"""
Decoupled (PyG-Batch-free) reimplementation of GNNUnet.forward for optimization
experiments.  Math is IDENTICAL to models/unetGnn_cfg.py:GNNUnet.forward — the
submodules are *shared by reference* from a baseline GNNUnet, so parameters are
byte-for-byte the same and outputs must match to fp tolerance.

The only change is the call signature: forward takes plain tensors + baked-in
ints instead of a torch_geometric Batch.  This removes the 3 dynamo graph breaks
(batch.batch / batch.num_nodes / PyG internals) and makes the whole forward
capturable by torch.compile / CUDA graphs.

Toggles:
  use_checkpoint : gradient-checkpoint each recurrent iteration (memory lever, #3)
  use_jit_fused  : fuse pointwise chains via torch.jit.script (fusion lever, #4)

GraphConv is left untouched (optimization #1 intentionally excluded).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.unetGnn_cfg import rms_norm_fixed_N


# ── fused pointwise helpers (NNC/TE fuser on CUDA, no Triton needed) ───────────
@torch.jit.script
def _pre_conv_fused(x, gamma, beta, num_graphs: int, N_per: int, eps: float):
    """rms_norm -> FiLM affine -> SiLU, fused. Returns (x_normed, x_modulated)."""
    C = x.shape[-1]
    x3 = x.view(num_graphs, N_per, C)
    rms = x3.pow(2).mean(dim=1, keepdim=True).sqrt()
    xn = (x3 / (rms + eps)).view(num_graphs * N_per, C)
    xm = F.silu(xn * (1.0 + gamma) + beta)
    return xn, xm


@torch.jit.script
def _post_conv_fused(xn, xc, alpha_node, h_proj, gate,
                     batch_vec, num_graphs: int, N_per: int):
    """residual add -> per-graph mean -> gated add, fused."""
    x = xn + alpha_node * xc + h_proj
    C = x.shape[-1]
    x_mean = x.view(num_graphs, N_per, C).mean(dim=1).index_select(0, batch_vec)
    return x + gate * x_mean


class GNNUnetPlain(nn.Module):
    def __init__(self, base: nn.Module,
                 use_checkpoint: bool = False, use_jit_fused: bool = False):
        super().__init__()
        # share submodules by reference -> identical weights, no state copy
        self.time_emb     = base.time_emb
        self.beta_emb     = base.beta_emb
        self.damp_proj    = base.damp_proj
        self.in_lin_proj  = base.in_lin_proj
        self.h_proj       = base.h_proj
        self.block        = base.block
        self.out_lin_proj = base.out_lin_proj
        self.act          = base.act
        self.global_gate  = base.global_gate
        self.beta_scale   = base.beta_scale
        self.num_iters    = base.num_iters
        self.use_checkpoint = use_checkpoint
        self.use_jit_fused  = use_jit_fused

    def _iter(self, x, gamma, beta_c, alpha_node, h_proj,
              edge_index, edge_weight, batch_vec, gate, num_graphs, N_per):
        blk = self.block
        if self.use_jit_fused:
            xn, xm = _pre_conv_fused(x.contiguous(), gamma, beta_c,
                                     num_graphs, N_per, 1e-6)
            xm = xm + blk.mlp(xm)
            xc = blk.conv0(xm, edge_index, edge_weight)
            return _post_conv_fused(xn, xc, alpha_node, h_proj, gate,
                                    batch_vec, num_graphs, N_per)
        else:
            xn = rms_norm_fixed_N(x.contiguous(), num_graphs, N_per)
            xc = blk(xn, edge_index, batch_vec, gamma, beta_c, edge_weight, num_graphs)
            x = xn + alpha_node * xc + h_proj
            x_mean = x.view(num_graphs, N_per, -1).mean(dim=1)[batch_vec]
            return x + gate * x_mean

    def forward(self, x_in, t, inv_temp, field_mean, field_std,
                edge_index, edge_weight, batch_vec, num_graphs: int, N_per: int):
        t_vec = self.time_emb(t)
        beta_vec = self.beta_emb(inv_temp * self.beta_scale)
        t_vec = t_vec + beta_vec
        alpha_node = torch.sigmoid(self.damp_proj(beta_vec))[batch_vec]

        field = x_in[:, -1:]
        h_signed_mean = field_mean.unsqueeze(-1)
        h_std = field_std.unsqueeze(-1)

        gamma, beta_c = self.block.precompute_cond(t_vec, batch_vec)
        gamma = torch.tanh(gamma)
        beta_c = beta_c.clamp(-2, 2)

        h_proj = self.h_proj(field)
        x = self.in_lin_proj(x_in) + h_proj
        gate = torch.sigmoid(self.global_gate)

        for _ in range(self.num_iters):
            if self.use_checkpoint and x.requires_grad:
                x = checkpoint(self._iter, x, gamma, beta_c, alpha_node, h_proj,
                               edge_index, edge_weight, batch_vec, gate,
                               num_graphs, N_per, use_reentrant=False)
            else:
                x = self._iter(x, gamma, beta_c, alpha_node, h_proj,
                               edge_index, edge_weight, batch_vec, gate,
                               num_graphs, N_per)

        x = self.act(x)
        x = torch.cat((x, h_signed_mean[batch_vec], h_std[batch_vec],
                       (h_std * inv_temp.unsqueeze(-1))[batch_vec]), dim=-1)
        return self.out_lin_proj(x)


def unpack_batch(batch):
    """Extract the plain-tensor args GNNUnetPlain.forward expects from a PyG Batch."""
    x_in = torch.cat([batch.x, batch.field], dim=1)
    num_graphs = batch.num_graphs
    N_per = batch.num_nodes // num_graphs
    return dict(
        x_in=x_in,
        inv_temp=batch.inv_temp,
        field_mean=batch.field_mean,
        field_std=batch.field_std,
        edge_index=batch.edge_index,
        edge_weight=batch.edge_weight,
        batch_vec=batch.batch,
        num_graphs=num_graphs,
        N_per=N_per,
    )
