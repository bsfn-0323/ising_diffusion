import torch
import torch.nn as nn
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.pool import global_mean_pool
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


def rms_norm_fixed_N(x, num_graphs, N_per_graph, eps=1e-6):
    """Per-channel RMS norm over nodes, using dense reshape (no scatter)."""
    B, C = num_graphs, x.shape[-1]
    x_3d = x.view(B, N_per_graph, C)
    rms  = x_3d.pow(2).mean(dim=1, keepdim=True).sqrt()
    return (x_3d / (rms + eps)).view(B * N_per_graph, C)


def rms_norm_scatter(x, batch_vec, num_graphs, eps=1e-6):
    """Same normalisation as rms_norm_fixed_N, but via scatter so the graphs in
    the batch may have different node counts.  Formula kept identical --
    x / (sqrt(mean(x^2)) + eps) -- so results match the dense path bit-for-bit
    up to reduction order when the batch happens to be uniform."""
    mean_sq = global_mean_pool(x.pow(2), batch_vec, size=num_graphs)   # [B, C]
    return x / (mean_sq.sqrt() + eps)[batch_vec]


def batch_is_uniform(batch, num_graphs):
    """True iff every graph in the batch has the same number of nodes.
    Costs one host sync, so callers should cache the answer."""
    if num_graphs <= 1:
        return True
    ptr = getattr(batch, "ptr", None)
    counts = (ptr[1:] - ptr[:-1]) if ptr is not None \
        else torch.bincount(batch.batch, minlength=num_graphs)
    return bool((counts == counts[0]).all())


class GNNLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.t_proj = nn.Sequential(nn.SiLU(), nn.LayerNorm(time_emb_dim), nn.Linear(time_emb_dim, in_ch*2))
        # self.t_proj = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(time_emb_dim, in_ch * 2),
        #     nn.LayerNorm(in_ch * 2, elementwise_affine=False),
        # )
        nn.init.zeros_(self.t_proj[2].weight)
        nn.init.zeros_(self.t_proj[2].bias)

        self.conv0 = GraphConv(in_ch, out_ch, bias=False)
        self.act   = nn.SiLU()
        self.mlp   = nn.Sequential(
            nn.Linear(in_ch, in_ch), nn.SiLU(), nn.Linear(in_ch, in_ch)
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def precompute_cond(self, t_vec, batch_vec):
        cond = self.t_proj(t_vec)[batch_vec]        # [N, in_ch*2]
        return cond.chunk(2, dim=-1)                # (gamma, beta)

    def forward(self, x, edge_index, batch, gamma, beta, edge_weight, num_graphs=None):
        x = x * (1.0 + gamma) + beta
        x = self.act(x)
        x = x + self.mlp(x)
        x = self.conv0(x, edge_index, edge_weight)
        return x


class GNNUnet(nn.Module):
    def __init__(self, base_ch: int, time_emb_dim: int,
                 discrete: bool = False, num_iters: int = 10,
                 beta_scale: float = 100.0, use_tilt_head: bool = False,
                 tilt_scale: float = 4.0, uniform_N=None,both=False):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.base_ch      = base_ch
        self.num_iters    = num_iters
        self.beta_scale   = beta_scale
        # Do all graphs in a batch have the same node count?
        #   True  -> dense reshape path (fastest; what every previous run used)
        #   False -> scatter path (required for mixed-size batches)
        #   None  -> detect on the first forward and cache the answer.
        # Detection costs a host sync, so pass it explicitly under torch.compile
        # or CUDA-graph capture.  This is a runtime switch only: it adds no
        # parameters, so old checkpoints load unchanged.
        self.uniform_N = uniform_N
        # Decoupled one-body tilt channel for the flip-asymmetry loss L_A.
        # OFF by default => forward is byte-identical to the original model.
        self.use_tilt_head = use_tilt_head
        # Bound |b_i| <= tilt_scale via tanh: the tilt must never drive the
        # logits into saturation, else the VLB -log(p) terms blow up and L_A
        # runs away (positive feedback: b -> saturation -> huge r_hat -> bigger b).
        self.tilt_scale = tilt_scale

        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim, bias=False),
        )
        self.beta_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim, bias=False),
        )
        # Null embedding replaces beta_emb when drop_beta=True (CFG).
        # self.null_beta_emb = nn.Parameter(torch.zeros(time_emb_dim))
        # __init__:
        self.global_gate = nn.Parameter(torch.full((1,), -3.0))  # sigmoid(-3) ≈ 0.05 initially

        

        self.damp_proj = nn.Linear(time_emb_dim, 1)
        nn.init.constant_(self.damp_proj.bias, 1.0)

        self.in_lin_proj = nn.Linear(3, base_ch, bias=False) if both else nn.Linear(2, base_ch, bias=False)
        self.h_proj      = nn.Linear(2, base_ch, bias=False) if both else nn.Linear(1,base_ch,bias=False)
        self.block       = GNNLayer(base_ch, base_ch, time_emb_dim)
        self.out_lin_proj = nn.Linear(base_ch + 3, 1, bias=True)
        self.act         = nn.SiLU()

        # One-body tilt head, PHYSICAL form: b_i = a_g(t,beta) * h_i, i.e. the
        # logit tilt is proportional to the local field (exactly the target
        # r*=2*beta*(h.s) => b_i = 2*beta*h_i).  Only a per-graph scalar
        # coefficient a_g is learned (from the time+beta conditioning).  This is
        # sufficient (matches the target exactly) and CANNOT collapse to a useless
        # constant the way a free MLP does: b_i is always field-proportional.
        # Reads no message-passing output x, so it cannot represent correlations.
        # Zero-init => a_g == 0 => b == 0 at start (no disruption).
        # if self.use_tilt_head:
        #     self.tilt_coeff = nn.Linear(time_emb_dim, 1)   # cond -> a_g (per graph)
        #     nn.init.zeros_(self.tilt_coeff.weight)
        #     nn.init.zeros_(self.tilt_coeff.bias)


    def forward(self, x_in, batch, t, drop_beta: bool = False,negative: bool = False,
                return_parts: bool = False):
        # print(x_in.shape)
        inv_temp   = batch.inv_temp
        num_graphs = batch.num_graphs
        batch_vec  = batch.batch

        t_vec = self.time_emb(t)
        sign = -1.0 if negative else 1.0

        beta_vec = self.beta_emb(inv_temp * self.beta_scale)

        t_vec = t_vec + beta_vec

        alpha_node = torch.sigmoid(self.damp_proj(beta_vec))[batch_vec]  # [N, 1]

        field         = x_in[:,1]                         # [N, 1]
        field = field[:,None]
        h_signed_mean =sign* batch.field_mean.unsqueeze(-1)        # [B, 1]
        h_std         = batch.field_std.unsqueeze(-1)         # [B, 1]

        edge_weight = batch.edge_weight

        gamma, beta = self.block.precompute_cond(t_vec, batch_vec)
        gamma = torch.tanh(gamma)
        beta  = beta.clamp(-2, 2)
        fields = x_in[:,1:]
        # print(fields.shape)
        h_proj      = self.h_proj(fields)                      # [N, base_ch]
        x           = self.in_lin_proj(x_in) + h_proj

        if self.uniform_N is None:
            self.uniform_N = batch_is_uniform(batch, num_graphs)
        uniform = self.uniform_N and batch.num_nodes % num_graphs == 0
        N_per_graph = batch.num_nodes // num_graphs if uniform else None

        for i in range(self.num_iters):
            # print(i)
            if uniform:
                x = rms_norm_fixed_N(x.contiguous(), num_graphs, N_per_graph)
            else:
                x = rms_norm_scatter(x, batch_vec, num_graphs)
            x = x + alpha_node * self.block(x, batch.edge_index, batch_vec,
                                             gamma, beta, edge_weight, num_graphs) + h_proj
            if uniform:
                x_mean = x.view(num_graphs, N_per_graph, -1).mean(dim=1)[batch_vec]  # [N, C]
            else:
                x_mean = global_mean_pool(x, batch_vec, size=num_graphs)[batch_vec]  # [N, C]
            x = x + torch.sigmoid(self.global_gate) * x_mean

        x = self.act(x)
        x = torch.cat((x,
                        h_signed_mean[batch_vec],
                        h_std[batch_vec],
                        (h_std * inv_temp.unsqueeze(-1))[batch_vec]), dim=-1)

        body_logit = self.out_lin_proj(x)                     # [N, 1] correlation body

        # if self.use_tilt_head:
        #     # Physical one-body tilt: b_i = a_g(t,beta) * h_i.  a_g is a per-graph
        #     # scalar from the time+beta conditioning (t_vec already includes
        #     # beta_vec); bounded by tilt_scale via tanh so it cannot run away.
        #     # Field-proportional => cannot collapse to a useless constant, and
        #     # depends on no message-passing output x => cannot touch correlations.
        #     a_g = self.tilt_scale * torch.tanh(self.tilt_coeff(t_vec))   # [B, 1] in (-tilt_scale, tilt_scale)
        #     b = a_g[batch_vec] * field                                  # [N, 1] tilt ∝ local field
        #     if return_parts:
        #         return body_logit, b
        #     return body_logit + b

        # if return_parts:
        #     return body_logit, torch.zeros_like(body_logit)
        return body_logit
