import torch
import torch.nn as nn
from torch_geometric.nn.conv import GraphConv
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


class GNNLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.t_proj = nn.Sequential(nn.SiLU(), nn.LayerNorm(time_emb_dim), nn.Linear(time_emb_dim, in_ch*2))
        nn.init.zeros_(self.t_proj[2].weight)
        nn.init.zeros_(self.t_proj[2].bias)

        self.conv0 = GraphConv(in_ch, in_ch, bias=False)
        self.conv1 = GraphConv(in_ch, out_ch, bias=False)
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
        x = self.act(x)
        x = self.conv1(x, edge_index, edge_weight)
        return x


class GNNUnet(nn.Module):
    """Unrolled variant of models/unetGnn_cfg.GNNUnet.

    Instead of applying one weight-shared block `num_iters` times, this stacks
    `num_iters` independent blocks, each with its own conditioning projection,
    damping gate and global-mean gate. Each block applies two graph
    convolutions.
    """

    def __init__(self, base_ch: int, time_emb_dim: int,
                 discrete: bool = False, num_iters: int = 10,
                 beta_scale: float = 100.0):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.base_ch      = base_ch
        self.num_iters    = num_iters
        self.beta_scale   = beta_scale

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
        # One gate per block: sigmoid(-3) ~= 0.05 initially
        self.global_gate = nn.Parameter(torch.full((num_iters,), -3.0))

        self.damp_proj = nn.ModuleList([nn.Linear(time_emb_dim, 1) for _ in range(num_iters)])
        for proj in self.damp_proj:
            nn.init.constant_(proj.bias, 1.0)

        self.in_lin_proj = nn.Linear(2, base_ch, bias=False)
        self.h_proj      = nn.Linear(1, base_ch, bias=False)
        self.blocks      = nn.ModuleList([
            GNNLayer(base_ch, base_ch, time_emb_dim) for _ in range(num_iters)
        ])
        self.out_lin_proj = nn.Linear(base_ch + 3, 1, bias=True)
        self.act         = nn.SiLU()

    def forward(self, x_in, batch, t, drop_beta: bool = False, negative: bool = False):
        inv_temp   = batch.inv_temp
        num_graphs = batch.num_graphs
        batch_vec  = batch.batch

        t_vec = self.time_emb(t)
        sign = -1.0 if negative else 1.0

        beta_vec = self.beta_emb(inv_temp * self.beta_scale)

        t_vec = t_vec + beta_vec

        field         = x_in[:, -1:]                          # [N, 1]
        h_signed_mean = sign * batch.field_mean.unsqueeze(-1)  # [B, 1]
        h_std         = batch.field_std.unsqueeze(-1)         # [B, 1]

        edge_weight = batch.edge_weight

        h_proj      = self.h_proj(field)                      # [N, base_ch]
        x           = self.in_lin_proj(x_in) + h_proj
        N_per_graph = batch.num_nodes // num_graphs

        for i, block in enumerate(self.blocks):
            alpha_node = torch.sigmoid(self.damp_proj[i](beta_vec))[batch_vec]  # [N, 1]

            gamma, beta = block.precompute_cond(t_vec, batch_vec)
            gamma = torch.tanh(gamma)
            beta  = beta.clamp(-2, 2)

            x = rms_norm_fixed_N(x.contiguous(), num_graphs, N_per_graph)
            x = x + alpha_node * block(x, batch.edge_index, batch_vec,
                                       gamma, beta, edge_weight, num_graphs) + h_proj
            x_mean = x.view(num_graphs, N_per_graph, -1).mean(dim=1)[batch_vec]  # [N, C]
            x = x + torch.sigmoid(self.global_gate[i]) * x_mean

        x = self.act(x)
        x = torch.cat((x,
                       h_signed_mean[batch_vec],
                       h_std[batch_vec],
                       (h_std * inv_temp.unsqueeze(-1))[batch_vec]), dim=-1)

        return self.out_lin_proj(x)
