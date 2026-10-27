import torch
import torch.nn as nn
import torch.nn.functional as F
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


class RMSGraphNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias   = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter('weight', None)
            self.bias = None

    def forward(self, x: torch.Tensor, batch: torch.Tensor, num_graphs: int = None):
        x_sq = x.pow(2)
        # Pass num_graphs so global_mean_pool skips the int(index.max())+1 .item()
        # call — that call is a Python-side scalar and breaks torch.compile / dynamo
        # tracing. Providing the size statically removes the graph break.
        mean_sq = global_mean_pool(x_sq, batch, size=num_graphs)
        inv_rms = torch.rsqrt(mean_sq[batch] + self.eps)
        y = x * inv_rms
        if self.affine and self.weight is not None:
            y = y * self.weight
            if self.bias is not None:
                y = y + self.bias
        return y


class GNNLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int,
                 encoder: bool = False, damping: float = 0.95):
        super().__init__()
        self.encoder = encoder
        # self.damping = damping
#        self.damping_logit = nn.Parameter(torch.logit(torch.tensor(float(damping))))

        self.norm   = RMSGraphNorm(channels=in_ch, affine=False)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, in_ch * 2))
        nn.init.zeros_(self.t_proj[1].weight)
        nn.init.zeros_(self.t_proj[1].bias)
        self.f_proj = nn.Linear(1, in_ch * 2, bias=False)
        self.conv0  = GraphConv(in_ch, in_ch, bias=False)
        self.act    = nn.SiLU()
        self.conv1  = GraphConv(in_ch, out_ch, bias=False)

        if not encoder:
            self.mlp = nn.Sequential(
                nn.Linear(in_ch, 2 * in_ch), nn.SiLU(), nn.Linear(2 * in_ch, in_ch)
            )
            nn.init.zeros_(self.mlp[2].weight)
            nn.init.zeros_(self.mlp[2].bias)

        self.shortcut = nn.Linear(in_ch, out_ch, bias=False) if in_ch != out_ch else nn.Identity()

    # ------------------------------------------------------------------ #
    # OPT-1: conditioning is constant across all num_iters iterations
    #   (t_vec and field don't change between iterations).
    #   Compute gamma/beta once in GNNUnet and pass them in, eliminating
    #   9 redundant t_proj + f_proj calls per forward pass.
    # ------------------------------------------------------------------ #
    def precompute_cond(self, t_vec, field, batch_vec):
        style_t = self.t_proj(t_vec)       # [B, in_ch*2]
        style_h = self.f_proj(field)       # [N_total, in_ch*2]
        cond = style_t[batch_vec] + style_h
        return cond.chunk(2, dim=-1)       # (gamma, beta)

    def forward(self, x, edge_index, batch, gamma, beta, num_graphs: int = None):
        x_in = x
        x = self.norm(x, batch, num_graphs)
        x = x * (1.0 + gamma) + beta
        x = self.act(x)
        if not self.encoder:
            x = x + self.mlp(x)
        x = self.conv0(x, edge_index)
        x = self.act(x)
        x = self.conv1(x, edge_index)
        return self.shortcut(x_in) + x


class GNNUnet(nn.Module):
    def __init__(self, base_ch: int, time_emb_dim: int,
                 discrete: bool = False, num_iters: int = 10, damping: float = 0.3):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.base_ch      = base_ch
        self.num_iters    = num_iters

        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim, bias=False),
        )
        self.field_stats_proj = nn.Sequential(
            nn.Linear(4, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        self.in_conv    = nn.Linear(2, base_ch, bias=False)
        self.block      = GNNLayer(base_ch, base_ch, time_emb_dim, encoder=False, damping=damping)
        self.final_norm = RMSGraphNorm(base_ch, affine=False)
        # +6: h_i (local field) + mean(h_i) (graph-level signed mean, direct phase signal)
        #      + mean(h_i^2) + mean(h_i * h_j) + std(h_i)
        self.out_conv   = nn.Linear(base_ch + 5, 1, bias=True)
        self.act        = nn.SiLU()

    def forward(self, x_in, batch, t):
        t_vec = self.time_emb(t)
        field = x_in[:, -1].unsqueeze(-1)

        batch_vec  = batch.batch
        num_graphs = batch.num_graphs

        # Global disorder statistics — all computable from the quenched disorder
        # at inference time, no leakage from the spin configuration.
        h_signed_mean = global_mean_pool(field,            batch_vec, size=num_graphs)  # [B,1] SIGNED
        h_sq_mean     = global_mean_pool(field.pow(2),     batch_vec, size=num_graphs)  # [B,1]
        h_std         = (h_sq_mean - h_signed_mean.pow(2)).clamp(min=0).sqrt()          # [B,1]

        # Edge frustration: mean(h_i * h_j) over all edges.
        src, dst    = batch.edge_index
        edge_prod   = field[src] * field[dst]                                           # [E,1]
        h_edge_corr = global_mean_pool(edge_prod, batch_vec[src], size=num_graphs)      # [B,1]

        h_stats = torch.cat([h_signed_mean, h_sq_mean, h_edge_corr, h_std], dim=-1)    # [B,4]
        t_vec   = t_vec + self.field_stats_proj(h_stats)

        x = self.in_conv(x_in)

        # OPT-1: precompute conditioning once for all num_iters iterations
        gamma, beta = self.block.precompute_cond(t_vec, field, batch_vec)

        for _ in range(self.num_iters):
            x = self.block(x, batch.edge_index, batch_vec, gamma, beta, num_graphs)

        x = self.final_norm(x, batch_vec, num_graphs)
        x = self.act(x)
        x = torch.cat((x, field, h_signed_mean[batch_vec], h_sq_mean[batch_vec], h_edge_corr[batch_vec], h_std[batch_vec]), dim=-1)
        return self.out_conv(x)
