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


def split_norm_fixed_N(x, num_graphs, N_per_graph, eps=1e-6):
    """Split per-channel node features into a DC (mean-over-nodes) component and a
    RMS-normalised fluctuation component.

    Only the *fluctuation* is normalised, so local structure stays well-scaled while
    the global DC offset (the basin-commitment signal) is preserved and returned
    separately instead of being washed out every iteration.
    """
    B, C = num_graphs, x.shape[-1]
    x_3d = x.view(B, N_per_graph, C)
    dc   = x_3d.mean(dim=1, keepdim=True)                       # [B, 1, C]
    fl   = x_3d - dc
    rms  = fl.pow(2).mean(dim=1, keepdim=True).sqrt()
    fl   = fl / (rms + eps)
    return fl.view(B * N_per_graph, C), dc.squeeze(1)           # [N, C], [B, C]


class GNNLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()

        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, in_ch * 2),
            nn.LayerNorm(in_ch * 2, elementwise_affine=False),
        )
        nn.init.zeros_(self.t_proj[1].weight)
        nn.init.zeros_(self.t_proj[1].bias)

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
                 beta_scale: float = 100.0, global_dim: int = 8):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.base_ch      = base_ch
        self.num_iters    = num_iters
        self.beta_scale   = beta_scale
        self.global_dim   = global_dim

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

        self.damp_proj = nn.Linear(time_emb_dim, 1)
        nn.init.constant_(self.damp_proj.bias, 1.0)

        # ---- global state (low-dim, un-normalised integrator) ----------------
        # The basin-commitment is essentially a low-dimensional decision (which way
        # the magnetisation tips and how hard), so the global channel is a tiny
        # `global_dim` vector. It is a PLAIN INTEGRATOR (g += read(pooled)) so it can
        # grow across iterations, broadcast back as an ADDITIVE per-channel bias --
        # i.e. a learned global effective field applied coherently to every site.
        self.g_read = nn.Linear(2 * base_ch, global_dim)        # [mean | field·x] -> dg
        self.g_cast = nn.Linear(global_dim, base_ch)            # dg -> additive node bias
        # Start inert: g_cast = 0 so the model begins at the local-only behaviour
        # and *learns* to commit (clean A/B vs unetGnn_cfg).
        nn.init.zeros_(self.g_cast.weight)
        nn.init.zeros_(self.g_cast.bias)

        self.in_lin_proj  = nn.Linear(2, base_ch, bias=False)
        self.h_proj       = nn.Linear(1, base_ch, bias=False)
        self.block        = GNNLayer(base_ch, base_ch, time_emb_dim)
        # readout sees the global state too (global_dim) plus field stats (3)
        self.out_lin_proj = nn.Linear(base_ch + global_dim + 3, 1, bias=True)
        # learnable output gain so logits can saturate the D3PM softmax (init -> scale 1)
        self.logit_scale  = nn.Parameter(torch.zeros(1))

    def forward(self, x_in, batch, t, drop_beta: bool = False):
        inv_temp   = batch.inv_temp
        num_graphs = batch.num_graphs
        batch_vec  = batch.batch

        t_vec = self.time_emb(t)


        beta_vec = self.beta_emb(inv_temp * self.beta_scale)
        t_vec = t_vec + beta_vec

        alpha_node = torch.sigmoid(self.damp_proj(beta_vec))[batch_vec]  # [N, 1]

        field         = x_in[:, -1:]                          # [N, 1]
        h_signed_mean = batch.field_mean.unsqueeze(-1)  # [B, 1]
        h_std         = batch.field_std.unsqueeze(-1)         # [B, 1]

        edge_weight = batch.edge_weight

        # local FiLM (kept bounded as before)
        gamma, beta = self.block.precompute_cond(t_vec, batch_vec)
        gamma = torch.tanh(gamma)
        beta  = beta.clamp(-2, 2)

        h_proj      = self.h_proj(field)                      # [N, base_ch]
        x           = self.in_lin_proj(x_in) + h_proj
        N_per_graph = batch.num_nodes // num_graphs

        # low-dim global state, integrated across iterations (starts at zero)
        g = x.new_zeros(num_graphs, self.global_dim)          # [B, dg]

        for _ in range(self.num_iters):
            x_loc, _ = split_norm_fixed_N(x.contiguous(), num_graphs, N_per_graph)

            # broadcast the global state back as an additive global effective field
            x_loc = x_loc + self.g_cast(g)[batch_vec]         # [N, C]

            x = x + alpha_node * self.block(x_loc, batch.edge_index, batch_vec,
                                             gamma, beta, edge_weight, num_graphs) + h_proj

            # pool net + field-weighted aggregates, integrate into g (no gating)
            x_3d   = x.view(num_graphs, N_per_graph, -1)
            mean_x = x_3d.mean(dim=1)                          # [B, C]
            hw_x   = (field * x).view(num_graphs, N_per_graph, -1).mean(dim=1)  # field-weighted
            g      = g + self.g_read(torch.cat([mean_x, hw_x], dim=-1))         # integrator

        x = torch.cat((x,
                        g[batch_vec],
                        h_signed_mean[batch_vec],
                        h_std[batch_vec],
                        (h_std * inv_temp.unsqueeze(-1))[batch_vec]), dim=-1)

        return self.out_lin_proj(x) * self.logit_scale.exp()
