import torch
import torch.nn as nn
import math

# ---------------------------------------------------------------------------
# Cavity/EA variant of unetGnn_cfg.GNNUnet.  Two changes vs the baseline:
#
#   (1) Edge-conditioned (FiLM) messages.  The baseline let the coupling enter
#       only as a scalar gain inside GraphConv:  agg_i = Σ_j (β J_ij) (W x_j).
#       Here the neighbour message is
#           msg_{j->i} = (β J_ij) (W x_j) ⊙ (1 + γ_e) + β_e,
#       where (γ_e, β_e) are per-channel affine params produced by a small MLP
#       of the edge feature β·J_ij.  γ_e, β_e are zero-initialised, so at init
#       the message reduces exactly to the baseline GraphConv term — training
#       starts from the known-good linear model and learns the (nonlinear,
#       bond-type-aware) correction on top.
#
#   (2) Self-consistent local (cavity) field.  Each iteration we read a
#       magnetisation estimate m_i = tanh(mag_readout(x)) and inject the
#       endogenous field  β h_i = β Σ_j J_ij m_j  through the (previously
#       external-field) h_proj pathway.  This is an unrolled mean-field / cavity
#       update: β h_i is the argument of m_i = tanh(β h_i), and it generalises
#       over J by locality.  mag_readout is zero-initialised so the injected
#       field starts at 0 and the whole model reduces to the baseline at init.
#
# Constructor signature is kept identical to unetGnn_cfg.GNNUnet so
# utils/loading.py can rebuild it by only swapping the import.
# ---------------------------------------------------------------------------


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
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, edge_hidden: int = 32):
        super().__init__()
        self.t_proj = nn.Sequential(nn.SiLU(), nn.LayerNorm(time_emb_dim), nn.Linear(time_emb_dim, in_ch * 2))
        nn.init.zeros_(self.t_proj[2].weight)
        nn.init.zeros_(self.t_proj[2].bias)

        # Split the old single GraphConv into an explicit root + neighbour map so
        # the neighbour message can carry the edge FiLM modulation (change 1).
        self.W_root  = nn.Linear(in_ch, out_ch, bias=False)
        self.W_neigh = nn.Linear(in_ch, out_ch, bias=False)
        self.act = nn.SiLU()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, in_ch), nn.SiLU(), nn.Linear(in_ch, in_ch)
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

        # Edge-conditioned FiLM: β·J_ij -> per-channel (γ_e, β_e). Zero-init the
        # last layer so the message reduces to the baseline GraphConv term.
        self.edge_film = nn.Sequential(
            nn.Linear(1, edge_hidden), nn.SiLU(), nn.Linear(edge_hidden, out_ch * 2)
        )
        nn.init.zeros_(self.edge_film[2].weight)
        nn.init.zeros_(self.edge_film[2].bias)

    def precompute_cond(self, t_vec, batch_vec):
        cond = self.t_proj(t_vec)[batch_vec]        # [N, in_ch*2]
        return cond.chunk(2, dim=-1)                # (gamma, beta)

    def precompute_edge_cond(self, edge_weight):
        """edge_weight [E] = β·J_ij  ->  per-channel (γ_e, β_e), each [E, out_ch].

        Constant for a whole sampling run, so this is computed once (outside the
        num_iters loop) and reused across every iteration / diffusion step.
        """
        ef = self.edge_film(edge_weight.view(-1, 1))   # [E, 2*out_ch]
        return ef.chunk(2, dim=-1)

    def forward(self, x, edge_index, gamma, beta, edge_weight, gamma_e, beta_e):
        x = x * (1.0 + gamma) + beta
        x = self.act(x)
        x = x + self.mlp(x)

        src, dst = edge_index[0], edge_index[1]
        h = self.W_neigh(x)                                    # [N, C]
        ew = edge_weight.view(-1, 1).to(h.dtype)               # [E, 1] = β·J
        # baseline physical gain (β J) · W x_j  +  learned per-channel FiLM
        msg = ew * h[src] * (1.0 + gamma_e) + beta_e           # [E, C]
        agg = h.new_zeros(x.size(0), h.size(-1))
        agg.index_add_(0, dst, msg)                            # Σ_{j∈∂i} msg
        return self.W_root(x) + agg


class GNNUnet(nn.Module):
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
        self.global_gate = nn.Parameter(torch.full((1,), -3.0))  # sigmoid(-3) ≈ 0.05

        self.damp_proj = nn.Linear(time_emb_dim, 1)
        nn.init.constant_(self.damp_proj.bias, 1.0)

        self.in_lin_proj = nn.Linear(2, base_ch, bias=False)
        # h_proj now injects the endogenous cavity field β·h_i (change 2) rather
        # than the (zero, for EA) external field.
        self.h_proj      = nn.Linear(1, base_ch, bias=False)
        # magnetisation readout m_i = tanh(mag_readout(x)); zero-init so the
        # injected cavity field starts at 0 -> model == baseline at init.
        self.mag_readout = nn.Linear(base_ch, 1)
        nn.init.zeros_(self.mag_readout.weight)
        nn.init.zeros_(self.mag_readout.bias)

        self.block        = GNNLayer(base_ch, base_ch, time_emb_dim)
        self.out_lin_proj = nn.Linear(base_ch, 1, bias=True)
        self.act          = nn.SiLU()

    def forward(self, x_in, batch, t, drop_beta: bool = False, negative: bool = False):
        inv_temp   = batch.inv_temp
        num_graphs = batch.num_graphs
        batch_vec  = batch.batch

        t_vec = self.time_emb(t)
        sign = -1.0 if negative else 1.0

        beta_vec = self.beta_emb(inv_temp * self.beta_scale)
        t_vec = t_vec + beta_vec

        alpha_node = torch.sigmoid(self.damp_proj(beta_vec))[batch_vec]  # [N, 1]

        # field         = x_in[:, -1:]                          # [N, 1] (0 for EA)
        # h_signed_mean = sign * batch.field_mean.unsqueeze(-1)
        # h_std         = batch.field_std.unsqueeze(-1)

        edge_index  = batch.edge_index
        edge_weight = batch.edge_weight                        # [E] = β·J
        src, dst    = edge_index[0], edge_index[1]

        gamma, beta = self.block.precompute_cond(t_vec, batch_vec)
        gamma = torch.tanh(gamma)
        beta  = beta.clamp(-2, 2)
        # edge FiLM params: constant across the loop, computed once (change 1)
        gamma_e, beta_e = self.block.precompute_edge_cond(edge_weight)

        x = self.in_lin_proj(x_in)
        N_per_graph = batch.num_nodes // num_graphs

        for _ in range(self.num_iters):
            x = rms_norm_fixed_N(x.contiguous(), num_graphs, N_per_graph)

            # --- self-consistent cavity field  β h_i = β Σ_j J_ij m_j  (change 2)
            m_est = torch.tanh(self.mag_readout(x))            # [N, 1] in (-1, 1)
            ew = edge_weight.view(-1, 1).to(m_est.dtype)
            h_local = m_est.new_zeros(x.size(0), 1)
            h_local.index_add_(0, dst, ew * m_est[src])        # [N, 1]
            field_feat = self.h_proj(h_local)                  # [N, C]

            x = x + alpha_node * self.block(x, edge_index, gamma, beta,
                                            edge_weight, gamma_e, beta_e) + field_feat
            
            x_mean = x.view(num_graphs, N_per_graph, -1).mean(dim=1)[batch_vec]
            x = x + torch.sigmoid(self.global_gate) * x_mean

        x = self.act(x)

        return self.out_lin_proj(x)
