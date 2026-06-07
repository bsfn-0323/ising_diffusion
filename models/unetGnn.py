import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GraphConv,GCNConv,ChebConv
# from torch_geometric.nn.norm import BatchNorm,GraphNorm,LayerNorm
# from torch.nn import GroupNorm
from torch_geometric.nn.pool import global_mean_pool
import math

def compute_local_mag(x,instance_idx,batch,ptr):
    device = x.device
    N = int(ptr[1] - ptr[0])
    A = int(instance_idx.max()) + 1
    x = x.view(-1)                              # [B*N]

    # per-node instance id and site id
    node_inst = instance_idx[batch]       # [B*N]
    site = (torch.arange(x.shape[0], device=device) - ptr[batch])  # [B*N], 0..N-1
    flat = node_inst * N + site                       # unique (instance,site) key

    numer = torch.zeros(A * N, device=device, dtype=x.dtype).scatter_add_(0, flat, x)
    cnt   = torch.zeros(A * N, device=device, dtype=x.dtype).scatter_add_(0, flat, torch.ones_like(x))
    local_mag = (numer / cnt.clamp(min=1)).view(A, N)         # [A, N]

    local_mag_node = local_mag.view(-1)[flat].view(-1, 1)     # [B*N, 1]
    return local_mag, local_mag_node

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
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        # 1. Square the node features
        x_sq = x.pow(2)
        
        # 2. Compute mean squared per graph dynamically using the batch vector
        # Returns shape: [Batch_size, Channels]
        mean_sq = global_mean_pool(x_sq, batch)
        
        # 3. Broadcast the graph-level mean back to all individual nodes
        # Returns shape: [Total_nodes_in_batch, Channels]
        mean_sq_expanded = mean_sq[batch]
        
        # 4. RMS Calculation
        inv_rms = torch.rsqrt(mean_sq_expanded + self.eps)
        y = x * inv_rms

        if self.affine and self.weight is not None:
            y = y * self.weight
            if self.bias is not None:
                y = y + self.bias

        return y
    
class GNNLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        
        # Size is gone!
        self.norm = RMSGraphNorm(channels=in_ch, affine=False)
        
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, in_ch * 2))
        nn.init.zeros_(self.t_proj[1].weight)
        nn.init.zeros_(self.t_proj[1].bias)

        self.g_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, in_ch * 2))
        nn.init.zeros_(self.g_proj[1].weight)
        nn.init.zeros_(self.g_proj[1].bias)
        
        self.f_proj = nn.Linear(1, in_ch * 2, bias=False)
        # self.f_proj = nn.Sequential(
        #     nn.Linear(1, in_ch),
        #     nn.SiLU(),
        #     nn.Linear(in_ch, in_ch * 2)
        # )
        nn.init.zeros_(self.f_proj.weight)
        # nn.init.zeros_(self.f_proj[2].weight)
        # nn.init.zeros_(self.f_proj[2].bias)
        
        self.conv0 = GraphConv(in_ch, in_ch, bias=False)
        self.act = nn.SiLU()
        self.conv1 = GraphConv(in_ch, out_ch, bias=False)
        
        if not encoder:
            self.mlp = nn.Sequential(nn.Linear(in_ch, 2 * in_ch), nn.SiLU(), nn.Linear(2 * in_ch, in_ch))
            nn.init.zeros_(self.mlp[2].weight)
            nn.init.zeros_(self.mlp[2].bias)
            
        self.shortcut = nn.Linear(in_ch, out_ch, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x, edge_index, batch, t_vec, field, field_emb=None):
        x_in = x
        
        # MUST pass batch here now for dynamic pooling
        x = self.norm(x, batch)
        
        style_t = self.t_proj(t_vec)
        style_h = self.f_proj(field)
        
        cond = style_t[batch] + style_h + self.g_proj(field_emb)[batch]
        # cond = style_t[batch] + style_h 
        gamma, beta = cond.chunk(2, dim=-1)
        
        x = x * (1.0 + gamma) + beta
        x = self.act(x)
        
        if not self.encoder:
            x = x + self.mlp(x)

        x = self.conv0(x, edge_index)
        x = self.act(x)
        x = self.conv1(x, edge_index)
        
        return x + self.shortcut(x_in)

class GNNUnet(nn.Module):
    # Removed `size` parameter from initialization
    def __init__(self, base_ch: int, ch_mult: list, time_emb_dim: int,discrete:bool=False):
        super().__init__()
        self.layer_len = len(ch_mult)
        self.time_emb_dim = time_emb_dim
        self.base_ch =base_ch
        self.ch_mult=ch_mult
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim, bias=False),
        )

        # Field-prediction branch: m_hat ~ true local magnetization <s_k>,
        # regressed from (h, J/topology) AND an empirical probe mk_emp of the
        # *current* diffused batch -- the per-instance average (over that
        # instance's chains in the batch) of the noisy x_t, via
        # compute_local_mag. For the symmetric binary channel,
        # E[x_t | x_0] = lambda_t * x_0 (lambda_t = the second eigenvalue of
        # Q_bar_t = prod(1-b_t')), so mk_emp ~= mk_i * lambda_t + noise(t):
        # a *deterministically rescaled, unbiased* probe of the truth at every
        # noise level -- not "the model's current guess" fed back on itself.
        # We deliberately do NOT divide out lambda_t (that blows up the noise
        # as t -> T, where lambda_t -> 0); instead we hand the network mk_emp
        # together with the time embedding and let it learn the known,
        # deterministic rescaling itself, in a naturally regularized way.
        #
        # Learning (h, J) -> <s_k> from scratch is, structurally, learning to
        # solve the belief-propagation/cavity fixed point: an iterative,
        # non-local computation (messages travel ~ the correlation length).
        # mk_emp gives the regressor a running empirical anchor on the
        # answer, turning "solve BP from nothing" into "calibrate a noisy
        # probe" -- much like the ground-truth-mk conditioning that already
        # works well, but computable without knowing the true mk in advance.
        #
        # The (h, J) -> <s_k> part is still hard and non-local, so -- as
        # before -- we share ONE GraphConv and iterate it (residual + act),
        # mirroring BP's fixed-point structure (a single local update rule
        # applied repeatedly) rather than stacking K independent transforms
        # tied to training-set depths/topologies.
        self.field_iters = 4
        self.field_in = nn.Linear(2, time_emb_dim, bias=False)
        self.field_t_proj = nn.Linear(time_emb_dim, time_emb_dim, bias=False)
        nn.init.zeros_(self.field_t_proj.weight)
        self.field_convs = nn.ModuleList([
            GraphConv(time_emb_dim, time_emb_dim, bias=False) for _ in range(self.field_iters)
        ])
        self.field_act = nn.SiLU()
        self.field_out_act = nn.Tanh()
        self.m_head = nn.Linear(time_emb_dim, 1, bias=False)
        nn.init.zeros_(self.m_head.weight)

        self.in_conv = nn.Linear(2, base_ch, bias=False)
        
        self.encoder = nn.ModuleList([
            GNNLayer(base_ch * ch_mult[i], base_ch * ch_mult[i+1], time_emb_dim, encoder=True) 
            for i in range(self.layer_len - 1)
        ])

        self.latent = GNNLayer(base_ch * ch_mult[-1], base_ch * ch_mult[-1], time_emb_dim, encoder=False)

        self.decoder = nn.ModuleList([
            GNNLayer(base_ch * ch_mult[i], base_ch * ch_mult[i-1], time_emb_dim, encoder=False)
            for i in reversed(range(1, self.layer_len))
        ])

        self.final_norm = RMSGraphNorm(base_ch, affine=False)
        if not discrete:
            self.out_conv = nn.Linear(base_ch + 1, 1, bias=True)
        else:
            self.out_conv = nn.Linear(base_ch + 1, 1, bias=True)
            
        self.act = nn.SiLU()

    def forward(self, x_in, batch, t):
        t_vec = self.time_emb(t)

        hh = x_in[:,-1].unsqueeze(-1) #xin = (B(N), 2) -> (B(N), 1)

        # Empirical probe of the local magnetization from the *current* noisy
        # batch: per-instance average of x_t over that instance's chains.
        # Detached -- it's a function of the sampled x_t (already
        # non-differentiable via multinomial), not something to backprop into.
        mk_emp = compute_local_mag(x_in[:,0], batch.instance_idx, batch.batch, batch.ptr)[1].detach()

        field_input = torch.cat((hh, mk_emp), dim=-1)  # (B(N), 2): [h_i, mk_emp_i]
        gf = self.field_in(field_input) + self.field_t_proj(t_vec)[batch.batch]  # (B(N), time_emb_dim)
        # Multi-hop message passing: field_iters independent GraphConv rounds
        # (residual + activation), each extending the receptive field by one
        # hop -- the version that empirically worked better than sharing one
        # GraphConv across iterations.
        for conv in self.field_convs:
            gf = gf + self.field_act(conv(gf, batch.edge_index))
            
        m_hat = self.field_out_act(self.m_head(gf))

        # Stop-gradient on m_hat only: its sole remaining consumer is loss_mk
        # (mse against batch.mk), so m_head is trained purely as a supervised
        # regressor for the true local magnetization -- the denoising loss can
        # never reach it. gf is left attached so field_emb still gets a training
        # signal from the denoising loss too, letting the shared representation
        # learn richer features than "whatever compresses into one scalar".
        field_emb = global_mean_pool(gf, batch.batch)   # [B, time_emb_dim]
        field = m_hat.detach()
        # field_emb =None

        x_cat = torch.cat((x_in[:, 0].unsqueeze(-1), field), dim=-1)
        x = self.in_conv(x_cat)

        # x = self.in_conv(x_in)
        
        x_residual = []
        for layer in self.encoder:
            x = layer(x, batch.edge_index, batch.batch, t_vec, field, field_emb)
            x_residual.append(x)

        x = self.latent(x, batch.edge_index, batch.batch, t_vec, field, field_emb)
        
        for i, layer in enumerate(self.decoder):
            res = x_residual[-(i + 1)]
            x = x + res 
            x = layer(x.contiguous(), batch.edge_index, batch.batch, t_vec, field, field_emb)

        # Final norm needs batch vector too
        x = self.final_norm(x, batch.batch)
        x = self.act(x)
        x = torch.cat((x, field), dim=-1)
        x = self.out_conv(x)

        return x, m_hat