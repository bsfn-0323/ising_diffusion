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

    def forward(self, x, edge_index, batch, t_vec, field):
        x_in = x

        # MUST pass batch here now for dynamic pooling
        x = self.norm(x, batch)

        style_t = self.t_proj(t_vec)
        style_h = self.f_proj(field)

        cond = style_t[batch] + style_h
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
    # Weight-shared iterated message passing: the same GNNLayer is applied
    # `num_iters` times, mirroring belief propagation / the cavity method --
    # a single local update rule run to a fixed point, rather than a
    # feature-pyramid stack of distinct per-depth layers. This forces a fixed
    # message width across iterations (just like real BP messages are
    # constant-size sufficient statistics regardless of subtree size), and is
    # the inductive bias that should let the network generalize the implicit
    # "solve (h, J, topology) -> <s_k>" computation across new disorder
    # instances from few training examples.
    def __init__(self, base_ch: int, time_emb_dim: int, discrete: bool = False, num_iters: int = 10):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.base_ch = base_ch
        self.num_iters = num_iters
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim, bias=False),
        )

        self.in_conv = nn.Linear(2, base_ch, bias=False)

        self.block = GNNLayer(base_ch, base_ch, time_emb_dim, encoder=False)

        self.final_norm = RMSGraphNorm(base_ch, affine=False)
        self.out_conv = nn.Linear(base_ch + 1, 1, bias=True)

        self.act = nn.SiLU()

    def forward(self, x_in, batch, t):
        t_vec = self.time_emb(t)

        # Conditioning is just the raw local field h_i (x_in = [x_t, h_i],
        # already (B(N), 2) -- matches in_conv's expected width directly).
        # No learned field/global embedding: the denoiser's GraphConv message
        # passing is the only thing that has to turn (h, J/topology) into
        # something resembling <s_k> -- see compute_loss/loss_mk for how that
        # is now supervised directly through the network's own x0 predictions.
        hh = x_in[:, -1].unsqueeze(-1)
        field = hh

        x = self.in_conv(x_in)

        for _ in range(self.num_iters):
            x = self.block(x, batch.edge_index, batch.batch, t_vec, field)

        # Final norm needs batch vector too
        x = self.final_norm(x, batch.batch)
        x = self.act(x)
        x = torch.cat((x, field), dim=-1)
        x = self.out_conv(x)

        return x