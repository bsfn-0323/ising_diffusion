import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GraphConv,GCNConv,ChebConv
# from torch_geometric.nn.norm import BatchNorm,GraphNorm,LayerNorm
# from torch.nn import GroupNorm
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
        nn.init.zeros_(self.f_proj.weight)
        
        self.conv0 = GraphConv(in_ch, in_ch, bias=False)
        self.act = nn.SiLU()
        self.conv1 = GraphConv(in_ch, out_ch, bias=False)
        
        if not encoder:
            self.mlp = nn.Sequential(nn.Linear(in_ch, 2 * in_ch), nn.SiLU(), nn.Linear(2 * in_ch, in_ch))
            nn.init.zeros_(self.mlp[2].weight)
            nn.init.zeros_(self.mlp[2].bias)
            
        self.shortcut = nn.Linear(in_ch, out_ch, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x, edge_index,edge_weight, batch, t_vec, field):
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

        x = self.conv0(x, edge_index,edge_weight.view(-1))
        x = self.act(x)
        x = self.conv1(x, edge_index,edge_weight.view(-1))
        
        return x + self.shortcut(x_in)

class GNNUnet(nn.Module):
    # Removed `size` parameter from initialization
    def __init__(self, base_ch: int, ch_mult: list, time_emb_dim: int,discrete:bool=False):
        super().__init__()
        self.layer_len = len(ch_mult)
        self.time_emb_dim = time_emb_dim
        
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim, bias=False),
        )

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
            self.out_conv = nn.Linear(base_ch + 1, 2, bias=True)
            
        self.act = nn.SiLU()

    def forward(self, x_in, edge_index,edge_weight, batch, t):
        t_vec = self.time_emb(t)
        field = x_in[:, -1].unsqueeze(-1)
        
        x = self.in_conv(x_in)
        
        x_residual = []
        for layer in self.encoder:
            x = layer(x, edge_index,edge_weight, batch, t_vec, field)
            x_residual.append(x)

        x = self.latent(x, edge_index,edge_weight, batch, t_vec, field)
        
        for i, layer in enumerate(self.decoder):
            res = x_residual[-(i + 1)]
            x = x + res 
            x = layer(x.contiguous(), edge_index,edge_weight, batch, t_vec, field)

        # Final norm needs batch vector too
        x = self.final_norm(x, batch)
        x = self.act(x)
        x = torch.cat((x, field), dim=-1)
        x = self.out_conv(x)
        
        return x