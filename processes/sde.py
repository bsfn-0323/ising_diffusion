import math
import torch
import torch.nn.functional as F
from .base import DiffusionProcess

def get_cosine_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Generates a cosine noise schedule."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def get_linear_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Generates a linear noise schedule."""
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class ContinuousVPSDE(DiffusionProcess):
    """Variance Preserving SDE for continuous graph-based spins."""
    
    def __init__(self, betas: torch.Tensor, device: torch.device):
        self.device = device
        self.timesteps = len(betas)
        self.beta_t = betas.to(device).view(-1, 1)
        self.alpha_t = 1.0 - self.beta_t
        self.alphacum_t = torch.cumprod(self.alpha_t, dim=0)
        self.sqrt_alphacum = torch.sqrt(self.alphacum_t)
        self.sqrt_one_minus_alphacum = torch.sqrt(1.0 - self.alphacum_t)

    def _add_noise(self, x_0: torch.Tensor, t: torch.Tensor, batch_vec: torch.Tensor):
        mean = self.sqrt_alphacum[t][batch_vec] * x_0
        std = self.sqrt_one_minus_alphacum[t][batch_vec]
        noise = torch.randn_like(x_0)
        return mean + std * noise, noise

    def compute_loss(self, model: torch.nn.Module, batch, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_noised, noise = self._add_noise(batch.x, t, batch.batch)
        x_input = torch.cat([x_noised, batch.field], dim=1)
        
        # temb = model.time_emb(t.float())
        noise_pred = model(x_input, batch.edge_index,batch.edge_weight, batch.batch, t).reshape(noise.shape[0], 1)
        
        loss_mse = F.mse_loss(noise_pred, noise)
        
        # Predict x_0 analytically from noise_pred for correlation loss
        std_coeff = self.sqrt_one_minus_alphacum[t][batch.batch]
        mean_coeff = self.sqrt_alphacum[t][batch.batch]
        x0_pred = (x_noised - std_coeff * noise_pred) / mean_coeff
        
        return loss_mse, x0_pred

    def ddpm_step(self, model: torch.nn.Module, x_t: torch.Tensor, t: torch.Tensor, batch_vec: torch.Tensor, edge_index: torch.Tensor,edge_weight: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        # real_model = model.module if hasattr(model, 'module') else model
        # temb = real_model.time_emb(t.float())
        x_input = torch.cat([x_t, field], dim=1)
        eps_pred = model(x_input, edge_index,edge_weight, batch_vec, t.float()).reshape(-1, 1)

        t_val = t[0]
        alpha_bar_t = self.alphacum_t[t_val]
        alpha_bar_prev = self.alphacum_t[t_val-1] if t_val > 0 else torch.tensor(1.0, device=self.device)
        beta_t = 1 - (alpha_bar_t / alpha_bar_prev)

        coef_x0 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
        coef_xt = (torch.sqrt(alpha_bar_t / alpha_bar_prev) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)

        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        posterior_mean = coef_x0 * x0_pred + coef_xt * x_t

        if t_val > 0:
            noise = torch.randn_like(x_t)
            posterior_var = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
            return posterior_mean + torch.exp(0.5 * torch.log(posterior_var.clamp(min=1e-20))) * noise
        return posterior_mean

    def ddim_step(self, model: torch.nn.Module, x_t: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, batch_vec: torch.Tensor, edge_index: torch.Tensor,edge_weight: torch.Tensor, field: torch.Tensor, eta: float) -> torch.Tensor:
        # temb = model.time_emb(t.float())
        # x_input = torch.cat([x_t, field], dim=1)
        eps_pred = model(x_input, edge_index,edge_weight, batch_vec, t.float()).reshape(-1, 1)

        t_val, t_next_val = t[0], t_next[0]
        alpha_t = self.alphacum_t[t_val]
        alpha_t_next = self.alphacum_t[t_next_val] if t_next_val >= 0 else torch.tensor(1.0, device=self.device)
        
        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
        sigma = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next))
        
        noise = torch.randn_like(x_t) if t_next_val > 0 else 0.0

        return (torch.sqrt(alpha_t_next) * x0_pred + 
                torch.sqrt(1 - alpha_t_next - sigma**2) * eps_pred + 
                sigma * noise)
        
    def d3pm_step(self, *args, **kwargs):
        raise NotImplementedError("For continous processes, use ddpm_step or ddim_step instead of d3pm_step.")
