import torch
import torch.nn.functional as F
from .base import DiffusionProcess

class D3PMProcess(DiffusionProcess):
    def __init__(self, betas: torch.Tensor, device: torch.device, lambda_aux: float = 0.01):
        self.device = device
        self.timesteps = len(betas) # Inherits whatever schedule length you pass
        self.K = 2 
        self.lambda_aux = lambda_aux

        self.Q_t = torch.zeros((self.timesteps, self.K, self.K), device=device)
        for t in range(self.timesteps):
            b = betas[t] # Applies your custom schedule step-by-step
            self.Q_t[t] = torch.tensor([
                [1.0 - b/2, b/2],
                [b/2, 1.0 - b/2]
            ], device=device)

        self.Q_bar_t = torch.zeros_like(self.Q_t)
        self.Q_bar_t[0] = self.Q_t[0]
        for t in range(1, self.timesteps):
            self.Q_bar_t[t] = torch.matmul(self.Q_bar_t[t-1], self.Q_t[t])

    def _spin_to_idx(self, x: torch.Tensor) -> torch.Tensor:
        """Maps physical spins {-1, 1} to indices {0, 1}."""
        return ((x + 1) / 2).long().squeeze()

    def _idx_to_spin(self, idx: torch.Tensor) -> torch.Tensor:
        """Maps indices {0, 1} back to physical spins {-1, 1}."""
        return (idx * 2 - 1).float().unsqueeze(-1)

    # def compute_loss(self, model: torch.nn.Module, batch, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #     x0_idx = self._spin_to_idx(batch.x)
    #     N = len(x0_idx)
    #     idx_arange = torch.arange(N, device=self.device)
        
    #     # 1. Forward Sample q(x_t | x_0) using direct indexing
    #     q_bar_t_batch = self.Q_bar_t[t[batch.batch]] 
    #     p_xt = q_bar_t_batch[idx_arange, x0_idx] 
        
    #     xt_idx = torch.multinomial(p_xt, num_samples=1).squeeze()
    #     xt_physical = self._idx_to_spin(xt_idx) 

    #     # 2. Model Prediction
    #     x_input = torch.cat([xt_physical, batch.field], dim=1)
    #     logits = model(x_input, batch.edge_index, batch.edge_weight, batch.batch, t.float()).float() 
    #     p_theta_x0 = F.softmax(logits, dim=-1)

    #     # 3. Auxiliary Loss
    #     loss_aux = F.cross_entropy(logits, x0_idx)

    #     # 4. Variational Lower Bound
    #     t_prev = torch.clamp(t - 1, min=0)
    #     q_t_batch = self.Q_t[t[batch.batch]]
    #     q_bar_t_prev_batch = self.Q_bar_t[t_prev[batch.batch]]
        
    #     denom = p_xt.gather(1, xt_idx.unsqueeze(1)).clamp(min=1e-6)
        
    #     # Numerator pieces via direct indexing
    #     qt_trans = q_t_batch.transpose(1, 2)[idx_arange, xt_idx] 
    #     qbar_prev_trans = q_bar_t_prev_batch[idx_arange, x0_idx] 
        
    #     q_posterior = (qt_trans * qbar_prev_trans) / denom
        
    #     p_theta_xt_prev = torch.zeros_like(p_theta_x0)
    #     for possible_x0 in range(self.K):
    #         # Directly slice the transition matrices for the hypothetical x_0
    #         test_p_xt = q_bar_t_batch[:, possible_x0, :]
    #         test_denom = test_p_xt.gather(1, xt_idx.unsqueeze(1)).clamp(min=1e-6)
            
    #         test_qbar_prev = q_bar_t_prev_batch[:, possible_x0, :]
    #         test_posterior = (qt_trans * test_qbar_prev) / test_denom
            
    #         p_theta_xt_prev += test_posterior * p_theta_x0[:, possible_x0].unsqueeze(1)

    #     loss_vb_node = F.kl_div(torch.log(p_theta_xt_prev.clamp(min=1e-6)), q_posterior, reduction='none').sum(dim=-1)
    #     loss_vb = torch.where(t[batch.batch] == 0, torch.zeros_like(loss_vb_node), loss_vb_node).mean()

    #     total_loss = loss_vb + self.lambda_aux * loss_aux
    #     x0_pred_idx = torch.argmax(logits, dim=-1)

    #     return total_loss, self._idx_to_spin(x0_pred_idx)

    # def d3pm_step(self, model, x_t, t, batch_vec, edge_index, edge_weight, field):
    #     """Optimized Discrete Denoising Step."""
    #     xt_idx = self._spin_to_idx(x_t)
    #     N = len(xt_idx)
    #     idx_arange = torch.arange(N, device=self.device)
        
    #     x_input = torch.cat([x_t, field], dim=1)
    #     logits = model(x_input, edge_index, edge_weight, batch_vec, t.float()).float()
    #     p_theta_x0 = F.softmax(logits, dim=-1)
        
    #     t_val = t[0]
    #     if t_val == 0:
    #         return self._idx_to_spin(torch.argmax(logits, dim=-1))

    #     q_t_batch = self.Q_t[t_val].unsqueeze(0).expand(N, -1, -1)
    #     q_bar_prev_batch = self.Q_bar_t[t_val - 1].unsqueeze(0).expand(N, -1, -1)
    #     q_bar_t_batch = self.Q_bar_t[t_val].unsqueeze(0).expand(N, -1, -1)
        
    #     p_theta_xt_prev = torch.zeros_like(p_theta_x0)
        
    #     # Direct indexing
    #     qt_trans = q_t_batch.transpose(1, 2)[idx_arange, xt_idx]
        
    #     for possible_x0 in range(self.K):
    #         test_denom = q_bar_t_batch[:, possible_x0, :].gather(1, xt_idx.unsqueeze(1)).clamp(min=1e-6)
    #         test_qbar_prev = q_bar_prev_batch[:, possible_x0, :]
            
    #         test_posterior = (qt_trans * test_qbar_prev) / test_denom
    #         p_theta_xt_prev += test_posterior * p_theta_x0[:, possible_x0].unsqueeze(1)
            
    #     xt_prev_idx = torch.multinomial(p_theta_xt_prev, num_samples=1).squeeze()
    #     return self._idx_to_spin(xt_prev_idx)
    
    def compute_loss(self, model: torch.nn.Module, batch, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0_idx = self._spin_to_idx(batch.x)
        N = len(x0_idx)
        idx_arange = torch.arange(N, device=self.device)
        
        # 1. Forward Sample q(x_t | x_0)
        q_bar_t_batch = self.Q_bar_t[t[batch.batch]] 
        p_xt = q_bar_t_batch[idx_arange, x0_idx] 
        xt_idx = torch.multinomial(p_xt, num_samples=1).squeeze()
        xt_physical = self._idx_to_spin(xt_idx) 

        # 2. Model Prediction (BCE Setup)
        x_input = torch.cat([xt_physical, batch.field], dim=1)
        # Squeeze the (N, 1) output to (N,)
        logits = model(x_input, batch.edge_index, batch.batch, t.float()).float().squeeze(-1)
        
        # Reconstruct full probability distribution for KL Divergence
        p_plus = torch.sigmoid(logits) 
        p_theta_x0 = torch.stack([1.0 - p_plus, p_plus], dim=-1) # Shape: (N, 2)

        # 3. Auxiliary Loss (using numerically stable BCE)
        loss_aux = F.binary_cross_entropy_with_logits(logits, x0_idx.float())

        # 4. Variational Lower Bound
        t_prev = torch.clamp(t - 1, min=0)
        q_t_batch = self.Q_t[t[batch.batch]]
        q_bar_t_prev_batch = self.Q_bar_t[t_prev[batch.batch]]
        
        denom = p_xt.gather(1, xt_idx.unsqueeze(1)).clamp(min=1e-6)
        
        qt_trans = q_t_batch.transpose(1, 2)[idx_arange, xt_idx] 
        qbar_prev_trans = q_bar_t_prev_batch[idx_arange, x0_idx] 
        
        q_posterior = (qt_trans * qbar_prev_trans) / denom
        
        p_theta_xt_prev = torch.zeros_like(p_theta_x0)
        for possible_x0 in range(self.K):
            test_p_xt = q_bar_t_batch[:, possible_x0, :]
            test_denom = test_p_xt.gather(1, xt_idx.unsqueeze(1)).clamp(min=1e-6)
            
            test_qbar_prev = q_bar_t_prev_batch[:, possible_x0, :]
            test_posterior = (qt_trans * test_qbar_prev) / test_denom
            
            p_theta_xt_prev += test_posterior * p_theta_x0[:, possible_x0].unsqueeze(1)

        loss_vb_node = F.kl_div(torch.log(p_theta_xt_prev.clamp(min=1e-6)), q_posterior, reduction='none').sum(dim=-1)
        loss_vb = torch.where(t[batch.batch] == 0, torch.zeros_like(loss_vb_node), loss_vb_node).mean()

        total_loss = loss_vb + self.lambda_aux * loss_aux
        
        # Logit > 0 corresponds to probability > 0.5
        x0_pred_idx = (logits > 0.0).long()

        return total_loss, self._idx_to_spin(x0_pred_idx)

    def d3pm_step(self, model, x_t, t, batch_vec, edge_index, field):
        xt_idx = self._spin_to_idx(x_t)
        N = len(xt_idx)
        idx_arange = torch.arange(N, device=self.device)
        
        x_input = torch.cat([x_t, field], dim=1)
        logits = model(x_input, edge_index, batch_vec, t.float()).float().squeeze(-1)
        
        p_plus = torch.sigmoid(logits)
        p_theta_x0 = torch.stack([1.0 - p_plus, p_plus], dim=-1)
        
        t_val = t[0]
        if t_val == 0:
            return self._idx_to_spin((logits > 0.0).long())

        q_t_batch = self.Q_t[t_val].unsqueeze(0).expand(N, -1, -1)
        q_bar_prev_batch = self.Q_bar_t[t_val - 1].unsqueeze(0).expand(N, -1, -1)
        q_bar_t_batch = self.Q_bar_t[t_val].unsqueeze(0).expand(N, -1, -1)
        
        p_theta_xt_prev = torch.zeros_like(p_theta_x0)
        qt_trans = q_t_batch.transpose(1, 2)[idx_arange, xt_idx]
        
        for possible_x0 in range(self.K):
            test_denom = q_bar_t_batch[:, possible_x0, :].gather(1, xt_idx.unsqueeze(1)).clamp(min=1e-6)
            test_qbar_prev = q_bar_prev_batch[:, possible_x0, :]
            
            test_posterior = (qt_trans * test_qbar_prev) / test_denom
            p_theta_xt_prev += test_posterior * p_theta_x0[:, possible_x0].unsqueeze(1)
            
        xt_prev_idx = torch.multinomial(p_theta_xt_prev, num_samples=1).squeeze()
        return self._idx_to_spin(xt_prev_idx)
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        """Forward diffuses clean fixed spins x_0 to time t."""
        x0_idx = self._spin_to_idx(x_0)
        idx_arange = torch.arange(len(x0_idx), device=self.device)
        
        # Retrieve the cumulative transition matrices for the batch
        q_bar_t_batch = self.Q_bar_t[t[batch_vec]]
        
        # Get transition probabilities for the specific clean state
        p_xt = q_bar_t_batch[idx_arange, x0_idx]
        
        # Sample the noisy fixed state
        xt_idx = torch.multinomial(p_xt, num_samples=1).squeeze()
        return self._idx_to_spin(xt_idx)

    def ddpm_step(self, *args, **kwargs):
        raise NotImplementedError("For discrete processes, use d3pm_step instead of ddpm_step.")

    def ddim_step(self, *args, **kwargs):
        raise NotImplementedError("DDIM skipping is not mathematically supported for discrete Markov chains.")
