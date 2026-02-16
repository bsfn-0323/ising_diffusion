import time
import torch
import numpy as np
from tqdm import tqdm

class IsingSampler:
    def __init__(self, model: torch.nn.Module, process, device: torch.device, accelerator):
        self.model = model.eval()
        self.process = process # Abstraction: ContinuousVPSDE or D3PMProcess
        self.device = device
        self.accelerator = accelerator
        # self.N = model.size

    def _prepare_batch(self, batch_size: int, edge_index_single: torch.Tensor,edge_weight_single: torch.Tensor, field_single: torch.Tensor):
        N = field_single.shape[0]
        # Ensure field is (N, 1) before expanding
        field_single = field_single.reshape(N, 1).to(self.device)
        field_expand = field_single.repeat(batch_size, 1)

        # Initialize noise dynamically based on the process type
        if hasattr(self.process, 'K'): 
            # D3PM (Discrete): Uniformly sample categorical states and map to {-1, 1}
            x = (torch.randint(0, self.process.K, (batch_size * N, 1), device=self.device) * 2 - 1).float()
        else:
            # Continuous SDE: Gaussian noise
            x = torch.randn(batch_size * N, 1, device=self.device)
        
        offset = (torch.arange(batch_size, device=self.device) * N).view(-1, 1, 1)
        edge_index = (edge_index_single.to(self.device).unsqueeze(0) + offset).permute(1, 0, 2).reshape(2, -1)
        edge_weight = edge_weight_single.to(self.device).repeat(batch_size, 1)
        batch_vec = torch.arange(batch_size, device=self.device).repeat_interleave(N)
        
        return x, field_expand, edge_index,edge_weight, batch_vec,N

    @torch.no_grad()
    def sample_ddpm(self, batch_size: int, edge_index_single: torch.Tensor,edge_weight_single: torch.Tensor, field_single: torch.Tensor):
        """Standard Ancestral Sampling abstracted through the Process interface."""
        x, field_expand, edge_index,edge_weight, batch_vec,N = self._prepare_batch(batch_size, edge_index_single,edge_weight_single, field_single)

        start_time = time.perf_counter()
        for i in tqdm(reversed(range(0, self.process.timesteps)), desc="DDPM Sampling", leave=False):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # Delegate the mathematical step to the process
            x = self.process.ddpm_step(
                model=self.model, x_t=x, t=t, 
                batch_vec=batch_vec, edge_index=edge_index,edge_weight=edge_weight, field=field_expand
            )

        print(f"Sampling finished in {time.perf_counter() - start_time:.4f}s")
        return x.view(batch_size, N, 1)

    @torch.no_grad()
    def sample_ddim(self, batch_size: int, num_steps: int, edge_index_single: torch.Tensor,edge_weight_single: torch.Tensor, field_single: torch.Tensor, eta: float = 0.0):
        x, field_expand, edge_index,edge_weight, batch_vec,N = self._prepare_batch(batch_size, edge_index_single,edge_weight_single, field_single)
        taus = np.linspace(self.process.timesteps - 1, 0, num_steps, dtype=int)

        for i in tqdm(range(len(taus) - 1), desc="DDIM Sampling", leave=False):
            t, t_next = int(taus[i]), int(taus[i+1])
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_next_tensor = torch.full((batch_size,), t_next, device=self.device, dtype=torch.long)

            # Delegate to the process
            x = self.process.ddim_step(
                model=self.model, x_t=x, t=t_tensor, t_next=t_next_tensor, 
                batch_vec=batch_vec, edge_index=edge_index,edge_weight=edge_weight, field=field_expand, eta=eta
            )

        return x.view(batch_size, N, 1)
    
    @torch.no_grad()
    def sample_d3pm(self, batch_size: int, edge_index_single: torch.Tensor, edge_weight_single: torch.Tensor, field_single: torch.Tensor):
        x, field_expand, edge_index, edge_weight, batch_vec, N = self._prepare_batch(batch_size, edge_index_single, edge_weight_single, field_single)

        start_time = time.perf_counter()
        for i in tqdm(reversed(range(0, self.process.timesteps)), desc="D3PM Sampling", leave=False):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # Call the newly named d3pm_step
            x = self.process.d3pm_step(
                model=self.model, x_t=x, t=t, 
                batch_vec=batch_vec, edge_index=edge_index, edge_weight=edge_weight, field=field_expand
            )

        print(f"Sampling finished in {time.perf_counter() - start_time:.4f}s")
        return x.view(batch_size, N, 1)

    @torch.no_grad()
    def sample_distributed(self, total_samples: int, method: str, max_vram_batch: int = 256, **kwargs):
        """
        Memory-safe distributed sampling using sub-batching.
        max_vram_batch: The maximum number of Ising systems to process on one GPU at a time.
        """
        if self.accelerator is None:
            raise ValueError("Accelerator required.")

        # 1. Determine local workload
        indices = list(range(total_samples))
        with self.accelerator.split_between_processes(indices) as local_indices:
            n_local = len(local_indices)

        local_results = []
        
        # 2. Sequential Chunking: Process n_local in small pieces
        # This prevents VRAM spikes for large N
        for i in range(0, n_local, max_vram_batch):
            current_batch_size = min(max_vram_batch, n_local - i)
            
            if method == 'd3pm':
                chunk = self.sample_d3pm(batch_size=current_batch_size, **kwargs)
            elif method == 'ddpm':
                chunk = self.sample_ddpm(batch_size=current_batch_size, **kwargs)
            
            # Move to CPU immediately to free up VRAM for the next chunk
            local_results.append(chunk.cpu()) 

        # 3. Concatenate local CPU chunks
        if len(local_results) > 0:
            local_samples_cpu = torch.cat(local_results, dim=0)
        else:
            local_samples_cpu = torch.empty(0, kwargs['field_single'].shape[0], 1)

        # 4. Gather to Main Process (using CPU to avoid VRAM gather crash)
        # We use a simple gather because gather_for_metrics usually expects GPU tensors
        all_samples_list = [torch.zeros_like(local_samples_cpu) for _ in range(self.accelerator.num_processes)]
        
        # This gathers the CPU tensors across nodes/GPUs
        if self.accelerator.num_processes > 1:
            import torch.distributed as dist
            # Note: Accelerator's gather works better with its own utilities, 
            # but for very large N, we often manual-save to disk per rank.
            all_samples = self.accelerator.gather(local_samples_cpu.to(self.device)).cpu()
        else:
            all_samples = local_samples_cpu

        # Trim padding if necessary
        return all_samples[:total_samples]