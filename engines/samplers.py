import time
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch


# class MultiFieldIsingDataset(Dataset):
#     def __init__(self, x_list=None, magn_list=None, field_list=None, edge_index_list=None, augment=False, load_path=None):
#         super().__init__()
#         self.augment = augment
        
#         if load_path and os.path.exists(load_path):
#             # Load the processed tensors from disk
#             data_dict = torch.load(load_path)
#             self.x_all = data_dict['x_all']
#             self.mks = data_dict['mks']
#             self.fields = data_dict['fields']
#             self.edge_indices = data_dict['edge_indices']
#             self.sample_to_instance = data_dict['sample_to_instance']
#             print(f"[*] Dataset loaded from {load_path}")
#         else:
#             # Process the raw lists into tensors
#             self.edge_indices = [torch.tensor(ei, dtype=torch.long) for ei in edge_index_list]
#             self.mks = [torch.tensor(m, dtype=torch.float32).view(-1, 1) for m in magn_list]
#             self.fields = [torch.tensor(f, dtype=torch.float32).view(-1, 1) for f in field_list]

#             all_x = []
#             sample_to_instance = []
            
#             for i, x_samples in enumerate(x_list):
#                 x_tensor = torch.tensor(x_samples, dtype=torch.float32) # [N_samples, 144, 1]
#                 all_x.append(x_tensor)
#                 # Map these samples to the i-th instance (field/topology)
#                 sample_to_instance.extend([i] * x_tensor.size(0))
            
#             self.x_all = torch.cat(all_x, dim=0)
#             self.sample_to_instance = torch.tensor(sample_to_instance, dtype=torch.long)
#             print(f"[*] Dataset built: {self.x_all.size(0)} total samples.")

#     def save(self, path):
#         """Saves the processed dataset to a single file."""
#         torch.save({
#             'x_all': self.x_all,
#             'mks': self.mks,
#             'fields': self.fields,
#             'edge_indices': self.edge_indices,
#             'sample_to_instance': self.sample_to_instance
#         }, path)
#         print(f"[*] Dataset saved to {path}")
    
#     def len(self):
#         return self.x_all.size(0)

#     def get(self, idx):
#         # 1. Identify which realization (instance) this sample belongs to
#         instance_idx = self.sample_to_instance[idx].item()
        
#         # 2. Slice the configuration [144, 1]
#         x = self.x_all[idx].view(-1, 1) 
        
#         # 3. Retrieve the correct field [144, 1] and edge_index [2, E]
#         mk = self.mks[instance_idx]
#         field = self.fields[instance_idx]
#         edge_index = self.edge_indices[instance_idx]
        
#         # Consistent Z2 symmetry augmentation
#         if self.augment and torch.rand(1).item() < 0.5:
#             x, field = -x, -field
#             mk = -mk
#         return Data(x=x,mk=mk, field=field, edge_index=edge_index,instance_idx=torch.tensor(instance_idx, dtype=torch.long))


class IsingSampler:
    def __init__(self, model: torch.nn.Module, process, device: torch.device, accelerator):
        self.model = model.eval()
        self.process = process # Abstraction: ContinuousVPSDE or D3PMProcess
        self.device = device
        self.accelerator = accelerator
        # self.N = model.size

    def _prepare_batch(self, batch_size: int, edge_index_single: torch.Tensor, field_single: torch.Tensor):
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
        # edge_weight = edge_weight_single.to(self.device).repeat(batch_size, 1)
        batch_vec = torch.arange(batch_size, device=self.device).repeat_interleave(N)
        
        return x, field_expand, edge_index, batch_vec,N
        # return x, field_expand, edge_index,edge_weight, batch_vec,N

    # def _prepare_batch_v2(self,batch_size:int, edge_index_single: torch.Tensor, field_single: torch.Tensor,mk_single : torch.Tensor ):
    #     N = field_single.shape[0]
    #     # Ensure field is (N, 1) before expanding
    #     field_single = field_single.reshape(N, 1).to(self.device)
    #     field_expand = field_single.repeat(batch_size, 1)

    #     # Initialize noise dynamically based on the process type
    #     if hasattr(self.process, 'K'): 
    #         # D3PM (Discrete): Uniformly sample categorical states and map to {-1, 1}
    #         x = (torch.randint(0, self.process.K, (batch_size, N, 1), device=self.device) * 2 - 1).float()
    #     else:
    #         # Continuous SDE: Gaussian noise
    #         x = torch.randn(batch_size, N, 1, device=self.device)

    #     edge_index_single = edge_index_single.to(self.device)
    #     field_single = field_single.to(self.device)
    #     mk_single = mk_single.to(self.device)

    #     ds = MultiFieldIsingDataset([x],[mk_single], [field_single], [edge_index_single], augment=False)
    #     data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    #     return x,next(iter(data_loader))
    def _prepare_batch_v2(self, batch_size, edge_index_single, field_single, mk_single=None):
        N = field_single.shape[0]
        device = self.device
        field_single = field_single.reshape(N, 1).to(device)
        edge_index_single = edge_index_single.to(device)

        # init noise
        if hasattr(self.process, 'K'):
            x = (torch.randint(0, self.process.K, (batch_size * N, 1), device=device) * 2 - 1).float()
        else:
            x = torch.randn(batch_size * N, 1, device=device)

        # build K chains as K graphs of the SAME instance
        data_list = []
        for k in range(batch_size):
            data_list.append(
                Data(
                x=x[k*N:(k+1)*N],                       # [N,1]
                field=field_single,                      # [N,1]
                edge_index=edge_index_single,            # [2,E]
                instance_idx=torch.zeros(1, dtype=torch.long, device=device),
            )
            )
        batch = Batch.from_data_list(data_list).to(device)
        return x, batch
    @torch.no_grad()
    def sample_ddpm(self, batch_size: int, edge_index_single: torch.Tensor, field_single: torch.Tensor):
        """Standard Ancestral Sampling abstracted through the Process interface."""
        # x, field_expand, edge_index,edge_weight, batch_vec,N = self._prepare_batch(batch_size, edge_index_single,edge_weight_single, field_single)
        x, field_expand, edge_index, batch_vec,N = self._prepare_batch(batch_size, edge_index_single, field_single)

        start_time = time.perf_counter()
        for i in tqdm(reversed(range(0, self.process.timesteps)), desc="DDPM Sampling", leave=False):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # Delegate the mathematical step to the process
            x = self.process.ddpm_step(
                model=self.model, x_t=x, t=t, 
                batch_vec=batch_vec, edge_index=edge_index, field=field_expand
            )

        print(f"Sampling finished in {time.perf_counter() - start_time:.4f}s")
        return x.view(batch_size, N, 1)

    @torch.no_grad()
    def sample_ddim(self, batch_size: int, num_steps: int, edge_index_single: torch.Tensor, field_single: torch.Tensor, eta: float = 0.0):
        x, field_expand, edge_index, batch_vec,N = self._prepare_batch(batch_size, edge_index_single, field_single)
        # x, field_expand, edge_index,edge_weight, batch_vec,N = self._prepare_batch(batch_size, edge_index_single,edge_weight_single, field_single)
        
        taus = np.linspace(self.process.timesteps - 1, 0, num_steps, dtype=int)

        for i in tqdm(range(len(taus) - 1), desc="DDIM Sampling", leave=False):
            t, t_next = int(taus[i]), int(taus[i+1])
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_next_tensor = torch.full((batch_size,), t_next, device=self.device, dtype=torch.long)

            # Delegate to the process
            x = self.process.ddim_step(
                model=self.model, x_t=x, t=t_tensor, t_next=t_next_tensor, 
                batch_vec=batch_vec, edge_index=edge_index, field=field_expand, eta=eta
            )

        return x.view(batch_size, N, 1)
    
    @torch.no_grad()
    def sample_d3pm(self, batch_size: int, edge_index_single: torch.Tensor, field_single: torch.Tensor):
        # x, field_expand, edge_index, edge_weight, batch_vec, N = self._prepare_batch(batch_size, edge_index_single, edge_weight_single, field_single)
        # x, field_expand, edge_index, batch_vec, N = self._prepare_batch(batch_size, edge_index_single, field_single)
        x,batch = self._prepare_batch_v2(batch_size, edge_index_single, field_single,field_single)
        start_time = time.perf_counter()
        for i in tqdm(reversed(range(0, self.process.timesteps)), desc="D3PM Sampling", leave=False):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # Call the newly named d3pm_step
            # x = self.process.d3pm_step(
            #     model=self.model, x_t=x, t=t, 
            #     batch_vec=batch_vec, edge_index=edge_index,field=field_expand
            # )
            x = self.process.d3pm_step(
                model=self.model, x_t=x, t=t, 
                batch=batch
            )

        print(f"Sampling finished in {time.perf_counter() - start_time:.4f}s")
        return x.view(batch_size, -1, 1)
    
    @torch.no_grad()
    def sample_d3pm_conditional(self, batch_size: int, edge_index_single: torch.Tensor, field_single: torch.Tensor, mask_single: torch.Tensor, x0_fixed_single: torch.Tensor):
        x, field_expand, edge_index, batch_vec, N = self._prepare_batch(batch_size, edge_index_single, field_single)
        
        # Expand masks and fixed states across the batch dimension
        mask_expand = mask_single.to(self.device).unsqueeze(1).repeat(batch_size, 1)
        x0_fixed_expand = x0_fixed_single.to(self.device).repeat(batch_size, 1)
        
        start_time = time.perf_counter()
        for i in tqdm(reversed(range(0, self.process.timesteps)), desc="D3PM Conditional Sampling", leave=False):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # 1. Enforce known boundary state at noise level t
            x_known = self.process.q_sample(x0_fixed_expand, t, batch_vec)
            
            # 2. Overwrite the boundary nodes (False in mask) with the known noisy state
            x = torch.where(mask_expand, x, x_known)
            
            # 3. Predict the next less-noisy state using the GNN
            x = self.process.d3pm_step(
                model=self.model, x_t=x, t=t, 
                batch_vec=batch_vec, edge_index=edge_index, field=field_expand
            )

        # 4. Final step: Ensure the boundary exactly matches the clean ground truth
        x = torch.where(mask_expand, x, x0_fixed_expand)

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

        # If using batched conditioning, extract them from kwargs
        if method == 'd3pm_cond_batched':
            full_x0_batch = kwargs.pop('x0_fixed_batch')
            # full_field_batch = kwargs.pop('field_batch')
            
        # 1. Determine local workload
        indices = list(range(total_samples))
        with self.accelerator.split_between_processes(indices) as local_indices:
            n_local = len(local_indices)

        # IMPORTANT: Slice the full batched conditions to match the GPU's local workload
            if method == 'd3pm_cond_batched':
                local_x0 = full_x0_batch[local_indices]
                # local_field = full_field_batch[local_indices]
                
        local_results = []
        
        # 2. Sequential Chunking: Process n_local in small pieces
        # This prevents VRAM spikes for large N
        for i in range(0, n_local, max_vram_batch):
            current_batch_size = min(max_vram_batch, n_local - i)
            
            if method == 'd3pm':
                chunk = self.sample_d3pm(batch_size=current_batch_size, **kwargs)
            elif method == 'ddpm':
                chunk = self.sample_ddpm(batch_size=current_batch_size, **kwargs)
            elif method == 'd3pm_bound_cond':
                chunk = self.sample_d3pm_conditional(batch_size=current_batch_size, **kwargs)
            elif method == 'd3pm_cond_batched':
                # Slice the exact sub-chunk for this VRAM pass
                chunk_x0 = local_x0[i : i + current_batch_size]
                
                chunk = self.sample_d3pm_conditional_batched(
                    batch_size=current_batch_size,
                    edge_index_single=kwargs['edge_index_single'],
                    mask_single=kwargs['mask_single'],
                    field_single=kwargs['field_single'],
                    x0_fixed_batch=chunk_x0
                )
            # elif method == 'd3pm_repaint_cond':
            #     chunk = self.sample_d3pm_repaint_conditional(
            #         batch_size=current_batch_size, 
            #         jump_len=20, 
            #         jump_n_sample=20, 
            #         **kwargs
            #     )
            elif method == 'd3pm_repaint_cond':
                chunk = self.sample_d3pm_repaint_optimized(
                    batch_size=current_batch_size, 
                    edge_index_single=kwargs['edge_index_single'],
                    field_single=kwargs['field_single'],
                    mask_single=kwargs['mask_single'],
                    x0_fixed_single=kwargs['x0_fixed_single'],
                    jump_len=5, 
                    jump_n_sample=5, 
                    t_jump_threshold=800  # Triggers time-travel only for t < 100
                )
            
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
        # if self.accelerator.num_processes > 1:
        #     import torch.distributed as dist
        #     # Note: Accelerator's gather works better with its own utilities, 
        #     # but for very large N, we often manual-save to disk per rank.
        #     all_samples = self.accelerator.gather(local_samples_cpu.to(self.device)).cpu()
        # else:
        #     all_samples = local_samples_cpu
        if self.accelerator.num_processes > 1:
            import torch.distributed as dist
            # Create a gloo group for CPU transfers if not already done
            if not hasattr(self, '_gloo_group'):
                self._gloo_group = dist.new_group(backend='gloo')
                gathered = [torch.zeros_like(local_samples_cpu) for _ in range(self.accelerator.num_processes)]
                dist.gather(local_samples_cpu, gathered if self.accelerator.is_main_process else None, dst=0, group=self._gloo_group)
                all_samples = torch.cat(gathered, dim=0) if self.accelerator.is_main_process else local_samples_cpu
        else:
            all_samples = local_samples_cpu

        # Trim padding if necessary
        return all_samples[:total_samples]
    
    # @torch.no_grad()
    # def sample_d3pm_conditional_batched(self, batch_size: int, edge_index_single: torch.Tensor, mask_single: torch.Tensor, field_single: torch.Tensor, x0_fixed_batch: torch.Tensor):
    #     """
    #     batch_size: Number of samples in this VRAM chunk
    #     field_single: Tensor [N, 1] (Static field for the subgraph)
    #     x0_fixed_batch: Tensor [B, N, 1] (Different boundary conditions per sample)
    #     """
    #     N = field_single.shape[0]
        
    #     # 1. Prepare Graph Topology (Shared across batch)
    #     offset = (torch.arange(batch_size, device=self.device) * N).view(-1, 1, 1)
    #     edge_index = (edge_index_single.to(self.device).unsqueeze(0) + offset).permute(1, 0, 2).reshape(2, -1)
    #     batch_vec = torch.arange(batch_size, device=self.device).repeat_interleave(N)
        
    #     # 2. Expand Shared Conditions
    #     mask_expand = mask_single.to(self.device).unsqueeze(1).repeat(batch_size, 1)
    #     field_expand = field_single.to(self.device).repeat(batch_size, 1)
        
    #     # 3. Flatten Unique Batched Boundaries to [B*N, 1]
    #     x0_fixed_expand = x0_fixed_batch.to(self.device).view(batch_size * N, 1)

    #     # Initialize full noise for the batch
    #     x = (torch.randint(0, self.process.K, (batch_size * N, 1), device=self.device) * 2 - 1).float()

    #     start_time = time.perf_counter()
    #     for i in tqdm(reversed(range(0, self.process.timesteps)), desc="Batched Cond Sampling", leave=False):
    #         t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
    #         # Forward diffuse the unique batched boundary states
    #         x_known = self.process.q_sample(x0_fixed_expand, t, batch_vec)
            
    #         # Enforce the unique boundaries for each sample
    #         x = torch.where(mask_expand, x, x_known)
            
    #         # Neural Network Step (using the shared expanded field)
    #         x = self.process.d3pm_step(
    #             model=self.model, x_t=x, t=t, 
    #             batch_vec=batch_vec, edge_index=edge_index, field=field_expand
    #         )

    #     # Final cleanup to ensure exact matching of boundaries
    #     x = torch.where(mask_expand, x, x0_fixed_expand)

    #     # print(f"Sampling finished in {time.perf_counter() - start_time:.4f}s")
    #     return x.view(batch_size, N, 1)
    def sample_d3pm_conditional_batched(self, batch_size: int, edge_index_single: torch.Tensor, mask_single: torch.Tensor, field_single: torch.Tensor, x0_fixed_batch: torch.Tensor):
        # Infer N from the first dimension
        N = field_single.shape[0]
        
        # 1. Prepare Graph Topology
        offset = (torch.arange(batch_size, device=self.device) * N).view(-1, 1, 1)
        edge_index = (edge_index_single.to(self.device).unsqueeze(0) + offset).permute(1, 0, 2).reshape(2, -1)
        batch_vec = torch.arange(batch_size, device=self.device).repeat_interleave(N)
        
        # 2. Expand Shared Conditions (FIXED)
        # Force field and mask into [N, 1] before repeating to avoid [B, N] broadcast errors
        mask_expand = mask_single.to(self.device).view(-1, 1).repeat(batch_size, 1)
        field_expand = field_single.to(self.device).view(-1, 1).repeat(batch_size, 1)
        
        # 3. Flatten Unique Batched Boundaries to [B*N, 1]
        x0_fixed_expand = x0_fixed_batch.to(self.device).view(batch_size * N, 1)

        # Initialize full noise for the batch
        x = (torch.randint(0, self.process.K, (batch_size * N, 1), device=self.device) * 2 - 1).float()

        start_time = time.perf_counter()
        for i in tqdm(reversed(range(0, self.process.timesteps)), desc="Batched Cond Sampling", leave=False):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # Forward diffuse the unique batched boundary states
            x_known = self.process.q_sample(x0_fixed_expand, t, batch_vec)
            
            # Enforce the unique boundaries for each sample
            x = torch.where(mask_expand, x, x_known)
            
            # Neural Network Step
            x = self.process.d3pm_step(
                model=self.model, x_t=x, t=t, 
                batch_vec=batch_vec, edge_index=edge_index, field=field_expand
            )

        # Final cleanup to ensure exact matching of boundaries
        x = torch.where(mask_expand, x, x0_fixed_expand)

        return x.view(batch_size, N, 1)
    
    @torch.no_grad()
    def sample_d3pm_repaint_conditional(self, batch_size: int, edge_index_single: torch.Tensor, field_single: torch.Tensor, mask_single: torch.Tensor, x0_fixed_single: torch.Tensor, jump_len: int = 10, jump_n_sample: int = 10):
        # 1. Prepare standard batch
        x, field_expand, edge_index, batch_vec, N = self._prepare_batch(batch_size, edge_index_single, field_single)
        
        # Expand single condition mask and fixed states across the batch dimension
        mask_expand = mask_single.to(self.device).unsqueeze(1).repeat(batch_size, 1)
        x0_fixed_expand = x0_fixed_single.to(self.device).repeat(batch_size, 1)
        
        # 2. Compute RePaint Schedule
        schedule = []
        t = self.process.timesteps - 1
        jumps = {}
        for j in range(0, self.process.timesteps - jump_len, jump_len):
            jumps[j] = jump_n_sample - 1
            
        while t >= 0:
            schedule.append(("backward", t))
            t -= 1
            # Time Travel: Step forward 'jump_len' times if we hit a jump point
            if t in jumps and jumps[t] > 0:
                jumps[t] -= 1
                for _ in range(jump_len):
                    t += 1
                    schedule.append(("forward", t))

        start_time = time.perf_counter()
        
        # 3. Execute Schedule
        for op, t_val in tqdm(schedule, desc="RePaint Conditional", leave=False):
            t_tensor = torch.full((batch_size,), t_val, device=self.device, dtype=torch.long)
            
            if op == "backward":
                # Enforce known boundary state at noise level t
                x_known = self.process.q_sample(x0_fixed_expand, t_tensor, batch_vec)
                x = torch.where(mask_expand, x, x_known)
                
                # Predict the next less-noisy state using the GNN
                x = self.process.d3pm_step(
                    model=self.model, x_t=x, t=t_tensor, 
                    batch_vec=batch_vec, edge_index=edge_index, field=field_expand
                )
            
            elif op == "forward":
                # Forward Noise Step (t-1 -> t) using the 1-step discrete matrix Q_t
                idx_prev = self.process._spin_to_idx(x)
                q_t = self.process.Q_t[t_val] # Shape: [2, 2]
                
                # Get transition probabilities and sample
                p_xt = q_t[idx_prev] # Shape: [B*N, 2]
                xt_idx = torch.multinomial(p_xt, num_samples=1).squeeze(-1)
                x = self.process._idx_to_spin(xt_idx)

        # 4. Final step: Ensure the boundary exactly matches the clean ground truth
        x = torch.where(mask_expand, x, x0_fixed_expand)

        print(f"Sampling finished in {time.perf_counter() - start_time:.4f}s")
        return x.view(batch_size, N, 1)
    
    @torch.no_grad()
    def sample_d3pm_repaint_optimized(self, batch_size: int, edge_index_single: torch.Tensor, field_single: torch.Tensor, mask_single: torch.Tensor, x0_fixed_single: torch.Tensor, jump_len: int = 10, jump_n_sample: int = 10, t_jump_threshold: int = 100):
        # 1. Prepare standard batch
        x, field_expand, edge_index, batch_vec, N = self._prepare_batch(batch_size, edge_index_single, field_single)
        
        # Expand single condition mask and fixed states across the batch dimension
        mask_expand = mask_single.to(self.device).unsqueeze(1).repeat(batch_size, 1)
        x0_fixed_expand = x0_fixed_single.to(self.device).repeat(batch_size, 1)
        
        # 2. Compute Optimized RePaint Schedule
        schedule = []
        t = self.process.timesteps - 1
        jumps = {}
        
        # OPTIMIZATION: Only allocate jumps if t is strictly below the threshold
        for j in range(0, t_jump_threshold - jump_len + 1, jump_len):
            jumps[j] = jump_n_sample - 1
            
        while t >= 0:
            schedule.append(("backward", t))
            t -= 1
            # Time Travel: Step forward 'jump_len' times if we hit a jump point
            if t in jumps and jumps[t] > 0:
                jumps[t] -= 1
                for _ in range(jump_len):
                    t += 1
                    schedule.append(("forward", t))

        start_time = time.perf_counter()
        
        # 3. Execute Schedule
        for op, t_val in tqdm(schedule, desc="Optimized RePaint", leave=False):
            t_tensor = torch.full((batch_size,), t_val, device=self.device, dtype=torch.long)
            
            if op == "backward":
                # Enforce known boundary state at noise level t
                x_known = self.process.q_sample(x0_fixed_expand, t_tensor, batch_vec)
                x = torch.where(mask_expand, x, x_known)
                
                # Predict the next less-noisy state using the GNN
                x = self.process.d3pm_step(
                    model=self.model, x_t=x, t=t_tensor, 
                    batch_vec=batch_vec, edge_index=edge_index, field=field_expand
                )
            
            elif op == "forward":
                # Forward Noise Step (t-1 -> t) using the 1-step discrete matrix Q_t
                idx_prev = self.process._spin_to_idx(x)
                q_t = self.process.Q_t[t_val] # Shape: [2, 2]
                
                # Get transition probabilities and sample
                p_xt = q_t[idx_prev] # Shape: [B*N, 2]
                xt_idx = torch.multinomial(p_xt, num_samples=1).squeeze(-1)
                x = self.process._idx_to_spin(xt_idx)

        # 4. Final step: Ensure the boundary exactly matches the clean ground truth
        x = torch.where(mask_expand, x, x0_fixed_expand)

        print(f"Sampling finished in {time.perf_counter() - start_time:.4f}s")
        return x.view(batch_size, N, 1)