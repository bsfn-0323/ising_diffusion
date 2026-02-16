import torch
import os
import numpy as np
from accelerate import Accelerator

# Import our refactored modules
from processes.sde import ContinuousVPSDE, get_cosine_schedule
from models.unetGnn import GNNUnet
from engines.trainer import IsingTrainer
from engines.samplers import IsingSampler

# Mock dataset imports (replace with your actual PyG Dataset)
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data,Dataset
from torch.utils.data import random_split

class MyDataset(Dataset):
    def __init__(self, x_all, field, edge_index,edge_weight, augment=False):
        super().__init__()
        # Ensure everything is a float32 tensor
        self.x_all = torch.tensor(x_all, dtype=torch.float32)
        self.field = torch.tensor(field, dtype=torch.float32)
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.augment = augment

    def len(self):
        return self.x_all.size(0)

    def get(self, idx):
        x = self.x_all[idx].view(-1, 1)      # [N, 1]
        field = self.field.view(-1, 1)       # [N, 1]
        edge_index = self.edge_index
        edge_weight = self.edge_weight
        if self.augment and torch.rand(1).item() < 0.5:
            x = -x
            field = -field

        # Crucial: edge_index must be inside the Data object
        return Data(x=x, field=field, edge_index=self.edge_index,edge_weight=self.edge_weight)

def main():
    # 1. Hardware Initialization
    # V100 uses "fp16". When moving to H200, change this to "bf16" for better numerical stability.
    Tmin=1.5
    Tmax=2.5
    dB = (1/Tmin - 1/Tmax)/20
    betas = np.linspace(1/Tmax, 1/Tmin -dB ,20)
    
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    print(f"[*] Running on device: {device} | Precision: {accelerator.mixed_precision}")

    # 2. Mathematical Definition
    timesteps = 1000
    betas = get_cosine_schedule(timesteps)
    process = ContinuousVPSDE(betas=betas, device=device)

    # 3. Model Architecture
    model = GNNUnet(base_ch=32, ch_mult=[1, 2, 4], time_emb_dim=32)
    model = model.to(device)

    # ---------------------------------------------------------
    # 4. HIGH-PERFORMANCE COMPILATION (V100 & H200)
    # ---------------------------------------------------------
    # - mode="reduce-overhead": Best for models with many small layers (like GNNs).
    # - dynamic=True: Prevents recompilation when graph sizes (N) change across batches.
    # - fullgraph=True: Optional, but forces the compiler to error out if it finds a Python fallback (graph break).
    #   Remove fullgraph=True if you encounter unresolvable PyG compilation issues.
    print("[*] Compiling model...")
    # compiled_model = torch.compile(model,backend="cudagraphs")
    compiled_model = model
    # 5. Data Setup (Placeholder)
    # Replace with your actual Ising dataset loading
    data = np.fromfile("/mnt/beegfs/2a/sb12724/rfim_pt/rfim_rrg_L12_Tmin1.500_Tmax2.500_hstd0.200_M20_10/config_rank0.bin",dtype = np.int32).reshape(-1,144,1)
    # dummy_data = [Data(x=torch.randn(100, 1), edge_index=torch.randint(0, 100, (2, 300)), field=torch.randn(100,1)) for _ in range(2048)]
    Js = np.fromfile("/mnt/beegfs/2a/sb12724/rfim_pt/rfim_rrg_L12_Tmin1.500_Tmax2.500_hstd0.200_M20_10/Jmat.bin",dtype=np.int32).reshape(144,144)
    Js = torch.from_numpy(Js)
    mk = data.mean(0)
    edge_index = torch.tensor(np.array(np.nonzero(Js)), dtype=torch.long)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.t().contiguous() # Force [2, E]
        
    edge_weight = torch.full((edge_index.size(1), 1), betas[18])
    ds= MyDataset(data[:10240],mk.reshape(144,1),edge_index,edge_weight)
    train_ds ,val_ds = random_split(ds, [1-0.2, 0.2])
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True,num_workers = 4,pin_memory=False,prefetch_factor=2, persistent_workers=True)
    batch = next(iter(train_loader))
    print(f"Batch edge_index shape: {batch.edge_index.shape}")
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    # Physics parameters (Placeholder for J and h matrices)
    # Js = torch.randn(128, 100, 100) # [Batch, N, N]
    hs = torch.randn(128, 100)      # [Batch, N]

    # 6. Optimization
    optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=0.004, weight_decay=1e-5)
    total_steps = len(train_loader) * 1000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.004/100)

    # 7. Accelerate Preparation
    # compiled_model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    #     compiled_model, optimizer, train_loader, val_loader, scheduler
    # )
    state_dict = torch.load("/mnt/beegfs/2a/sb12724/ising_diffusion_project/lib/checkpoints/ising_diffusion/epoch_best.bin",map_location = "cuda",weights_only=True)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    compiled_model.load_state_dict(state_dict)
    compiled_model.to("cuda")
    compiled_model.eval()
    compiled_model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        compiled_model, optimizer, train_loader, val_loader, scheduler
    )
    # 8. Unified Trainer
    save_path = "./checkpoints/ising_diffusion"
    trainer = IsingTrainer(
        model=compiled_model,
        process=process,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        Js=Js,
        hs=hs,
        epochs=1000,
        save_path=save_path
    )

    # print("[*] Starting Training...")
    # trainer.train() # Uncomment to train

    # 9. Unified Sampler (Generation)
    print("[*] Initializing Sampler...")
    sampler = IsingSampler(model=compiled_model, process=process, device=device)
    
    # Example Sampling Call
    # N = 100, extracting edges and field from the first dummy graph
    # sample_edges = dummy_data[0].edge_index
    # sample_field = dummy_data[0].field
    
    generated_spins = sampler.sample_ddpm(batch_size=2048, edge_index_single=edge_index,edge_weight_single=edge_weight, field_single=torch.from_numpy(mk).float())
    # print(f"[*] Generated Batch Shape: {generated_spins.shape}") # Expected: [8, 100, 1]
    import matplotlib.pyplot as plt
    plt.hist(generated_spins.sign().reshape(-1,144).cpu().mean(-1),bins = 31,range=(-1,1),alpha=0.5,density=True)
    plt.hist(data[:10240].reshape(-1,144).mean(-1),bins = 31,range=(-1,1),alpha=0.5,density=True)
    plt.savefig("gnegne.pdf")

if __name__ == "__main__":
    main()