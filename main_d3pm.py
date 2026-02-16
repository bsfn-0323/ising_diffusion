import torch
import os
import numpy as np
from accelerate import Accelerator
import matplotlib.pyplot as plt

# Imports
from processes.sde import get_cosine_schedule
from processes.discrete import D3PMProcess 
from models.unetGnn import GNNUnet
from engines.trainer import IsingTrainer
from engines.samplers import IsingSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split
from utils.tracker import ExperimentTracker # Our new tracker

class MyDataset(Dataset):
    def __init__(self, x_all, field, edge_index, edge_weight, augment=False):
        super().__init__()
        self.x_all = torch.tensor(x_all, dtype=torch.float32)
        self.field = torch.tensor(field, dtype=torch.float32)
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.augment = augment

    def len(self):
        return self.x_all.size(0)

    def get(self, idx):
        x = self.x_all[idx].view(-1, 1)
        field = self.field.view(-1, 1)
        if self.augment and torch.rand(1).item() < 0.5:
            x, field = -x, -field
        return Data(x=x, field=field, edge_index=self.edge_index, edge_weight=self.edge_weight)

def main():
    # --- MODE SELECTIONTrue---
    LOAD_CHECKPOINT = False  # Set to True to skip training and just sample
    CHECKPOINT_PATH = "/home/bae/ising_diffsion_lib/lib_diffusion_ising/checkpoints/RFIM_D3PM_RRG/2026-02-15/h0.2_L12_lr0.0002/"

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    if LOAD_CHECKPOINT:
        # 1. Load existing model and config
        model, config = load_model_from_checkpoint(CHECKPOINT_PATH, device)
        exp_args = config["training_params"]
        
        # 2. Reconstruct process with saved betas
        betas_schedule = torch.tensor(config["diffusion"]["betas"])
        process = D3PMProcess(betas=betas_schedule, device=device, 
                              lambda_aux=config["diffusion"]["lambda_aux"])
        
        # We still need to prepare the model with accelerator for distributed sampling
        # model = accelerator.prepare(model)
        tracker_path = CHECKPOINT_PATH # Use existing path for results
    else:
        # # --- STANDARD TRAINING FLOW ---
        # exp_args = { "hstd": 0.2, "L": 12, "N": 144, "lr": 0.0002, "epochs": 1000, 
        #             "batch_size": 1024, "base_ch": 32, "time_emb_dim": 32 }
        
        # timesteps = 1000
        # betas_schedule = get_cosine_schedule(timesteps)
        # process = D3PMProcess(betas=betas_schedule, device=device, lambda_aux=0.1)
        # model = GNNUnet(base_ch=exp_args["base_ch"], ch_mult=[1, 2, 4], 
        #                 time_emb_dim=exp_args["time_emb_dim"])
    
        # 1. Config & Metadata
        exp_args = {
            "hstd": 0.200,
            "L": 12,
            "N": 144,
            "lr": 0.0002,
            "batch_size": 1024,
            "epochs": 1000,
            "base_ch": 32,
            "time_emb_dim": 32,
            "diffusion": "D3PM",
            "dataset_path": "/home/bae/rfim_pt/rfim_rrg_L12_Tmin1.500_Tmax2.500_hstd0.000_M20_0/config_rank14.bin"
        }
            # 3. Model & Process
        timesteps = 1000
        betas_schedule = get_cosine_schedule(timesteps)
        process = D3PMProcess(betas=betas_schedule, device=device, lambda_aux=0.1)

        model = GNNUnet(
            base_ch=exp_args["base_ch"], 
            ch_mult=[1, 2, 4], 
            time_emb_dim=exp_args["time_emb_dim"],
            # discrete=True # Set to True for D3PM logit output
        )

    # 2. Hardware & Accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    
    # 4. Data Preparation
    data = np.fromfile(exp_args["dataset_path"], dtype=np.int32).reshape(-1, 144, 1)
    Js_raw = np.fromfile("/home/bae/rfim_pt/rfim_rrg_L12_Tmin1.500_Tmax2.500_hstd0.000_M20_0/Jmat.bin", dtype=np.int32).reshape(144, 144)
    Js = torch.from_numpy(Js_raw).float()
    mk = data.mean(0)
    
    edge_index = torch.from_numpy(np.stack(np.nonzero(Js_raw))).long()
    if edge_index.shape[0] != 2:
        edge_index = edge_index.t().contiguous()
    edge_weight = torch.full((edge_index.size(1), 1), 1.0)
    
    ds = MyDataset(data[:10240], mk.reshape(144, 1), edge_index, edge_weight)
    train_ds, val_ds = random_split(ds, [0.8, 0.2])
    
    train_loader = DataLoader(train_ds, batch_size=exp_args["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    # 5. Tracker Initialization
    tracker = ExperimentTracker(
        base_path="./checkpoints", 
        experiment_name="RFIM_D3PM_RRG", 
        model=model, 
        process=process, 
        accelerator=accelerator, 
        args=exp_args
    )

    # 6. Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_args["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*exp_args["epochs"])

    # 7. Accelerate Prepare
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # 8. Training
    # save_path = "./checkpoints/ising_diffusion"
    # trainer = IsingTrainer(
    #     model=model,
    #     process=process,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     accelerator=accelerator,
    #     tracker=tracker, # Pass the tracker here
    #     epochs=exp_args["epochs"],
    #     save_path=save_path
    # )

    # print(f"[*] Starting Training on {device}...")
    # trainer.train()

    # 9. Distributed Sampling & Analysis
    # Note: Every GPU must enter this block to participate in the workload
    print(f"[*] Starting Distributed Sampling on Rank {accelerator.process_index}...")
    
    # Initialize sampler with accelerator for distributed logic
    sampler = IsingSampler(model=model, process=process, device=device, accelerator=accelerator)
    
    sampling_kwargs = {
        "edge_index_single": edge_index,
        "edge_weight_single": edge_weight,
        "field_single": torch.from_numpy(mk).float()
    }

    # Split 2048 samples across all GPUs. 
    # 'max_vram_batch' ensures we don't OOM by processing in smaller sub-chunks.
    generated_spins = sampler.sample_distributed(
        total_samples=8192, 
        method='d3pm', 
        max_vram_batch=4096, # Adjust this if N increases
        **sampling_kwargs
    )

    # 10. Results Processing (Main Process Only)
    if accelerator.is_main_process:
        # Move gathered results back to GPU for final computation if they were on CPU
        final_spins = generated_spins.to(device) 
        
        plt.figure(figsize=(8, 5))
        # .sign() handles the D3PM discrete output mapping to physical spins
        m_gen = final_spins.sign().reshape(-1, 144).cpu().mean(-1)
        m_true = data[:10240].reshape(-1, 144).mean(-1)

        plt.hist(m_gen, bins=31, range=(-1, 1), alpha=0.5, density=True, label="Generated")
        plt.hist(m_true, bins=31, range=(-1, 1), alpha=0.5, density=True, label="True")
        plt.title("Magnetization PDF: D3PM vs Data")
        plt.xlabel("Magnetization M")
        plt.ylabel("Density")
        plt.legend()
        
        # Save analysis to the run folder
        plot_path = os.path.join(tracker.get_save_path(), "magn_pdf_d3pm.pdf")
        plt.savefig(plot_path)
        
        # Save raw samples for physics analysis (Energy, Correlations, etc.)
        samples_path = os.path.join(tracker.get_save_path(), "generated_samples.bin")
        final_spins.cpu().numpy().tofile(samples_path)
        
        print(f"[*] Analysis complete.")
        print(f"[*] Samples saved: {samples_path}")
        print(f"[*] Plot saved: {plot_path}")

if __name__ == "__main__":
    main()