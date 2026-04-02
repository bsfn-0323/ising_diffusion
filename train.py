import os
import argparse
import torch
import numpy as np
from accelerate import Accelerator
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset

from processes.sde import ContinuousVPSDE, get_cosine_schedule
from processes.discrete import D3PMProcess
from models.unetGnn import GNNUnet
from engines.trainer import IsingTrainer
from utils.tracker import ExperimentTracker

class MultiFieldIsingDataset(Dataset):
    def __init__(self, x_list=None, magn_list=None, edge_index_list=None, augment=False, load_path=None):
        super().__init__()
        self.augment = augment
        
        if load_path and os.path.exists(load_path):
            # Load the processed tensors from disk
            data_dict = torch.load(load_path)
            self.x_all = data_dict['x_all']
            self.fields = data_dict['fields']
            self.edge_indices = data_dict['edge_indices']
            self.sample_to_instance = data_dict['sample_to_instance']
            print(f"[*] Dataset loaded from {load_path}")
        else:
            # Process the raw lists into tensors
            self.edge_indices = [torch.tensor(ei, dtype=torch.long) for ei in edge_index_list]
            self.fields = [torch.tensor(m, dtype=torch.float32).view(-1, 1) for m in magn_list]
            
            all_x = []
            sample_to_instance = []
            
            for i, x_samples in enumerate(x_list):
                x_tensor = torch.tensor(x_samples, dtype=torch.float32) # [N_samples, 144, 1]
                all_x.append(x_tensor)
                # Map these samples to the i-th instance (field/topology)
                sample_to_instance.extend([i] * x_tensor.size(0))
            
            self.x_all = torch.cat(all_x, dim=0)
            self.sample_to_instance = torch.tensor(sample_to_instance, dtype=torch.long)
            print(f"[*] Dataset built: {self.x_all.size(0)} total samples.")

    def save(self, path):
        """Saves the processed dataset to a single file."""
        torch.save({
            'x_all': self.x_all,
            'fields': self.fields,
            'edge_indices': self.edge_indices,
            'sample_to_instance': self.sample_to_instance
        }, path)
        print(f"[*] Dataset saved to {path}")

    def len(self):
        return self.x_all.size(0)

    def get(self, idx):
        # 1. Identify which realization (instance) this sample belongs to
        instance_idx = self.sample_to_instance[idx].item()
        
        # 2. Slice the configuration [144, 1]
        x = self.x_all[idx].view(-1, 1) 
        
        # 3. Retrieve the correct field [144, 1] and edge_index [2, E]
        field = self.fields[instance_idx]
        edge_index = self.edge_indices[instance_idx]
        
        # Consistent Z2 symmetry augmentation
        if self.augment and torch.rand(1).item() < 0.5:
            x, field = -x, -field
            
        return Data(x=x, field=field, edge_index=edge_index)
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

def parse_args():
    parser = argparse.ArgumentParser(description="Ising Diffusion Training Script")
    parser.add_argument("--Tmin", type=float, default=1.5)
    parser.add_argument("--Tmax", type=float, default=2.5)
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--ch_mult", type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument("--time_emb_dim", type=int, default=32)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--L", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--batch_size_val", type=int, default=512)
    parser.add_argument("--save_path", type=str, default="./checkpoints")
    parser.add_argument("--discrete", action="store_true", help="Use D3PM instead of Continuous SDE")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--rank", type=int, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    betas_schedule = get_cosine_schedule(args.timesteps)
    if args.discrete:
        process = D3PMProcess(betas=betas_schedule, device=device, lambda_aux=0.1)
    else:
        process = ContinuousVPSDE(betas=betas_schedule, device=device)

    model = GNNUnet(
        base_ch=args.base_ch,
        ch_mult=args.ch_mult,
        time_emb_dim=args.time_emb_dim,
        discrete=args.discrete
    )

    data_dir = os.path.dirname(args.data_path)
    jmat_path = os.path.join(args.data_path, "Jmat.bin")
    config_path = os.path.join(args.data_path, f"config_rank{args.rank}.bin")
    N = args.L * args.L
    # data = np.fromfile(config_path, dtype=np.int32).reshape(-1, N, 1)
    # np.random.seed(seed=19990323+args.rank)
    # idxs = np.random.choice(20000,12500,replace = False)
    # data = data[idxs]
    # Js_raw = np.fromfile(jmat_path, dtype=np.int32).reshape(N, N)
    # Js = torch.from_numpy(Js_raw).float()
    # mk = data.mean(0)

    # edge_index = torch.from_numpy(np.stack(np.nonzero(Js_raw))).long()
    # if edge_index.shape[0] != 2:
    #     edge_index = edge_index.t().contiguous()
    # edge_weight = torch.full((edge_index.size(1), 1), 1.0)

    # ds = MyDataset(data, mk.reshape(N, 1), edge_index, edge_weight)
    # ds = torch.load("/mnt/beegfs/2a/sb12724/rfim_learn_fields/processed_dataset_L12.pt")
    ds = MultiFieldIsingDataset(load_path="/mnt/beegfs/2a/sb12724/rfim_learn_fields/processed_dataset_L12.pt", augment=True)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size_val, shuffle=False)

    exp_args = vars(args)
    tracker = ExperimentTracker(
        base_path=args.save_path,
        experiment_name="D3PM_RRG" if args.discrete else "SDE_RRG",
        model=model,
        process=process,
        accelerator=accelerator,
        args=exp_args
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs, eta_min=args.lr/100)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    trainer = IsingTrainer(
        model=model,
        process=process,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        tracker=tracker,
        epochs=args.epochs,
        save_path=tracker.get_save_path()
    )

    trainer.train()

if __name__ == "__main__":
    main()