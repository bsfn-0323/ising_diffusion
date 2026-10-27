import os
import argparse
import torch
import numpy as np
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from processes.sde import ContinuousVPSDE, get_cosine_schedule
from processes.discrete import D3PMProcess
from models.unetGnn_cfg import GNNUnet
from engines.trainer import IsingTrainer
from utils.tracker import ExperimentTracker


class MultiFieldIsingDataset(Dataset):
    def __init__(self, x_list=None, magn_list=None, beta_list=None, field_list=None, edge_index_list=None, augment=False, load_path=None):
        super().__init__()
        self.augment = augment

        if load_path and os.path.exists(load_path):
            data_dict = torch.load(load_path)
            self.x_all = data_dict['x_all']
            self.mks = data_dict['mks']
            self.fields = data_dict['fields']
            self.beta_list = data_dict['beta_list']
            self.edge_indices = data_dict['edge_indices']
            self.sample_to_instance = data_dict['sample_to_instance']
            print(f"[*] Dataset loaded from {load_path}")
        else:
            self.edge_indices = [torch.tensor(ei, dtype=torch.long) for ei in edge_index_list]
            self.mks = [torch.tensor(m, dtype=torch.float32).view(-1, 1) for m in magn_list]
            self.fields = [torch.tensor(f, dtype=torch.float32).view(-1, 1) for f in field_list]
            self.beta_list = [torch.tensor(b, dtype=torch.float32) for b in beta_list]

            all_x = []
            sample_to_instance = []
            for i, x_samples in enumerate(x_list):
                x_tensor = torch.tensor(x_samples, dtype=torch.float32)
                all_x.append(x_tensor)
                sample_to_instance.extend([i] * x_tensor.size(0))

            self.x_all = torch.cat(all_x, dim=0)
            self.sample_to_instance = torch.tensor(sample_to_instance, dtype=torch.long)
            print(f"[*] Dataset built: {self.x_all.size(0)} total samples.")

        self.fields_mean = torch.tensor([f.mean() for f in self.fields], dtype=torch.float32).view(-1, 1)
        self.fields_std = torch.tensor([f.std() for f in self.fields], dtype=torch.float32).view(-1, 1)
        self.deg_norms = []
        for i, ei in enumerate(self.edge_indices):
            N = self.fields[i].shape[0]
            src, dst = ei
            deg = torch.zeros(N, dtype=torch.float32)
            deg.scatter_add_(0, dst, torch.ones(src.shape[0], dtype=torch.float32))
            deg_inv_sqrt = deg.pow(-0.5)
            self.deg_norms.append((deg_inv_sqrt[src] * deg_inv_sqrt[dst]).contiguous())

    def save(self, path):
        torch.save({
            'x_all': self.x_all,
            'mks': self.mks,
            'fields': self.fields,
            'beta_list': self.beta_list,
            'edge_indices': self.edge_indices,
            'sample_to_instance': self.sample_to_instance,
        }, path)
        print(f"[*] Dataset saved to {path}")

    def len(self):
        return self.x_all.size(0)

    def get(self, idx):
        instance_idx = self.sample_to_instance[idx].item()
        x = self.x_all[idx].view(-1, 1)
        mk = self.mks[instance_idx]
        field = self.fields[instance_idx]
        field_mean = self.fields_mean[instance_idx]
        field_std = self.fields_std[instance_idx]
        edge_index = self.edge_indices[instance_idx]

        
        if self.augment and torch.rand(1).item() < 0.5:
            x, field = -x, -field
            mk = -mk
            field_mean = -field_mean

        beta = self.beta_list[instance_idx]
        deg_norm   = self.deg_norms[instance_idx]          # [E], precomputed
        edge_weight = (beta * deg_norm).contiguous()        # scalar * [E]

        # edge_weight = beta.expand(edge_index.size(1)).contiguous()
        return Data(
            x=x, mk=mk, field=field,
            field_mean=field_mean, field_std=field_std,
            edge_index=edge_index, edge_weight=edge_weight,
            instance_idx=torch.tensor(instance_idx, dtype=torch.long),
            inv_temp=beta,

        )


def parse_args():
    parser = argparse.ArgumentParser(description="Ising Diffusion Training Script (CFG)")
    parser.add_argument("--Tmin", type=float, default=1.5)
    parser.add_argument("--Tmax", type=float, default=2.5)
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--time_emb_dim", type=int, default=32)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--validation_path", type=str, default=None)
    parser.add_argument("--L", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--batch_size_val", type=int, default=512)
    parser.add_argument("--save_path", type=str, default="./checkpoints")
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--p_cfg", type=float, default=0.0,
                        help="Probability of dropping beta conditioning per training step (CFG dropout)")
    parser.add_argument("--use_mk", action="store_true",
                        help="Condition on m_k (magnetization) in addition to h_i (local field); this is the default for the D3PM paper, but not used in the SDE experiments.")

    # --- Flip-asymmetry ("free-energy") loss L_A (fixes Z2 mode weights); OFF by default ---
    parser.add_argument("--lambda_fa", type=float, default=0.0,
                        help="flip-asymmetry loss weight; 0 = off")
    parser.add_argument("--fa_estimator", type=str, default="unbiased",
                        choices=["unbiased", "single"],
                        help="L_A estimator: 'unbiased' (2-sample) or 'single' (1-sample, biased)")
    parser.add_argument("--fa_probe", action="store_true",
                        help="log flip-asymmetry r_hat vs tau diagnostics without training on L_A")
    return parser.parse_args()


def main():
    args = parse_args()
    # find_unused_parameters=True: the flip-asymmetry loss runs several extra model
    # forwards per step; this keeps DDP happy when a param isn't in every sub-graph.
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[ddp_kwargs])
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    betas_schedule = get_cosine_schedule(args.timesteps)
    if args.discrete:
        process = D3PMProcess(betas=betas_schedule, device=device, lambda_aux=0.1, lambda_mk=0.0, p_cfg=args.p_cfg,
                              lambda_fa=0.0, fa_estimator=args.fa_estimator)
    else:
        process = ContinuousVPSDE(betas=betas_schedule, device=device)

    model = GNNUnet(
        base_ch=args.base_ch,
        time_emb_dim=args.time_emb_dim,
        discrete=args.discrete,
        num_iters=args.num_iters,
        use_tilt_head=False,  # decoupled tilt channel; on iff L_A is on
    )
    print(f"[*] Model parameters: {sum(p.numel() for p in model.parameters())}")

    train_ds = MultiFieldIsingDataset(load_path=args.data_path, augment=False,)
    val_ds = MultiFieldIsingDataset(load_path=args.validation_path, augment=False) if args.validation_path else None

    frac = 0.01
    n = int(len(val_ds) * frac)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(val_ds), generator=g)[:n]
    subset = Subset(val_ds, idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(subset, batch_size=args.batch_size_val, shuffle=False)

    exp_args = vars(args)
    tracker = ExperimentTracker(
        base_path=args.save_path,
        experiment_name="D3PM_CFG" if args.discrete else "SDE_CFG",
        model=model,
        process=process,
        accelerator=accelerator,
        args=exp_args,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs, eta_min=args.lr / 100
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    print(f"Using mk: {args.use_mk}")
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
        save_path=tracker.get_save_path(),
        use_mk=args.use_mk,  # this experiment conditions on h_i (not m_k)
        fa_probe=False,
    )

    trainer.train()


if __name__ == "__main__":
    main()
