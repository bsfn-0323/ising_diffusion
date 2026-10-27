"""Same as train_cfg.py, but --data_path / --validation_path each take one or
more .pt files (e.g. different system sizes L) and train on their union via
torch.utils.data.ConcatDataset. No padding/masking needed: samples are PyG
Data graphs collated by torch_geometric's DataLoader, and the D3PM loss
(processes/discrete.py: compute_loss / _vb_kl_node) is purely per-node via
batch.batch, so a batch can freely mix graphs of different N. The model
(models/unetGnn_cfg.py) also falls back to the non-uniform-N path
(rms_norm_scatter / global_mean_pool) whenever a batch isn't all one size.
"""
import os
import argparse
import torch
from accelerate import Accelerator
from torch.utils.data import Subset, ConcatDataset
from torch_geometric.loader import DataLoader
from processes.sde import ContinuousVPSDE, get_cosine_schedule
from processes.discrete import D3PMProcess
from models.unetGnn_cfg import GNNUnet
from engines.trainer import IsingTrainer
from utils.tracker import ExperimentTracker
from train_cfg import MultiFieldIsingDataset


def load_concat(paths, augment=False):
    datasets = [MultiFieldIsingDataset(load_path=p, augment=augment) for p in paths]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def parse_args():
    parser = argparse.ArgumentParser(description="Ising Diffusion Training Script (CFG, multi-dataset)")
    parser.add_argument("--Tmin", type=float, default=1.5)
    parser.add_argument("--Tmax", type=float, default=2.5)
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--time_emb_dim", type=int, default=32)
    parser.add_argument("--data_path", type=str, nargs="+", required=True,
                        help="one or more dataset .pt files (may mix system sizes)")
    parser.add_argument("--validation_path", type=str, nargs="+", default=None,
                        help="one or more validation .pt files (may mix system sizes)")
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
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    betas_schedule = get_cosine_schedule(args.timesteps)
    if args.discrete:
        process = D3PMProcess(betas=betas_schedule, device=device, lambda_aux=0.1, lambda_mk=0.0, p_cfg=args.p_cfg,
                              lambda_fa=0.0, fa_estimator=args.fa_estimator,uniform_N=False)
    else:
        process = ContinuousVPSDE(betas=betas_schedule, device=device)

    model = GNNUnet(
        base_ch=args.base_ch,
        time_emb_dim=args.time_emb_dim,
        discrete=args.discrete,
        num_iters=args.num_iters,
        use_tilt_head=False,  # decoupled tilt channel; on iff L_A is on
        uniform_N=False
    )
    print(f"[*] Model parameters: {sum(p.numel() for p in model.parameters())}")

    train_ds = load_concat(args.data_path, augment=False)
    val_ds = load_concat(args.validation_path, augment=False) if args.validation_path else None
    print(f"[*] Train dataset: {len(train_ds)} samples from {len(args.data_path)} file(s).")
    if val_ds is not None:
        print(f"[*] Val dataset: {len(val_ds)} samples from {len(args.validation_path)} file(s).")

    frac = 0.05
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
