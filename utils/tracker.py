import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class ExperimentTracker:
    def __init__(self, base_path, experiment_name, model, process, accelerator, args):
        self.accelerator = accelerator
        self.best_val_loss = float('inf')
        
        # 1. Directory Setup
        date_str = datetime.now().strftime("%Y-%m-%d")
        param_suffix = f"h{args.get('hstd', 'na')}_L{args.get('L', 'na')}_lr{args.get('lr')}"
        self.save_path = os.path.join(base_path, experiment_name, date_str, param_suffix)
        self.step_ckpt_path = os.path.join(self.save_path, "step_checkpoints")

        if self.accelerator.is_main_process:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(self.step_ckpt_path, exist_ok=True)
            self._save_config(model, process, args)
            
            self.metrics_path = os.path.join(self.save_path, "metrics.csv")
            if not os.path.exists(self.metrics_path):
                with open(self.metrics_path, "w") as f:
                    f.write("epoch,train_loss,val_loss,lr\n")

    def _save_config(self, model, process, args):
        """Extracts and saves metadata based on the specific D3PM implementation."""
        unwrapped = self.accelerator.unwrap_model(model)
        
        # Determine diffusion parameters dynamically
        diffusion_info = {
            "type": type(process).__name__,
            "num_steps": getattr(process, 'timesteps', args.get('timesteps')),
            "K_categories": getattr(process, 'K', 2),
            "lambda_aux": getattr(process, 'lambda_aux', None),
        }

        # If it's D3PM, we want to know the transition matrix schedule
        # Note: D3PMProcess doesn't store 'betas' directly, but we have them in args
        if "betas" in args:
            diffusion_info["beta_schedule"] = args["betas"] if isinstance(args["betas"], list) else "custom_tensor"

        config = {
            "timestamp": datetime.now().isoformat(),
            "architecture": {
                "type": type(unwrapped).__name__,
                "base_ch": getattr(unwrapped, 'base_ch', None),
                "time_emb_dim": getattr(unwrapped, 'time_emb_dim', None),
                "discrete_mode": getattr(unwrapped, 'discrete', False),
                "ch_mult": getattr(unwrapped, 'ch_mult', None)
            },
            "diffusion": diffusion_info,
            "training_params": {k: v for k, v in args.items() if k != 'betas'}
        }
        
        with open(os.path.join(self.save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    def log_step(self, step, model, mode="linear", interval=5000):
        if not self.accelerator.is_main_process: return

        should_save = False
        if mode == "linear" and step % interval == 0:
            should_save = True
        elif mode == "log":
            log10 = np.log10(step) if step > 0 else 0
            if log10 % 1 == 0 or (step / 2) % (10**(int(log10))) == 0 or (step / 5) % (10**(int(log10))) == 0:
                should_save = True

        if should_save:
            unwrapped = self.accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), os.path.join(self.step_ckpt_path, f"model_step_{step}.pt"))

    def log_epoch(self, epoch, train_loss, val_loss, model, optimizer):
        if not self.accelerator.is_main_process: return

        current_lr = optimizer.param_groups[0]['lr']
        with open(self.metrics_path, "a") as f:
            f.write(f"{epoch},{train_loss:.8f},{val_loss:.8f},{current_lr:.8e}\n")

        unwrapped = self.accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), os.path.join(self.save_path, "model_last.pt"))
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(unwrapped.state_dict(), os.path.join(self.save_path, "model_best.pt"))

    def finalize(self):
        """Generates the final loss plot at the end of training."""
        if not self.accelerator.is_main_process: return
        try:
            df = pd.read_csv(self.metrics_path)
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
            plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
            plt.yscale('log')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(self.save_path, "loss_curve.png"))
            plt.close()
        except Exception as e:
            print(f"[!] Plotting failed: {e}")

    def get_save_path(self):
        return self.save_path