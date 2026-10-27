import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
class IsingTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        process, # Abstraction: ContinuousVPSDE or D3PMProcess
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        accelerator,
        # Js: torch.Tensor,
        # hs: torch.Tensor,
        epochs: int,
        save_path: str,
        tracker,
        use_mk: bool = False,  # New parameter to control the use of mk
        use_both: bool= False,
        fa_probe: bool = False,  # diagnostic-only: log flip-asymmetry r_hat vs tau, do not train on it
    ):
        self.model = model
        self.process = process
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.device = accelerator.device
        self.tracker=tracker
        self.use_mk = use_mk  # Store the new parameter
        self.use_both = use_both
        self.fa_probe = fa_probe
        # self.Js = Js.to(self.device)
        # self.hs = hs.to(self.device)
        self.epochs = epochs
        self.save_path = save_path
        self.best_val_loss = float('inf')

    def correlation_loss(self, pred_x0: torch.Tensor, target_x0: torch.Tensor, edge_index: torch.Tensor):
        """Computes the physical energy correlation loss."""
        pred_src, pred_dst = pred_x0[edge_index[0]], pred_x0[edge_index[1]]
        target_src, target_dst = target_x0[edge_index[0]], target_x0[edge_index[1]]
        return F.mse_loss(pred_src * pred_dst, target_src * target_dst)

    def train_step(self, batch):
        self.model.train()
        B = batch.batch_size
        time_idx = torch.randint(0, self.process.timesteps - 1, (B,), device=self.device)
        # with self.accelerator.accumulate(self.model):
        with self.accelerator.autocast():
            # 1. Ask the Process for the primary loss and the predicted clean state
            base_loss, _ = self.process.compute_loss(self.model, batch, time_idx,include_L0=False, use_mk=self.use_mk,use_both=self.use_both)
            # base_loss, _ = self.process.compute_loss(self.model, batch, time_idx,include_L0=True)
            
            # 2. Enforce Physical Constraints
            # loss_edge = self.correlation_loss(x0_pred.sign(), batch.x, batch.edge_index)
            # loss = base_loss + 0.5 * loss_edge
            loss = base_loss

            # 3. Flip-asymmetry ("free-energy") loss L_A -- fixes Z2 mode weights.
            # if self.fa_probe:
            #     # Diagnostic-first: compute L_A + r_hat vs tau and log, but DO NOT train on L_A.
            #     # no_grad is essential: the extra forward passes must not build an
            #     # autograd graph, else DDP sees params used in forward but never
            #     # reduced (only base_loss is backpropagated) and errors next step.
            #     with torch.no_grad():
            #         fa_val, fa_diag = self.process.flip_asymmetry_loss(
            #             self.model, batch, use_mk=self.use_mk, return_diag=True)
            #     fa_diag["fa_LA"] = fa_val.item()
            #     self._last_fa_diag = fa_diag
            # elif getattr(self.process, "lambda_fa", 0.0) > 0.0:   # OFF when lambda_fa == 0
            #     fa = self.process.flip_asymmetry_loss(self.model, batch, use_mk=self.use_mk)
            #     loss = loss + self.process.lambda_fa * fa

        self.accelerator.backward(loss)
        
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
        return loss.detach().item()

    @torch.no_grad()
    def val_step(self, batch):
        self.model.eval()
        B = batch.batch_size
        time_idx = torch.randint(0, self.process.timesteps - 1, (B,), device=self.device)

        base_loss, p_plus = self.process.compute_loss(self.model, batch, time_idx,include_L0=False, use_mk=self.use_mk,use_both=self.use_both)
        x0_pred = (p_plus * 2 - 1).float().unsqueeze(-1)
        val_energy = self.correlation_loss(x0_pred, batch.x, batch.edge_index).item()

        val_mk = F.mse_loss(p_plus[:,None], batch.mk.float()).item()

        return base_loss.detach().item(), val_energy, val_mk

    def train(self):
        epoch_iter = range(self.epochs)
        global_step = 0 # Track total steps across all epochs
        
        if self.accelerator.is_main_process:
            epoch_iter = tqdm(epoch_iter, desc="Training Progress")
        
        for epoch in epoch_iter:
            self.model.train()
            train_loss_sum = 0.0
            
            for batch in self.train_loader:
                # 1. Perform training step
                loss = self.train_step(batch)
                train_loss_sum += loss
                global_step += 1
                
                # 2. Log intermediate step checkpoints (e.g., every 5000 steps)
                # You can change mode to "log" for logarithmic saving
                self.tracker.log_step(global_step, self.model, mode="linear", interval=50)

                # Diagnostic-first probe: report flip-asymmetry r_hat vs tau scale/corr.
                if self.fa_probe and self.accelerator.is_main_process and global_step % 50 == 0:
                    d = getattr(self, "_last_fa_diag", None)
                    if d is not None:
                        print(f"[FA probe] step {global_step}  L_A={d['fa_LA']:.3e}  "
                              f"mean_rhat={d['fa_mean_rhat']:+.4f}  mean_tau={d['fa_mean_tau']:+.4f}  "
                              f"corr={d['fa_corr']:+.3f}  mean|Kflip-Kreal|={d['fa_mean_absKdiff']:.2e}",
                              flush=True)

                self.scheduler.step()

            # 3. Validation and Epoch Logging
            # Adjust frequency as needed (e.g., every 10 or 100 epochs)
            
            if epoch % 10 == 0:
                val_loss_sum = 0.0
                val_energy_sum = 0.0
                val_mk_sum = 0.0
                self.model.eval()
                if self.val_loader is not None:
                    with torch.no_grad():
                        for batch in self.val_loader:
                            single_val_loss,val_energy, val_mk = self.val_step(batch)
                            val_loss_sum += single_val_loss
                            val_energy_sum += val_energy
                            val_mk_sum += val_mk

                    val_loss = val_loss_sum / len(self.val_loader)
                    val_energy = val_energy_sum / len(self.val_loader)
                    val_mk = val_mk_sum / len(self.val_loader)
                else:
                    val_loss = 0.0
                    val_energy = 0.0
                    val_mk = 0.0
                train_loss = train_loss_sum / len(self.train_loader)

                # 4. Use the tracker to handle model saving and CSV logging
                # This replaces your old self._save_checkpoint(val_loss, epoch)
                self.tracker.log_epoch(epoch, train_loss, val_loss,val_energy,val_mk, self.model, self.optimizer)
                
                if self.accelerator.is_main_process:
                    # Update TQDM postfix with current loss info
                    epoch_iter.set_postfix({"train": f"{train_loss:.4f}", "val": f"{val_loss:.4f}", "energy": f"{val_energy:.4f}", "mk": f"{val_mk:.4f}"})
                    # m = self.accelerator.unwrap_model(self.model)
                    # epoch_iter.set_postfix({"damping": f"{(m.block.dmin+(1-m.block.dmin)*m.block.damping_logit.sigmoid()).item():.4f}"})
        # 5. Finalize: Generate the loss curves once training is complete
        self.tracker.finalize()
        self.accelerator.wait_for_everyone()

    # def _save_checkpoint(self, val_loss, epoch):
    #     """Saves the best model based on validation metrics."""
    #     if val_loss < self.best_val_loss:
    #         self.best_val_loss = val_loss
    #         os.makedirs(self.save_path, exist_ok=True)
    #         save_file = os.path.join(self.save_path, "epoch_best.bin")
            
    #         unwrapped_model = self.accelerator.unwrap_model(self.model)
    #         self.accelerator.save(unwrapped_model.state_dict(), save_file)
    #         print(f"[*] Best model saved at epoch {epoch} (Loss: {val_loss:.4f})")