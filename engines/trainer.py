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
        tracker
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
        
        with self.accelerator.autocast():
            # 1. Ask the Process for the primary loss and the predicted clean state
            base_loss, x0_pred = self.process.compute_loss(self.model, batch, time_idx)
            
            # 2. Enforce Physical Constraints
            loss_edge = self.correlation_loss(x0_pred.sign(), batch.x, batch.edge_index)
            loss = base_loss + 0.5 * loss_edge

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
        
        base_loss, _ = self.process.compute_loss(self.model, batch, time_idx)
        return base_loss.detach().item()

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
                self.tracker.log_step(global_step, self.model, mode="linear", interval=5000)
            
            self.scheduler.step()

            # 3. Validation and Epoch Logging
            # Adjust frequency as needed (e.g., every 10 or 100 epochs)
            if epoch % 10 == 0:
                val_loss_sum = 0.0
                self.model.eval()
                
                with torch.no_grad():
                    for batch in self.val_loader:
                        val_loss_sum += self.val_step(batch)
                
                train_loss = train_loss_sum / len(self.train_loader)
                val_loss = val_loss_sum / len(self.val_loader)
                
                # 4. Use the tracker to handle model saving and CSV logging
                # This replaces your old self._save_checkpoint(val_loss, epoch)
                self.tracker.log_epoch(epoch, train_loss, val_loss, self.model, self.optimizer)
                
                if self.accelerator.is_main_process:
                    # Update TQDM postfix with current loss info
                    epoch_iter.set_postfix({"train": f"{train_loss:.4f}", "val": f"{val_loss:.4f}"})

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