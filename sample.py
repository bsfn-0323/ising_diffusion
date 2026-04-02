import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from accelerate import Accelerator

from utils.loading import load_model_from_checkpoint
from engines.samplers import IsingSampler
from processes.sde import ContinuousVPSDE, get_cosine_schedule
from processes.discrete import D3PMProcess

def parse_args():
    parser = argparse.ArgumentParser(description="Ising Diffusion Sampling Script")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the directory containing config.json and model weights")
    parser.add_argument("--num_samples", type=int, default=8192, help="Total samples to generate")
    parser.add_argument("--batch_size", type=int, default=2048, help="Max VRAM batch size per GPU")
    parser.add_argument("--load_best", action="store_true", help="Load model_best.pt instead of model_last.pt")
    return parser.parse_args()

def main():
    args = parse_args()
    
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    print(f"[*] Running sampling on device: {device}")

    # 1. Load Model and Config
    model, config = load_model_from_checkpoint(args.checkpoint_path, device, load_best=args.load_best)
    diff_config = config["diffusion"]
    train_args = config.get("training_params", {})
    
    # 2. Reconstruct Diffusion Process
    num_steps = diff_config.get("num_steps", 1000)
    if "beta_schedule" in diff_config and isinstance(diff_config["beta_schedule"], list):
        betas_schedule = torch.tensor(diff_config["beta_schedule"], dtype=torch.float32, device=device)
    else:
        betas_schedule = get_cosine_schedule(num_steps).to(device)

    is_discrete = "D3PM" in diff_config.get("type", "")
    if is_discrete:
        process = D3PMProcess(betas=betas_schedule, device=device, lambda_aux=diff_config.get("lambda_aux", 0.1))
        method = 'd3pm'
    else:
        process = ContinuousVPSDE(betas=betas_schedule, device=device)
        method = 'ddpm'

    # 3. Data Preparation from Config
    data_path = train_args.get("data_path")
    if not data_path or not os.path.exists(data_path):
        raise ValueError(f"Valid data_path not found in config.json: {data_path}")
        
    L = train_args.get("L", 12)
    N = L * L
    # data_dir = os.path.dirname(data_path)
    jmat_path = os.path.join(data_path, "Jmat.bin")
    config_path = os.path.join(data_path, f"config_rank{train_args.get('rank')}.bin")
    
    data_raw = np.fromfile(config_path, dtype=np.int32).reshape(-1, N, 1)
    mk = data_raw.mean(0)
    
    Js_raw = np.fromfile(jmat_path, dtype=np.int32).reshape(N, N)
    edge_index = torch.from_numpy(np.stack(np.nonzero(Js_raw))).long()
    if edge_index.shape[0] != 2:
        edge_index = edge_index.t().contiguous()
    edge_weight = torch.full((edge_index.size(1), 1), 1.0)

    # 4. Sampling
    sampler = IsingSampler(model=model, process=process, device=device, accelerator=accelerator)
    sampling_kwargs = {
        "edge_index_single": edge_index,
        "edge_weight_single": edge_weight,
        "field_single": torch.from_numpy(mk).float()
    }

    print(f"[*] Starting {method.upper()} Distributed Sampling...")
    generated_spins = sampler.sample_distributed(
        total_samples=args.num_samples, 
        method=method, 
        max_vram_batch=args.batch_size, 
        **sampling_kwargs
    )

    # 5. Save Results to Checkpoint Folder
    if accelerator.is_main_process:
        final_spins = generated_spins.to(device)
        samples_path = os.path.join(args.checkpoint_path, "generated_samples.bin")
        final_spins.cpu().numpy().astype(np.float32).tofile(samples_path)
        
        plt.figure(figsize=(8, 5))
        m_gen = final_spins.sign().reshape(-1, N).cpu().mean(-1)
        m_true = data_raw[:min(args.num_samples, len(data_raw))].reshape(-1, N).mean(-1)

        plt.hist(m_gen, bins=31, range=(-1, 1), alpha=0.5, density=True, label="Generated")
        plt.hist(m_true, bins=31, range=(-1, 1), alpha=0.5, density=True, label="True Data")
        plt.title(f"Magnetization PDF: {method.upper()} vs True Data")
        plt.xlabel("Magnetization M")
        plt.ylabel("Density")
        plt.legend()
        
        plot_path = os.path.join(args.checkpoint_path, f"magn_pdf_{method}.pdf")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"[*] Done. Samples: {samples_path} | Plot: {plot_path}")

if __name__ == "__main__":
    main()