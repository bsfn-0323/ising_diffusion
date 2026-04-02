import os
import json
import torch
from models.unetGnn import GNNUnet

def load_model_from_checkpoint(checkpoint_dir, device, load_best=False):
    """
    Reconstructs the GNNUnet and loads weights based on the checkpoint metadata.
    """
    # 1. Load the configuration
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in {checkpoint_dir}")
        
    with open(config_path, "r") as f:
        config = json.load(f)
    
    arch = config["architecture"]
    train_args = config.get("training_params",{})
    # 2. Reconstruct Architecture
    # Ensure these keys match what the ExperimentTracker saved
    print(arch)
    model = GNNUnet(
        base_ch=arch["base_ch"],
        ch_mult=arch["ch_mult"], # This was hardcoded in main, but you could save it too
        time_emb_dim=arch["time_emb_dim"],
        discrete=arch.get("discrete_mode", True) 
    )
    
    # 3. Choose which weights to load
    weight_file = "model_best.pt" if load_best else "model_last.pt"
    weight_path = os.path.join(checkpoint_dir, weight_file)
    print("CARICATO DIO")
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file {weight_file} not found in {checkpoint_dir}")

    # 4. Load weights with DDP prefix handling
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    
    # If the model was saved via Accelerator/DDP, keys start with 'module.'
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    print(f"[*] Successfully loaded {weight_file} from {checkpoint_dir}")
    return model, config