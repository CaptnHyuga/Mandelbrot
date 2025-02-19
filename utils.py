import os
import json
import glob
import torch
from config import CHECKPOINT_DIR, LOG_LEVEL

def log_message(level, msg):
    """Simple logging function that respects config.LOG_LEVEL."""
    levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    if levels.index(level) >= levels.index(LOG_LEVEL):
        print(f"[{level}] {msg}")

def fourier_encode(positions, num_frequencies):
    """
    Encode positions (N x 2) using a Fourier feature mapping.
    Args:
        positions: tensor of shape (N, 2)
        num_frequencies: int, number of frequency pairs
    Returns:
        tensor of shape (N, 4*num_frequencies)
    """
    if not isinstance(positions, torch.Tensor):
        positions = torch.from_numpy(positions).float()
    
    # Validate input shape
    if positions.shape[-1] != 2:
        raise ValueError(f"Expected positions to have shape (N, 2), got {positions.shape}")
    
    x, y = positions[..., 0], positions[..., 1]
    
    # Preallocate output tensor
    batch_size = positions.shape[0]
    output_dim = 4 * num_frequencies
    encoded = torch.zeros((batch_size, output_dim), dtype=torch.float32)
    
    # Fill encoded tensor
    for i in range(num_frequencies):
        encoded[:, 4*i:4*i+4] = torch.stack([
            torch.sin((i + 1) * torch.pi * x),
            torch.cos((i + 1) * torch.pi * x),
            torch.sin((i + 1) * torch.pi * y),
            torch.cos((i + 1) * torch.pi * y)
        ], dim=1)
    
    return encoded

def save_checkpoint(network, generation, loss):
    """
    Save network checkpoint at the specified generation and loss to a JSON file.
    """
    state_dict = network.save_state_dict()
    state_dict['generation'] = generation
    state_dict['loss'] = loss
    filename = os.path.join(CHECKPOINT_DIR, f'checkpoint_{generation:06d}.json')
    temp_filename = filename + '.tmp'
    try:
        with open(temp_filename, 'w') as f:
            json.dump(state_dict, f)
        os.replace(temp_filename, filename)
        log_message("INFO", f"Checkpoint saved: {filename}")
    except Exception as e:
        log_message("ERROR", f"Error saving checkpoint: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def load_latest_checkpoint():
    """
    Find and load the latest checkpoint from CHECKPOINT_DIR.
    Returns the state dict and the generation index.
    """
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, 'checkpoint_*.json'))
    if not checkpoints:
        log_message("INFO", "No checkpoints found.")
        return None, 0
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    try:
        with open(latest, 'r') as f:
            state_dict = json.load(f)
        log_message("INFO", f"Loading checkpoint: {latest}")
        return state_dict, state_dict.get('generation', 0)
    except FileNotFoundError:
        log_message("ERROR", f"Error loading checkpoint: File not found: {latest}")
        return None, 0
    except Exception as e:
        log_message("ERROR", f"Error loading checkpoint: {e}")
        return None, 0