"""Shared device detection for all SeqRec training and inference code."""
import torch


def get_device() -> torch.device:
    """
    Return the best available device in priority order:
      1. CUDA (NVIDIA GPU) — if torch.cuda.is_available()
      2. MPS  (Apple Silicon) — if torch.backends.mps.is_available()
      3. CPU  — fallback
    Prints the selected device at INFO level.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] Using: {device}")
    return device
