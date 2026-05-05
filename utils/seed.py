"""
utils/seed.py — Centralized seed management for reproducibility.
All experiments must call set_all_seeds(seed) before any random operation.
"""

import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Args:
        seed: Integer seed value. Use values from config['evaluation']['seeds'].
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic CUDA operations (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Return GPU device if available, else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] GPU not available. Using CPU.")
    return device
