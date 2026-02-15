"""Utility functions."""

import os
import random
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist. Return path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
