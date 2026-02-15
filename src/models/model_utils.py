"""
Model utilities: weight initialization, parameter counting, save/load.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


def init_weights(module: nn.Module) -> None:
    """
    Xavier for linear layers, orthogonal for LSTM.
    Research: proper initialization improves optimization landscape (see Li et al.).
    """
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[dict] = None,
) -> None:
    """Save model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if metrics is not None:
        state["metrics"] = metrics
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load checkpoint. Returns extra state (epoch, metrics) if present.
    """
    path = Path(path)
    state = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(state["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return {k: v for k, v in state.items() if k not in ("model_state_dict", "optimizer_state_dict")}
