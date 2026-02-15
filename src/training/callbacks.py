"""
Training callbacks: early stopping, checkpointing, TensorBoard.
"""

import torch
from pathlib import Path
from typing import Optional


class EarlyStopping:
    """Stop when validation loss does not improve for patience epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # "min" for loss, "max" for accuracy
        self.counter = 0
        self.best = float("inf") if mode == "min" else float("-inf")

    def __call__(self, value: float) -> bool:
        if self.mode == "min":
            improved = value < (self.best - self.min_delta)
        else:
            improved = value > (self.best + self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class ModelCheckpoint:
    """Save best model based on validation metric."""

    def __init__(self, path: Path, mode: str = "min"):
        self.path = Path(path)
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")

    def __call__(self, model: torch.nn.Module, value: float, optimizer=None, epoch=None) -> bool:
        if self.mode == "min":
            save = value < self.best
        else:
            save = value > self.best
        if save:
            self.best = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            state = {"model_state_dict": model.state_dict()}
            if optimizer is not None:
                state["optimizer_state_dict"] = optimizer.state_dict()
            if epoch is not None:
                state["epoch"] = epoch
            torch.save(state, self.path)
            return True
        return False
