"""
Trainer class: data loading, training loop, validation, checkpointing.
Research: SGD convergence and learning rate scheduling. See docs/research_questions.md.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm

from src.models.lstm_model import LSTMModel
from src.models.model_utils import save_checkpoint
from src.training.callbacks import EarlyStopping, ModelCheckpoint


class Trainer:
    """
    Train LSTM with BCE loss, SGD + momentum, cosine annealing, gradient clipping.
    """

    def __init__(
        self,
        model: LSTMModel,
        device: torch.device,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        max_epochs: int = 50,
        early_stopping_patience: int = 10,
        save_dir: Optional[Path] = None,
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        self.criterion = nn.BCELoss()
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode="min")
        self.save_dir = Path(save_dir) if save_dir else None
        self.checkpoint = ModelCheckpoint(Path(save_dir) / "best_model.pt", mode="min") if save_dir else None

    def train_epoch(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        n = 0
        # Simple batch loop (assume X, y are already batched or we use full batch)
        X = X.to(self.device)
        y = y.to(self.device)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        self.optimizer.zero_grad()
        out = self.model(X)
        loss = self.criterion(out, y)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        total_loss = loss.item()
        pred = (out.detach() >= 0.5).float()
        correct = (pred == y).sum().item()
        n = y.size(0)
        return total_loss, correct / n if n else 0

    def train_batched(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
    ) -> Tuple[list, list, list, list]:
        """Train with mini-batches. Returns train_losses, train_accs, val_losses, val_accs."""
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        n = len(X_train)
        for epoch in range(self.max_epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_n = 0
            self.model.train()
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                Xb = torch.tensor(X_train[idx], dtype=torch.float32, device=self.device)
                yb = torch.tensor(y_train[idx], dtype=torch.float32, device=self.device)
                if yb.dim() == 1:
                    yb = yb.unsqueeze(1)
                self.optimizer.zero_grad()
                out = self.model(Xb)
                loss = self.criterion(out, yb)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                epoch_loss += loss.item() * len(idx)
                epoch_correct += ((out.detach() >= 0.5).float() == yb).sum().item()
                epoch_n += len(idx)
            self.scheduler.step()
            train_loss = epoch_loss / epoch_n if epoch_n else 0
            train_acc = epoch_correct / epoch_n if epoch_n else 0
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation
            v_loss, v_acc = self.evaluate(X_val, y_val, batch_size)
            val_losses.append(v_loss)
            val_accs.append(v_acc)

            if self.checkpoint:
                self.checkpoint(self.model, v_loss, self.optimizer, epoch)
            if self.early_stopping(v_loss):
                break
        return train_losses, train_accs, val_losses, val_accs

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        n = len(X)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                Xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=self.device)
                yb = torch.tensor(y[start:start + batch_size], dtype=torch.float32, device=self.device)
                if yb.dim() == 1:
                    yb = yb.unsqueeze(1)
                out = self.model(Xb)
                loss = self.criterion(out, yb)
                total_loss += loss.item() * Xb.size(0)
                correct += ((out >= 0.5).float() == yb).sum().item()
        return total_loss / n if n else 0, correct / n if n else 0
