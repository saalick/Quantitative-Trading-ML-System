"""
Ensemble of LSTM and/or baselines for robustness.
"""

import torch
import numpy as np
from typing import List, Union
from src.models.lstm_model import LSTMModel


class EnsembleModel:
    """
    Average predictions from multiple models (e.g. several LSTMs with different seeds).
    Research: ensemble reduces variance and can improve calibration.
    """

    def __init__(self, models: List[Union[LSTMModel, "BaselineModel"]]):
        self.models = models

    def predict_proba(self, X: Union[torch.Tensor, np.ndarray], device: torch.device) -> np.ndarray:
        preds = []
        for m in self.models:
            if isinstance(m, LSTMModel):
                m.eval()
                with torch.no_grad():
                    x = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32, device=device)
                    p = m(x).cpu().numpy()
            else:
                p = m.predict_proba(X if isinstance(X, np.ndarray) else X.cpu().numpy())
            preds.append(p)
        return np.mean(preds, axis=0)
