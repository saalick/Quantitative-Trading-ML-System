"""
Baseline models: simple predictors for comparison.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Optional


class BaselineModel:
    """
    Sklearn-based baseline (e.g. logistic regression or random forest).
    Used for ablation and to show that sequence model adds value.
    """

    def __init__(self, kind: str = "logistic", **kwargs):
        if kind == "logistic":
            self.model = LogisticRegression(max_iter=1000, **kwargs)
        elif kind == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, **kwargs)
        else:
            raise ValueError(f"Unknown baseline: {kind}")
        self.kind = kind

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineModel":
        """X: (n_samples, n_features) - use last time step of sequence or flattened."""
        if X.ndim == 3:
            X = X[:, -1, :]  # last step
        self.model.fit(X, (y >= 0.5).astype(int) if y.dtype == np.float32 else y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X[:, -1, :]
        return self.model.predict_proba(X)[:, 1:2]  # (n, 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X).ravel() >= 0.5).astype(int)
