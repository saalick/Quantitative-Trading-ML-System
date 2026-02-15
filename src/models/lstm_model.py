"""
LSTM model for directional prediction.
Architecture: 2-layer LSTM (128 units each), dense 64, dropout, output 1.
Research: Why does this over-parameterized model generalize? See docs/research_questions.md.
"""

import torch
import torch.nn as nn
from src.models.model_utils import init_weights, count_parameters


class LSTMModel(nn.Module):
    """
    LSTM for sequence-to-one direction score (regression for probability).
    Input: (batch, seq_len=60, features=45).
    Output: (batch, 1) in [0, 1] for probability of positive return.
    """

    def __init__(
        self,
        input_size: int = 45,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        dense_units: int = 64,
        output_size: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc1 = nn.Linear(hidden_size, dense_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(dense_units, output_size)
        self.sigmoid = nn.Sigmoid()

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use last time step
        last = lstm_out[:, -1, :]  # (batch, hidden_size)
        out = self.fc1(last)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out)

    def summary(self) -> str:
        """Print model summary and parameter count."""
        n = count_parameters(self)
        return (
            f"LSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers})\n"
            f"Total trainable parameters: {n:,}"
        )
