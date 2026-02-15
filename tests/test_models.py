"""Tests for LSTM and model utils: init, forward shape, save/load."""

import numpy as np
import torch
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.lstm_model import LSTMModel
from src.models.model_utils import count_parameters, save_checkpoint, load_checkpoint


def test_lstm_forward_shape():
    model = LSTMModel(input_size=45, hidden_size=128, num_layers=2, dropout=0.2)
    x = torch.randn(4, 60, 45)
    out = model(x)
    assert out.shape == (4, 1)


def test_lstm_output_range():
    model = LSTMModel(input_size=45, hidden_size=128, num_layers=2, dropout=0.2)
    x = torch.randn(2, 60, 45)
    out = model(x)
    assert (out >= 0).all() and (out <= 1).all()


def test_count_parameters():
    model = LSTMModel(input_size=10, hidden_size=16, num_layers=1, dropout=0)
    n = count_parameters(model)
    assert n > 0


def test_save_load():
    model = LSTMModel(input_size=5, hidden_size=8, num_layers=1, dropout=0)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.pt"
        save_checkpoint(model, path)
        assert path.exists()
        model2 = LSTMModel(input_size=5, hidden_size=8, num_layers=1, dropout=0)
        load_checkpoint(path, model2)
        model.eval()
        model2.eval()
        x = torch.randn(2, 10, 5)
        with torch.no_grad():
            assert torch.allclose(model(x), model2(x))


def test_different_input_sizes():
    for size in [10, 45, 100]:
        model = LSTMModel(input_size=size, hidden_size=32, num_layers=1, dropout=0)
        x = torch.randn(2, 20, size)
        out = model(x)
        assert out.shape == (2, 1)
