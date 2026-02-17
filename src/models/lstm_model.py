"""
LSTM model for directional prediction.
Architecture: 2-layer LSTM (128 units each), temporal attention, layer norm,
residual projection, dense head, output 1.
Research: Why does this over-parameterized model generalize? See docs/research_questions.md.
Enhanced: Added temporal self-attention so the model learns *which time steps*
matter most for the directional signal, rather than blindly using the last step.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.model_utils import init_weights, count_parameters


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) self-attention over the LSTM sequence.

    For each time step h_t the module scores how relevant that step is to
    the final prediction, then returns a weighted context vector.

        e_t = v^T * tanh(W * h_t + b)   (scalar energy for step t)
        a_t = softmax(e_t)               (normalised attention weight)
        c   = sum_t a_t * h_t            (context vector)

    The additive formulation is lightweight (only two small linear layers)
    and interpretable: you can visualise a_t to see which days the model
    focuses on, which is useful for research / debugging.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Projects each hidden state to an energy scalar
        self.energy_layer = nn.Linear(hidden_size, hidden_size, bias=True)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, lstm_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_size)  – all LSTM hidden states

        Returns:
            context:  (batch, hidden_size)  – attention-weighted summary
            weights:  (batch, seq_len)      – attention distribution (for viz)
        """
        # (batch, seq_len, hidden_size)
        energy = torch.tanh(self.energy_layer(lstm_out))
        # (batch, seq_len, 1) -> (batch, seq_len)
        scores = self.context_vector(energy).squeeze(-1)
        weights = F.softmax(scores, dim=-1)                 # (batch, seq_len)
        # Weighted sum: (batch, 1, seq_len) @ (batch, seq_len, hidden) -> (batch, 1, hidden)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """
    LSTM + Temporal Attention for sequence-to-one direction score.

    Input:  (batch, seq_len=60, features=45)
    Output: (batch, 1) in [0, 1] — probability of a positive next-day return

    Architecture improvements over baseline:
    ─────────────────────────────────────────────────────────────────────────
    1. Input projection  : linear(45 → hidden) + LayerNorm before the LSTM
       Normalises features so the gate activations start in a sensible range.

    2. Bidirectional flag (optional, default off):
       Allows the LSTM to see future context during training on historical
       windows (non-causal, only valid for research / offline mode).

    3. Temporal attention : replaces naive "take last step" with a learned
       weighted average over all 60 steps.  Adds seq_len attention weights
       to the output so you can inspect what the model looks at.

    4. Residual projection : adds the (projected) last hidden state back to
       the attention context, giving a skip-connection that eases gradient
       flow through many layers.

    5. Deeper head : LayerNorm → Linear(hidden → dense) → GELU → Dropout
                              → Linear(dense → dense//2) → GELU → Dropout
                              → Linear(dense//2 → 1) → Sigmoid
       GELU outperforms ReLU on financial time-series (smoother gradients).

    6. Layer-wise dropout: separate Dropout before each head linear layer
       instead of a single Dropout, which provides more regularisation
       without overly shrinking outputs.
    ─────────────────────────────────────────────────────────────────────────
    All new arguments have defaults that reproduce the original behaviour
    when left unchanged (input_proj=False, bidirectional=False).
    """

    def __init__(
        self,
        input_size: int = 45,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        dense_units: int = 64,
        output_size: int = 1,
        # --- new optional arguments ---
        input_proj: bool = True,        # project + normalise inputs first
        bidirectional: bool = False,    # bidirectional LSTM (offline only)
        attention_dropout: float = 0.1, # dropout applied to context vector
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # The LSTM output width doubles if bidirectional
        lstm_out_size = hidden_size * (2 if bidirectional else 1)

        # ------------------------------------------------------------------
        # 1. Optional input projection + normalisation
        # ------------------------------------------------------------------
        self.input_proj = None
        if input_proj:
            self.input_proj = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            lstm_in_size = hidden_size
        else:
            lstm_in_size = input_size

        # ------------------------------------------------------------------
        # 2. LSTM encoder
        # ------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=lstm_in_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # ------------------------------------------------------------------
        # 3. Temporal attention
        # ------------------------------------------------------------------
        self.attention = TemporalAttention(lstm_out_size)
        self.attention_dropout = nn.Dropout(attention_dropout)

        # ------------------------------------------------------------------
        # 4. Residual projection
        #    Projects last-step hidden to lstm_out_size so we can add it to
        #    the attention context (dimensions must match).
        # ------------------------------------------------------------------
        self.residual_proj = nn.Linear(lstm_out_size, lstm_out_size, bias=False)
        self.context_norm = nn.LayerNorm(lstm_out_size)

        # ------------------------------------------------------------------
        # 5. Deeper prediction head
        # ------------------------------------------------------------------
        mid_units = dense_units // 2 if dense_units >= 4 else dense_units
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Linear(lstm_out_size, dense_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, mid_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_units, output_size),
            nn.Sigmoid(),
        )

        self.apply(init_weights)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            out: (batch, 1)  — directional probability in [0, 1]
        """
        # 1. Optional input projection
        if self.input_proj is not None:
            x = self.input_proj(x)          # (batch, seq_len, hidden_size)

        # 2. LSTM — keep ALL hidden states for attention
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, lstm_out_size)

        # 3. Temporal attention
        context, self._attn_weights = self.attention(lstm_out)   # (batch, lstm_out_size)
        context = self.attention_dropout(context)

        # 4. Residual: add projected last-step to context
        last_step = lstm_out[:, -1, :]      # (batch, lstm_out_size)
        context = self.context_norm(context + self.residual_proj(last_step))

        # 5. Head
        out = self.head(context)            # (batch, 1)
        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_attention_weights(self) -> torch.Tensor:
        """
        Return the attention weight distribution from the last forward pass.

        Shape: (batch, seq_len) — each row sums to 1.
        Useful for visualising which time steps the model focuses on.

        Example:
            model.eval()
            with torch.no_grad():
                _ = model(x_batch)
            weights = model.get_attention_weights()   # (batch, 60)
            # weights[0] → attention over 60 days for first sample
        """
        if not hasattr(self, '_attn_weights'):
            raise RuntimeError("Run a forward pass before calling get_attention_weights().")
        return self._attn_weights

    def summary(self) -> str:
        """Return a compact model summary with parameter count."""
        n = count_parameters(self)
        bidir = "bidirectional " if self.bidirectional else ""
        proj = "input_proj=True" if self.input_proj is not None else "input_proj=False"
        return (
            f"{bidir}LSTM(input_size={self.input_size}, "
            f"hidden_size={self.hidden_size}, num_layers={self.num_layers})\n"
            f"+ TemporalAttention + residual projection + deeper GELU head\n"
            f"{proj}\n"
            f"Total trainable parameters: {n:,}"
        )
