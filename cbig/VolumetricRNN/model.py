"""
RNN model for multivariate volumetric trajectory prediction.

Given a sequence of visits for a subject (each visit: 151-dim input),
the model predicts the 145 brain volumes at each visit.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VolumetricRNN(nn.Module):
    """
    LSTM (or GRU) that maps a sequence of visits to predicted brain volumes.

    Input  per step : 151 features (145 baseline vols + 5 clinical + 1 time)
    Output per step : 145 predicted brain volumes
    """

    def __init__(
        self,
        input_size: int = 151,
        hidden_size: int = 256,
        output_size: int = 145,
        n_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = 'LSTM',
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_type = rnn_type

        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        # dropout between layers only when n_layers > 1
        rnn_dropout = dropout if n_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size, hidden_size, n_layers,
            batch_first=True, dropout=rnn_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, T_max, input_size)  — padded input sequences
        lengths : (B,)                    — true sequence length per sample
        returns : (B, T_max, output_size) — predicted volumes (padded)
        """
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)  # (B, T_max, hidden)
        out = self.dropout(out)
        return self.fc(out)                                  # (B, T_max, 145)
