import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Linear, ReLU
import numpy as np

# import transformer encoder model from torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def TransformerEncoderForcaster(Module):
    def __init__(
        self,
        seq_length,
        embed_dim,
        num_heads,
        num_layers,
        hidden_size,
        hidden_nodes,
        forecast_length,
        device="cuda",
        dropout=0.1,
    ):
        super(TransformerEncoderForcaster, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.device = device
        self.seq_length = seq_length
        self.forecast_length = forecast_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.seq_length = seq_length
        self.other_dim = other_dim

        self.pos_encoder = PositionalEncoding(
            d_model=embed_dim,
            dropout=dropout,
            max_len=seq_length,
        )

        self.transformer_encoder = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_nodes,
            dropout=dropout,
            batch_first=True,
        )

        self.linear = Sequential(
            Linear(seq_length * embed_dim, hidden_size),
            ReLU(),
            Linear(hidden_size, int(hidden_size / 2)),
            ReLU(),
            Linear(int(hidden_size / 2), int(hidden_size / 4)),
            ReLU(),
            Linear(int(hidden_size / 4), int(hidden_size / 8)),
            ReLU(),
            Linear(int(hidden_size / 8), int(hidden_size / 16)),
            ReLU(),
            Linear(int(hidden_size / 16), forecast_length),
        )

    def forward(self, x):
        # input dim should be (batch_size, seq_length, embed_dim)
        x = self.pos_encoder(x)
        # output dim should be (seq_length, batch_size, embed_dim)
        output = self.transformer_encoder(x)
        # output dim should still be (seq_length, batch_size, embed_dim)
        # flatten the output on the second dimension
        output = output.view(output.size(0), -1)
        # output dim should be (batch_size, seq_length*embed_dim)
        output = self.linear(output)
        return output
