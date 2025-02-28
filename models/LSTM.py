import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ForecastLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        input_size,
        hidden_size,
        num_layers,
        seq_length,
        other_dim,
        hidden_nodes,
        linear_layers,
        device="cuda",
        bidirectional=False,
        dropout=0.1,
    ):
        super(ForecastLSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.seq_length = seq_length
        self.num_directions = 2 if bidirectional else 1
        self.other_dim = other_dim

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout / 2,
        )

        layers = []

        for i in range(linear_layers - 1):
            if i == 0:
                layers.append(
                    nn.Linear(hidden_size * self.num_directions, hidden_nodes)
                )
            else:
                layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_nodes, num_classes))

        self.linear = nn.Sequential(*layers)

    def forward(self, x):

        hidden_states = Variable(
            torch.zeros(
                self.num_layers * self.num_directions, x.size(0), self.hidden_size
            )
        ).to(self.device)

        cell_states = Variable(
            torch.zeros(
                self.num_layers * self.num_directions, x.size(0), self.hidden_size
            )
        ).to(self.device)

        h_all, (h_out, c_out) = self.lstm(x, (hidden_states, cell_states))

        out = self.linear(h_all).squeeze(-1)

        return out
