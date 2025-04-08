import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ForecastLSTM(nn.Module):
    def __init__(
        self,
        num_output_seq,
        input_sequence_len,
        n_other_features,
        hidden_size,
        num_layers,
        other_dim,
        hidden_nodes,
        linear_layers,
        device="cuda",
        bidirectional=False,
        dropout=0.1,
    ):
        super(ForecastLSTM, self).__init__()

        self.num_classes = num_output_seq
        self.num_layers = num_layers
        self.input_seq_len = input_sequence_len
        self.hidden_size = hidden_size
        self.device = device
        self.n_other_feat = n_other_features
        self.num_directions = 2 if bidirectional else 1
        self.other_dim = other_dim

        self.lstm = nn.LSTM(
            input_size=input_sequence_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout / 2,
        )

        layers = []
        self.encoding_layer = nn.Sequential(
            nn.Linear(self.n_other_feat, hidden_nodes),
            nn.Relu(),
            nn.Linear(hidden_nodes, self.n_other_feat)
        )
        for i in range(linear_layers - 1):
            if i == 0:
                layers.append(
                    nn.Linear(hidden_size * self.num_directions + self.n_other_feat, hidden_nodes)
                )
            else:
                layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_nodes, num_classes))

        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        other_feat_x = x[:self.n_other_feat]
        input_seq_x = x[self.n_other_feat:]
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

        h_all, (h_out, c_out) = self.lstm(input_seq_x, (hidden_states, cell_states))
        en_x = self.encoding_layer(other_feat_x)
        # Unsqueeze en_x to add a sequence dimension, then expand to match h_all's seq_len
        en_x_expanded = en_x.unsqueeze(0).expand(h_all.size(0), -1, -1)  # shape: [seq_len, batch_size, feature_size]
        
        # Concatenate h_all and the expanded en_x along the last dimension
        concat = torch.cat((h_all, en_x_expanded), dim=2)
        
        out = self.linear(concat).squeeze(-1)
        return out
