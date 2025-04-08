import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvLSTM(nn.Module):
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
        self.channels = 4

        self.conv_in_dim = seq_length
        self.kernel_size = 8
        self.conv_dim1 = int(
            ((self.conv_in_dim) - 1 * (self.kernel_size - 1) - 1) / 1 + 1
        )
        self.conv_dim2 = int(
            ((self.conv_dim1) - 1 * (self.kernel_size - 1) - 1) / 1 + 1
        )

        # Convolutional Preprocessor
        decode_conv = []
        decode_conv.append(nn.Unflatten(1, (1, self.conv_in_dim)))
        decode_conv.append(nn.Conv1d(1, self.channels, self.kernel_size, bias=False))
        decode_conv.append(nn.ReLU())
        decode_conv.append(nn.Flatten())

        self.decode_conv = nn.Sequential(*decode_conv)

        preprocess_decode = []
        preprocess_decode.append(nn.Dropout(dropout))
        preprocess_decode.append(
            nn.Linear(
                self.conv_dim1 * self.channels + self.other_dim,
                self.input_size * (self.seq_length),
            )
        )
        preprocess_decode.append(nn.ReLU())
        preprocess_decode.append(nn.Dropout(dropout))
        # preprocess_decode.append(nn.Linear(self.input_size*(self.seq_length//2), self.input_size*self.seq_length))
        # preprocess_decode.append(nn.ReLU())

        self.decode = nn.Sequential(*preprocess_decode)

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
        x_preprocess = self.decode_conv(x[:, self.other_dim :])
        x_preprocess = torch.cat((x_preprocess, x[:, : self.other_dim]), dim=1)
        x_preprocess = self.decode(x_preprocess)
        x_preprocess = x_preprocess.reshape(
            x.shape[0], self.seq_length, self.input_size
        )

        h_0 = Variable(
            torch.zeros(
                self.num_layers * self.num_directions, x.size(0), self.hidden_size
            )
        ).to(self.device)

        c_0 = Variable(
            torch.zeros(
                self.num_layers * self.num_directions, x.size(0), self.hidden_size
            )
        ).to(self.device)

        h_all, (h_out, c_out) = self.lstm(x_preprocess, (h_0, c_0))

        out = self.linear(h_all).squeeze(-1)

        return out
