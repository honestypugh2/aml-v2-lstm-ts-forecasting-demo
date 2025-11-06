"""
LSTM Model for Time Series Forecasting using PyTorch
"""
from typing import Tuple

import torch
import torch.nn as nn


class LSTMTimeSeriesModel(nn.Module):
    """
    LSTM model for time series forecasting
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 50,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(LSTMTimeSeriesModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Calculate linear layer input size
        linear_input_size = hidden_size * 2 if bidirectional else hidden_size

        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state and cell state
        batch_size = x.size(0)
        h0, c0 = self.init_hidden(batch_size, x.device)

        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # Take the last output
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state and cell state

        Args:
            batch_size: Batch size
            device: Device (CPU or GPU)

        Returns:
            Tuple of hidden state and cell state tensors
        """
        num_directions = 2 if self.bidirectional else 1

        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(device)

        c0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(device)

        return h0, c0

class LSTMConfig:
    """Configuration class for LSTM model"""

    def __init__(self):
        self.input_size = 1
        self.hidden_size = 50
        self.num_layers = 2
        self.output_size = 1
        self.dropout = 0.2
        self.bidirectional = False
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.sequence_length = 60
        self.train_split = 0.8
        self.val_split = 0.1
