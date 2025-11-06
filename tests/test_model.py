"""
Unit tests for LSTM model
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.preprocessor import (  # type: ignore  # noqa: E402
    TimeSeriesPreprocessor,
    load_sample_data,
)
from models.lstm_model import (  # type: ignore  # noqa: E402
    LSTMConfig,
    LSTMTimeSeriesModel,
)


class TestLSTMModel:
    """Test cases for LSTM model"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = LSTMConfig()  # type: ignore
        self.model = LSTMTimeSeriesModel(  # type: ignore
            input_size=self.config.input_size,  # type: ignore
            hidden_size=self.config.hidden_size,  # type: ignore
            num_layers=self.config.num_layers,  # type: ignore
            output_size=self.config.output_size,  # type: ignore
            dropout=self.config.dropout  # type: ignore
        )

    def test_model_initialization(self):
        """Test model initialization"""
        assert self.model is not None  # type: ignore
        assert isinstance(self.model, LSTMTimeSeriesModel)  # type: ignore

    def test_model_forward_pass(self):
        """Test forward pass through the model"""
        batch_size = 32
        sequence_length = 60

        # Create random input tensor
        input_tensor = torch.randn(batch_size, sequence_length, self.config.input_size)  # type: ignore

        # Forward pass
        output = self.model(input_tensor)  # type: ignore

        assert output.shape == (batch_size, self.config.output_size)  # type: ignore

    def test_hidden_initialization(self):
        """Test hidden state initialization"""
        batch_size = 3
        device = torch.device("cpu")

        h0, c0 = self.model.init_hidden(batch_size, device)

        expected_shape = (self.config.num_layers, batch_size, self.config.hidden_size)
        assert h0.shape == expected_shape
        assert c0.shape == expected_shape
        assert h0.device == device
        assert c0.device == device

class TestTimeSeriesPreprocessor:
    """Test cases for time series preprocessor"""

    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = TimeSeriesPreprocessor(sequence_length=60)
        self.sample_data = load_sample_data()

    def test_data_loading(self):
        """Test sample data loading"""
        assert isinstance(self.sample_data, pd.DataFrame)
        assert 'value' in self.sample_data.columns
        assert 'date' in self.sample_data.columns
        assert len(self.sample_data) > 0

    def test_fit_transform(self):
        """Test data scaling"""
        scaled_data = self.preprocessor.fit_transform(self.sample_data)

        assert isinstance(scaled_data, np.ndarray)
        assert len(scaled_data) == len(self.sample_data)
        assert 0 <= scaled_data.min() <= 1
        assert 0 <= scaled_data.max() <= 1
        assert self.preprocessor.is_fitted

    def test_sequence_creation(self):
        """Test sequence creation"""
        scaled_data = self.preprocessor.fit_transform(self.sample_data)
        sequences, targets = self.preprocessor.create_sequences(scaled_data)

        expected_num_sequences = len(scaled_data) - self.preprocessor.sequence_length
        assert len(sequences) == expected_num_sequences
        assert len(targets) == expected_num_sequences
        assert sequences.shape[1] == self.preprocessor.sequence_length
        assert targets.shape[1] == 1  # forecast_horizon = 1

    def test_data_split(self):
        """Test data splitting"""
        scaled_data = self.preprocessor.fit_transform(self.sample_data)
        sequences, targets = self.preprocessor.create_sequences(scaled_data)

        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(
            sequences, targets
        )

        total_samples = len(sequences)
        expected_train = int(total_samples * 0.8)
        expected_val = int(total_samples * 0.1)

        assert len(X_train) <= expected_train + 1  # Allow for rounding
        assert len(X_val) <= expected_val + 1
        assert len(X_test) > 0

        # Check that all data is accounted for
        total_split = len(X_train) + len(X_val) + len(X_test)
        assert total_split == total_samples

class TestModelConfig:
    """Test cases for model configuration"""

    def test_config_creation(self):
        """Test configuration creation"""
        config = LSTMConfig()

        assert config.input_size > 0
        assert config.hidden_size > 0
        assert config.num_layers > 0
        assert config.output_size > 0
        assert 0 <= config.dropout <= 1
        assert config.learning_rate > 0
        assert config.batch_size > 0
        assert config.num_epochs > 0
        assert config.sequence_length > 0

if __name__ == "__main__":
    pytest.main([__file__])
