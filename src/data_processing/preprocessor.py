"""
Data preprocessing utilities for time series forecasting
"""
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset): # type: ignore
    """Custom Dataset for time series data"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray): # type: ignore
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):  # type: ignore
        return self.sequences[idx], self.targets[idx]

class TimeSeriesPreprocessor:
    """Preprocessor for time series data"""

    def __init__(self, sequence_length: int = 60, target_column: str = 'value'):
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:  # type: ignore
        """Fit scaler and transform data"""
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")

        # Extract target values
        values = data[self.target_column].values.reshape(-1, 1)  # type: ignore

        # Fit and transform
        scaled_data = self.scaler.fit_transform(values)  # type: ignore
        self.is_fitted = True

        logger.info(f"Data scaled. Shape: {scaled_data.shape}")
        return scaled_data.flatten()  # type: ignore

    def transform(self, data: pd.DataFrame) -> np.ndarray:  # type: ignore
        """Transform data using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")

        values = data[self.target_column].values.reshape(-1, 1)  # type: ignore
        scaled_data = self.scaler.transform(values)  # type: ignore
        return scaled_data.flatten()  # type: ignore

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:  # type: ignore
        """Inverse transform scaled data"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted.")

        if data.ndim == 1:
            data = data.reshape(-1, 1)  # type: ignore

        return self.scaler.inverse_transform(data).flatten()  # type: ignore

    def create_sequences(
        self,
        data: np.ndarray,  # type: ignore
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore
        """Create sequences for training"""
        sequences = []
        targets = []

        for i in range(len(data) - self.sequence_length - forecast_horizon + 1):  # type: ignore
            # Input sequence
            seq = data[i:i + self.sequence_length]  # type: ignore
            # Target (next value(s))
            target = data[  # type: ignore
                i + self.sequence_length : i + self.sequence_length + forecast_horizon
            ]

            sequences.append(seq)  # type: ignore
            targets.append(target)  # type: ignore

        return np.array(sequences), np.array(targets)  # type: ignore

    def split_data(
        self,
        sequences: np.ndarray,  # type: ignore
        targets: np.ndarray,  # type: ignore
        train_size: float = 0.8,
        val_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # type: ignore
        """Split data into train, validation, and test sets"""

        # First split: separate test data
        X_temp, X_test, y_temp, y_test = train_test_split(  # type: ignore
            sequences, targets,
            test_size=1-train_size-val_size,
            random_state=42,
            shuffle=False
        )

        # Second split: separate train and validation
        val_size_adjusted = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(  # type: ignore
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=42,
            shuffle=False
        )

        logger.info(
            f"Data split - Train: {X_train.shape}, "
            f"Val: {X_val.shape}, " # type: ignore
            f"Test: {X_test.shape}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_data_loaders(
        self,
        X_train: np.ndarray,  # type: ignore
        X_val: np.ndarray,  # type: ignore
        X_test: np.ndarray,  # type: ignore
        y_train: np.ndarray,  # type: ignore
        y_val: np.ndarray,  # type: ignore
        y_test: np.ndarray,  # type: ignore
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:  # type: ignore
        """Create PyTorch DataLoaders"""

        # Reshape for LSTM (add feature dimension if needed)
        if X_train.ndim == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # type: ignore
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)  # type: ignore
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)  # type: ignore

        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)  # type: ignore
        val_dataset = TimeSeriesDataset(X_val, y_val)  # type: ignore
        test_dataset = TimeSeriesDataset(X_test, y_test)  # type: ignore

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # type: ignore
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # type: ignore
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # type: ignore

        return train_loader, val_loader, test_loader

def load_sample_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load sample time series data"""
    if file_path:
        return pd.read_csv(file_path)  # type: ignore
    else:
        # Generate sample data
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')  # type: ignore
        np.random.seed(42)
        trend = np.linspace(100, 200, 1000)  # type: ignore
        seasonal = 10 * np.sin(2 * np.pi * np.arange(1000) / 365.25)  # type: ignore
        noise = np.random.normal(0, 5, 1000)  # type: ignore
        values = trend + seasonal + noise  # type: ignore

        return pd.DataFrame({  # type: ignore
            'date': dates,
            'value': values
        })
