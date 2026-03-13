"""
Unit tests for the TimeSeriesPreprocessor class.

This module contains comprehensive tests for data preprocessing functionality
including scaling, sequence creation, and data splitting.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.preprocessor import (  # noqa: E402
    TimeSeriesPreprocessor,  # type: ignore
)


class TestTimeSeriesPreprocessor:  # type: ignore
    """Test suite for TimeSeriesPreprocessor."""

    @pytest.fixture  # type: ignore
    def sample_data(self):  # type: ignore
        """Create sample time series data for testing."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        np.random.seed(42)

        data = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(len(dates)) * 10 + 100,
        })
        return data

    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance."""
        return TimeSeriesPreprocessor(
            target_column='value',
            sequence_length=10,
        )

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = TimeSeriesPreprocessor(
            target_column='target',
            sequence_length=20,
        )

        assert preprocessor.target_column == 'target'
        assert preprocessor.sequence_length == 20
        assert preprocessor.is_fitted is False

    def test_fit_transform(self, preprocessor, sample_data):
        """Test fit_transform method."""
        scaled_data = preprocessor.fit_transform(sample_data)

        assert preprocessor.is_fitted is True
        assert preprocessor.scaler is not None
        assert isinstance(scaled_data, np.ndarray)
        assert len(scaled_data) == len(sample_data)
        # MinMaxScaler should scale to [0, 1]
        assert scaled_data.min() >= 0.0 - 1e-9
        assert scaled_data.max() <= 1.0 + 1e-9

    def test_fit_transform_with_missing_column(self, preprocessor):
        """Test fit_transform with missing target column."""
        invalid_data = pd.DataFrame({
            'wrong_column': np.random.randn(10)
        })

        with pytest.raises(ValueError, match="not found"):
            preprocessor.fit_transform(invalid_data)

    def test_transform_before_fit(self, preprocessor, sample_data):
        """Test that transform raises error before fitting."""
        with pytest.raises(ValueError, match="not fitted"):
            preprocessor.transform(sample_data)

    def test_transform_method(self, preprocessor, sample_data):
        """Test the transform method."""
        preprocessor.fit_transform(sample_data)
        transformed_data = preprocessor.transform(sample_data)

        assert isinstance(transformed_data, np.ndarray)
        assert len(transformed_data) == len(sample_data)
        # Values should be scaled to [0, 1]
        assert transformed_data.min() >= 0.0 - 1e-9
        assert transformed_data.max() <= 1.0 + 1e-9

    def test_create_sequences(self, preprocessor, sample_data):
        """Test sequence creation."""
        scaled_data = preprocessor.fit_transform(sample_data)
        X, y = preprocessor.create_sequences(scaled_data, forecast_horizon=1)

        expected_samples = len(scaled_data) - preprocessor.sequence_length - 1 + 1
        assert X.shape == (expected_samples, preprocessor.sequence_length)
        assert y.shape == (expected_samples, 1)

    def test_create_sequences_with_forecast_horizon(self, preprocessor, sample_data):
        """Test sequence creation with multi-step forecast horizon."""
        scaled_data = preprocessor.fit_transform(sample_data)
        forecast_horizon = 7
        X, y = preprocessor.create_sequences(scaled_data, forecast_horizon=forecast_horizon)

        expected_samples = len(scaled_data) - preprocessor.sequence_length - forecast_horizon + 1
        assert X.shape == (expected_samples, preprocessor.sequence_length)
        assert y.shape == (expected_samples, forecast_horizon)

    def test_create_sequences_insufficient_data(self):
        """Test sequence creation with insufficient data."""
        preprocessor = TimeSeriesPreprocessor(sequence_length=60)
        short_data = pd.DataFrame({
            'value': np.random.randn(5),
        })

        scaled_data = preprocessor.fit_transform(short_data)
        X, y = preprocessor.create_sequences(scaled_data, forecast_horizon=1)

        # Not enough data for any sequences
        assert len(X) == 0
        assert len(y) == 0

    def test_inverse_transform(self, preprocessor, sample_data):
        """Test inverse transformation recovers original values."""
        original_values = sample_data['value'].values
        scaled_data = preprocessor.fit_transform(sample_data)
        recovered = preprocessor.inverse_transform(scaled_data)

        np.testing.assert_allclose(original_values, recovered, rtol=1e-5)

    def test_inverse_transform_before_fit(self, preprocessor):
        """Test that inverse_transform raises error before fitting."""
        with pytest.raises(ValueError, match="not fitted"):
            preprocessor.inverse_transform(np.array([0.5, 0.6]))

    def test_sequence_consistency(self, preprocessor, sample_data):
        """Test that sequences are created consistently."""
        scaled_data = preprocessor.fit_transform(sample_data)

        X1, y1 = preprocessor.create_sequences(scaled_data)
        X2, y2 = preprocessor.create_sequences(scaled_data)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_sequence_parameters(self, sample_data):
        """Test with different sequence length."""
        preprocessor = TimeSeriesPreprocessor(
            target_column='value',
            sequence_length=30,
        )

        scaled_data = preprocessor.fit_transform(sample_data)
        X, y = preprocessor.create_sequences(scaled_data, forecast_horizon=7)

        assert X.shape[1] == 30  # sequence_length
        assert y.shape[1] == 7   # forecast_horizon

    def test_edge_case_single_feature(self):
        """Test with minimal data that still works."""
        data = pd.DataFrame({
            'value': np.random.randn(50)
        })

        preprocessor = TimeSeriesPreprocessor(
            target_column='value',
            sequence_length=10,
        )

        scaled_data = preprocessor.fit_transform(data)
        X, y = preprocessor.create_sequences(scaled_data, forecast_horizon=1)

        assert X.shape[1] == 10
        assert y.shape[1] == 1
        assert len(X) == 50 - 10 - 1 + 1

    def test_reproducibility(self, sample_data):
        """Test that preprocessing is reproducible."""
        preprocessor1 = TimeSeriesPreprocessor(
            target_column='value',
            sequence_length=10,
        )

        preprocessor2 = TimeSeriesPreprocessor(
            target_column='value',
            sequence_length=10,
        )

        scaled1 = preprocessor1.fit_transform(sample_data)
        scaled2 = preprocessor2.fit_transform(sample_data)

        X1, y1 = preprocessor1.create_sequences(scaled1)
        X2, y2 = preprocessor2.create_sequences(scaled2)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_split_data(self, preprocessor, sample_data):
        """Test data splitting into train/val/test sets."""
        scaled_data = preprocessor.fit_transform(sample_data)
        X, y = preprocessor.create_sequences(scaled_data)

        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

        # All data should be accounted for
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert len(y_train) + len(y_val) + len(y_test) == len(y)

        # Train should be the largest split
        assert len(X_train) > len(X_val)
        assert len(X_train) > len(X_test)
