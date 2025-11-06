"""
Unit tests for the TimeSeriesPreprocessor class.

This module contains comprehensive tests for data preprocessing functionality
including scaling, sequence creation, and feature engineering.
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
            'feature_1': np.random.randn(len(dates)) * 5 + 50,
            'feature_2': np.random.exponential(2, len(dates))
        })
        return data

    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance."""
        return TimeSeriesPreprocessor(
            target_column='value',
            date_column='date',
            sequence_length=10,
            forecast_horizon=1
        )

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = TimeSeriesPreprocessor(
            target_column='target',
            date_column='timestamp',
            sequence_length=20,
            forecast_horizon=5
        )

        assert preprocessor.target_column == 'target'
        assert preprocessor.date_column == 'timestamp'
        assert preprocessor.sequence_length == 20
        assert preprocessor.forecast_horizon == 5
        assert preprocessor.is_fitted is False

    def test_fit_method(self, preprocessor, sample_data):
        """Test the fit method."""
        preprocessor.fit(sample_data)

        assert preprocessor.is_fitted is True
        assert preprocessor.target_scaler is not None
        assert preprocessor.feature_scaler is not None

        # Check that feature columns are identified correctly
        expected_features = ['feature_1', 'feature_2']
        assert set(preprocessor.feature_columns) == set(expected_features)

    def test_fit_with_missing_columns(self, preprocessor):
        """Test fit method with missing required columns."""
        invalid_data = pd.DataFrame({
            'wrong_date': pd.date_range('2020-01-01', periods=10),
            'wrong_value': np.random.randn(10)
        })

        with pytest.raises((KeyError, ValueError)):
            preprocessor.fit(invalid_data)

    def test_transform_before_fit(self, preprocessor, sample_data):
        """Test that transform raises error before fitting."""
        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(sample_data)

    def test_create_sequences_before_fit(self, preprocessor, sample_data):
        """Test that create_sequences raises error before fitting."""
        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.create_sequences(sample_data)

    def test_transform_method(self, preprocessor, sample_data):
        """Test the transform method."""
        preprocessor.fit(sample_data)
        transformed_data = preprocessor.transform(sample_data)

        # Check that data is transformed correctly
        assert isinstance(transformed_data, pd.DataFrame)
        assert len(transformed_data) == len(sample_data)

        # Check that target values are scaled
        original_mean = sample_data['value'].mean()
        transformed_mean = transformed_data['value'].mean()
        assert abs(transformed_mean) < abs(original_mean)  # Should be closer to 0

    def test_create_sequences(self, preprocessor, sample_data):
        """Test sequence creation."""
        preprocessor.fit(sample_data)  # type: ignore
        X, y = preprocessor.create_sequences(sample_data)  # type: ignore

        # Check shapes
        expected_samples = (  # type: ignore
            len(sample_data) - preprocessor.sequence_length  # type: ignore
            - preprocessor.forecast_horizon + 1  # type: ignore
        )
        expected_features = len(preprocessor.feature_columns) + 1  # type: ignore  # +1 for target

        assert X.shape == (  # type: ignore
            expected_samples, preprocessor.sequence_length, expected_features
        )
        assert y.shape == (expected_samples, preprocessor.forecast_horizon)  # type: ignore

        # Check data types
        assert X.dtype == np.float32  # type: ignore
        assert y.dtype == np.float32  # type: ignore

    def test_create_sequences_insufficient_data(self, preprocessor):  # type: ignore
        """Test sequence creation with insufficient data."""
        # Create data shorter than sequence length
        short_data = pd.DataFrame({  # type: ignore
            'date': pd.date_range('2020-01-01', periods=5),  # type: ignore
            'value': np.random.randn(5),  # type: ignore
            'feature_1': np.random.randn(5)  # type: ignore
        })

        preprocessor.fit(short_data)  # type: ignore

        with pytest.raises(ValueError, match="not enough data"):  # type: ignore
            preprocessor.create_sequences(short_data)  # type: ignore

    def test_fit_transform(self, preprocessor, sample_data):  # type: ignore
        """Test fit_transform method."""
        transformed_data = preprocessor.fit_transform(sample_data)  # type: ignore

        assert preprocessor.is_fitted is True  # type: ignore
        assert isinstance(transformed_data, pd.DataFrame)  # type: ignore
        assert len(transformed_data) == len(sample_data)  # type: ignore

    def test_inverse_transform_target(self, preprocessor, sample_data):  # type: ignore
        """Test inverse transformation of target values."""
        preprocessor.fit(sample_data)  # type: ignore
        transformed_data = preprocessor.transform(sample_data)  # type: ignore

        # Transform and inverse transform target
        original_target = sample_data['value'].values.reshape(-1, 1)  # type: ignore
        transformed_target = transformed_data['value'].values.reshape(-1, 1)  # type: ignore
        inverse_transformed = preprocessor.inverse_transform_target(transformed_target)  # type: ignore

        # Check that inverse transformation recovers original values
        np.testing.assert_allclose(original_target, inverse_transformed, rtol=1e-5)  # type: ignore

    def test_sequence_consistency(self, preprocessor, sample_data):  # type: ignore
        """Test that sequences are created consistently."""
        preprocessor.fit(sample_data)  # type: ignore

        # Create sequences multiple times
        X1, y1 = preprocessor.create_sequences(sample_data)  # type: ignore
        X2, y2 = preprocessor.create_sequences(sample_data)  # type: ignore

        # Should be identical
        np.testing.assert_array_equal(X1, X2)  # type: ignore
        np.testing.assert_array_equal(y1, y2)  # type: ignore

    def test_date_sorting(self, preprocessor):  # type: ignore
        """Test that data is sorted by date correctly."""
        # Create unsorted data
        dates = pd.date_range('2020-01-01', periods=10)  # type: ignore
        unsorted_data = pd.DataFrame({  # type: ignore
            'date': dates,  # type: ignore
            'value': np.arange(10),  # type: ignore
            'feature_1': np.arange(10) * 2  # type: ignore
        })

        # Shuffle the data
        unsorted_data = unsorted_data.sample(frac=1).reset_index(drop=True)  # type: ignore

        preprocessor.fit(unsorted_data)  # type: ignore
        transformed_data = preprocessor.transform(unsorted_data)  # type: ignore

        # Check that dates are sorted
        assert transformed_data['date'].is_monotonic_increasing  # type: ignore

    def test_missing_values_handling(self, preprocessor):  # type: ignore
        """Test handling of missing values."""
        data_with_nan = pd.DataFrame({  # type: ignore
            'date': pd.date_range('2020-01-01', periods=20),  # type: ignore
            'value': [1, 2, np.nan, 4, 5] * 4,  # type: ignore
            'feature_1': np.random.randn(20)  # type: ignore
        })

        # Should handle missing values gracefully
        with pytest.raises(ValueError, match="contains missing values"):  # type: ignore
            preprocessor.fit(data_with_nan)  # type: ignore

    def test_feature_scaling(self, preprocessor, sample_data):  # type: ignore
        """Test that features are scaled properly."""
        preprocessor.fit(sample_data)  # type: ignore
        transformed_data = preprocessor.transform(sample_data)  # type: ignore

        # Check that features are approximately standardized
        for feature in preprocessor.feature_columns:  # type: ignore
            feature_values = transformed_data[feature].values  # type: ignore
            assert abs(feature_values.mean()) < 0.1  # type: ignore  # Should be close to 0
            assert abs(feature_values.std() - 1.0) < 0.1  # type: ignore  # Should be close to 1

    def test_different_sequence_parameters(self, sample_data):  # type: ignore
        """Test with different sequence length and forecast horizon."""
        preprocessor = TimeSeriesPreprocessor(  # type: ignore
            target_column='value',
            date_column='date',
            sequence_length=30,
            forecast_horizon=7
        )

        preprocessor.fit(sample_data)  # type: ignore
        X, y = preprocessor.create_sequences(sample_data)  # type: ignore

        assert X.shape[1] == 30  # type: ignore  # sequence_length
        assert y.shape[1] == 7  # type: ignore   # forecast_horizon

    def test_edge_case_single_feature(self):  # type: ignore
        """Test with only target column (no additional features)."""
        data = pd.DataFrame({  # type: ignore
            'date': pd.date_range('2020-01-01', periods=50),  # type: ignore
            'value': np.random.randn(50)  # type: ignore
        })

        preprocessor = TimeSeriesPreprocessor(  # type: ignore
            target_column='value',
            date_column='date',
            sequence_length=10,
            forecast_horizon=1
        )

        preprocessor.fit(data)  # type: ignore
        X, y = preprocessor.create_sequences(data)  # type: ignore

        # Should work with only target column
        assert X.shape[2] == 1  # type: ignore  # Only target feature
        assert y.shape[1] == 1  # type: ignore  # forecast_horizon

    def test_reproducibility(self, preprocessor, sample_data):  # type: ignore
        """Test that preprocessing is reproducible."""
        # Fit twice and compare results
        preprocessor1 = TimeSeriesPreprocessor(  # type: ignore
            target_column='value',
            date_column='date',
            sequence_length=10,
            forecast_horizon=1
        )

        preprocessor2 = TimeSeriesPreprocessor(  # type: ignore
            target_column='value',
            date_column='date',
            sequence_length=10,
            forecast_horizon=1
        )

        preprocessor1.fit(sample_data)  # type: ignore
        preprocessor2.fit(sample_data)  # type: ignore

        X1, y1 = preprocessor1.create_sequences(sample_data)  # type: ignore
        X2, y2 = preprocessor2.create_sequences(sample_data)  # type: ignore

        # Results should be identical
        np.testing.assert_array_equal(X1, X2)  # type: ignore
        np.testing.assert_array_equal(y1, y2)  # type: ignore
