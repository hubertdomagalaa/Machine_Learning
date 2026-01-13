"""
Unit tests for Cancer Detection System - Data Loader Module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.cancer.data_loader import CancerDataLoader, DataValidationError


class TestCancerDataLoader:
    """Test suite for CancerDataLoader class."""
    
    def test_load_data_returns_correct_types(self):
        """Test that load_data returns numpy arrays."""
        loader = CancerDataLoader()
        X, y = loader.load_data()
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
    
    def test_load_data_correct_shape(self):
        """Test that loaded data has correct dimensions."""
        loader = CancerDataLoader()
        X, y = loader.load_data()
        
        # Wisconsin dataset has 569 samples and 30 features
        assert X.shape == (569, 30)
        assert y.shape == (569,)
    
    def test_load_data_as_dataframe(self):
        """Test loading data as pandas DataFrame."""
        loader = CancerDataLoader()
        X, y = loader.load_data(return_dataframe=True)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X.columns) == 30
    
    def test_target_values_are_binary(self):
        """Test that target values are only 0 or 1."""
        loader = CancerDataLoader()
        X, y = loader.load_data()
        
        unique_vals = np.unique(y)
        assert len(unique_vals) == 2
        assert 0 in unique_vals
        assert 1 in unique_vals
    
    def test_no_nan_values(self):
        """Test that there are no NaN values in features."""
        loader = CancerDataLoader()
        X, y = loader.load_data()
        
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
    
    def test_get_feature_names(self):
        """Test that feature names are returned correctly."""
        loader = CancerDataLoader()
        names = loader.get_feature_names()
        
        assert len(names) == 30
        assert "mean radius" in names
        assert "worst fractal dimension" in names
    
    def test_get_target_names(self):
        """Test that target names are returned correctly."""
        loader = CancerDataLoader()
        names = loader.get_target_names()
        
        assert len(names) == 2
        assert "malignant" in names
        assert "benign" in names
    
    def test_get_data_info(self):
        """Test data info dictionary."""
        loader = CancerDataLoader()
        info = loader.get_data_info()
        
        assert info["n_samples"] == 569
        assert info["n_features"] == 30
        assert "class_distribution" in info
        assert "feature_stats" in info
    
    def test_data_caching(self):
        """Test that data is cached after first load."""
        loader = CancerDataLoader()
        
        # First load
        X1, y1 = loader.load_data()
        
        # Second load should use cache
        X2, y2 = loader.load_data()
        
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)
    
    def test_invalid_source_raises_error(self):
        """Test that invalid source raises ValueError."""
        loader = CancerDataLoader()
        
        with pytest.raises(ValueError, match="Unknown source"):
            loader.load_data(source="invalid")
    
    def test_csv_without_path_raises_error(self):
        """Test that CSV source without path raises ValueError."""
        loader = CancerDataLoader()
        
        with pytest.raises(ValueError, match="file_path required"):
            loader.load_data(source="csv")
    
    def test_csv_file_not_found_raises_error(self):
        """Test that missing CSV file raises FileNotFoundError."""
        loader = CancerDataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_data(source="csv", file_path=Path("nonexistent.csv"))


class TestDataValidation:
    """Test suite for data validation functionality."""
    
    def test_validation_passes_for_valid_data(self):
        """Test that validation passes for sklearn dataset."""
        loader = CancerDataLoader()
        # Should not raise
        X, y = loader.load_data()
        assert X is not None
        assert y is not None
    
    def test_validation_detects_nan(self):
        """Test that NaN values are detected."""
        loader = CancerDataLoader()
        
        # Create data with NaN
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        y = np.array([0, 1])
        
        with pytest.raises(DataValidationError, match="NaN"):
            loader._validate_data(X, y)
    
    def test_validation_detects_inf(self):
        """Test that infinite values are detected."""
        loader = CancerDataLoader()
        
        X = np.array([[1.0, np.inf], [3.0, 4.0]])
        y = np.array([0, 1])
        
        with pytest.raises(DataValidationError, match="infinite"):
            loader._validate_data(X, y)
    
    def test_validation_detects_dimension_mismatch(self):
        """Test that sample/target mismatch is detected."""
        loader = CancerDataLoader()
        
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1, 0])  # Wrong length
        
        with pytest.raises(DataValidationError, match="mismatch"):
            loader._validate_data(X, y)
    
    def test_validation_detects_non_binary_targets(self):
        """Test that non-binary targets are detected."""
        loader = CancerDataLoader()
        
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 2])  # Should be 0 or 1
        
        with pytest.raises(DataValidationError, match="binary"):
            loader._validate_data(X, y)
