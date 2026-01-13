"""
Cancer Detection System - Data Loading Module.

This module provides functionality to load and validate the
Wisconsin Breast Cancer Dataset from sklearn or external sources.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from pathlib import Path
import logging

from .config import default_config, DataConfig


logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Exception raised when data validation fails."""
    pass


class CancerDataLoader:
    """
    Data loader for the Wisconsin Breast Cancer Dataset.
    
    This class handles loading data from sklearn's built-in dataset
    or from external CSV files, with comprehensive validation.
    
    Attributes:
        config: Data configuration object
        
    Example:
        >>> loader = CancerDataLoader()
        >>> X, y = loader.load_data()
        >>> print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize the data loader.
        
        Args:
            config: Optional data configuration. Uses default if not provided.
        """
        self.config = config or default_config.data
        self._data_cache: Optional[Dict[str, Any]] = None
    
    def load_data(
        self,
        source: str = "sklearn",
        file_path: Optional[Path] = None,
        return_dataframe: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the breast cancer dataset.
        
        Args:
            source: Data source - "sklearn" for built-in dataset, 
                   "csv" for external file
            file_path: Path to CSV file (required if source="csv")
            return_dataframe: If True, return pandas DataFrame instead of numpy arrays
            
        Returns:
            Tuple of (features, targets) as numpy arrays or pandas objects
            
        Raises:
            DataValidationError: If data validation fails
            FileNotFoundError: If CSV file not found
        """
        if source == "sklearn":
            X, y = self._load_from_sklearn()
        elif source == "csv":
            if file_path is None:
                raise ValueError("file_path required for CSV source")
            X, y = self._load_from_csv(file_path)
        else:
            raise ValueError(f"Unknown source: {source}. Use 'sklearn' or 'csv'")
        
        # Validate data
        self._validate_data(X, y)
        
        logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
        
        if return_dataframe:
            X = pd.DataFrame(X, columns=self.config.feature_names)
            y = pd.Series(y, name="target")
        
        return X, y
    
    def _load_from_sklearn(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from sklearn's built-in dataset."""
        if self._data_cache is None:
            data = load_breast_cancer()
            self._data_cache = data
        else:
            data = self._data_cache
            
        return data.data, data.target
    
    def _load_from_csv(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from a CSV file.
        
        Expected format:
        - Features in columns matching config.feature_names
        - Target in 'target' or 'diagnosis' column
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Find target column
        target_col = None
        for col in ["target", "diagnosis", "label"]:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            raise DataValidationError("No target column found in CSV")
        
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        
        return X, y
    
    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate loaded data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Raises:
            DataValidationError: If validation fails
        """
        # Check dimensions
        if X.ndim != 2:
            raise DataValidationError(f"Expected 2D feature matrix, got {X.ndim}D")
        
        if y.ndim != 1:
            raise DataValidationError(f"Expected 1D target vector, got {y.ndim}D")
        
        # Check sample match
        if len(X) != len(y):
            raise DataValidationError(
                f"Feature/target mismatch: {len(X)} samples vs {len(y)} targets"
            )
        
        # Check for NaN values
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            raise DataValidationError(f"Found {nan_count} NaN values in features")
        
        # Check for infinite values
        if np.isinf(X).any():
            inf_count = np.isinf(X).sum()
            raise DataValidationError(f"Found {inf_count} infinite values in features")
        
        # Check target values (should be binary: 0 or 1)
        unique_targets = np.unique(y)
        if not np.all(np.isin(unique_targets, [0, 1])):
            raise DataValidationError(
                f"Expected binary targets (0, 1), got: {unique_targets}"
            )
        
        logger.debug("Data validation passed")
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset statistics and metadata
        """
        X, y = self.load_data()
        
        return {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "feature_names": self.config.feature_names,
            "target_names": self.config.target_names,
            "class_distribution": {
                "malignant (0)": int(np.sum(y == 0)),
                "benign (1)": int(np.sum(y == 1)),
            },
            "class_balance": float(np.mean(y)),
            "feature_stats": {
                "mean": X.mean(axis=0).tolist(),
                "std": X.std(axis=0).tolist(),
                "min": X.min(axis=0).tolist(),
                "max": X.max(axis=0).tolist(),
            }
        }
    
    def get_feature_names(self) -> list:
        """Return list of feature names."""
        return self.config.feature_names.copy()
    
    def get_target_names(self) -> list:
        """Return list of target class names."""
        return self.config.target_names.copy()
