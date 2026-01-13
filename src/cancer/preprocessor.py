"""
Cancer Detection System - Data Preprocessing Module.

This module provides feature scaling, normalization, and 
transformation utilities for the cancer detection pipeline.
"""

from typing import Tuple, Optional, List
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib
from pathlib import Path
import logging

from .config import default_config, ModelConfig, PathConfig


logger = logging.getLogger(__name__)


class CancerPreprocessor:
    """
    Preprocessor for cancer detection features.
    
    Handles feature scaling, train/test splitting, and optional
    dimensionality reduction using PCA.
    
    Attributes:
        scaler: Fitted StandardScaler for feature normalization
        pca: Optional fitted PCA transformer
        is_fitted: Whether the preprocessor has been fitted
        
    Example:
        >>> preprocessor = CancerPreprocessor()
        >>> X_train, X_test, y_train, y_test = preprocessor.fit_transform_split(X, y)
        >>> X_new_scaled = preprocessor.transform(X_new)
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        path_config: Optional[PathConfig] = None,
        scaler_type: str = "standard",
        use_pca: bool = False,
        pca_components: Optional[int] = None
    ):
        """
        Initialize the preprocessor.
        
        Args:
            model_config: Model configuration for train/test split params
            path_config: Path configuration for saving artifacts
            scaler_type: Type of scaler - "standard" or "minmax"
            use_pca: Whether to apply PCA dimensionality reduction
            pca_components: Number of PCA components (None = keep all)
        """
        self.model_config = model_config or default_config.model
        self.path_config = path_config or default_config.paths
        
        # Initialize scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.scaler_type = scaler_type
        
        # PCA settings
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca: Optional[PCA] = None
        if use_pca:
            self.pca = PCA(n_components=pca_components)
        
        self.is_fitted = False
        self._feature_names: Optional[List[str]] = None
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "CancerPreprocessor":
        """
        Fit the preprocessor to the data.
        
        Args:
            X: Feature matrix to fit on
            feature_names: Optional list of feature names
            
        Returns:
            self for method chaining
        """
        self.scaler.fit(X)
        
        if self.use_pca and self.pca is not None:
            X_scaled = self.scaler.transform(X)
            self.pca.fit(X_scaled)
            logger.info(
                f"PCA fitted: {self.pca.n_components_} components explain "
                f"{sum(self.pca.explained_variance_ratio_):.2%} of variance"
            )
        
        self._feature_names = feature_names
        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler (and optional PCA).
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix
            
        Raises:
            RuntimeError: If preprocessor not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.use_pca and self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        return X_scaled
    
    def fit_transform(
        self, 
        X: np.ndarray, 
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix
            feature_names: Optional feature names
            
        Returns:
            Transformed feature matrix
        """
        self.fit(X, feature_names)
        return self.transform(X)
    
    def fit_transform_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit preprocessor on training data and return train/test split.
        
        This method ensures proper data leakage prevention by:
        1. Splitting data first
        2. Fitting only on training data
        3. Transforming both train and test sets
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional feature names
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Split first (before fitting scaler)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.model_config.test_size,
            random_state=self.model_config.random_state,
            stratify=y  # Maintain class balance
        )
        
        logger.info(
            f"Data split: {len(X_train)} train, {len(X_test)} test "
            f"(test_size={self.model_config.test_size})"
        )
        
        # Fit on training data only
        self.fit(X_train, feature_names)
        
        # Transform both sets
        X_train_scaled = self.transform(X_train)
        X_test_scaled = self.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features back to original scale.
        
        Note: This only works when PCA is not used.
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            Features in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        if self.use_pca:
            raise ValueError("Inverse transform not supported when PCA is enabled")
        
        return self.scaler.inverse_transform(X_scaled)
    
    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save the fitted preprocessor to disk.
        
        Args:
            path: Optional path to save to. Uses default if not provided.
            
        Returns:
            Path where preprocessor was saved
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")
        
        path = path or self.path_config.scaler_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as dictionary with all components
        artifact = {
            "scaler": self.scaler,
            "scaler_type": self.scaler_type,
            "pca": self.pca,
            "use_pca": self.use_pca,
            "pca_components": self.pca_components,
            "feature_names": self._feature_names,
        }
        
        joblib.dump(artifact, path)
        logger.info(f"Preprocessor saved to {path}")
        
        return path
    
    @classmethod
    def load(cls, path: Path) -> "CancerPreprocessor":
        """
        Load a fitted preprocessor from disk.
        
        Args:
            path: Path to saved preprocessor
            
        Returns:
            Loaded CancerPreprocessor instance
        """
        artifact = joblib.load(path)
        
        preprocessor = cls(
            scaler_type=artifact.get("scaler_type", "standard"),
            use_pca=artifact.get("use_pca", False),
            pca_components=artifact.get("pca_components"),
        )
        
        preprocessor.scaler = artifact["scaler"]
        preprocessor.pca = artifact.get("pca")
        preprocessor._feature_names = artifact.get("feature_names")
        preprocessor.is_fitted = True
        
        logger.info(f"Preprocessor loaded from {path}")
        
        return preprocessor
    
    def get_feature_importance_from_pca(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores from PCA components.
        
        Returns:
            Feature importance array or None if PCA not used
        """
        if not self.use_pca or self.pca is None:
            return None
        
        # Aggregate absolute component weights
        importance = np.abs(self.pca.components_).sum(axis=0)
        importance = importance / importance.sum()  # Normalize
        
        return importance
    
    @property
    def n_features_in(self) -> int:
        """Number of input features."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted")
        return self.scaler.n_features_in_
    
    @property
    def n_features_out(self) -> int:
        """Number of output features (after PCA if used)."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted")
        if self.use_pca and self.pca is not None:
            return self.pca.n_components_
        return self.scaler.n_features_in_
