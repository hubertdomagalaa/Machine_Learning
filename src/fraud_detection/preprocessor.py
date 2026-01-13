"""
Fraud Detection System - Preprocessing Module.

Handles feature scaling and class imbalance through SMOTE
and other resampling techniques.
"""

from typing import Tuple, Optional, List
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import logging

from .config import default_config, ModelConfig, PathConfig


logger = logging.getLogger(__name__)


# Try to import imblearn, fall back gracefully if not available
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger.warning("imbalanced-learn not installed. SMOTE will be unavailable.")


class FraudPreprocessor:
    """
    Preprocessor for fraud detection features with class imbalance handling.
    
    Handles feature scaling and provides multiple strategies for
    dealing with severely imbalanced fraud data.
    
    Attributes:
        scaler: Fitted RobustScaler (better for outliers in fraud data)
        resampling_strategy: Strategy for handling class imbalance
        is_fitted: Whether the preprocessor has been fitted
    """
    
    RESAMPLING_STRATEGIES = ["smote", "adasyn", "undersample", "smote_tomek", "none"]
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        path_config: Optional[PathConfig] = None,
        resampling_strategy: str = "smote",
        scaler_type: str = "robust"
    ):
        """
        Initialize the preprocessor.
        
        Args:
            model_config: Model configuration
            path_config: Path configuration
            resampling_strategy: Strategy for handling class imbalance
            scaler_type: Type of scaler ("robust" or "standard")
        """
        self.model_config = model_config or default_config.model
        self.path_config = path_config or default_config.paths
        
        if resampling_strategy not in self.RESAMPLING_STRATEGIES:
            raise ValueError(f"Unknown strategy: {resampling_strategy}")
        
        self.resampling_strategy = resampling_strategy
        self.scaler_type = scaler_type
        
        # RobustScaler is better for fraud data (handles outliers)
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        self.is_fitted = False
        self._feature_names: Optional[List[str]] = None
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "FraudPreprocessor":
        """
        Fit the preprocessor to training data.
        
        Args:
            X: Feature matrix
            feature_names: Optional feature names
            
        Returns:
            self for method chaining
        """
        self.scaler.fit(X)
        self._feature_names = feature_names
        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled features
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, feature_names)
        return self.transform(X)
    
    def fit_transform_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit, split, and resample training data.
        
        This method:
        1. Splits data first (before any resampling)
        2. Fits scaler on training data only
        3. Applies resampling to training data only (not test!)
        4. Transforms both train and test sets
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional feature names
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            Note: X_train and y_train may be resampled
        """
        # Split FIRST (avoid data leakage!)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.model_config.test_size,
            random_state=self.model_config.random_state,
            stratify=y
        )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")
        logger.info(f"Original train fraud rate: {y_train.mean():.4%}")
        
        # Fit scaler on training data
        self.fit(X_train, feature_names)
        
        # Scale training data
        X_train_scaled = self.transform(X_train)
        
        # Resample training data (AFTER scaling, only on training)
        if self.resampling_strategy != "none":
            X_train_resampled, y_train_resampled = self._resample(X_train_scaled, y_train)
            logger.info(
                f"Resampled: {len(X_train_scaled)} -> {len(X_train_resampled)} samples, "
                f"new fraud rate: {y_train_resampled.mean():.4%}"
            )
        else:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train
        
        # Scale test data (no resampling on test!)
        X_test_scaled = self.transform(X_test)
        
        return X_train_resampled, X_test_scaled, y_train_resampled, y_test
    
    def _resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply resampling strategy to handle class imbalance."""
        if not IMBLEARN_AVAILABLE:
            logger.warning("imbalanced-learn not available, skipping resampling")
            return X, y
        
        cfg = self.model_config
        
        if self.resampling_strategy == "smote":
            resampler = SMOTE(
                sampling_strategy=cfg.smote_sampling_strategy,
                k_neighbors=min(cfg.smote_k_neighbors, sum(y == 1) - 1),
                random_state=cfg.random_state
            )
        elif self.resampling_strategy == "adasyn":
            resampler = ADASYN(
                sampling_strategy=cfg.smote_sampling_strategy,
                random_state=cfg.random_state
            )
        elif self.resampling_strategy == "undersample":
            resampler = RandomUnderSampler(
                sampling_strategy=cfg.smote_sampling_strategy,
                random_state=cfg.random_state
            )
        elif self.resampling_strategy == "smote_tomek":
            resampler = SMOTETomek(random_state=cfg.random_state)
        else:
            return X, y
        
        try:
            X_resampled, y_resampled = resampler.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            logger.warning(f"Resampling failed: {e}, using original data")
            return X, y
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save the fitted preprocessor."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")
        
        path = path or self.path_config.scaler_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        artifact = {
            "scaler": self.scaler,
            "scaler_type": self.scaler_type,
            "resampling_strategy": self.resampling_strategy,
            "feature_names": self._feature_names,
        }
        
        joblib.dump(artifact, path)
        logger.info(f"Preprocessor saved to {path}")
        
        return path
    
    @classmethod
    def load(cls, path: Path) -> "FraudPreprocessor":
        """Load a fitted preprocessor."""
        artifact = joblib.load(path)
        
        preprocessor = cls(
            scaler_type=artifact.get("scaler_type", "robust"),
            resampling_strategy=artifact.get("resampling_strategy", "smote"),
        )
        
        preprocessor.scaler = artifact["scaler"]
        preprocessor._feature_names = artifact.get("feature_names")
        preprocessor.is_fitted = True
        
        logger.info(f"Preprocessor loaded from {path}")
        
        return preprocessor
