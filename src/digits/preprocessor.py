"""
Digits Recognition System - Preprocessing Module.
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import logging

from .config import default_config, ModelConfig, PathConfig


logger = logging.getLogger(__name__)


class DigitsPreprocessor:
    """Preprocessor for digit images with optional PCA."""
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        path_config: Optional[PathConfig] = None,
        use_pca: bool = False,
        pca_components: int = 40
    ):
        self.model_config = model_config or default_config.model
        self.path_config = path_config or default_config.paths
        
        self.scaler = StandardScaler()
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = PCA(n_components=pca_components) if use_pca else None
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> "DigitsPreprocessor":
        """Fit the preprocessor."""
        self.scaler.fit(X)
        
        if self.use_pca:
            X_scaled = self.scaler.transform(X)
            self.pca.fit(X_scaled)
            explained = sum(self.pca.explained_variance_ratio_)
            logger.info(f"PCA: {self.pca_components} components explain {explained:.2%} variance")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted")
        
        X_scaled = self.scaler.transform(X)
        
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        
        return X_scaled
    
    def fit_transform_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data and fit/transform."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.model_config.test_size,
            random_state=self.model_config.random_state,
            stratify=y
        )
        
        self.fit(X_train)
        X_train_scaled = self.transform(X_train)
        X_test_scaled = self.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save preprocessor."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")
        
        path = path or self.path_config.scaler_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        artifact = {
            "scaler": self.scaler,
            "pca": self.pca,
            "use_pca": self.use_pca,
            "pca_components": self.pca_components,
        }
        
        joblib.dump(artifact, path)
        return path
    
    @classmethod
    def load(cls, path: Path) -> "DigitsPreprocessor":
        """Load preprocessor."""
        artifact = joblib.load(path)
        
        preprocessor = cls(
            use_pca=artifact["use_pca"],
            pca_components=artifact["pca_components"]
        )
        preprocessor.scaler = artifact["scaler"]
        preprocessor.pca = artifact["pca"]
        preprocessor.is_fitted = True
        
        return preprocessor
