"""
Digits Recognition System - Data Loading Module.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from sklearn.datasets import load_digits
import logging

from .config import default_config, DataConfig


logger = logging.getLogger(__name__)


class DigitsDataLoader:
    """Loader for handwritten digits dataset."""
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or default_config.data
        self._data_cache = None
    
    def load_data(
        self,
        return_images: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load digits dataset.
        
        Args:
            return_images: If True, return 8x8 images instead of flat vectors
        """
        if self._data_cache is None:
            self._data_cache = load_digits()
        
        data = self._data_cache
        
        if return_images:
            X = data.images  # Shape: (n_samples, 8, 8)
        else:
            X = data.data  # Shape: (n_samples, 64)
        
        y = data.target
        
        logger.info(f"Loaded {len(X)} digit images")
        
        return X, y
    
    def get_sample_images(self, n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Get sample images for visualization."""
        X, y = self.load_data(return_images=True)
        indices = np.random.choice(len(X), n, replace=False)
        return X[indices], y[indices]
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        X, y = self.load_data()
        
        return {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "image_size": (8, 8),
            "class_distribution": {
                str(i): int(np.sum(y == i)) for i in range(10)
            },
        }
