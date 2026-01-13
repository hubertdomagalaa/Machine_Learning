"""
Digits Recognition System - Prediction Interface.
"""

from typing import Optional, Dict, Any, List, Union
import numpy as np
from pathlib import Path
import logging

from .config import default_config, PathConfig
from .data_loader import DigitsDataLoader
from .preprocessor import DigitsPreprocessor
from .model import DigitsClassifier


logger = logging.getLogger(__name__)


class DigitsPredictor:
    """High-level prediction interface for digit recognition."""
    
    def __init__(
        self,
        classifier: DigitsClassifier,
        preprocessor: DigitsPreprocessor,
        path_config: Optional[PathConfig] = None
    ):
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.path_config = path_config or default_config.paths
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict digits."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.preprocessor.transform(X)
        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)
        
        return {
            "predictions": predictions.tolist(),
            "digits": [str(d) for d in predictions],
            "confidences": np.max(probabilities, axis=1).tolist(),
            "probabilities": probabilities.tolist(),
            "n_samples": len(predictions),
        }
    
    def predict_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict from 8x8 image.
        
        Args:
            image: 8x8 grayscale image (values 0-16)
        """
        if image.shape != (8, 8):
            raise ValueError(f"Expected 8x8 image, got {image.shape}")
        
        X = image.flatten().reshape(1, -1)
        return self.predict(X)
    
    def save(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        """Save predictor components."""
        return {
            "model": self.classifier.save(model_path),
            "preprocessor": self.preprocessor.save(scaler_path),
        }
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None
    ) -> "DigitsPredictor":
        """Load pretrained predictor."""
        config = default_config.paths
        
        model_path = Path(model_path or config.model_path)
        scaler_path = Path(scaler_path or config.scaler_path)
        
        classifier = DigitsClassifier.load(model_path)
        preprocessor = DigitsPreprocessor.load(scaler_path)
        
        return cls(classifier=classifier, preprocessor=preprocessor)
    
    @classmethod
    def train_new(cls, model_type: str = "svm", save: bool = True) -> "DigitsPredictor":
        """Train new predictor."""
        logger.info(f"Training new {model_type} digits predictor...")
        
        # Load data
        loader = DigitsDataLoader()
        X, y = loader.load_data()
        
        # Preprocess
        preprocessor = DigitsPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.fit_transform_split(X, y)
        
        # Train
        classifier = DigitsClassifier(model_type=model_type)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)
        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
        
        predictor = cls(classifier=classifier, preprocessor=preprocessor)
        
        if save:
            predictor.save()
        
        return predictor
