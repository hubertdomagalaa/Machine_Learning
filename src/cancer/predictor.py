"""
Cancer Detection System - Prediction Interface Module.

This module provides a high-level inference interface that combines
data loading, preprocessing, and model prediction into a single API.
"""

from typing import Optional, Dict, Any, List, Union
import numpy as np
from pathlib import Path
import logging

from .config import default_config, PathConfig
from .data_loader import CancerDataLoader
from .preprocessor import CancerPreprocessor
from .model import CancerClassifier


logger = logging.getLogger(__name__)


class CancerPredictor:
    """
    High-level prediction interface for cancer detection.
    
    Combines preprocessor and model into a single inference API,
    handling feature validation, scaling, and prediction.
    
    Attributes:
        classifier: Trained CancerClassifier instance
        preprocessor: Fitted CancerPreprocessor instance
        feature_names: List of expected feature names
        
    Example:
        >>> predictor = CancerPredictor.from_pretrained()
        >>> result = predictor.predict_single({
        ...     "mean radius": 17.99,
        ...     "mean texture": 10.38,
        ...     # ... all 30 features
        ... })
        >>> print(f"Diagnosis: {result['diagnosis']}, Confidence: {result['confidence']:.2%}")
    """
    
    def __init__(
        self,
        classifier: CancerClassifier,
        preprocessor: CancerPreprocessor,
        feature_names: Optional[List[str]] = None,
        path_config: Optional[PathConfig] = None
    ):
        """
        Initialize the predictor with trained components.
        
        Args:
            classifier: Trained CancerClassifier
            preprocessor: Fitted CancerPreprocessor
            feature_names: Optional feature names list
            path_config: Path configuration
        """
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.path_config = path_config or default_config.paths
        
        # Get feature names from data config
        data_loader = CancerDataLoader()
        self.feature_names = feature_names or data_loader.get_feature_names()
        
        # Validate components
        if not classifier.is_fitted:
            raise ValueError("Classifier must be trained before creating predictor")
        if not preprocessor.is_fitted:
            raise ValueError("Preprocessor must be fitted before creating predictor")
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions on feature matrix.
        
        Args:
            X: Feature matrix of shape (n_samples, 30)
            
        Returns:
            Dictionary with predictions, probabilities, and metadata
        """
        # Validate input shape
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {X.shape[1]}"
            )
        
        # Scale features
        X_scaled = self.preprocessor.transform(X)
        
        # Get predictions and probabilities
        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)
        
        # Format results
        results = {
            "predictions": predictions.tolist(),
            "diagnoses": [
                "Benign" if p == 1 else "Malignant" 
                for p in predictions
            ],
            "probabilities": {
                "malignant": probabilities[:, 0].tolist(),
                "benign": probabilities[:, 1].tolist(),
            },
            "confidences": np.max(probabilities, axis=1).tolist(),
            "n_samples": len(predictions),
        }
        
        return results
    
    def predict_single(
        self,
        features: Union[Dict[str, float], List[float], np.ndarray]
    ) -> Dict[str, Any]:
        """
        Make prediction for a single sample.
        
        Args:
            features: Either a dict mapping feature names to values,
                     a list of 30 feature values, or a numpy array
                     
        Returns:
            Dictionary with diagnosis, confidence, and probabilities
        """
        # Convert to numpy array
        if isinstance(features, dict):
            X = self._dict_to_array(features)
        elif isinstance(features, list):
            X = np.array(features)
        else:
            X = features
        
        X = X.reshape(1, -1)
        
        # Get prediction
        results = self.predict(X)
        
        # Format single-sample result
        return {
            "diagnosis": results["diagnoses"][0],
            "prediction": results["predictions"][0],
            "confidence": results["confidences"][0],
            "probabilities": {
                "malignant": results["probabilities"]["malignant"][0],
                "benign": results["probabilities"]["benign"][0],
            }
        }
    
    def _dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary to numpy array.
        
        Args:
            features: Dictionary mapping feature names to values
            
        Returns:
            Numpy array with features in correct order
        """
        # Validate all features present
        missing = set(self.feature_names) - set(features.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        # Create array in correct order
        return np.array([features[name] for name in self.feature_names])
    
    def predict_batch(
        self,
        samples: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple samples.
        
        Args:
            samples: List of feature dictionaries
            
        Returns:
            List of prediction results for each sample
        """
        # Convert all samples to array
        X = np.array([self._dict_to_array(s) for s in samples])
        
        # Get bulk predictions
        results = self.predict(X)
        
        # Split into individual results
        return [
            {
                "diagnosis": results["diagnoses"][i],
                "prediction": results["predictions"][i],
                "confidence": results["confidences"][i],
                "probabilities": {
                    "malignant": results["probabilities"]["malignant"][i],
                    "benign": results["probabilities"]["benign"][i],
                }
            }
            for i in range(len(samples))
        ]
    
    def get_feature_names(self) -> List[str]:
        """Return list of expected feature names."""
        return self.feature_names.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the underlying model."""
        return self.classifier.get_model_info()
    
    def save(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None) -> Dict[str, Path]:
        """
        Save predictor components to disk.
        
        Args:
            model_path: Optional path for model
            scaler_path: Optional path for preprocessor
            
        Returns:
            Dictionary with paths where components were saved
        """
        model_path = self.classifier.save(model_path)
        scaler_path = self.preprocessor.save(scaler_path)
        
        return {
            "model": model_path,
            "preprocessor": scaler_path,
        }
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        path_config: Optional[PathConfig] = None
    ) -> "CancerPredictor":
        """
        Load a pretrained predictor from saved artifacts.
        
        Args:
            model_path: Optional path to saved model
            scaler_path: Optional path to saved preprocessor
            path_config: Path configuration
            
        Returns:
            Loaded CancerPredictor instance
        """
        config = path_config or default_config.paths
        
        model_path = Path(model_path or config.model_path)
        scaler_path = Path(scaler_path or config.scaler_path)
        
        # Verify files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {scaler_path}")
        
        # Load components
        classifier = CancerClassifier.load(model_path)
        preprocessor = CancerPreprocessor.load(scaler_path)
        
        logger.info("Loaded pretrained predictor")
        
        return cls(classifier=classifier, preprocessor=preprocessor, path_config=config)
    
    @classmethod
    def train_new(
        cls,
        model_type: str = "random_forest",
        save: bool = True
    ) -> "CancerPredictor":
        """
        Train a new predictor from scratch using sklearn's dataset.
        
        Args:
            model_type: Type of classifier to train
            save: Whether to save the trained components
            
        Returns:
            Trained CancerPredictor instance
        """
        logger.info(f"Training new {model_type} predictor...")
        
        # Load data
        loader = CancerDataLoader()
        X, y = loader.load_data()
        feature_names = loader.get_feature_names()
        
        # Preprocess
        preprocessor = CancerPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.fit_transform_split(
            X, y, feature_names
        )
        
        # Train model
        classifier = CancerClassifier(model_type=model_type)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)
        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Create predictor
        predictor = cls(
            classifier=classifier,
            preprocessor=preprocessor,
            feature_names=feature_names
        )
        
        # Save if requested
        if save:
            predictor.save()
            logger.info("Predictor saved to disk")
        
        return predictor
