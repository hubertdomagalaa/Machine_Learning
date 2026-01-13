"""
Fraud Detection System - Prediction Interface.

High-level interface for fraud prediction combining all components.
"""

from typing import Optional, Dict, Any, List, Union
import numpy as np
from pathlib import Path
import logging

from .config import default_config, PathConfig
from .data_loader import FraudDataLoader
from .preprocessor import FraudPreprocessor
from .model import FraudDetector


logger = logging.getLogger(__name__)


class FraudPredictor:
    """
    High-level prediction interface for fraud detection.
    
    Example:
        >>> predictor = FraudPredictor.from_pretrained()
        >>> result = predictor.predict_transaction({
        ...     "amount": 50000,
        ...     "oldbalanceOrg": 60000,
        ...     "type": "TRANSFER",
        ...     ...
        ... })
        >>> print(f"Fraud: {result['is_fraud']}, Risk: {result['fraud_probability']:.2%}")
    """
    
    def __init__(
        self,
        detector: FraudDetector,
        preprocessor: FraudPreprocessor,
        feature_names: Optional[List[str]] = None,
        path_config: Optional[PathConfig] = None
    ):
        """Initialize predictor with trained components."""
        self.detector = detector
        self.preprocessor = preprocessor
        self.path_config = path_config or default_config.paths
        self.feature_names = feature_names or []
        
        if not detector.is_fitted:
            raise ValueError("Detector must be trained")
        if not preprocessor.is_fitted:
            raise ValueError("Preprocessor must be fitted")
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Make fraud predictions on feature matrix.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        X_scaled = self.preprocessor.transform(X)
        
        # Get predictions
        predictions = self.detector.predict(X_scaled)
        probabilities = self.detector.predict_proba(X_scaled)
        
        return {
            "predictions": predictions.tolist(),
            "is_fraud": [bool(p) for p in predictions],
            "fraud_probabilities": probabilities[:, 1].tolist(),
            "risk_levels": [
                self._get_risk_level(p) for p in probabilities[:, 1]
            ],
            "threshold": self.detector.threshold,
            "n_samples": len(predictions),
        }
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level."""
        if probability >= 0.8:
            return "CRITICAL"
        elif probability >= 0.6:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        elif probability >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def predict_single(
        self,
        features: Union[Dict[str, float], List[float], np.ndarray]
    ) -> Dict[str, Any]:
        """
        Make prediction for single transaction.
        
        Args:
            features: Feature dict, list, or array
            
        Returns:
            Single prediction result
        """
        if isinstance(features, dict):
            X = np.array([features.get(name, 0) for name in self.feature_names])
        elif isinstance(features, list):
            X = np.array(features)
        else:
            X = features
        
        X = X.reshape(1, -1)
        results = self.predict(X)
        
        return {
            "is_fraud": results["is_fraud"][0],
            "fraud_probability": results["fraud_probabilities"][0],
            "risk_level": results["risk_levels"][0],
            "threshold": results["threshold"],
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.detector.get_model_info()
    
    def save(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None
    ) -> Dict[str, Path]:
        """Save predictor components."""
        model_path = self.detector.save(model_path)
        scaler_path = self.preprocessor.save(scaler_path)
        
        return {"model": model_path, "preprocessor": scaler_path}
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        path_config: Optional[PathConfig] = None
    ) -> "FraudPredictor":
        """Load pretrained predictor."""
        config = path_config or default_config.paths
        
        model_path = Path(model_path or config.model_path)
        scaler_path = Path(scaler_path or config.scaler_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {scaler_path}")
        
        detector = FraudDetector.load(model_path)
        preprocessor = FraudPreprocessor.load(scaler_path)
        
        return cls(detector=detector, preprocessor=preprocessor, path_config=config)
    
    @classmethod
    def train_new(
        cls,
        model_type: str = "xgboost",
        resampling_strategy: str = "smote",
        sample_size: Optional[int] = None,
        save: bool = True
    ) -> "FraudPredictor":
        """
        Train new fraud predictor.
        
        Args:
            model_type: Type of model
            resampling_strategy: Strategy for class imbalance
            sample_size: Optional sample size
            save: Whether to save trained components
            
        Returns:
            Trained FraudPredictor
        """
        logger.info(f"Training new {model_type} fraud predictor...")
        
        # Load data
        loader = FraudDataLoader()
        X, y = loader.load_data(sample_size=sample_size)
        
        # Get feature names from a sample
        df_X, _ = loader.load_data(sample_size=100, return_dataframe=True)
        feature_names = list(df_X.columns)
        
        # Preprocess with resampling
        preprocessor = FraudPreprocessor(resampling_strategy=resampling_strategy)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform_resample(
            X, y, feature_names
        )
        
        # Train model
        detector = FraudDetector(model_type=model_type)
        detector.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        
        # Evaluate
        metrics = detector.evaluate(X_test, y_test)
        logger.info(f"Test Recall: {metrics['recall']:.4f}")
        logger.info(f"Test AUC: {metrics['roc_auc']:.4f}")
        
        # Create predictor
        predictor = cls(
            detector=detector,
            preprocessor=preprocessor,
            feature_names=feature_names
        )
        
        if save:
            predictor.save()
            logger.info("Predictor saved")
        
        return predictor
