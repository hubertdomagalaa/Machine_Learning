"""
Digits Recognition System - Model Module.

Multi-class classifier for handwritten digits (0-9).
"""

from typing import Optional, Dict, Any, List
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import joblib
from pathlib import Path
import logging
from datetime import datetime

from .config import default_config, ModelConfig, PathConfig


logger = logging.getLogger(__name__)


class DigitsClassifier:
    """
    Handwritten digit classifier with support for SVM, RF, and MLP.
    
    SVM with RBF kernel typically achieves ~98% accuracy on digits dataset.
    """
    
    SUPPORTED_MODELS = ["svm", "random_forest", "mlp"]
    
    def __init__(
        self,
        model_type: str = "svm",
        model_config: Optional[ModelConfig] = None,
        path_config: Optional[PathConfig] = None,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model_type}")
        
        self.model_type = model_type
        self.model_config = model_config or default_config.model
        self.path_config = path_config or default_config.paths
        
        self.model = self._create_model()
        self.is_fitted = False
        self.training_metrics = None
        self._training_date = None
    
    def _create_model(self):
        """Create the underlying model."""
        cfg = self.model_config
        
        if self.model_type == "svm":
            return SVC(
                kernel=cfg.svm_kernel,
                C=cfg.svm_C,
                gamma=cfg.svm_gamma,
                probability=True,
                random_state=cfg.random_state
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=cfg.rf_n_estimators,
                max_depth=cfg.rf_max_depth,
                random_state=cfg.random_state,
                n_jobs=-1
            )
        elif self.model_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=cfg.mlp_hidden_layers,
                max_iter=cfg.mlp_max_iter,
                alpha=cfg.mlp_alpha,
                random_state=cfg.random_state
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DigitsClassifier":
        """Train the model."""
        logger.info(f"Training {self.model_type} on {len(X)} samples...")
        
        self.model.fit(X, y)
        self.is_fitted = True
        self._training_date = datetime.now()
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=self.model_config.cv_folds,
            scoring="accuracy"
        )
        
        self.training_metrics = {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "n_samples": len(X),
        }
        
        logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict digit classes."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained")
        
        y_pred = self.predict(X)
        
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(
                y, y_pred,
                target_names=[str(i) for i in range(10)],
                output_dict=True
            )
        }
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save model."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save untrained model")
        
        path = path or self.path_config.model_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        artifact = {
            "model": self.model,
            "model_type": self.model_type,
            "training_metrics": self.training_metrics,
            "training_date": self._training_date.isoformat() if self._training_date else None,
        }
        
        joblib.dump(artifact, path)
        return path
    
    @classmethod
    def load(cls, path: Path) -> "DigitsClassifier":
        """Load model."""
        artifact = joblib.load(path)
        
        classifier = cls(model_type=artifact["model_type"])
        classifier.model = artifact["model"]
        classifier.is_fitted = True
        classifier.training_metrics = artifact.get("training_metrics")
        
        return classifier
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model info."""
        return {
            "model_type": self.model_type,
            "is_fitted": self.is_fitted,
            "training_metrics": self.training_metrics,
        }
