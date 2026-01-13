"""
Fraud Detection System - Model Module.

Implements fraud detection models optimized for severely imbalanced
datasets with focus on minimizing false negatives (missed fraud).
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import joblib
from pathlib import Path
import logging
from datetime import datetime

from .config import default_config, ModelConfig, PathConfig


logger = logging.getLogger(__name__)


# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed. Using GradientBoosting as fallback.")


class FraudDetector:
    """
    Fraud detection model optimized for imbalanced data.
    
    Uses XGBoost (or GradientBoosting fallback) with scale_pos_weight
    to handle class imbalance. Includes threshold optimization for
    precision-recall trade-off.
    
    Attributes:
        model: Trained classifier
        threshold: Decision threshold (default 0.5, can be optimized)
        is_fitted: Whether model has been trained
    """
    
    SUPPORTED_MODELS = ["xgboost", "random_forest", "logistic_regression", "gradient_boosting"]
    
    def __init__(
        self,
        model_type: str = "xgboost",
        model_config: Optional[ModelConfig] = None,
        path_config: Optional[PathConfig] = None,
    ):
        """
        Initialize the fraud detector.
        
        Args:
            model_type: Type of model to use
            model_config: Model configuration
            path_config: Path configuration
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model_type}. Supported: {self.SUPPORTED_MODELS}")
        
        # Fall back to gradient_boosting if XGBoost not available
        if model_type == "xgboost" and not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, using gradient_boosting")
            model_type = "gradient_boosting"
        
        self.model_type = model_type
        self.model_config = model_config or default_config.model
        self.path_config = path_config or default_config.paths
        
        self.model = self._create_model()
        self.threshold = self.model_config.default_threshold
        self.is_fitted = False
        self.training_metrics: Optional[Dict[str, Any]] = None
        self._training_date: Optional[datetime] = None
        self._optimal_threshold: Optional[float] = None
    
    def _create_model(self) -> Any:
        """Create the underlying model."""
        cfg = self.model_config
        
        if self.model_type == "xgboost":
            return XGBClassifier(
                n_estimators=cfg.xgb_n_estimators,
                max_depth=cfg.xgb_max_depth,
                learning_rate=cfg.xgb_learning_rate,
                subsample=cfg.xgb_subsample,
                colsample_bytree=cfg.xgb_colsample_bytree,
                scale_pos_weight=cfg.xgb_scale_pos_weight,
                random_state=cfg.random_state,
                eval_metric='auc',
                use_label_encoder=False
            )
        
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=cfg.xgb_n_estimators,
                max_depth=cfg.xgb_max_depth,
                learning_rate=cfg.xgb_learning_rate,
                subsample=cfg.xgb_subsample,
                random_state=cfg.random_state
            )
        
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=cfg.rf_n_estimators,
                max_depth=cfg.rf_max_depth,
                class_weight=cfg.rf_class_weight,
                random_state=cfg.random_state,
                n_jobs=-1
            )
        
        elif self.model_type == "logistic_regression":
            return LogisticRegression(
                C=cfg.lr_C,
                max_iter=cfg.lr_max_iter,
                class_weight=cfg.lr_class_weight,
                random_state=cfg.random_state
            )
        
        raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimize_threshold: bool = True,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "FraudDetector":
        """
        Train the fraud detection model.
        
        Args:
            X: Training features
            y: Training targets
            optimize_threshold: Whether to find optimal decision threshold
            X_val: Optional validation features for threshold optimization
            y_val: Optional validation targets
            
        Returns:
            self for method chaining
        """
        logger.info(f"Training {self.model_type} fraud detector on {len(X)} samples...")
        logger.info(f"Training fraud rate: {y.mean():.4%}")
        
        self.model.fit(X, y)
        self.is_fitted = True
        self._training_date = datetime.now()
        
        # Optimize threshold using validation data or training data
        if optimize_threshold:
            if X_val is not None and y_val is not None:
                self._optimize_threshold(X_val, y_val)
            else:
                self._optimize_threshold(X, y)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=min(self.model_config.cv_folds, sum(y == 1)),
            scoring="roc_auc"
        )
        
        self.training_metrics = {
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "n_samples": len(X),
            "fraud_rate": float(y.mean()),
            "optimal_threshold": self.threshold,
        }
        
        logger.info(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        logger.info(f"Optimal threshold: {self.threshold:.4f}")
        
        return self
    
    def _optimize_threshold(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Find optimal decision threshold using precision-recall curve.
        
        Optimizes for F1 score or a custom metric that balances
        fraud detection (recall) with false alarm rate (precision).
        """
        y_proba = self.model.predict_proba(X)[:, 1]
        
        # Get precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
        
        # Calculate F1 for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        # Find threshold with best F1
        best_idx = np.argmax(f1_scores)
        
        # Handle edge case where best_idx is at the end
        if best_idx < len(thresholds):
            self.threshold = float(thresholds[best_idx])
        else:
            self.threshold = 0.5
        
        self._optimal_threshold = self.threshold
        
        logger.info(
            f"Optimized threshold: {self.threshold:.4f} "
            f"(F1={f1_scores[best_idx]:.4f}, Recall={recalls[best_idx]:.4f})"
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using optimized threshold."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        y_proba = self.model.predict_proba(X)[:, 1]
        return (y_proba >= self.threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model with fraud-specific metrics.
        
        In fraud detection:
        - High RECALL is critical (catch as many frauds as possible)
        - Precision determines false alarm rate
        - False negatives = missed fraud = LOST MONEY
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_proba)),
            "average_precision": float(average_precision_score(y, y_proba)),
            "threshold": self.threshold,
        }
        
        if detailed:
            metrics["confusion_matrix"] = cm.tolist()
            metrics["true_positives"] = int(tp)
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            
            # Fraud-specific metrics
            metrics["fraud_detection_rate"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            metrics["false_alarm_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics["missed_fraud_rate"] = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
            
            # Cost-based metric (assuming fraud costs 10x more than false alarm)
            fraud_cost = fn * 10  # Missed fraud
            false_alarm_cost = fp * 1  # False alarm
            metrics["total_cost_score"] = fraud_cost + false_alarm_cost
            
            metrics["classification_report"] = classification_report(
                y, y_pred,
                target_names=["Legitimate", "Fraud"],
                output_dict=True
            )
        
        logger.info(
            f"Evaluation: Recall={metrics['recall']:.4f}, "
            f"Precision={metrics['precision']:.4f}, AUC={metrics['roc_auc']:.4f}"
        )
        
        return metrics
    
    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None
    ) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        if not hasattr(self.model, "feature_importances_"):
            return None
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        sorted_idx = np.argsort(importances)[::-1]
        
        return {
            feature_names[i]: float(importances[i])
            for i in sorted_idx
        }
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save trained model."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save untrained model")
        
        path = path or self.path_config.model_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        artifact = {
            "model": self.model,
            "model_type": self.model_type,
            "threshold": self.threshold,
            "training_metrics": self.training_metrics,
            "training_date": self._training_date.isoformat() if self._training_date else None,
            "version": "1.0.0",
        }
        
        joblib.dump(artifact, path)
        logger.info(f"Model saved to {path}")
        
        return path
    
    @classmethod
    def load(cls, path: Path) -> "FraudDetector":
        """Load trained model."""
        artifact = joblib.load(path)
        
        detector = cls(model_type=artifact.get("model_type", "xgboost"))
        detector.model = artifact["model"]
        detector.threshold = artifact.get("threshold", 0.5)
        detector.is_fitted = True
        detector.training_metrics = artifact.get("training_metrics")
        
        if artifact.get("training_date"):
            detector._training_date = datetime.fromisoformat(artifact["training_date"])
        
        logger.info(f"Model loaded from {path}")
        
        return detector
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": self.model_type,
            "is_fitted": self.is_fitted,
            "threshold": self.threshold,
            "training_date": self._training_date.isoformat() if self._training_date else None,
            "training_metrics": self.training_metrics,
            "version": "1.0.0",
        }
