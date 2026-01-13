"""
Cancer Detection System - Model Training Module.

This module provides the main classification model with support for
multiple algorithms, cross-validation, and comprehensive evaluation.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
from pathlib import Path
import logging
from datetime import datetime

from .config import default_config, ModelConfig, PathConfig


logger = logging.getLogger(__name__)


class CancerClassifier:
    """
    Cancer classification model with multiple algorithm support.
    
    Supports Random Forest, SVM, Logistic Regression, KNN, and
    ensemble voting classifiers. Includes cross-validation,
    hyperparameter tuning, and comprehensive evaluation metrics.
    
    Attributes:
        model: Trained sklearn classifier
        model_type: Type of model being used
        is_fitted: Whether model has been trained
        training_metrics: Metrics from model evaluation
        
    Example:
        >>> classifier = CancerClassifier(model_type="random_forest")
        >>> classifier.fit(X_train, y_train)
        >>> predictions = classifier.predict(X_test)
        >>> metrics = classifier.evaluate(X_test, y_test)
    """
    
    SUPPORTED_MODELS = ["random_forest", "svm", "logistic_regression", "knn", "ensemble"]
    
    def __init__(
        self,
        model_type: str = "random_forest",
        model_config: Optional[ModelConfig] = None,
        path_config: Optional[PathConfig] = None,
    ):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of model - one of SUPPORTED_MODELS
            model_config: Model configuration with hyperparameters
            path_config: Path configuration for saving models
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )
        
        self.model_type = model_type
        self.model_config = model_config or default_config.model
        self.path_config = path_config or default_config.paths
        
        self.model = self._create_model()
        self.is_fitted = False
        self.training_metrics: Optional[Dict[str, Any]] = None
        self._training_date: Optional[datetime] = None
        self._feature_importances: Optional[np.ndarray] = None
    
    def _create_model(self) -> Any:
        """Create the underlying sklearn model."""
        cfg = self.model_config
        
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=cfg.rf_n_estimators,
                max_depth=cfg.rf_max_depth,
                min_samples_split=cfg.rf_min_samples_split,
                min_samples_leaf=cfg.rf_min_samples_leaf,
                random_state=cfg.rf_random_state,
                n_jobs=-1
            )
        
        elif self.model_type == "svm":
            return SVC(
                kernel=cfg.svm_kernel,
                C=cfg.svm_C,
                gamma=cfg.svm_gamma,
                probability=True,  # Enable predict_proba
                random_state=cfg.random_state
            )
        
        elif self.model_type == "logistic_regression":
            return LogisticRegression(
                C=cfg.lr_C,
                max_iter=cfg.lr_max_iter,
                solver=cfg.lr_solver,
                random_state=cfg.random_state
            )
        
        elif self.model_type == "knn":
            return KNeighborsClassifier(
                n_neighbors=cfg.knn_n_neighbors,
                weights=cfg.knn_weights,
                metric=cfg.knn_metric
            )
        
        elif self.model_type == "ensemble":
            return VotingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(
                        n_estimators=cfg.rf_n_estimators,
                        max_depth=cfg.rf_max_depth,
                        random_state=cfg.rf_random_state,
                        n_jobs=-1
                    )),
                    ("svm", SVC(
                        kernel=cfg.svm_kernel,
                        C=cfg.svm_C,
                        probability=True,
                        random_state=cfg.random_state
                    )),
                    ("lr", LogisticRegression(
                        C=cfg.lr_C,
                        max_iter=cfg.lr_max_iter,
                        random_state=cfg.random_state
                    ))
                ],
                voting="soft"
            )
        
        raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validate: bool = True
    ) -> "CancerClassifier":
        """
        Train the model on provided data.
        
        Args:
            X: Training features (scaled)
            y: Training targets
            validate: Whether to perform cross-validation during training
            
        Returns:
            self for method chaining
        """
        logger.info(f"Training {self.model_type} model on {len(X)} samples...")
        
        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True
        self._training_date = datetime.now()
        
        # Store feature importances if available
        if hasattr(self.model, "feature_importances_"):
            self._feature_importances = self.model.feature_importances_
        
        # Perform cross-validation if requested
        if validate:
            cv_scores = cross_val_score(
                self.model, X, y,
                cv=self.model_config.cv_folds,
                scoring="accuracy"
            )
            logger.info(
                f"Cross-validation accuracy: {cv_scores.mean():.4f} "
                f"(+/- {cv_scores.std() * 2:.4f})"
            )
            
            self.training_metrics = {
                "cv_accuracy_mean": float(cv_scores.mean()),
                "cv_accuracy_std": float(cv_scores.std()),
                "cv_scores": cv_scores.tolist(),
                "n_samples": len(X),
            }
        
        logger.info(f"Model training completed at {self._training_date}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict (must be scaled)
            
        Returns:
            Array of predicted classes (0=malignant, 1=benign)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict (must be scaled)
            
        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
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
        Evaluate model performance on test data.
        
        Args:
            X: Test features (scaled)
            y: True labels
            detailed: Whether to include detailed metrics
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_proba)),
        }
        
        if detailed:
            cm = confusion_matrix(y, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            metrics["classification_report"] = classification_report(
                y, y_pred,
                target_names=["Malignant", "Benign"],
                output_dict=True
            )
            
            # Calculate false negative rate (critical for cancer detection)
            tn, fp, fn, tp = cm.ravel()
            metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics["n_test_samples"] = len(y)
        
        logger.info(f"Evaluation: Accuracy={metrics['accuracy']:.4f}, AUC={metrics['roc_auc']:.4f}")
        
        return metrics
    
    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Optional[Dict[str, List]] = None
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X: Training features
            y: Training targets
            param_grid: Parameter grid for search. Uses defaults if not provided.
            
        Returns:
            Dictionary with best parameters and scores
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        logger.info(f"Starting hyperparameter tuning for {self.model_type}...")
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=self.model_config.cv_folds,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best estimator
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        self._training_date = datetime.now()
        
        results = {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "cv_results": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in grid_search.cv_results_.items()
                if k in ["mean_test_score", "std_test_score", "params"]
            }
        }
        
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best score: {results['best_score']:.4f}")
        
        return results
    
    def _get_default_param_grid(self) -> Dict[str, List]:
        """Get default parameter grid for hyperparameter tuning."""
        if self.model_type == "random_forest":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            }
        elif self.model_type == "svm":
            return {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            }
        elif self.model_type == "logistic_regression":
            return {
                "C": [0.1, 1, 10],
                "solver": ["lbfgs", "newton-cg"],
            }
        elif self.model_type == "knn":
            return {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
            }
        return {}
    
    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None
    ) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self._feature_importances is None:
            return None
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self._feature_importances))]
        
        # Sort by importance
        sorted_idx = np.argsort(self._feature_importances)[::-1]
        
        return {
            feature_names[i]: float(self._feature_importances[i])
            for i in sorted_idx
        }
    
    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save the trained model to disk.
        
        Args:
            path: Optional path to save to. Uses default if not provided.
            
        Returns:
            Path where model was saved
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save untrained model")
        
        path = path or self.path_config.model_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model with metadata
        artifact = {
            "model": self.model,
            "model_type": self.model_type,
            "training_metrics": self.training_metrics,
            "training_date": self._training_date.isoformat() if self._training_date else None,
            "feature_importances": self._feature_importances,
            "version": "1.0.0",
        }
        
        joblib.dump(artifact, path)
        logger.info(f"Model saved to {path}")
        
        return path
    
    @classmethod
    def load(cls, path: Path) -> "CancerClassifier":
        """
        Load a trained model from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded CancerClassifier instance
        """
        artifact = joblib.load(path)
        
        classifier = cls(model_type=artifact.get("model_type", "random_forest"))
        classifier.model = artifact["model"]
        classifier.is_fitted = True
        classifier.training_metrics = artifact.get("training_metrics")
        classifier._feature_importances = artifact.get("feature_importances")
        
        if artifact.get("training_date"):
            classifier._training_date = datetime.fromisoformat(artifact["training_date"])
        
        logger.info(f"Model loaded from {path}")
        
        return classifier
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_type": self.model_type,
            "is_fitted": self.is_fitted,
            "training_date": self._training_date.isoformat() if self._training_date else None,
            "training_metrics": self.training_metrics,
            "version": "1.0.0",
        }
