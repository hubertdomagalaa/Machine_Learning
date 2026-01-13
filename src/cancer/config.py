"""
Cancer Detection System Configuration Module.

This module provides centralized configuration management for the cancer
detection system, including model parameters, paths, and hyperparameters.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List
import yaml
import os


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_random_state: int = 42
    
    # SVM parameters
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    svm_gamma: str = "scale"
    
    # Logistic Regression parameters
    lr_C: float = 1.0
    lr_max_iter: int = 1000
    lr_solver: str = "lbfgs"
    
    # KNN parameters
    knn_n_neighbors: int = 5
    knn_weights: str = "uniform"
    knn_metric: str = "minkowski"
    
    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "random_forest": {
                "n_estimators": self.rf_n_estimators,
                "max_depth": self.rf_max_depth,
                "min_samples_split": self.rf_min_samples_split,
                "min_samples_leaf": self.rf_min_samples_leaf,
                "random_state": self.rf_random_state,
            },
            "svm": {
                "kernel": self.svm_kernel,
                "C": self.svm_C,
                "gamma": self.svm_gamma,
            },
            "logistic_regression": {
                "C": self.lr_C,
                "max_iter": self.lr_max_iter,
                "solver": self.lr_solver,
            },
            "knn": {
                "n_neighbors": self.knn_n_neighbors,
                "weights": self.knn_weights,
                "metric": self.knn_metric,
            },
            "training": {
                "test_size": self.test_size,
                "random_state": self.random_state,
                "cv_folds": self.cv_folds,
            },
        }


@dataclass
class DataConfig:
    """Configuration for data handling."""
    
    # Feature names from Wisconsin Breast Cancer Dataset
    feature_names: List[str] = field(default_factory=lambda: [
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity",
        "mean concave points", "mean symmetry", "mean fractal dimension",
        "radius error", "texture error", "perimeter error", "area error",
        "smoothness error", "compactness error", "concavity error",
        "concave points error", "symmetry error", "fractal dimension error",
        "worst radius", "worst texture", "worst perimeter", "worst area",
        "worst smoothness", "worst compactness", "worst concavity",
        "worst concave points", "worst symmetry", "worst fractal dimension"
    ])
    
    # Target names
    target_names: List[str] = field(default_factory=lambda: ["malignant", "benign"])
    
    # Number of features
    n_features: int = 30


@dataclass 
class PathConfig:
    """Configuration for file paths."""
    
    model_dir: Path = field(default_factory=lambda: MODELS_DIR)
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    
    # Model artifact paths
    model_path: Path = field(default_factory=lambda: MODELS_DIR / "cancer_classifier.pkl")
    scaler_path: Path = field(default_factory=lambda: MODELS_DIR / "cancer_scaler.pkl")
    metadata_path: Path = field(default_factory=lambda: MODELS_DIR / "cancer_metadata.yaml")
    
    def ensure_dirs(self) -> None:
        """Create directories if they don't exist."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration class combining all config sections."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.paths = PathConfig()
    
    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "model": self.model.to_dict(),
            "data": {
                "feature_names": self.data.feature_names,
                "target_names": self.data.target_names,
                "n_features": self.data.n_features,
            },
            "paths": {
                "model_dir": str(self.paths.model_dir),
                "data_dir": str(self.paths.data_dir),
            }
        }
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        # Load model config
        if "model" in config_dict:
            model_cfg = config_dict["model"]
            if "random_forest" in model_cfg:
                config.model.rf_n_estimators = model_cfg["random_forest"].get("n_estimators", 100)
                config.model.rf_max_depth = model_cfg["random_forest"].get("max_depth", 10)
            if "training" in model_cfg:
                config.model.test_size = model_cfg["training"].get("test_size", 0.2)
                config.model.random_state = model_cfg["training"].get("random_state", 42)
        
        return config


# Global default configuration
default_config = Config()
