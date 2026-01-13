"""
Fraud Detection System Configuration Module.

Centralized configuration for fraud detection with handling
of severely imbalanced datasets.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List


PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class ModelConfig:
    """Configuration for fraud detection models."""
    
    # XGBoost parameters (primary model for fraud detection)
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_scale_pos_weight: float = 10.0  # For imbalanced data
    
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 15
    rf_class_weight: str = "balanced"
    
    # Logistic Regression parameters
    lr_C: float = 1.0
    lr_max_iter: int = 1000
    lr_class_weight: str = "balanced"
    
    # SMOTE parameters for handling class imbalance
    smote_sampling_strategy: float = 0.5  # Ratio of minority to majority
    smote_k_neighbors: int = 5
    use_smote: bool = True
    
    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # Threshold tuning (fraud detection often needs custom thresholds)
    default_threshold: float = 0.5
    optimize_threshold: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "xgboost": {
                "n_estimators": self.xgb_n_estimators,
                "max_depth": self.xgb_max_depth,
                "learning_rate": self.xgb_learning_rate,
                "subsample": self.xgb_subsample,
                "colsample_bytree": self.xgb_colsample_bytree,
                "scale_pos_weight": self.xgb_scale_pos_weight,
            },
            "random_forest": {
                "n_estimators": self.rf_n_estimators,
                "max_depth": self.rf_max_depth,
                "class_weight": self.rf_class_weight,
            },
            "smote": {
                "sampling_strategy": self.smote_sampling_strategy,
                "k_neighbors": self.smote_k_neighbors,
                "enabled": self.use_smote,
            },
            "training": {
                "test_size": self.test_size,
                "random_state": self.random_state,
                "cv_folds": self.cv_folds,
            },
        }


@dataclass
class DataConfig:
    """Configuration for fraud detection data."""
    
    # Feature columns
    feature_names: List[str] = field(default_factory=lambda: [
        "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"
    ])
    
    # Engineered features
    engineered_features: List[str] = field(default_factory=lambda: [
        "isPayment", "isMovement", "transactionRatio", 
        "accountDiff", "balanceChange", "destBalanceChange"
    ])
    
    # Target column
    target_column: str = "isFraud"
    
    # Transaction types
    transaction_types: List[str] = field(default_factory=lambda: [
        "PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"
    ])


@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    model_dir: Path = field(default_factory=lambda: MODELS_DIR)
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    
    model_path: Path = field(default_factory=lambda: MODELS_DIR / "fraud_detector.pkl")
    scaler_path: Path = field(default_factory=lambda: MODELS_DIR / "fraud_scaler.pkl")
    metadata_path: Path = field(default_factory=lambda: MODELS_DIR / "fraud_metadata.yaml")
    
    def ensure_dirs(self) -> None:
        """Create directories if they don't exist."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.paths = PathConfig()


default_config = Config()
