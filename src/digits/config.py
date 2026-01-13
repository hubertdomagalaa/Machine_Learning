"""
Digits Recognition System Configuration.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List


PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class ModelConfig:
    """Model configuration for digit recognition."""
    
    # SVM (best for digits)
    svm_kernel: str = "rbf"
    svm_C: float = 10.0
    svm_gamma: float = 0.001
    
    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: int = 20
    
    # MLP Neural Network
    mlp_hidden_layers: tuple = (100, 50)
    mlp_max_iter: int = 500
    mlp_alpha: float = 0.0001
    
    # Training
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # PCA
    use_pca: bool = False
    pca_components: int = 40


@dataclass
class DataConfig:
    """Data configuration."""
    
    n_classes: int = 10
    image_size: tuple = (8, 8)
    n_features: int = 64


@dataclass
class PathConfig:
    """Path configuration."""
    
    model_dir: Path = field(default_factory=lambda: MODELS_DIR)
    model_path: Path = field(default_factory=lambda: MODELS_DIR / "digits_classifier.pkl")
    scaler_path: Path = field(default_factory=lambda: MODELS_DIR / "digits_scaler.pkl")
    
    def ensure_dirs(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.paths = PathConfig()


default_config = Config()
