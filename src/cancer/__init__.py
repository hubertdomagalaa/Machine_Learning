"""
Cancer Detection System - Package Initialization.

This package provides a production-ready cancer detection system
based on the Wisconsin Breast Cancer Dataset.
"""

from .config import Config, default_config
from .data_loader import CancerDataLoader
from .preprocessor import CancerPreprocessor
from .model import CancerClassifier
from .predictor import CancerPredictor

__version__ = "1.0.0"
__author__ = "Hubert Domagała"

__all__ = [
    "Config",
    "default_config",
    "CancerDataLoader",
    "CancerPreprocessor", 
    "CancerClassifier",
    "CancerPredictor",
]
