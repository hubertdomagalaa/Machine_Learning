"""
Fraud Detection System - Package Initialization.

Production-ready fraud detection system for financial transactions.
"""

from .config import Config, default_config
from .data_loader import FraudDataLoader
from .preprocessor import FraudPreprocessor
from .model import FraudDetector
from .predictor import FraudPredictor

__version__ = "1.0.0"
__author__ = "Hubert Domagała"

__all__ = [
    "Config",
    "default_config",
    "FraudDataLoader",
    "FraudPreprocessor",
    "FraudDetector",
    "FraudPredictor",
]
