"""
Digits Recognition System - Package Initialization.

Production-ready handwritten digit recognition using sklearn's digits dataset.
"""

from .config import Config, default_config
from .data_loader import DigitsDataLoader
from .preprocessor import DigitsPreprocessor
from .model import DigitsClassifier
from .predictor import DigitsPredictor

__version__ = "1.0.0"
__author__ = "Hubert Domagała"

__all__ = [
    "Config",
    "default_config",
    "DigitsDataLoader",
    "DigitsPreprocessor",
    "DigitsClassifier",
    "DigitsPredictor",
]
