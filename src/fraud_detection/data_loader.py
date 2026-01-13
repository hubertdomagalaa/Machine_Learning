"""
Fraud Detection System - Data Loading Module.

Handles loading and validation of financial transaction data
with support for highly imbalanced fraud datasets.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from .config import default_config, DataConfig


logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Exception raised when data validation fails."""
    pass


class FraudDataLoader:
    """
    Data loader for fraud detection datasets.
    
    Handles loading transaction data with proper validation
    and class imbalance statistics.
    
    Example:
        >>> loader = FraudDataLoader()
        >>> X, y = loader.load_data()
        >>> print(f"Fraud rate: {y.mean():.4%}")
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize the data loader."""
        self.config = config or default_config.data
        self._data_cache: Optional[pd.DataFrame] = None
    
    def load_data(
        self,
        file_path: Optional[Path] = None,
        sample_size: Optional[int] = None,
        return_dataframe: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load fraud detection dataset.
        
        Args:
            file_path: Path to CSV file with transaction data
            sample_size: Optional sample size for faster processing
            return_dataframe: If True, return pandas objects
            
        Returns:
            Tuple of (features, targets)
        """
        if file_path is not None:
            df = self._load_from_csv(file_path)
        else:
            # Generate synthetic data for demonstration
            df = self._generate_synthetic_data()
        
        if sample_size and len(df) > sample_size:
            # Stratified sampling to maintain fraud ratio
            df = self._stratified_sample(df, sample_size)
        
        # Engineer features
        df = self._engineer_features(df)
        
        # Prepare features and target
        feature_cols = self._get_feature_columns(df)
        X = df[feature_cols].values
        y = df[self.config.target_column].values
        
        # Validate
        self._validate_data(X, y)
        
        # Log class distribution
        fraud_rate = y.mean()
        logger.info(f"Loaded {len(X)} transactions, fraud rate: {fraud_rate:.4%}")
        
        if return_dataframe:
            return df[feature_cols], df[self.config.target_column]
        
        return X, y
    
    def _load_from_csv(self, file_path: Path) -> pd.DataFrame:
        """Load data from CSV file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        return df
    
    def _generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic transaction data for demonstration.
        
        Creates realistic-looking transaction data with ~1.17% fraud rate
        (typical for real fraud datasets).
        """
        np.random.seed(42)
        
        n_fraud = int(n_samples * 0.0117)  # ~1.17% fraud rate
        n_normal = n_samples - n_fraud
        
        # Transaction types
        types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
        
        data = {
            "step": np.random.randint(1, 744, n_samples),  # Hours in a month
            "type": np.random.choice(types, n_samples, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
            "amount": np.concatenate([
                np.random.exponential(5000, n_normal),
                np.random.exponential(50000, n_fraud)  # Fraud tends to be larger
            ]),
            "oldbalanceOrg": np.random.exponential(50000, n_samples),
            "isFlaggedFraud": np.zeros(n_samples),
            "isFraud": np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
        }
        
        # Calculate derived columns
        data["newbalanceOrig"] = np.maximum(0, data["oldbalanceOrg"] - data["amount"])
        data["oldbalanceDest"] = np.random.exponential(30000, n_samples)
        data["newbalanceDest"] = data["oldbalanceDest"] + data["amount"]
        
        # Shuffle
        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Generated synthetic data with {n_fraud} fraud cases ({n_fraud/n_samples:.2%})")
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for fraud detection.
        
        Features designed based on real fraud patterns:
        - Fraudulent transactions often empty accounts
        - Large transactions relative to balance are suspicious
        - Certain transaction types are more prone to fraud
        """
        df = df.copy()
        
        # Binary features for transaction type
        df["isPayment"] = (df["type"].isin(["PAYMENT", "DEBIT"])).astype(int)
        df["isMovement"] = (df["type"].isin(["CASH_OUT", "TRANSFER"])).astype(int)
        df["isCashOut"] = (df["type"] == "CASH_OUT").astype(int)
        df["isTransfer"] = (df["type"] == "TRANSFER").astype(int)
        
        # Transaction ratio (amount relative to balance)
        df["transactionRatio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
        
        # Account balance difference
        df["accountDiff"] = abs(df["oldbalanceOrg"] - df["oldbalanceDest"])
        
        # Balance changes
        df["balanceChange"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
        df["destBalanceChange"] = df["newbalanceDest"] - df["oldbalanceDest"]
        
        # Suspicious patterns
        df["emptyBalance"] = (df["newbalanceOrig"] == 0).astype(int)
        df["largeTransaction"] = (df["amount"] > df["oldbalanceOrg"]).astype(int)
        
        # Error in balance (mismatch between expected and actual)
        expected_new = df["oldbalanceOrg"] - df["amount"]
        df["balanceError"] = abs(df["newbalanceOrig"] - expected_new.clip(lower=0))
        
        return df
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns to use."""
        # Exclude non-feature columns
        exclude = ["type", "nameOrig", "nameDest", self.config.target_column]
        
        # Use all numeric columns except excluded ones
        feature_cols = [
            col for col in df.columns
            if col not in exclude and df[col].dtype in [np.float64, np.int64, np.int32, np.float32]
        ]
        
        return feature_cols
    
    def _stratified_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Stratified sampling maintaining class distribution."""
        fraud_ratio = df[self.config.target_column].mean()
        n_fraud = max(1, int(n * fraud_ratio))
        n_normal = n - n_fraud
        
        fraud_df = df[df[self.config.target_column] == 1].sample(
            n=min(n_fraud, len(df[df[self.config.target_column] == 1])),
            random_state=42
        )
        normal_df = df[df[self.config.target_column] == 0].sample(
            n=min(n_normal, len(df[df[self.config.target_column] == 0])),
            random_state=42
        )
        
        return pd.concat([fraud_df, normal_df]).sample(frac=1, random_state=42)
    
    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate loaded data."""
        if X.ndim != 2:
            raise DataValidationError(f"Expected 2D feature matrix, got {X.ndim}D")
        
        if y.ndim != 1:
            raise DataValidationError(f"Expected 1D target vector, got {y.ndim}D")
        
        if len(X) != len(y):
            raise DataValidationError(
                f"Feature/target mismatch: {len(X)} vs {len(y)}"
            )
        
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            raise DataValidationError(f"Found {nan_count} NaN values")
        
        unique_targets = np.unique(y)
        if not np.all(np.isin(unique_targets, [0, 1])):
            raise DataValidationError(f"Expected binary targets, got: {unique_targets}")
    
    def get_data_info(self, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Get information about the dataset."""
        X, y = self.load_data(file_path)
        
        return {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "fraud_count": int(np.sum(y == 1)),
            "normal_count": int(np.sum(y == 0)),
            "fraud_rate": float(np.mean(y)),
            "class_imbalance_ratio": float(np.sum(y == 0) / max(1, np.sum(y == 1))),
        }
