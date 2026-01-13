"""
Unit tests for Fraud Detection System.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.fraud_detection.data_loader import FraudDataLoader, DataValidationError
from src.fraud_detection.preprocessor import FraudPreprocessor
from src.fraud_detection.model import FraudDetector


class TestFraudDataLoader:
    """Tests for FraudDataLoader."""
    
    def test_load_synthetic_data(self):
        """Test loading synthetic data."""
        loader = FraudDataLoader()
        X, y = loader.load_data()
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y)
    
    def test_fraud_rate_is_low(self):
        """Test that synthetic data has realistic low fraud rate."""
        loader = FraudDataLoader()
        X, y = loader.load_data()
        
        fraud_rate = y.mean()
        assert 0.001 < fraud_rate < 0.1  # Between 0.1% and 10%
    
    def test_engineered_features_created(self):
        """Test that engineered features are created."""
        loader = FraudDataLoader()
        X, y = loader.load_data()
        
        # Should have more than basic features
        assert X.shape[1] >= 10
    
    def test_sample_size_works(self):
        """Test sample size limiting."""
        loader = FraudDataLoader()
        X, y = loader.load_data(sample_size=1000)
        
        assert len(X) <= 1000
    
    def test_get_data_info(self):
        """Test data info retrieval."""
        loader = FraudDataLoader()
        info = loader.get_data_info()
        
        assert "n_samples" in info
        assert "fraud_count" in info
        assert "fraud_rate" in info
        assert "class_imbalance_ratio" in info
    
    def test_no_nan_values(self):
        """Test no NaN values in data."""
        loader = FraudDataLoader()
        X, y = loader.load_data()
        
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
    
    def test_binary_targets(self):
        """Test targets are binary."""
        loader = FraudDataLoader()
        X, y = loader.load_data()
        
        assert set(np.unique(y)).issubset({0, 1})


class TestFraudPreprocessor:
    """Tests for FraudPreprocessor."""
    
    @pytest.fixture
    def sample_data(self):
        """Get sample data."""
        loader = FraudDataLoader()
        X, y = loader.load_data(sample_size=1000)
        return X, y
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform works."""
        X, y = sample_data
        preprocessor = FraudPreprocessor(resampling_strategy="none")
        
        X_scaled = preprocessor.fit_transform(X)
        
        assert preprocessor.is_fitted
        assert X_scaled.shape == X.shape
    
    def test_transform_scales_data(self, sample_data):
        """Test that transform scales data."""
        X, y = sample_data
        preprocessor = FraudPreprocessor(resampling_strategy="none")
        
        X_scaled = preprocessor.fit_transform(X)
        
        # Scaled data should have different statistics
        # RobustScaler centers around median
        assert np.abs(np.median(X_scaled, axis=0)).mean() < np.abs(np.median(X, axis=0)).mean() + 1
    
    def test_fit_transform_resample(self, sample_data):
        """Test fit_transform_resample returns splits."""
        X, y = sample_data
        preprocessor = FraudPreprocessor(resampling_strategy="none")
        
        X_train, X_test, y_train, y_test = preprocessor.fit_transform_resample(X, y)
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
    
    def test_save_and_load(self, sample_data):
        """Test save and load functionality."""
        X, y = sample_data
        preprocessor = FraudPreprocessor(resampling_strategy="none")
        preprocessor.fit(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preprocessor.pkl"
            preprocessor.save(path)
            
            loaded = FraudPreprocessor.load(path)
            
            assert loaded.is_fitted
            # Transform should produce same results
            X_orig = preprocessor.transform(X)
            X_loaded = loaded.transform(X)
            np.testing.assert_array_almost_equal(X_orig, X_loaded)


class TestFraudDetector:
    """Tests for FraudDetector."""
    
    @pytest.fixture
    def prepared_data(self):
        """Prepare train/test data."""
        loader = FraudDataLoader()
        X, y = loader.load_data(sample_size=2000)
        
        preprocessor = FraudPreprocessor(resampling_strategy="none")
        X_train, X_test, y_train, y_test = preprocessor.fit_transform_resample(X, y)
        
        return X_train, X_test, y_train, y_test
    
    def test_create_detector(self):
        """Test detector creation."""
        detector = FraudDetector(model_type="random_forest")
        
        assert detector.model_type == "random_forest"
        assert detector.is_fitted == False
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError):
            FraudDetector(model_type="invalid")
    
    def test_fit_detector(self, prepared_data):
        """Test fitting detector."""
        X_train, X_test, y_train, y_test = prepared_data
        
        detector = FraudDetector(model_type="random_forest")
        detector.fit(X_train, y_train, optimize_threshold=False)
        
        assert detector.is_fitted
    
    def test_predict(self, prepared_data):
        """Test prediction."""
        X_train, X_test, y_train, y_test = prepared_data
        
        detector = FraudDetector(model_type="random_forest")
        detector.fit(X_train, y_train, optimize_threshold=False)
        
        predictions = detector.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba(self, prepared_data):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = prepared_data
        
        detector = FraudDetector(model_type="random_forest")
        detector.fit(X_train, y_train, optimize_threshold=False)
        
        probas = detector.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_evaluate(self, prepared_data):
        """Test evaluation metrics."""
        X_train, X_test, y_train, y_test = prepared_data
        
        detector = FraudDetector(model_type="random_forest")
        detector.fit(X_train, y_train, optimize_threshold=False)
        
        metrics = detector.evaluate(X_test, y_test)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "roc_auc" in metrics
        assert "fraud_detection_rate" in metrics
    
    def test_threshold_optimization(self, prepared_data):
        """Test threshold optimization."""
        X_train, X_test, y_train, y_test = prepared_data
        
        detector = FraudDetector(model_type="random_forest")
        detector.fit(X_train, y_train, optimize_threshold=True, X_val=X_test, y_val=y_test)
        
        # Threshold should be between 0 and 1
        assert 0 < detector.threshold < 1
    
    def test_save_and_load(self, prepared_data):
        """Test save and load."""
        X_train, X_test, y_train, y_test = prepared_data
        
        detector = FraudDetector(model_type="random_forest")
        detector.fit(X_train, y_train, optimize_threshold=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            detector.save(path)
            
            loaded = FraudDetector.load(path)
            
            assert loaded.is_fitted
            
            # Same predictions
            orig_pred = detector.predict(X_test)
            loaded_pred = loaded.predict(X_test)
            np.testing.assert_array_equal(orig_pred, loaded_pred)


class TestModelPerformance:
    """Test model achieves reasonable performance."""
    
    def test_random_forest_performance(self):
        """Test Random Forest achieves reasonable AUC."""
        loader = FraudDataLoader()
        X, y = loader.load_data(sample_size=5000)
        
        preprocessor = FraudPreprocessor(resampling_strategy="none")
        X_train, X_test, y_train, y_test = preprocessor.fit_transform_resample(X, y)
        
        detector = FraudDetector(model_type="random_forest")
        detector.fit(X_train, y_train, optimize_threshold=False)
        
        metrics = detector.evaluate(X_test, y_test)
        
        # Should achieve reasonable AUC on synthetic data
        assert metrics["roc_auc"] >= 0.70
