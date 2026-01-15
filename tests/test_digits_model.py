"""
Unit tests for Digits Recognition System.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.digits.data_loader import DigitsDataLoader
from src.digits.preprocessor import DigitsPreprocessor
from src.digits.model import DigitsClassifier


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Load and prepare sample data for testing."""
    loader = DigitsDataLoader()
    X, y = loader.load_data()
    
    # Use subset for faster testing
    indices = np.random.RandomState(42).choice(len(X), size=200, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    
    preprocessor = DigitsPreprocessor()
    X_scaled = preprocessor.fit_transform(X_sample)
    
    return X_scaled, y_sample, preprocessor


# =============================================================================
# Data Loader Tests
# =============================================================================

class TestDigitsDataLoader:
    """Test suite for DigitsDataLoader class."""
    
    def test_load_data_returns_arrays(self):
        """Test that load_data returns numpy arrays."""
        loader = DigitsDataLoader()
        X, y = loader.load_data()
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
    
    def test_load_data_correct_shape(self):
        """Test that loaded data has correct shape."""
        loader = DigitsDataLoader()
        X, y = loader.load_data()
        
        # UCI digits dataset: 1797 samples, 64 features (8x8 images)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 64
    
    def test_labels_are_digits(self):
        """Test that labels are digits 0-9."""
        loader = DigitsDataLoader()
        _, y = loader.load_data()
        
        assert np.all(y >= 0)
        assert np.all(y <= 9)
        assert len(np.unique(y)) == 10


# =============================================================================
# Preprocessor Tests
# =============================================================================

class TestDigitsPreprocessor:
    """Test suite for DigitsPreprocessor class."""
    
    def test_fit_transform_returns_scaled_data(self, sample_data):
        """Test that fit_transform returns scaled data."""
        X_scaled, _, _ = sample_data
        
        # Check that data is scaled (mean close to 0, std close to 1)
        assert np.abs(X_scaled.mean()) < 0.5
        assert 0.5 < X_scaled.std() < 1.5
    
    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises error."""
        preprocessor = DigitsPreprocessor()
        X = np.random.rand(10, 64)
        
        with pytest.raises(RuntimeError):
            preprocessor.transform(X)
    
    def test_transform_preserves_shape(self):
        """Test that transform preserves data shape."""
        loader = DigitsDataLoader()
        X, _ = loader.load_data()
        
        preprocessor = DigitsPreprocessor()
        X_scaled = preprocessor.fit_transform(X)
        
        assert X_scaled.shape == X.shape


# =============================================================================
# Model Tests
# =============================================================================

class TestDigitsClassifier:
    """Test suite for DigitsClassifier class."""
    
    def test_create_svm(self):
        """Test creating an SVM classifier."""
        classifier = DigitsClassifier(model_type="svm")
        assert classifier.model_type == "svm"
        assert not classifier.is_fitted
    
    def test_create_random_forest(self):
        """Test creating a random forest classifier."""
        classifier = DigitsClassifier(model_type="random_forest")
        assert classifier.model_type == "random_forest"
    
    def test_invalid_model_type_raises_error(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError):
            DigitsClassifier(model_type="invalid_model")
    
    def test_fit_sets_is_fitted(self, sample_data):
        """Test that fitting sets is_fitted flag."""
        X, y, _ = sample_data
        classifier = DigitsClassifier(model_type="svm")
        classifier.fit(X, y)
        
        assert classifier.is_fitted
    
    def test_predict_returns_correct_shape(self, sample_data):
        """Test that predict returns correct shape."""
        X, y, _ = sample_data
        classifier = DigitsClassifier(model_type="svm")
        classifier.fit(X, y)
        
        predictions = classifier.predict(X[:10])
        
        assert predictions.shape == (10,)
    
    def test_predict_returns_valid_digits(self, sample_data):
        """Test that predictions are valid digits (0-9)."""
        X, y, _ = sample_data
        classifier = DigitsClassifier(model_type="svm")
        classifier.fit(X, y)
        
        predictions = classifier.predict(X)
        
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 9)
    
    def test_predict_before_fit_raises_error(self):
        """Test that predicting before fitting raises error."""
        classifier = DigitsClassifier(model_type="svm")
        X = np.random.rand(10, 64)
        
        with pytest.raises(RuntimeError):
            classifier.predict(X)
    
    def test_model_accuracy_is_reasonable(self, sample_data):
        """Test that model achieves reasonable accuracy on digits."""
        X, y, _ = sample_data
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        classifier = DigitsClassifier(model_type="svm")
        classifier.fit(X_train, y_train)
        
        predictions = classifier.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        
        # SVM on digits should achieve > 90% accuracy
        assert accuracy > 0.85
    
    def test_evaluate_returns_metrics(self, sample_data):
        """Test that evaluate returns expected metrics."""
        X, y, _ = sample_data
        
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        classifier = DigitsClassifier(model_type="svm")
        classifier.fit(X_train, y_train)
        
        metrics = classifier.evaluate(X_test, y_test)
        
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1


# =============================================================================
# Model Serialization Tests
# =============================================================================

class TestModelSerialization:
    """Test suite for model save/load functionality."""
    
    def test_save_and_load_model(self, sample_data):
        """Test saving and loading a trained model."""
        X, y, _ = sample_data
        
        classifier = DigitsClassifier(model_type="svm")
        classifier.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            classifier.save(path)
            
            assert path.exists()
            
            loaded = DigitsClassifier.load(path)
            assert loaded.is_fitted
    
    def test_loaded_model_produces_same_predictions(self, sample_data):
        """Test that loaded model produces same predictions."""
        X, y, _ = sample_data
        
        classifier = DigitsClassifier(model_type="svm")
        classifier.fit(X, y)
        
        original_predictions = classifier.predict(X[:5])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            classifier.save(path)
            loaded = DigitsClassifier.load(path)
            
            loaded_predictions = loaded.predict(X[:5])
            
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
