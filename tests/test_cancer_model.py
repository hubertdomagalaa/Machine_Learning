"""
Unit tests for Cancer Detection System - Model Module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.cancer.data_loader import CancerDataLoader
from src.cancer.preprocessor import CancerPreprocessor
from src.cancer.model import CancerClassifier


@pytest.fixture
def sample_data():
    """Load and prepare sample data for testing."""
    loader = CancerDataLoader()
    X, y = loader.load_data()
    
    preprocessor = CancerPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform_split(X, y)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
    }


class TestCancerClassifier:
    """Test suite for CancerClassifier class."""
    
    def test_create_random_forest(self):
        """Test creating a random forest classifier."""
        clf = CancerClassifier(model_type="random_forest")
        assert clf.model_type == "random_forest"
        assert clf.is_fitted == False
    
    def test_create_svm(self):
        """Test creating an SVM classifier."""
        clf = CancerClassifier(model_type="svm")
        assert clf.model_type == "svm"
    
    def test_create_logistic_regression(self):
        """Test creating a logistic regression classifier."""
        clf = CancerClassifier(model_type="logistic_regression")
        assert clf.model_type == "logistic_regression"
    
    def test_create_knn(self):
        """Test creating a KNN classifier."""
        clf = CancerClassifier(model_type="knn")
        assert clf.model_type == "knn"
    
    def test_create_ensemble(self):
        """Test creating an ensemble classifier."""
        clf = CancerClassifier(model_type="ensemble")
        assert clf.model_type == "ensemble"
    
    def test_invalid_model_type_raises_error(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            CancerClassifier(model_type="invalid")
    
    def test_fit_sets_is_fitted(self, sample_data):
        """Test that fitting sets is_fitted flag."""
        clf = CancerClassifier(model_type="random_forest")
        clf.fit(sample_data["X_train"], sample_data["y_train"], validate=False)
        
        assert clf.is_fitted == True
    
    def test_fit_returns_self(self, sample_data):
        """Test that fit returns self for chaining."""
        clf = CancerClassifier(model_type="random_forest")
        result = clf.fit(sample_data["X_train"], sample_data["y_train"], validate=False)
        
        assert result is clf
    
    def test_predict_returns_correct_shape(self, sample_data):
        """Test that predict returns correct shape."""
        clf = CancerClassifier(model_type="random_forest")
        clf.fit(sample_data["X_train"], sample_data["y_train"], validate=False)
        
        predictions = clf.predict(sample_data["X_test"])
        
        assert predictions.shape == sample_data["y_test"].shape
    
    def test_predict_returns_binary_values(self, sample_data):
        """Test that predictions are binary (0 or 1)."""
        clf = CancerClassifier(model_type="random_forest")
        clf.fit(sample_data["X_train"], sample_data["y_train"], validate=False)
        
        predictions = clf.predict(sample_data["X_test"])
        
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba_returns_probabilities(self, sample_data):
        """Test that predict_proba returns valid probabilities."""
        clf = CancerClassifier(model_type="random_forest")
        clf.fit(sample_data["X_train"], sample_data["y_train"], validate=False)
        
        probas = clf.predict_proba(sample_data["X_test"])
        
        # Check shape
        assert probas.shape == (len(sample_data["X_test"]), 2)
        
        # Check probabilities sum to 1
        assert np.allclose(probas.sum(axis=1), 1.0)
        
        # Check all probabilities are between 0 and 1
        assert (probas >= 0).all()
        assert (probas <= 1).all()
    
    def test_predict_before_fit_raises_error(self):
        """Test that predicting before fitting raises error."""
        clf = CancerClassifier(model_type="random_forest")
        X = np.random.rand(10, 30)
        
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict(X)
    
    def test_evaluate_returns_metrics(self, sample_data):
        """Test that evaluate returns expected metrics."""
        clf = CancerClassifier(model_type="random_forest")
        clf.fit(sample_data["X_train"], sample_data["y_train"], validate=False)
        
        metrics = clf.evaluate(sample_data["X_test"], sample_data["y_test"])
        
        expected_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        for key in expected_keys:
            assert key in metrics
            assert 0 <= metrics[key] <= 1
    
    def test_evaluate_detailed_includes_confusion_matrix(self, sample_data):
        """Test that detailed evaluation includes confusion matrix."""
        clf = CancerClassifier(model_type="random_forest")
        clf.fit(sample_data["X_train"], sample_data["y_train"], validate=False)
        
        metrics = clf.evaluate(sample_data["X_test"], sample_data["y_test"], detailed=True)
        
        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics
        assert "false_negative_rate" in metrics
    
    def test_model_accuracy_is_reasonable(self, sample_data):
        """Test that model achieves reasonable accuracy."""
        clf = CancerClassifier(model_type="random_forest")
        clf.fit(sample_data["X_train"], sample_data["y_train"], validate=False)
        
        metrics = clf.evaluate(sample_data["X_test"], sample_data["y_test"])
        
        # Random Forest should achieve at least 90% accuracy on this dataset
        assert metrics["accuracy"] >= 0.90
        assert metrics["roc_auc"] >= 0.90
    
    def test_cross_validation_during_fit(self, sample_data):
        """Test that cross-validation is performed during fit."""
        clf = CancerClassifier(model_type="random_forest")
        clf.fit(sample_data["X_train"], sample_data["y_train"], validate=True)
        
        assert clf.training_metrics is not None
        assert "cv_accuracy_mean" in clf.training_metrics
        assert "cv_scores" in clf.training_metrics
    
    def test_feature_importance_for_random_forest(self, sample_data):
        """Test feature importance for tree-based models."""
        clf = CancerClassifier(model_type="random_forest")
        clf.fit(sample_data["X_train"], sample_data["y_train"], validate=False)
        
        importance = clf.get_feature_importance()
        
        assert importance is not None
        assert len(importance) == 30
        assert all(v >= 0 for v in importance.values())
    
    def test_get_model_info(self, sample_data):
        """Test get_model_info returns expected data."""
        clf = CancerClassifier(model_type="random_forest")
        clf.fit(sample_data["X_train"], sample_data["y_train"], validate=False)
        
        info = clf.get_model_info()
        
        assert info["model_type"] == "random_forest"
        assert info["is_fitted"] == True
        assert info["training_date"] is not None


class TestModelSerialization:
    """Test suite for model save/load functionality."""
    
    def test_save_and_load_model(self, sample_data):
        """Test saving and loading a trained model."""
        clf = CancerClassifier(model_type="random_forest")
        clf.fit(sample_data["X_train"], sample_data["y_train"], validate=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            
            # Verify file exists
            assert path.exists()
            
            # Load and verify
            loaded_clf = CancerClassifier.load(path)
            assert loaded_clf.is_fitted == True
            assert loaded_clf.model_type == "random_forest"
    
    def test_loaded_model_produces_same_predictions(self, sample_data):
        """Test that loaded model produces same predictions."""
        clf = CancerClassifier(model_type="random_forest")
        clf.fit(sample_data["X_train"], sample_data["y_train"], validate=False)
        
        original_predictions = clf.predict(sample_data["X_test"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            
            loaded_clf = CancerClassifier.load(path)
            loaded_predictions = loaded_clf.predict(sample_data["X_test"])
            
            assert np.array_equal(original_predictions, loaded_predictions)
    
    def test_save_untrained_model_raises_error(self):
        """Test that saving untrained model raises error."""
        clf = CancerClassifier(model_type="random_forest")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            
            with pytest.raises(RuntimeError, match="untrained"):
                clf.save(path)


class TestDifferentModelTypes:
    """Test different model types work correctly."""
    
    @pytest.mark.parametrize("model_type", [
        "random_forest",
        "svm", 
        "logistic_regression",
        "knn",
        "ensemble"
    ])
    def test_all_model_types_can_train_and_predict(self, sample_data, model_type):
        """Test that all model types can train and make predictions."""
        clf = CancerClassifier(model_type=model_type)
        clf.fit(sample_data["X_train"], sample_data["y_train"], validate=False)
        
        predictions = clf.predict(sample_data["X_test"])
        probas = clf.predict_proba(sample_data["X_test"])
        
        assert len(predictions) == len(sample_data["y_test"])
        assert probas.shape == (len(sample_data["X_test"]), 2)
    
    @pytest.mark.parametrize("model_type", [
        "random_forest",
        "svm",
        "logistic_regression", 
        "knn",
        "ensemble"
    ])
    def test_all_model_types_achieve_reasonable_accuracy(self, sample_data, model_type):
        """Test that all model types achieve reasonable accuracy."""
        clf = CancerClassifier(model_type=model_type)
        clf.fit(sample_data["X_train"], sample_data["y_train"], validate=False)
        
        metrics = clf.evaluate(sample_data["X_test"], sample_data["y_test"])
        
        # All models should achieve at least 85% accuracy on this dataset
        assert metrics["accuracy"] >= 0.85
