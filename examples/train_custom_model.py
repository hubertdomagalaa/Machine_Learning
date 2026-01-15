"""
Custom Model Training Example

This script shows how to train a custom cancer detection model
with different hyperparameters or algorithms.

Usage:
    python examples/train_custom_model.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cancer.data_loader import CancerDataLoader
from src.cancer.preprocessor import CancerPreprocessor
from src.cancer.model import CancerClassifier


def train_custom_model(model_type: str = "random_forest"):
    """
    Train a custom cancer detection model.
    
    Args:
        model_type: One of 'random_forest', 'svm', 'logistic_regression', 
                   'knn', or 'ensemble'
    """
    print(f"🚀 Training Custom {model_type.upper()} Model")
    print("=" * 50)
    
    # Step 1: Load data
    print("\n📊 Loading Wisconsin Breast Cancer dataset...")
    loader = CancerDataLoader()
    X, y = loader.load_data()
    print(f"   Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Step 2: Preprocess
    print("\n⚙️ Preprocessing data...")
    preprocessor = CancerPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(X, y)
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Step 3: Train model
    print(f"\n🧠 Training {model_type} classifier...")
    classifier = CancerClassifier(model_type=model_type)
    classifier.fit(X_train, y_train, validate=True)
    
    # Step 4: Evaluate
    print("\n📈 Evaluating model...")
    metrics = classifier.evaluate(X_test, y_test)
    
    print("\n" + "=" * 50)
    print("📊 MODEL PERFORMANCE")
    print("=" * 50)
    print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   Precision: {metrics['precision']*100:.2f}%")
    print(f"   Recall:    {metrics['recall']*100:.2f}%")
    print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"   False Negative Rate: {metrics['false_negative_rate']*100:.2f}%")
    print("=" * 50)
    
    # Step 5: Save model (optional)
    save_path = project_root / "models" / f"custom_{model_type}.pkl"
    classifier.save(save_path)
    print(f"\n💾 Model saved to: {save_path}")
    
    return classifier, metrics


def compare_models():
    """Compare all available model types."""
    print("🔄 Comparing All Model Types")
    print("=" * 60)
    
    model_types = ["random_forest", "svm", "logistic_regression", "knn", "ensemble"]
    results = {}
    
    for model_type in model_types:
        print(f"\n--- {model_type.upper()} ---")
        _, metrics = train_custom_model(model_type)
        results[model_type] = metrics
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Recall':>10} {'ROC-AUC':>10}")
    print("-" * 60)
    for model, metrics in results.items():
        print(f"{model:<20} {metrics['accuracy']*100:>9.1f}% {metrics['recall']*100:>9.1f}% {metrics['roc_auc']:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train custom cancer detection model")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="random_forest",
        choices=["random_forest", "svm", "logistic_regression", "knn", "ensemble", "compare"],
        help="Model type to train (or 'compare' to compare all)"
    )
    
    args = parser.parse_args()
    
    if args.model == "compare":
        compare_models()
    else:
        train_custom_model(args.model)
