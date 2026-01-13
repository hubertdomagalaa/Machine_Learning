#!/usr/bin/env python
"""
Cancer Detection Model Training Script.

This script trains and saves a production-ready cancer detection model.
Run this script to create model artifacts before starting the API.

Usage:
    python scripts/train_cancer_model.py
    python scripts/train_cancer_model.py --model-type ensemble --tune
"""

import argparse
import logging
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cancer import CancerDataLoader, CancerPreprocessor, CancerClassifier, CancerPredictor
from src.cancer.config import default_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model(
    model_type: str = "random_forest",
    tune: bool = False,
    compare_models: bool = False
) -> None:
    """
    Train and save a cancer detection model.
    
    Args:
        model_type: Type of model to train
        tune: Whether to perform hyperparameter tuning
        compare_models: Whether to compare all model types
    """
    print("\n" + "=" * 60)
    print("CANCER DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model type: {model_type}")
    print(f"Hyperparameter tuning: {'Yes' if tune else 'No'}")
    print("=" * 60 + "\n")
    
    # Ensure model directory exists
    default_config.paths.ensure_dirs()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    loader = CancerDataLoader()
    X, y = loader.load_data()
    feature_names = loader.get_feature_names()
    data_info = loader.get_data_info()
    
    print(f"  ✓ Loaded {data_info['n_samples']} samples with {data_info['n_features']} features")
    print(f"  ✓ Class distribution: Malignant={data_info['class_distribution']['malignant (0)']}, "
          f"Benign={data_info['class_distribution']['benign (1)']}")
    
    # Step 2: Preprocess data
    print("\nStep 2: Preprocessing data...")
    preprocessor = CancerPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform_split(
        X, y, feature_names
    )
    
    print(f"  ✓ Training set: {len(X_train)} samples")
    print(f"  ✓ Test set: {len(X_test)} samples")
    
    # Save preprocessor
    scaler_path = preprocessor.save()
    print(f"  ✓ Preprocessor saved to: {scaler_path}")
    
    # Step 3: Compare models (optional)
    if compare_models:
        print("\nStep 3: Comparing all model types...")
        results = {}
        
        for mtype in CancerClassifier.SUPPORTED_MODELS:
            print(f"\n  Training {mtype}...")
            clf = CancerClassifier(model_type=mtype)
            clf.fit(X_train, y_train, validate=False)
            metrics = clf.evaluate(X_test, y_test, detailed=False)
            results[mtype] = metrics
            print(f"    Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        
        # Find best model
        best_model = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        print(f"\n  Best model: {best_model} (AUC: {results[best_model]['roc_auc']:.4f})")
        
        if model_type == "auto":
            model_type = best_model
    
    # Step 4: Train final model
    print(f"\nStep 3: Training {model_type} model...")
    classifier = CancerClassifier(model_type=model_type)
    
    if tune:
        print("  Performing hyperparameter tuning (this may take a while)...")
        tune_results = classifier.tune_hyperparameters(X_train, y_train)
        print(f"  ✓ Best parameters: {tune_results['best_params']}")
        print(f"  ✓ Best CV score: {tune_results['best_score']:.4f}")
    else:
        classifier.fit(X_train, y_train, validate=True)
        print(f"  ✓ Training complete with cross-validation")
    
    # Step 5: Evaluate model
    print("\nStep 4: Evaluating model on test set...")
    metrics = classifier.evaluate(X_test, y_test, detailed=True)
    
    print("\n" + "-" * 40)
    print("EVALUATION RESULTS")
    print("-" * 40)
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  F1 Score:           {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
    print(f"  Specificity:        {metrics['specificity']:.4f}")
    print("-" * 40)
    
    # Print confusion matrix
    cm = metrics['confusion_matrix']
    print("\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Mal   Ben")
    print(f"  Actual Mal  |  {cm[0][0]:3d}   {cm[0][1]:3d}")
    print(f"         Ben  |  {cm[1][0]:3d}   {cm[1][1]:3d}")
    
    # Step 6: Save model
    print("\nStep 5: Saving model...")
    model_path = classifier.save()
    print(f"  ✓ Model saved to: {model_path}")
    
    # Step 7: Feature importance
    importance = classifier.get_feature_importance(feature_names)
    if importance:
        print("\nTop 10 Most Important Features:")
        for i, (name, score) in enumerate(list(importance.items())[:10]):
            print(f"  {i+1:2d}. {name}: {score:.4f}")
    
    # Save metadata
    metadata = {
        "model_type": model_type,
        "training_date": datetime.now().isoformat(),
        "metrics": {k: v for k, v in metrics.items() if k != 'confusion_matrix'},
        "data_info": {
            "n_samples": data_info['n_samples'],
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        "feature_names": feature_names,
    }
    
    metadata_path = default_config.paths.metadata_path
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\n  ✓ Metadata saved to: {metadata_path}")
    
    # Verify by loading
    print("\nStep 6: Verifying saved model...")
    try:
        loaded_predictor = CancerPredictor.from_pretrained()
        sample_result = loaded_predictor.predict(X_test[:1])
        print(f"  ✓ Model verified: can make predictions")
        print(f"  ✓ Sample prediction: {sample_result['diagnoses'][0]}")
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model Type:     {model_type}")
    print(f"Accuracy:       {metrics['accuracy']:.2%}")
    print(f"ROC-AUC:        {metrics['roc_auc']:.4f}")
    print(f"Model Path:     {model_path}")
    print(f"Scaler Path:    {scaler_path}")
    print("=" * 60)
    print("\nTo start the API server, run:")
    print("  uvicorn api.cancer_api:app --reload")
    print("\nTo use the CLI, run:")
    print("  python -m src.cancer.cli predict '<features>'")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train a cancer detection model"
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "svm", "logistic_regression", "knn", "ensemble"],
        default="random_forest",
        help="Type of model to train (default: random_forest)"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all model types before training"
    )
    
    args = parser.parse_args()
    
    train_model(
        model_type=args.model_type,
        tune=args.tune,
        compare_models=args.compare
    )


if __name__ == "__main__":
    main()
