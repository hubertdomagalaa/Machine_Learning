"""
Cancer Detection System - Command Line Interface.

This module provides a CLI for training, evaluating, and making
predictions with the cancer detection system.
"""

import click
import json
from pathlib import Path
import logging
import sys

from .data_loader import CancerDataLoader
from .preprocessor import CancerPreprocessor
from .model import CancerClassifier
from .predictor import CancerPredictor
from .config import default_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Cancer Detection System CLI.
    
    Train, evaluate, and make predictions with the breast cancer
    detection model.
    """
    pass


@cli.command()
@click.option(
    "--model-type", "-m",
    type=click.Choice(["random_forest", "svm", "logistic_regression", "knn", "ensemble"]),
    default="random_forest",
    help="Type of model to train"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to save trained model"
)
@click.option(
    "--tune/--no-tune",
    default=False,
    help="Perform hyperparameter tuning"
)
def train(model_type: str, output_dir: Path, tune: bool):
    """Train a new cancer detection model."""
    click.echo(f"Training {model_type} model...")
    
    # Load data
    loader = CancerDataLoader()
    X, y = loader.load_data()
    feature_names = loader.get_feature_names()
    
    # Preprocess
    preprocessor = CancerPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform_split(
        X, y, feature_names
    )
    click.echo(f"Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Create and train model
    classifier = CancerClassifier(model_type=model_type)
    
    if tune:
        click.echo("Performing hyperparameter tuning...")
        tune_results = classifier.tune_hyperparameters(X_train, y_train)
        click.echo(f"Best parameters: {tune_results['best_params']}")
    else:
        classifier.fit(X_train, y_train)
    
    # Evaluate
    metrics = classifier.evaluate(X_test, y_test)
    
    click.echo("\n" + "=" * 50)
    click.echo("EVALUATION RESULTS")
    click.echo("=" * 50)
    click.echo(f"Accuracy:  {metrics['accuracy']:.4f}")
    click.echo(f"Precision: {metrics['precision']:.4f}")
    click.echo(f"Recall:    {metrics['recall']:.4f}")
    click.echo(f"F1 Score:  {metrics['f1_score']:.4f}")
    click.echo(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    click.echo(f"False Negative Rate: {metrics['false_negative_rate']:.4f}")
    click.echo("=" * 50)
    
    # Save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "cancer_classifier.pkl"
        scaler_path = output_dir / "cancer_scaler.pkl"
    else:
        model_path = None
        scaler_path = None
    
    classifier.save(model_path)
    preprocessor.save(scaler_path)
    
    click.echo(f"\nModel saved successfully!")
    
    # Show feature importance if available
    importance = classifier.get_feature_importance(feature_names)
    if importance:
        click.echo("\nTop 10 Important Features:")
        for i, (name, score) in enumerate(list(importance.items())[:10]):
            click.echo(f"  {i+1}. {name}: {score:.4f}")


@cli.command()
@click.option(
    "--model-path", "-m",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to trained model"
)
def evaluate(model_path: Path):
    """Evaluate a trained model on test data."""
    # Load model
    if model_path:
        classifier = CancerClassifier.load(model_path)
    else:
        model_path = default_config.paths.model_path
        if not model_path.exists():
            click.echo("Error: No trained model found. Run 'train' first.", err=True)
            sys.exit(1)
        classifier = CancerClassifier.load(model_path)
    
    # Load preprocessor
    scaler_path = default_config.paths.scaler_path
    if not scaler_path.exists():
        click.echo("Error: Preprocessor not found.", err=True)
        sys.exit(1)
    preprocessor = CancerPreprocessor.load(scaler_path)
    
    # Load and preprocess test data
    loader = CancerDataLoader()
    X, y = loader.load_data()
    X_scaled = preprocessor.transform(X)
    
    # Evaluate
    metrics = classifier.evaluate(X_scaled, y)
    
    click.echo("\n" + "=" * 50)
    click.echo("MODEL EVALUATION")
    click.echo("=" * 50)
    click.echo(json.dumps(metrics, indent=2, default=str))


@cli.command()
@click.argument("features", type=str)
@click.option(
    "--model-path", "-m",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to trained model"
)
def predict(features: str, model_path: Path):
    """
    Make a prediction for given features.
    
    FEATURES: JSON string with feature values, e.g.:
    '{"mean radius": 17.99, "mean texture": 10.38, ...}'
    
    Or a comma-separated list of 30 values.
    """
    try:
        # Load predictor
        predictor = CancerPredictor.from_pretrained(model_path=model_path)
        
        # Parse input
        try:
            feature_dict = json.loads(features)
        except json.JSONDecodeError:
            # Try comma-separated values
            values = [float(v.strip()) for v in features.split(",")]
            if len(values) != 30:
                click.echo(f"Error: Expected 30 features, got {len(values)}", err=True)
                sys.exit(1)
            feature_names = predictor.get_feature_names()
            feature_dict = dict(zip(feature_names, values))
        
        # Make prediction
        result = predictor.predict_single(feature_dict)
        
        click.echo("\n" + "=" * 50)
        click.echo("PREDICTION RESULT")
        click.echo("=" * 50)
        click.echo(f"Diagnosis:  {result['diagnosis']}")
        click.echo(f"Confidence: {result['confidence']:.2%}")
        click.echo(f"P(Malignant): {result['probabilities']['malignant']:.4f}")
        click.echo(f"P(Benign):    {result['probabilities']['benign']:.4f}")
        click.echo("=" * 50)
        
    except FileNotFoundError as e:
        click.echo(f"Error: {e}. Run 'train' first.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def info():
    """Show information about the trained model."""
    try:
        predictor = CancerPredictor.from_pretrained()
        info = predictor.get_model_info()
        
        click.echo("\n" + "=" * 50)
        click.echo("MODEL INFORMATION")
        click.echo("=" * 50)
        click.echo(json.dumps(info, indent=2, default=str))
        
        click.echo("\nExpected Features:")
        for i, name in enumerate(predictor.get_feature_names()):
            click.echo(f"  {i+1}. {name}")
            
    except FileNotFoundError:
        click.echo("No trained model found. Run 'train' first.", err=True)
        sys.exit(1)


@cli.command()
def data_info():
    """Show information about the dataset."""
    loader = CancerDataLoader()
    info = loader.get_data_info()
    
    click.echo("\n" + "=" * 50)
    click.echo("DATASET INFORMATION")
    click.echo("=" * 50)
    click.echo(f"Samples: {info['n_samples']}")
    click.echo(f"Features: {info['n_features']}")
    click.echo(f"Class Distribution:")
    for cls, count in info['class_distribution'].items():
        click.echo(f"  {cls}: {count}")
    click.echo(f"Class Balance (proportion of benign): {info['class_balance']:.2%}")


if __name__ == "__main__":
    cli()
