"""
Quick Start Example - Cancer Detection API

This script demonstrates how to use the Cancer Detection system
for making predictions programmatically.

Usage:
    python examples/quick_start.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cancer.predictor import CancerPredictor


def main():
    """Run a quick prediction example."""
    
    print("🏥 Cancer Detection System - Quick Start")
    print("=" * 50)
    
    # Initialize predictor (loads trained model)
    print("\n📦 Loading model...")
    predictor = CancerPredictor()
    print("✅ Model loaded successfully!")
    
    # Example patient features (30 features from FNA)
    sample_features = {
        "mean_radius": 17.99,
        "mean_texture": 10.38,
        "mean_perimeter": 122.8,
        "mean_area": 1001.0,
        "mean_smoothness": 0.1184,
        "mean_compactness": 0.2776,
        "mean_concavity": 0.3001,
        "mean_concave_points": 0.1471,
        "mean_symmetry": 0.2419,
        "mean_fractal_dimension": 0.07871,
        "se_radius": 1.095,
        "se_texture": 0.9053,
        "se_perimeter": 8.589,
        "se_area": 153.4,
        "se_smoothness": 0.006399,
        "se_compactness": 0.04904,
        "se_concavity": 0.05373,
        "se_concave_points": 0.01587,
        "se_symmetry": 0.03003,
        "se_fractal_dimension": 0.006193,
        "worst_radius": 25.38,
        "worst_texture": 17.33,
        "worst_perimeter": 184.6,
        "worst_area": 2019.0,
        "worst_smoothness": 0.1622,
        "worst_compactness": 0.6656,
        "worst_concavity": 0.7119,
        "worst_concave_points": 0.2654,
        "worst_symmetry": 0.4601,
        "worst_fractal_dimension": 0.1189,
    }
    
    # Make prediction
    print("\n🔬 Analyzing patient features...")
    result = predictor.predict_single(sample_features)
    
    # Display results
    print("\n" + "=" * 50)
    print("📊 PREDICTION RESULT")
    print("=" * 50)
    print(f"   Diagnosis: {result['diagnosis']}")
    print(f"   Confidence: {result['confidence']*100:.1f}%")
    print(f"   Malignant Probability: {result['probabilities']['malignant']*100:.1f}%")
    print(f"   Benign Probability: {result['probabilities']['benign']*100:.1f}%")
    print("=" * 50)
    
    # Disclaimer
    print("\n⚠️  DISCLAIMER: This is for educational purposes only.")
    print("    Always consult qualified healthcare professionals.")


if __name__ == "__main__":
    main()
