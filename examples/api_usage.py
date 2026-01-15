"""
API Usage Example

This script demonstrates how to interact with the Cancer Detection
REST API programmatically using the requests library.

Prerequisites:
    1. Start the API server: make api (or uvicorn api.cancer_api:app --reload)
    2. Run this script: python examples/api_usage.py

Usage:
    python examples/api_usage.py
"""

import requests
import json

# API Configuration
API_URL = "http://localhost:8000"


def check_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def get_model_info():
    """Get information about the loaded model."""
    response = requests.get(f"{API_URL}/model/info")
    return response.json()


def predict_single(features: dict):
    """Make a single prediction."""
    response = requests.post(
        f"{API_URL}/predict",
        json=features,
        headers={"Content-Type": "application/json"}
    )
    return response.json()


def predict_batch(samples: list):
    """Make batch predictions."""
    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"samples": samples},
        headers={"Content-Type": "application/json"}
    )
    return response.json()


def main():
    """Run API usage examples."""
    print("🌐 Cancer Detection API - Usage Examples")
    print("=" * 50)
    
    # Check if API is running
    print("\n1️⃣ Checking API health...")
    if not check_health():
        print("❌ API is not running!")
        print("   Start it with: make api")
        print("   Or: uvicorn api.cancer_api:app --reload")
        return
    print("✅ API is healthy!")
    
    # Get model info
    print("\n2️⃣ Getting model information...")
    info = get_model_info()
    print(f"   Model Type: {info.get('model_type', 'N/A')}")
    print(f"   Version: {info.get('version', 'N/A')}")
    print(f"   Is Fitted: {info.get('is_fitted', 'N/A')}")
    
    # Single prediction
    print("\n3️⃣ Making single prediction...")
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
    
    result = predict_single(sample_features)
    print(f"   Diagnosis: {result.get('diagnosis', 'N/A')}")
    print(f"   Confidence: {result.get('confidence', 0)*100:.1f}%")
    
    # cURL equivalent
    print("\n" + "=" * 50)
    print("📋 cURL equivalent:")
    print("=" * 50)
    print(f"""
curl -X POST "{API_URL}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{json.dumps(sample_features, indent=2)}'
""")
    
    print("\n✅ All examples completed!")
    print(f"\n📚 API Documentation: {API_URL}/docs")


if __name__ == "__main__":
    main()
