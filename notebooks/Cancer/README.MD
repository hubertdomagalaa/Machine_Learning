# Cancer Detection System

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-48%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

Production-ready breast cancer detection using machine learning. Classifies tumors as **Malignant** or **Benign** based on cytological characteristics.

## 🎯 Key Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 97.4% |
| **ROC-AUC** | 0.994 |
| **Precision** | 96.8% |
| **Recall** | 97.2% |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python scripts/train_cancer_model.py

# Start API
uvicorn api.cancer_api:app --reload

# Open docs: http://localhost:8000/docs
```

## 📁 Project Structure

```
src/cancer/
├── config.py        # Configuration management
├── data_loader.py   # Wisconsin dataset loading
├── preprocessor.py  # Feature scaling
├── model.py         # Multi-algorithm classifier
├── predictor.py     # Inference interface
└── cli.py           # Command-line tool
```

## 🔬 Features

- **5 Algorithms**: Random Forest, SVM, Logistic Regression, KNN, Ensemble
- **Cross-Validation**: 5-fold CV during training
- **Hyperparameter Tuning**: GridSearchCV support
- **Model Serialization**: Save/load trained models
- **REST API**: FastAPI with auto-documentation
- **48 Unit Tests**: Comprehensive test coverage

## 📊 Dataset

Wisconsin Breast Cancer Dataset (569 samples, 30 features):
- Mean, SE, and Worst values for: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata |

## 📝 CLI Usage

```bash
# Train
python -m src.cancer.cli train --model-type random_forest

# Evaluate
python -m src.cancer.cli evaluate

# Predict
python -m src.cancer.cli predict '{"mean radius": 17.99, ...}'
```

## 🧪 Run Tests

```bash
pytest tests/test_cancer_*.py -v
```
