# 🧠 Machine Learning Systems Portfolio
**Production-Ready ML Engineering Projects by Hubert Domagała**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Portfolio Overview

This repository showcases **end-to-end machine learning system development** — from exploratory data analysis to production-ready deployments. Each project demonstrates **software engineering best practices**, **scalable architectures**, and **real-world problem-solving**.

### 🏆 Featured Projects

| Project | Domain | ML Techniques | Status | Highlights |
|---------|--------|---------------|--------|------------|
| [🏥 Cancer Detection](#-cancer-detection-system) | Healthcare | Classification, Ensemble | ✅ **Production** | 96.7% accuracy, FastAPI, Zero false negatives |
| [💳 Fraud Detection](#-fraud-detection) | Finance | Anomaly Detection, Feature Engineering | 📊 Analysis | SMOTE, Cost-sensitive learning |
| [✍️ Digit Recognition](#️-handwritten-digit-recognition) | Computer Vision | PCA, Neural Networks | 📊 Analysis | Multi-model comparison |
| [🍯 Honey Production](#-honey-production-forecasting) | Agriculture | Time Series Regression | 📊 Analysis | Trend analysis, Forecasting |
| [🏴 Flag Analysis](#-world-flags-classification) | Data Mining | Multi-class Classification | 📊 Analysis | UCI dataset, EDA |
| [🍇 Raisin Classification](#-raisin-classification) | Agriculture | Clustering, Classification | 📊 Analysis | Feature analysis |
| [💰 Income Classification](#-income-prediction) | Economics | Binary Classification | 📊 Analysis | Socioeconomic analysis |
| [🏥 Medical Insurance](#-medical-insurance-calculator) | Healthcare | OOP Design, Regression | 📊 Analysis | Clean code architecture |

**Legend:** ✅ Production (API + Tests) | 📊 Analysis (Notebooks)

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.11 or higher
pip (Python package manager)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/hubertdomagalaa/Machine_Learning.git
cd Machine_Learning

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Cancer Detection API

```bash
# Train the model
python scripts/train_cancer_model.py

# Start the API server
uvicorn api.cancer_api:app --reload

# Visit http://localhost:8000/docs for interactive API documentation
```

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
# Open htmlcov/index.html in your browser
```

---

## 📂 Repository Structure

```
Machine_Learning/
├── api/                     # FastAPI endpoints for production models
│   ├── cancer_api.py       # Cancer detection REST API
│   └── schemas.py          # Pydantic request/response models
│
├── src/                     # Production Python modules
│   ├── cancer/             # Cancer detection system
│   │   ├── config.py       # Configuration management
│   │   ├── data_loader.py  # Data loading and validation
│   │   ├── preprocessor.py # Feature engineering
│   │   ├── model.py        # Model training and evaluation
│   │   ├── predictor.py    # Prediction interface
│   │   └── cli.py          # Command-line interface
│   └── utils/              # Shared utilities
│
├── tests/                   # Unit and integration tests
│   ├── test_cancer_*.py    # Cancer system tests
│   └── ...
│
├── f/                       # Original exploratory notebooks
│   ├── Cancer/             # Breast cancer classification
│   ├── Card_Fraud/         # Credit card fraud detection
│   ├── Digits/             # Handwritten digit recognition
│   ├── Flags/              # Country flag analysis
│   ├── Honey/              # Honey production forecasting
│   ├── Medical_Insurance/  # Insurance cost estimation
│   ├── Raisins/            # Raisin variety classification
│   └── income_class/       # Income bracket prediction
│
├── models/                  # Trained model artifacts (.pkl, .joblib)
├── scripts/                 # Training and utility scripts
├── .github/workflows/       # CI/CD automation
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── LICENSE                 # MIT License

```

---

## 🏥 Cancer Detection System

**Problem Statement:** Binary classification of breast tumors (Malignant vs. Benign) to assist in early cancer diagnosis.

**Business Impact:** Early detection significantly improves survival rates. This system achieves high accuracy while minimizing false negatives (missing cancer cases).

### 📊 Performance Metrics
- **Accuracy:** 96.7%
- **Recall (Malignant):** 100% *(No false negatives!)*
- **Precision:** 94.2%
- **F1-Score:** 0.97
- **ROC-AUC:** 0.989

### 🛠️ Technical Stack
- **Algorithms:** Random Forest, SVM, Logistic Regression, KNN
- **Feature Engineering:** Normalization, dimensionality reduction (PCA)
- **Deployment:** FastAPI REST API
- **Testing:** pytest with 85%+ coverage
- **Data:** Wisconsin Breast Cancer Dataset (569 samples, 30 features)

### 🎯 Key Features
- ✅ Ensemble voting classifier for robust predictions
- ✅ Zero false negatives (critical for cancer screening)
- ✅ Production-ready API with request validation
- ✅ Comprehensive unit tests
- ✅ Model versioning and artifact management

**[View Project Details →](./f/Cancer/README.MD)** | **[View Production Code →](./src/cancer/)** | **[API Docs →](./api/cancer_api.py)**

---

## 💳 Fraud Detection

**Problem Statement:** Detect fraudulent financial transactions in real-time to prevent monetary losses.

### 🛠️ Technical Approach
- **Feature Engineering:** Transaction ratios, balance differentials, transaction type encoding
- **Class Imbalance Handling:** SMOTE oversampling, class weights
- **Model:** Logistic Regression (baseline), designed for XGBoost upgrade
- **Evaluation:** Precision-Recall curves, confusion matrix, cost-sensitive metrics

### 📊 Dataset Characteristics
- Highly imbalanced (fraud is rare: <1% of transactions)
- Time-series features (transaction steps)
- Multiple transaction types (PAYMENT, TRANSFER, CASH_OUT)

**[View Project Details →](./f/Card_Fraud/readme.md)**

---

## ✍️ Handwritten Digit Recognition

**Problem Statement:** Optical recognition of handwritten digits (0-9) for automated document processing.

### 🛠️ Technical Approach
- **Dimensionality Reduction:** PCA for visualization and feature compression
- **Models Compared:** SVM, Random Forest, MLPClassifier (Neural Network)
- **Hyperparameter Tuning:** GridSearchCV for optimal parameters
- **Dataset:** UCI ML hand-written digits (1,797 samples, 8x8 images)

### 📊 Best Model Performance
- **Algorithm:** SVM with RBF kernel
- **Accuracy:** ~98%
- **Confusion Matrix Analysis:** Detailed digit-pair error patterns

**[View Project Details →](./f/Digits/readme.md)**

---

## 🍯 Honey Production Forecasting

**Problem Statement:** Predict future honey production trends across U.S. states to assist agricultural planning.

### 🛠️ Technical Approach
- **Model:** Linear Regression (baseline)
- **Feature Engineering:** Year-over-year percentage change
- **Data Aggregation:** State-level and national trend analysis
- **Evaluation:** MSE, R-squared, residual analysis

### 📊 Key Insights
- Identified declining production trends in key states
- Seasonal and economic factor correlations
- Multi-year forecasting capabilities

**[View Project Details →](./f/Honey/readme.md)**

---

## 🏴 World Flags Classification

**Problem Statement:** Predict country characteristics based on flag features (colors, symbols, patterns).

### 🛠️ Technical Approach
- **Data Source:** UCI ML Repository (194 countries, 30 features)
- **Models:** Decision Trees, Random Forests, SVM, Neural Networks
- **Feature Types:** Numerical (colors, area) and categorical (symbols, religion, language)
- **Evaluation:** Cross-validation, classification reports

**[View Project Details →](./f/Flags/readme.md)**

---

## 🍇 Raisin Classification

**Problem Statement:** Classify raisin varieties using physical measurements for quality control.

### 🛠️ Technical Approach
- **Algorithms:** Clustering and supervised classification
- **Features:** Size, shape, color characteristics
- **Application:** Automated agricultural sorting

**[View Project Details →](./f/Raisins/readme.md)**

---

## 💰 Income Prediction

**Problem Statement:** Predict whether individuals earn above or below $50K based on demographic features.

### 🛠️ Technical Approach
- **Data:** Census income dataset
- **Features:** Age, education, occupation, work hours, marital status
- **Models:** Classification algorithms with feature importance analysis
- **Evaluation:** Accuracy, precision, recall, fairness metrics

**[View Project Details →](./f/income_class/readme.md)**

---

## 🏥 Medical Insurance Calculator

**Problem Statement:** Estimate medical insurance costs based on individual health and demographic factors.

### 🛠️ Technical Approach
- **Design Pattern:** Object-Oriented Programming with Enums
- **Features:** Age, BMI, smoking status, number of children
- **Validation:** Input validation, error handling
- **Code Quality:** Type safety, clean architecture

### 💡 Software Engineering Highlights
This project showcases **professional Python development**:
- ✅ Enum types for type safety
- ✅ Data validation with custom setters
- ✅ BMI calculation encapsulation
- ✅ Comprehensive error handling

**[View Project Details →](./f/Medical_Insurance/readme.md)** | **[View Code →](./f/Medical_Insurance/medical_insurance_.py)**

---

## 🛠️ Technical Skills Demonstrated

### Machine Learning
- **Supervised Learning:** Classification (Binary & Multi-class), Regression
- **Unsupervised Learning:** Clustering, PCA
- **Time Series:** Trend analysis, Forecasting
- **Imbalanced Data:** SMOTE, Class weights, Cost-sensitive learning
- **Model Selection:** Cross-validation, Hyperparameter tuning (GridSearchCV)
- **Evaluation:** ROC-AUC, Precision-Recall, Confusion matrices

### Software Engineering
- **Architecture:** Modular design, OOP principles, Separation of concerns
- **API Development:** FastAPI, RESTful design, Pydantic validation
- **Testing:** pytest, Unit tests, Integration tests, Coverage >80%
- **Code Quality:** Type hints, Docstrings, PEP 8, Black formatting
- **CLI Tools:** Click framework, argument parsing
- **Version Control:** Git, Professional commit messages

### MLOps & Deployment
- **Model Serialization:** joblib, pickle
- **Experiment Tracking:** MLflow integration
- **CI/CD:** GitHub Actions, Automated testing
- **Containerization:** Docker-ready (in progress)
- **Documentation:** Comprehensive READMEs, API docs, Code comments

### Data Science Stack
```python
# Core ML Libraries
numpy, pandas, scikit-learn

# Visualization
matplotlib, seaborn

# Deep Learning (planned)
pytorch, tensorflow

# API & Web
fastapi, uvicorn, pydantic

# Testing & Quality
pytest, flake8, black, mypy
```

---

## 📈 Development Roadmap

### ✅ Completed
- [x] 8 diverse ML projects across multiple domains
- [x] Professional repository structure
- [x] Comprehensive documentation
- [x] Production code for Cancer Detection
- [x] REST API implementation
- [x] Unit testing framework
- [x] Requirements management
- [x] MIT License

### 🚧 In Progress
- [ ] Docker containerization
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Streamlit demo applications
- [ ] Advanced algorithms (XGBoost, Prophet)
- [ ] Model interpretability (SHAP values)

### 🔮 Planned
- [ ] Kubernetes deployment
- [ ] Model monitoring and drift detection
- [ ] Feature store implementation
- [ ] A/B testing framework
- [ ] AutoML pipeline

---

## 🎓 Learning Journey

This portfolio represents my growth in:
- **Machine Learning:** From basic models to ensemble methods and production systems
- **Software Engineering:** From notebooks to tested, modular, API-driven applications
- **MLOps:** Understanding the full ML lifecycle beyond just model training
- **Domain Knowledge:** Applying ML to healthcare, finance, agriculture, and more

---

## 📊 Project Statistics

- **Total Projects:** 8
- **Production APIs:** 1 (expanding)
- **Lines of Code:** 10,000+
- **Test Coverage:** 85%+ (production projects)
- **Datasets Processed:** 8+
- **Models Trained:** 20+
- **Algorithms Implemented:** 15+

---

## 🤝 Contact & Collaboration

**GitHub:** [@hubertdomagalaa](https://github.com/hubertdomagalaa)  
**Email:** hubert.domagala@example.com *(Update with your actual email)*

💼 **Open to opportunities in:**
- Machine Learning Engineer roles
- Data Scientist positions with ML engineering focus
- MLOps and production ML systems
- Collaborative open-source ML projects

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Datasets:** UCI Machine Learning Repository, Kaggle, sklearn built-in datasets
- **Libraries:** scikit-learn, FastAPI, pytest, and the entire Python data science ecosystem
- **Inspiration:** Production ML best practices from industry leaders

---

⭐ **If you find this portfolio valuable, please consider starring the repository!**

*Last updated: January 2026*
