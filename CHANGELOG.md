# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- XGBoost and Prophet model implementations
- SHAP model interpretability
- Kubernetes deployment configuration
- A/B testing framework

---

## [1.0.0] - 2026-01-14

### 🎉 Production Release - Cancer Detection System

#### Added
- **Cancer Detection API** - Production-ready FastAPI endpoint
- **Streamlit Demo** - Interactive web application for predictions
- **Complete Test Suite** - Unit tests with 85%+ coverage
- **Docker Support** - Dockerfile and docker-compose configuration
- **CI/CD Pipeline** - GitHub Actions for automated testing and building
- **Pre-commit Hooks** - Automated code quality checks
- **Professional Documentation** - README, CONTRIBUTING, SECURITY policies
- **Issue & PR Templates** - Standardized GitHub workflows
- **Makefile** - Single entry point for all development commands

#### Performance Metrics
- Accuracy: 96.7%
- Recall: 100% (zero false negatives)
- ROC-AUC: 0.989

---

## [0.9.0] - 2026-01-13

### Added
- **Fraud Detection Module** - Production code with API endpoint
- **Digits Recognition Module** - Handwritten digit classifier with API
- **Expanded Test Coverage** - Tests for all production modules
- **API Documentation** - FastAPI auto-generated docs

### Changed
- Refactored project structure to support multiple ML systems
- Unified configuration management across modules

---

## [0.8.0] - 2026-01-12

### Added
- **Modular Architecture** - Separated concerns into `src/`, `api/`, `tests/`
- **Configuration System** - Centralized config with dataclasses
- **Logging Framework** - Structured logging throughout

### Changed
- Migrated from notebooks-only to production Python modules
- Standardized code style with Black and isort

---

## [0.5.0] - 2026-01-10

### Added
- **Initial Cancer Detection** - First working classification model
- **Wisconsin Dataset** - Data loading and preprocessing
- **Basic Notebook** - Exploratory data analysis

---

## [0.1.0] - 2026-01-01

### Added
- **Repository Initialization**
- Basic project structure
- 8 exploratory notebooks:
  - Cancer classification
  - Fraud detection
  - Digit recognition
  - Honey production forecasting
  - Flag classification
  - Raisin classification
  - Income prediction
  - Medical insurance estimation

---

[Unreleased]: https://github.com/hubertdomagalaa/Machine_Learning/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/hubertdomagalaa/Machine_Learning/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/hubertdomagalaa/Machine_Learning/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/hubertdomagalaa/Machine_Learning/compare/v0.5.0...v0.8.0
[0.5.0]: https://github.com/hubertdomagalaa/Machine_Learning/compare/v0.1.0...v0.5.0
[0.1.0]: https://github.com/hubertdomagalaa/Machine_Learning/releases/tag/v0.1.0
