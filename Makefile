# Makefile for ML Systems Portfolio
# Usage: make <target>

.PHONY: help install install-dev test lint format type-check train api demo docker clean

# Default target
help:
	@echo "ML Systems Portfolio - Available Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make install-all  Install all dependencies including demo"
	@echo ""
	@echo "Quality:"
	@echo "  make test         Run tests with coverage"
	@echo "  make lint         Run flake8 linting"
	@echo "  make format       Format code with Black"
	@echo "  make type-check   Run mypy type checking"
	@echo "  make check        Run all quality checks"
	@echo ""
	@echo "Run:"
	@echo "  make train        Train the cancer detection model"
	@echo "  make api          Start FastAPI server"
	@echo "  make demo         Run Streamlit demo"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-run   Run Docker container"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        Remove cache and build files"
	@echo "  make pre-commit   Install pre-commit hooks"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-asyncio httpx flake8 black mypy pre-commit

install-all:
	pip install -e ".[all]"

pre-commit:
	pip install pre-commit
	pre-commit install
	@echo "Pre-commit hooks installed!"

# =============================================================================
# Quality Checks
# =============================================================================

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

lint:
	flake8 src/ api/ tests/ --max-line-length=100 --extend-ignore=E203,W503

format:
	black src/ api/ tests/ --line-length=100
	isort src/ api/ tests/ --profile=black --line-length=100

type-check:
	mypy src/ --ignore-missing-imports

check: format lint type-check test
	@echo "All checks passed!"

# =============================================================================
# Run Applications
# =============================================================================

train:
	python scripts/train_cancer_model.py --model-type random_forest
	@echo "Model trained successfully!"

api:
	uvicorn api.cancer_api:app --reload --host 0.0.0.0 --port 8000

demo:
	streamlit run app.py

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t cancer-detection-api:latest .

docker-run:
	docker run -d -p 8000:8000 --name ml-api cancer-detection-api:latest
	@echo "API running at http://localhost:8000"

docker-stop:
	docker stop ml-api && docker rm ml-api

# =============================================================================
# Maintenance
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleaned up cache files!"
