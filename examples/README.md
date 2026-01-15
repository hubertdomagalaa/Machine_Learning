# Examples

This directory contains example scripts demonstrating how to use the ML Systems Portfolio.

## Available Examples

### 1. Quick Start (`quick_start.py`)
Simple example showing how to make predictions using the Cancer Detection system.

```bash
python examples/quick_start.py
```

### 2. Custom Model Training (`train_custom_model.py`)
Advanced example demonstrating how to train custom models with different algorithms.

```bash
# Train specific model type
python examples/train_custom_model.py --model random_forest
python examples/train_custom_model.py --model svm
python examples/train_custom_model.py --model ensemble

# Compare all models
python examples/train_custom_model.py --model compare
```

### 3. API Usage (`api_usage.py`)
Example showing how to interact with the REST API programmatically.

```bash
# First, start the API server
make api

# In another terminal, run the example
python examples/api_usage.py
```

## Prerequisites

Before running examples, ensure you have:

1. Installed dependencies:
   ```bash
   make install
   ```

2. Trained the model:
   ```bash
   make train
   ```

## Need Help?

See the main [README](../README.md) or open an [issue](https://github.com/hubertdomagalaa/Machine_Learning/issues).
