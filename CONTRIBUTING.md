# Contributing to Machine Learning Systems Portfolio

Thank you for considering contributing to this project! 🎉

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Python version
   - Operating system
   - Steps to reproduce
   - Expected vs actual behavior

### Suggesting Features

1. Open an issue with the "enhancement" label
2. Describe the feature and its use case
3. Include mockups/examples if applicable

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Follow code style guidelines (Black, flake8)
4. Add tests for new functionality
5. Update documentation as needed
6. Commit with clear messages: `git commit -m 'Add amazing feature'`
7. Push and open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Machine_Learning.git
cd Machine_Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=src
```

## Code Style

- **Formatter**: Black (line length 100)
- **Linter**: flake8
- **Type hints**: Required for public functions
- **Docstrings**: Google style

```bash
# Format code
black src/ api/ tests/

# Check lint
flake8 src/ api/ tests/

# Type check
mypy src/
```

## Testing

- Write unit tests for new features
- Maintain >80% coverage
- Use pytest fixtures for common setup

## Questions?

Open an issue or reach out to the maintainers.

Thank you for contributing! 🙏
