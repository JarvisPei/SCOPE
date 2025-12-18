# Contributing to SCOPE

Thank you for your interest in contributing to SCOPE! This document provides guidelines for contributing.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/JarvisPei/SCOPE.git
cd scope

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks (optional but recommended)
pre-commit install
```

## Code Style

We use the following tools to maintain code quality:

- **Black** for formatting
- **Ruff** for linting
- **MyPy** for type checking (optional)

```bash
# Format code
black scope/

# Check linting
ruff check scope/

# Run type checking
mypy scope/
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=scope --cov-report=html

# Run specific test file
pytest tests/test_basic.py -v
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and add tests
4. **Run tests** to ensure everything passes:
   ```bash
   pytest tests/
   ruff check scope/
   ```
5. **Commit** with a clear message:
   ```bash
   git commit -m "feat: add X feature"
   ```
6. **Push** and create a Pull Request

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding/updating tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

## Adding New Features

When adding new features:

1. **Write tests first** (TDD encouraged)
2. **Update documentation** (README, docstrings)
3. **Add examples** if applicable
4. **Update CHANGELOG.md**

## Reporting Issues

When reporting bugs, please include:

- Python version
- SCOPE version (`pip show scope-optimizer`)
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn

## Questions?

- Open a [GitHub Discussion](https://github.com/JarvisPei/SCOPE/discussions)
- Check existing [Issues](https://github.com/JarvisPei/SCOPE/issues)

Thank you for contributing! 🎯

