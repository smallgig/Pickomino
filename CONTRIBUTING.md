# Contributing to Pickomino-Env

## Development Setup

```bash
git clone https://github.com/smallgig/Pickomino.git
cd Pickomino
pip install -e ".[dev]"
pre-commit install
```

## Before Submitting a Pull Request

Run pre-commit run --all-files
Ensure tests pass: pytest
Add tests for new features (maintain 95%+ coverage)

## Code Style

Ruff, black, pylint, mypy (strict mode)
Google-style docstrings
Type hints required.

## Questions?

Open an issue on GitHub.
