# Testing Infrastructure

This directory contains the test suite for the Ternary VAE for Bioinformatics project.

## Structure

```
tests/
├── unit/               # Unit tests for individual components
│   ├── test_spectral_encoder.py
│   ├── test_codon_encoder.py
│   └── ...
├── suites/             # Test suites (e.g., specific scenarios)
├── integration/        # Integration tests (e.g. training loop)
├── fixtures/           # Shared pytest fixtures
└── conftest.py         # Root conftest configuration
```

## Running Tests

We use `pytest` as the test runner.

**Run all tests:**

```bash
python -m pytest
```

**Run specific test file:**

```bash
python -m pytest tests/unit/test_spectral_encoder.py
```

**Run with coverage:**

```bash
python -m pytest --cov=src --cov-report=term-missing
```

## Configuration

- **pytest.ini**: Main configuration file. Sets python path, test discovery patterns, and markers.
- **.coveragerc**: Coverage configuration (omits init files, defines branches, etc.).

## Guidelines

1.  **Fixtures**: Place shared fixtures in `tests/conftest.py` or `tests/fixtures/`.
2.  **Markers**: Use `@pytest.mark.unit` or `@pytest.mark.integration` to categorize tests.
3.  **Imports**: Imports are handled via `pythonpath = .` in `pytest.ini`. Do not use `sys.path.append` hacks.
