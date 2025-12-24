# Task: Unit Testing & CI Overhaul

**Objective**: Establish a robust testing infrastructure to prevent regression and ensure code reliability.
**Source**: `COMPREHENSIVE_CODE_HEALTH.md` (Missing tests findings)

## High-Level Goals

- [ ] **Coverage**: Increase test coverage from <5% to >40% for core modules.
- [ ] **CI Pipeline**: Setup GitHub Actions or local hooks for automated testing.

## Detailed Tasks (Implementation)

### 1. Infrastructure Setup

- [ ] **Pytest Configuration**: Create `pytest.ini` with standard markers (unit, integration, slow).
- [ ] **Test Structure**: create `tests/` mirrors `src/`.
  - `tests/geometry/`
  - `tests/models/`
  - `tests/training/`

### 2. Core Unit Tests

- [ ] **Geometry Tests**: Test `src/geometry/poincare.py` math against `geoopt` reference. (Critical for P0_GEOMETRY).
- [ ] **Decoder Tests**: Verify shape correctness for `src/models/decoders.py`.
- [ ] **Encoder Tests**: Verify shape correctness for `src/models/encoders.py`.

### 3. Integration Tests

- [ ] **Pipeline Test**: A simplified `train_step` that runs on CPU for 1 epoch to verify the loop.
- [ ] **Data Loader Test**: Verify `bio_loader.py` (ScanPy) correctly reads `.h5ad` files.

## Deliverables

- [ ] Passing Test Suite (green build).
- [ ] Coverage Report (`htmlcov/`).
