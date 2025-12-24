# Codebase Structure Guide

This document provides a detailed breakdown of the repository structure, explaining the purpose of each directory and key file.

## Root Directory

| File/Folder        | Purpose                                                                                   |
| ------------------ | ----------------------------------------------------------------------------------------- |
| `configs/`         | **Single Source of Truth** for configuration. Contains `ternary.yaml`.                    |
| `docs/`            | Project documentation archive, legal info, plans, and analysis reports.                   |
| `experiments/`     | Sandbox for research scripts (Bioinformatics, Mathematics). **No heavy data here.**       |
| `guides/`          | User-facing documentation organized by audience (Biologists, Developers, Mathematicians). |
| `results/`         | **All outputs.** Checkpoints, logs, run history, and AlphaFold predictions.               |
| `scripts/`         | Executable scripts for training, evaluation, and benchmarking.                            |
| `src/`             | **Production Code.** The core Python package `ternary_vae`.                               |
| `tests/`           | Integration and regression tests.                                                         |
| `setup.py`         | Package installation script (v5.11.0).                                                    |
| `requirements.txt` | Python dependency list.                                                                   |
| `.env.example`     | Template for environment variables (API keys, path overrides).                            |

---

## `src/` - The Core Package

The source code follows the **Single Responsibility Principle (SRP)**.

### `src/models/`

Defines the Neural Network architectures.

- **`ternary_vae.py`**: Main entry point. Implements `TernaryVAEV5_11` (Frozen Encoder).
- **`hyperbolic_projection.py`**: Implements the trainable projection layer to Poincare ball.
- **`differentiable_controller.py`**: The StateNet controller (Meta-Learning).

### `src/losses/`

Loss functions are decoupled from models.

- **`hyperbolic_prior.py`**: Geometrically correct prior (Wrapped Normal).
- **`hyperbolic_recon.py`**: Geodesic reconstruction loss.
- **`padic_losses.py`**: 3-adic ranking losses.

### `src/training/`

Orchestration logic.

- **`hyperbolic_trainer.py`**: specialized trainer for v5.11 geometry.
- **`monitor.py`**: Unified logging (TensorBoard + Console).

### `src/metrics/`

- **`hyperbolic.py`**: Mathematics for computing Poincare distance and correlation.

---

## `scripts/` - Entry Points

Use these scripts to interact with the codebase. They wrap `src` components.

- **`scripts/train/`**: Training loops.
  - `train_ternary_v5_10.py`: The main training script (works for v5.11 too).
- **`scripts/eval/`**: Evaluation tools.
  - `evaluate_coverage.py`: Checks if all 19,683 operations are learned.
- **`scripts/benchmark/`**: Performance testing.
- **`scripts/visualization/`**: Tools to generate plots (Poincare disk visualization).

---

## `results/` - Data Management

To keep the repo clean, all heavy artifacts go here.

- `checkpoints/`: `.pt` model files.
- `logs/`: Text logs from training.
- `training_runs/`: TensorBoard event files.
- `alphafold_predictions/`: Large ZIP files from protein folding experiments.
