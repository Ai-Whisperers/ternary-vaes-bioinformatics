# DDG Multimodal VAE Architecture

**Doc-Type:** Module Documentation · Version 1.0 · 2026-01-29

## Overview

This module implements a multi-stage DDG (ΔΔG) prediction system that combines multiple data regimes through a multimodal VAE architecture.

### Key Innovation

Instead of training a single model on heterogeneous data, we:
1. Train **specialist VAEs** on different data regimes
2. **Fuse** their representations through attention
3. **Refine** predictions with MLP and transformer heads
4. Provide both **fuzzy** (uncertainty) and **precise** outputs

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DDG Multimodal Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │  VAE-S669   │  │VAE-ProTherm │  │  VAE-Wide   │            │
│   │  (N=669)    │  │  (N=2000+)  │  │ (N=500K+)   │            │
│   │  Benchmark  │  │ High-quality│  │  Diversity  │            │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│          │                │                │                    │
│          │   z_s669       │  z_protherm    │   z_wide          │
│          │   (16-dim)     │   (32-dim)     │   (64-dim)        │
│          └────────────────┼────────────────┘                    │
│                           │                                     │
│                    ┌──────▼──────┐                              │
│                    │Cross-Modal  │                              │
│                    │  Attention  │                              │
│                    │   Fusion    │                              │
│                    └──────┬──────┘                              │
│                           │ z_fused (128-dim)                   │
│                    ┌──────▼──────┐                              │
│                    │    MLP      │                              │
│                    │  Refiner    │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│          ┌────────────────┼────────────────┐                    │
│          │                │                │                    │
│   ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐            │
│   │ Fuzzy Head  │  │  Full-Seq   │  │Hierarchical │            │
│   │(VAE + σ)    │  │ Transformer │  │ Transformer │            │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│          │                │                │                    │
│          └────────────────┼────────────────┘                    │
│                           │                                     │
│                    ┌──────▼──────┐                              │
│                    │  Ensemble   │                              │
│                    │ (Learned    │                              │
│                    │  Weights)   │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│                    ┌──────▼──────┐                              │
│                    │ DDG ± σ     │                              │
│                    └─────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

The module is part of the ternary-vaes-bioinformatics package. No separate installation required.

```python
from src.bioinformatics import (
    DatasetRegistry,
    DDGVAE,
    MultimodalDDGVAE,
    DDGEnsemble,
    BenchmarkRunner,
)
```

## Quick Start

### 1. Load Data

```python
from src.bioinformatics.data import DatasetRegistry

# Initialize registry
registry = DatasetRegistry()

# Load S669 benchmark
s669_dataset = registry.get_dataset("s669")
print(f"S669: {len(s669_dataset)} samples")

# Load ProTherm curated
protherm_dataset = registry.get_dataset("protherm")

# Get combined dataset
combined = registry.get_combined_dataset(["protherm", "s669"])
```

### 2. Train Specialist VAE

```python
from src.bioinformatics.models import DDGVAE
from src.bioinformatics.training import DDGVAETrainer, TrainingConfig

# Create S669 specialist
model = DDGVAE.create_s669_variant(use_hyperbolic=True)

# Configure training
config = TrainingConfig(
    epochs=100,
    batch_size=32,
    learning_rate=1e-4,
)

# Train
trainer = DDGVAETrainer(
    model=model,
    train_dataset=train_ds,
    val_dataset=val_ds,
    config=config,
)
trainer.train()
```

### 3. Run Benchmark

```python
from src.bioinformatics.evaluation import BenchmarkRunner

runner = BenchmarkRunner()
result = runner.run_benchmark(
    model=model,
    X=features,
    y=ddg_values,
    dataset_name="s669_full",
    model_name="VAE-S669",
)
print(result)  # Includes comparison with Rosetta, FoldX, etc.
```

## Directory Structure

```
src/bioinformatics/
├── __init__.py
├── README.md
│
├── data/                        # Data loaders
│   ├── protherm_loader.py       # ProTherm curated (N=176+)
│   ├── s669_loader.py           # S669 benchmark (N=669)
│   ├── proteingym_loader.py     # ProteinGym (N=500K+)
│   ├── dataset_registry.py      # Unified registry
│   └── preprocessing.py         # Feature extraction
│
├── models/                      # Model architectures
│   ├── ddg_vae.py               # Base DDG VAE
│   ├── multimodal_ddg_vae.py    # Multimodal fusion
│   ├── ddg_mlp_refiner.py       # MLP refinement
│   ├── ddg_transformer.py       # Transformer heads
│   └── ddg_ensemble.py          # Ensemble + fuzzy head
│
├── training/                    # Training pipelines
│   ├── deterministic.py         # Reproducibility utilities
│   ├── train_ddg_vae.py         # Single-VAE training
│   ├── train_multimodal.py      # Multimodal training
│   └── train_transformer.py     # Transformer training
│
├── evaluation/                  # Evaluation tools
│   ├── metrics.py               # DDG-specific metrics
│   ├── cross_validation.py      # LOO and k-fold CV
│   └── benchmark_runner.py      # Benchmark suite
│
└── configs/                     # Configuration files
    ├── base.yaml                # Base configuration
    ├── vae_s669.yaml            # S669 specialist
    ├── vae_protherm.yaml        # ProTherm specialist
    ├── vae_wide.yaml            # Wide specialist
    ├── multimodal.yaml          # Multimodal fusion
    └── transformer.yaml         # Transformer heads
```

## Datasets

| Dataset | N | Description | CV Method |
|---------|--:|-------------|-----------|
| **ProTherm Curated** | 176+ | High-quality alanine scanning | LOO |
| **S669 Full** | 669 | Standard benchmark | 5-fold × 3 |
| **S669 Curated** | 52 | Selected subset | LOO |
| **ProteinGym** | 500K+ | Large-scale diverse | 5-fold |

## Expected Results

| Stage | Model | Dataset | Spearman ρ |
|-------|-------|---------|:----------:|
| 3 | VAE-S669 | S669 full | 0.40 |
| 3 | VAE-ProTherm | ProTherm | 0.90+ |
| 3 | VAE-Wide | ProteinGym | 0.55 |
| 4 | Multimodal | S669 full | 0.55+ |
| 5 | + MLP Refiner | S669 full | 0.58+ |
| 6 | + Transformers | S669 full | 0.65+ |
| 7 | Ensemble | S669 full | **0.68+** |

**Target:** Match ESM-1v (0.51) on S669, approach Rosetta (0.69).

## Memory Optimization

For RTX 3050 (6GB VRAM):

```yaml
# configs/transformer.yaml
training:
  batch_size: 4
  accumulation_steps: 8  # Effective batch = 32

hardware:
  mixed_precision: true

full_sequence:
  max_seq_len: 256  # Reduced from 512
  d_model: 128      # Reduced from 256
  use_gradient_checkpointing: true
```

## Literature Comparison

IMPORTANT: Our N=52 curated results are NOT directly comparable to N=669 benchmarks.

| Method | Spearman | Dataset | Type |
|--------|:--------:|---------|------|
| Rosetta ddg_monomer | 0.69 | N=669 | Structure |
| Mutate Everything | 0.56 | N=669 | Sequence |
| ESM-1v | 0.51 | N=669 | Sequence |
| ELASPIC-2 | 0.50 | N=669 | Sequence |
| FoldX | 0.48 | N=669 | Structure |
| **Our Ensemble** | **0.68+** | N=669 | Sequence (target) |
| Our Current | 0.52 | N=52 | Sequence |

## API Reference

See docstrings in each module for detailed API documentation.

### Key Classes

- `DDGVAE` - Base VAE for DDG prediction
- `MultimodalDDGVAE` - Multimodal fusion VAE
- `DDGEnsemble` - Final ensemble with uncertainty
- `BenchmarkRunner` - Standardized benchmarking
- `DatasetRegistry` - Unified data access

## Citation

If you use this code, please cite:

```bibtex
@software{ternary_vaes_bioinformatics,
  title = {Ternary VAE Bioinformatics: DDG Multimodal Architecture},
  year = {2026},
  author = {AI Whisperers},
  url = {https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics}
}
```
