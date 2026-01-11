# TernaryVAE Mathematical Framework v5.12.5

**Pure mathematical foundation for p-adic VAE learning with hyperbolic geometry (default)**

## Overview

This repository contains the mathematical foundation of TernaryVAE, separated from bioinformatics applications. It implements:

- **3-adic number theory**: Operations on the 19,683-element ternary space (3^9)
- **Hyperbolic geometry**: Poincaré ball embeddings with curvature c=1.0 (default)
- **Variational Autoencoders**: Dual-encoder architecture with frozen components
- **Homeostatic control**: Dynamic training orchestration for mathematical objectives

## Quick Start

```python
from src_math import TERNARY, TernaryVAEV5_11_PartialFreeze, poincare_distance

# Load mathematical foundation checkpoint
model = TernaryVAEV5_11_PartialFreeze.load_checkpoint(
    'checkpoints/tier-1/v5_12_4_best_Q.pt'
)

# Generate embeddings for ternary operations
operations = TERNARY.generate_all_operations()
embeddings = model.encode(operations)

# Compute hyperbolic distances (default geometry)
distances = poincare_distance(embeddings[0], embeddings[1], c=1.0)
```

## Mathematical Foundation

### Core Components

- `src-math/core/` - P-adic mathematics, ternary operations, tensor utilities
- `src-math/geometry/` - Hyperbolic Poincaré ball operations (default)
- `src-math/models/` - TernaryVAE architecture with homeostatic control
- `src-math/losses/` - Mathematical loss functions (p-adic geodesic, ranking)
- `src-math/training/` - Domain-agnostic training infrastructure

### TIER-1 Checkpoints

- `v5_12_4_best_Q.pt` (981K) - Current production with improved components
- `homeostatic_rich_best.pt` (421K) - Hierarchy-richness balance reference
- `v5_11_structural_best.pt` (1.4M) - Contact prediction baseline
- `v5_11_homeostasis_best.pt` (845K) - Multi-encoder orchestration

### Mathematical Properties

- **Coverage**: 100% reconstruction of all 19,683 ternary operations
- **Hierarchy**: Spearman correlation ≤ -0.8321 (mathematical ceiling)
- **Richness**: Geometric diversity ≥ 0.006 (proven achievable)
- **Hyperbolic Constraint**: All embeddings within Poincaré ball (radius < 1.0)

## Training Pipelines

### Mathematical Foundation (Recommended)
```bash
python scripts/mathematical/train_homeostatic_rich.py \
    --config configs/mathematical/v5_12_4.yaml \
    --profile mathematical_foundation
```

### Coverage Focus
```bash
python scripts/mathematical/train.py \
    --config configs/mathematical/ternary.yaml \
    --profile coverage_focused
```

### Quick Test
```bash
python scripts/mathematical/quick_train.py \
    --config configs/mathematical/ternary_fast_test.yaml
```

## Mathematical Validation

```python
from src_math.core import ComprehensiveMetrics

# Validate mathematical properties
metrics = ComprehensiveMetrics.compute(model, test_data)
assert metrics.coverage >= 0.9999  # Perfect reconstruction
assert metrics.hierarchy <= -0.832  # Approaching mathematical ceiling
assert metrics.richness >= 0.002   # Geometric diversity preserved
```

## Dependencies

Mathematical framework only (no bioinformatics):
- PyTorch ≥ 2.0
- geoopt (hyperbolic optimization)
- scipy (statistical functions)
- numpy (numerical operations)

## Architecture

**V5.12.5 Enhanced Features:**
- **Hyperbolic Default**: All operations use Poincaré ball geometry
- **Mathematical Precision**: Enhanced numerical stability
- **Improved Components**: SiLU activation, LayerNorm, Dropout
- **Enhanced Controller**: 12-dim input for comprehensive metrics
- **Validation Suite**: Mathematical property verification

**Proven Patterns:**
- Frozen encoder strategy (preserves 100% coverage)
- Homeostatic control (dynamic freeze/unfreeze)
- Hierarchy-richness balance (NOT mutually exclusive)
- Mathematical ceiling respect (-0.8321 limit)

## Research Applications

This mathematical foundation supports:
- P-adic analysis and ultrametric learning
- Hyperbolic representation learning
- Hierarchical structure discovery
- Multi-modal embedding research
- Topological data analysis

**Note**: This is the mathematical substrate. Bioinformatics applications (protein analysis, drug design) are available in the main repository branch.