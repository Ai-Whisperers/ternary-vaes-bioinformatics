# Ternary VAE Mathematical Framework

A pure mathematical framework for training Variational Autoencoders on 3-adic (p-adic) hierarchical structures over the complete space of 19,683 ternary operations (3^9).

## Overview

This framework provides the mathematical substrate for learning hyperbolic embeddings that preserve p-adic structure in ternary operation spaces. It implements a dual-encoder system where mathematical hierarchy and coverage objectives are balanced through homeostatic control mechanisms.

### Key Mathematical Concepts

- **3-adic Valuation**: Mathematical ordering based on highest power of 3 dividing operation indices
- **Hyperbolic Geometry**: Poincaré ball embeddings with curvature-aware distance metrics
- **Dual-Encoder Architecture**: VAE-A (coverage) and VAE-B (hierarchy) with complementary learning
- **P-adic Geodesic Loss**: Unified objective aligning embedding distances with mathematical structure

## Quick Start

### Installation

```bash
# Clone and install
git clone <repository-url>
cd ternary-vae-framework
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Basic Training

```python
from src.models import TernaryVAEV5_11_PartialFreeze
from src.training import TernaryVAETrainer
from src.data import generate_all_ternary_operations

# Initialize model
model = TernaryVAEV5_11_PartialFreeze(
    latent_dim=16,
    hidden_dim=64,
    use_controller=True,
    use_dual_projection=True
)

# Generate complete ternary operation dataset
operations = generate_all_ternary_operations()  # (19683, 9)

# Train with mathematical configuration
trainer = TernaryVAETrainer(
    model=model,
    config_path="configs/mathematical_framework.yaml"
)

trainer.train(operations, epochs=100)
```

### Mathematical Evaluation

```python
from src.core import TERNARY
from src.core.metrics import compute_coverage, compute_hierarchy, compute_richness
from src.geometry import poincare_distance

# Evaluate learned structure
with torch.no_grad():
    output = model(operations, compute_control=False)

    # Extract hyperbolic embeddings
    z_hyp_B = output['z_B_hyp']  # Use VAE-B for hierarchy

    # Compute mathematical metrics
    coverage = compute_coverage(output['logits'], operations)

    # Compute hierarchy (target: negative correlation)
    valuations = TERNARY.valuation(torch.arange(19683))
    origin = torch.zeros_like(z_hyp_B)
    radii = poincare_distance(z_hyp_B, origin, c=1.0)
    hierarchy = spearmanr(valuations.cpu(), radii.cpu())[0]

    # Compute richness (geometric diversity)
    richness = compute_richness(z_hyp_B, valuations)

    print(f"Coverage: {coverage:.1%}")
    print(f"Hierarchy: {hierarchy:.3f} (target: -0.82 to -1.0)")
    print(f"Richness: {richness:.6f} (higher = more structure)")
```

## Architecture

### Core Components

| Module | Description | Key Files |
|--------|-------------|-----------|
| **`src.core`** | Mathematical foundations | `ternary.py`, `padic_math.py`, `metrics.py` |
| **`src.geometry`** | Hyperbolic operations | `poincare.py` |
| **`src.models`** | VAE architectures | `ternary_vae.py`, `homeostasis.py` |
| **`src.losses`** | Mathematical objectives | `padic_geodesic.py`, `dual_vae_loss.py` |
| **`src.training`** | Training orchestration | `trainer.py`, `optimizations.py` |

### Dual-Encoder System

```
Input: (batch, 9) ternary operations
  ↓
┌─────────────────────────────────────┐
│ FROZEN ENCODERS (preserves coverage) │
│ ├─ VAE-A: mu_A, logvar_A           │
│ └─ VAE-B: mu_B, logvar_B           │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ HYPERBOLIC PROJECTIONS (trainable)  │
│ ├─ z_A → z_hyp_A (Euclidean→Poincaré) │
│ └─ z_B → z_hyp_B (geometry learning) │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ DIFFERENTIABLE CONTROLLER           │
│ Input: batch statistics             │
│ Output: loss weights (tensor-valued) │
└─────────────────────────────────────┘
```

## Mathematical Metrics

### Primary Metrics

| Metric | Range | Target | Interpretation |
|--------|-------|--------|----------------|
| **Coverage** | [0, 100%] | 100% | Complete ternary operation reconstruction |
| **Hierarchy_B** | [-1, +1] | -0.82 to -1.0 | v=0 at edge, v=9 at center (VAE-B) |
| **Richness** | [0, ∞) | 0.006+ | Geometric diversity preservation |
| **Q-metric** | [0, ∞) | 1.8+ | Structure quality: dist_corr + 1.5×\|hierarchy\| |

### Mathematical Limits

- **Hierarchy Ceiling**: -0.8321 (mathematical limit due to v=0 containing 66.7% of samples)
- **Coverage Requirement**: 100% for complete mathematical validity
- **Richness Trade-off**: Higher richness preserves geometric structure but may reduce hierarchy

## Configuration

The framework uses YAML configurations for mathematical training:

```yaml
# configs/mathematical_framework.yaml
model:
  name: TernaryVAEV5_11_PartialFreeze
  latent_dim: 16
  use_controller: true
  use_dual_projection: true

loss:
  rich_hierarchy:
    hierarchy_weight: 5.0      # P-adic structure importance
    coverage_weight: 1.0       # Complete space coverage
    richness_weight: 2.0       # Geometric diversity
    separation_weight: 3.0     # Valuation level separation

training:
  epochs: 100
  batch_size: 512
  use_stratified: true         # Ensure high-valuation representation
  high_v_budget_ratio: 0.25
```

## Advanced Usage

### Custom Loss Functions

```python
from src.losses import PAdicGeodesicLoss, RichHierarchyLoss

# P-adic geodesic alignment
geodesic_loss = PAdicGeodesicLoss(
    curvature=1.0,
    max_target_distance=3.0,
    n_pairs=2000
)

# Hierarchical structure preservation
hierarchy_loss = RichHierarchyLoss(
    hierarchy_weight=5.0,
    richness_weight=2.0
)
```

### Homeostatic Control

```python
from src.models import HomeostasisController

# Dynamic freeze/unfreeze management
controller = HomeostasisController(
    coverage_freeze_threshold=0.995,
    hierarchy_plateau_threshold=0.001,
    enable_annealing=True
)

# Training with homeostatic control
trainer = TernaryVAETrainer(
    model=model,
    homeostasis_controller=controller
)
```

### Custom Embeddings

```python
# Extract embeddings for downstream tasks
with torch.no_grad():
    embeddings = model.encode(operations)
    z_hyp_A = embeddings['z_A_hyp']  # Coverage-optimized
    z_hyp_B = embeddings['z_B_hyp']  # Hierarchy-optimized

    # Use hyperbolic distances for similarity
    distances = poincare_distance(z_hyp_B[0:1], z_hyp_B[1:], c=1.0)
```

## Mathematical Correctness

### Validation Protocol

1. **Coverage Verification**: All 19,683 operations must be reconstructable
2. **Hierarchy Validation**: Spearman correlation between valuation and radius
3. **Richness Assessment**: Within-level geometric diversity preservation
4. **P-adic Consistency**: Distance relationships match 3-adic metric properties

### Known Mathematical Constraints

- **Valuation Distribution**: v=0 has 13,122 samples (66.7%), creating hierarchy ceiling
- **Hyperbolic Constraints**: All embeddings must satisfy |z| < max_radius
- **Convergence Guarantees**: Homeostatic control ensures stable mathematical structure

## Dependencies

### Core Requirements
- **PyTorch ≥2.0**: Automatic differentiation and GPU acceleration
- **geoopt ≥0.5**: Riemannian optimization for hyperbolic geometry
- **NumPy ≥1.21**: Numerical computations
- **SciPy ≥1.7**: Statistical functions (Spearman correlation)

### Optional Extensions
- **matplotlib, seaborn**: Visualization of embeddings and training curves
- **tensorboard**: Training monitoring and metric tracking
- **pytest**: Test suite execution

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run mathematical correctness tests
pytest tests/test_ternary_operations.py
pytest tests/test_padic_math.py
pytest tests/test_hyperbolic_geometry.py
```

### Contributing

This is a pure mathematical framework. Contributions should:

1. Preserve mathematical correctness
2. Maintain numerical stability
3. Follow p-adic and hyperbolic geometry principles
4. Include appropriate mathematical validation tests

## License

**Software**: PolyForm Non-Commercial 1.0.0 (academic/research use)
**Research Outputs**: CC-BY-4.0 (free reuse with attribution)

## Citation

```bibtex
@software{ternary_vae_framework,
  title={Ternary VAE Mathematical Framework},
  author={{AI Whisperers}},
  year={2026},
  url={https://github.com/Ai-Whisperers/ternary-vae-framework},
  note={Mathematical substrate for 3-adic hyperbolic VAEs}
}
```

---

**Framework Version**: 1.0.0 | **Updated**: 2026-01-08 | **Mathematical Focus**: Pure 3-adic structure learning