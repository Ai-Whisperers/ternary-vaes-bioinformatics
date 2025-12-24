# Add New Feature (Bio-Protocol)

Add a new bioinformatics module or experiment following rigorous scientific standards.

## When to Use

- **Use for**: New VAE variants, hyperbolic layers, or experiment protocols.
- **Critical for**: Reproducibility, tensor shape consistency, and geometric validity.

## Good Pattern - Scientific Protocol Format

Before implementing, document the feature using this structure:

```markdown
## Hypothesis / Goal

As a [researcher], I want to [implement X] to test [hypothesis Y].

## Mathematical Specification

**Latent Space**: [Euclidean | Poincare | Lorentz]
**Manifold Curvature**: [Fixed (-1) | Learnable]
**Metric Tensor**: $g_{ij} = ...$

## Implementation Plan

**Scenario 1: Forward Pass**

- Input: `[Batch, Channels, SeqLen]`
- Transformation: `ExpMap(x)` -> Hyperbolic Space
- Output: `[Batch, LatentDim]`

**Scenario 2: Edge Cases**

- Numerical Instability: Implement `clamp(min=1e-5)`
- Nan Gradients: Use `geoopt.Manifold.retr`

## Technical Requirements

- [ ] New `nn.Module` in `src/models/`
- [ ] Unit Test in `tests/unit/` (Gradient check required)
- [ ] Integration Test (Smoke run)
```

## Implementation Steps

### 1. PyTorch Module

Create in `src/models/layers/feature_name.py`:

```python
import torch
import torch.nn as nn
from geoopt.manifolds import PoincareBall

class HyperbolicFeature(nn.Module):
    def __init__(self, in_features, out_features, c=1.0):
        super().__init__()
        self.manifold = PoincareBall(c=c)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ALWAYS validate shapes
        assert x.dim() == 3, f"Expected 3D input, got {x.shape}"

        # Euclidean -> Hyperbolic
        x_euc = self.linear(x)
        x_hyp = self.manifold.expmap0(x_euc)

        return x_hyp
```

### 2. Tests

Create in `tests/unit/test_feature_name.py`:

```python
import pytest
import torch
from src.models.layers.feature_name import HyperbolicFeature

def test_hyperbolic_feature_shape():
    layer = HyperbolicFeature(10, 5)
    x = torch.randn(32, 1, 10) # Batch x Ch x Seq
    out = layer(x)

    assert out.shape == (32, 1, 5)
    assert not torch.isnan(out).any(), "Output contains NaNs"

def test_gradient_flow():
    layer = HyperbolicFeature(10, 5)
    x = torch.randn(32, 1, 10, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert layer.linear.weight.grad is not None
```

## Final Must-Pass Checklist

- [ ] Mathematical spec documented
- [ ] Tensor shapes validated with `assert` or Type Hints
- [ ] `NaN` checks implemented for hyperbolic ops
- [ ] Gradient flow verified
- [ ] Reproducibility seed set in tests
