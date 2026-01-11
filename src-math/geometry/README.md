# Geometry Module

Hyperbolic geometry operations for the Poincaré ball model.

## Purpose

This module provides geoopt-backed implementations of hyperbolic geometry operations, enabling:
- Stable numerical operations on the Poincaré ball manifold
- Riemannian optimization (gradient descent respecting curvature)
- Holographic boundary projections for the p-adic framework

## Core Operations

### Poincaré Ball Operations

```python
from src.geometry import (
    get_manifold,
    poincare_distance,
    project_to_poincare,
    exp_map_zero,
    log_map_zero,
    mobius_add,
)

# Get the Poincaré ball manifold (curvature=-1 by default)
manifold = get_manifold(curvature=-1.0)

# Project points to Poincaré ball (ensures ||x|| < 1)
x_projected = project_to_poincare(x, max_norm=0.95)

# Compute hyperbolic distance between points
dist = poincare_distance(x, y, curvature=-1.0)

# Exponential map from origin (tangent space -> manifold)
x_on_manifold = exp_map_zero(v, curvature=-1.0)

# Logarithmic map to origin (manifold -> tangent space)
v_tangent = log_map_zero(x, curvature=-1.0)

# Möbius addition (hyperbolic addition)
z = mobius_add(x, y, curvature=-1.0)
```

### Distance Matrix

```python
from src.geometry import poincare_distance_matrix

# Compute pairwise hyperbolic distances
D = poincare_distance_matrix(embeddings, curvature=-1.0)
```

### Manifold Parameters

```python
from src.geometry import ManifoldParameter, create_manifold_parameter

# Create a learnable parameter on the Poincaré ball
param = create_manifold_parameter(shape=(batch, dim), curvature=-1.0)
```

## Riemannian Optimization

```python
from src.geometry import RiemannianAdam, get_riemannian_optimizer

# Create Riemannian Adam optimizer
optimizer = get_riemannian_optimizer(
    model.parameters(),
    lr=1e-3,
    optimizer_type="radam"
)

# Or directly
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
```

## Holographic Extensions

The holographic module provides boundary/bulk correspondence:

```python
from src.geometry import (
    HolographicPoincareManifold,
    HolographicProjection,
    HolographicLoss,
    BoundaryPoint,
)

# Create holographic manifold with boundary
manifold = HolographicPoincareManifold(curvature=-1.0)

# Project from boundary (radius=1) to bulk
projection = HolographicProjection(curvature=-1.0)
bulk_point = projection(boundary_point)

# Holographic loss encourages boundary organization
loss = HolographicLoss()(embeddings)
```

## Files

| File | Description |
|------|-------------|
| `poincare.py` | Core Poincaré ball operations via geoopt |
| `holographic_poincare.py` | Holographic boundary projections |

## Mathematical Background

The Poincaré ball model represents hyperbolic space as the unit ball:
- **Points**: ||x|| < 1
- **Curvature**: K < 0 (usually K = -1)
- **Distance**: Hyperbolic distance grows exponentially near boundary
- **Geodesics**: Circular arcs perpendicular to boundary

Key property: Hierarchical structures naturally embed in hyperbolic space with low distortion.
