# Config Module

Centralized configuration management for the Ternary VAE framework.

## Purpose

This module provides:
1. **Constants**: All magic numbers and defaults in one place
2. **Schema**: Typed, validated configuration classes
3. **Loader**: YAML + environment variable configuration loading

## Loading Configuration

```python
from src.config import load_config, save_config

# Load from YAML file
config = load_config("configs/training.yaml")

# Access configuration
print(config.training.epochs)
print(config.geometry.curvature)

# Save configuration
save_config(config, "configs/output.yaml")
```

## Configuration Schema

### Training Configuration

```python
from src.config import TrainingConfig

config = TrainingConfig(
    epochs=500,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-5,
    patience=50,        # Early stopping patience
    grad_clip=1.0,      # Gradient clipping
)
```

### Geometry Configuration

```python
from src.config import GeometryConfig

config = GeometryConfig(
    curvature=-1.0,     # Poincaré ball curvature
    max_radius=0.95,    # Maximum embedding radius
    latent_dim=16,      # Latent space dimension
)
```

### Loss Weights

```python
from src.config import LossWeights

weights = LossWeights(
    reconstruction=1.0,
    kl_divergence=0.1,
    geodesic=0.5,
    radial=0.3,
)
```

### VAE Configuration

```python
from src.config import VAEConfig

config = VAEConfig(
    latent_dim=16,
    hidden_dim=64,
    use_dual_projection=False,
    learnable_curvature=False,
)
```

## Constants

### Numerical Stability

```python
from src.config.constants import (
    EPSILON,       # 1e-8 (general stability)
    EPSILON_LOG,   # 1e-10 (log operations)
    EPSILON_NORM,  # 1e-6 (normalization)
)
```

### Geometry Defaults

```python
from src.config.constants import (
    DEFAULT_CURVATURE,    # -1.0
    DEFAULT_MAX_RADIUS,   # 0.95
    DEFAULT_LATENT_DIM,   # 16
    HYPERBOLIC_CURVATURE, # -1.0
    HYPERBOLIC_MAX_NORM,  # 0.95
)
```

### Training Defaults

```python
from src.config.constants import (
    DEFAULT_EPOCHS,         # 500
    DEFAULT_BATCH_SIZE,     # 256
    DEFAULT_LEARNING_RATE,  # 1e-3
    DEFAULT_WEIGHT_DECAY,   # 1e-5
    DEFAULT_PATIENCE,       # 50
    DEFAULT_GRAD_CLIP,      # 1.0
)
```

### Ternary Space

```python
from src.config.constants import (
    TERNARY_BASE,          # 3
    N_TERNARY_DIGITS,      # 9
    N_TERNARY_OPERATIONS,  # 19683 (3^9)
    MAX_VALUATION,         # 9
)
```

## Environment Configuration

```python
from src.config import get_env_config, Environment

env = get_env_config()
print(env.environment)  # Environment.DEVELOPMENT

# Check environment
if env.environment == Environment.PRODUCTION:
    # Production-specific logic
    pass
```

## Files

| File | Description |
|------|-------------|
| `constants.py` | All magic numbers and defaults |
| `schema.py` | Typed configuration dataclasses |
| `loader.py` | YAML loading and validation |
| `environment.py` | Environment detection |

## Best Practices

1. **Use constants** instead of magic numbers in code
2. **Use schema classes** for validated configuration
3. **Load from YAML** for reproducible experiments
4. **Override with environment variables** for deployment
