# Factories Module

Factory patterns for model creation and configuration.

## Purpose

This module provides factory patterns for instantiating complex model components with consistent configuration:
- Dependency injection for testing
- Configuration-driven model creation
- Consistent component initialization

## TernaryModelFactory

Create TernaryVAE models and components:

```python
from src.factories import TernaryModelFactory

# Configuration dictionary
config = {
    "latent_dim": 16,
    "hidden_dim": 64,
    "max_radius": 0.95,
    "curvature": 1.0,
    "use_dual_projection": False,
    "use_controller": False,
}

# Create complete model
model = TernaryModelFactory.create_model(config)
```

### Component Creation

Create individual components for custom assembly:

```python
from src.factories import TernaryModelFactory

config = {"latent_dim": 16, "hidden_dim": 64}

# Create individual components
components = TernaryModelFactory.create_components(config)

encoder_A = components["encoder_A"]
encoder_B = components["encoder_B"]
decoder_A = components["decoder_A"]
projection = components["projection"]
controller = components["controller"]  # None if not configured
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `latent_dim` | int | 16 | Latent space dimension |
| `hidden_dim` | int | 64 | Hidden layer dimension |
| `max_radius` | float | 0.95 | Maximum Poincaré ball radius |
| `curvature` | float | 1.0 | Hyperbolic curvature |
| `use_dual_projection` | bool | False | Use dual projection layer |
| `use_controller` | bool | False | Add differentiable controller |
| `projection_layers` | int | 1 | Number of projection layers |
| `projection_dropout` | float | 0.0 | Dropout in projection |
| `learnable_curvature` | bool | False | Learn curvature parameter |

## Usage with YAML Config

```python
from src.config import load_config
from src.factories import TernaryModelFactory

# Load config from YAML
config = load_config("configs/ternary.yaml")

# Create model from config
model = TernaryModelFactory.create_model(config.model)
```

## Testing with Factories

Factories enable easy testing with mock components:

```python
from src.factories import TernaryModelFactory
from unittest.mock import MagicMock

# Create model with test configuration
test_config = {
    "latent_dim": 8,  # Smaller for faster tests
    "hidden_dim": 16,
}
model = TernaryModelFactory.create_model(test_config)

# Or inject mock components
components = TernaryModelFactory.create_components(test_config)
components["encoder_A"] = MagicMock()  # Mock for testing
```

## Files

| File | Description |
|------|-------------|
| `model_factory.py` | TernaryModelFactory implementation |

## Design Principles

1. **Single Responsibility**: Factory only creates objects, doesn't configure training
2. **Configuration-Driven**: All options come from config dict
3. **Testable**: Easy to inject mocks or test configurations
4. **Extensible**: Add new factory methods for new model types
