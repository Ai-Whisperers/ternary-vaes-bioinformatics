# Models: The Geometric Engine

This directory contains the implementations of the Ternary VAE and its components.

## Core Architecture

### 1. `ternary_vae.py` (The Main Brain)

- **Role:** Orchestrates the Encoder, Decoder, and Geometric Projection.
- **Key Feature:** Implements the "Frozen Encoder" strategy (v5.11) where discrete ternary logic is learned first, then projected into continuous space.

### 2. `hyperbolic_projection.py` (The Geometry)

- **Role:** Maps flat Euclidean vectors into the **Poincaré Ball**.
- **Math:** Implements `exp_map` (Exponential Map) to push points onto the manifold.
- **Topological Significance:** This layer forces the model to learn hierarchies. Points near the center are "General" (e.g., "Mammal"), points near the edge are "Specific" (e.g., "Poodle").

### 3. `differentiable_controller.py` (The Navigator)

- **Role:** The "StateNet". It learns to navigate the hyperbolic space over time.
- **Feature:** Instead of jumping telepathically, it traces a continuous path (trajectory) between states. This models **Evolutionary Time**.

### 4. `homeostasis.py` (The Regulator)

- **Role:** Enforces the "Regenerative Axis".
- **Feature:** A specialized loss function that penalizes deviation from the "Healthy" submanifold.

---

## Usage Example

```python
from src.models.ternary_vae import TernaryVAE

# Initialize model with Hyperbolic Geometry
model = TernaryVAE(
    latent_dim=16,
    manifold='poincare',  # Activate hyperbolic mode
    curvature=1.0         # Flatness of the disk
)

# Forward pass
reconstruction, latent_ops = model(sequence_data)
```
