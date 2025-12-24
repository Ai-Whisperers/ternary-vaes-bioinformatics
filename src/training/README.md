# Training: Hyperbolic Curriculum Learning

This module implements the specialized training loop required for Ternary VAEs.

## Key Components

### 1. `hyperbolic_trainer.py`

- **Role:** The Trainer.
- **Difference from Standard PyTorch:** Euclidean optimizers (Adam) fail on the Poincaré ball because "straight lines" are curved. This trainer uses **Riemannian Gradients** (via `geoopt` or manual projection) to update weights.

### 2. `curriculum.py` (in models, but used here)

- **Role:** The Teacher.
- **Mechanism:** Humans learn simple concepts first. Our model does too.
  - _Epoch 0-10:_ Learn short sequences (Simple tasks).
  - _Epoch 10-50:_ Learn long sequences (Complex tasks).
  - _Epoch 50+:_ Turn on the "Hyperbolic Prior" (Learn the hierarchy).

### 3. `monitor.py`

- **Role:** The Observer.
- **Metric:** Tracks the "Collapse" of the latent space. If all points bunch to the center, it stops training.

## Usage

```python
from src.training.hyperbolic_trainer import HyperbolicTrainer
trainer = HyperbolicTrainer(model, optimizer, curvature=1.0)
trainer.train(dataloader)
```
