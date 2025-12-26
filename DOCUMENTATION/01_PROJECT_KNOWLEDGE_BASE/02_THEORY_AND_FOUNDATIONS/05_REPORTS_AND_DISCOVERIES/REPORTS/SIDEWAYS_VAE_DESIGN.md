# Sideways VAE: Meta-Learning for Checkpoint Exploration

**Date:** December 26, 2025
**Status:** Design Document
**Author:** Research Team

---

## Executive Summary

The **Sideways VAE** is a meta-learning architecture that explores the space of model checkpoints to discover optimal training trajectories. Instead of training forward through epochs, it learns "sideways" across checkpoint variations to identify:

1. Which weight configurations preserve coverage while improving structure
2. Optimal interpolation paths between checkpoints
3. Better initialization points for new training runs

---

## Motivation

### The Problem

Our progressive unfreeze experiments revealed a fundamental tension:

| Approach | Coverage | Distance Corr | Issue |
|----------|----------|---------------|-------|
| Frozen encoder | 100% | 0.58 | Structure limited |
| Aggressive unfreeze | 0.1% | 0.96 | Coverage destroyed |
| Tiny LR unfreeze | 59.5% | 0.93 | Still loses coverage |

**Key insight:** There may exist weight configurations that achieve both high coverage AND high structure - we just haven't found the right path to them.

### The Hypothesis

The space of model weights forms a manifold where:
- Coverage-preserving regions exist
- Structure-optimal regions exist
- The intersection (Pareto frontier) is navigable with the right guidance

A Sideways VAE can learn this manifold and guide training toward the Pareto frontier.

---

## Architecture

### Core Concept

```
                    ┌─────────────────────────────────────┐
                    │         SIDEWAYS VAE                │
                    │                                     │
Checkpoint_1 ──────►│  ┌─────────┐    ┌─────────┐        │
Checkpoint_2 ──────►│  │ Encoder │───►│ Latent  │        │
Checkpoint_3 ──────►│  │ (Ckpt)  │    │  Space  │        │
      ...    ──────►│  └─────────┘    └────┬────┘        │
Checkpoint_N ──────►│                      │              │
                    │              ┌───────┴───────┐      │
                    │              ▼               ▼      │
                    │        ┌─────────┐    ┌─────────┐  │
                    │        │ Decoder │    │ Metric  │  │
                    │        │ (Ckpt)  │    │Predictor│  │
                    │        └────┬────┘    └────┬────┘  │
                    │             │              │        │
                    │             ▼              ▼        │
                    │      New Checkpoint   [Coverage,    │
                    │                        DistCorr,    │
                    │                        RadHier]     │
                    └─────────────────────────────────────┘
```

### Components

#### 1. Checkpoint Encoder

Encodes model weights into a low-dimensional manifold:

```python
class CheckpointEncoder(nn.Module):
    """
    Encodes checkpoint weights into latent space.

    Input: Flattened model weights (or key weight matrices)
    Output: Latent vector z ∈ R^d (d << weight_dim)
    """

    def __init__(self, weight_dim: int, latent_dim: int = 32):
        super().__init__()
        # Use attention over weight blocks
        self.block_embedder = WeightBlockEmbedder()
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self.to_latent = nn.Linear(64, latent_dim * 2)  # mu, logvar

    def forward(self, weights: dict) -> tuple[Tensor, Tensor]:
        # Embed each weight block
        blocks = self.block_embedder(weights)  # [n_blocks, 64]

        # Self-attention across blocks
        attended, _ = self.attention(blocks, blocks, blocks)

        # Pool and project to latent
        pooled = attended.mean(dim=0)
        mu, logvar = self.to_latent(pooled).chunk(2, dim=-1)

        return mu, logvar
```

#### 2. Metric Predictor

Predicts performance metrics from latent position:

```python
class MetricPredictor(nn.Module):
    """
    Predicts [coverage, distance_corr, radial_hierarchy] from latent z.

    This learns the metric landscape over checkpoint space.
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # [coverage, dist_corr, rad_hier]
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)
```

#### 3. Checkpoint Decoder

Reconstructs weights from latent (for interpolation):

```python
class CheckpointDecoder(nn.Module):
    """
    Decodes latent vector back to model weights.

    Enables:
    - Checkpoint interpolation
    - Novel weight generation
    - Pareto frontier exploration
    """

    def __init__(self, latent_dim: int = 32, weight_dim: int = ...):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, weight_dim)
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)
```

---

## Training the Sideways VAE

### Data Collection

Collect checkpoints from multiple training runs with varied hyperparameters:

```python
def collect_checkpoint_dataset():
    """
    Collect (weights, metrics) pairs from all training runs.
    """
    dataset = []

    for run_dir in Path("sandbox-training/checkpoints").iterdir():
        for ckpt_path in run_dir.glob("*.pt"):
            ckpt = torch.load(ckpt_path, map_location="cpu")

            # Extract key weights (not all weights - too high dimensional)
            weights = extract_key_weights(ckpt["model_state_dict"])

            # Get metrics
            metrics = ckpt.get("metrics", {})

            dataset.append({
                "weights": weights,
                "coverage": metrics.get("coverage", 0),
                "distance_corr": metrics.get("distance_corr_A", 0),
                "radial_hier": metrics.get("radial_corr_A", 0),
            })

    return dataset
```

### Loss Function

```python
def sideways_vae_loss(
    weights_pred: Tensor,
    weights_true: Tensor,
    metrics_pred: Tensor,
    metrics_true: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 0.1
) -> Tensor:
    """
    Combined loss for Sideways VAE.

    Components:
    1. Weight reconstruction (optional, can be sparse)
    2. Metric prediction accuracy
    3. KL divergence for latent regularization
    """
    # Metric prediction loss (primary objective)
    metric_loss = F.mse_loss(metrics_pred, metrics_true)

    # Weight reconstruction (sparse, only key layers)
    recon_loss = F.mse_loss(weights_pred, weights_true)

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return metric_loss + 0.1 * recon_loss + beta * kl_loss
```

---

## Applications

### 1. Pareto Frontier Discovery

Find checkpoints that optimize the coverage-structure tradeoff:

```python
def find_pareto_frontier(sideways_vae, n_samples=1000):
    """
    Sample latent space to find Pareto-optimal configurations.
    """
    z_samples = torch.randn(n_samples, latent_dim)
    metrics = sideways_vae.predict_metrics(z_samples)

    # Find Pareto frontier
    pareto_mask = is_pareto_efficient(metrics)  # [coverage, dist_corr]

    return z_samples[pareto_mask], metrics[pareto_mask]
```

### 2. Checkpoint Interpolation

Create smooth paths between checkpoints:

```python
def interpolate_checkpoints(
    sideways_vae,
    ckpt_frozen: dict,  # 100% coverage, 0.58 dist_corr
    ckpt_unfrozen: dict,  # 60% coverage, 0.93 dist_corr
    n_steps: int = 10
) -> list[dict]:
    """
    Interpolate in latent space, decode to weight space.

    May discover intermediate configurations with better tradeoffs.
    """
    z_frozen = sideways_vae.encode(ckpt_frozen)
    z_unfrozen = sideways_vae.encode(ckpt_unfrozen)

    interpolated = []
    for alpha in torch.linspace(0, 1, n_steps):
        z_interp = (1 - alpha) * z_frozen + alpha * z_unfrozen
        weights = sideways_vae.decode(z_interp)
        metrics = sideways_vae.predict_metrics(z_interp)

        interpolated.append({
            "weights": weights,
            "predicted_metrics": metrics,
            "alpha": alpha.item()
        })

    return interpolated
```

### 3. Guided Training Initialization

Find optimal starting points for new training runs:

```python
def find_optimal_initialization(
    sideways_vae,
    target_coverage: float = 0.95,
    target_dist_corr: float = 0.85
) -> dict:
    """
    Search latent space for initialization that likely achieves targets.
    """
    # Optimize in latent space
    z = torch.randn(1, latent_dim, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=0.01)

    for _ in range(1000):
        metrics = sideways_vae.predict_metrics(z)
        coverage, dist_corr, _ = metrics[0]

        # Loss: distance from targets
        loss = (coverage - target_coverage)**2 + (dist_corr - target_dist_corr)**2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Decode to weights
    return sideways_vae.decode(z.detach())
```

### 4. Training Trajectory Prediction

Predict where training will go before running:

```python
def predict_training_trajectory(
    sideways_vae,
    initial_ckpt: dict,
    hyperparams: dict,
    n_epochs: int = 100
) -> list[Tensor]:
    """
    Predict metric trajectory given hyperparameters.

    Requires training with (ckpt, hyperparams, next_ckpt) tuples.
    """
    z_current = sideways_vae.encode(initial_ckpt)
    trajectory = [sideways_vae.predict_metrics(z_current)]

    for epoch in range(n_epochs):
        # Predict next latent position given hyperparams
        z_next = sideways_vae.predict_next_z(z_current, hyperparams, epoch)
        trajectory.append(sideways_vae.predict_metrics(z_next))
        z_current = z_next

    return trajectory
```

---

## Implementation Plan

### Phase 1: Data Collection (1-2 days)

1. Extract weights from all existing checkpoints (~100+)
2. Normalize and compress weight representations
3. Create (weights, metrics) dataset

### Phase 2: Basic Sideways VAE (2-3 days)

1. Implement CheckpointEncoder
2. Implement MetricPredictor
3. Train on collected data
4. Validate metric predictions

### Phase 3: Interpolation & Exploration (2-3 days)

1. Implement CheckpointDecoder
2. Test checkpoint interpolation
3. Find Pareto frontier candidates
4. Validate interpolated checkpoints by running them

### Phase 4: Training Guidance (3-5 days)

1. Collect training trajectory data
2. Add trajectory prediction
3. Implement guided initialization
4. Integrate with training pipeline

---

## Key Innovations

### 1. Weight-Space Meta-Learning

Unlike hyperparameter tuning (which operates on scalars), Sideways VAE learns over the full weight manifold.

### 2. Pareto-Aware Exploration

Explicitly optimizes for multi-objective tradeoffs (coverage vs structure).

### 3. Interpolation-Based Discovery

May find "shortcut" weight configurations not reachable by gradient descent.

### 4. Predictive Training

Simulates training outcomes before expensive GPU runs.

---

## Expected Outcomes

| Capability | Benefit |
|------------|---------|
| Metric prediction | Skip bad hyperparameter combinations |
| Checkpoint interpolation | Find intermediate sweet spots |
| Pareto discovery | Identify best coverage/structure tradeoffs |
| Training guidance | Better initialization for new runs |

---

## Technical Requirements

- ~100 checkpoints for initial training
- GPU for encoder/decoder training
- ~1 day to train basic Sideways VAE
- Integration with existing training pipeline

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| High-dimensional weights | Compress to key layers only |
| Overfitting to seen checkpoints | Regularization, diverse training runs |
| Interpolation instability | Validate decoded checkpoints |
| Computational cost | Use lightweight encoders |

---

*This design enables a fundamentally new approach to training optimization: instead of blindly exploring hyperparameter space, we learn the structure of the checkpoint manifold and navigate it intelligently.*
