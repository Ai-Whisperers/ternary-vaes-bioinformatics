# API Reference - Ternary VAE v5.10

**Version:** 5.10.1 (Pure Hyperbolic Geometry)
**Last Updated:** 2025-12-12

---

## Quick Links

- [Training](#training-module)
- [Models](#model-module)
- [Losses](#loss-module)
- [Metrics](#metrics-module)
- [Data](#data-module)
- [Artifacts](#artifacts-module)

---

## Training Module

### HyperbolicVAETrainer (v5.10)

Orchestrates pure hyperbolic training with homeostatic adaptation.

```python
from src.training import HyperbolicVAETrainer

trainer = HyperbolicVAETrainer(
    base_trainer: TernaryVAETrainer,
    model: DualNeuralVAEV5_10,
    device: str,
    config: Dict[str, Any],
    monitor: TrainingMonitor
)
```

**Methods:**

#### `train_epoch(train_loader, val_loader, epoch) -> Dict`

Execute one training epoch with hyperbolic losses.

```python
losses = trainer.train_epoch(train_loader, val_loader, epoch)
# Returns dict with: loss, cov_A, cov_B, corr_A_hyp, corr_B_hyp,
#                    corr_A_euc, corr_B_euc, mean_radius_A, mean_radius_B,
#                    hyp_kl_A, hyp_kl_B, centroid_loss, radial_loss, etc.
```

---

### TernaryVAETrainer

Base training orchestrator.

```python
from src.training import TernaryVAETrainer

trainer = TernaryVAETrainer(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: str = 'cuda'
)
```

**Methods:**

#### `train_epoch(train_loader) -> Dict`

Execute one base training epoch.

#### `validate(val_loader) -> Dict`

Run validation pass.

---

### TrainingMonitor

Unified logging and metrics tracking.

```python
from src.training import TrainingMonitor

monitor = TrainingMonitor(
    eval_num_samples: int = 1000,
    tensorboard_dir: str = 'runs',
    log_dir: str = 'logs',
    log_to_file: bool = True,
    experiment_name: str = None
)
```

**Methods:**

#### `log_epoch_summary(...)`

Log comprehensive epoch metrics.

```python
monitor.log_epoch_summary(
    epoch, total_epochs, loss,
    cov_A, cov_B,
    corr_A_hyp, corr_B_hyp,
    corr_A_euc, corr_B_euc,
    mean_radius_A, mean_radius_B,
    ranking_weight,
    coverage_evaluated=True,
    correlation_evaluated=True,
    hyp_kl_A=0, hyp_kl_B=0,
    centroid_loss=0, radial_loss=0,
    homeostatic_metrics=None
)
```

#### `log_hyperbolic_epoch(...)`

Log v5.10 hyperbolic-specific metrics.

```python
monitor.log_hyperbolic_epoch(
    epoch, corr_A_hyp, corr_B_hyp,
    corr_A_euc, corr_B_euc,
    mean_radius_A, mean_radius_B,
    ranking_weight, ranking_loss_hyp,
    radial_loss, hyp_kl_A, hyp_kl_B,
    centroid_loss, homeostatic_metrics
)
```

#### `log_batch(epoch, batch_idx, losses, ranking_weight)`

Log batch-level metrics to TensorBoard.

#### `evaluate_coverage(model, num_samples, device, vae='A') -> Tuple[int, float]`

Evaluate operation coverage.

```python
unique_ops, coverage_pct = monitor.evaluate_coverage(
    model, num_samples=1000, device='cuda', vae='A'
)
```

#### `_log(message: str)`

Unified logging to console and file.

---

### Schedulers

#### TemperatureScheduler

```python
from src.training import TemperatureScheduler

scheduler = TemperatureScheduler(
    config: Dict[str, Any],
    phase_4_start: int,
    temp_lag: int = 0
)

temp_A = scheduler.get_temperature(epoch: int, vae: str = 'A')
temp_B = scheduler.get_temperature(epoch: int, vae: str = 'B')
```

#### BetaScheduler

```python
from src.training import BetaScheduler

scheduler = BetaScheduler(
    config: Dict[str, Any],
    beta_phase_lag: float = 0.0
)

beta_A = scheduler.get_beta(epoch: int, vae: str = 'A')
beta_B = scheduler.get_beta(epoch: int, vae: str = 'B')
```

#### LearningRateScheduler

```python
from src.training import LearningRateScheduler

scheduler = LearningRateScheduler(
    lr_schedule: List[Dict[str, Any]]
)

lr = scheduler.get_lr(epoch: int)
```

---

## Model Module

### DualNeuralVAEV5_10 (v5.10)

Pure hyperbolic VAE with StateNet v4.

```python
from src.models import DualNeuralVAEV5_10

model = DualNeuralVAEV5_10(
    input_dim: int = 9,
    latent_dim: int = 16,
    rho_min: float = 0.1,
    rho_max: float = 0.7,
    lambda3_base: float = 0.3,
    lambda3_amplitude: float = 0.15,
    eps_kl: float = 0.0005,
    gradient_balance: bool = True,
    adaptive_scheduling: bool = True,
    use_statenet: bool = True,
    statenet_lr_scale: float = 0.1,
    statenet_lambda_scale: float = 0.02,
    statenet_ranking_scale: float = 0.3,
    statenet_hyp_sigma_scale: float = 0.05,      # v5.10
    statenet_hyp_curvature_scale: float = 0.02   # v5.10
)
```

**Methods:**

#### `forward(x, temp_A, temp_B, beta_A, beta_B) -> Dict`

Forward pass through both VAEs.

```python
outputs = model(x, temp_A=1.0, temp_B=0.9, beta_A=0.3, beta_B=0.2)
# Returns: logits_A, logits_B, mu_A, mu_B, logvar_A, logvar_B,
#          z_A, z_B, z_A_tilde, z_B_tilde, H_A, H_B, beta_A, beta_B
```

#### `sample(batch_size, device, vae='A') -> torch.Tensor`

Sample from latent space.

```python
samples = model.sample(1000, 'cuda', 'A')
# Shape: (1000, 9), Values: {-1, 0, 1}
```

#### `update_gradient_norms()`

Update EMA of gradient norms for balancing.

#### `get_statenet_state() -> torch.Tensor`

Get 18D state vector for StateNet v4.

```python
state = model.get_statenet_state()
# Shape: (18,) containing H_A, H_B, KL_A, KL_B, grad_ratio, rho, lambda3,
#        coverage_A, coverage_B, r_A, r_B, delta_ranking,
#        mean_radius_A, mean_radius_B, prior_sigma, curvature
```

---

### DualNeuralVAEV5_7

VAE with StateNet v3 and metric attention.

```python
from src.models import DualNeuralVAEV5_7

model = DualNeuralVAEV5_7(
    input_dim: int = 9,
    latent_dim: int = 16,
    # ... same base params as v5.6 ...
    statenet_ranking_scale: float = 0.3  # v5.7 addition
)
```

---

### DualNeuralVAEV5 (v5.6)

Base dual VAE architecture.

```python
from src.models import DualNeuralVAEV5

model = DualNeuralVAEV5(
    input_dim: int = 9,
    latent_dim: int = 16,
    rho_min: float = 0.1,
    rho_max: float = 0.9,
    lambda3_base: float = 0.3,
    lambda3_amplitude: float = 0.15,
    eps_kl: float = 0.01,
    gradient_balance: bool = True,
    adaptive_scheduling: bool = True,
    use_statenet: bool = True,
    statenet_lr_scale: float = 0.05,
    statenet_lambda_scale: float = 0.01
)
```

---

### StateNetV4 (v5.10)

Meta-controller with hyperbolic awareness.

```python
from src.models import StateNetV4

statenet = StateNetV4(
    input_dim: int = 18,   # v5.10: adds 4 hyperbolic dims
    hidden_dim: int = 32,
    output_dim: int = 7    # v5.10: adds 2 hyperbolic outputs
)
```

**Input dimensions (18D):**
- H_A, H_B, KL_A, KL_B (4D) - base metrics
- grad_ratio, rho, lambda3 (3D) - adaptive params
- coverage_A, coverage_B (2D) - coverage
- r_A, r_B, delta_ranking (3D) - v5.7 ranking
- mean_radius_A, mean_radius_B (2D) - v5.10 hyperbolic
- prior_sigma, curvature (2D) - v5.10 hyperbolic

**Output dimensions (7D):**
- lr_correction, lambda1_corr, lambda2_corr, lambda3_corr (4D)
- ranking_weight_correction (1D) - v5.7
- delta_sigma, delta_curvature (2D) - v5.10

---

## Loss Module

### Hyperbolic Losses (v5.10)

#### HyperbolicPrior

Wrapped Normal on Poincare ball.

```python
from src.losses import HyperbolicPrior

prior = HyperbolicPrior(
    latent_dim: int = 16,
    curvature: float = 1.0,
    prior_sigma: float = 1.0,
    max_norm: float = 0.95
)

kl_loss = prior(mu, logvar, z)
```

#### HomeostaticHyperbolicPrior

Adaptive prior with bounds.

```python
from src.losses import HomeostaticHyperbolicPrior

prior = HomeostaticHyperbolicPrior(
    latent_dim: int = 16,
    curvature: float = 2.0,
    prior_sigma: float = 1.0,
    max_norm: float = 0.95,
    sigma_min: float = 0.3,
    sigma_max: float = 2.0,
    curvature_min: float = 0.5,
    curvature_max: float = 4.0,
    adaptation_rate: float = 0.01
)

kl_loss = prior(mu, logvar, z)
prior.adapt(mean_radius, target_radius=0.5)
```

#### HyperbolicReconLoss

Geodesic reconstruction loss.

```python
from src.losses import HyperbolicReconLoss

recon = HyperbolicReconLoss(
    curvature: float = 2.0,
    max_norm: float = 0.95,
    mode: str = 'weighted_ce',  # 'geodesic', 'weighted_ce', 'hybrid'
    radius_weighting: bool = True,
    radius_power: float = 2.0
)

loss = recon(logits, x, z)
```

#### HomeostaticReconLoss

Adaptive reconstruction.

```python
from src.losses import HomeostaticReconLoss

recon = HomeostaticReconLoss(
    curvature: float = 2.0,
    max_norm: float = 0.95,
    mode: str = 'weighted_ce',
    geodesic_weight_min: float = 0.1,
    geodesic_weight_max: float = 0.8,
    radius_power_min: float = 1.0,
    radius_power_max: float = 4.0,
    adaptation_rate: float = 0.01
)

loss = recon(logits, x, z)
recon.adapt(coverage, target_coverage=0.95)
```

#### HyperbolicCentroidLoss

Frechet mean clustering.

```python
from src.losses import HyperbolicCentroidLoss

centroid = HyperbolicCentroidLoss(
    max_level: int = 4,      # Tree depth (3^4 = 81 clusters)
    curvature: float = 2.0,
    max_norm: float = 0.95,
    weight: float = 0.2
)

loss = centroid(z_A, z_B, x)
```

---

### P-adic Losses

#### PAdicRankingLossHyperbolic

Triplet loss with Poincare distance.

```python
from src.losses import PAdicRankingLossHyperbolic

ranking = PAdicRankingLossHyperbolic(
    base_margin: float = 0.05,
    margin_scale: float = 0.15,
    n_triplets: int = 500,
    hard_negative_ratio: float = 0.5,
    curvature: float = 2.0,
    radial_weight: float = 0.4,
    max_norm: float = 0.95
)

loss, radial_loss = ranking(z_A, z_B, x)
```

---

### Base Losses

#### DualVAELoss

Complete loss for dual VAE.

```python
from src.losses import DualVAELoss

loss_fn = DualVAELoss(
    free_bits: float = 0.3,
    repulsion_sigma: float = 0.5
)

losses = loss_fn(
    x, outputs,
    lambda1, lambda2, lambda3,
    entropy_weight_B, repulsion_weight_B,
    grad_norm_A_ema, grad_norm_B_ema,
    gradient_balance, training
)
```

**Returns Dict:**
- `loss`: Total loss
- `ce_A`, `ce_B`: Cross-entropy
- `kl_A`, `kl_B`: KL divergence
- `loss_A`, `loss_B`: VAE losses
- `entropy_B`, `repulsion_B`: Regularization
- `H_A`, `H_B`: Entropies
- `grad_scale_A`, `grad_scale_B`: Gradient scales

#### Component Losses

```python
from src.losses import (
    ReconstructionLoss,
    KLDivergenceLoss,
    EntropyRegularization,
    RepulsionLoss
)

# Reconstruction
recon = ReconstructionLoss()
loss = recon(logits, x)

# KL Divergence
kl = KLDivergenceLoss(free_bits=0.3)
loss = kl(mu, logvar)

# Entropy
entropy = EntropyRegularization()
loss = entropy(logits)

# Repulsion
repulsion = RepulsionLoss(sigma=0.5)
loss = repulsion(z)
```

---

## Metrics Module

### Hyperbolic Metrics (v5.10)

```python
from src.metrics import (
    project_to_poincare,
    poincare_distance,
    compute_3adic_valuation,
    compute_ranking_correlation_hyperbolic
)
```

#### `project_to_poincare(z, max_norm=0.95) -> torch.Tensor`

Project latents to Poincare ball.

```python
z_projected = project_to_poincare(z, max_norm=0.95)
```

#### `poincare_distance(u, v, c=1.0) -> torch.Tensor`

Compute hyperbolic distance.

```python
dist = poincare_distance(u, v, c=2.0)
```

#### `compute_3adic_valuation(x) -> torch.Tensor`

Compute 3-adic valuation for ultrametric.

```python
valuation = compute_3adic_valuation(x)
```

#### `compute_ranking_correlation_hyperbolic(...) -> Dict`

Main evaluation metric.

```python
results = compute_ranking_correlation_hyperbolic(
    z_A, z_B, x,
    curvature=2.0,
    max_norm=0.95,
    n_samples=1000
)
# Returns: corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc,
#          mean_radius_A, mean_radius_B
```

---

## Data Module

### Generation

```python
from src.data import (
    generate_all_ternary_operations,
    count_ternary_operations,
    generate_ternary_operation_by_index
)

# All operations
operations = generate_all_ternary_operations()
# Shape: (19683, 9), Values: {-1, 0, 1}

# Count
total = count_ternary_operations()  # 19683

# By index
op = generate_ternary_operation_by_index(1000)
# Returns: List[int] of length 9
```

### Dataset

```python
from src.data import TernaryOperationDataset

dataset = TernaryOperationDataset(operations)

# Length
len(dataset)  # 19683

# Access
op = dataset[0]  # Shape: (9,)

# Statistics
stats = dataset.get_statistics()
```

---

## Artifacts Module

### CheckpointManager

```python
from src.artifacts import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir: Path,
    checkpoint_freq: int = 10
)
```

#### `save_checkpoint(...)`

```python
manager.save_checkpoint(
    epoch=100,
    model=model,
    optimizer=optimizer,
    metadata={
        'coverage_A': 19000,
        'best_corr_hyp': 0.95
    },
    is_best=True
)
```

#### `load_checkpoint(...)`

```python
checkpoint = manager.load_checkpoint(
    model=model,
    optimizer=optimizer,
    checkpoint_name='best',  # or 'latest', 'epoch_50'
    device='cuda'
)
```

#### `list_checkpoints()`

```python
checkpoints = manager.list_checkpoints()
# {'special': ['latest', 'best'], 'epochs': ['epoch_10', ...]}
```

---

## Complete Example (v5.10)

```python
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from src.models import DualNeuralVAEV5_10
from src.training import TernaryVAETrainer, HyperbolicVAETrainer, TrainingMonitor
from src.data import generate_all_ternary_operations, TernaryOperationDataset

# Load config
with open('configs/ternary_v5_10.yaml') as f:
    config = yaml.safe_load(f)

# Setup monitor
monitor = TrainingMonitor(
    eval_num_samples=1000,
    tensorboard_dir='runs',
    log_dir='logs',
    log_to_file=True
)

# Generate data
operations = generate_all_ternary_operations()
dataset = TernaryOperationDataset(operations)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, _ = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# Initialize model
model = DualNeuralVAEV5_10(
    input_dim=9,
    latent_dim=16,
    use_statenet=True,
    statenet_hyp_sigma_scale=0.05,
    statenet_hyp_curvature_scale=0.02
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize trainers
base_trainer = TernaryVAETrainer(model, config, device)
trainer = HyperbolicVAETrainer(base_trainer, model, device, config, monitor)

# Training loop
for epoch in range(300):
    losses = trainer.train_epoch(train_loader, val_loader, epoch)

    monitor.log_epoch_summary(
        epoch, 300, losses['loss'],
        losses['cov_A'], losses['cov_B'],
        losses['corr_A_hyp'], losses['corr_B_hyp'],
        losses['corr_A_euc'], losses['corr_B_euc'],
        losses['mean_radius_A'], losses['mean_radius_B'],
        losses['ranking_weight']
    )

# Save final
torch.save({
    'model_state_dict': model.state_dict(),
    'best_corr_hyp': trainer.best_corr_hyp,
    'config': config
}, 'checkpoints/final_model.pt')
```

---

## See Also

- **Architecture:** `docs/ARCHITECTURE.md`
- **Mathematical Foundations:** `docs/theory/MATHEMATICAL_FOUNDATIONS.md`
- **Installation:** `docs/INSTALLATION_AND_USAGE.md`
