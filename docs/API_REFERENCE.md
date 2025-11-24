# API Reference - Refactored Ternary VAE

**Version:** 5.5 (Refactored)
**Last Updated:** 2025-11-24

---

## Quick Links

- [Training](#training-module)
- [Losses](#loss-module)
- [Data](#data-module)
- [Artifacts](#artifacts-module)
- [Models](#model-module)

---

## Training Module

### TernaryVAETrainer

Main training orchestrator.

```python
from src.training import TernaryVAETrainer

trainer = TernaryVAETrainer(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: str = 'cuda'
)
```

**Methods:**

#### `train(train_loader, val_loader)`

Execute complete training loop.

```python
trainer.train(train_loader, val_loader)
```

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

**Parameters:**
- `epoch`: Current epoch
- `vae`: 'A' or 'B'

**Returns:** Temperature value (float)

---

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

**Parameters:**
- `epoch`: Current epoch
- `vae`: 'A' or 'B'

**Returns:** Beta value (float)

**Features:**
- KL warmup to prevent posterior collapse
- Phase lag between VAE-A and VAE-B

---

#### LearningRateScheduler

```python
from src.training import LearningRateScheduler

scheduler = LearningRateScheduler(
    lr_schedule: List[Dict[str, Any]]
)

lr = scheduler.get_lr(epoch: int)
```

**Example schedule:**
```python
lr_schedule = [
    {'epoch': 0, 'lr': 0.001},
    {'epoch': 50, 'lr': 0.0005},
    {'epoch': 100, 'lr': 0.0001}
]
```

---

### TrainingMonitor

```python
from src.training import TrainingMonitor

monitor = TrainingMonitor(eval_num_samples: int = 100000)
```

**Methods:**

#### `update_histories(H_A, H_B, coverage_A, coverage_B)`

Update tracked metrics.

#### `check_best(val_loss) -> bool`

Check if current validation loss is best.

```python
is_best = monitor.check_best(val_losses['loss'])
```

#### `evaluate_coverage(model, num_samples, device, vae='A') -> Tuple[int, float]`

Evaluate operation coverage.

```python
unique_ops, coverage_pct = monitor.evaluate_coverage(
    model, num_samples=50000, device='cuda', vae='A'
)
```

**Returns:**
- `unique_ops`: Number of unique operations sampled
- `coverage_pct`: Coverage percentage (0-100)

#### `log_epoch(...)`

Print epoch results to console.

#### `get_metadata() -> Dict[str, Any]`

Get all tracked metrics for checkpointing.

---

## Loss Module

### DualVAELoss

Complete loss computation for Dual VAE system.

```python
from src.losses import DualVAELoss

loss_fn = DualVAELoss(
    free_bits: float = 0.0,
    repulsion_sigma: float = 0.5
)

losses = loss_fn(
    x: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    lambda1: float,
    lambda2: float,
    lambda3: float,
    entropy_weight_B: float,
    repulsion_weight_B: float,
    grad_norm_A_ema: torch.Tensor,
    grad_norm_B_ema: torch.Tensor,
    gradient_balance: bool,
    training: bool
)
```

**Returns:** Dict with keys:
- `loss`: Total loss (scalar)
- `ce_A`, `ce_B`: Cross-entropy losses
- `kl_A`, `kl_B`: KL divergences
- `loss_A`, `loss_B`: VAE losses
- `entropy_B`: VAE-B entropy
- `repulsion_B`: VAE-B repulsion
- `entropy_align`: Entropy alignment
- `H_A`, `H_B`: Entropies
- `grad_scale_A`, `grad_scale_B`: Gradient scales
- `lambda1`, `lambda2`, `lambda3`: Lambda values

---

### Component Losses

#### ReconstructionLoss

```python
from src.losses import ReconstructionLoss

recon_loss = ReconstructionLoss()
loss = recon_loss(logits, x)
```

**Parameters:**
- `logits`: Model logits (batch_size, 9, 3)
- `x`: Input data (batch_size, 9) in {-1, 0, 1}

---

#### KLDivergenceLoss

```python
from src.losses import KLDivergenceLoss

kl_loss = KLDivergenceLoss(free_bits: float = 0.0)
kl = kl_loss(mu, logvar)
```

**Free bits:** Minimum KL per dimension before penalty applies (prevents collapse).

---

#### EntropyRegularization

```python
from src.losses import EntropyRegularization

entropy_loss = EntropyRegularization()
loss = entropy_loss(logits)
```

Returns negative entropy (lower = more diverse).

---

#### RepulsionLoss

```python
from src.losses import RepulsionLoss

repulsion_loss = RepulsionLoss(sigma: float = 0.5)
loss = repulsion_loss(z)
```

Encourages diversity in latent space via RBF kernel.

---

## Data Module

### Generation

```python
from src.data import (
    generate_all_ternary_operations,
    count_ternary_operations,
    generate_ternary_operation_by_index
)
```

#### `generate_all_ternary_operations() -> np.ndarray`

Generate all 19,683 ternary operations.

```python
operations = generate_all_ternary_operations()
# Shape: (19683, 9)
# Values: {-1, 0, 1}
```

#### `count_ternary_operations() -> int`

Return total count (3^9 = 19683).

#### `generate_ternary_operation_by_index(index: int) -> List[int]`

Generate specific operation by index.

```python
op = generate_ternary_operation_by_index(1000)
# Returns: [1, -1, 0, ...] (9 elements)
```

---

### Dataset

```python
from src.data import TernaryOperationDataset

dataset = TernaryOperationDataset(
    operations: Union[np.ndarray, torch.Tensor]
)
```

**Methods:**

#### `__len__() -> int`

Return dataset size.

#### `__getitem__(idx: int) -> torch.Tensor`

Get operation at index.

```python
op = dataset[0]  # Shape: (9,)
```

#### `get_statistics() -> Dict[str, Any]`

Get dataset statistics.

```python
stats = dataset.get_statistics()
# Returns:
# {
#     'size': 19683,
#     'shape': (19683, 9),
#     'mean': 0.0,
#     'std': 0.816,
#     'min': -1.0,
#     'max': 1.0,
#     'value_counts': {'-1': ..., '0': ..., '1': ...}
# }
```

---

## Artifacts Module

### CheckpointManager

Manage checkpoint saving and loading.

```python
from src.artifacts import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir: Path,
    checkpoint_freq: int = 10
)
```

**Methods:**

#### `save_checkpoint(epoch, model, optimizer, metadata, is_best=False)`

Save checkpoint with metadata.

```python
manager.save_checkpoint(
    epoch=10,
    model=model,
    optimizer=optimizer,
    metadata={'coverage_A': 19000, 'coverage_B': 18500},
    is_best=True
)
```

**Saves:**
- `latest.pt`: Always
- `best.pt`: If is_best=True
- `epoch_N.pt`: Every checkpoint_freq epochs

---

#### `load_checkpoint(model, optimizer=None, checkpoint_name='latest', device='cuda')`

Load checkpoint and restore state.

```python
checkpoint = manager.load_checkpoint(
    model=model,
    optimizer=optimizer,
    checkpoint_name='best',  # or 'latest', 'epoch_50'
    device='cuda'
)

# Returns metadata dict
epoch = checkpoint['epoch']
coverage = checkpoint['coverage_A']
```

---

#### `list_checkpoints() -> Dict[str, List]`

List available checkpoints.

```python
checkpoints = manager.list_checkpoints()
# Returns:
# {
#     'special': ['latest', 'best'],
#     'epochs': ['epoch_10', 'epoch_20', 'epoch_30']
# }
```

---

#### `get_latest_epoch() -> Optional[int]`

Get epoch number of latest checkpoint.

```python
latest_epoch = manager.get_latest_epoch()
# Returns: 103 (or None if no checkpoints)
```

---

## Model Module

### DualNeuralVAEV5

Dual-pathway VAE architecture.

```python
from src.models.ternary_vae_v5_5 import DualNeuralVAEV5

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

**Methods:**

#### `forward(x, temp_A, temp_B, beta_A, beta_B)`

Forward pass through both VAEs.

```python
outputs = model(
    x=batch_data,
    temp_A=1.0,
    temp_B=0.9,
    beta_A=0.3,
    beta_B=0.2
)
```

**Returns:** Dict with keys:
- `logits_A`, `logits_B`: Decoder outputs
- `mu_A`, `mu_B`: Encoder means
- `logvar_A`, `logvar_B`: Encoder log variances
- `z_A`, `z_B`: Sampled latents
- `z_A_tilde`, `z_B_tilde`: Cross-injected latents
- `H_A`, `H_B`: Entropies
- `beta_A`, `beta_B`: Beta values

---

#### `sample(batch_size, device, vae='A')`

Sample from latent space.

```python
samples = model.sample(
    batch_size=1000,
    device='cuda',
    vae='A'  # or 'B'
)
# Shape: (1000, 9)
# Values: {-1, 0, 1}
```

---

#### `update_gradient_norms()`

Update EMA of gradient norms (for balancing).

```python
# Call after backward, before optimizer.step()
loss.backward()
model.update_gradient_norms()
optimizer.step()
```

---

#### `update_adaptive_lambdas(grad_ratio, coverage_A, coverage_B)`

Update lambda1, lambda2 adaptively.

```python
model.update_adaptive_lambdas(
    grad_ratio=0.8,
    coverage_A=19000,
    coverage_B=18500
)
# Updates: model.lambda1, model.lambda2
```

---

#### `compute_phase_scheduled_rho(epoch, phase_4_start)`

Compute phase-scheduled permeability.

```python
rho = model.compute_phase_scheduled_rho(
    epoch=50,
    phase_4_start=200
)
# Returns: float in [rho_min, rho_max]
```

---

#### `compute_cyclic_lambda3(epoch, period=30)`

Compute cyclic entropy alignment weight.

```python
lambda3 = model.compute_cyclic_lambda3(epoch=25, period=30)
# Returns: lambda3_base ± lambda3_amplitude
```

---

## Utility Functions

### Scheduling

```python
from src.training import linear_schedule, cyclic_schedule

# Linear interpolation
value = linear_schedule(
    epoch=50,
    start_val=1.0,
    end_val=0.5,
    total_epochs=100,
    start_epoch=0
)

# Cyclic oscillation
value = cyclic_schedule(
    epoch=25,
    base_val=1.0,
    amplitude=0.1,
    period=30
)
```

---

## Complete Example

```python
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, random_split

# Import modules
from src.models.ternary_vae_v5_5 import DualNeuralVAEV5
from src.training import TernaryVAETrainer
from src.data import generate_all_ternary_operations, TernaryOperationDataset

# Load config
with open('configs/ternary_v5_5.yaml') as f:
    config = yaml.safe_load(f)

# Generate data
operations = generate_all_ternary_operations()
dataset = TernaryOperationDataset(operations)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Initialize model
model = DualNeuralVAEV5(
    input_dim=9,
    latent_dim=16,
    rho_min=0.1,
    rho_max=0.9,
    lambda3_base=0.3,
    lambda3_amplitude=0.15,
    eps_kl=0.01,
    gradient_balance=True,
    adaptive_scheduling=True,
    use_statenet=True
)

# Initialize trainer
trainer = TernaryVAETrainer(model, config, device='cuda')

# Train
trainer.train(train_loader, val_loader)

# Load best checkpoint
from src.artifacts import CheckpointManager
manager = CheckpointManager(Path(config['checkpoint_dir']))
checkpoint = manager.load_checkpoint(model, checkpoint_name='best')

# Evaluate
unique_A, cov_A = trainer.monitor.evaluate_coverage(model, 50000, 'cuda', 'A')
print(f"Coverage: {cov_A:.2f}%")
```

---

## Type Hints

All functions use type hints:

```python
def get_temperature(self, epoch: int, vae: str = 'A') -> float:
    ...

def evaluate_coverage(
    self,
    model: torch.nn.Module,
    num_samples: int,
    device: str,
    vae: str = 'A'
) -> Tuple[int, float]:
    ...
```

---

## Error Handling

```python
# Dataset validation
try:
    dataset = TernaryOperationDataset(invalid_data)
except ValueError as e:
    print(f"Invalid data: {e}")

# Checkpoint loading
from src.artifacts import CheckpointManager
try:
    checkpoint = manager.load_checkpoint(model, checkpoint_name='missing')
except FileNotFoundError as e:
    print(f"Checkpoint not found: {e}")

# Index bounds
try:
    op = generate_ternary_operation_by_index(20000)  # > 19683
except ValueError as e:
    print(f"Index out of range: {e}")
```

---

## Best Practices

1. **Always validate data:**
   ```python
   dataset = TernaryOperationDataset(operations)  # Validates automatically
   stats = dataset.get_statistics()  # Check statistics
   ```

2. **Use dependency injection:**
   ```python
   # Good: Dependencies explicit
   trainer = TernaryVAETrainer(model, config, device)

   # Bad: Hidden dependencies
   # trainer = TernaryVAETrainer(config)  # Where's the model?
   ```

3. **Leverage schedulers:**
   ```python
   # Don't hardcode schedules
   temp_A = temp_scheduler.get_temperature(epoch, 'A')

   # Avoid magic numbers
   # temp_A = 1.1 - 0.25 * epoch / total_epochs  # Bad
   ```

4. **Monitor training:**
   ```python
   is_best = monitor.check_best(val_loss)
   unique_A, cov_A = monitor.evaluate_coverage(model, ...)
   monitor.log_epoch(epoch, train_losses, val_losses, ...)
   ```

5. **Save checkpoints regularly:**
   ```python
   checkpoint_manager.save_checkpoint(
       epoch, model, optimizer, metadata, is_best
   )
   ```

---

## Performance Tips

1. **Batch size:** Use 256 for good GPU utilization
2. **Num workers:** Set to 0 on Windows, 4-8 on Linux
3. **Eval samples:** 50,000 for accurate coverage, 10,000 for quick checks
4. **Checkpoint freq:** Save every 10 epochs to balance disk space and recovery

---

## See Also

- [Architecture Documentation](ARCHITECTURE.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [Training Report](../reports/training_report_2025-11-23.md)
