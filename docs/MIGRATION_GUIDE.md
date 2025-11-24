# Migration Guide: Original → Refactored Architecture

**Version:** Original v5.5 → Refactored v5.5
**Date:** 2025-11-24

---

## Overview

This guide helps you migrate from the original monolithic implementation to the refactored Single Responsibility Principle (SRP) architecture.

**Key Change:** The codebase has been refactored from a monolithic structure into modular components, each with a single clear responsibility.

---

## What Changed

### Before (Original)

```
ternary-vaes/
├── src/
│   ├── models/
│   │   └── ternary_vae_v5_5.py   (632 lines - model + loss + tracking)
│   └── utils/
│       ├── data.py                (199 lines - everything data-related)
│       └── metrics.py             (276 lines - all metrics)
├── scripts/
│   └── train/
│       └── train_ternary_v5_5.py (549 lines - data + trainer + main)
```

**Problems:**
- Model contained loss computation (violations of SRP)
- Trainer contained scheduling, monitoring, checkpointing
- Hard to test individual components
- Difficult to reuse components

### After (Refactored)

```
ternary-vaes/
├── src/
│   ├── training/
│   │   ├── trainer.py      (350 lines - orchestration only)
│   │   ├── schedulers.py   (210 lines - parameter scheduling)
│   │   └── monitor.py      (150 lines - logging & metrics)
│   ├── losses/
│   │   └── dual_vae_loss.py (270 lines - all loss computation)
│   ├── data/
│   │   ├── generation.py   (65 lines - data generation)
│   │   └── dataset.py      (75 lines - dataset classes)
│   ├── artifacts/
│   │   └── checkpoint_manager.py (120 lines - checkpoint I/O)
│   └── models/
│       └── ternary_vae_v5_5.py (499 lines - architecture only)
├── scripts/
│   └── train/
│       └── train_ternary_v5_5_refactored.py (115 lines - minimal glue)
```

**Benefits:**
- ✅ Each component has single responsibility
- ✅ Easy to test independently
- ✅ Components can be reused
- ✅ Clear dependencies

---

## Step-by-Step Migration

### 1. Update Imports

**Before:**
```python
# Old monolithic import
from scripts.train.train_ternary_v5_5 import DNVAETrainerV5
```

**After:**
```python
# New modular imports
from src.models.ternary_vae_v5_5 import DualNeuralVAEV5
from src.training import TernaryVAETrainer
from src.data import generate_all_ternary_operations, TernaryOperationDataset
```

---

### 2. Data Generation

**Before:**
```python
# Embedded in training script
def generate_all_ternary_operations():
    operations = []
    for i in range(3**9):
        # ... generation logic ...
    return np.array(operations, dtype=np.float32)

operations = generate_all_ternary_operations()
```

**After:**
```python
# Import from data module
from src.data import generate_all_ternary_operations, TernaryOperationDataset

operations = generate_all_ternary_operations()
dataset = TernaryOperationDataset(operations)
```

---

### 3. Model Initialization

**Before:**
```python
# Model initialized inside trainer
trainer = DNVAETrainerV5(config, device)
# Model is private: trainer.model
```

**After:**
```python
# Model created separately
model = DualNeuralVAEV5(
    input_dim=config['model']['input_dim'],
    latent_dim=config['model']['latent_dim'],
    rho_min=config['model']['rho_min'],
    rho_max=config['model']['rho_max'],
    lambda3_base=config['model']['lambda3_base'],
    lambda3_amplitude=config['model']['lambda3_amplitude'],
    eps_kl=config['model']['eps_kl'],
    gradient_balance=config['model'].get('gradient_balance', True),
    adaptive_scheduling=config['model'].get('adaptive_scheduling', True),
    use_statenet=config['model'].get('use_statenet', True),
    statenet_lr_scale=config['model'].get('statenet_lr_scale', 0.05),
    statenet_lambda_scale=config['model'].get('statenet_lambda_scale', 0.01)
)

# Pass model to trainer
trainer = TernaryVAETrainer(model, config, device)
```

**Why:** Separation of concerns - model definition separate from training logic.

---

### 4. Training

**Before:**
```python
trainer = DNVAETrainerV5(config, device)
trainer.train(train_loader, val_loader)
```

**After:**
```python
model = DualNeuralVAEV5(...)
trainer = TernaryVAETrainer(model, config, device)
trainer.train(train_loader, val_loader)
```

**Note:** Training interface is identical, but now with dependency injection.

---

### 5. Loss Computation

**Before (if using model.loss_function directly):**
```python
outputs = model(x, temp_A, temp_B, beta_A, beta_B)
losses = model.loss_function(
    x, outputs,
    entropy_weight_B=0.05,
    repulsion_weight_B=0.01,
    free_bits=0.0
)
```

**After:**
```python
from src.losses import DualVAELoss

# Initialize loss function
loss_fn = DualVAELoss(free_bits=0.0, repulsion_sigma=0.5)

# Forward pass
outputs = model(x, temp_A, temp_B, beta_A, beta_B)

# Compute losses
losses = loss_fn(
    x, outputs,
    model.lambda1, model.lambda2, model.lambda3,
    entropy_weight_B, repulsion_weight_B,
    model.grad_norm_A_ema, model.grad_norm_B_ema,
    model.gradient_balance, training=True
)
```

**Why:** Loss computation separated from model architecture.

---

### 6. Checkpointing

**Before:**
```python
# Embedded in trainer
trainer.save_checkpoint(is_best=True)
```

**After:**
```python
from src.artifacts import CheckpointManager

checkpoint_manager = CheckpointManager(
    checkpoint_dir=Path('checkpoints'),
    checkpoint_freq=10
)

checkpoint_manager.save_checkpoint(
    epoch=epoch,
    model=model,
    optimizer=optimizer,
    metadata=metadata,
    is_best=is_best
)
```

**Note:** Trainer automatically creates and uses CheckpointManager internally.

---

### 7. Scheduling

**Before:**
```python
# Embedded in trainer
temp_A = self.get_temperature(epoch, 'A')
```

**After:**
```python
from src.training import TemperatureScheduler, BetaScheduler, LearningRateScheduler

temp_scheduler = TemperatureScheduler(config, phase_4_start=200, temp_lag=5)
temp_A = temp_scheduler.get_temperature(epoch, 'A')

beta_scheduler = BetaScheduler(config, beta_phase_lag=1.5708)
beta_A = beta_scheduler.get_beta(epoch, 'A')

lr_scheduler = LearningRateScheduler(config['optimizer']['lr_schedule'])
lr = lr_scheduler.get_lr(epoch)
```

**Note:** Trainer automatically creates and uses these schedulers internally.

---

## Configuration Changes

**No configuration changes required!** The config format remains identical:

```yaml
# configs/ternary_v5_5.yaml - works with both versions
model:
  input_dim: 9
  latent_dim: 16
  ...

vae_a:
  temp_start: 1.1
  temp_end: 0.85
  ...

optimizer:
  lr_start: 0.001
  lr_schedule:
    - {epoch: 0, lr: 0.001}
    - {epoch: 50, lr: 0.0005}
```

---

## Testing Your Migration

### Quick Test (3 epochs)

```bash
python scripts/train/train_ternary_v5_5_refactored.py \
    --config configs/ternary_v5_5.yaml
# Ctrl+C after 3 epochs
```

### Full Validation (50 epochs)

```bash
# Update config: total_epochs: 50
python scripts/train/train_ternary_v5_5_refactored.py \
    --config configs/ternary_v5_5.yaml
```

**Expected:**
- Same loss values as original
- Same coverage metrics
- Same training curves

---

## Troubleshooting

### Import Errors

**Problem:**
```python
ModuleNotFoundError: No module named 'src.training'
```

**Solution:**
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
```

### Loss Values Differ

**Problem:** Loss values don't match original implementation.

**Solution:** Ensure you're passing all required arguments to `DualVAELoss`:
```python
losses = loss_fn(
    x, outputs,
    model.lambda1, model.lambda2, model.lambda3,  # ← Must pass these
    entropy_weight_B, repulsion_weight_B,
    model.grad_norm_A_ema, model.grad_norm_B_ema,  # ← And these
    model.gradient_balance, training                # ← And these
)
```

### Checkpoint Loading

**Problem:** Can't load old checkpoints.

**Solution:** Old checkpoints are compatible. Use:
```python
checkpoint = torch.load('old_checkpoint.pt')
model.load_state_dict(checkpoint['model'])
```

---

## Benefits of Migration

### Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **Testability** | Hard to test components | Easy unit tests |
| **Reusability** | Components tightly coupled | Reusable modules |
| **Maintainability** | Large monolithic files | Small focused files |
| **Understandability** | Mixed responsibilities | Clear single responsibilities |
| **Extensibility** | Hard to modify | Easy to extend |

### Example: Adding Custom Loss

**Before:** Modify 632-line model file, risk breaking other features.

**After:** Create new loss class inheriting from `nn.Module`, use in trainer:
```python
from src.losses import DualVAELoss

class CustomDualVAELoss(DualVAELoss):
    def forward(self, x, outputs, ...):
        losses = super().forward(x, outputs, ...)
        # Add custom loss term
        losses['custom'] = ...
        losses['loss'] += weight * losses['custom']
        return losses

# Use in trainer
trainer.loss_fn = CustomDualVAELoss(...)
```

---

## Compatibility

### Checkpoint Compatibility

✅ **Forward compatible:** Old checkpoints work with new code
✅ **Backward compatible:** New checkpoints work with old code
✅ **Same format:** No changes to checkpoint structure

### Config Compatibility

✅ **Fully compatible:** Same config files work with both versions
✅ **No migration needed:** Existing configs work as-is

---

## Gradual Migration

You can migrate gradually:

1. **Keep both versions:** Original in `main`, refactored in `refactor/srp-implementation`
2. **Test in parallel:** Run both versions, compare results
3. **Migrate piece by piece:** Start with data module, then losses, etc.
4. **Full switch:** When confident, merge refactored to `main`

---

## Complete Example

### Original Code

```python
# scripts/train/train_ternary_v5_5.py
from scripts.train.train_ternary_v5_5 import DNVAETrainerV5

config = yaml.safe_load(open('config.yaml'))
trainer = DNVAETrainerV5(config, 'cuda')
trainer.train(train_loader, val_loader)
```

### Refactored Code

```python
# scripts/train/train_ternary_v5_5_refactored.py
from src.models.ternary_vae_v5_5 import DualNeuralVAEV5
from src.training import TernaryVAETrainer
from src.data import generate_all_ternary_operations, TernaryOperationDataset

# Load config
config = yaml.safe_load(open('config.yaml'))

# Generate data
operations = generate_all_ternary_operations()
dataset = TernaryOperationDataset(operations)
train_loader, val_loader = create_dataloaders(dataset, config)

# Initialize model
model = DualNeuralVAEV5(
    input_dim=config['model']['input_dim'],
    latent_dim=config['model']['latent_dim'],
    rho_min=config['model']['rho_min'],
    rho_max=config['model']['rho_max'],
    lambda3_base=config['model']['lambda3_base'],
    lambda3_amplitude=config['model']['lambda3_amplitude'],
    eps_kl=config['model']['eps_kl'],
    gradient_balance=config['model'].get('gradient_balance', True),
    adaptive_scheduling=config['model'].get('adaptive_scheduling', True),
    use_statenet=config['model'].get('use_statenet', True)
)

# Initialize trainer (dependency injection)
trainer = TernaryVAETrainer(model, config, 'cuda')

# Train (same interface)
trainer.train(train_loader, val_loader)
```

---

## Need Help?

- **Architecture docs:** `docs/ARCHITECTURE.md`
- **Refactoring plan:** `reports/SRP_REFACTORING_PLAN.md`
- **Progress report:** `reports/REFACTORING_PROGRESS.md`
- **GitHub issues:** [Report issues](https://github.com/gesttaltt/ternary-vaes/issues)

---

## Summary

**Migration is straightforward:**
1. Update imports from monolithic to modular
2. Create model separately, pass to trainer
3. Use same training interface
4. Same config files work as-is
5. Same checkpoints work as-is

**No breaking changes to:**
- Configuration format
- Checkpoint format
- Training results
- Model behavior

**Benefits:**
- Cleaner architecture
- Better testability
- More maintainable
- Easier to extend
