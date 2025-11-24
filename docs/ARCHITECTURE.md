# Ternary VAE Architecture Documentation

**Version:** 5.5 (Refactored)
**Last Updated:** 2025-11-24
**Status:** Production-ready

---

## Overview

The Ternary VAE v5.5 implements a dual-pathway variational autoencoder for learning all 19,683 possible ternary operations (3^9 space). The architecture has been refactored following Single Responsibility Principle (SRP) for clean separation of concerns.

---

## System Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Script                          │
│  (scripts/train/train_ternary_v5_5_refactored.py)          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │      TernaryVAETrainer             │
         │  (Orchestrates training loop)      │
         └────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
  ┌──────────┐   ┌──────────────┐   ┌──────────┐
  │Schedulers│   │ DualVAELoss  │   │ Monitor  │
  └──────────┘   └──────────────┘   └──────────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   DualNeuralVAEV5     │
              │  (Model architecture)  │
              └───────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
         ┌────────┐              ┌────────┐
         │ VAE-A  │              │ VAE-B  │
         │Chaotic │              │Frozen  │
         └────────┘              └────────┘
```

---

## Module Responsibilities

### 1. Training Module (`src/training/`)

**Purpose:** Orchestrate training process with clean separation of concerns.

#### TernaryVAETrainer (`trainer.py`)

**Responsibility:** Training loop orchestration only

**Key Methods:**
- `__init__(model, config, device)`: Initialize trainer with dependencies
- `train_epoch(train_loader)`: Execute one training epoch
- `validate(val_loader)`: Run validation
- `train(train_loader, val_loader)`: Main training loop

**Dependencies:**
- Receives model, config, device via constructor
- Uses TemperatureScheduler for temperature parameters
- Uses BetaScheduler for KL weights
- Uses LearningRateScheduler for optimizer LR
- Uses TrainingMonitor for logging
- Uses CheckpointManager for persistence
- Uses DualVAELoss for loss computation

**Does NOT:**
- Compute losses (delegated to DualVAELoss)
- Schedule parameters (delegated to Schedulers)
- Log metrics (delegated to TrainingMonitor)
- Save checkpoints (delegated to CheckpointManager)

#### Schedulers (`schedulers.py`)

**Responsibility:** Parameter scheduling only

**Classes:**

1. **TemperatureScheduler**
   - Linear annealing with optional cyclic modulation
   - Phase 4 ultra-exploration boost
   - Lag support for VAE-B

2. **BetaScheduler**
   - KL warmup to prevent posterior collapse
   - Phase lag between VAE-A and VAE-B

3. **LearningRateScheduler**
   - Step-based learning rate scheduling
   - Reads from config schedule

**Utility Functions:**
- `linear_schedule()`: Linear interpolation
- `cyclic_schedule()`: Cosine-based oscillation

#### TrainingMonitor (`monitor.py`)

**Responsibility:** Monitoring and logging only

**Key Methods:**
- `update_histories()`: Track entropy and coverage
- `check_best()`: Determine best validation loss
- `evaluate_coverage()`: Compute operation coverage
- `log_epoch()`: Print epoch results
- `get_metadata()`: Export tracking data
- `print_training_summary()`: Final summary

**Tracks:**
- Best validation loss
- Entropy history (VAE-A, VAE-B)
- Coverage history (VAE-A, VAE-B)
- Patience counter for early stopping

---

### 2. Loss Module (`src/losses/`)

**Purpose:** All loss computation separated from model architecture.

#### DualVAELoss (`dual_vae_loss.py`)

**Responsibility:** Compute complete loss for dual VAE system

**Component Losses:**

1. **ReconstructionLoss**
   - Cross-entropy for ternary operations
   - Converts {-1, 0, 1} to class indices {0, 1, 2}

2. **KLDivergenceLoss**
   - KL divergence with free bits support
   - Prevents posterior collapse

3. **EntropyRegularization**
   - Encourages output diversity
   - Computed over batch-averaged distributions

4. **RepulsionLoss**
   - Encourages latent space diversity
   - RBF kernel-based repulsion

**Forward Method:**
```python
loss_dict = dual_vae_loss(
    x,                      # Input data
    outputs,                # Model outputs
    lambda1, lambda2, lambda3,  # Loss weights
    entropy_weight_B,       # VAE-B entropy weight
    repulsion_weight_B,     # VAE-B repulsion weight
    grad_norm_A_ema,        # Gradient norm for balancing
    grad_norm_B_ema,
    gradient_balance,       # Enable/disable balancing
    training                # Training mode flag
)
```

**Returns:**
- Complete loss dictionary with all components
- Individual losses (ce_A, ce_B, kl_A, kl_B)
- Regularization terms (entropy, repulsion, alignment)
- Gradient scaling factors
- Lambda values for logging

---

### 3. Data Module (`src/data/`)

**Purpose:** Data generation and loading.

#### Generation (`generation.py`)

**Responsibility:** Generate ternary operations

**Key Functions:**
- `generate_all_ternary_operations()`: Generate all 19,683 operations
- `count_ternary_operations()`: Return total count (3^9)
- `generate_ternary_operation_by_index(idx)`: Generate specific operation

**Operation Format:**
- 9-element vectors with values in {-1, 0, 1}
- Represents truth table for ternary logic function
- Total space: 3^9 = 19,683 operations

#### Dataset (`dataset.py`)

**Responsibility:** PyTorch dataset interface

**TernaryOperationDataset:**
- Wraps numpy/torch arrays as PyTorch Dataset
- Validates shape and value ranges
- Provides statistics via `get_statistics()`

---

### 4. Artifacts Module (`src/artifacts/`)

**Purpose:** Checkpoint and artifact lifecycle management.

#### CheckpointManager (`checkpoint_manager.py`)

**Responsibility:** Checkpoint I/O only

**Key Methods:**
- `save_checkpoint(epoch, model, optimizer, metadata, is_best)`
  - Saves latest.pt always
  - Saves best.pt if is_best=True
  - Saves epoch_N.pt at checkpoint_freq intervals

- `load_checkpoint(model, optimizer, checkpoint_name, device)`
  - Loads checkpoint and restores state
  - Supports 'latest', 'best', or 'epoch_N'

- `list_checkpoints()`: Enumerate available checkpoints
- `get_latest_epoch()`: Get epoch of latest checkpoint

**Artifact Lifecycle:**
```
Training → artifacts/raw/        (direct outputs)
         ↓
    Validation
         ↓
         artifacts/validated/    (passed validation)
         ↓
    Approval
         ↓
         artifacts/production/   (deployment-ready)
```

---

### 5. Model Module (`src/models/`)

**Purpose:** Neural network architecture only.

#### DualNeuralVAEV5 (`ternary_vae_v5_5.py`)

**Responsibility:** Architecture definition and forward pass only

**What it DOES:**
- Define encoder/decoder architectures
- Implement forward pass
- Sample from latent space
- Compute entropies
- Track gradient norms (for balancing)
- Manage adaptive parameters (lambda1, lambda2, lambda3, rho)
- StateNet meta-controller

**What it DOES NOT:**
- ~~Compute losses~~ (delegated to DualVAELoss)
- ~~Schedule parameters~~ (delegated to Schedulers)
- ~~Log metrics~~ (delegated to TrainingMonitor)
- ~~Save checkpoints~~ (delegated to CheckpointManager)

**Key Components:**

1. **VAE-A (Chaotic Regime)**
   - TernaryEncoderA + TernaryDecoderA
   - Exploratory pathway
   - Higher temperature, variable beta

2. **VAE-B (Frozen Regime)**
   - TernaryEncoderB + TernaryDecoderB
   - Conservative pathway with residual blocks
   - Lower temperature, stabilizing influence

3. **StateNet**
   - Meta-controller for adaptive parameter tuning
   - Adjusts lambda1, lambda2, lambda3, and learning rate
   - Small overhead: 1,068 parameters (0.63% of total)

**Architecture Details:**
- Total parameters: 168,770
- VAE-A: 50,203 params
- VAE-B: 117,499 params
- StateNet: 1,068 params

---

## Training Flow

### Initialization

```python
# 1. Load configuration
config = yaml.safe_load('config.yaml')

# 2. Generate data
operations = generate_all_ternary_operations()
dataset = TernaryOperationDataset(operations)
train_loader, val_loader = create_data_loaders(dataset)

# 3. Initialize model
model = DualNeuralVAEV5(
    input_dim=9,
    latent_dim=16,
    ...
)

# 4. Initialize trainer (injects all dependencies)
trainer = TernaryVAETrainer(model, config, device)
# Trainer creates: schedulers, monitor, checkpoint_manager, loss_fn

# 5. Train
trainer.train(train_loader, val_loader)
```

### Training Loop (per epoch)

```python
for epoch in range(total_epochs):
    # 1. Update model parameters
    trainer._update_model_parameters(epoch)

    # 2. Get scheduled values
    temp_A = temp_scheduler.get_temperature(epoch, 'A')
    temp_B = temp_scheduler.get_temperature(epoch, 'B')
    beta_A = beta_scheduler.get_beta(epoch, 'A')
    beta_B = beta_scheduler.get_beta(epoch, 'B')
    lr = lr_scheduler.get_lr(epoch)

    for batch in train_loader:
        # 3. Forward pass
        outputs = model(batch, temp_A, temp_B, beta_A, beta_B)

        # 4. Compute losses
        losses = loss_fn(
            batch, outputs,
            model.lambda1, model.lambda2, model.lambda3,
            entropy_weight, repulsion_weight,
            model.grad_norm_A_ema, model.grad_norm_B_ema,
            model.gradient_balance, training=True
        )

        # 5. Backward and optimize
        optimizer.zero_grad()
        losses['loss'].backward()
        model.update_gradient_norms()
        optimizer.step()

    # 6. Validate
    val_losses = trainer.validate(val_loader)

    # 7. Monitor
    is_best = monitor.check_best(val_losses['loss'])
    unique_A, cov_A = monitor.evaluate_coverage(model, ...)
    monitor.log_epoch(epoch, train_losses, val_losses, ...)

    # 8. Save checkpoint
    checkpoint_manager.save_checkpoint(
        epoch, model, optimizer, metadata, is_best
    )
```

---

## Dependency Injection

The refactored architecture uses **constructor-based dependency injection**:

```python
# Trainer receives all dependencies
trainer = TernaryVAETrainer(
    model=model,           # Model to train
    config=config,         # Training configuration
    device=device          # Device (cuda/cpu)
)

# Internally, trainer creates:
# - self.temp_scheduler = TemperatureScheduler(config, ...)
# - self.beta_scheduler = BetaScheduler(config, ...)
# - self.lr_scheduler = LearningRateScheduler(config['lr_schedule'])
# - self.monitor = TrainingMonitor(config['eval_num_samples'])
# - self.checkpoint_manager = CheckpointManager(config['checkpoint_dir'])
# - self.loss_fn = DualVAELoss(config['free_bits'])
```

**Benefits:**
- Easy to test (can mock dependencies)
- Clear dependencies visible in constructor
- Can swap implementations
- No hidden global state

---

## Configuration

Training is configured via YAML files:

```yaml
# Model architecture
model:
  input_dim: 9
  latent_dim: 16
  gradient_balance: true
  use_statenet: true

# VAE-A parameters
vae_a:
  temp_start: 1.1
  temp_end: 0.85
  beta_start: 0.0
  beta_end: 0.4
  beta_warmup_epochs: 50

# VAE-B parameters
vae_b:
  temp_start: 0.9
  temp_end: 0.8
  beta_start: 0.0
  beta_end: 0.3
  entropy_weight: 0.05
  repulsion_weight: 0.01

# Optimizer
optimizer:
  lr_start: 0.001
  lr_schedule:
    - {epoch: 0, lr: 0.001}
    - {epoch: 50, lr: 0.0005}

# Training
total_epochs: 400
batch_size: 256
checkpoint_freq: 10
patience: 100
```

---

## Testing

### Unit Testing

Each component can be tested independently:

```python
# Test scheduler
scheduler = TemperatureScheduler(config, phase_4_start=200, temp_lag=5)
assert scheduler.get_temperature(0, 'A') == 1.1
assert scheduler.get_temperature(100, 'A') < 1.1

# Test loss
loss_fn = DualVAELoss(free_bits=0.0)
losses = loss_fn(x, outputs, lambda1, lambda2, lambda3, ...)
assert 'loss' in losses
assert losses['ce_A'] >= 0

# Test dataset
dataset = TernaryOperationDataset(operations)
assert len(dataset) == 19683
assert dataset[0].shape == (9,)
```

### Integration Testing

```python
# Run full training loop for N epochs
trainer = TernaryVAETrainer(model, config, device)
trainer.train(train_loader, val_loader)
```

---

## Migration Guide

### From Original to Refactored

**Old code:**
```python
from scripts.train.train_ternary_v5_5 import DNVAETrainerV5
trainer = DNVAETrainerV5(config, device)
trainer.train(train_loader, val_loader)
```

**New code:**
```python
from src.training import TernaryVAETrainer
from src.models.ternary_vae_v5_5 import DualNeuralVAEV5

model = DualNeuralVAEV5(...)
trainer = TernaryVAETrainer(model, config, device)
trainer.train(train_loader, val_loader)
```

**Key changes:**
- Trainer is now a pure orchestrator
- Model created separately and passed to trainer
- Data generation moved to `src.data` module
- Loss computation separated into `src.losses`

---

## Performance

**Validation Results:**
- 50-epoch test completed successfully
- Best validation loss: -0.2562 (epoch 6)
- Coverage: 90%+ achievable
- Training time: ~90 minutes for 50 epochs (CUDA)

**No performance regression:**
- Refactored code matches original implementation exactly
- Same loss values
- Same gradient flow
- Same training curves

---

## Future Enhancements

Possible improvements maintaining SRP:

1. **Metrics Module**: Extract coverage/entropy computation
2. **Validation Module**: Separate validation logic
3. **Callbacks**: Add training callbacks for extensibility
4. **Artifact Repository**: Promotion workflow (raw → validated → production)
5. **Experiment Tracking**: Integration with W&B/MLflow

---

## References

- **Original Paper**: Dual-Neural VAE architecture
- **SRP Refactoring Plan**: `reports/SRP_REFACTORING_PLAN.md`
- **Progress Report**: `reports/REFACTORING_PROGRESS.md`
- **Training Report**: `reports/training_report_2025-11-23.md`
