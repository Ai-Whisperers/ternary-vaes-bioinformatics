# Installation and Usage Guide

**Version:** 5.10.1 (Pure Hyperbolic Geometry)
**Last Updated:** 2025-12-12

---

## Quick Start

### 1. Install Dependencies

```bash
cd ternary-vaes

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch numpy scipy pyyaml tqdm tensorboard
```

### 2. Run Training (v5.10)

```bash
python scripts/train/train_ternary_v5_10.py --config configs/ternary_v5_10.yaml
```

### 3. Monitor with TensorBoard

```bash
tensorboard --logdir=runs
# Open http://localhost:6006
```

---

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, CPU works but slow)
- 4GB+ GPU VRAM
- 8GB+ system RAM

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected:
```
PyTorch: 2.x.x
CUDA: True
```

---

## Training v5.10

### Full Training (300 epochs)

```bash
python scripts/train/train_ternary_v5_10.py \
    --config configs/ternary_v5_10.yaml \
    --log-dir logs
```

**Expected duration:** 5-10 hours on modern GPU (RTX 3080+)

### Training Output

```
logs/
├── training_YYYYMMDD_HHMMSS.log    # Full training log
runs/
├── ternary_vae_YYYYMMDD_HHMMSS/    # TensorBoard metrics
sandbox-training/checkpoints/v5_10/
├── checkpoint_epoch_N.pt           # Periodic checkpoints
├── final_model.pt                  # Final model
```

### Key Metrics to Watch

| Metric | Good Value | Description |
|--------|------------|-------------|
| corr_A_hyp | > 0.95 | Hyperbolic correlation VAE-A |
| corr_B_hyp | > 0.95 | Hyperbolic correlation VAE-B |
| cov_A | > 95% | Coverage VAE-A |
| cov_B | > 95% | Coverage VAE-B |
| mean_radius_A | 0.7-0.9 | VAE-A boundary exploration |
| mean_radius_B | 0.3-0.5 | VAE-B origin anchoring |

---

## Usage Examples

### Example 1: Basic Training

```python
import yaml
from src.models import DualNeuralVAEV5_10
from src.training import TernaryVAETrainer, HyperbolicVAETrainer, TrainingMonitor
from src.data import generate_all_ternary_operations, TernaryOperationDataset
from torch.utils.data import DataLoader, random_split

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

# Create dataset
operations = generate_all_ternary_operations()
dataset = TernaryOperationDataset(operations)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# Initialize model
model = DualNeuralVAEV5_10(
    input_dim=9,
    latent_dim=16,
    use_statenet=True
)

# Train
device = 'cuda'
base_trainer = TernaryVAETrainer(model, config, device)
trainer = HyperbolicVAETrainer(base_trainer, model, device, config, monitor)

for epoch in range(100):
    losses = trainer.train_epoch(train_loader, val_loader, epoch)
    print(f"Epoch {epoch}: corr_hyp={losses['corr_A_hyp']:.4f}")
```

### Example 2: Load and Sample

```python
import torch
from src.models import DualNeuralVAEV5_10

# Load model
model = DualNeuralVAEV5_10(input_dim=9, latent_dim=16)
checkpoint = torch.load('sandbox-training/checkpoints/v5_10/final_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate samples
samples_A = model.sample(1000, device='cuda', vae='A')
samples_B = model.sample(1000, device='cuda', vae='B')

print(f"VAE-A samples: {samples_A.shape}")  # (1000, 9)
print(f"VAE-B samples: {samples_B.shape}")  # (1000, 9)
```

### Example 3: Evaluate Hyperbolic Correlation

```python
from src.metrics import compute_ranking_correlation_hyperbolic

# With model outputs
with torch.no_grad():
    outputs = model(x, temp_A=0.3, temp_B=0.2, beta_A=0.8, beta_B=0.5)

results = compute_ranking_correlation_hyperbolic(
    outputs['z_A'], outputs['z_B'], x,
    curvature=2.0,
    max_norm=0.95,
    n_samples=1000
)

print(f"Hyperbolic correlation A: {results['corr_A_hyp']:.4f}")
print(f"Hyperbolic correlation B: {results['corr_B_hyp']:.4f}")
print(f"Mean radius A: {results['mean_radius_A']:.4f}")
print(f"Mean radius B: {results['mean_radius_B']:.4f}")
```

---

## Configuration

### v5.10 Config Structure

```yaml
config_version: "5.10"

model:
  input_dim: 9
  latent_dim: 16
  use_statenet: true
  statenet_hyp_sigma_scale: 0.05
  statenet_hyp_curvature_scale: 0.02

padic_losses:
  enable_ranking_loss_hyperbolic: true
  ranking_hyperbolic:
    curvature: 2.0
    radial_weight: 0.4

  hyperbolic_v10:
    use_hyperbolic_prior: true
    use_hyperbolic_recon: true
    use_centroid_loss: true

# Evaluation intervals
coverage_check_interval: 5
eval_interval: 20
eval_num_samples: 1000

# Logging
log_dir: logs
tensorboard_dir: runs
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `curvature` | 2.0 | Poincare ball curvature (higher = sharper tree) |
| `radial_weight` | 0.4 | Hierarchy enforcement weight |
| `prior_sigma` | 1.0 | Prior spread (homeostatic adapts) |
| `coverage_check_interval` | 5 | Epochs between coverage checks |
| `eval_interval` | 20 | Epochs between correlation checks |

---

## Monitoring

### TensorBoard Metrics

```bash
tensorboard --logdir=runs
```

**Available Metrics:**
- `loss/total`: Total training loss
- `correlation/hyp_A`, `correlation/hyp_B`: Hyperbolic correlations
- `correlation/euc_A`, `correlation/euc_B`: Euclidean correlations
- `coverage/A`, `coverage/B`: Operation coverage
- `radii/mean_A`, `radii/mean_B`: Poincare ball positioning
- `hyperbolic/kl_A`, `hyperbolic/kl_B`: Hyperbolic KL divergence
- `hyperbolic/centroid_loss`: Frechet mean clustering
- `hyperbolic/radial_loss`: Radial hierarchy enforcement

### Log Files

Logs are written to `logs/training_YYYYMMDD_HHMMSS.log`:

```
2025-12-12 04:05:00 | Starting Pure Hyperbolic Training
2025-12-12 04:05:00 | Evaluation Intervals:
2025-12-12 04:05:00 |   Coverage check: every 5 epochs
2025-12-12 04:05:00 |   Correlation check: every 20 epochs
...
Epoch 20/300 | Loss: 0.4532 | Cov: 85.2%/83.1% | r_hyp: 0.7823/0.7654
```

---

## Troubleshooting

### CUDA Out of Memory

```yaml
# Reduce batch size in config
batch_size: 128  # from 256
```

### Slow Training

```yaml
# Increase evaluation intervals
coverage_check_interval: 10  # from 5
eval_interval: 50  # from 20
```

### Import Errors

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
```

### Correlation Not Improving

Check homeostatic adaptation is enabled:
```yaml
hyperbolic_v10:
  prior:
    homeostatic: true
  recon:
    homeostatic: true
```

---

## Project Structure

```
ternary-vaes/
├── configs/
│   ├── ternary_v5_10.yaml       # v5.10 config (active)
│   └── archive/                  # Legacy configs
├── docs/
│   ├── ARCHITECTURE.md          # System architecture
│   ├── API_REFERENCE.md         # API documentation
│   └── theory/                   # Mathematical foundations
├── scripts/
│   └── train/
│       └── train_ternary_v5_10.py  # Training entry point
├── src/
│   ├── models/                   # VAE architectures
│   ├── training/                 # Training orchestration
│   ├── losses/                   # Loss functions
│   ├── metrics/                  # Evaluation metrics
│   ├── data/                     # Dataset handling
│   └── artifacts/                # Checkpoint management
├── logs/                         # Training logs
├── runs/                         # TensorBoard data
└── sandbox-training/             # Checkpoints
```

---

## Next Steps

- **Architecture details:** See `docs/ARCHITECTURE.md`
- **API reference:** See `docs/API_REFERENCE.md`
- **Mathematical theory:** See `docs/theory/MATHEMATICAL_FOUNDATIONS.md`

---

## Support

- **GitHub Issues:** Report bugs and feature requests
- **Documentation:** See `docs/` directory
