# Quick Start

Get up and running with Ternary VAE in under 5 minutes.

## 60-Second Overview

```python
from src.models import TernaryVAE
from src.losses import create_registry_from_training_config
from src.config import TrainingConfig
import torch

# 1. Create config
config = TrainingConfig(epochs=10, batch_size=32)

# 2. Create model
model = TernaryVAE(input_dim=19683, latent_dim=16)

# 3. Create sample data (19683 = 3^9 ternary operations)
x = torch.randint(0, 19683, (32,))  # Batch of indices
x_onehot = torch.zeros(32, 19683).scatter_(1, x.unsqueeze(1), 1)

# 4. Forward pass
outputs = model(x_onehot)

# 5. Compute loss
registry = create_registry_from_training_config(config)
result = registry.compose(outputs, x)

print(f"Loss: {result.total.item():.4f}")
print(f"Latent shape: {outputs['z_hyperbolic'].shape}")
```

## Complete Training Example

### Step 1: Setup

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import TernaryVAE
from src.config import TrainingConfig
from src.losses import create_registry_from_training_config
from src.geometry import RiemannianAdam

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Step 2: Create Synthetic Data

```python
# Generate random ternary operation indices
n_samples = 1000
data = torch.randint(0, 19683, (n_samples,))

# Create one-hot encoding
data_onehot = torch.zeros(n_samples, 19683)
data_onehot.scatter_(1, data.unsqueeze(1), 1)

# Create DataLoader
dataset = TensorDataset(data_onehot, data)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"Created {n_samples} samples")
```

### Step 3: Initialize Model

```python
# Configuration
config = TrainingConfig(
    epochs=50,
    batch_size=64,
    geometry={"curvature": 1.0, "latent_dim": 16},
    loss_weights={"reconstruction": 1.0, "kl_divergence": 0.5},
)

# Model
model = TernaryVAE(
    input_dim=19683,
    latent_dim=config.geometry.latent_dim,
    curvature=config.geometry.curvature,
).to(device)

# Optimizer (Riemannian for hyperbolic space)
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)

# Loss
loss_registry = create_registry_from_training_config(config)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Step 4: Training Loop

```python
# Training
model.train()
for epoch in range(config.epochs):
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)

        # Compute loss
        result = loss_registry.compose(outputs, batch_y)

        # Backward pass
        result.total.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += result.total.item()

    avg_loss = total_loss / len(train_loader)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f}")
```

### Step 5: Evaluate

```python
# Evaluation
model.eval()
with torch.no_grad():
    # Get latent representations
    sample_x, sample_y = next(iter(train_loader))
    sample_x = sample_x.to(device)

    outputs = model(sample_x)
    z = outputs["z_hyperbolic"]

    print(f"\nLatent space statistics:")
    print(f"  Mean norm: {z.norm(dim=1).mean():.4f}")
    print(f"  Max norm: {z.norm(dim=1).max():.4f}")

    # Reconstruction accuracy
    preds = outputs["reconstruction"].argmax(dim=1)
    accuracy = (preds == sample_y.to(device)).float().mean()
    print(f"  Reconstruction accuracy: {accuracy:.2%}")
```

## Using Configuration Files

### Create config.yaml

```yaml
# config.yaml
seed: 42
epochs: 100
batch_size: 64

geometry:
  curvature: 1.0
  max_radius: 0.95
  latent_dim: 16

optimizer:
  type: adamw
  learning_rate: 0.001
  weight_decay: 0.01

loss_weights:
  reconstruction: 1.0
  kl_divergence: 0.5
  ranking: 0.1
```

### Load and Use

```python
from src.config import load_config

config = load_config("config.yaml")
print(f"Epochs: {config.epochs}")
print(f"Curvature: {config.geometry.curvature}")
```

## Working with Real Data

### Codon Sequences

```python
from src.data import CodonDataset

# Load codon data
dataset = CodonDataset(
    fasta_file="data/sequences.fasta",
    max_length=100,
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Custom Dataset

```python
from torch.utils.data import Dataset

class MyTernaryDataset(Dataset):
    def __init__(self, operations):
        """
        Args:
            operations: List of ternary operation indices (0-19682)
        """
        self.operations = torch.tensor(operations, dtype=torch.long)

    def __len__(self):
        return len(self.operations)

    def __getitem__(self, idx):
        op = self.operations[idx]
        # One-hot encode
        onehot = torch.zeros(19683)
        onehot[op] = 1.0
        return onehot, op
```

## Visualizing Results

### Latent Space

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Get latent representations
model.eval()
all_z = []
all_y = []

with torch.no_grad():
    for batch_x, batch_y in train_loader:
        outputs = model(batch_x.to(device))
        all_z.append(outputs["z_hyperbolic"].cpu())
        all_y.append(batch_y)

z = torch.cat(all_z).numpy()
y = torch.cat(all_y).numpy()

# t-SNE projection (for high-dim latent spaces)
if z.shape[1] > 2:
    z_2d = TSNE(n_components=2).fit_transform(z)
else:
    z_2d = z

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y % 10, cmap='tab10', alpha=0.5, s=10)
plt.colorbar(label='Operation class (mod 10)')
plt.title('Latent Space Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('latent_space.png', dpi=150)
plt.show()
```

### Training Curves

```python
# Track metrics during training
history = {"loss": [], "kl": [], "recon": []}

for epoch in range(config.epochs):
    epoch_metrics = {"loss": 0, "kl": 0, "recon": 0}

    for batch_x, batch_y in train_loader:
        # ... training step ...
        result = loss_registry.compose(outputs, batch_y)

        epoch_metrics["loss"] += result.total.item()
        epoch_metrics["kl"] += result.components.get("kl", 0)
        epoch_metrics["recon"] += result.components.get("reconstruction", 0)

    for key in epoch_metrics:
        history[key].append(epoch_metrics[key] / len(train_loader))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, values) in zip(axes, history.items()):
    ax.plot(values)
    ax.set_title(name.capitalize())
    ax.set_xlabel("Epoch")
plt.tight_layout()
plt.savefig('training_curves.png')
```

## Command Line Training

```bash
# Basic training
python scripts/train/train.py --config configs/default.yaml

# With overrides
python scripts/train/train.py \
    --config configs/default.yaml \
    --epochs 200 \
    --batch_size 128 \
    --learning_rate 0.0005

# Resume from checkpoint
python scripts/train/train.py \
    --config configs/default.yaml \
    --resume checkpoints/epoch_50.pt
```

## Next Steps

| Goal | Resource |
|------|----------|
| Understand the math | [[Geometry]] |
| Customize losses | [[Loss-Functions]] |
| Full training guide | [[Training]] |
| Tune hyperparameters | [[Configuration]] |
| Debug issues | [[Troubleshooting]] |

---

*See also: [[Tutorials]] for in-depth walkthroughs*
