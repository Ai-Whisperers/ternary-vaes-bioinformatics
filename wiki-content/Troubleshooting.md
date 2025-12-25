# Troubleshooting

Solutions to common issues when using Ternary VAE.

---

## Installation Issues

### ModuleNotFoundError: No module named 'src'

**Cause**: Python can't find the project modules.

**Solutions**:

```bash
# Option 1: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Option 2: Install in editable mode
pip install -e .

# Option 3: Run from project root
cd /path/to/ternary-vaes-bioinformatics
python -m src.models.ternary_vae
```

### ImportError: geoopt not installed

**Solution**:
```bash
pip install geoopt
```

### CUDA not available

**Diagnosis**:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.version.cuda)         # Should match your CUDA
```

**Solutions**:
1. Install CUDA-enabled PyTorch:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

2. Verify NVIDIA drivers:
```bash
nvidia-smi  # Should show GPU info
```

### Microsoft Visual C++ required (Windows)

**Solution**: Install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

---

## Training Issues

### Loss is NaN

**Cause 1: Learning rate too high**
```python
# Bad
optimizer = RiemannianAdam(model.parameters(), lr=0.01)

# Good
optimizer = RiemannianAdam(model.parameters(), lr=0.001)
```

**Cause 2: Points outside Poincare ball**
```python
# Add stricter projection
config = TrainingConfig(
    geometry={"max_radius": 0.9}  # Default is 0.95
)
```

**Cause 3: Numerical instability**
```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Cause 4: Bad initialization**
```python
# Check for NaN in weights
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN in {name}")
```

### Loss not decreasing

**Cause 1: Learning rate too low**
```python
# Try higher learning rate
optimizer = RiemannianAdam(model.parameters(), lr=0.01)
```

**Cause 2: Posterior collapse**
```python
# Check KL divergence
print(f"KL: {result.components['kl']}")  # Should be > 0

# Solutions:
# 1. Use free bits
loss = KLDivergenceLossComponent(free_bits=1.0)

# 2. Lower KL weight
config = TrainingConfig(loss_weights={"kl_divergence": 0.1})

# 3. Use homeostasis
from src.models import HomeostasisController
controller = HomeostasisController(target_kl=1.0)
```

**Cause 3: Data issues**
```python
# Check data distribution
print(f"Data mean: {x.mean()}")
print(f"Data std: {x.std()}")
print(f"Data min/max: {x.min()}, {x.max()}")
```

### CUDA out of memory

**Solutions**:

```python
# 1. Reduce batch size
config = TrainingConfig(batch_size=32)  # Was 128

# 2. Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    outputs = model(x)

# 4. Reduce model size
model = TernaryVAE(latent_dim=8, hidden_dims=[256, 128])

# 5. Clear cache
torch.cuda.empty_cache()
```

### Training is very slow

**Solutions**:

```python
# 1. Use GPU
device = torch.device("cuda")
model = model.to(device)

# 2. Use DataLoader workers
loader = DataLoader(dataset, num_workers=4, pin_memory=True)

# 3. Disable gradient computation for evaluation
with torch.no_grad():
    outputs = model(x)

# 4. Use torch.compile (PyTorch 2.0+)
model = torch.compile(model)
```

### Reconstruction accuracy stuck at ~0.01%

This is ~1/19683, meaning random guessing.

**Causes and solutions**:

1. **Posterior collapse**: Model ignores latent space
   - Lower KL weight
   - Use free bits
   - Increase latent dimension

2. **Decoder too weak**:
   - Increase decoder capacity
   - Add more layers

3. **Wrong loss function**:
   - Ensure using CrossEntropyLoss, not MSE

### Latent space collapse (all points same location)

**Cause**: KL divergence is too strong.

**Solutions**:
```python
# 1. Lower KL weight
config = TrainingConfig(loss_weights={"kl_divergence": 0.01})

# 2. Use annealing
beta = min(1.0, epoch / warmup_epochs)
kl_loss = beta * kl_raw

# 3. Add repulsion loss
from src.losses import RepulsionLossComponent
registry.register("repulsion", RepulsionLossComponent(weight=0.1))
```

---

## Geometry Issues

### Points escaping Poincare ball (norm > 1)

**Cause**: Projection not applied correctly.

**Solution**:
```python
from src.geometry import project_to_poincare

# Always project after operations
z = model.encode(x)
z = project_to_poincare(z, max_radius=0.95, curvature=1.0)
```

### Gradient explosion near boundary

**Cause**: Conformal factor explodes as ||x|| → 1.

**Solutions**:
```python
# 1. Use smaller max_radius
config = TrainingConfig(geometry={"max_radius": 0.9})

# 2. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# 3. Use homeostasis controller
from src.models import HomeostasisController
```

### Curvature going negative (learnable curvature)

**Solution**:
```python
# Ensure curvature stays positive
curvature = torch.nn.functional.softplus(self.raw_curvature) + 0.1
```

---

## Data Issues

### Data loading is slow

```python
# 1. Use multiple workers
loader = DataLoader(dataset, num_workers=4)

# 2. Use memory mapping for large files
import numpy as np
data = np.load("large_file.npy", mmap_mode='r')

# 3. Pre-process and cache
torch.save(processed_data, "processed.pt")
data = torch.load("processed.pt")
```

### Sequences have different lengths

```python
# 1. Pad sequences
from torch.nn.utils.rnn import pad_sequence
padded = pad_sequence(sequences, batch_first=True)

# 2. Use fixed-length chunks
def chunk_sequence(seq, chunk_size=100):
    return [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]
```

### Invalid codon indices

```python
# Validate input range
assert (x >= 0).all() and (x < 19683).all(), "Invalid codon index"
```

---

## Callback Issues

### EarlyStopping not triggering

**Cause**: Monitor metric not in logs.

**Solution**:
```python
# Ensure metric is passed to on_epoch_end
callbacks.on_epoch_end(epoch, {
    "val_loss": val_loss,  # Must match monitor="val_loss"
})
```

### Checkpoints not saving

**Cause**: Directory doesn't exist.

**Solution**:
```python
import os
os.makedirs("checkpoints/", exist_ok=True)
```

---

## Common Error Messages

### "RuntimeError: Expected all tensors to be on the same device"

```python
# Ensure all tensors on same device
model = model.to(device)
x = x.to(device)
targets = targets.to(device)
```

### "ValueError: Expected input batch_size to match target batch_size"

```python
# Check shapes
print(f"Input: {outputs['reconstruction'].shape}")  # Should be (B, 19683)
print(f"Target: {targets.shape}")  # Should be (B,)
```

### "AssertionError: Curvature must be positive"

```python
# Check config
config = TrainingConfig(geometry={"curvature": 1.0})  # Not 0 or negative
```

---

## Getting Help

If your issue isn't listed:

1. **Search existing issues**: [GitHub Issues](https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics/issues)

2. **Open new issue** with:
   - Error message (full traceback)
   - Minimal reproduction code
   - Environment info (`pip list`, `nvidia-smi`)

3. **Ask on Discussions**: [GitHub Discussions](https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics/discussions)

---

*See also: [[FAQ]], [[Installation]]*
