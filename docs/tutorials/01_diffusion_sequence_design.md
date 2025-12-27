# Tutorial: Codon Sequence Design with Diffusion Models

This tutorial covers how to use the discrete diffusion models in `src.diffusion` for generating and designing codon sequences.

## Overview

The diffusion module implements **Discrete Denoising Diffusion Probabilistic Models (D3PM)** specifically designed for codon sequences. Unlike continuous diffusion (used for images), discrete diffusion works with categorical tokens - perfect for the 64 codons in the genetic code.

## Key Concepts

### 1. Absorbing State Diffusion

In D3PM, noise is added by randomly replacing tokens with a special "absorbing" state (like a [MASK] token):

```
Original:  AUG UUU GCU AAA ...
t=10:      AUG [M] GCU [M] ...
t=50:      [M] [M] [M] [M] ...
t=100:     [M] [M] [M] [M] ...  (fully masked)
```

The model learns to reverse this process - predicting the original tokens from masked sequences.

### 2. Noise Schedules

The rate of masking is controlled by a noise schedule:

- **Linear**: Constant masking rate
- **Cosine**: Slower at start/end, faster in middle (recommended)
- **Sigmoid**: S-curve transition
- **Exponential**: Accelerating masking

## Basic Usage

### Unconditional Generation

```python
import torch
from src.diffusion import CodonDiffusion

# Create model
model = CodonDiffusion(
    n_steps=1000,        # Diffusion steps (more = better quality, slower)
    vocab_size=64,       # 64 codons
    hidden_dim=256,      # Model capacity
    n_layers=6,          # Transformer layers
    schedule_type="cosine"
)

# Generate sequences
model.eval()
with torch.no_grad():
    sequences = model.sample(
        n_samples=10,      # Number of sequences
        seq_length=100,    # Codons per sequence
        device="cuda"      # Use GPU if available
    )

print(sequences.shape)  # (10, 100) - codon indices
```

### Decoding to Nucleotides

```python
# Codon lookup table (UCAG ordering)
CODONS = [f"{b1}{b2}{b3}" for b1 in "UCAG" for b2 in "UCAG" for b3 in "UCAG"]

def decode_sequence(codon_indices):
    """Convert codon indices to nucleotide string."""
    codons = [CODONS[idx] for idx in codon_indices]
    return " ".join(codons)

# Decode first sequence
print(decode_sequence(sequences[0].tolist()))
# Output: AUG UUU GCU AAA CGU ...
```

## Training Your Own Model

### 1. Prepare Data

```python
from torch.utils.data import DataLoader, TensorDataset

# Your codon sequences as indices (0-63)
# Shape: (n_sequences, seq_length)
train_data = torch.randint(0, 64, (1000, 100))

dataset = TensorDataset(train_data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 2. Training Loop

```python
model = CodonDiffusion(
    n_steps=1000,
    vocab_size=64,
    hidden_dim=256,
    n_layers=6,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(100):
    for batch in loader:
        sequences = batch[0]

        optimizer.zero_grad()

        # Forward pass returns loss dict
        result = model.training_step(sequences)
        loss = result["loss"]

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss={result['loss']:.4f}, acc={result['accuracy']:.4f}")
```

### 3. Sampling Strategies

**DDPM (standard):** Full reverse diffusion
```python
samples = model.sample(n_samples=10, seq_length=100)
```

**DDIM (faster):** Skip steps for faster generation
```python
samples = model.sample_ddim(
    n_samples=10,
    seq_length=100,
    n_steps=50,  # Only 50 steps instead of 1000
)
```

## Structure-Conditioned Design

The real power is generating sequences that fold into a specific structure:

```python
from src.diffusion import StructureConditionedGen

# Create structure-conditioned model
model = StructureConditionedGen(
    hidden_dim=256,
    n_diffusion_steps=1000,
    n_layers=6,
    n_encoder_layers=3,
)

# Load backbone coordinates (Ca atoms)
# Shape: (batch, n_residues, 3)
backbone = torch.load("my_structure.pt")  # Or from PDB

# Design sequences for this structure
model.eval()
with torch.no_grad():
    designed_sequences = model.design(
        backbone,
        n_designs=100,  # Generate 100 candidates
    )

# Each sequence is designed to fold into the input structure!
```

### Multi-Objective Design

Optimize for multiple properties simultaneously:

```python
from src.diffusion import MultiObjectiveDesigner

designer = MultiObjectiveDesigner(
    hidden_dim=256,
    use_codon_bias=True,       # Optimize codon usage
    use_mrna_stability=True,   # Optimize mRNA stability
)

# Design with custom weights
sequences = designer.design_optimized(
    backbone,
    n_candidates=1000,
    n_select=10,
    weights={
        "structure": 1.0,      # Structure compatibility
        "codon_bias": 0.5,     # Codon optimization
        "mrna_stability": 0.3, # mRNA stability
    }
)
```

## Advanced: Custom Denoiser

You can replace the default Transformer denoiser:

```python
import torch.nn as nn
from src.diffusion import CodonDiffusion

class MyDenoiser(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        # Your custom architecture
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=3, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, t, context=None):
        h = self.embed(x)
        h, _ = self.lstm(h)
        return self.output(h)

# Use custom denoiser
model = CodonDiffusion(n_steps=1000, vocab_size=64, hidden_dim=256)
model.denoiser = MyDenoiser(64, 256)
```

## Tips & Best Practices

1. **More steps = better quality** but slower. Start with 100 steps for testing, use 1000 for production.

2. **Cosine schedule** generally works best for discrete diffusion.

3. **Temperature sampling**: Lower temperature = more conservative sequences
   ```python
   samples = model.sample(n_samples=10, seq_length=100, temperature=0.8)
   ```

4. **Batch generation** is much faster than generating one at a time.

5. **GPU acceleration** provides ~10-50x speedup for generation.

## Next Steps

- See `scripts/examples/diffusion_sequence_design.py` for runnable examples
- Check `src/diffusion/structure_gen.py` for structure encoding details
- Explore `src/equivariant/` for structure-aware neural networks
