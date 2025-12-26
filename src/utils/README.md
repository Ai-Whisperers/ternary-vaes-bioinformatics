# Utils Module

Utility functions and performance optimizations.

## Purpose

This module provides:
- Precomputed lookup tables (P1 optimization)
- P-adic arithmetic operations
- Reproducibility utilities
- Checkpoint management
- Coverage tracking

## Precomputed LUTs (P1 Optimization)

Precomputed lookup tables for fast ternary operations:

```python
from src.utils import VALUATION_LUT, TERNARY_LUT

# Valuation LUT: index -> valuation (0-9)
val = VALUATION_LUT[index]

# Ternary LUT: index -> 9-digit ternary representation
ternary = TERNARY_LUT[index]
```

### Batch Operations

```python
from src.utils import get_valuation_batch, get_ternary_batch, get_3adic_distance_batch

# Batch valuation lookup
valuations = get_valuation_batch(indices)  # (N,)

# Batch ternary conversion
ternary = get_ternary_batch(indices)  # (N, 9)

# Batch pairwise 3-adic distance
distances = get_3adic_distance_batch(i, j)  # (N,)
```

## P-adic Operations

### Basic P-adic Arithmetic

```python
from src.utils import padic_valuation, padic_norm, padic_distance

# Valuation: largest power of p dividing x
v = padic_valuation(x, p=3)

# Norm: p^(-valuation)
norm = padic_norm(x, p=3)

# Distance: p^(-valuation(x-y))
d = padic_distance(x, y, p=3)
```

### P-adic Shift

```python
from src.utils import padic_shift, PAdicShiftResult

# Compute p-adic shift between values
result = padic_shift(original=100, modified=103, p=3)
print(result.shift_magnitude)
print(result.shift_direction)
```

### Codon P-adic Distance

```python
from src.utils import codon_padic_distance, codon_to_index

# Convert codon to index
idx = codon_to_index("ATG")

# Compute p-adic distance between codons
d = codon_padic_distance("ATG", "ATT")
```

### Sequence Encoding

```python
from src.utils import PAdicSequenceEncoder, sequence_padic_encoding

# Encode a DNA sequence
encoder = PAdicSequenceEncoder()
encoding = encoder.encode("ATGCGATCGATCG")

# Or use the function directly
encoding = sequence_padic_encoding("ATGCGATCGATCG")
```

## Reproducibility

```python
from src.utils import set_seed, get_generator

# Set global seed for reproducibility
set_seed(42)

# Get seeded random generator for a specific operation
gen = get_generator(seed=42)
random_tensor = torch.randn(10, generator=gen)
```

## Checkpoint Management

```python
from src.utils import save_checkpoint, load_checkpoint_compat

# Save checkpoint
save_checkpoint(
    path="checkpoints/model.pt",
    model=model,
    optimizer=optimizer,
    epoch=100,
    metrics={"loss": 0.5}
)

# Load checkpoint (handles backwards compatibility)
checkpoint = load_checkpoint_compat("checkpoints/model.pt")
model.load_state_dict(checkpoint["model"])
```

### Backwards Compatibility

```python
from src.utils import NumpyBackwardsCompatUnpickler, extract_model_state

# Load old checkpoints with numpy arrays
checkpoint = load_checkpoint_compat(
    "old_checkpoint.pt",
    unpickler=NumpyBackwardsCompatUnpickler
)

# Extract just the model state from complex checkpoint
state_dict = extract_model_state(checkpoint)
```

## Coverage Tracking

```python
from src.utils import CoverageTracker, evaluate_coverage

# Track unique operations during training
tracker = CoverageTracker(n_operations=19683)

# Update with batch
tracker.update(predicted_indices)

# Get coverage statistics
coverage = tracker.coverage
unique = tracker.unique_count

# Or evaluate directly
stats = evaluate_coverage(model, dataloader)
```

### Diversity Score

```python
from src.utils import compute_diversity_score, compute_latent_entropy

# Measure embedding diversity
diversity = compute_diversity_score(embeddings)

# Measure latent space entropy
entropy = compute_latent_entropy(latent_samples)
```

## Files

| File | Description |
|------|-------------|
| `ternary_lut.py` | Precomputed ternary lookup tables |
| `padic_shift.py` | P-adic arithmetic operations |
| `reproducibility.py` | Seed management |
| `checkpoint.py` | Checkpoint save/load utilities |
| `metrics.py` | Coverage and diversity metrics |

## Performance Notes

- **LUTs are precomputed**: O(1) lookup instead of O(log n) computation
- **Batch operations**: Vectorized for GPU efficiency
- **Coverage tracking**: Uses bitset for memory efficiency
