# Core Module

Fundamental domain layer - the mathematical foundation for ternary operations.

## Purpose

This module contains the fundamental domain concepts that all other modules depend on. It defines the ternary algebra operations used throughout the codebase.

## Ternary Space

The ternary space consists of 19,683 (3^9) operations, each represented as a 9-digit base-3 number.

```python
from src.core import TERNARY, N_OPERATIONS, N_DIGITS

# Constants
print(N_OPERATIONS)  # 19683 (3^9)
print(N_DIGITS)      # 9
```

## P-adic Valuation

The p-adic valuation measures how "divisible by 3" a ternary representation is:

```python
from src.core import TERNARY

# Valuation of an index
v = TERNARY.valuation(0)      # 9 (all zeros)
v = TERNARY.valuation(13122)  # 0 (starts with non-zero)

# Batch computation
valuations = TERNARY.valuation(torch.tensor([0, 1, 3, 9]))
```

## P-adic Distance

The p-adic distance uses the valuation to measure similarity:

```python
from src.core import TERNARY

# Distance between two indices
d = TERNARY.distance(0, 1)    # Large distance (differ at position 0)
d = TERNARY.distance(0, 729)  # Small distance (same first 6 digits)

# Distance matrix for a batch
distances = TERNARY.pairwise_distance(indices)
```

## Ternary Conversion

Convert between decimal indices and ternary representations:

```python
from src.core import to_ternary, from_ternary

# Decimal to ternary (returns 9-digit tensor)
ternary = to_ternary(13122)  # tensor([2, 2, 2, 2, 2, 2, 0, 0, 0])

# Ternary to decimal
index = from_ternary(torch.tensor([2, 2, 2, 2, 2, 2, 0, 0, 0]))  # 13122
```

## The TERNARY Singleton

All operations are available through the `TERNARY` singleton:

```python
from src.core import TERNARY

# Properties
TERNARY.N_OPERATIONS   # 19683
TERNARY.N_DIGITS       # 9
TERNARY.MAX_VALUATION  # 9

# Methods
TERNARY.valuation(index)
TERNARY.distance(i, j)
TERNARY.to_ternary(index)
TERNARY.from_ternary(ternary)
```

## Mathematical Background

### P-adic Numbers

P-adic numbers provide an alternative metric on integers where "closeness" is determined by divisibility rather than magnitude. For base-3 (ternary):

- Two numbers are **close** if their difference is divisible by a high power of 3
- The **valuation** ν₃(x) counts how many times 3 divides x
- The **distance** d(x,y) = 3^(-ν₃(x-y))

### Ultrametricity

P-adic distance satisfies the **ultrametric inequality**:
```
d(x, z) ≤ max(d(x, y), d(y, z))
```

This is stronger than the triangle inequality and means:
- All triangles are isosceles
- Hierarchical structures embed with zero distortion
- Natural fit for biological hierarchies (evolutionary trees, protein families)

## Files

| File | Description |
|------|-------------|
| `ternary.py` | TernarySpace class and operations |
