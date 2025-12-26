# Encoders Module

Specialized encoders for biological sequence and structure embedding.

## Purpose

This module provides a collection of encoders that transform biological data into geometric representations suitable for the p-adic hyperbolic framework.

## Encoder Categories

### Codon Encoding

P-adic encoding of DNA/RNA codons:

```python
from src.encoders import CodonEncoder

encoder = CodonEncoder(prime=3)

# Encode a sequence
embeddings = encoder.encode_sequence("ATGGCGATC")

# Encode individual codons
embedding = encoder.encode_codon("ATG")
```

### Post-Translational Modification (PTM)

```python
from src.encoders import PTMGoldilocksEncoder, GoldilocksZone, PTMType

encoder = PTMGoldilocksEncoder()

# Encode PTM sites with Goldilocks zone
embedding = encoder.encode(
    site="K123",
    ptm_type=PTMType.ACETYLATION,
    goldilocks_zone=GoldilocksZone.STABLE
)
```

### Motor Proteins / Ternary Logic

```python
from src.encoders import TernaryMotorEncoder, ATPSynthaseEncoder

# Encode ternary motor states
encoder = TernaryMotorEncoder()
state = encoder.encode(rotation_angle=120.0)

# ATP synthase specific
atp_encoder = ATPSynthaseEncoder()
embedding = atp_encoder.encode(gamma_rotation=240.0)
```

### Circadian / Toroidal

Temporal cycle encoding on torus:

```python
from src.encoders import CircadianCycleEncoder, ToroidalEmbedding

# Encode circadian time
encoder = CircadianCycleEncoder()
embedding = encoder.encode(hour=14.5)

# General toroidal embedding
torus = ToroidalEmbedding(dim=2)
point = torus.embed(theta1=1.5, theta2=2.3)
```

### Spectral / Holographic

Multi-scale spectral graph encoding:

```python
from src.encoders import (
    HolographicEncoder,
    GraphLaplacianEncoder,
    PPINetworkEncoder,
)

# Holographic spectral encoding
encoder = HolographicEncoder(n_scales=5)
embedding = encoder.encode(adjacency_matrix)

# PPI network encoding
ppi_encoder = PPINetworkEncoder()
node_embeddings = ppi_encoder.encode(ppi_graph)
```

### Diffusion Maps

Nonlinear dimensionality reduction:

```python
from src.encoders import DiffusionMapEncoder, DiffusionPseudotime

# Diffusion map embedding
encoder = DiffusionMapEncoder(n_components=10, alpha=0.5)
embedding = encoder.fit_transform(data)

# Pseudotime inference
pseudotime = DiffusionPseudotime()
trajectory = pseudotime.compute(embeddings, root_cell=0)
```

### Geometric Vector Perceptron (GVP)

SE(3)-equivariant neural networks:

```python
from src.encoders import GVPLayer, ProteinGVPEncoder, PAdicGVP

# Protein structure encoding
encoder = ProteinGVPEncoder(hidden_dim=64)
embedding = encoder.encode(coordinates, sequence)

# P-adic enhanced GVP
padic_gvp = PAdicGVP(hidden_dim=64)
embedding = padic_gvp(node_features, edge_features)
```

### Surface / MaSIF-style

Molecular surface fingerprinting:

```python
from src.encoders import (
    MaSIFEncoder,
    SurfacePatchEncoder,
    GeodesicConv,
)

# Full MaSIF-style encoding
encoder = MaSIFEncoder()
surface_embedding = encoder.encode(surface_mesh)

# Local surface patch encoding
patch_encoder = SurfacePatchEncoder(radius=12.0)
patches = patch_encoder.encode(surface, centers)
```

## Files

| File | Description |
|------|-------------|
| `codon_encoder.py` | P-adic codon encoding |
| `ptm_encoder.py` | Post-translational modifications |
| `motor_encoder.py` | Molecular motor states |
| `circadian_encoder.py` | Circadian/toroidal encoding |
| `holographic_encoder.py` | Spectral graph encoding |
| `diffusion_encoder.py` | Diffusion map embedding |
| `geometric_vector_perceptron.py` | GVP layers |
| `surface_encoder.py` | MaSIF-style surface encoding |

## Design Principles

1. **Geometric consistency**: All encoders produce embeddings compatible with hyperbolic operations
2. **P-adic awareness**: Encodings preserve hierarchical structure
3. **Batch support**: All encoders handle batched inputs efficiently
4. **Modularity**: Encoders can be composed for complex data types
