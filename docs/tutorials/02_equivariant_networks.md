# Tutorial: Equivariant Networks for Protein Structure

This tutorial covers the use of SO(3)/SE(3)-equivariant neural networks and codon symmetry layers for structure-aware protein modeling.

## Why Equivariance?

Proteins exist in 3D space. If you rotate a protein, its properties don't change - they're **invariant** to rotation. However, some outputs (like forces, gradients) should rotate WITH the protein - they're **equivariant**.

Standard neural networks don't respect these symmetries. Equivariant networks do, leading to:
- Better generalization
- Fewer parameters needed
- Physically meaningful representations

## Symmetry Groups

| Group | Transformation | Example |
|-------|---------------|---------|
| SO(3) | 3D Rotations | Rotating a protein |
| SE(3) | Rotations + Translations | Moving a protein in space |
| E(3) | SE(3) + Reflections | Includes mirror images |

## Spherical Harmonics Basics

Equivariant networks use **spherical harmonics** as basis functions. They're like Fourier modes but on a sphere:

```python
from src.equivariant import SphericalHarmonics

# Create spherical harmonics up to l=2
sh = SphericalHarmonics(lmax=2)

# Input: 3D directions (unit vectors)
directions = torch.randn(100, 3)
directions = directions / directions.norm(dim=-1, keepdim=True)

# Output: Spherical harmonic features
# l=0: 1 feature (scalar)
# l=1: 3 features (vector-like)
# l=2: 5 features (matrix-like)
# Total: 1 + 3 + 5 = 9 features
features = sh(directions)
print(features.shape)  # (100, 9)
```

## SO(3)-Equivariant Layers

### Basic SO(3) Layer

```python
from src.equivariant import SO3Layer

layer = SO3Layer(
    in_features=16,
    out_features=32,
    lmax=2,           # Maximum angular momentum
    use_radial=True,  # Use radial basis functions
)

# Input: node features + positions
positions = torch.randn(batch, n_nodes, 3)  # 3D coordinates
features = torch.randn(batch, n_nodes, 16)  # Node features

# Edges for message passing
edge_index = ...  # (2, n_edges)

# Forward pass
output = layer(features, positions, edge_index)
```

### SO(3) Graph Neural Network

```python
from src.equivariant import SO3GNN

model = SO3GNN(
    in_channels=16,
    hidden_channels=64,
    out_channels=32,
    n_layers=4,
    lmax=2,
)

# Protein graph
positions = torch.randn(1, 100, 3)  # 100 residues
features = torch.randn(1, 100, 16)
edge_index = build_knn_graph(positions, k=10)

# Get equivariant embeddings
embeddings = model(features, positions, edge_index)
```

## SE(3)-Equivariant Networks

SE(3) adds translation equivariance. Positions can also be updated:

### EGNN (E(n) Equivariant Graph Neural Network)

```python
from src.equivariant import EGNN

model = EGNN(
    in_channels=16,
    hidden_channels=64,
    out_channels=32,
    n_layers=4,
)

# Input
positions = torch.randn(1, 100, 3)
features = torch.randn(1, 100, 16)
edge_index = ...

# Output: updated features AND positions
new_features, new_positions = model(features, positions, edge_index)
```

### SE(3) Transformer

More powerful but computationally intensive:

```python
from src.equivariant import SE3Transformer

model = SE3Transformer(
    in_channels=16,
    hidden_channels=64,
    out_channels=32,
    n_layers=4,
    n_heads=8,
)

new_features, new_positions = model(features, positions, edge_index)
```

## Codon Symmetry Layers

Codons have biological symmetries that standard networks ignore:

1. **Synonymous codons**: Multiple codons encode the same amino acid
2. **Wobble position**: 3rd position is more flexible

### Codon Embedding

```python
from src.equivariant import CodonEmbedding

embed = CodonEmbedding(
    embed_dim=64,
    respect_synonymy=True,  # Group synonymous codons
)

# Input: codon indices (0-63)
codons = torch.randint(0, 64, (batch, seq_len))

# Output: symmetry-aware embeddings
embeddings = embed(codons)
```

### Synonymous Pooling

Pool information across synonymous codons:

```python
from src.equivariant import SynonymousPooling

pool = SynonymousPooling(hidden_dim=64)

# Input: per-codon features
codon_features = torch.randn(batch, 64, 64)  # (batch, n_codons, features)

# Output: per-amino-acid features
aa_features = pool(codon_features)  # (batch, 21, 64)
```

### Wobble-Aware Convolution

Convolution that respects wobble position flexibility:

```python
from src.equivariant import WobbleAwareConv

conv = WobbleAwareConv(
    in_channels=64,
    out_channels=64,
    wobble_weight=0.5,  # Reduce weight on 3rd position
)

# Apply to codon sequence
output = conv(codon_embeddings)
```

### Full Codon Transformer

Complete transformer respecting codon symmetries:

```python
from src.equivariant import CodonTransformer

model = CodonTransformer(
    hidden_dim=256,
    n_layers=6,
    n_heads=8,
    dropout=0.1,
)

# Input: codon sequence
codons = torch.randint(0, 64, (batch, seq_len))

# Output: contextualized representations
output = model(codons)
```

## Combining Structure and Sequence

For inverse folding (structure → sequence):

```python
from src.equivariant import EGNN, CodonTransformer
from src.diffusion import StructureConditionedGen
import torch.nn as nn

class StructureAwareSequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encode structure with EGNN
        self.structure_encoder = EGNN(
            in_channels=16,
            hidden_channels=128,
            out_channels=64,
            n_layers=4,
        )
        # Process sequence with codon transformer
        self.sequence_encoder = CodonTransformer(
            hidden_dim=64,
            n_layers=4,
        )
        # Cross-attention between structure and sequence
        self.cross_attn = nn.MultiheadAttention(64, 8)

    def forward(self, positions, codons):
        # Encode structure
        struct_feat, _ = self.structure_encoder(
            torch.zeros(positions.shape[0], positions.shape[1], 16),
            positions,
            build_graph(positions)
        )

        # Encode sequence
        seq_feat = self.sequence_encoder(codons)

        # Cross-attention
        output, _ = self.cross_attn(seq_feat, struct_feat, struct_feat)
        return output
```

## Verifying Equivariance

Test that your model is actually equivariant:

```python
import torch
from src.equivariant import SO3GNN

model = SO3GNN(in_channels=16, hidden_channels=32, out_channels=16, n_layers=2)

# Random rotation matrix
def random_rotation():
    q = torch.randn(4)
    q = q / q.norm()
    # Quaternion to rotation matrix
    r = torch.zeros(3, 3)
    r[0, 0] = 1 - 2*(q[2]**2 + q[3]**2)
    r[0, 1] = 2*(q[1]*q[2] - q[3]*q[0])
    r[0, 2] = 2*(q[1]*q[3] + q[2]*q[0])
    r[1, 0] = 2*(q[1]*q[2] + q[3]*q[0])
    r[1, 1] = 1 - 2*(q[1]**2 + q[3]**2)
    r[1, 2] = 2*(q[2]*q[3] - q[1]*q[0])
    r[2, 0] = 2*(q[1]*q[3] - q[2]*q[0])
    r[2, 1] = 2*(q[2]*q[3] + q[1]*q[0])
    r[2, 2] = 1 - 2*(q[1]**2 + q[2]**2)
    return r

R = random_rotation()

# Original input
pos = torch.randn(1, 50, 3)
feat = torch.randn(1, 50, 16)
edge_index = torch.randint(0, 50, (2, 100))

# Rotated input
pos_rot = pos @ R.T

# Outputs
out1 = model(feat, pos, edge_index)
out2 = model(feat, pos_rot, edge_index)

# For invariant outputs, should be equal
print(f"Invariance error: {(out1 - out2).abs().max():.6f}")
```

## Best Practices

1. **Start with EGNN** - simpler and often sufficient
2. **Use lmax=1 or 2** - higher is expensive and rarely needed
3. **Build sparse graphs** - k-NN with k=10-30 works well
4. **Normalize positions** - center and scale your coordinates
5. **Test equivariance** - verify with random rotations

## Next Steps

- See `src/scripts/examples/equivariant_networks.py` for runnable code
- Combine with `src.diffusion` for structure-conditioned generation
- Check `src.graphs` for hyperbolic alternatives
