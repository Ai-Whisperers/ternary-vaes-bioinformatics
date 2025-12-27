# Tutorial: Hyperbolic Graph Neural Networks

This tutorial covers hyperbolic geometry for learning hierarchical representations of biological networks.

## Why Hyperbolic Space?

Biological data is often **hierarchical**:
- Protein families → subfamilies → individual proteins
- Phylogenetic trees
- Protein domain organization
- Codon → amino acid → protein hierarchy

Euclidean space struggles with hierarchies. Tree structures require exponentially growing space. **Hyperbolic space** grows exponentially by nature - perfect for trees!

## Hyperbolic Geometry Basics

### The Poincare Ball

The Poincare ball is the unit disk/ball where:
- Center = root of hierarchy
- Boundary = leaves (infinitely far from center)
- Distance grows exponentially toward boundary

```python
from src.graphs import PoincareOperations

poincare = PoincareOperations(curvature=1.0)

# Points must have norm < 1
x = torch.randn(10, 8) * 0.3  # Stay away from boundary
y = torch.randn(10, 8) * 0.3

# Hyperbolic distance (much larger than Euclidean near boundary)
dist = poincare.distance(x, y)
print(f"Hyperbolic distance: {dist.mean():.3f}")
```

### Key Operations

**Mobius Addition** (hyperbolic "plus"):
```python
z = poincare.mobius_add(x, y)  # x ⊕ y in hyperbolic space
```

**Exponential Map** (Euclidean → Hyperbolic):
```python
tangent_vector = torch.randn(10, 8) * 0.1
base_point = torch.zeros(1, 8)
hyperbolic_point = poincare.exp_map(tangent_vector, base_point)
```

**Logarithmic Map** (Hyperbolic → Euclidean):
```python
tangent = poincare.log_map(hyperbolic_point, base_point)
```

### The Lorentz Model

Alternative hyperbolic model on a hyperboloid:

```python
from src.graphs import LorentzOperations

lorentz = LorentzOperations(curvature=1.0)

# Points on hyperboloid: -x0² + x1² + ... + xn² = -1/c
space = torch.randn(10, 7) * 0.3
time = torch.sqrt(1 + (space**2).sum(dim=-1, keepdim=True))
x = torch.cat([time, space], dim=-1)

# Lorentzian inner product
inner = lorentz.lorentzian_inner(x, x)  # Should be ≈ -1
```

## Hyperbolic Neural Network Layers

### Hyperbolic Linear Layer

Linear transformation in the Poincare ball:

```python
from src.graphs import HyperbolicLinear

layer = HyperbolicLinear(
    in_features=32,
    out_features=64,
    curvature=1.0,
    bias=True,
)

# Input must be in Poincare ball (norm < 1)
x = torch.randn(batch, 32) * 0.3
y = layer(x)  # Output also in Poincare ball
```

### Hyperbolic Graph Convolution

Message passing in hyperbolic space:

```python
from src.graphs import HyperbolicGraphConv

conv = HyperbolicGraphConv(
    in_channels=32,
    out_channels=64,
    curvature=1.0,
    use_attention=True,  # Hyperbolic attention weights
)

# Graph data
x = torch.randn(n_nodes, 32) * 0.3  # Node features
edge_index = torch.randint(0, n_nodes, (2, n_edges))

# Message passing in hyperbolic space
out = conv(x, edge_index)
```

### Lorentz MLP

MLP operating in Lorentz model:

```python
from src.graphs import LorentzMLP

mlp = LorentzMLP(
    in_features=32,
    hidden_features=64,
    out_features=32,
    n_layers=3,
    curvature=1.0,
)

# Input on hyperboloid
x = ...  # Lorentz vectors
out = mlp(x)
```

## Spectral Wavelet Decomposition

Multi-scale analysis on graphs:

```python
from src.graphs import SpectralWavelet

wavelet = SpectralWavelet(n_scales=4)

# Graph Laplacian
adj = build_adjacency_matrix(edge_index, n_nodes)
degree = adj.sum(dim=1)
laplacian = torch.diag(degree) - adj

# Node features
x = torch.randn(n_nodes, 16)

# Multi-scale wavelet features
# Different scales capture different frequency information
wavelets = wavelet(x, laplacian)  # (n_nodes, n_scales, features)
```

## HyboWaveNet: Full Architecture

Combines wavelets + hyperbolic GNN:

```python
from src.graphs import HyboWaveNet

model = HyboWaveNet(
    in_channels=16,
    hidden_channels=64,
    out_channels=32,
    n_scales=4,       # Wavelet scales
    curvature=1.0,    # Hyperbolic curvature
    n_layers=3,       # GNN layers
)

# Protein graph
x = torch.randn(n_nodes, 16)  # Node features
edge_index = ...

# Node-level embeddings (in hyperbolic space)
node_embeddings = model(x, edge_index)

# Graph-level embedding (for classification)
graph_embedding = model.encode_graph(x, edge_index)
```

## Practical Example: Protein Family Classification

```python
from src.graphs import HyboWaveNet
import torch.nn as nn

class ProteinFamilyClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = HyboWaveNet(
            in_channels=20,      # 20 amino acids
            hidden_channels=64,
            out_channels=32,
            n_scales=4,
            curvature=1.0,
        )
        # Classification head (in tangent space)
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, edge_index, batch):
        # Encode protein graph
        graph_emb = self.encoder.encode_graph(x, edge_index)

        # Classify (map to tangent space first for Euclidean classifier)
        logits = self.classifier(graph_emb)
        return logits

# Usage
model = ProteinFamilyClassifier(n_classes=100)
logits = model(node_features, edge_index, batch)
```

## Building Protein Graphs

```python
def build_protein_graph(ca_coords, threshold=10.0):
    """Build graph from Ca coordinates.

    Args:
        ca_coords: (n_residues, 3) Ca atom coordinates
        threshold: Distance threshold for edges (Angstroms)

    Returns:
        edge_index: (2, n_edges) edge connectivity
    """
    # Pairwise distances
    dist = torch.cdist(ca_coords, ca_coords)

    # Create edges for nearby residues
    mask = (dist < threshold) & (dist > 0)
    edge_index = mask.nonzero().T

    return edge_index

def build_knn_graph(coords, k=10):
    """Build k-nearest neighbor graph."""
    dist = torch.cdist(coords, coords)
    _, indices = dist.topk(k + 1, largest=False)  # +1 for self
    indices = indices[:, 1:]  # Remove self-loops

    n = coords.shape[0]
    src = torch.arange(n).unsqueeze(1).expand(-1, k).flatten()
    dst = indices.flatten()

    return torch.stack([src, dst])
```

## Curvature Selection

The curvature parameter controls how "curved" the space is:

| Curvature | Effect | Use Case |
|-----------|--------|----------|
| c = 0 | Euclidean (flat) | No hierarchy |
| c = 0.1 | Slightly curved | Weak hierarchy |
| c = 1.0 | Standard hyperbolic | Moderate hierarchy |
| c = 10.0 | Highly curved | Deep hierarchy |

**Learnable curvature:**
```python
class LearnableCurvatureGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_curvature = nn.Parameter(torch.zeros(1))

    @property
    def curvature(self):
        return torch.exp(self.log_curvature)  # Always positive
```

## Tips & Best Practices

1. **Keep points away from boundary**: Norm > 0.95 causes numerical issues
   ```python
   x = x.clamp(max=0.9 / x.norm(dim=-1, keepdim=True))
   ```

2. **Use learnable curvature** if unsure about hierarchy depth

3. **Gradient clipping** helps stability in hyperbolic optimization

4. **Riemannian optimizers** work better than SGD/Adam:
   ```python
   from geoopt.optim import RiemannianAdam
   optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
   ```

5. **Initialize near origin**: Start with small norms

## Next Steps

- See `scripts/examples/hyperbolic_gnn_demo.py` for runnable code
- Explore `src.topology` for topological features
- Combine with `src.physics` for spin glass analysis
