# Example Scripts

This directory contains example scripts demonstrating the advanced modules in `src/`.

## Available Examples

### `diffusion_sequence_design.py`
Demonstrates discrete diffusion models for codon sequence generation:
- Unconditional sequence generation
- Structure-conditioned inverse folding
- Training loop example

```bash
python scripts/examples/diffusion_sequence_design.py
```

### `equivariant_networks.py`
Demonstrates SO(3)/SE(3)-equivariant networks and codon symmetry:
- Spherical harmonics computation
- SO(3)-equivariant layers
- EGNN (SE(3)-equivariant graph neural network)
- Codon symmetry layers (wobble, synonymy)
- Full codon transformer

```bash
python scripts/examples/equivariant_networks.py
```

### `hyperbolic_gnn_demo.py`
Demonstrates hyperbolic graph neural networks:
- Poincare ball operations (Mobius addition, exp/log maps)
- Lorentz (hyperboloid) model
- Hyperbolic linear layers
- Hyperbolic graph convolution
- Spectral wavelet decomposition
- HyboWaveNet full model

```bash
python scripts/examples/hyperbolic_gnn_demo.py
```

### `protein_design_workflow.py`
Complete end-to-end protein design pipeline:
- Structure analysis with topology
- Structure encoding with EGNN
- Codon sequence generation with diffusion
- Hyperbolic geometry analysis

```bash
python scripts/examples/protein_design_workflow.py
```

### `protein_family_classification.py`
Few-shot protein family classification with meta-learning:
- Contrastive pretraining with p-adic loss
- MAML for few-shot adaptation
- Evaluation on novel families

```bash
python scripts/examples/protein_family_classification.py
```

## Module Overview

| Module | Description | Example |
|--------|-------------|---------|
| `src.diffusion` | Discrete diffusion for codons | `diffusion_sequence_design.py`, `protein_design_workflow.py` |
| `src.equivariant` | SO(3)/SE(3) equivariant networks | `equivariant_networks.py`, `protein_design_workflow.py` |
| `src.graphs` | Hyperbolic GNNs | `hyperbolic_gnn_demo.py`, `protein_design_workflow.py` |
| `src.topology` | Persistent homology | `protein_design_workflow.py` |
| `src.contrastive` | P-adic contrastive learning | `protein_family_classification.py` |
| `src.meta` | Meta-learning (MAML, Reptile) | `protein_family_classification.py` |

## Requirements

All examples require the base package dependencies. Some modules have optional dependencies:
- `src.equivariant`: Optionally uses `e3nn` for optimized spherical harmonics
- `src.topology`: Optionally uses `ripser` or `gudhi` for persistent homology
