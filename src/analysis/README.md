# Analysis Module

Analysis tools for biological sequences and structures through geometric and p-adic lenses.

## Purpose

This module provides specialized analysis tools for:
- Geometric analysis of embeddings (hyperbolicity metrics)
- Extremophile organism codon usage patterns
- Asteroid/meteorite amino acid analysis
- CRISPR off-target landscape analysis
- Protein energy landscape analysis

## Geometric Analysis

```python
from src.analysis import compute_pairwise_distances, compute_delta_hyperbolicity

# Compute pairwise hyperbolic distances
distances = compute_pairwise_distances(embeddings)

# Measure hyperbolicity (δ-hyperbolicity, lower is more tree-like)
delta = compute_delta_hyperbolicity(embeddings)
```

## Extremophile Codon Analysis

Analyze codon usage patterns in organisms adapted to extreme environments:

```python
from src.analysis import ExtremophileCodonAnalyzer, ExtremophileCategory

analyzer = ExtremophileCodonAnalyzer()

# Analyze thermophile codon usage
result = analyzer.analyze(
    sequence="ATGGCG...",
    category=ExtremophileCategory.THERMOPHILE
)
```

## Extraterrestrial Amino Acids

Analyze amino acids found in asteroids and meteorites:

```python
from src.analysis import (
    AsteroidAminoAcidAnalyzer,
    ExtraterrestrialSample,
    AminoAcidSource
)

analyzer = AsteroidAminoAcidAnalyzer()

# Create sample from meteorite data
sample = ExtraterrestrialSample(
    source=AminoAcidSource.MURCHISON,
    amino_acids=["glycine", "alanine", "...]
)

# Analyze compatibility with terrestrial biology
result = analyzer.analyze_compatibility(sample)
```

## CRISPR Off-Target Analysis

Analyze CRISPR guide RNA off-target landscapes using hyperbolic geometry:

```python
from src.analysis import (
    CRISPROfftargetAnalyzer,
    GuideDesignOptimizer,
    HyperbolicOfftargetEmbedder,
)

# Analyze off-target sites for a guide
analyzer = CRISPROfftargetAnalyzer()
offtargets = analyzer.find_offtargets(guide_rna="ACGT...NGG")

# Embed off-targets in hyperbolic space
embedder = HyperbolicOfftargetEmbedder()
embeddings = embedder.embed(offtargets)

# Optimize guide design for specificity
optimizer = GuideDesignOptimizer()
best_guides = optimizer.optimize(target_region, num_guides=5)
```

## Protein Energy Landscape

Analyze protein folding landscapes:

```python
from src.analysis import (
    ProteinLandscapeAnalyzer,
    FoldingFunnelAnalyzer,
    TransitionStateAnalyzer,
)

# Analyze folding funnel topology
funnel = FoldingFunnelAnalyzer()
metrics = funnel.analyze(conformations, energies)

# Find transition states between conformations
ts_analyzer = TransitionStateAnalyzer()
transitions = ts_analyzer.find_transitions(state_A, state_B)
```

## Files

| File | Description |
|------|-------------|
| `geometry.py` | Geometric distance and hyperbolicity metrics |
| `extremophile_codons.py` | Extremophile codon usage analysis |
| `extraterrestrial_aminoacids.py` | Meteorite amino acid analysis |
| `crispr_offtarget.py` | CRISPR off-target landscape analysis |
| `protein_landscape.py` | Protein energy landscape analysis |

## Key Concepts

### δ-Hyperbolicity

Measures how "tree-like" a metric space is. Lower values indicate better fit for hyperbolic embedding. Biological hierarchies (evolutionary trees, protein families) typically show low δ.

### P-adic Distance

Used in CRISPR analysis to measure sequence similarity in a way that respects mutational hierarchy. Mismatches at seed region have higher p-adic weight.

### Folding Funnel

Protein energy landscapes are funnel-shaped, with native state at minimum. Hyperbolic geometry captures this funnel topology naturally.
