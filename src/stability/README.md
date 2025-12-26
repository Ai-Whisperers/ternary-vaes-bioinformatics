# Stability Module

mRNA and protein stability prediction.

## Purpose

This module provides tools for predicting stability of biological sequences:
- mRNA secondary structure and stability
- Codon optimization for expression
- UTR optimization
- Minimum free energy (MFE) estimation

## mRNA Stability Prediction

```python
from src.stability import mRNAStabilityPredictor, StabilityPrediction

predictor = mRNAStabilityPredictor()

# Predict mRNA stability
prediction = predictor.predict(
    sequence="AUGGCGAAUUCU...",
    include_structure=True
)

print(f"Half-life: {prediction.half_life:.1f} hours")
print(f"Stability score: {prediction.stability_score:.2f}")
print(f"MFE: {prediction.mfe:.2f} kcal/mol")
```

## Codon Stability Scores

Access precomputed codon stability scores:

```python
from src.stability import CODON_STABILITY_SCORES

# Get stability score for a codon
score = CODON_STABILITY_SCORES["GCU"]  # Alanine codon
```

## Secondary Structure Prediction

```python
from src.stability import SecondaryStructurePredictor

predictor = SecondaryStructurePredictor()

# Predict secondary structure
structure = predictor.predict("AUGGCGAAUUCU...")

print(f"Structure: {structure.dot_bracket}")
print(f"Base pairs: {len(structure.pairs)}")
```

## UTR Optimization

Optimize untranslated regions for expression:

```python
from src.stability import UTROptimizer

optimizer = UTROptimizer()

# Optimize 5' UTR for translation efficiency
optimized = optimizer.optimize_5utr(
    cds_start="AUGGCG...",
    target_organism="human"
)

print(f"Optimized UTR: {optimized.sequence}")
print(f"Translation efficiency: {optimized.efficiency:.2f}")
```

## MFE Estimation

Estimate minimum free energy without full folding:

```python
from src.stability import MFEEstimator

estimator = MFEEstimator()

# Quick MFE estimate
mfe = estimator.estimate(sequence="AUGGCGAAUUCU...")
print(f"Estimated MFE: {mfe:.2f} kcal/mol")
```

## Files

| File | Description |
|------|-------------|
| `mrna_stability.py` | mRNA stability prediction |

## Key Concepts

### Half-life Prediction

mRNA half-life depends on:
- Sequence composition (AU-rich elements destabilize)
- Secondary structure (stable hairpins protect from degradation)
- Codon optimality (rare codons can slow decay)

### Codon Stability Index

Each codon contributes to mRNA stability:
- **Optimal codons**: Match abundant tRNAs, stabilize mRNA
- **Non-optimal codons**: Can trigger decay pathways

### UTR Elements

5' and 3' UTRs critically affect:
- Translation initiation efficiency
- mRNA localization
- Stability and decay rates
