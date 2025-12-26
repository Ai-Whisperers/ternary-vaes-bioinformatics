# Evolution Module

Viral mutation prediction and evolutionary analysis.

## Purpose

This module provides tools for predicting and analyzing viral evolution, particularly:
- Immune escape mutation prediction
- Mutation hotspot detection
- Evolutionary pressure analysis
- Selection type classification

## Viral Evolution Predictor

```python
from src.evolution import ViralEvolutionPredictor, SelectionType

predictor = ViralEvolutionPredictor()

# Predict escape mutations
predictions = predictor.predict_escape_mutations(
    sequence="MFVFLVLLPLVSSQCVN...",
    immune_epitopes=["YLQPRTFLL", "RLQSLQTYV"],
    n_predictions=10
)

for pred in predictions:
    print(f"Position {pred.position}: {pred.wildtype} -> {pred.mutant}")
    print(f"  Escape probability: {pred.probability:.2f}")
    print(f"  Fitness cost: {pred.fitness_cost:.2f}")
```

## Escape Mutation Analysis

```python
from src.evolution import EscapeMutation, EscapePrediction

# Create escape mutation record
mutation = EscapeMutation(
    position=484,
    wildtype="E",
    mutant="K",
    epitope="STEIYQAGS",
    escape_score=0.85
)

# Get prediction with confidence
prediction = EscapePrediction(
    mutation=mutation,
    probability=0.85,
    fitness_cost=0.12,
    selection_type=SelectionType.POSITIVE
)
```

## Mutation Hotspot Detection

Identify regions with elevated mutation rates:

```python
from src.evolution import ViralEvolutionPredictor, MutationHotspot

predictor = ViralEvolutionPredictor()

# Find mutation hotspots
hotspots = predictor.find_hotspots(
    sequences=["SEQ1...", "SEQ2...", ...],
    window_size=20
)

for hotspot in hotspots:
    print(f"Hotspot at {hotspot.start}-{hotspot.end}")
    print(f"  Mutation rate: {hotspot.rate:.4f}")
    print(f"  Selection: {hotspot.selection_type}")
```

## Evolutionary Pressure

Analyze selection pressure at each position:

```python
from src.evolution import EvolutionaryPressure

# Compute dN/dS ratios
pressure = EvolutionaryPressure(
    sequences=alignment,
    reference="reference_seq"
)

# Get pressure at position
dnds = pressure.get_dnds(position=484)
if dnds > 1:
    print("Positive selection (adaptive)")
elif dnds < 1:
    print("Purifying selection (conserved)")
```

## Selection Types

```python
from src.evolution import SelectionType

SelectionType.POSITIVE   # dN/dS > 1, adaptive evolution
SelectionType.NEGATIVE   # dN/dS < 1, purifying selection
SelectionType.NEUTRAL    # dN/dS ≈ 1, neutral drift
```

## P-adic Integration

The predictor uses p-adic distance to assess mutation impact:
- **High p-adic distance**: Major structural change
- **Low p-adic distance**: Conservative substitution

```python
# Mutations with high p-adic distance are flagged
# as potentially having larger phenotypic effects
pred = predictor.predict_with_padic_distance(
    wildtype_codon="GAA",
    mutant_codon="AAA"
)
print(f"P-adic distance: {pred.padic_distance}")
```

## Files

| File | Description |
|------|-------------|
| `viral_evolution.py` | Viral evolution prediction |

## See Also

- `src.diseases` - Disease-specific analysis
- `src.biology` - Genetic code and amino acid properties
- `src.analysis` - CRISPR and protein landscape analysis
