# Validation Module

Hypothesis validation against experimental data.

## Purpose

This module provides tools for validating theoretical predictions (especially the Goldilocks Zone hypothesis) against experimental immune recognition data.

## Nobel Immune Validator

Validate predictions using Nobel Prize-winning immune recognition thresholds:

```python
from src.validation import NobelImmuneValidator, ValidationResult

validator = NobelImmuneValidator()

# Validate Goldilocks Zone prediction
result = validator.validate(
    peptide="YLQPRTFLL",
    hla="HLA-A*02:01",
    predicted_zone="goldilocks"
)

print(f"Valid: {result.is_valid}")
print(f"Experimental binding: {result.experimental_binding}")
print(f"Prediction matches: {result.prediction_matches}")
```

## Goldilocks Zone Validator

Validate the Goldilocks Zone hypothesis across peptide sets:

```python
from src.validation import GoldilocksZoneValidator

validator = GoldilocksZoneValidator()

# Validate across a peptide set
results = validator.validate_set(
    peptides=["YLQPRTFLL", "RLQSLQTYV", ...],
    hla="HLA-A*02:01"
)

print(f"Accuracy: {results.accuracy:.2%}")
print(f"Precision: {results.precision:.2f}")
print(f"Recall: {results.recall:.2f}")
```

## Immune Threshold Data

Access curated immune threshold data:

```python
from src.validation import ImmuneThresholdData

data = ImmuneThresholdData()

# Get threshold for specific HLA
threshold = data.get_threshold("HLA-A*02:01")

# Get all validated thresholds
all_thresholds = data.get_all()
```

## Validation Result

Structure for validation results:

```python
from src.validation import ValidationResult

result = ValidationResult(
    is_valid=True,
    experimental_value=0.65,
    predicted_value=0.70,
    confidence=0.85,
    source="IEDB"
)

# Check if prediction is within tolerance
if result.within_tolerance(0.1):
    print("Prediction validated")
```

## Files

| File | Description |
|------|-------------|
| `nobel_immune.py` | Immune recognition validation |

## The Goldilocks Zone Hypothesis

The hypothesis states that certain peptide-HLA binding affinities create a "Goldilocks Zone" for autoimmunity:

- **Too weak**: No T cell activation
- **Too strong**: Deleted by central tolerance
- **Just right**: Escape tolerance, trigger autoimmunity

This module validates whether observed autoimmune epitopes fall within predicted Goldilocks zones.

## Data Sources

Validation uses data from:
- IEDB (Immune Epitope Database)
- Published binding affinity studies
- Clinical autoimmunity cohorts
