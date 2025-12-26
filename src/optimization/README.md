# Optimization Module

Sequence optimization for biological safety.

## Purpose

This module provides optimization algorithms for designing biological sequences with specific properties, focused on:
- Avoiding autoimmune triggers
- Optimizing citrullination boundaries
- P-adic codon selection

## Citrullination Boundary Optimizer

Optimize codons to avoid autoimmune citrullination:

```python
from src.optimization import CitrullinationBoundaryOptimizer

optimizer = CitrullinationBoundaryOptimizer()

# Optimize arginine codons to avoid citrullination risk
result = optimizer.optimize(
    sequence="MVSKRGEEDNM...",
    target_positions=[4, 15, 32],  # Arginine positions
    hla_context="DRB1*04:01"
)

print(f"Optimized sequence: {result.sequence}")
print(f"Risk reduction: {result.risk_reduction:.2%}")
```

## P-adic Boundary Analyzer

Analyze p-adic boundaries for citrullination:

```python
from src.optimization import PAdicBoundaryAnalyzer

analyzer = PAdicBoundaryAnalyzer()

# Find p-adic boundaries in codon space
boundaries = analyzer.find_boundaries(
    codon="CGG",  # Arginine
    tolerance=2
)

# Analyze boundary crossing risk
risk = analyzer.analyze_crossing_risk(
    from_codon="CGG",
    to_state="citrullinated"
)
```

## Codon Context Optimizer

Optimize codon context for expression and safety:

```python
from src.optimization import CodonContextOptimizer, CodonChoice

optimizer = CodonContextOptimizer()

# Get optimal codon for amino acid in context
choice = optimizer.choose_codon(
    amino_acid="R",  # Arginine
    upstream="ATG",
    downstream="GAA",
    criteria=["citrullination_safety", "expression"]
)

print(f"Recommended: {choice.codon}")
print(f"Score: {choice.score:.3f}")
print(f"Rationale: {choice.rationale}")
```

## Optimization Result

```python
from src.optimization import OptimizationResult

result = OptimizationResult(
    sequence="optimized_sequence...",
    original_sequence="original_sequence...",
    changes=[
        {"position": 4, "from": "CGG", "to": "AGA"},
        {"position": 15, "from": "CGC", "to": "AGG"}
    ],
    metrics={
        "citrullination_risk": 0.12,
        "expression_score": 0.85
    }
)
```

## Codon Choice

```python
from src.optimization import CodonChoice

choice = CodonChoice(
    codon="AGA",
    amino_acid="R",
    score=0.92,
    rationale="Low citrullination risk, good tRNA availability",
    alternatives=["AGG", "CGA"]
)
```

## Files

| File | Description |
|------|-------------|
| `citrullination_optimizer.py` | Citrullination-safe codon optimization |

## The Citrullination Problem

Citrullination converts arginine to citrulline:
- Occurs at certain arginine positions
- Creates neo-epitopes for autoimmunity
- Position-dependent based on surrounding sequence

P-adic analysis helps identify:
- Codon choices that reduce citrullination
- Boundaries that separate safe/risky configurations
- Optimal synonymous substitutions
