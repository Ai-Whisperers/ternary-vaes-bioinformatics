# Objectives Module

Multi-objective optimization functions for biological sequence design.

## Purpose

This module provides objective functions for Pareto-optimal design of:
- Vaccines
- Therapeutic proteins
- mRNA sequences
- Antibodies

Each objective returns a scalar score where **LOWER is BETTER** (for minimization).

## Objective Registry

Manage multiple objectives:

```python
from src.objectives import ObjectiveRegistry, SolubilityObjective, StabilityObjective

registry = ObjectiveRegistry()

# Register objectives with weights
registry.register("solubility", SolubilityObjective(), weight=1.0)
registry.register("stability", StabilityObjective(), weight=0.5)

# Evaluate all objectives
scores = registry.evaluate(latent_vectors, decoded_sequences)

# Get weighted sum
total = registry.weighted_sum(scores)
```

## Available Objectives

### Solubility Objective

Predict protein solubility for expression:

```python
from src.objectives import SolubilityObjective

objective = SolubilityObjective()

# Lower score = more soluble
score = objective.evaluate(sequence="MVSKGEEDNM...")
```

### Stability Objective

Thermodynamic stability prediction:

```python
from src.objectives import StabilityObjective

objective = StabilityObjective()

# Lower score = more stable
score = objective.evaluate(sequence="MVSKGEEDNM...")
```

### Binding Objective

Predicted binding affinity to target:

```python
from src.objectives import BindingObjective

objective = BindingObjective(target="ACE2")

# Lower score = stronger binding
score = objective.evaluate(sequence="MVSKGEEDNM...")
```

### Manufacturability Objective

Production feasibility:

```python
from src.objectives import ManufacturabilityObjective

objective = ManufacturabilityObjective()

# Lower score = easier to manufacture
score = objective.evaluate(
    sequence="MVSKGEEDNM...",
    expression_system="CHO"
)
```

### Production Cost Objective

Estimate production costs:

```python
from src.objectives import ProductionCostObjective

objective = ProductionCostObjective()

# Returns estimated cost per gram
cost = objective.evaluate(
    sequence="MVSKGEEDNM...",
    scale="clinical"  # or "commercial"
)
```

## Custom Objectives

Create custom objectives:

```python
from src.objectives import Objective, ObjectiveResult

class CustomObjective(Objective):
    def evaluate(self, sequence: str, **kwargs) -> ObjectiveResult:
        # Your scoring logic
        score = compute_score(sequence)

        return ObjectiveResult(
            score=score,
            details={"raw_value": score}
        )

# Register custom objective
registry.register("custom", CustomObjective())
```

## Multi-Objective Optimization

Use with Pareto optimization:

```python
from src.objectives import ObjectiveRegistry
from src.optimizers import NSGAII

registry = ObjectiveRegistry()
registry.register("solubility", SolubilityObjective())
registry.register("stability", StabilityObjective())
registry.register("binding", BindingObjective())

# Define evaluation function
def evaluate(candidate):
    scores = registry.evaluate(candidate.latent, candidate.sequence)
    return [scores[name].score for name in registry.names]

# Run Pareto optimization
optimizer = NSGAII(pop_size=100)
pareto_front = optimizer.optimize(evaluate, n_generations=50)
```

## Files

| File | Description |
|------|-------------|
| `base.py` | Objective base class and registry |
| `binding.py` | Binding affinity prediction |
| `solubility.py` | Solubility and stability |
| `manufacturability.py` | Production feasibility |

## Design Philosophy

1. **Lower is better**: All objectives minimize (for NSGA-II compatibility)
2. **Composable**: Objectives can be combined via registry
3. **Extensible**: Easy to add custom objectives
4. **Deterministic**: Same input produces same output (for caching)
