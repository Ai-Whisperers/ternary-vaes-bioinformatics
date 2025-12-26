# Optimizers Module

Optimizers for hyperbolic and multi-objective training.

## Purpose

This module provides specialized optimization algorithms for:
1. Training on hyperbolic (Riemannian) manifolds
2. Multi-objective optimization with Pareto fronts

## Riemannian Optimization

### Mixed Riemannian Optimizer

Combines Euclidean and Riemannian optimization:

```python
from src.optimizers import MixedRiemannianOptimizer, OptimizerConfig

config = OptimizerConfig(
    lr=1e-3,
    riemannian_lr=1e-4,
    weight_decay=1e-5
)

optimizer = MixedRiemannianOptimizer(
    euclidean_params=model.euclidean_parameters(),
    riemannian_params=model.hyperbolic_parameters(),
    config=config
)

# Training loop
optimizer.zero_grad()
loss.backward()
optimizer.step()  # Respects manifold geometry
```

### Hyperbolic Scheduler

Learning rate scheduling for hyperbolic space:

```python
from src.optimizers import HyperbolicScheduler

scheduler = HyperbolicScheduler(
    optimizer,
    warmup_epochs=10,
    decay_rate=0.95
)

for epoch in range(100):
    train_epoch()
    scheduler.step()
```

### Factory Function

```python
from src.optimizers import create_optimizer, OptimizerConfig

config = OptimizerConfig(
    lr=1e-3,
    optimizer_type="radam",  # Riemannian Adam
    curvature=-1.0
)

optimizer = create_optimizer(model.parameters(), config)
```

## Multi-Objective Optimization

### NSGA-II

Non-dominated Sorting Genetic Algorithm II:

```python
from src.optimizers import NSGAII, NSGAConfig

config = NSGAConfig(
    pop_size=100,
    n_generations=50,
    mutation_rate=0.1,
    crossover_rate=0.9
)

nsga = NSGAII(config)

# Define objectives (list of functions)
objectives = [objective1, objective2, objective3]

# Run optimization
pareto_front = nsga.optimize(objectives)

# Get best solutions
for solution in pareto_front:
    print(f"Objectives: {solution.objectives}")
    print(f"Variables: {solution.variables}")
```

### Pareto Front Optimizer

General Pareto optimization:

```python
from src.optimizers import ParetoFrontOptimizer

optimizer = ParetoFrontOptimizer(n_objectives=3)

# Add solutions
optimizer.add(solution1)
optimizer.add(solution2)

# Get Pareto front
front = optimizer.get_pareto_front()
```

### Utility Functions

```python
from src.optimizers import fast_non_dominated_sort, compute_crowding_distance

# Sort population by dominance
fronts = fast_non_dominated_sort(population)

# Compute crowding distance for diversity
distances = compute_crowding_distance(front)
```

## Configuration

### OptimizerConfig

```python
from src.optimizers import OptimizerConfig

config = OptimizerConfig(
    lr=1e-3,                    # Learning rate
    riemannian_lr=1e-4,         # Riemannian learning rate
    weight_decay=1e-5,          # L2 regularization
    momentum=0.9,               # Momentum (if applicable)
    curvature=-1.0,             # Hyperbolic curvature
    optimizer_type="radam",     # radam, rsgd, or mixed
)
```

### NSGAConfig

```python
from src.optimizers import NSGAConfig

config = NSGAConfig(
    pop_size=100,               # Population size
    n_generations=50,           # Number of generations
    mutation_rate=0.1,          # Mutation probability
    crossover_rate=0.9,         # Crossover probability
    tournament_size=3,          # Tournament selection size
)
```

## Files

| File | Description |
|------|-------------|
| `riemannian.py` | Riemannian optimization |
| `multi_objective.py` | NSGA-II and Pareto optimization |

## Mathematical Background

### Riemannian Gradient Descent

In hyperbolic space, gradient descent must respect the manifold geometry:
1. Compute Euclidean gradient
2. Project to tangent space (Riemannian gradient)
3. Retract along geodesic (exponential map)

### Pareto Optimality

A solution is Pareto-optimal if no objective can be improved without worsening another. NSGA-II finds the Pareto front efficiently using:
- Fast non-dominated sorting
- Crowding distance for diversity
- Elitist selection
