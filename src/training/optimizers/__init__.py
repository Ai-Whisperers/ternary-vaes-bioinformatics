# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Training optimizers module.

Provides specialized optimizers for training:
- MultiObjectiveOptimizer: Multi-objective optimization support (NSGA-II, Pareto)

Note: Riemannian/hyperbolic optimizers (RiemannianAdam, get_riemannian_optimizer)
are in src.geometry, not here. Use:
    from src.geometry import get_riemannian_optimizer
"""

from src.training.optimizers.multi_objective import (
    NSGAII,
    NSGAConfig,
    ParetoFrontOptimizer,
    compute_crowding_distance,
    fast_non_dominated_sort,
)

__all__ = [
    # Multi-objective optimization
    "ParetoFrontOptimizer",
    "NSGAII",
    "NSGAConfig",
    "fast_non_dominated_sort",
    "compute_crowding_distance",
]
