# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Optimization modules for sequence and structure design.

This package provides optimization algorithms for designing biological
sequences with specific properties, particularly focused on avoiding
autoimmune triggers through p-adic analysis.

Modules:
    - citrullination_optimizer: Codon optimization for citrullination safety
"""

from .citrullination_optimizer import (
    CitrullinationBoundaryOptimizer,
    CodonChoice,
    CodonContextOptimizer,
    OptimizationResult,
    PAdicBoundaryAnalyzer,
)

__all__ = [
    "CitrullinationBoundaryOptimizer",
    "PAdicBoundaryAnalyzer",
    "CodonContextOptimizer",
    "OptimizationResult",
    "CodonChoice",
]
