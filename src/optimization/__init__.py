# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DEPRECATED: Use src.analysis.codon_optimization instead."""

import warnings

warnings.warn(
    "src.optimization is deprecated. Use src.analysis.codon_optimization instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.analysis.codon_optimization import (
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
