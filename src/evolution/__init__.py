# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DEPRECATED: Use src.analysis.evolution instead."""

import warnings

warnings.warn(
    "src.evolution is deprecated. Use src.analysis.evolution instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.analysis.evolution import (
    AMINO_ACID_PROPERTIES,
    EscapeMutation,
    EscapePrediction,
    EvolutionaryPressure,
    MutationHotspot,
    SelectionType,
    ViralEvolutionPredictor,
)

__all__ = [
    "ViralEvolutionPredictor",
    "EscapeMutation",
    "EscapePrediction",
    "MutationHotspot",
    "EvolutionaryPressure",
    "SelectionType",
    "AMINO_ACID_PROPERTIES",
]
