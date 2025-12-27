# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DEPRECATED: Use src.analysis.mrna_stability instead."""

import warnings

warnings.warn(
    "src.stability is deprecated. Use src.analysis.mrna_stability instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.analysis.mrna_stability import (
    CODON_STABILITY_SCORES,
    MFEEstimator,
    mRNAStabilityPredictor,
    SecondaryStructurePredictor,
    StabilityPrediction,
    UTROptimizer,
)

__all__ = [
    "mRNAStabilityPredictor",
    "StabilityPrediction",
    "SecondaryStructurePredictor",
    "UTROptimizer",
    "MFEEstimator",
    "CODON_STABILITY_SCORES",
]
