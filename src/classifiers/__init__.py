# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DEPRECATED: Use src.analysis.classifiers instead.

This module has been consolidated into src/analysis/ for better organization.
"""

import warnings

warnings.warn(
    "src.classifiers is deprecated. Use src.analysis.classifiers instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.analysis.classifiers import (
    ClassificationResult,
    CodonClassifier,
    GoldilocksZoneClassifier,
    PAdicClassifierBase,
    PAdicHierarchicalClassifier,
    PAdicKNN,
)

__all__ = [
    "ClassificationResult",
    "PAdicClassifierBase",
    "PAdicKNN",
    "GoldilocksZoneClassifier",
    "CodonClassifier",
    "PAdicHierarchicalClassifier",
]
