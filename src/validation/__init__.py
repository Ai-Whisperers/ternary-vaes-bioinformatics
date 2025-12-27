# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DEPRECATED: Use src.analysis.immune_validation instead."""

import warnings

warnings.warn(
    "src.validation is deprecated. Use src.analysis.immune_validation instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.analysis.immune_validation import (
    GoldilocksZoneValidator,
    ImmuneThresholdData,
    NobelImmuneValidator,
    ValidationResult,
)

__all__ = [
    "NobelImmuneValidator",
    "GoldilocksZoneValidator",
    "ImmuneThresholdData",
    "ValidationResult",
]
