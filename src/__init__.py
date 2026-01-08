# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Ternary VAE Mathematical Framework.

This package provides the mathematical substrate for training Variational
Autoencoders on 3-adic (p-adic) ternary structures over the complete space
of 19,683 ternary operations (3^9).

Core Mathematical Components:
- core: Ternary algebra, p-adic mathematics, evaluation metrics
- geometry: Hyperbolic Poincaré ball operations
- models: Dual-encoder VAE architectures with homeostatic control
- losses: P-adic geodesic loss functions for structure learning
- training: Mathematical training infrastructure with freeze/unfreeze
- data: Complete ternary operation space generation
"""

__version__ = "1.0.0"
__author__ = "AI Whisperers"
__license__ = "PolyForm-Noncommercial-1.0.0"

# Core mathematical components
from .core import (
    TERNARY,
    padic_distance_vectorized,
    compute_comprehensive_metrics,
    ComprehensiveMetrics,
)

# Data generation
from .data.generation import generate_all_ternary_operations

# GPU utilities
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
except ImportError:
    DEVICE = "cpu"
    N_GPUS = 0

# Version info
version_info = tuple(map(int, __version__.split(".")))

__all__ = [
    # Core mathematics
    "TERNARY",
    "padic_distance_vectorized",
    "compute_comprehensive_metrics",
    "ComprehensiveMetrics",

    # Data generation
    "generate_all_ternary_operations",

    # Utilities
    "DEVICE",
    "N_GPUS",
    "version_info",
]