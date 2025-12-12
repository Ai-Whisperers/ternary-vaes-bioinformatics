"""Metrics computation components.

This module contains evaluation metrics:
- hyperbolic: 3-adic ranking correlation using Poincare geometry (v5.10)
- coverage: Coverage evaluation (unique operations learned)
- entropy: Latent space entropy computation
- reconstruction: Reconstruction accuracy metrics
- tracking: Coverage and metrics tracking
"""

from .hyperbolic import (
    project_to_poincare,
    poincare_distance,
    compute_3adic_valuation,
    compute_ranking_correlation_hyperbolic
)

__all__ = [
    'project_to_poincare',
    'poincare_distance',
    'compute_3adic_valuation',
    'compute_ranking_correlation_hyperbolic'
]
