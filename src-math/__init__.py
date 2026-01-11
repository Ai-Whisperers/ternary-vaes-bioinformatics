# TernaryVAE Mathematical Framework
#
# This package contains the pure mathematical foundation for TernaryVAE:
# - P-adic number theory and 3-adic operations
# - Hyperbolic geometry on the Poincaré ball (default)
# - Variational Autoencoder architectures
# - Mathematical loss functions and training infrastructure
#
# Version: v5.12.5 Mathematical Foundation

__version__ = "5.12.5"
__mathematical_foundation__ = True

# Core mathematical components
from .core import TERNARY, padic_distance, padic_valuation
from .geometry import poincare_distance, exp_map_zero, get_manifold
from .models import TernaryVAEV5_11_PartialFreeze, HomeostasisController
from .losses import RichHierarchyLoss, PAdicGeodesicLoss
from .training import Trainer, TrainingMonitor

__all__ = [
    # Core mathematics
    'TERNARY', 'padic_distance', 'padic_valuation',
    # Hyperbolic geometry (default)
    'poincare_distance', 'exp_map_zero', 'get_manifold',
    # VAE models
    'TernaryVAEV5_11_PartialFreeze', 'HomeostasisController',
    # Mathematical losses
    'RichHierarchyLoss', 'PAdicGeodesicLoss',
    # Training infrastructure
    'Trainer', 'TrainingMonitor'
]