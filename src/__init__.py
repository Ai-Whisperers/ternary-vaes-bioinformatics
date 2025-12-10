"""Ternary VAE v5.6 - Production Package."""

__version__ = "5.6.0"
__author__ = "AI Whisperers"
__license__ = "MIT"

from .models.ternary_vae_v5_6 import DualNeuralVAEV5
from .utils.data import (
    generate_all_ternary_operations,
    TernaryOperationDataset,
    split_dataset,
    create_dataloader
)
from .utils.metrics import (
    evaluate_coverage,
    compute_latent_entropy,
    compute_diversity_score,
    CoverageTracker
)

__all__ = [
    'DualNeuralVAEV5',
    'generate_all_ternary_operations',
    'TernaryOperationDataset',
    'split_dataset',
    'create_dataloader',
    'evaluate_coverage',
    'compute_latent_entropy',
    'compute_diversity_score',
    'CoverageTracker',
]
