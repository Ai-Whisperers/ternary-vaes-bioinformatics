"""Loss function components.

This module contains loss computation separated from model architecture:
- ReconstructionLoss: Cross-entropy reconstruction loss
- KLDivergenceLoss: KL divergence with free bits support
- EntropyRegularization: Entropy-based regularization
- RepulsionLoss: Latent space repulsion
- DualVAELoss: Aggregated loss for dual VAE system
- PAdicMetricLoss: 3-adic metric alignment (Phase 1A)
- PAdicRankingLossV2: Hard negative mining + hierarchical margin (v5.8)
- PAdicRankingLossHyperbolic: Poincare distance ranking (v5.9)
- PAdicNormLoss: MSB/LSB hierarchy regularizer (Phase 1B)
- AdaptiveRankingLoss: Multi-scale ranking loss for ultrametric approximation
- HierarchicalNormLoss: MSB/LSB variance hierarchy
- CuriosityModule: Density-based exploration drive
- SymbioticBridge: MI-based coupling between VAEs
- AlgebraicClosureLoss: Homomorphism constraint
- HyperbolicPrior: Wrapped normal on Poincare ball (v5.10)
- HomeostaticHyperbolicPrior: Adaptive hyperbolic prior (v5.10)
- HyperbolicReconLoss: Geodesic reconstruction loss (v5.10)
- HomeostaticReconLoss: Adaptive reconstruction loss (v5.10)
- HyperbolicCentroidLoss: Frechet mean clustering (v5.10)
- ConsequencePredictor: Purpose feedback (experimental)
"""

from .dual_vae_loss import (
    ReconstructionLoss,
    KLDivergenceLoss,
    EntropyRegularization,
    RepulsionLoss,
    DualVAELoss
)

from .padic_losses import (
    PAdicMetricLoss,
    PAdicRankingLoss,
    PAdicRankingLossV2,
    PAdicRankingLossHyperbolic,
    PAdicNormLoss
)

from .appetitive_losses import (
    AdaptiveRankingLoss,
    HierarchicalNormLoss,
    CuriosityModule,
    SymbioticBridge,
    AlgebraicClosureLoss,
    ViolationBuffer
)

from .hyperbolic_prior import (
    HyperbolicPrior,
    HomeostaticHyperbolicPrior
)

from .hyperbolic_recon import (
    HyperbolicReconLoss,
    HomeostaticReconLoss,
    HyperbolicCentroidLoss
)

from .consequence_predictor import (
    ConsequencePredictor,
    evaluate_addition_accuracy
)

__all__ = [
    'ReconstructionLoss',
    'KLDivergenceLoss',
    'EntropyRegularization',
    'RepulsionLoss',
    'DualVAELoss',
    'PAdicMetricLoss',
    'PAdicRankingLoss',
    'PAdicRankingLossV2',
    'PAdicRankingLossHyperbolic',
    'PAdicNormLoss',
    'AdaptiveRankingLoss',
    'HierarchicalNormLoss',
    'CuriosityModule',
    'SymbioticBridge',
    'AlgebraicClosureLoss',
    'ViolationBuffer',
    'HyperbolicPrior',
    'HomeostaticHyperbolicPrior',
    'HyperbolicReconLoss',
    'HomeostaticReconLoss',
    'HyperbolicCentroidLoss',
    'ConsequencePredictor',
    'evaluate_addition_accuracy'
]
