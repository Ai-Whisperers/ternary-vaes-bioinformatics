"""Model definitions for Ternary VAE v5.6, v5.7, v5.10, v5.11, and Appetitive VAE."""

from .ternary_vae_v5_6 import DualNeuralVAEV5
from .ternary_vae_v5_7 import DualNeuralVAEV5_7, StateNetV3
from .ternary_vae_v5_10 import DualNeuralVAEV5_10, StateNetV4, StateNetV5
from .ternary_vae_v5_11 import TernaryVAEV5_11, TernaryVAEV5_11_OptionC, FrozenEncoder, FrozenDecoder
from .hyperbolic_projection import HyperbolicProjection, DualHyperbolicProjection
from .differentiable_controller import DifferentiableController, ThreeBodyController
from .appetitive_vae import AppetitiveDualVAE, create_appetitive_vae

__all__ = [
    # V5.6
    'DualNeuralVAEV5',
    # V5.7
    'DualNeuralVAEV5_7',
    'StateNetV3',
    # V5.10
    'DualNeuralVAEV5_10',
    'StateNetV4',
    'StateNetV5',
    # V5.11
    'TernaryVAEV5_11',
    'TernaryVAEV5_11_OptionC',
    'FrozenEncoder',
    'FrozenDecoder',
    'HyperbolicProjection',
    'DualHyperbolicProjection',
    'DifferentiableController',
    'ThreeBodyController',
    # Appetitive
    'AppetitiveDualVAE',
    'create_appetitive_vae'
]
