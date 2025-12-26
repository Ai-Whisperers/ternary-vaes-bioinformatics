# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Factory module for creating and configuring Ternary VAE components.

This module provides factory patterns for instantiating complex model components
with consistent configuration. It centralizes construction logic and enables:

1. Dependency injection for testing
2. Configuration-driven model creation
3. Consistent component initialization

Available Factories:
    TernaryModelFactory: Creates TernaryVAE models and components
    HyperbolicLossFactory: Creates hyperbolic loss components from config

Data Classes:
    HyperbolicLossComponents: Container for all hyperbolic loss modules

Example:
    >>> from src.factories import TernaryModelFactory
    >>> config = {"latent_dim": 16, "hidden_dim": 64}
    >>> model = TernaryModelFactory.create_model(config)

    >>> # Or create individual components for custom assembly
    >>> components = TernaryModelFactory.create_components(config)
    >>> encoder = components["encoder_A"]
    >>> projection = components["projection"]

    >>> # Create hyperbolic loss components
    >>> from src.factories import HyperbolicLossFactory
    >>> factory = HyperbolicLossFactory()
    >>> loss_components = factory.create_all(config, device="cuda")
"""

from .loss_factory import HyperbolicLossComponents, HyperbolicLossFactory
from .model_factory import TernaryModelFactory

__all__ = [
    "TernaryModelFactory",
    "HyperbolicLossFactory",
    "HyperbolicLossComponents",
]
