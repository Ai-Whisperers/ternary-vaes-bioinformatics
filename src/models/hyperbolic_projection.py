"""Hyperbolic Projection Layer for V5.11.

Trainable projection from Euclidean latent space to Poincaré ball.

Key V5.11 innovation: Separate networks for direction and radius learning.
This decouples angular structure from radial hierarchy, allowing each
to be learned independently.

Architecture:
  z_euclidean → [direction_net] → normalized direction
              → [radius_net]    → radius in [0, max_radius]
              → direction * radius = z_hyp (in Poincaré ball)

Single responsibility: Euclidean to hyperbolic projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HyperbolicProjection(nn.Module):
    """Trainable projection to Poincaré ball.

    Learns BOTH direction (angular) AND radius (hierarchy) independently.

    The key insight: in v5.5, the coverage is perfect but radial hierarchy
    is inverted. By separating direction and radius networks, we can:
    - Preserve learned angular structure (direction_net can be identity-initialized)
    - Learn correct radial hierarchy from scratch (radius_net)

    Direction: Preserves angular relationships between points
    Radius: Maps to correct position in tree hierarchy
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        max_radius: float = 0.95,
        curvature: float = 1.0,
        init_identity: bool = True,
        n_layers: int = 1,
        dropout: float = 0.0
    ):
        """Initialize HyperbolicProjection.

        Args:
            latent_dim: Dimension of input/output latent space
            hidden_dim: Hidden dimension for projection networks
            max_radius: Maximum radius in Poincaré ball (must be < 1)
            curvature: Hyperbolic curvature parameter
            init_identity: If True, initialize direction_net as identity
            n_layers: Number of hidden layers (1=shallow, 2+=deep)
            dropout: Dropout rate for regularization (default: 0.0)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_radius = max_radius
        self.curvature = curvature
        self.n_layers = n_layers
        self.dropout_rate = dropout

        # Direction network: learns angular structure
        # Output is a residual added to input, then normalized
        if n_layers == 1:
            # Original shallow network
            layers = [
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU()
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, latent_dim))
            self.direction_net = nn.Sequential(*layers)
        else:
            # Deeper network with residual-friendly structure
            layers = [
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU()
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(n_layers - 1):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU()
                ])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, latent_dim))
            self.direction_net = nn.Sequential(*layers)

        # Radius network: learns radial hierarchy
        # Output is in [0, 1], scaled to [0, max_radius]
        radius_hidden = max(32, hidden_dim // 2)
        if n_layers == 1:
            # Original shallow network
            layers = [
                nn.Linear(latent_dim, radius_hidden),
                nn.SiLU()
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.extend([
                nn.Linear(radius_hidden, 1),
                nn.Sigmoid()
            ])
            self.radius_net = nn.Sequential(*layers)
        else:
            # Deeper radius network
            layers = [
                nn.Linear(latent_dim, radius_hidden),
                nn.SiLU()
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(n_layers - 1):
                layers.extend([
                    nn.Linear(radius_hidden, radius_hidden),
                    nn.SiLU()
                ])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.extend([
                nn.Linear(radius_hidden, 1),
                nn.Sigmoid()
            ])
            self.radius_net = nn.Sequential(*layers)

        # Initialize for stability
        if init_identity:
            self._init_identity()

    def _init_identity(self):
        """Initialize direction_net as near-identity.

        This preserves the angular structure from frozen encoder initially,
        allowing the network to learn corrections rather than from scratch.
        """
        # Zero out direction_net output weights (residual starts as 0)
        with torch.no_grad():
            self.direction_net[-1].weight.zero_()
            self.direction_net[-1].bias.zero_()

            # Initialize radius to mid-range (0.5 * max_radius)
            # Bias the final layer to output ~0.5 after sigmoid
            self.radius_net[-2].bias.zero_()

    def forward(self, z_euclidean: torch.Tensor) -> torch.Tensor:
        """Project Euclidean latent to Poincaré ball.

        Args:
            z_euclidean: Euclidean latent codes (batch, latent_dim)

        Returns:
            z_hyp: Points in Poincaré ball (batch, latent_dim)
        """
        # Direction: input + learned residual, then normalize
        direction_residual = self.direction_net(z_euclidean)
        direction = F.normalize(z_euclidean + direction_residual, dim=-1)

        # Radius: learned mapping to [0, max_radius]
        radius = self.radius_net(z_euclidean) * self.max_radius

        # Combine: z_hyp = direction * radius
        z_hyp = direction * radius

        return z_hyp

    def forward_with_components(
        self,
        z_euclidean: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project with explicit direction/radius outputs.

        Useful for monitoring and debugging.

        Returns:
            Tuple of (z_hyp, direction, radius)
        """
        direction_residual = self.direction_net(z_euclidean)
        direction = F.normalize(z_euclidean + direction_residual, dim=-1)
        radius = self.radius_net(z_euclidean) * self.max_radius
        z_hyp = direction * radius

        return z_hyp, direction, radius.squeeze(-1)


class DualHyperbolicProjection(nn.Module):
    """Dual hyperbolic projections for VAE-A and VAE-B.

    Each VAE gets its own projection layer, allowing them to learn
    different positions in the hyperbolic space while maintaining
    the Three-Body opposition dynamics.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        max_radius: float = 0.95,
        curvature: float = 1.0,
        share_direction: bool = False,
        n_layers: int = 1,
        dropout: float = 0.0
    ):
        """Initialize DualHyperbolicProjection.

        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension for networks
            max_radius: Maximum Poincaré ball radius
            curvature: Hyperbolic curvature
            share_direction: If True, share direction_net between A and B
            n_layers: Number of hidden layers in projection networks
            dropout: Dropout rate for regularization (default: 0.0)
        """
        super().__init__()
        self.share_direction = share_direction
        self.n_layers = n_layers
        self.dropout_rate = dropout

        # VAE-A projection (chaotic, explores boundary)
        self.proj_A = HyperbolicProjection(
            latent_dim, hidden_dim, max_radius, curvature,
            n_layers=n_layers, dropout=dropout
        )

        if share_direction:
            # VAE-B shares direction but has own radius
            radius_hidden = max(32, hidden_dim // 2)
            layers = [
                nn.Linear(latent_dim, radius_hidden),
                nn.SiLU()
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.extend([
                nn.Linear(radius_hidden, 1),
                nn.Sigmoid()
            ])
            self.proj_B_radius = nn.Sequential(*layers)
            self.max_radius = max_radius
        else:
            # VAE-B has completely separate projection
            self.proj_B = HyperbolicProjection(
                latent_dim, hidden_dim, max_radius, curvature,
                n_layers=n_layers, dropout=dropout
            )

    def forward(
        self,
        z_A: torch.Tensor,
        z_B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project both VAE latents to Poincaré ball.

        Args:
            z_A: VAE-A Euclidean latent (batch, latent_dim)
            z_B: VAE-B Euclidean latent (batch, latent_dim)

        Returns:
            Tuple of (z_A_hyp, z_B_hyp)
        """
        z_A_hyp = self.proj_A(z_A)

        if self.share_direction:
            # Use A's direction network for B
            direction_residual = self.proj_A.direction_net(z_B)
            direction = F.normalize(z_B + direction_residual, dim=-1)
            radius = self.proj_B_radius(z_B) * self.max_radius
            z_B_hyp = direction * radius
        else:
            z_B_hyp = self.proj_B(z_B)

        return z_A_hyp, z_B_hyp


__all__ = [
    'HyperbolicProjection',
    'DualHyperbolicProjection'
]
