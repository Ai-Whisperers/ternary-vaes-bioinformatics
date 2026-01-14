# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Loss functions for different manifold organization types in TernaryVAE."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Union
from scipy.stats import spearmanr

from src.core import TERNARY
from src.geometry import poincare_distance


class ValuationOptimalLoss:
    """Loss function optimizing for p-adic semantic hierarchy (negative correlation)."""

    def __init__(self,
                 target_hierarchy: float = -0.80,
                 hierarchy_weight: float = 5.0,
                 richness_weight: float = 2.0,
                 separation_weight: float = 3.0,
                 curvature: float = 1.0):
        """Initialize valuation-optimal loss.

        Args:
            target_hierarchy: Target negative correlation (e.g., -0.80)
            hierarchy_weight: Weight for hierarchy loss term
            richness_weight: Weight for within-level variance preservation
            separation_weight: Weight for level separation
            curvature: Hyperbolic curvature parameter
        """
        self.target_hierarchy = target_hierarchy
        self.hierarchy_weight = hierarchy_weight
        self.richness_weight = richness_weight
        self.separation_weight = separation_weight
        self.curvature = curvature

    def __call__(self,
                 embeddings_A: torch.Tensor,
                 embeddings_B: torch.Tensor,
                 indices: torch.Tensor,
                 return_components: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute valuation-optimal loss.

        Args:
            embeddings_A: Hyperbolic embeddings from VAE-A, shape (N, latent_dim)
            embeddings_B: Hyperbolic embeddings from VAE-B, shape (N, latent_dim)
            indices: Operation indices, shape (N,)
            return_components: Whether to return loss components

        Returns:
            Total loss or dict of loss components
        """
        device = embeddings_B.device

        # Compute radii using hyperbolic distance from origin
        origin_B = torch.zeros_like(embeddings_B)
        radii_B = poincare_distance(embeddings_B, origin_B, c=self.curvature)

        # Get p-adic valuations
        valuations = TERNARY.valuation(indices).to(device).float()

        # 1. Hierarchy loss: push toward negative correlation
        hierarchy_loss = self._compute_hierarchy_loss(radii_B, valuations)

        # 2. Richness loss: preserve within-level variance
        richness_loss = self._compute_richness_loss(radii_B, valuations)

        # 3. Separation loss: ensure proper level ordering
        separation_loss = self._compute_separation_loss(radii_B, valuations)

        # Combine losses
        total_loss = (self.hierarchy_weight * hierarchy_loss +
                     self.richness_weight * richness_loss +
                     self.separation_weight * separation_loss)

        if return_components:
            return {
                'total_loss': total_loss,
                'hierarchy_loss': hierarchy_loss,
                'richness_loss': richness_loss,
                'separation_loss': separation_loss,
                'radii_B': radii_B,
                'valuations': valuations
            }
        else:
            return total_loss

    def _compute_hierarchy_loss(self, radii: torch.Tensor, valuations: torch.Tensor) -> torch.Tensor:
        """Compute loss for hierarchy correlation."""
        # Convert to numpy for scipy (better numerical stability)
        radii_np = radii.detach().cpu().numpy()
        valuations_np = valuations.detach().cpu().numpy()

        # Compute current correlation
        current_corr = spearmanr(valuations_np, radii_np)[0]
        current_corr = float(current_corr) if not np.isnan(current_corr) else 0.0

        # Loss: squared difference from target
        hierarchy_loss = (current_corr - self.target_hierarchy) ** 2

        return torch.tensor(hierarchy_loss, device=radii.device, dtype=radii.dtype)

    def _compute_richness_loss(self, radii: torch.Tensor, valuations: torch.Tensor) -> torch.Tensor:
        """Preserve within-level variance (richness)."""
        richness_losses = []

        for v in range(10):  # v=0 to v=9
            mask = (valuations == v)
            if torch.sum(mask) > 1:  # Need at least 2 points for variance
                v_radii = radii[mask]
                variance = torch.var(v_radii)
                # Encourage non-zero variance (penalize collapse)
                richness_losses.append(torch.exp(-variance * 100))  # Higher penalty for low variance

        if richness_losses:
            return torch.stack(richness_losses).mean()
        else:
            return torch.tensor(0.0, device=radii.device)

    def _compute_separation_loss(self, radii: torch.Tensor, valuations: torch.Tensor) -> torch.Tensor:
        """Encourage proper radial separation between levels."""
        # Target radii: v=0 → high radius (edge), v=9 → low radius (center)
        target_radii = 0.85 - (valuations / 9.0) * 0.75  # Linear mapping

        # MSE loss for target radii
        return F.mse_loss(radii, target_radii)


class FrequencyOptimalLoss:
    """Loss function optimizing for Shannon information efficiency (positive correlation)."""

    def __init__(self,
                 target_hierarchy: float = +0.70,
                 frequency_weight: float = 5.0,
                 density_weight: float = 3.0,
                 compression_weight: float = 2.0,
                 curvature: float = 1.0):
        """Initialize frequency-optimal loss.

        Args:
            target_hierarchy: Target positive correlation (e.g., +0.70)
            frequency_weight: Weight for frequency hierarchy loss
            density_weight: Weight for density allocation loss
            compression_weight: Weight for compression optimization
            curvature: Hyperbolic curvature parameter
        """
        self.target_hierarchy = target_hierarchy
        self.frequency_weight = frequency_weight
        self.density_weight = density_weight
        self.compression_weight = compression_weight
        self.curvature = curvature

        # Pre-compute operation frequencies
        self._setup_frequencies()

    def _setup_frequencies(self):
        """Pre-compute frequency distribution across valuation levels."""
        # Frequencies for each valuation level (based on 19,683 total operations)
        self.level_frequencies = {
            0: 13122 / 19683,  # 66.7% at v=0
            1: 4374 / 19683,   # 22.2% at v=1
            2: 1458 / 19683,   # 7.4% at v=2
            3: 486 / 19683,    # 2.5% at v=3
            4: 162 / 19683,    # 0.8% at v=4
            5: 54 / 19683,     # 0.3% at v=5
            6: 18 / 19683,     # 0.1% at v=6
            7: 6 / 19683,      # <0.1% at v=7
            8: 2 / 19683,      # <0.1% at v=8
            9: 1 / 19683       # <0.1% at v=9
        }

    def __call__(self,
                 embeddings_A: torch.Tensor,
                 embeddings_B: torch.Tensor,
                 indices: torch.Tensor,
                 return_components: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute frequency-optimal loss.

        Args:
            embeddings_A: Hyperbolic embeddings from VAE-A, shape (N, latent_dim)
            embeddings_B: Hyperbolic embeddings from VAE-B, shape (N, latent_dim)
            indices: Operation indices, shape (N,)
            return_components: Whether to return loss components

        Returns:
            Total loss or dict of loss components
        """
        device = embeddings_B.device

        # Compute radii using hyperbolic distance from origin
        origin_B = torch.zeros_like(embeddings_B)
        radii_B = poincare_distance(embeddings_B, origin_B, c=self.curvature)

        # Get p-adic valuations
        valuations = TERNARY.valuation(indices).to(device).float()

        # 1. Frequency hierarchy loss: push toward positive correlation
        frequency_loss = self._compute_frequency_hierarchy_loss(radii_B, valuations)

        # 2. Density allocation loss: volume proportional to frequency
        density_loss = self._compute_density_allocation_loss(radii_B, valuations)

        # 3. Compression loss: optimize for efficient encoding
        compression_loss = self._compute_compression_loss(radii_B, valuations)

        # Combine losses
        total_loss = (self.frequency_weight * frequency_loss +
                     self.density_weight * density_loss +
                     self.compression_weight * compression_loss)

        if return_components:
            return {
                'total_loss': total_loss,
                'frequency_loss': frequency_loss,
                'density_loss': density_loss,
                'compression_loss': compression_loss,
                'radii_B': radii_B,
                'valuations': valuations
            }
        else:
            return total_loss

    def _compute_frequency_hierarchy_loss(self, radii: torch.Tensor, valuations: torch.Tensor) -> torch.Tensor:
        """Compute loss for positive hierarchy correlation."""
        # Convert to numpy for scipy
        radii_np = radii.detach().cpu().numpy()
        valuations_np = valuations.detach().cpu().numpy()

        # Compute current correlation
        current_corr = spearmanr(valuations_np, radii_np)[0]
        current_corr = float(current_corr) if not np.isnan(current_corr) else 0.0

        # Loss: squared difference from target (positive)
        hierarchy_loss = (current_corr - self.target_hierarchy) ** 2

        return torch.tensor(hierarchy_loss, device=radii.device, dtype=radii.dtype)

    def _compute_density_allocation_loss(self, radii: torch.Tensor, valuations: torch.Tensor) -> torch.Tensor:
        """Allocate geometric volume proportional to frequency."""
        device = radii.device

        # Get target radii based on frequency (frequent → center, rare → edge)
        target_radii = torch.zeros_like(radii)

        for v in range(10):
            mask = (valuations == v)
            if torch.sum(mask) > 0:
                frequency = self.level_frequencies.get(v, 0.0)
                # High frequency → low radius (center), low frequency → high radius (edge)
                target_radius = 0.9 - (frequency * 1.2)  # Map frequency to radius
                target_radius = max(0.1, min(0.9, target_radius))  # Clamp to reasonable range
                target_radii[mask] = target_radius

        # MSE loss for density-proportional allocation
        return F.mse_loss(radii, target_radii)

    def _compute_compression_loss(self, radii: torch.Tensor, valuations: torch.Tensor) -> torch.Tensor:
        """Optimize for compression efficiency."""
        # Compression efficiency: frequent items should be easily accessible (center)
        # This is implicitly captured by density allocation, but we add explicit term

        compression_losses = []

        for v in range(10):
            mask = (valuations == v)
            if torch.sum(mask) > 0:
                v_radii = radii[mask]
                frequency = self.level_frequencies.get(v, 0.0)

                # High frequency items should have low variance (consistent placement)
                variance = torch.var(v_radii)
                # Weight by frequency: more important to have consistent placement for frequent items
                frequency_weighted_variance = variance * frequency * 10  # Scale for numerical stability
                compression_losses.append(frequency_weighted_variance)

        if compression_losses:
            return torch.stack(compression_losses).mean()
        else:
            return torch.tensor(0.0, device=radii.device)


class AdaptiveLoss:
    """Adaptive loss that lets the model choose the optimal organization type."""

    def __init__(self,
                 structure_weight: float = 3.0,
                 consistency_weight: float = 2.0,
                 curvature: float = 1.0,
                 adaptation_patience: int = 10):
        """Initialize adaptive loss.

        Args:
            structure_weight: Weight for any strong organization
            consistency_weight: Weight for maintaining chosen organization
            curvature: Hyperbolic curvature parameter
            adaptation_patience: Epochs to wait before switching organization type
        """
        self.structure_weight = structure_weight
        self.consistency_weight = consistency_weight
        self.curvature = curvature
        self.adaptation_patience = adaptation_patience

        self.current_type = None
        self.type_history = []

    def __call__(self,
                 embeddings_A: torch.Tensor,
                 embeddings_B: torch.Tensor,
                 indices: torch.Tensor,
                 return_components: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute adaptive loss.

        Args:
            embeddings_A: Hyperbolic embeddings from VAE-A, shape (N, latent_dim)
            embeddings_B: Hyperbolic embeddings from VAE-B, shape (N, latent_dim)
            indices: Operation indices, shape (N,)
            return_components: Whether to return loss components

        Returns:
            Total loss or dict of loss components
        """
        device = embeddings_B.device

        # Compute radii
        origin_B = torch.zeros_like(embeddings_B)
        radii_B = poincare_distance(embeddings_B, origin_B, c=self.curvature)

        # Get valuations
        valuations = TERNARY.valuation(indices).to(device).float()

        # Determine current organization tendency
        radii_np = radii_B.detach().cpu().numpy()
        valuations_np = valuations.detach().cpu().numpy()
        current_corr = spearmanr(valuations_np, radii_np)[0]
        current_corr = float(current_corr) if not np.isnan(current_corr) else 0.0

        # Update type history
        if abs(current_corr) > 0.3:  # Significant organization
            if current_corr < 0:
                detected_type = "valuation_optimal"
            else:
                detected_type = "frequency_optimal"
        else:
            detected_type = "unorganized"

        self.type_history.append(detected_type)

        # Determine stable organization type
        if len(self.type_history) >= self.adaptation_patience:
            recent_types = self.type_history[-self.adaptation_patience:]
            if all(t == recent_types[0] for t in recent_types) and recent_types[0] != "unorganized":
                self.current_type = recent_types[0]

        # 1. Structure loss: encourage any strong organization
        abs_corr = abs(current_corr)
        structure_loss = torch.exp(-abs_corr * 5)  # Exponential penalty for weak organization

        # 2. Consistency loss: maintain chosen organization type
        if self.current_type == "valuation_optimal":
            # Encourage negative correlation
            consistency_loss = torch.exp(current_corr * 3) if current_corr > 0 else torch.tensor(0.0, device=device)
        elif self.current_type == "frequency_optimal":
            # Encourage positive correlation
            consistency_loss = torch.exp(-current_corr * 3) if current_corr < 0 else torch.tensor(0.0, device=device)
        else:
            consistency_loss = torch.tensor(0.0, device=device)

        # Combine losses
        total_loss = (self.structure_weight * structure_loss +
                     self.consistency_weight * consistency_loss)

        if return_components:
            return {
                'total_loss': total_loss,
                'structure_loss': structure_loss,
                'consistency_loss': consistency_loss,
                'detected_type': detected_type,
                'stable_type': self.current_type,
                'current_correlation': current_corr,
                'radii_B': radii_B,
                'valuations': valuations
            }
        else:
            return total_loss


# Convenience factory function
def create_manifold_loss(manifold_type: str, **kwargs):
    """Create appropriate loss function for manifold type.

    Args:
        manifold_type: One of 'valuation_optimal', 'frequency_optimal', 'adaptive'
        **kwargs: Arguments for the specific loss function

    Returns:
        Configured loss function
    """
    if manifold_type == "valuation_optimal":
        return ValuationOptimalLoss(**kwargs)
    elif manifold_type == "frequency_optimal":
        return FrequencyOptimalLoss(**kwargs)
    elif manifold_type == "adaptive":
        return AdaptiveLoss(**kwargs)
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")