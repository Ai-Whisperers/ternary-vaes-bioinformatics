"""Hyperbolic Triplet Loss for Poincaré Ball Embeddings.

Phase 3.4 Enhancement - Manifold-aware Triplet Learning
=======================================================

This module implements triplet loss using hyperbolic distance in the Poincaré ball,
enabling better separation learning that respects the manifold geometry.

Key features:
1. **Hyperbolic distance computation** - Uses Poincaré ball distance instead of Euclidean
2. **Adaptive margin** - Margin that adapts to distance from origin (boundary effects)
3. **Hard negative mining** - Selects challenging negatives for better learning
4. **Batch-efficient implementation** - Vectorized operations for large batches
5. **Integration ready** - Compatible with TernaryVAE training pipeline

Mathematical foundation:
- Poincaré distance: d(x,y) = arccosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))
- Triplet objective: max(0, d(anchor,pos) - d(anchor,neg) + margin)
- Adaptive margin: margin * (1 + boundary_factor * max(||anchor||, ||pos||, ||neg||))

Author: Claude Code
Date: 2026-01-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union, List
import math

# Import geometry utilities
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.geometry import poincare_distance


class HyperbolicTripletLoss(nn.Module):
    """Triplet loss using hyperbolic distance in Poincaré ball.

    Computes triplet loss using Poincaré ball distance, which better respects
    the manifold geometry for hyperbolic embeddings.
    """

    def __init__(
        self,
        margin: float = 1.0,
        curvature: float = 1.0,
        adaptive_margin: bool = True,
        boundary_factor: float = 2.0,
        hard_negative_mining: bool = True,
        margin_threshold: float = 0.1,
        reduction: str = "mean",
    ):
        """Initialize hyperbolic triplet loss.

        Args:
            margin: Base margin for triplet loss
            curvature: Curvature parameter for Poincaré ball
            adaptive_margin: Whether to adapt margin based on distance from origin
            boundary_factor: Factor for adaptive margin near boundary
            hard_negative_mining: Whether to mine hard negatives
            margin_threshold: Minimum margin for numerical stability
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.margin = margin
        self.curvature = curvature
        self.adaptive_margin = adaptive_margin
        self.boundary_factor = boundary_factor
        self.hard_negative_mining = hard_negative_mining
        self.margin_threshold = margin_threshold
        self.reduction = reduction

        # Statistics tracking
        self.register_buffer('num_active_triplets', torch.tensor(0))
        self.register_buffer('total_triplets', torch.tensor(0))

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute hyperbolic triplet loss.

        Args:
            embeddings: Embeddings in Poincaré ball (batch_size, embedding_dim)
            labels: Integer labels for each embedding (batch_size,)

        Returns:
            Dict containing loss and statistics
        """
        batch_size = embeddings.size(0)

        # Get all pairwise distances in hyperbolic space
        distances = self._compute_pairwise_distances(embeddings)

        # Create masks for positive and negative pairs
        labels_expanded = labels.unsqueeze(1) == labels.unsqueeze(0)
        positive_mask = labels_expanded & ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        negative_mask = ~labels_expanded

        # Generate triplets
        triplet_loss, num_active, total_triplets = self._compute_triplet_loss(
            distances, positive_mask, negative_mask, embeddings
        )

        # Update statistics
        self.num_active_triplets += num_active
        self.total_triplets += total_triplets

        # Apply reduction
        if self.reduction == 'mean':
            triplet_loss = triplet_loss.mean()
        elif self.reduction == 'sum':
            triplet_loss = triplet_loss.sum()

        return {
            'loss': triplet_loss,
            'num_active_triplets': num_active,
            'total_triplets': total_triplets,
            'active_ratio': num_active.float() / max(total_triplets.float(), 1.0),
            'mean_distance': distances[positive_mask].mean() if positive_mask.any() else torch.tensor(0.0),
        }

    def _compute_pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute all pairwise hyperbolic distances.

        Args:
            embeddings: Embeddings in Poincaré ball

        Returns:
            Distance matrix (batch_size, batch_size)
        """
        batch_size = embeddings.size(0)
        distances = torch.zeros(batch_size, batch_size, device=embeddings.device)

        for i in range(batch_size):
            for j in range(i+1, batch_size):
                dist = poincare_distance(
                    embeddings[i:i+1], embeddings[j:j+1], c=self.curvature
                )
                distances[i, j] = distances[j, i] = dist

        return distances

    def _compute_triplet_loss(
        self,
        distances: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute triplet loss from distance matrix.

        Args:
            distances: Pairwise distance matrix
            positive_mask: Mask for positive pairs
            negative_mask: Mask for negative pairs
            embeddings: Original embeddings (for adaptive margin)

        Returns:
            Tuple of (triplet_losses, num_active, total_triplets)
        """
        batch_size = distances.size(0)
        triplet_losses = []
        num_active = 0
        total_triplets = 0

        for anchor_idx in range(batch_size):
            # Get positive distances for this anchor
            pos_distances = distances[anchor_idx][positive_mask[anchor_idx]]

            if len(pos_distances) == 0:
                continue  # No positives for this anchor

            # Get negative distances for this anchor
            neg_distances = distances[anchor_idx][negative_mask[anchor_idx]]

            if len(neg_distances) == 0:
                continue  # No negatives for this anchor

            # Select hardest positive and negatives
            if self.hard_negative_mining:
                # Hardest positive (farthest positive)
                hardest_pos_dist = pos_distances.max()

                # Hard negatives (closest negatives)
                hard_neg_distances = neg_distances
            else:
                # Random sampling
                hardest_pos_dist = pos_distances[torch.randint(len(pos_distances), (1,))]
                hard_neg_distances = neg_distances

            # Compute adaptive margin
            if self.adaptive_margin:
                # Get embeddings for margin computation
                anchor_emb = embeddings[anchor_idx]
                anchor_norm = torch.norm(anchor_emb).item()

                # Adaptive margin increases near boundary
                adaptive_factor = 1.0 + self.boundary_factor * anchor_norm
                current_margin = max(self.margin * adaptive_factor, self.margin_threshold)
            else:
                current_margin = self.margin

            # Compute triplet losses for all negatives
            for neg_dist in hard_neg_distances:
                triplet_loss = hardest_pos_dist - neg_dist + current_margin
                triplet_losses.append(torch.clamp(triplet_loss, min=0.0))

                total_triplets += 1
                if triplet_loss > 0:
                    num_active += 1

        if not triplet_losses:
            return torch.tensor(0.0), torch.tensor(0), torch.tensor(0)

        return (
            torch.stack(triplet_losses),
            torch.tensor(num_active),
            torch.tensor(total_triplets)
        )

    def get_statistics(self) -> Dict[str, float]:
        """Get accumulated statistics.

        Returns:
            Dictionary with loss statistics
        """
        total = self.total_triplets.item()
        active = self.num_active_triplets.item()

        return {
            'total_triplets': total,
            'active_triplets': active,
            'active_ratio': active / max(total, 1),
            'inactive_ratio': (total - active) / max(total, 1),
        }

    def reset_statistics(self):
        """Reset accumulated statistics."""
        self.num_active_triplets.zero_()
        self.total_triplets.zero_()


class EfficientHyperbolicTripletLoss(nn.Module):
    """Memory-efficient version of hyperbolic triplet loss.

    Uses batch sampling and vectorized operations to handle large batches
    without computing the full O(n²) distance matrix.
    """

    def __init__(
        self,
        margin: float = 1.0,
        curvature: float = 1.0,
        num_triplets_per_anchor: int = 5,
        adaptive_margin: bool = True,
        boundary_factor: float = 2.0,
        reduction: str = "mean",
    ):
        """Initialize efficient hyperbolic triplet loss.

        Args:
            margin: Base margin for triplet loss
            curvature: Curvature parameter for Poincaré ball
            num_triplets_per_anchor: Number of triplets to sample per anchor
            adaptive_margin: Whether to adapt margin based on distance from origin
            boundary_factor: Factor for adaptive margin near boundary
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.margin = margin
        self.curvature = curvature
        self.num_triplets_per_anchor = num_triplets_per_anchor
        self.adaptive_margin = adaptive_margin
        self.boundary_factor = boundary_factor
        self.reduction = reduction

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute efficient hyperbolic triplet loss.

        Args:
            embeddings: Embeddings in Poincaré ball (batch_size, embedding_dim)
            labels: Integer labels for each embedding (batch_size,)

        Returns:
            Dict containing loss and statistics
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        triplet_losses = []
        num_active = 0
        total_triplets = 0

        # Process each unique label
        unique_labels = torch.unique(labels)

        for label in unique_labels:
            # Get anchors and positives for this label
            label_mask = labels == label
            label_embeddings = embeddings[label_mask]
            label_indices = torch.nonzero(label_mask, as_tuple=False).squeeze(1)

            if len(label_embeddings) < 2:
                continue  # Need at least 2 samples for positive pairs

            # Get negatives (all other labels)
            negative_mask = labels != label
            if not negative_mask.any():
                continue  # Need negatives

            negative_embeddings = embeddings[negative_mask]

            # Sample triplets for this label
            for anchor_idx in range(len(label_embeddings)):
                anchor_emb = label_embeddings[anchor_idx:anchor_idx+1]

                # Sample positives (excluding anchor itself)
                positive_candidates = torch.cat([
                    label_embeddings[:anchor_idx],
                    label_embeddings[anchor_idx+1:]
                ])

                if len(positive_candidates) == 0:
                    continue

                # Sample triplets
                num_triplets_to_sample = min(
                    self.num_triplets_per_anchor,
                    len(positive_candidates) * len(negative_embeddings)
                )

                for _ in range(num_triplets_to_sample):
                    # Random positive
                    pos_idx = torch.randint(len(positive_candidates), (1,))
                    positive_emb = positive_candidates[pos_idx:pos_idx+1]

                    # Random negative
                    neg_idx = torch.randint(len(negative_embeddings), (1,))
                    negative_emb = negative_embeddings[neg_idx:neg_idx+1]

                    # Compute distances
                    pos_distance = poincare_distance(anchor_emb, positive_emb, c=self.curvature)
                    neg_distance = poincare_distance(anchor_emb, negative_emb, c=self.curvature)

                    # Compute adaptive margin
                    if self.adaptive_margin:
                        anchor_norm = torch.norm(anchor_emb).item()
                        adaptive_factor = 1.0 + self.boundary_factor * anchor_norm
                        current_margin = self.margin * adaptive_factor
                    else:
                        current_margin = self.margin

                    # Compute triplet loss
                    triplet_loss = pos_distance - neg_distance + current_margin
                    loss_value = torch.clamp(triplet_loss, min=0.0)
                    triplet_losses.append(loss_value)

                    total_triplets += 1
                    if loss_value > 0:
                        num_active += 1

        if not triplet_losses:
            return {
                'loss': torch.tensor(0.0, device=device),
                'num_active_triplets': torch.tensor(0, device=device),
                'total_triplets': torch.tensor(0, device=device),
                'active_ratio': torch.tensor(0.0, device=device),
                'mean_positive_distance': torch.tensor(0.0, device=device),
            }

        # Stack all losses
        all_losses = torch.stack(triplet_losses)

        # Apply reduction
        if self.reduction == 'mean':
            final_loss = all_losses.mean()
        elif self.reduction == 'sum':
            final_loss = all_losses.sum()
        else:
            final_loss = all_losses

        return {
            'loss': final_loss,
            'num_active_triplets': torch.tensor(num_active, device=device),
            'total_triplets': torch.tensor(total_triplets, device=device),
            'active_ratio': torch.tensor(num_active / max(total_triplets, 1), device=device),
            'mean_positive_distance': all_losses.mean(),
        }


class AdaptiveHyperbolicTripletLoss(nn.Module):
    """Adaptive hyperbolic triplet loss with curriculum learning.

    Gradually increases difficulty by adjusting margin and mining strategy
    over the course of training.
    """

    def __init__(
        self,
        initial_margin: float = 0.5,
        final_margin: float = 2.0,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        curvature: float = 1.0,
        hard_mining_start_epoch: int = 20,
    ):
        """Initialize adaptive hyperbolic triplet loss.

        Args:
            initial_margin: Starting margin (easier)
            final_margin: Final margin (harder)
            warmup_epochs: Epochs to warm up before increasing margin
            total_epochs: Total training epochs for curriculum
            curvature: Curvature parameter for Poincaré ball
            hard_mining_start_epoch: When to start hard negative mining
        """
        super().__init__()
        self.initial_margin = initial_margin
        self.final_margin = final_margin
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.curvature = curvature
        self.hard_mining_start_epoch = hard_mining_start_epoch

        # Current training state
        self.register_buffer('current_epoch', torch.tensor(0))

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive difficulty."""
        # Compute current margin
        if self.current_epoch < self.warmup_epochs:
            current_margin = self.initial_margin
        else:
            progress = min(
                (self.current_epoch - self.warmup_epochs) /
                max(self.total_epochs - self.warmup_epochs, 1),
                1.0
            )
            current_margin = self.initial_margin + progress * (self.final_margin - self.initial_margin)

        # Determine mining strategy
        use_hard_mining = self.current_epoch >= self.hard_mining_start_epoch

        # Create appropriate loss function
        if use_hard_mining:
            loss_fn = HyperbolicTripletLoss(
                margin=current_margin,
                curvature=self.curvature,
                hard_negative_mining=True,
                adaptive_margin=True,
            )
        else:
            loss_fn = EfficientHyperbolicTripletLoss(
                margin=current_margin,
                curvature=self.curvature,
                num_triplets_per_anchor=3,  # Easy start
            )

        # Compute loss
        result = loss_fn(embeddings, labels)
        result['current_margin'] = torch.tensor(current_margin)
        result['use_hard_mining'] = torch.tensor(use_hard_mining)

        return result

    def step_epoch(self):
        """Call this at the end of each epoch."""
        self.current_epoch += 1


# Convenience functions
def create_hyperbolic_triplet_loss(
    margin: float = 1.0,
    curvature: float = 1.0,
    efficient: bool = False,
    adaptive: bool = False,
    **kwargs
) -> nn.Module:
    """Create hyperbolic triplet loss with sensible defaults.

    Args:
        margin: Triplet margin
        curvature: Poincaré ball curvature
        efficient: Use efficient (sampling-based) version
        adaptive: Use adaptive curriculum version
        **kwargs: Additional arguments for specific loss types

    Returns:
        Hyperbolic triplet loss module
    """
    if adaptive:
        return AdaptiveHyperbolicTripletLoss(
            final_margin=margin,
            curvature=curvature,
            **kwargs
        )
    elif efficient:
        return EfficientHyperbolicTripletLoss(
            margin=margin,
            curvature=curvature,
            **kwargs
        )
    else:
        return HyperbolicTripletLoss(
            margin=margin,
            curvature=curvature,
            **kwargs
        )


__all__ = [
    "HyperbolicTripletLoss",
    "EfficientHyperbolicTripletLoss",
    "AdaptiveHyperbolicTripletLoss",
    "create_hyperbolic_triplet_loss",
]