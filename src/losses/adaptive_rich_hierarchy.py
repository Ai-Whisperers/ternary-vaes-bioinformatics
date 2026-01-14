"""
Adaptive RichHierarchyLoss for TernaryVAE V5.12.5

Enhanced version with dynamic loss weighting based on:
- Training progress (curriculum learning)
- Valuation difficulty (adaptive per-level weighting)
- Performance-based rebalancing

Author: Claude Code
Date: 2026-01-14
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from pathlib import Path
import warnings

from src.core import TERNARY
from src.geometry import poincare_distance


class AdaptiveWeightConfig:
    """Configuration for adaptive loss weighting."""

    def __init__(
        self,
        # Base weights (fallback)
        hierarchy_weight: float = 5.0,
        coverage_weight: float = 1.0,
        richness_weight: float = 2.0,
        separation_weight: float = 3.0,

        # Adaptive mechanisms
        enable_curriculum: bool = True,
        enable_difficulty_adaptive: bool = True,
        enable_performance_rebalancing: bool = True,

        # Curriculum learning
        curriculum_warmup_epochs: int = 10,
        curriculum_transition_epochs: int = 20,
        coverage_priority_early: float = 2.0,  # Emphasize coverage early
        hierarchy_priority_late: float = 1.5,  # Emphasize hierarchy later

        # Difficulty-adaptive weighting
        difficulty_adaptation_rate: float = 0.1,
        difficulty_smoothing: float = 0.9,  # EMA smoothing

        # Performance rebalancing
        rebalancing_interval: int = 5,  # Epochs between rebalancing
        rebalancing_sensitivity: float = 0.2,
        target_hierarchy_correlation: float = -0.83,
        target_richness_ratio: float = 0.5,
    ):
        """Initialize adaptive weighting configuration."""
        self.hierarchy_weight = hierarchy_weight
        self.coverage_weight = coverage_weight
        self.richness_weight = richness_weight
        self.separation_weight = separation_weight

        self.enable_curriculum = enable_curriculum
        self.enable_difficulty_adaptive = enable_difficulty_adaptive
        self.enable_performance_rebalancing = enable_performance_rebalancing

        self.curriculum_warmup_epochs = curriculum_warmup_epochs
        self.curriculum_transition_epochs = curriculum_transition_epochs
        self.coverage_priority_early = coverage_priority_early
        self.hierarchy_priority_late = hierarchy_priority_late

        self.difficulty_adaptation_rate = difficulty_adaptation_rate
        self.difficulty_smoothing = difficulty_smoothing

        self.rebalancing_interval = rebalancing_interval
        self.rebalancing_sensitivity = rebalancing_sensitivity
        self.target_hierarchy_correlation = target_hierarchy_correlation
        self.target_richness_ratio = target_richness_ratio


class AdaptiveRichHierarchyLoss(nn.Module):
    """Enhanced RichHierarchyLoss with adaptive weighting mechanisms.

    Key improvements over standard RichHierarchyLoss:
    1. Curriculum learning: Coverage → Hierarchy → Richness progression
    2. Difficulty-adaptive: Higher weights for harder valuation levels
    3. Performance rebalancing: Adjust weights based on training metrics
    4. Smooth transitions: Gradual weight changes to prevent instability
    """

    def __init__(
        self,
        inner_radius: float = 0.08,
        outer_radius: float = 0.9,
        min_richness_ratio: float = 0.5,
        curvature: float = 1.0,
        adaptive_config: Optional[AdaptiveWeightConfig] = None,
    ):
        super().__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.min_richness_ratio = min_richness_ratio
        self.max_valuation = 9
        self.curvature = curvature

        # Adaptive configuration
        self.adaptive_config = adaptive_config or AdaptiveWeightConfig()

        # Training state tracking
        self.register_buffer('current_epoch', torch.tensor(0))
        self.register_buffer('total_steps', torch.tensor(0))

        # Difficulty tracking per valuation level
        self.register_buffer('valuation_difficulty', torch.ones(10))  # Initialize to 1.0
        self.register_buffer('valuation_error_ema', torch.zeros(10))

        # Performance history for rebalancing
        self.performance_history = []

        # Current adaptive weights (start with base weights)
        self.register_buffer('current_hierarchy_weight',
                           torch.tensor(self.adaptive_config.hierarchy_weight))
        self.register_buffer('current_coverage_weight',
                           torch.tensor(self.adaptive_config.coverage_weight))
        self.register_buffer('current_richness_weight',
                           torch.tensor(self.adaptive_config.richness_weight))
        self.register_buffer('current_separation_weight',
                           torch.tensor(self.adaptive_config.separation_weight))

        # Precompute target radii for each valuation level
        target_radii = torch.tensor([
            outer_radius - (v / self.max_valuation) * (outer_radius - inner_radius)
            for v in range(10)
        ])
        self.register_buffer('target_radii', target_radii)

    def update_training_state(self, epoch: int, step: int):
        """Update training state for adaptive mechanisms."""
        self.current_epoch = torch.tensor(epoch, device=self.current_epoch.device)
        self.total_steps = torch.tensor(step, device=self.total_steps.device)

    def update_performance_metrics(
        self,
        hierarchy_correlation: float,
        richness_ratio: float,
        coverage_accuracy: float,
    ):
        """Update performance history for rebalancing."""
        self.performance_history.append({
            'epoch': self.current_epoch.item(),
            'hierarchy_correlation': hierarchy_correlation,
            'richness_ratio': richness_ratio,
            'coverage_accuracy': coverage_accuracy,
        })

        # Keep only recent history
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]

    def compute_curriculum_weights(self) -> Tuple[float, float, float, float]:
        """Compute curriculum-based weights based on training progress."""
        if not self.adaptive_config.enable_curriculum:
            return (
                self.adaptive_config.hierarchy_weight,
                self.adaptive_config.coverage_weight,
                self.adaptive_config.richness_weight,
                self.adaptive_config.separation_weight
            )

        epoch = self.current_epoch.item()
        warmup = self.adaptive_config.curriculum_warmup_epochs
        transition = self.adaptive_config.curriculum_transition_epochs

        if epoch < warmup:
            # Early phase: Emphasize coverage
            coverage_mult = self.adaptive_config.coverage_priority_early
            hierarchy_mult = 0.5  # Reduced hierarchy pressure
            richness_mult = 0.3   # Minimal richness pressure
            separation_mult = 1.0
        elif epoch < warmup + transition:
            # Transition phase: Gradually shift to hierarchy
            progress = (epoch - warmup) / transition
            coverage_mult = self.adaptive_config.coverage_priority_early * (1 - progress) + 1.0 * progress
            hierarchy_mult = 0.5 * (1 - progress) + self.adaptive_config.hierarchy_priority_late * progress
            richness_mult = 0.3 * (1 - progress) + 1.0 * progress
            separation_mult = 1.0
        else:
            # Late phase: Full hierarchy and richness focus
            coverage_mult = 1.0
            hierarchy_mult = self.adaptive_config.hierarchy_priority_late
            richness_mult = 1.0
            separation_mult = 1.0

        return (
            self.adaptive_config.hierarchy_weight * hierarchy_mult,
            self.adaptive_config.coverage_weight * coverage_mult,
            self.adaptive_config.richness_weight * richness_mult,
            self.adaptive_config.separation_weight * separation_mult
        )

    def update_difficulty_weights(
        self,
        valuations: torch.Tensor,
        hierarchy_errors: torch.Tensor,
    ):
        """Update difficulty-based weights for each valuation level."""
        if not self.adaptive_config.enable_difficulty_adaptive:
            return

        device = valuations.device
        unique_vals = torch.unique(valuations)

        # Compute per-valuation error rates
        for v in unique_vals:
            if 0 <= v <= 9:
                mask = valuations == v
                if mask.sum() > 0:
                    val_error = hierarchy_errors[mask].mean()

                    # Update EMA of error for this valuation (avoid in-place operations)
                    alpha = self.adaptive_config.difficulty_smoothing
                    old_ema = self.valuation_error_ema[v].detach()
                    new_ema = alpha * old_ema + (1 - alpha) * val_error.detach()

                    # Create new tensors to avoid in-place modification
                    new_error_ema = self.valuation_error_ema.clone()
                    new_error_ema[v] = new_ema
                    self.valuation_error_ema = new_error_ema

                    # Convert error to difficulty multiplier
                    # Higher error → higher difficulty → higher weight
                    difficulty = 1.0 + self.adaptive_config.difficulty_adaptation_rate * new_ema.detach()
                    new_difficulty = self.valuation_difficulty.clone()
                    new_difficulty[v] = difficulty
                    self.valuation_difficulty = new_difficulty

    def compute_performance_rebalancing(self) -> Tuple[float, float, float, float]:
        """Compute performance-based weight adjustments."""
        if (not self.adaptive_config.enable_performance_rebalancing or
            len(self.performance_history) < 5):
            return 1.0, 1.0, 1.0, 1.0

        # Check if it's time to rebalance
        epoch = self.current_epoch.item()
        if epoch % self.adaptive_config.rebalancing_interval != 0:
            return 1.0, 1.0, 1.0, 1.0

        # Analyze recent performance
        recent_metrics = self.performance_history[-5:]
        avg_hierarchy = sum(m['hierarchy_correlation'] for m in recent_metrics) / len(recent_metrics)
        avg_richness = sum(m['richness_ratio'] for m in recent_metrics) / len(recent_metrics)
        avg_coverage = sum(m['coverage_accuracy'] for m in recent_metrics) / len(recent_metrics)

        # Compute adjustment factors
        sensitivity = self.adaptive_config.rebalancing_sensitivity

        # Hierarchy adjustment
        hierarchy_gap = abs(avg_hierarchy - self.adaptive_config.target_hierarchy_correlation)
        hierarchy_mult = 1.0 + sensitivity * hierarchy_gap

        # Richness adjustment
        richness_gap = max(0, self.adaptive_config.target_richness_ratio - avg_richness)
        richness_mult = 1.0 + sensitivity * richness_gap

        # Coverage adjustment (maintain high accuracy)
        coverage_mult = 1.0 + sensitivity * max(0, 0.95 - avg_coverage)

        return hierarchy_mult, coverage_mult, richness_mult, 1.0

    def forward(
        self,
        z_hyp: torch.Tensor,
        indices: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        original_radii: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive weighting."""
        device = z_hyp.device

        # V5.12.2: Use hyperbolic distance
        origin = torch.zeros_like(z_hyp)
        radii = poincare_distance(z_hyp, origin, c=self.curvature)
        valuations = TERNARY.valuation(indices).long().to(device)

        # === Compute individual losses (same as base implementation) ===

        # 1. Hierarchy loss with difficulty weighting
        hierarchy_loss = torch.tensor(0.0, device=device)
        hierarchy_errors = torch.zeros_like(radii)
        unique_vals = torch.unique(valuations)

        for v in unique_vals:
            mask = valuations == v
            if mask.sum() > 0:
                mean_r = radii[mask].mean()
                target_r = self.target_radii[v]
                error = (mean_r - target_r) ** 2

                # Apply difficulty weighting
                difficulty_weight = self.valuation_difficulty[v] if self.adaptive_config.enable_difficulty_adaptive else 1.0
                weighted_error = error * difficulty_weight
                hierarchy_loss = hierarchy_loss + weighted_error

                # Track errors for difficulty adaptation
                hierarchy_errors[mask] = error.detach()

        hierarchy_loss = hierarchy_loss / len(unique_vals)

        # Update difficulty weights based on this batch
        self.update_difficulty_weights(valuations, hierarchy_errors)

        # 2. Coverage loss
        coverage_loss = F.cross_entropy(
            logits.view(-1, 3),
            (targets + 1).long().view(-1),
        )

        # 3. Richness loss (same as base)
        richness_loss = torch.tensor(0.0, device=device)
        if original_radii is not None:
            if original_radii.shape[0] == 19683:
                orig_radii_batch = original_radii[indices]
            else:
                orig_radii_batch = original_radii

            for v in unique_vals:
                mask = valuations == v
                if mask.sum() > 1:
                    new_var = radii[mask].var()
                    orig_var = orig_radii_batch[mask].var() + 1e-8
                    ratio = new_var / orig_var
                    if ratio < self.min_richness_ratio:
                        richness_loss = richness_loss + (self.min_richness_ratio - ratio) ** 2
            richness_loss = richness_loss / max(len(unique_vals), 1)

        # 4. Separation loss (same as base)
        separation_loss = torch.tensor(0.0, device=device)
        mean_radii_list = []
        for v in sorted(unique_vals.tolist()):
            mask = valuations == v
            if mask.sum() > 0:
                mean_radii_list.append((v, radii[mask].mean()))

        for i in range(len(mean_radii_list) - 1):
            v1, r1 = mean_radii_list[i]
            v2, r2 = mean_radii_list[i + 1]
            margin = 0.01
            violation = F.relu(r2 - r1 + margin)
            separation_loss = separation_loss + violation

        # === Adaptive Weight Computation ===

        # 1. Curriculum-based weights
        curriculum_weights = self.compute_curriculum_weights()

        # 2. Performance-based adjustments
        performance_adjustments = self.compute_performance_rebalancing()

        # 3. Combine adaptive weights
        final_hierarchy_weight = curriculum_weights[0] * performance_adjustments[0]
        final_coverage_weight = curriculum_weights[1] * performance_adjustments[1]
        final_richness_weight = curriculum_weights[2] * performance_adjustments[2]
        final_separation_weight = curriculum_weights[3] * performance_adjustments[3]

        # Update current weights (for logging) - avoid in-place operations
        self.current_hierarchy_weight = torch.tensor(final_hierarchy_weight, device=self.current_hierarchy_weight.device)
        self.current_coverage_weight = torch.tensor(final_coverage_weight, device=self.current_coverage_weight.device)
        self.current_richness_weight = torch.tensor(final_richness_weight, device=self.current_richness_weight.device)
        self.current_separation_weight = torch.tensor(final_separation_weight, device=self.current_separation_weight.device)

        # === Combine with adaptive weights ===
        total = (
            final_hierarchy_weight * hierarchy_loss +
            final_coverage_weight * coverage_loss +
            final_richness_weight * richness_loss +
            final_separation_weight * separation_loss
        )

        return {
            'total': total,
            'hierarchy_loss': hierarchy_loss,
            'coverage_loss': coverage_loss,
            'richness_loss': richness_loss,
            'separation_loss': separation_loss,
            # Adaptive weight information
            'adaptive_weights': {
                'hierarchy': final_hierarchy_weight,
                'coverage': final_coverage_weight,
                'richness': final_richness_weight,
                'separation': final_separation_weight,
            },
            'curriculum_phase': self._get_curriculum_phase(),
            'valuation_difficulty': self.valuation_difficulty.clone(),
        }

    def _get_curriculum_phase(self) -> str:
        """Get current curriculum phase for logging."""
        epoch = self.current_epoch.item()
        warmup = self.adaptive_config.curriculum_warmup_epochs
        transition = self.adaptive_config.curriculum_transition_epochs

        if epoch < warmup:
            return "coverage_focus"
        elif epoch < warmup + transition:
            return "hierarchy_transition"
        else:
            return "full_optimization"

    def extra_repr(self) -> str:
        return (
            f"inner_radius={self.inner_radius}, "
            f"outer_radius={self.outer_radius}, "
            f"adaptive_curriculum={self.adaptive_config.enable_curriculum}, "
            f"adaptive_difficulty={self.adaptive_config.enable_difficulty_adaptive}, "
            f"adaptive_rebalancing={self.adaptive_config.enable_performance_rebalancing}"
        )


def create_adaptive_rich_hierarchy_loss(config_dict: Dict) -> AdaptiveRichHierarchyLoss:
    """Create AdaptiveRichHierarchyLoss from configuration."""
    rich_hierarchy_config = config_dict.get('rich_hierarchy', {})
    adaptive_loss_config = config_dict.get('adaptive_loss', {})

    # Create adaptive config
    adaptive_config = AdaptiveWeightConfig(
        hierarchy_weight=rich_hierarchy_config.get('hierarchy_weight', 5.0),
        coverage_weight=rich_hierarchy_config.get('coverage_weight', 1.0),
        richness_weight=rich_hierarchy_config.get('richness_weight', 2.0),
        separation_weight=rich_hierarchy_config.get('separation_weight', 3.0),

        enable_curriculum=adaptive_loss_config.get('enable_curriculum', True),
        enable_difficulty_adaptive=adaptive_loss_config.get('enable_difficulty_adaptive', True),
        enable_performance_rebalancing=adaptive_loss_config.get('enable_performance_rebalancing', True),

        curriculum_warmup_epochs=adaptive_loss_config.get('curriculum_warmup_epochs', 10),
        curriculum_transition_epochs=adaptive_loss_config.get('curriculum_transition_epochs', 20),
        coverage_priority_early=adaptive_loss_config.get('coverage_priority_early', 2.0),
        hierarchy_priority_late=adaptive_loss_config.get('hierarchy_priority_late', 1.5),

        difficulty_adaptation_rate=adaptive_loss_config.get('difficulty_adaptation_rate', 0.1),
        difficulty_smoothing=adaptive_loss_config.get('difficulty_smoothing', 0.9),

        rebalancing_interval=adaptive_loss_config.get('rebalancing_interval', 5),
        rebalancing_sensitivity=adaptive_loss_config.get('rebalancing_sensitivity', 0.2),
        target_hierarchy_correlation=adaptive_loss_config.get('target_hierarchy_correlation', -0.83),
        target_richness_ratio=adaptive_loss_config.get('target_richness_ratio', 0.5),
    )

    return AdaptiveRichHierarchyLoss(
        inner_radius=rich_hierarchy_config.get('inner_radius', 0.08),
        outer_radius=rich_hierarchy_config.get('outer_radius', 0.9),
        min_richness_ratio=rich_hierarchy_config.get('min_richness_ratio', 0.5),
        curvature=1.0,  # Use model curvature
        adaptive_config=adaptive_config,
    )