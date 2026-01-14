"""Enhanced Differentiable Controller with Hierarchical Outputs.

Phase 3.1 Enhancement - Hierarchical Control Architecture
=========================================================

The original DifferentiableController outputs 6 flat control signals.
The Enhanced version organizes outputs into hierarchical levels:

1. **Strategic Level**: Long-term training objectives (hierarchy vs richness balance)
2. **Tactical Level**: Medium-term loss weighting (geodesic, radial, KL)
3. **Operational Level**: Immediate dynamics (rho injection, tau curriculum)
4. **Attention Level**: Component-wise attention weights for adaptive focus

Key improvements:
- Multi-timescale control (strategic/tactical/operational)
- Attention-based component weighting
- Hierarchical loss organization
- Backward compatibility with original controller
- Self-adaptive temperature for exploration

Author: Claude Code
Date: 2026-01-14
"""

from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# Import MLPBuilder for standardized architecture
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.nn_factory import create_controller_mlp


@dataclass
class ControllerOutputs:
    """Hierarchical controller outputs organized by control level."""

    # Strategic Level (long-term objectives)
    hierarchy_priority: torch.Tensor  # [0, 1] - focus on hierarchy vs richness
    coverage_priority: torch.Tensor   # [0, 1] - focus on coverage preservation
    exploration_rate: torch.Tensor    # [0, 1] - exploration vs exploitation

    # Tactical Level (medium-term loss weighting)
    geodesic_weight: torch.Tensor     # [0.1, inf) - geodesic loss importance
    radial_weight: torch.Tensor       # [0, inf) - radial loss importance
    regularization_strength: torch.Tensor  # [0, 1] - overall regularization

    # Operational Level (immediate dynamics)
    rho_injection: torch.Tensor       # [0, 0.5] - cross-injection strength
    tau_curriculum: torch.Tensor      # [0, 1] - curriculum position
    learning_rate_scale: torch.Tensor # [0.5, 2.0] - LR scaling factor

    # Attention Level (component-wise focus)
    encoder_attention: torch.Tensor   # [N_encoders] - attention over encoders
    loss_attention: torch.Tensor      # [N_losses] - attention over loss components
    layer_attention: torch.Tensor     # [N_layers] - attention over model layers

    # Backward compatibility (flat outputs matching original controller)
    flat_outputs: Dict[str, torch.Tensor]


class HierarchicalAttention(nn.Module):
    """Multi-head attention for component-wise control."""

    def __init__(self, input_dim: int, num_components: int, num_heads: int = 4):
        super().__init__()
        self.num_components = num_components
        self.num_heads = num_heads

        # Attention layers
        self.query = nn.Linear(input_dim, num_heads * num_components)
        self.key = nn.Linear(input_dim, num_heads * num_components)
        self.value = nn.Linear(input_dim, num_heads * num_components)

        # Output projection
        self.output_proj = nn.Linear(num_components, num_components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention weights over components.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            Attention weights (batch, num_components) - softmax normalized
        """
        batch_size = x.size(0)

        # Compute Q, K, V
        q = self.query(x).view(batch_size, self.num_heads, self.num_components)
        k = self.key(x).view(batch_size, self.num_heads, self.num_components)
        v = self.value(x).view(batch_size, self.num_heads, self.num_components)

        # Scaled dot-product attention (simplified for component attention)
        scale = (self.num_components ** -0.5)

        # Compute attention scores: q * k (element-wise, then sum over components)
        attn_scores = (q * k * scale).sum(dim=-1)  # (batch, num_heads)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, num_heads)

        # Apply attention to values: weighted sum over heads
        attended = (attn_weights.unsqueeze(-1) * v).sum(dim=1)  # (batch, num_components)

        # Project to final dimension and normalize
        output = self.output_proj(attended)
        return F.softmax(output, dim=-1)


class TemperatureAdaptive(nn.Module):
    """Adaptive temperature for exploration control."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.temp_net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute adaptive temperature.

        Args:
            x: Input features

        Returns:
            Temperature value [0.1, 2.0]
        """
        temp_logit = self.temp_net(x)
        return F.softplus(temp_logit) * 1.9 / 6.0 + 0.1  # Scale to [0.1, 2.0]


class EnhancedDifferentiableController(nn.Module):
    """Enhanced controller with hierarchical outputs and multi-timescale control.

    Architecture enhancements:
    1. Hierarchical output organization (strategic/tactical/operational/attention)
    2. Multi-head attention for component-wise control
    3. Adaptive temperature for exploration/exploitation balance
    4. Backward compatibility with original flat outputs
    5. Self-monitoring and adaptation capabilities
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,  # Increased from 32 for more capacity
        num_strategic: int = 3,
        num_tactical: int = 3,
        num_operational: int = 3,
        num_encoders: int = 2,
        num_losses: int = 5,
        num_layers: int = 4,
        attention_heads: int = 4,
        output_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        enable_hierarchical: bool = True,
    ):
        """Initialize Enhanced Differentiable Controller.

        Args:
            input_dim: Dimension of batch statistics input
            hidden_dim: Hidden layer dimension (increased for more capacity)
            num_strategic: Number of strategic-level outputs
            num_tactical: Number of tactical-level outputs
            num_operational: Number of operational-level outputs
            num_encoders: Number of encoders for attention
            num_losses: Number of loss components for attention
            num_layers: Number of model layers for attention
            attention_heads: Number of attention heads
            output_bounds: Dict of {name: (min, max)} for output clamping
            enable_hierarchical: If False, falls back to original flat controller
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.enable_hierarchical = enable_hierarchical

        # Core feature extraction network (using MLPBuilder)
        self.feature_extractor = create_controller_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )

        if enable_hierarchical:
            # Hierarchical output heads
            self.strategic_head = create_controller_mlp(hidden_dim, hidden_dim//2, num_strategic)
            self.tactical_head = create_controller_mlp(hidden_dim, hidden_dim//2, num_tactical)
            self.operational_head = create_controller_mlp(hidden_dim, hidden_dim//2, num_operational)

            # Attention modules
            self.encoder_attention = HierarchicalAttention(hidden_dim, num_encoders, attention_heads)
            self.loss_attention = HierarchicalAttention(hidden_dim, num_losses, attention_heads)
            self.layer_attention = HierarchicalAttention(hidden_dim, num_layers, attention_heads)

            # Adaptive temperature
            self.temperature_adaptive = TemperatureAdaptive(hidden_dim)

        # Backward compatibility: flat outputs (always available)
        self.flat_head = create_controller_mlp(hidden_dim, hidden_dim//2, 6)

        # Output bounds
        self.output_bounds = output_bounds or {
            "rho": (0.0, 0.5),
            "weight_geodesic": (0.1, 10.0),
            "weight_radial": (0.0, 5.0),
            "beta_A": (0.5, 5.0),
            "beta_B": (0.5, 5.0),
            "tau": (0.0, 1.0),
        }

        # Initialize for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize for stable starting values."""
        with torch.no_grad():
            # Initialize all final layers to near-zero for stable start
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    if hasattr(module, '_is_final_layer'):  # Mark final layers
                        module.weight.mul_(0.01)
                        module.bias.zero_()

            # Initialize flat head specifically for backward compatibility
            self.flat_head[-1].weight.mul_(0.01)
            self.flat_head[-1].bias.zero_()

    def _apply_bounded_activations(
        self,
        raw_outputs: torch.Tensor,
        bounds: List[Tuple[float, float]],
        names: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Apply bounded activations to raw outputs.

        Args:
            raw_outputs: Raw network outputs
            bounds: List of (min, max) bounds for each output
            names: List of output names

        Returns:
            Dict of bounded outputs
        """
        outputs = {}

        for i, (name, (min_val, max_val)) in enumerate(zip(names, bounds)):
            raw = raw_outputs[:, i]

            if min_val == 0.0 and max_val == 1.0:
                # Sigmoid for [0, 1] range
                outputs[name] = torch.sigmoid(raw)
            elif min_val == 0.0 and max_val == 0.5:
                # Sigmoid scaled for [0, 0.5] range
                outputs[name] = torch.sigmoid(raw) * 0.5
            elif min_val > 0.0 and max_val == float('inf'):
                # Softplus with offset for [min, inf) range
                outputs[name] = F.softplus(raw) + min_val
            elif min_val > 0.0 and max_val > min_val:
                # Sigmoid scaled and shifted for [min, max] range
                outputs[name] = torch.sigmoid(raw) * (max_val - min_val) + min_val
            else:
                # Default: tanh scaled for symmetric range
                midpoint = (min_val + max_val) / 2
                half_range = (max_val - min_val) / 2
                outputs[name] = torch.tanh(raw) * half_range + midpoint

        return outputs

    def forward(self, batch_stats: torch.Tensor) -> ControllerOutputs:
        """Compute hierarchical control signals from batch statistics.

        Args:
            batch_stats: Tensor of batch statistics
                [mean_radius_A, mean_radius_B, H_A, H_B, kl_A, kl_B, geo_loss, rad_loss]

        Returns:
            ControllerOutputs with hierarchical and flat outputs
        """
        # Ensure batch dimension
        if batch_stats.dim() == 1:
            batch_stats = batch_stats.unsqueeze(0)

        # Extract features
        features = self.feature_extractor(batch_stats)

        # Always compute flat outputs for backward compatibility
        flat_raw = self.flat_head(features)
        flat_outputs = {
            "rho": torch.sigmoid(flat_raw[:, 0]) * 0.5,  # [0, 0.5]
            "weight_geodesic": F.softplus(flat_raw[:, 1]) + 0.1,  # [0.1, inf)
            "weight_radial": F.softplus(flat_raw[:, 2]),  # [0, inf)
            "beta_A": F.softplus(flat_raw[:, 3]) + 0.5,  # [0.5, inf)
            "beta_B": F.softplus(flat_raw[:, 4]) + 0.5,  # [0.5, inf)
            "tau": torch.sigmoid(flat_raw[:, 5]),  # [0, 1]
        }

        if not self.enable_hierarchical:
            # Return simplified outputs for backward compatibility
            return ControllerOutputs(
                # Strategic (derived from flat outputs)
                hierarchy_priority=flat_outputs["tau"],
                coverage_priority=1.0 - flat_outputs["tau"],
                exploration_rate=torch.ones_like(flat_outputs["tau"]) * 0.5,

                # Tactical (use flat outputs)
                geodesic_weight=flat_outputs["weight_geodesic"],
                radial_weight=flat_outputs["weight_radial"],
                regularization_strength=torch.ones_like(flat_outputs["tau"]) * 0.5,

                # Operational (use flat outputs)
                rho_injection=flat_outputs["rho"],
                tau_curriculum=flat_outputs["tau"],
                learning_rate_scale=torch.ones_like(flat_outputs["tau"]),

                # Attention (uniform weights)
                encoder_attention=torch.ones(batch_stats.size(0), 2) * 0.5,
                loss_attention=torch.ones(batch_stats.size(0), 5) * 0.2,
                layer_attention=torch.ones(batch_stats.size(0), 4) * 0.25,

                # Flat outputs
                flat_outputs=flat_outputs,
            )

        # Compute hierarchical outputs
        strategic_raw = self.strategic_head(features)
        tactical_raw = self.tactical_head(features)
        operational_raw = self.operational_head(features)

        # Apply bounded activations
        strategic = self._apply_bounded_activations(
            strategic_raw,
            [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            ["hierarchy_priority", "coverage_priority", "exploration_rate"]
        )

        tactical = self._apply_bounded_activations(
            tactical_raw,
            [(0.1, 10.0), (0.0, 5.0), (0.0, 1.0)],
            ["geodesic_weight", "radial_weight", "regularization_strength"]
        )

        operational = self._apply_bounded_activations(
            operational_raw,
            [(0.0, 0.5), (0.0, 1.0), (0.5, 2.0)],
            ["rho_injection", "tau_curriculum", "learning_rate_scale"]
        )

        # Compute attention weights
        encoder_attn = self.encoder_attention(features)
        loss_attn = self.loss_attention(features)
        layer_attn = self.layer_attention(features)

        return ControllerOutputs(
            # Strategic Level
            hierarchy_priority=strategic["hierarchy_priority"],
            coverage_priority=strategic["coverage_priority"],
            exploration_rate=strategic["exploration_rate"],

            # Tactical Level
            geodesic_weight=tactical["geodesic_weight"],
            radial_weight=tactical["radial_weight"],
            regularization_strength=tactical["regularization_strength"],

            # Operational Level
            rho_injection=operational["rho_injection"],
            tau_curriculum=operational["tau_curriculum"],
            learning_rate_scale=operational["learning_rate_scale"],

            # Attention Level
            encoder_attention=encoder_attn,
            loss_attention=loss_attn,
            layer_attention=layer_attn,

            # Backward compatibility
            flat_outputs=flat_outputs,
        )

    def get_adaptive_temperature(self, batch_stats: torch.Tensor) -> torch.Tensor:
        """Get adaptive temperature for exploration control.

        Args:
            batch_stats: Input batch statistics

        Returns:
            Adaptive temperature [0.1, 2.0]
        """
        if not self.enable_hierarchical:
            return torch.ones(batch_stats.size(0), 1)

        if batch_stats.dim() == 1:
            batch_stats = batch_stats.unsqueeze(0)

        features = self.feature_extractor(batch_stats)
        return self.temperature_adaptive(features)

    def get_control_summary(self, outputs: ControllerOutputs) -> Dict[str, float]:
        """Get human-readable summary of control signals.

        Args:
            outputs: Controller outputs

        Returns:
            Dict with summary statistics
        """
        return {
            # Strategic summary
            "hierarchy_focus": float(outputs.hierarchy_priority.mean()),
            "coverage_focus": float(outputs.coverage_priority.mean()),
            "exploration_rate": float(outputs.exploration_rate.mean()),

            # Tactical summary
            "geodesic_importance": float(outputs.geodesic_weight.mean()),
            "radial_importance": float(outputs.radial_weight.mean()),
            "regularization": float(outputs.regularization_strength.mean()),

            # Operational summary
            "rho_injection": float(outputs.rho_injection.mean()),
            "curriculum_progress": float(outputs.tau_curriculum.mean()),
            "lr_scaling": float(outputs.learning_rate_scale.mean()),

            # Attention summary
            "encoder_entropy": float(-torch.sum(outputs.encoder_attention * torch.log(outputs.encoder_attention + 1e-8), dim=1).mean()),
            "loss_entropy": float(-torch.sum(outputs.loss_attention * torch.log(outputs.loss_attention + 1e-8), dim=1).mean()),
            "layer_entropy": float(-torch.sum(outputs.layer_attention * torch.log(outputs.layer_attention + 1e-8), dim=1).mean()),
        }


# Convenience functions for backward compatibility
def create_enhanced_controller(
    input_dim: int = 8,
    hidden_dim: int = 64,
    enable_hierarchical: bool = True,
    **kwargs
) -> EnhancedDifferentiableController:
    """Create enhanced controller with sensible defaults."""
    return EnhancedDifferentiableController(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        enable_hierarchical=enable_hierarchical,
        **kwargs
    )


def create_backward_compatible_controller(
    input_dim: int = 8,
    hidden_dim: int = 32,
) -> EnhancedDifferentiableController:
    """Create controller that mimics original DifferentiableController behavior."""
    return EnhancedDifferentiableController(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        enable_hierarchical=False,
    )


__all__ = [
    "EnhancedDifferentiableController",
    "ControllerOutputs",
    "HierarchicalAttention",
    "TemperatureAdaptive",
    "create_enhanced_controller",
    "create_backward_compatible_controller",
]