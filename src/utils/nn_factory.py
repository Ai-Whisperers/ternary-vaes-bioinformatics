"""Neural Network Factory - MLPBuilder utility for framework unification.

This module provides utilities to create standardized MLPs across the TernaryVAE
codebase, reducing code duplication and ensuring consistent architectures.

Identified Patterns to Unify:
1. CodonEncoderMLP: Linear -> LayerNorm -> SiLU -> Dropout
2. ImprovedEncoder: Linear -> LayerNorm -> SiLU -> Dropout (256->128->64)
3. ImprovedDecoder: Linear -> LayerNorm -> SiLU -> Dropout (latent->32->64->output)
4. DifferentiableController: Linear -> LayerNorm -> SiLU (no dropout)
5. Test patterns: Linear -> ReLU/GELU (simple)

Author: Claude Code
Date: 2026-01-14
"""

from typing import List, Optional, Union, Literal, Dict, Any
import torch.nn as nn


class MLPBuilder:
    """Builder for creating standardized Multi-Layer Perceptrons.

    Supports common patterns found across the TernaryVAE codebase:
    - Various activation functions (ReLU, GELU, SiLU, Tanh, LeakyReLU)
    - Optional normalization (LayerNorm, BatchNorm1d, none)
    - Optional dropout
    - Flexible layer dimensions
    - Weight initialization strategies
    """

    def __init__(self):
        self.layers: List[nn.Module] = []
        self._input_dim: Optional[int] = None
        self._layer_dims: List[int] = []

    @classmethod
    def create(
        cls,
        input_dim: int,
        hidden_dims: Union[int, List[int]],
        output_dim: int,
        activation: Literal["relu", "gelu", "silu", "tanh", "leaky_relu"] = "silu",
        normalization: Literal["layer_norm", "batch_norm", "none"] = "layer_norm",
        dropout: float = 0.0,
        final_activation: bool = False,
        final_normalization: bool = False,
        final_dropout: bool = False,
        bias: bool = True,
        init_strategy: Literal["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "none"] = "xavier_uniform",
    ) -> nn.Sequential:
        """Create a standardized MLP.

        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions (int or list of ints)
            output_dim: Output dimension
            activation: Activation function to use
            normalization: Normalization type to apply
            dropout: Dropout probability (0.0 = no dropout)
            final_activation: Whether to apply activation after final layer
            final_normalization: Whether to apply normalization after final layer
            final_dropout: Whether to apply dropout after final layer
            bias: Whether to use bias in Linear layers
            init_strategy: Weight initialization strategy

        Returns:
            nn.Sequential containing the MLP layers

        Examples:
            # CodonEncoderMLP pattern (12 -> 64 -> 64 -> 16)
            mlp = MLPBuilder.create(
                input_dim=12,
                hidden_dims=64,
                output_dim=16,
                activation="silu",
                normalization="layer_norm",
                dropout=0.1
            )

            # ImprovedEncoder pattern (9 -> 256 -> 128 -> 64)
            mlp = MLPBuilder.create(
                input_dim=9,
                hidden_dims=[256, 128],
                output_dim=64,
                activation="silu",
                normalization="layer_norm",
                dropout=0.1
            )

            # DifferentiableController pattern (8 -> 32 -> 32 -> 6)
            mlp = MLPBuilder.create(
                input_dim=8,
                hidden_dims=32,
                output_dim=6,
                activation="silu",
                normalization="layer_norm",
                dropout=0.0
            )

            # Simple test pattern (10 -> 32 -> 5)
            mlp = MLPBuilder.create(
                input_dim=10,
                hidden_dims=32,
                output_dim=5,
                activation="relu",
                normalization="none",
                dropout=0.0
            )
        """
        builder = cls()
        return builder._build(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
            final_activation=final_activation,
            final_normalization=final_normalization,
            final_dropout=final_dropout,
            bias=bias,
            init_strategy=init_strategy,
        )

    def _build(
        self,
        input_dim: int,
        hidden_dims: Union[int, List[int]],
        output_dim: int,
        activation: str,
        normalization: str,
        dropout: float,
        final_activation: bool,
        final_normalization: bool,
        final_dropout: bool,
        bias: bool,
        init_strategy: str,
    ) -> nn.Sequential:
        """Internal build method."""
        self.layers = []

        # Normalize hidden_dims to list
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Create layer dimension sequence
        all_dims = [input_dim] + hidden_dims + [output_dim]

        # Build layers
        for i in range(len(all_dims) - 1):
            in_dim = all_dims[i]
            out_dim = all_dims[i + 1]
            is_final_layer = (i == len(all_dims) - 2)

            # Add Linear layer
            linear = nn.Linear(in_dim, out_dim, bias=bias)
            self.layers.append(linear)

            # Add normalization (if not final layer or if explicitly requested)
            if not is_final_layer or final_normalization:
                if normalization == "layer_norm":
                    self.layers.append(nn.LayerNorm(out_dim))
                elif normalization == "batch_norm":
                    self.layers.append(nn.BatchNorm1d(out_dim))
                # "none" - no normalization

            # Add activation (if not final layer or if explicitly requested)
            if not is_final_layer or final_activation:
                activation_layer = self._get_activation(activation)
                self.layers.append(activation_layer)

            # Add dropout (if not final layer or if explicitly requested)
            if (not is_final_layer or final_dropout) and dropout > 0.0:
                self.layers.append(nn.Dropout(dropout))

        # Create sequential model
        model = nn.Sequential(*self.layers)

        # Apply weight initialization
        if init_strategy != "none":
            self._init_weights(model, init_strategy)

        return model

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.2),
        }

        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}. "
                           f"Available: {list(activation_map.keys())}")

        return activation_map[activation]

    def _init_weights(self, model: nn.Sequential, strategy: str):
        """Initialize weights according to strategy."""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if strategy == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif strategy == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif strategy == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif strategy == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

                # Initialize bias to zero
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                # LayerNorm initialization (near-identity for compatibility)
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm1d):
                # BatchNorm initialization
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


# Convenience functions for common patterns
def create_encoder_mlp(
    input_dim: int,
    hidden_dims: Union[int, List[int]],
    latent_dim: int,
    dropout: float = 0.1,
    activation: str = "silu",
) -> nn.Sequential:
    """Create standard encoder MLP pattern.

    Pattern: Linear -> LayerNorm -> SiLU -> Dropout (repeated)
    """
    return MLPBuilder.create(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=latent_dim,
        activation=activation,
        normalization="layer_norm",
        dropout=dropout,
        final_activation=False,
        final_normalization=False,
        final_dropout=False,
    )


def create_decoder_mlp(
    latent_dim: int,
    hidden_dims: Union[int, List[int]],
    output_dim: int,
    dropout: float = 0.1,
    activation: str = "silu",
) -> nn.Sequential:
    """Create standard decoder MLP pattern.

    Pattern: Linear -> LayerNorm -> SiLU -> Dropout (repeated)
    Final layer has no activation/normalization (raw logits)
    """
    return MLPBuilder.create(
        input_dim=latent_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=activation,
        normalization="layer_norm",
        dropout=dropout,
        final_activation=False,
        final_normalization=False,
        final_dropout=False,
    )


def create_controller_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    activation: str = "silu",
) -> nn.Sequential:
    """Create standard controller MLP pattern.

    Pattern: Linear -> LayerNorm -> SiLU (no dropout, lightweight)
    """
    return MLPBuilder.create(
        input_dim=input_dim,
        hidden_dims=hidden_dim,
        output_dim=output_dim,
        activation=activation,
        normalization="layer_norm",
        dropout=0.0,
        final_activation=False,
        final_normalization=False,
        final_dropout=False,
    )


def create_simple_mlp(
    input_dim: int,
    hidden_dims: Union[int, List[int]],
    output_dim: int,
    activation: str = "relu",
) -> nn.Sequential:
    """Create simple MLP pattern (for tests/benchmarks).

    Pattern: Linear -> Activation (no normalization/dropout)
    """
    return MLPBuilder.create(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=activation,
        normalization="none",
        dropout=0.0,
        final_activation=False,
        final_normalization=False,
        final_dropout=False,
    )


# Pattern validation helpers
def validate_mlp_architecture(model: nn.Sequential, expected_pattern: str) -> bool:
    """Validate that an MLP follows expected architectural patterns.

    Args:
        model: The MLP model to validate
        expected_pattern: Pattern name ("encoder", "decoder", "controller", "simple")

    Returns:
        True if model matches expected pattern
    """
    layers = list(model.children())

    if expected_pattern == "encoder":
        # Expect: Linear, LayerNorm, SiLU, Dropout pattern
        return _validate_encoder_pattern(layers)
    elif expected_pattern == "decoder":
        # Expect: Linear, LayerNorm, SiLU, Dropout pattern
        return _validate_decoder_pattern(layers)
    elif expected_pattern == "controller":
        # Expect: Linear, LayerNorm, SiLU pattern (no dropout)
        return _validate_controller_pattern(layers)
    elif expected_pattern == "simple":
        # Expect: Linear, Activation pattern
        return _validate_simple_pattern(layers)
    else:
        raise ValueError(f"Unknown pattern: {expected_pattern}")


def _validate_encoder_pattern(layers: List[nn.Module]) -> bool:
    """Validate encoder pattern: [Linear, LayerNorm, SiLU, Dropout]+ Linear"""
    if len(layers) < 2:  # At least one hidden layer + output
        return False

    # Check repeating pattern
    i = 0
    while i < len(layers) - 1:  # Don't check final layer yet
        # Should be: Linear, LayerNorm, SiLU, Dropout
        if not isinstance(layers[i], nn.Linear):
            return False
        if i + 1 < len(layers) - 1 and not isinstance(layers[i + 1], nn.LayerNorm):
            return False
        if i + 2 < len(layers) - 1 and not isinstance(layers[i + 2], nn.SiLU):
            return False
        if i + 3 < len(layers) - 1 and not isinstance(layers[i + 3], nn.Dropout):
            return False

        i += 4  # Move to next block

    # Final layer should be just Linear
    return isinstance(layers[-1], nn.Linear)


def _validate_decoder_pattern(layers: List[nn.Module]) -> bool:
    """Validate decoder pattern (same as encoder)."""
    return _validate_encoder_pattern(layers)


def _validate_controller_pattern(layers: List[nn.Module]) -> bool:
    """Validate controller pattern: [Linear, LayerNorm, SiLU]+ Linear"""
    if len(layers) < 2:
        return False

    # Check repeating pattern (no dropout)
    i = 0
    while i < len(layers) - 1:  # Don't check final layer yet
        # Should be: Linear, LayerNorm, SiLU
        if not isinstance(layers[i], nn.Linear):
            return False
        if i + 1 < len(layers) - 1 and not isinstance(layers[i + 1], nn.LayerNorm):
            return False
        if i + 2 < len(layers) - 1 and not isinstance(layers[i + 2], nn.SiLU):
            return False

        i += 3  # Move to next block

    # Final layer should be just Linear
    return isinstance(layers[-1], nn.Linear)


def _validate_simple_pattern(layers: List[nn.Module]) -> bool:
    """Validate simple pattern: [Linear, Activation]+ Linear"""
    if len(layers) < 2:
        return False

    # Check alternating Linear/Activation pattern
    for i, layer in enumerate(layers[:-1]):  # Don't check final layer
        if i % 2 == 0:  # Even indices should be Linear
            if not isinstance(layer, nn.Linear):
                return False
        else:  # Odd indices should be activation
            if not isinstance(layer, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.LeakyReLU)):
                return False

    # Final layer should be Linear
    return isinstance(layers[-1], nn.Linear)


# Migration helpers
class MLPMigrationHelper:
    """Helper for migrating existing MLPs to standardized versions."""

    @staticmethod
    def analyze_existing_mlp(model: nn.Sequential) -> Dict[str, Any]:
        """Analyze existing MLP and suggest MLPBuilder configuration.

        Args:
            model: Existing nn.Sequential MLP

        Returns:
            Dictionary with suggested MLPBuilder parameters
        """
        layers = list(model.children())

        # Extract dimensions
        input_dim = None
        output_dim = None
        hidden_dims = []

        linear_layers = [l for l in layers if isinstance(l, nn.Linear)]
        if linear_layers:
            input_dim = linear_layers[0].in_features
            output_dim = linear_layers[-1].out_features

            if len(linear_layers) > 1:
                hidden_dims = [l.out_features for l in linear_layers[:-1]]

        # Detect activation type
        activation = "relu"  # default
        for layer in layers:
            if isinstance(layer, nn.ReLU):
                activation = "relu"
                break
            elif isinstance(layer, nn.GELU):
                activation = "gelu"
                break
            elif isinstance(layer, nn.SiLU):
                activation = "silu"
                break
            elif isinstance(layer, nn.Tanh):
                activation = "tanh"
                break
            elif isinstance(layer, nn.LeakyReLU):
                activation = "leaky_relu"
                break

        # Detect normalization
        normalization = "none"
        for layer in layers:
            if isinstance(layer, nn.LayerNorm):
                normalization = "layer_norm"
                break
            elif isinstance(layer, nn.BatchNorm1d):
                normalization = "batch_norm"
                break

        # Detect dropout
        dropout = 0.0
        for layer in layers:
            if isinstance(layer, nn.Dropout):
                dropout = layer.p
                break

        return {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
            "activation": activation,
            "normalization": normalization,
            "dropout": dropout,
        }

    @staticmethod
    def suggest_replacement(model: nn.Sequential) -> str:
        """Suggest MLPBuilder code to replace existing MLP."""
        config = MLPMigrationHelper.analyze_existing_mlp(model)

        # Generate code suggestion
        hidden_dims_str = str(config["hidden_dims"]) if len(config["hidden_dims"]) > 1 else str(config["hidden_dims"][0]) if config["hidden_dims"] else "[]"

        code = f"""# Replace with MLPBuilder
MLPBuilder.create(
    input_dim={config["input_dim"]},
    hidden_dims={hidden_dims_str},
    output_dim={config["output_dim"]},
    activation="{config["activation"]}",
    normalization="{config["normalization"]}",
    dropout={config["dropout"]},
)"""

        return code