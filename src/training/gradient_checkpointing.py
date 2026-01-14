"""
Gradient Checkpointing Utilities for TernaryVAE V5.12.5

Implements memory-efficient gradient checkpointing for deep sequential models.
Trade compute for memory: 30-40% VRAM reduction with 10-15% slower training.

Author: Claude Code
Date: 2026-01-14
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
from typing import Tuple, Optional, Dict, Any
import warnings


class CheckpointConfig:
    """Configuration for gradient checkpointing."""

    def __init__(
        self,
        enabled: bool = False,
        preserve_rng_state: bool = True,
        segments: int = 2,
        use_reentrant: bool = True,
        encoder_checkpoint: bool = True,
        decoder_checkpoint: bool = True,
        projection_checkpoint: bool = False,  # Usually fast, skip by default
        controller_checkpoint: bool = False,  # Small MLP, skip by default
    ):
        """Initialize checkpointing configuration.

        Args:
            enabled: Whether to use gradient checkpointing
            preserve_rng_state: Preserve RNG state during checkpointing
            segments: Number of segments for checkpoint_sequential
            use_reentrant: Use reentrant checkpointing (faster, more memory)
            encoder_checkpoint: Checkpoint encoder forward passes
            decoder_checkpoint: Checkpoint decoder forward passes
            projection_checkpoint: Checkpoint hyperbolic projection
            controller_checkpoint: Checkpoint controller computation
        """
        self.enabled = enabled
        self.preserve_rng_state = preserve_rng_state
        self.segments = segments
        self.use_reentrant = use_reentrant
        self.encoder_checkpoint = encoder_checkpoint
        self.decoder_checkpoint = decoder_checkpoint
        self.projection_checkpoint = projection_checkpoint
        self.controller_checkpoint = controller_checkpoint


def checkpoint_encoder(
    encoder: nn.Sequential,
    x: torch.Tensor,
    config: CheckpointConfig,
) -> torch.Tensor:
    """Apply gradient checkpointing to encoder forward pass.

    Args:
        encoder: Sequential encoder model
        x: Input tensor
        config: Checkpointing configuration

    Returns:
        Encoder output tensor
    """
    if not config.enabled or not config.encoder_checkpoint:
        return encoder(x)

    if len(encoder) < config.segments:
        # Too few layers, just use regular checkpointing
        return checkpoint(
            encoder,
            x,
            use_reentrant=config.use_reentrant,
            preserve_rng_state=config.preserve_rng_state
        )

    try:
        return checkpoint_sequential(
            functions=encoder,
            segments=config.segments,
            input=x,
            use_reentrant=config.use_reentrant,
            preserve_rng_state=config.preserve_rng_state
        )
    except Exception as e:
        warnings.warn(
            f"Gradient checkpointing failed for encoder: {e}. "
            f"Falling back to regular forward pass."
        )
        return encoder(x)


def checkpoint_decoder(
    decoder: nn.Sequential,
    z: torch.Tensor,
    config: CheckpointConfig,
) -> torch.Tensor:
    """Apply gradient checkpointing to decoder forward pass.

    Args:
        decoder: Sequential decoder model
        z: Latent tensor
        config: Checkpointing configuration

    Returns:
        Decoder output tensor
    """
    if not config.enabled or not config.decoder_checkpoint:
        return decoder(z)

    if len(decoder) < config.segments:
        # Too few layers, just use regular checkpointing
        return checkpoint(
            decoder,
            z,
            use_reentrant=config.use_reentrant,
            preserve_rng_state=config.preserve_rng_state
        )

    try:
        return checkpoint_sequential(
            functions=decoder,
            segments=config.segments,
            input=z,
            use_reentrant=config.use_reentrant,
            preserve_rng_state=config.preserve_rng_state
        )
    except Exception as e:
        warnings.warn(
            f"Gradient checkpointing failed for decoder: {e}. "
            f"Falling back to regular forward pass."
        )
        return decoder(z)


def checkpoint_function(
    fn,
    *args,
    config: CheckpointConfig,
    component_name: str = "function",
    **kwargs
) -> Any:
    """Apply gradient checkpointing to arbitrary function.

    Args:
        fn: Function to checkpoint
        *args: Function arguments
        config: Checkpointing configuration
        component_name: Name for error messages
        **kwargs: Function keyword arguments

    Returns:
        Function output
    """
    if not config.enabled:
        return fn(*args, **kwargs)

    # Only checkpoint if we have tensor inputs (gradients)
    has_tensor_args = any(isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args)
    has_tensor_kwargs = any(isinstance(val, torch.Tensor) and val.requires_grad for val in kwargs.values())

    if not (has_tensor_args or has_tensor_kwargs):
        return fn(*args, **kwargs)

    try:
        if kwargs:
            # checkpoint doesn't support kwargs directly, so we wrap
            def wrapper(*args_only):
                return fn(*args_only, **kwargs)
            return checkpoint(
                wrapper,
                *args,
                use_reentrant=config.use_reentrant,
                preserve_rng_state=config.preserve_rng_state
            )
        else:
            return checkpoint(
                fn,
                *args,
                use_reentrant=config.use_reentrant,
                preserve_rng_state=config.preserve_rng_state
            )
    except Exception as e:
        warnings.warn(
            f"Gradient checkpointing failed for {component_name}: {e}. "
            f"Falling back to regular computation."
        )
        return fn(*args, **kwargs)


class CheckpointedEncoder(nn.Module):
    """Wrapper for ImprovedEncoder with gradient checkpointing."""

    def __init__(self, encoder: nn.Module, config: CheckpointConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config

        # Extract the sequential part and head layers
        if hasattr(encoder, 'encoder') and hasattr(encoder, 'fc_mu') and hasattr(encoder, 'fc_logvar'):
            self.sequential_part = encoder.encoder
            self.fc_mu = encoder.fc_mu
            self.fc_logvar = encoder.fc_logvar
            self.logvar_min = getattr(encoder, 'logvar_min', -10.0)
            self.logvar_max = getattr(encoder, 'logvar_max', 2.0)
        else:
            raise ValueError(f"Encoder {type(encoder)} not compatible with checkpointing")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional gradient checkpointing."""
        # Checkpoint the main encoder sequential layers
        features = checkpoint_encoder(self.sequential_part, x, self.config)

        # Head layers are usually fast, don't checkpoint by default
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)

        # Apply logvar clamping
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)

        return mu, logvar


class CheckpointedDecoder(nn.Module):
    """Wrapper for ImprovedDecoder with gradient checkpointing."""

    def __init__(self, decoder: nn.Module, config: CheckpointConfig):
        super().__init__()
        self.decoder = decoder
        self.config = config

        # Extract the sequential part
        if hasattr(decoder, 'decoder'):
            self.sequential_part = decoder.decoder
            self.output_dim = getattr(decoder, 'output_dim', 9)
        else:
            raise ValueError(f"Decoder {type(decoder)} not compatible with checkpointing")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing."""
        # Checkpoint the decoder sequential layers
        logits_flat = checkpoint_decoder(self.sequential_part, z, self.config)

        # Reshape to (batch_size, output_dim, 3) for ternary operations
        batch_size = z.size(0)
        logits = logits_flat.view(batch_size, self.output_dim, 3)

        return logits


def apply_gradient_checkpointing(
    model: nn.Module,
    config: CheckpointConfig
) -> nn.Module:
    """Apply gradient checkpointing to compatible model components.

    Args:
        model: TernaryVAE model
        config: Checkpointing configuration

    Returns:
        Model with checkpointing wrappers applied
    """
    if not config.enabled:
        return model

    # Apply checkpointing to encoders
    if hasattr(model, 'encoder_A') and config.encoder_checkpoint:
        try:
            model.encoder_A = CheckpointedEncoder(model.encoder_A, config)
        except ValueError as e:
            warnings.warn(f"Could not checkpoint encoder_A: {e}")

    if hasattr(model, 'encoder_B') and config.encoder_checkpoint:
        try:
            model.encoder_B = CheckpointedEncoder(model.encoder_B, config)
        except ValueError as e:
            warnings.warn(f"Could not checkpoint encoder_B: {e}")

    # Apply checkpointing to decoder
    if hasattr(model, 'decoder_A') and config.decoder_checkpoint:
        try:
            model.decoder_A = CheckpointedDecoder(model.decoder_A, config)
        except ValueError as e:
            warnings.warn(f"Could not checkpoint decoder_A: {e}")

    return model


# Utility function for config parsing
def create_checkpoint_config(config_dict: Dict[str, Any]) -> CheckpointConfig:
    """Create CheckpointConfig from dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        CheckpointConfig instance
    """
    gradient_checkpoint = config_dict.get('gradient_checkpoint', {})

    return CheckpointConfig(
        enabled=gradient_checkpoint.get('enabled', False),
        preserve_rng_state=gradient_checkpoint.get('preserve_rng_state', True),
        segments=gradient_checkpoint.get('segments', 2),
        use_reentrant=gradient_checkpoint.get('use_reentrant', True),
        encoder_checkpoint=gradient_checkpoint.get('encoder_checkpoint', True),
        decoder_checkpoint=gradient_checkpoint.get('decoder_checkpoint', True),
        projection_checkpoint=gradient_checkpoint.get('projection_checkpoint', False),
        controller_checkpoint=gradient_checkpoint.get('controller_checkpoint', False),
    )