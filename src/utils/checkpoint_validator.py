"""
Checkpoint Validation System for TernaryVAE Training Pipeline

This module provides validation to prevent the 0% coverage issue caused by
null checkpoint paths with V5.11+ architectures that require frozen components.

Created: 2026-01-12
Root Cause Fix: V5.11+ expects frozen components for coverage, null paths break training
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings


class CheckpointCompatibilityError(Exception):
    """Raised when checkpoint is incompatible with model architecture."""
    pass


class CheckpointValidator:
    """Validates checkpoint loading for TernaryVAE architectures."""

    # Known architecture compatibility matrix
    COMPATIBILITY_MATRIX = {
        'TernaryVAEV5_11': {
            'compatible_checkpoints': ['v5_5', 'v5_11'],
            'requires_frozen': True,
            'coverage_requirement': 'frozen_components'
        },
        'TernaryVAEV5_11_PartialFreeze': {
            'compatible_checkpoints': ['v5_5', 'v5_11', 'v5_12'],
            'requires_frozen': True,
            'coverage_requirement': 'frozen_components'
        },
        'TernaryVAEV5_12': {
            'compatible_checkpoints': ['v5_12', 'v5_12_4'],
            'requires_frozen': False,
            'coverage_requirement': 'trainable_components'
        }
    }

    @classmethod
    def validate_checkpoint_config(cls, config: Dict, model_name: str) -> Tuple[bool, List[str]]:
        """
        Validate checkpoint configuration for given model architecture.

        Args:
            config: Training configuration dictionary
            model_name: Model architecture name

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # Get frozen checkpoint config
        frozen_cfg = config.get('frozen_checkpoint', {})
        checkpoint_path = frozen_cfg.get('path')

        # Check if model requires frozen components
        arch_info = cls.COMPATIBILITY_MATRIX.get(model_name, {})
        requires_frozen = arch_info.get('requires_frozen', False)

        if requires_frozen:
            # V5.11+ architectures MUST have frozen checkpoint
            if checkpoint_path is None or checkpoint_path == 'null':
                errors.append(
                    f"CRITICAL: {model_name} requires frozen checkpoint for 100% coverage. "
                    f"Set frozen_checkpoint.path to a valid checkpoint, not null."
                )
                errors.append(
                    f"Recommended: Use 'sandbox-training/checkpoints/v5_12_4/best_Q.pt' "
                    f"or another compatible checkpoint."
                )
            else:
                # Validate checkpoint exists
                checkpoint_file = Path(checkpoint_path)
                if not checkpoint_file.exists():
                    errors.append(
                        f"CRITICAL: Checkpoint not found: {checkpoint_path}. "
                        f"Training will fail with 0% coverage."
                    )

        # Additional architecture-specific validation
        if 'V5_11' in model_name and checkpoint_path and 'v5_12_4' in str(checkpoint_path):
            # Check for dimension compatibility
            errors.append(
                f"WARNING: Loading v5_12_4 checkpoint into {model_name} may cause dimension mismatches. "
                f"Consider using v5_11 compatible checkpoint instead."
            )

        return len(errors) == 0, errors

    @classmethod
    def validate_checkpoint_dimensions(cls, checkpoint_path: str, model: torch.nn.Module) -> Tuple[bool, List[str]]:
        """
        Validate that checkpoint dimensions match model architecture.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model instance to check against

        Returns:
            (is_compatible, error_messages)
        """
        errors = []

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Get model state dict
            if 'model_state_dict' in checkpoint:
                ckpt_state = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                ckpt_state = checkpoint['model']
            else:
                ckpt_state = checkpoint

            # Get model state dict
            model_state = model.state_dict()

            # Check for dimension mismatches
            mismatches = []
            for key in ckpt_state:
                if key in model_state:
                    ckpt_shape = ckpt_state[key].shape
                    model_shape = model_state[key].shape
                    if ckpt_shape != model_shape:
                        mismatches.append(f"{key}: checkpoint {ckpt_shape} vs model {model_shape}")

            if mismatches:
                errors.append(f"Dimension mismatches found:")
                errors.extend([f"  - {m}" for m in mismatches])
                errors.append("Consider using strict=False loading or a compatible checkpoint.")

        except Exception as e:
            errors.append(f"Failed to validate checkpoint: {e}")

        return len(errors) == 0, errors

    @classmethod
    def suggest_compatible_checkpoint(cls, model_name: str, checkpoint_dir: str = "sandbox-training/checkpoints") -> List[str]:
        """
        Suggest compatible checkpoints for given model architecture.

        Args:
            model_name: Model architecture name
            checkpoint_dir: Directory containing checkpoints

        Returns:
            List of recommended checkpoint paths
        """
        arch_info = cls.COMPATIBILITY_MATRIX.get(model_name, {})
        compatible_types = arch_info.get('compatible_checkpoints', [])

        suggestions = []
        checkpoint_path = Path(checkpoint_dir)

        if checkpoint_path.exists():
            for ckpt_type in compatible_types:
                # Look for checkpoints matching this type
                pattern_dirs = list(checkpoint_path.glob(f"{ckpt_type}*"))
                for pattern_dir in pattern_dirs:
                    # Look for best.pt or best_Q.pt
                    for filename in ['best_Q.pt', 'best.pt']:
                        ckpt_file = pattern_dir / filename
                        if ckpt_file.exists():
                            suggestions.append(str(ckpt_file))

        return suggestions

    @classmethod
    def fix_null_checkpoint_config(cls, config: Dict, model_name: str) -> Dict:
        """
        Automatically fix null checkpoint configurations.

        Args:
            config: Training configuration dictionary
            model_name: Model architecture name

        Returns:
            Fixed configuration dictionary
        """
        # Make a copy to avoid modifying original
        fixed_config = config.copy()

        # Get frozen checkpoint config
        frozen_cfg = fixed_config.get('frozen_checkpoint', {})
        checkpoint_path = frozen_cfg.get('path')

        # Check if model requires frozen components but has null path
        arch_info = cls.COMPATIBILITY_MATRIX.get(model_name, {})
        requires_frozen = arch_info.get('requires_frozen', False)

        if requires_frozen and (checkpoint_path is None or checkpoint_path == 'null'):
            # Find a compatible checkpoint
            suggestions = cls.suggest_compatible_checkpoint(model_name)

            if suggestions:
                # Use the first suggestion (usually best_Q.pt from latest version)
                recommended = suggestions[0]
                fixed_config['frozen_checkpoint'] = {
                    'path': recommended,
                    'encoder_to_load': 'both',
                    'decoder_to_load': 'decoder_A'
                }
                print(f"AUTO-FIXED: Set checkpoint to {recommended} for {model_name}")
            else:
                warnings.warn(
                    f"Could not auto-fix checkpoint for {model_name}. "
                    f"No compatible checkpoints found. Training may fail with 0% coverage."
                )

        return fixed_config


def validate_training_config(config: Dict) -> Tuple[bool, List[str]]:
    """
    Main validation function for training configurations.

    Args:
        config: Training configuration dictionary

    Returns:
        (is_valid, error_messages)
    """
    model_name = config.get('model', {}).get('name', 'Unknown')

    # Validate checkpoint configuration
    is_valid, errors = CheckpointValidator.validate_checkpoint_config(config, model_name)

    # Additional validations can be added here

    return is_valid, errors


if __name__ == "__main__":
    # Example usage
    import yaml

    # Test with problematic config
    config_path = "configs/v5_12_4_extended_grokking.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    is_valid, errors = validate_training_config(config)

    if not is_valid:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"  {error}")

        # Try to auto-fix
        model_name = config.get('model', {}).get('name')
        fixed_config = CheckpointValidator.fix_null_checkpoint_config(config, model_name)
        print("\n🔧 Auto-fixed configuration available")
    else:
        print("✅ Configuration validation passed")