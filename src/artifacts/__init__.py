"""Artifact management components.

This module handles checkpoint and artifact lifecycle management:
- CheckpointManager: Save/load checkpoints with metadata
- ArtifactRepository: Artifact promotion (raw → validated → production)
- Metadata: Checkpoint and model metadata handling
"""

from .checkpoint_manager import CheckpointManager

__all__ = [
    'CheckpointManager'
]
