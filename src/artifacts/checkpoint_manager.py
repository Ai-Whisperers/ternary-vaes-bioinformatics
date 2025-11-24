"""Checkpoint management for training.

This module handles saving and loading model checkpoints with metadata.

Single responsibility: Checkpoint persistence only.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional


class CheckpointManager:
    """Manages saving and loading model checkpoints."""

    def __init__(self, checkpoint_dir: Path, checkpoint_freq: int = 10):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            checkpoint_freq: Frequency (in epochs) to save numbered checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: Dict[str, Any],
        is_best: bool = False
    ) -> None:
        """Save checkpoint with model, optimizer, and metadata.

        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save
            metadata: Additional metadata to save
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            **metadata
        }

        # Always save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

        # Save best if indicated
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')

        # Save numbered checkpoint at specified frequency
        if epoch % self.checkpoint_freq == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{epoch}.pt')

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_name: str = 'latest',
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """Load checkpoint and restore model/optimizer state.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            checkpoint_name: Name of checkpoint ('latest', 'best', or 'epoch_N')
            device: Device to load checkpoint on

        Returns:
            Checkpoint metadata dict

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        # Construct checkpoint path
        if checkpoint_name in ['latest', 'best']:
            checkpoint_path = self.checkpoint_dir / f'{checkpoint_name}.pt'
        else:
            checkpoint_path = self.checkpoint_dir / f'{checkpoint_name}.pt'

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Restore model state
        model.load_state_dict(checkpoint['model'])

        # Restore optimizer state if provided
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint

    def list_checkpoints(self) -> Dict[str, list]:
        """List all available checkpoints.

        Returns:
            Dict with 'special' (latest/best) and 'epochs' (numbered) checkpoints
        """
        special = []
        epochs = []

        for ckpt_file in self.checkpoint_dir.glob('*.pt'):
            name = ckpt_file.stem
            if name in ['latest', 'best']:
                special.append(name)
            elif name.startswith('epoch_'):
                epochs.append(name)

        return {
            'special': sorted(special),
            'epochs': sorted(epochs, key=lambda x: int(x.split('_')[1]))
        }

    def get_latest_epoch(self) -> Optional[int]:
        """Get the latest saved epoch number.

        Returns:
            Latest epoch number or None if no checkpoints exist
        """
        latest_path = self.checkpoint_dir / 'latest.pt'
        if not latest_path.exists():
            return None

        checkpoint = torch.load(latest_path, map_location='cpu')
        return checkpoint.get('epoch')
