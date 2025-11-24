"""Training monitoring and logging.

This module handles training progress monitoring:
- Loss and metrics logging
- Coverage evaluation and tracking
- Training history management

Single responsibility: Monitoring and logging only.
"""

import torch
from typing import Dict, Any, List, Optional
from collections import defaultdict


class TrainingMonitor:
    """Monitors and logs training progress."""

    def __init__(self, eval_num_samples: int = 100000):
        """Initialize training monitor.

        Args:
            eval_num_samples: Number of samples for coverage evaluation
        """
        self.eval_num_samples = eval_num_samples

        # Training history
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Coverage tracking
        self.coverage_A_history: List[int] = []
        self.coverage_B_history: List[int] = []

        # Entropy tracking
        self.H_A_history: List[float] = []
        self.H_B_history: List[float] = []

    def update_histories(
        self,
        H_A: float,
        H_B: float,
        coverage_A: int,
        coverage_B: int
    ) -> None:
        """Update all tracked histories.

        Args:
            H_A: VAE-A entropy
            H_B: VAE-B entropy
            coverage_A: VAE-A coverage count
            coverage_B: VAE-B coverage count
        """
        self.H_A_history.append(H_A)
        self.H_B_history.append(H_B)
        self.coverage_A_history.append(coverage_A)
        self.coverage_B_history.append(coverage_B)

    def check_best(self, val_loss: float) -> bool:
        """Check if current validation loss is best.

        Args:
            val_loss: Current validation loss

        Returns:
            True if this is the best loss so far
        """
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return is_best

    def should_stop(self, patience: int) -> bool:
        """Check if early stopping criterion is met.

        Args:
            patience: Patience threshold

        Returns:
            True if should stop training
        """
        return self.patience_counter >= patience

    def evaluate_coverage(
        self,
        model: torch.nn.Module,
        num_samples: int,
        device: str,
        vae: str = 'A'
    ) -> tuple[int, float]:
        """Evaluate operation coverage.

        Args:
            model: Model to evaluate
            num_samples: Number of samples to generate
            device: Device to run on
            vae: Which VAE to evaluate ('A' or 'B')

        Returns:
            Tuple of (unique_count, coverage_percentage)
        """
        model.eval()
        unique_ops = set()

        with torch.no_grad():
            batch_size = 1000
            num_batches = num_samples // batch_size

            for _ in range(num_batches):
                samples = model.sample(batch_size, device, vae)
                samples_rounded = torch.round(samples).long()

                for i in range(batch_size):
                    lut = samples_rounded[i]
                    lut_tuple = tuple(lut.cpu().tolist())
                    unique_ops.add(lut_tuple)

        coverage_pct = (len(unique_ops) / 19683) * 100
        return len(unique_ops), coverage_pct

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_losses: Dict[str, Any],
        val_losses: Dict[str, Any],
        unique_A: int,
        cov_A: float,
        unique_B: int,
        cov_B: float,
        is_best: bool,
        use_statenet: bool,
        grad_balance_achieved: bool
    ) -> None:
        """Log epoch results to console.

        Args:
            epoch: Current epoch
            total_epochs: Total epochs
            train_losses: Training losses dict
            val_losses: Validation losses dict
            unique_A: VAE-A unique operations
            cov_A: VAE-A coverage percentage
            unique_B: VAE-B unique operations
            cov_B: VAE-B coverage percentage
            is_best: Whether this is best validation loss
            use_statenet: Whether StateNet is enabled
            grad_balance_achieved: Whether gradient balance is achieved
        """
        print(f"\nEpoch {epoch}/{total_epochs}")
        print(f"  Loss: Train={train_losses['loss']:.4f} Val={val_losses['loss']:.4f}")
        print(f"  VAE-A: CE={train_losses['ce_A']:.4f} KL={train_losses['kl_A']:.4f} H={train_losses['H_A']:.3f}")
        print(f"  VAE-B: CE={train_losses['ce_B']:.4f} KL={train_losses['kl_B']:.4f} H={train_losses['H_B']:.3f}")
        print(f"  Weights: λ1={train_losses['lambda1']:.3f} λ2={train_losses['lambda2']:.3f} λ3={train_losses['lambda3']:.3f}")
        print(f"  Phase {train_losses['phase']}: ρ={train_losses['rho']:.3f} (balance: {'✓' if grad_balance_achieved else '○'})")
        print(f"  Grad: ratio={train_losses['grad_ratio']:.3f} EMA_α={train_losses['ema_momentum']:.2f}")
        print(f"  Temp: A={train_losses['temp_A']:.3f} B={train_losses['temp_B']:.3f} | β: A={train_losses['beta_A']:.3f} B={train_losses['beta_B']:.3f}")

        if use_statenet and 'lr_corrected' in train_losses:
            print(f"  LR: {train_losses['lr_scheduled']:.6f} → {train_losses['lr_corrected']:.6f} (Δ={train_losses.get('delta_lr', 0):+.3f})")
            print(f"  StateNet: Δλ1={train_losses.get('delta_lambda1', 0):+.3f} Δλ2={train_losses.get('delta_lambda2', 0):+.3f} Δλ3={train_losses.get('delta_lambda3', 0):+.3f}")
        else:
            print(f"  LR: {train_losses['lr_scheduled']:.6f}")

        print(f"  Coverage: A={unique_A} ({cov_A:.2f}%) | B={unique_B} ({cov_B:.2f}%)")

        if is_best:
            print(f"  ✓ Best val loss: {self.best_val_loss:.4f}")

    def get_metadata(self) -> Dict[str, Any]:
        """Get all tracked metadata for checkpointing.

        Returns:
            Dict of all tracked metrics and history
        """
        return {
            'best_val_loss': self.best_val_loss,
            'H_A_history': self.H_A_history,
            'H_B_history': self.H_B_history,
            'coverage_A_history': self.coverage_A_history,
            'coverage_B_history': self.coverage_B_history
        }

    def print_training_summary(self) -> None:
        """Print training completion summary."""
        print(f"\n{'='*80}")
        print("Training Complete")
        print(f"{'='*80}")
        print(f"Best val loss: {self.best_val_loss:.4f}")

        if self.coverage_A_history:
            final_cov_A = self.coverage_A_history[-1]
            final_cov_B = self.coverage_B_history[-1]
            print(f"Final Coverage: A={final_cov_A} ({final_cov_A/19683*100:.2f}%)")
            print(f"                B={final_cov_B} ({final_cov_B/19683*100:.2f}%)")
