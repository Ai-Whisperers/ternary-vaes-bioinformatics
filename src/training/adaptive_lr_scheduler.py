"""
Validation-Based Learning Rate Scheduler for TernaryVAE V5.12.5

Implements intelligent LR scheduling based on:
- Validation plateau detection
- Multiple metrics monitoring (hierarchy, coverage, richness)
- Adaptive reduction strategies
- Early stopping with recovery mechanisms

Author: Claude Code
Date: 2026-01-14
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import warnings
import math
from dataclasses import dataclass
from enum import Enum


class SchedulerMode(Enum):
    """Learning rate scheduler modes."""
    MIN = "min"  # Reduce LR when metric stops improving (lower is better)
    MAX = "max"  # Reduce LR when metric stops improving (higher is better)


class PlateauPhase(Enum):
    """Current plateau detection phase."""
    WARMUP = "warmup"
    MONITORING = "monitoring"
    PLATEAU_DETECTED = "plateau_detected"
    RECOVERY = "recovery"
    CONVERGED = "converged"


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    epoch: int
    primary_metric: float  # Main metric for plateau detection
    hierarchy_correlation: float
    coverage_accuracy: float
    richness_ratio: float
    loss_value: float


class AdaptiveLRScheduler:
    """Validation-based learning rate scheduler with plateau detection.

    Monitors multiple validation metrics and reduces learning rate when
    training plateaus. Includes recovery mechanisms and early stopping.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        primary_metric: str = "hierarchy_correlation",
        mode: SchedulerMode = SchedulerMode.MAX,
        patience: int = 8,
        factor: float = 0.5,
        min_lr: float = 1e-7,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        warmup_epochs: int = 5,
        verbose: bool = True,

        # Multi-metric monitoring
        secondary_metrics: Optional[List[str]] = None,
        metric_weights: Optional[Dict[str, float]] = None,

        # Advanced features
        adaptive_patience: bool = True,
        recovery_detection: bool = True,
        early_stopping: bool = False,
        early_stopping_patience: int = 20,

        # Plateau recovery
        recovery_factor: float = 1.5,  # Increase LR when recovering
        recovery_threshold: float = 0.01,
    ):
        """Initialize adaptive learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer
            primary_metric: Primary metric to monitor for plateau detection
            mode: Whether primary metric should be minimized or maximized
            patience: Number of epochs to wait before reducing LR
            factor: Factor to multiply LR by when reducing
            min_lr: Minimum learning rate
            threshold: Threshold for measuring improvement
            threshold_mode: 'rel' or 'abs' threshold mode
            cooldown: Number of epochs to wait after LR reduction
            warmup_epochs: Number of epochs before starting monitoring
            verbose: Print LR changes
            secondary_metrics: Additional metrics to monitor
            metric_weights: Weights for combining multiple metrics
            adaptive_patience: Increase patience over time
            recovery_detection: Detect recovery from plateaus
            early_stopping: Enable early stopping
            early_stopping_patience: Patience for early stopping
            recovery_factor: Factor to increase LR when recovering
            recovery_threshold: Threshold for detecting recovery
        """
        self.optimizer = optimizer
        self.primary_metric = primary_metric
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose

        # Multi-metric setup
        self.secondary_metrics = secondary_metrics or []
        self.metric_weights = metric_weights or {primary_metric: 1.0}
        if primary_metric not in self.metric_weights:
            self.metric_weights[primary_metric] = 1.0

        # Advanced features
        self.adaptive_patience = adaptive_patience
        self.recovery_detection = recovery_detection
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.recovery_factor = recovery_factor
        self.recovery_threshold = recovery_threshold

        # State tracking
        self.history: List[ValidationMetrics] = []
        self.best_metric = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.current_patience = patience
        self.last_lr_reduction_epoch = -1
        self.phase = PlateauPhase.WARMUP
        self.should_stop_early = False

        # Store initial LR for each parameter group
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == SchedulerMode.MAX:
            if self.threshold_mode == "rel":
                return current > best * (1 + self.threshold)
            else:  # abs
                return current > best + self.threshold
        else:  # MIN
            if self.threshold_mode == "rel":
                return current < best * (1 - self.threshold)
            else:  # abs
                return current < best - self.threshold

    def _compute_composite_metric(self, metrics: ValidationMetrics) -> float:
        """Compute weighted composite metric from multiple metrics."""
        if len(self.metric_weights) == 1:
            return getattr(metrics, self.primary_metric)

        composite = 0.0
        total_weight = 0.0

        for metric_name, weight in self.metric_weights.items():
            if hasattr(metrics, metric_name):
                value = getattr(metrics, metric_name)
                composite += weight * value
                total_weight += weight

        return composite / max(total_weight, 1e-8)

    def _detect_plateau(self) -> bool:
        """Detect if training has plateaued."""
        if len(self.history) < self.current_patience:
            return False

        # Check last `current_patience` epochs for improvement
        recent_metrics = self.history[-self.current_patience:]
        recent_values = [self._compute_composite_metric(m) for m in recent_metrics]

        if self.mode == SchedulerMode.MAX:
            # Look for increasing trend
            max_recent = max(recent_values)
            if self.best_metric is None:
                return False
            return max_recent <= self.best_metric
        else:
            # Look for decreasing trend
            min_recent = min(recent_values)
            if self.best_metric is None:
                return False
            return min_recent >= self.best_metric

    def _detect_recovery(self) -> bool:
        """Detect if training is recovering from plateau."""
        if not self.recovery_detection or len(self.history) < 3:
            return False

        # Check for significant improvement in recent epochs
        recent_values = [self._compute_composite_metric(m) for m in self.history[-3:]]

        if self.mode == SchedulerMode.MAX:
            improvement = (recent_values[-1] - recent_values[0]) / max(abs(recent_values[0]), 1e-8)
            return improvement > self.recovery_threshold
        else:
            improvement = (recent_values[0] - recent_values[-1]) / max(abs(recent_values[0]), 1e-8)
            return improvement > self.recovery_threshold

    def _reduce_lr(self, epoch: int):
        """Reduce learning rate for all parameter groups."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr

        self.last_lr_reduction_epoch = epoch
        self.cooldown_counter = self.cooldown
        self.num_bad_epochs = 0

        if self.adaptive_patience:
            # Gradually increase patience as training progresses
            self.current_patience = min(self.patience + epoch // 10, self.patience * 2)

        if self.verbose:
            print(f"  📉 LR Reduced: {old_lr:.6f} → {new_lr:.6f} (patience: {self.current_patience})")

    def _increase_lr(self, epoch: int):
        """Increase learning rate when recovering from plateau."""
        if not self.recovery_detection:
            return

        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = min(old_lr * self.recovery_factor, self.base_lrs[i])
            param_group['lr'] = new_lr

        if self.verbose:
            print(f"  📈 LR Increased (Recovery): {old_lr:.6f} → {new_lr:.6f}")

    def step(self, metrics: ValidationMetrics) -> Dict[str, Union[float, str, bool]]:
        """Update learning rate based on validation metrics.

        Args:
            metrics: ValidationMetrics containing current epoch metrics

        Returns:
            Dict with scheduler state information
        """
        self.history.append(metrics)
        epoch = metrics.epoch

        # Skip during warmup
        if epoch < self.warmup_epochs:
            self.phase = PlateauPhase.WARMUP
            return self._get_state()

        # Compute current metric value
        current_metric = self._compute_composite_metric(metrics)

        # Update best metric
        if self.best_metric is None or self._is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.num_bad_epochs = 0
            self.phase = PlateauPhase.MONITORING
        else:
            self.num_bad_epochs += 1

        # Handle cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self._get_state()

        # Detect plateau and recovery
        plateau_detected = self._detect_plateau()
        recovery_detected = self._detect_recovery()

        if recovery_detected and self.phase == PlateauPhase.PLATEAU_DETECTED:
            self.phase = PlateauPhase.RECOVERY
            self._increase_lr(epoch)

        elif plateau_detected and self.phase != PlateauPhase.PLATEAU_DETECTED:
            self.phase = PlateauPhase.PLATEAU_DETECTED
            self._reduce_lr(epoch)

        # Check for early stopping
        if self.early_stopping and self.num_bad_epochs >= self.early_stopping_patience:
            self.should_stop_early = True
            self.phase = PlateauPhase.CONVERGED
            if self.verbose:
                print(f"  ⏹️  Early stopping triggered after {self.num_bad_epochs} epochs without improvement")

        return self._get_state()

    def _get_state(self) -> Dict[str, Union[float, str, bool]]:
        """Get current scheduler state."""
        current_lrs = [group['lr'] for group in self.optimizer.param_groups]

        return {
            'phase': self.phase.value,
            'current_lr': current_lrs[0],  # Use first param group LR
            'best_metric': self.best_metric,
            'num_bad_epochs': self.num_bad_epochs,
            'current_patience': self.current_patience,
            'should_stop_early': self.should_stop_early,
            'last_reduction_epoch': self.last_lr_reduction_epoch,
            'cooldown_remaining': self.cooldown_counter,
        }

    def get_last_lr(self) -> List[float]:
        """Get current learning rates (compatibility with PyTorch schedulers)."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict:
        """Return scheduler state dict."""
        return {
            'history': self.history,
            'best_metric': self.best_metric,
            'num_bad_epochs': self.num_bad_epochs,
            'cooldown_counter': self.cooldown_counter,
            'current_patience': self.current_patience,
            'last_lr_reduction_epoch': self.last_lr_reduction_epoch,
            'phase': self.phase.value,
            'should_stop_early': self.should_stop_early,
            'base_lrs': self.base_lrs,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state dict."""
        self.history = state_dict['history']
        self.best_metric = state_dict['best_metric']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.cooldown_counter = state_dict['cooldown_counter']
        self.current_patience = state_dict['current_patience']
        self.last_lr_reduction_epoch = state_dict['last_lr_reduction_epoch']
        self.phase = PlateauPhase(state_dict['phase'])
        self.should_stop_early = state_dict['should_stop_early']
        self.base_lrs = state_dict['base_lrs']


def create_adaptive_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config_dict: Dict,
) -> AdaptiveLRScheduler:
    """Create AdaptiveLRScheduler from configuration.

    Args:
        optimizer: PyTorch optimizer
        config_dict: Configuration dictionary

    Returns:
        AdaptiveLRScheduler instance
    """
    adaptive_lr_config = config_dict.get('adaptive_lr', {})

    return AdaptiveLRScheduler(
        optimizer=optimizer,
        primary_metric=adaptive_lr_config.get('primary_metric', 'hierarchy_correlation'),
        mode=SchedulerMode(adaptive_lr_config.get('mode', 'max')),
        patience=adaptive_lr_config.get('patience', 8),
        factor=adaptive_lr_config.get('factor', 0.5),
        min_lr=adaptive_lr_config.get('min_lr', 1e-7),
        threshold=adaptive_lr_config.get('threshold', 1e-4),
        threshold_mode=adaptive_lr_config.get('threshold_mode', 'rel'),
        cooldown=adaptive_lr_config.get('cooldown', 0),
        warmup_epochs=adaptive_lr_config.get('warmup_epochs', 5),
        verbose=adaptive_lr_config.get('verbose', True),

        secondary_metrics=adaptive_lr_config.get('secondary_metrics', []),
        metric_weights=adaptive_lr_config.get('metric_weights', {}),

        adaptive_patience=adaptive_lr_config.get('adaptive_patience', True),
        recovery_detection=adaptive_lr_config.get('recovery_detection', True),
        early_stopping=adaptive_lr_config.get('early_stopping', False),
        early_stopping_patience=adaptive_lr_config.get('early_stopping_patience', 20),

        recovery_factor=adaptive_lr_config.get('recovery_factor', 1.5),
        recovery_threshold=adaptive_lr_config.get('recovery_threshold', 0.01),
    )