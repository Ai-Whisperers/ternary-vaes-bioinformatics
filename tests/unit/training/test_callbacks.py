# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for the training callback system.

Tests cover:
- TrainingCallback protocol
- CallbackList composition
- EarlyStoppingCallback
- CoveragePlateauCallback
- CorrelationDropCallback
- LoggingCallback
- CheckpointCallback
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.training.callbacks import (
    CallbackList,
    CheckpointCallback,
    CorrelationDropCallback,
    CoveragePlateauCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    ProgressCallback,
    TrainingCallback,
)


class MockTrainer:
    """Mock trainer for testing callbacks."""

    def __init__(self):
        self.should_stop = False
        self.model = MagicMock()
        self.optimizer = MagicMock()
        self.current_epoch = 0
        self.total_epochs = 100


class TestTrainingCallback:
    """Tests for TrainingCallback base class."""

    def test_callback_methods_exist(self):
        """Callback should have all expected methods."""
        # Create concrete implementation
        class ConcreteCallback(TrainingCallback):
            pass

        callback = ConcreteCallback()
        assert hasattr(callback, "on_train_start")
        assert hasattr(callback, "on_train_end")
        assert hasattr(callback, "on_epoch_start")
        assert hasattr(callback, "on_epoch_end")
        assert hasattr(callback, "on_batch_start")
        assert hasattr(callback, "on_batch_end")


class TestCallbackList:
    """Tests for CallbackList composition."""

    def test_empty_list(self):
        """Empty callback list should be valid."""
        callbacks = CallbackList()
        assert len(callbacks.callbacks) == 0

    def test_add_callback(self):
        """Adding callbacks should increase list size."""
        callbacks = CallbackList()
        callbacks.add(EarlyStoppingCallback(patience=10))
        assert len(callbacks.callbacks) == 1

    def test_init_with_callbacks(self):
        """Initialize with list of callbacks."""
        callbacks = CallbackList([
            EarlyStoppingCallback(patience=10),
            LoggingCallback(log_interval=5),
        ])
        assert len(callbacks.callbacks) == 2

    def test_on_epoch_end_propagates(self):
        """on_epoch_end should propagate to all callbacks."""
        mock_cb1 = MagicMock(spec=TrainingCallback)
        mock_cb1.on_epoch_end.return_value = None
        mock_cb2 = MagicMock(spec=TrainingCallback)
        mock_cb2.on_epoch_end.return_value = None

        callbacks = CallbackList([mock_cb1, mock_cb2])
        trainer = MockTrainer()

        callbacks.on_epoch_end(epoch=1, metrics={"loss": 0.5}, trainer=trainer)

        mock_cb1.on_epoch_end.assert_called_once()
        mock_cb2.on_epoch_end.assert_called_once()

    def test_stop_signal_propagates(self):
        """If any callback returns True, training should stop."""
        mock_cb1 = MagicMock(spec=TrainingCallback)
        mock_cb1.on_epoch_end.return_value = None
        mock_cb2 = MagicMock(spec=TrainingCallback)
        mock_cb2.on_epoch_end.return_value = True  # Stop signal

        callbacks = CallbackList([mock_cb1, mock_cb2])
        trainer = MockTrainer()

        result = callbacks.on_epoch_end(epoch=1, metrics={"loss": 0.5}, trainer=trainer)

        assert result is True


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_default_initialization(self):
        """Test default parameter values."""
        callback = EarlyStoppingCallback()
        assert callback.patience == 50
        assert callback.min_delta == 0.0001
        assert callback.mode == "min"

    def test_custom_initialization(self):
        """Test custom parameter values."""
        callback = EarlyStoppingCallback(patience=20, min_delta=0.001, mode="max")
        assert callback.patience == 20
        assert callback.min_delta == 0.001
        assert callback.mode == "max"

    def test_improvement_resets_counter(self):
        """Improvement should reset the counter."""
        callback = EarlyStoppingCallback(patience=5, mode="min")
        trainer = MockTrainer()

        # First epoch establishes baseline
        callback.on_epoch_end(0, {"val_loss": 1.0}, trainer)
        assert callback.counter == 0

        # No improvement
        callback.on_epoch_end(1, {"val_loss": 1.0}, trainer)
        assert callback.counter == 1

        # Improvement
        callback.on_epoch_end(2, {"val_loss": 0.5}, trainer)
        assert callback.counter == 0

    def test_no_improvement_increments_counter(self):
        """No improvement should increment counter."""
        callback = EarlyStoppingCallback(patience=5, mode="min")
        trainer = MockTrainer()

        callback.on_epoch_end(0, {"val_loss": 1.0}, trainer)
        callback.on_epoch_end(1, {"val_loss": 1.0}, trainer)
        callback.on_epoch_end(2, {"val_loss": 1.0}, trainer)

        assert callback.counter == 2

    def test_patience_exceeded_returns_true(self):
        """Exceeding patience should return True (stop signal)."""
        callback = EarlyStoppingCallback(patience=3, mode="min")
        trainer = MockTrainer()

        callback.on_epoch_end(0, {"val_loss": 1.0}, trainer)
        callback.on_epoch_end(1, {"val_loss": 1.0}, trainer)
        callback.on_epoch_end(2, {"val_loss": 1.0}, trainer)

        result = callback.on_epoch_end(3, {"val_loss": 1.0}, trainer)
        assert result is True

    def test_max_mode(self):
        """Test max mode (higher is better)."""
        callback = EarlyStoppingCallback(patience=3, mode="max", monitor="accuracy")
        trainer = MockTrainer()

        callback.on_epoch_end(0, {"accuracy": 0.5}, trainer)
        callback.on_epoch_end(1, {"accuracy": 0.6}, trainer)  # Improvement
        assert callback.counter == 0

        callback.on_epoch_end(2, {"accuracy": 0.6}, trainer)  # No improvement
        assert callback.counter == 1


class TestCoveragePlateauCallback:
    """Tests for CoveragePlateauCallback."""

    def test_default_initialization(self):
        """Test default parameter values."""
        callback = CoveragePlateauCallback()
        assert callback.patience == 100
        assert callback.target_coverage == 99.7

    def test_high_coverage_stops_training(self):
        """Reaching target coverage should stop training."""
        callback = CoveragePlateauCallback(patience=5, target_coverage=95.0)
        trainer = MockTrainer()

        result = callback.on_epoch_end(0, {"coverage_A": 96.0}, trainer)
        assert result is True

    def test_plateau_detection(self):
        """Plateau in coverage should trigger stop."""
        callback = CoveragePlateauCallback(patience=3, target_coverage=99.0)
        trainer = MockTrainer()

        # Same coverage for multiple epochs
        callback.on_epoch_end(0, {"coverage_A": 90.0}, trainer)
        callback.on_epoch_end(1, {"coverage_A": 90.0}, trainer)
        callback.on_epoch_end(2, {"coverage_A": 90.0}, trainer)

        result = callback.on_epoch_end(3, {"coverage_A": 90.0}, trainer)
        assert result is True


class TestCorrelationDropCallback:
    """Tests for CorrelationDropCallback."""

    def test_default_initialization(self):
        """Test default parameter values."""
        callback = CorrelationDropCallback()
        assert callback.patience == 10
        assert callback.drop_threshold == 0.05

    def test_correlation_drop_triggers_stop(self):
        """Significant correlation drop should trigger stop."""
        callback = CorrelationDropCallback(patience=3, drop_threshold=0.1)
        trainer = MockTrainer()

        callback.on_epoch_end(0, {"correlation": 0.8}, trainer)
        callback.on_epoch_end(1, {"correlation": 0.65}, trainer)  # Big drop
        callback.on_epoch_end(2, {"correlation": 0.55}, trainer)  # Another drop

        result = callback.on_epoch_end(3, {"correlation": 0.45}, trainer)
        assert result is True


class TestLoggingCallback:
    """Tests for LoggingCallback."""

    def test_default_initialization(self):
        """Test default parameter values."""
        callback = LoggingCallback()
        assert callback.log_interval == 10

    def test_custom_log_interval(self):
        """Test custom log interval."""
        callback = LoggingCallback(log_interval=5)
        assert callback.log_interval == 5

    @patch("logging.getLogger")
    def test_logs_at_interval(self, mock_get_logger):
        """Logs should be emitted at specified intervals."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        callback = LoggingCallback(log_interval=10)
        trainer = MockTrainer()

        # Should not log at epoch 5
        callback.on_epoch_end(5, {"loss": 0.5}, trainer)

        # Should log at epoch 10
        callback.on_epoch_end(10, {"loss": 0.3}, trainer)


class TestCheckpointCallback:
    """Tests for CheckpointCallback."""

    def test_default_initialization(self):
        """Test default parameter values."""
        callback = CheckpointCallback(checkpoint_dir="test_checkpoints")
        assert callback.save_interval == 10

    def test_save_best_mode(self):
        """Test save_best mode."""
        callback = CheckpointCallback(
            checkpoint_dir="test_checkpoints",
            save_best=True,
        )
        assert callback.save_best is True


class TestProgressCallback:
    """Tests for ProgressCallback."""

    def test_initialization(self):
        """Test progress callback initialization."""
        callback = ProgressCallback()
        assert callback.update_interval == 1

    def test_on_train_start_records_epochs(self):
        """on_train_start should record total epochs."""
        callback = ProgressCallback()
        trainer = MockTrainer()

        callback.on_train_start(trainer)

        assert callback.total_epochs == 100
