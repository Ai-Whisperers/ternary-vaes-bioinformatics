"""Tests for hyperbolic trainer module.

Tests cover:
- HyperbolicVAETrainer initialization
- Feedback controller initialization
- Loss module initialization
- Basic training loop structure
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_A = nn.Linear(64, latent_dim)
        self.encoder_B = nn.Linear(64, latent_dim)
        self.decoder_A = nn.Linear(latent_dim, 64)
        self.decoder_B = nn.Linear(latent_dim, 64)

    def forward(self, x):
        return {"z_A": self.encoder_A(x), "z_B": self.encoder_B(x)}


class MockBaseTrainer:
    """Mock base trainer for testing."""

    def __init__(self):
        self.monitor = MagicMock()
        self.model = MockModel()
        self.optimizer = MagicMock()
        self.scheduler = MagicMock()


class TestHyperbolicVAETrainerInit:
    """Test HyperbolicVAETrainer initialization."""

    def test_basic_initialization(self):
        """Trainer should initialize with required arguments."""
        from src.training.hyperbolic_trainer import HyperbolicVAETrainer

        base_trainer = MockBaseTrainer()
        model = MockModel()
        config = {
            "total_epochs": 100,
            "curvature": 1.0,
            "ranking_temperature": 0.1,
        }

        trainer = HyperbolicVAETrainer(
            base_trainer=base_trainer,
            model=model,
            device="cpu",
            config=config,
        )

        assert trainer.model is model
        assert trainer.device == "cpu"
        assert trainer.total_epochs == 100

    def test_monitor_fallback(self):
        """Trainer should use base_trainer monitor if none provided."""
        from src.training.hyperbolic_trainer import HyperbolicVAETrainer

        base_trainer = MockBaseTrainer()
        model = MockModel()
        config = {"curvature": 1.0}

        trainer = HyperbolicVAETrainer(
            base_trainer=base_trainer,
            model=model,
            device="cpu",
            config=config,
        )

        assert trainer.monitor is base_trainer.monitor

    def test_custom_monitor(self):
        """Trainer should use provided monitor over base_trainer monitor."""
        from src.training.hyperbolic_trainer import HyperbolicVAETrainer
        from src.training.monitor import TrainingMonitor

        base_trainer = MockBaseTrainer()
        model = MockModel()
        config = {"curvature": 1.0}
        custom_monitor = TrainingMonitor(log_to_file=False)

        trainer = HyperbolicVAETrainer(
            base_trainer=base_trainer,
            model=model,
            device="cpu",
            config=config,
            monitor=custom_monitor,
        )

        assert trainer.monitor is custom_monitor


class TestFeedbackControllers:
    """Test feedback controller initialization."""

    def test_feedback_controllers_exist(self):
        """Trainer should have feedback controller attributes."""
        from src.training.hyperbolic_trainer import HyperbolicVAETrainer

        base_trainer = MockBaseTrainer()
        model = MockModel()
        config = {"curvature": 1.0}

        trainer = HyperbolicVAETrainer(
            base_trainer=base_trainer,
            model=model,
            device="cpu",
            config=config,
        )

        # exploration_boost is the main feedback controller
        assert hasattr(trainer, "exploration_boost")


class TestHyperbolicLosses:
    """Test hyperbolic loss initialization."""

    def test_loss_modules_initialized(self):
        """Trainer should initialize hyperbolic loss modules."""
        from src.training.hyperbolic_trainer import HyperbolicVAETrainer

        base_trainer = MockBaseTrainer()
        model = MockModel()
        config = {
            "curvature": 1.0,
            "ranking_temperature": 0.1,
            "geodesic_weight": 0.5,
        }

        trainer = HyperbolicVAETrainer(
            base_trainer=base_trainer,
            model=model,
            device="cpu",
            config=config,
        )

        # Check that loss modules exist as attributes
        assert hasattr(trainer, "ranking_loss_hyp")
        assert hasattr(trainer, "hyperbolic_prior_A")
        assert hasattr(trainer, "hyperbolic_prior_B")


class TestTrainingConfig:
    """Test configuration handling."""

    def test_default_config_values(self):
        """Trainer should use default values when config is minimal."""
        from src.training.hyperbolic_trainer import HyperbolicVAETrainer

        base_trainer = MockBaseTrainer()
        model = MockModel()
        config = {}

        trainer = HyperbolicVAETrainer(
            base_trainer=base_trainer,
            model=model,
            device="cpu",
            config=config,
        )

        assert trainer.total_epochs == 100  # Default
        assert trainer.histogram_interval == 10  # Default
        assert trainer.log_interval == 10  # Default

    def test_custom_config_values(self):
        """Trainer should respect custom config values."""
        from src.training.hyperbolic_trainer import HyperbolicVAETrainer

        base_trainer = MockBaseTrainer()
        model = MockModel()
        config = {
            "total_epochs": 50,
            "histogram_interval": 5,
            "log_interval": 20,
        }

        trainer = HyperbolicVAETrainer(
            base_trainer=base_trainer,
            model=model,
            device="cpu",
            config=config,
        )

        assert trainer.total_epochs == 50
        assert trainer.histogram_interval == 5
        assert trainer.log_interval == 20


class TestCurriculumInit:
    """Test curriculum module initialization."""

    def test_curriculum_attribute_exists(self):
        """Trainer should have curriculum attribute."""
        from src.training.hyperbolic_trainer import HyperbolicVAETrainer

        base_trainer = MockBaseTrainer()
        model = MockModel()
        config = {"curvature": 1.0}

        trainer = HyperbolicVAETrainer(
            base_trainer=base_trainer,
            model=model,
            device="cpu",
            config=config,
        )

        assert hasattr(trainer, "curriculum")


class TestRadialStratification:
    """Test radial stratification loss initialization."""

    def test_radial_stratification_attributes(self):
        """Trainer should have radial stratification attributes."""
        from src.training.hyperbolic_trainer import HyperbolicVAETrainer

        base_trainer = MockBaseTrainer()
        model = MockModel()
        config = {"curvature": 1.0}

        trainer = HyperbolicVAETrainer(
            base_trainer=base_trainer,
            model=model,
            device="cpu",
            config=config,
        )

        assert hasattr(trainer, "radial_stratification_A")
        assert hasattr(trainer, "radial_stratification_B")
