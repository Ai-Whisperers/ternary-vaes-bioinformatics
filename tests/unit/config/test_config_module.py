# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for the centralized config module.

Tests cover:
- Constants module
- Schema validation
- Config loading
- Environment variable overrides
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.config import (
    ConfigValidationError,
    GeometryConfig,
    LossWeights,
    OptimizerConfig,
    RankingConfig,
    TrainingConfig,
    VAEConfig,
    load_config,
    save_config,
)
from src.config.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CURVATURE,
    DEFAULT_EPOCHS,
    DEFAULT_LATENT_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_RADIUS,
    EPSILON,
    N_TERNARY_OPERATIONS,
)


class TestConstants:
    """Tests for configuration constants."""

    def test_epsilon_is_small_positive(self):
        """EPSILON should be a small positive number."""
        assert EPSILON > 0
        assert EPSILON < 1e-6

    def test_ternary_operations_is_3_to_9(self):
        """N_TERNARY_OPERATIONS should be 3^9 = 19683."""
        assert N_TERNARY_OPERATIONS == 3**9
        assert N_TERNARY_OPERATIONS == 19683

    def test_default_curvature_is_positive(self):
        """DEFAULT_CURVATURE should be positive."""
        assert DEFAULT_CURVATURE > 0

    def test_default_max_radius_in_valid_range(self):
        """DEFAULT_MAX_RADIUS should be in (0, 1)."""
        assert 0 < DEFAULT_MAX_RADIUS < 1

    def test_default_latent_dim_is_reasonable(self):
        """DEFAULT_LATENT_DIM should be >= 2."""
        assert DEFAULT_LATENT_DIM >= 2


class TestGeometryConfig:
    """Tests for GeometryConfig dataclass."""

    def test_default_values(self):
        """Test default geometry configuration."""
        config = GeometryConfig()
        assert config.curvature == DEFAULT_CURVATURE
        assert config.max_radius == DEFAULT_MAX_RADIUS
        assert config.latent_dim == DEFAULT_LATENT_DIM
        assert config.learnable_curvature is False

    def test_custom_values(self):
        """Test custom geometry configuration."""
        config = GeometryConfig(curvature=2.0, max_radius=0.9, latent_dim=32)
        assert config.curvature == 2.0
        assert config.max_radius == 0.9
        assert config.latent_dim == 32

    def test_invalid_curvature_raises(self):
        """Negative curvature should raise error."""
        with pytest.raises(ConfigValidationError):
            GeometryConfig(curvature=-1.0)

    def test_invalid_max_radius_raises(self):
        """max_radius outside (0, 1) should raise error."""
        with pytest.raises(ConfigValidationError):
            GeometryConfig(max_radius=1.5)
        with pytest.raises(ConfigValidationError):
            GeometryConfig(max_radius=0.0)

    def test_invalid_latent_dim_raises(self):
        """latent_dim < 2 should raise error."""
        with pytest.raises(ConfigValidationError):
            GeometryConfig(latent_dim=1)


class TestLossWeights:
    """Tests for LossWeights dataclass."""

    def test_default_values(self):
        """Test default loss weights."""
        weights = LossWeights()
        assert weights.reconstruction == 1.0
        assert weights.kl_divergence == 1.0
        assert weights.ranking == 0.5

    def test_custom_values(self):
        """Test custom loss weights."""
        weights = LossWeights(reconstruction=2.0, kl_divergence=0.5)
        assert weights.reconstruction == 2.0
        assert weights.kl_divergence == 0.5

    def test_negative_weight_raises(self):
        """Negative weights should raise error."""
        with pytest.raises(ConfigValidationError):
            LossWeights(reconstruction=-1.0)


class TestOptimizerConfig:
    """Tests for OptimizerConfig dataclass."""

    def test_default_values(self):
        """Test default optimizer configuration."""
        config = OptimizerConfig()
        assert config.type == "adamw"
        assert config.learning_rate == DEFAULT_LEARNING_RATE
        assert config.schedule == "constant"

    def test_custom_values(self):
        """Test custom optimizer configuration."""
        config = OptimizerConfig(type="adam", learning_rate=1e-3, schedule="cosine")
        assert config.type == "adam"
        assert config.learning_rate == 1e-3
        assert config.schedule == "cosine"


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.seed == 42
        assert config.epochs == DEFAULT_EPOCHS
        assert config.batch_size == DEFAULT_BATCH_SIZE
        assert isinstance(config.geometry, GeometryConfig)
        assert isinstance(config.optimizer, OptimizerConfig)
        assert isinstance(config.loss_weights, LossWeights)

    def test_custom_values(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            seed=123,
            epochs=500,
            batch_size=128,
        )
        assert config.seed == 123
        assert config.epochs == 500
        assert config.batch_size == 128

    def test_invalid_epochs_raises(self):
        """Invalid epochs should raise error."""
        with pytest.raises(ConfigValidationError):
            TrainingConfig(epochs=0)

    def test_invalid_batch_size_raises(self):
        """Invalid batch_size should raise error."""
        with pytest.raises(ConfigValidationError):
            TrainingConfig(batch_size=0)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "epochs": 200,
            "batch_size": 64,
            "geometry": {"curvature": 2.0},
            "optimizer": {"learning_rate": 0.001},
        }
        config = TrainingConfig.from_dict(data)
        assert config.epochs == 200
        assert config.batch_size == 64
        assert config.geometry.curvature == 2.0
        assert config.optimizer.learning_rate == 0.001

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TrainingConfig(epochs=100)
        data = config.to_dict()
        assert data["epochs"] == 100
        assert "geometry" in data
        assert "optimizer" in data


class TestLoadConfig:
    """Tests for config loading functionality."""

    def test_load_defaults(self):
        """Test loading with all defaults."""
        config = load_config()
        assert isinstance(config, TrainingConfig)
        assert config.epochs == DEFAULT_EPOCHS

    def test_load_with_overrides(self):
        """Test loading with explicit overrides."""
        config = load_config(overrides={"epochs": 999, "batch_size": 256})
        assert config.epochs == 999
        assert config.batch_size == 256

    def test_load_from_yaml(self):
        """Test loading from YAML file."""
        yaml_content = """
epochs: 500
batch_size: 128
geometry:
  curvature: 2.0
  latent_dim: 32
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            config = load_config(f.name)

            assert config.epochs == 500
            assert config.batch_size == 128
            assert config.geometry.curvature == 2.0
            assert config.geometry.latent_dim == 32

        os.unlink(f.name)

    def test_load_nonexistent_file_raises(self):
        """Loading nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


class TestSaveConfig:
    """Tests for config saving functionality."""

    def test_save_creates_file(self):
        """Test that save_config creates a valid YAML file."""
        original = TrainingConfig(
            epochs=123,
            batch_size=64,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            save_config(original, config_path)

            # Verify file was created
            assert config_path.exists()

            # Verify content is valid YAML (can be parsed with unsafe loader)
            import yaml

            with open(config_path) as f:
                data = yaml.unsafe_load(f)

            assert data["epochs"] == 123
            assert data["batch_size"] == 64


class TestEnvironmentVariables:
    """Tests for environment variable overrides."""

    def test_env_override_epochs(self, monkeypatch):
        """Test TVAE_EPOCHS environment variable."""
        monkeypatch.setenv("TVAE_EPOCHS", "777")
        config = load_config()
        assert config.epochs == 777

    def test_env_override_batch_size(self, monkeypatch):
        """Test TVAE_BATCH_SIZE environment variable."""
        monkeypatch.setenv("TVAE_BATCH_SIZE", "256")
        config = load_config()
        assert config.batch_size == 256

    def test_env_override_learning_rate(self, monkeypatch):
        """Test TVAE_OPTIMIZER_LEARNING_RATE environment variable."""
        monkeypatch.setenv("TVAE_OPTIMIZER_LEARNING_RATE", "0.0005")
        config = load_config()
        assert config.optimizer.learning_rate == 0.0005

    def test_explicit_override_beats_env(self, monkeypatch):
        """Explicit overrides should beat environment variables."""
        monkeypatch.setenv("TVAE_EPOCHS", "100")
        config = load_config(overrides={"epochs": 200})
        assert config.epochs == 200
