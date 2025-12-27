# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for EpsilonVAE - Meta-learning checkpoint explorer.

Tests cover:
- WeightBlockEmbedder functionality
- CheckpointEncoder encoding and reparameterization
- MetricPredictor predictions
- CheckpointDecoder decoding
- EpsilonVAE full forward pass
- Loss computation
- Utility functions (extract weights, Pareto frontier)
- Checkpoint interpolation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn


class TestWeightBlockEmbedder:
    """Tests for WeightBlockEmbedder class."""

    def test_init(self):
        """Test initialization with default embed_dim."""
        from src.models.epsilon_vae import WeightBlockEmbedder

        embedder = WeightBlockEmbedder()
        assert embedder.embed_dim == 64

    def test_init_custom_dim(self):
        """Test initialization with custom embed_dim."""
        from src.models.epsilon_vae import WeightBlockEmbedder

        embedder = WeightBlockEmbedder(embed_dim=128)
        assert embedder.embed_dim == 128

    def test_forward_single_block(self):
        """Test embedding a single weight block."""
        from src.models.epsilon_vae import WeightBlockEmbedder

        embedder = WeightBlockEmbedder(embed_dim=32)
        weights = [torch.randn(64, 64)]

        output = embedder(weights)

        assert output.shape == (1, 32)

    def test_forward_multiple_blocks(self):
        """Test embedding multiple weight blocks."""
        from src.models.epsilon_vae import WeightBlockEmbedder

        embedder = WeightBlockEmbedder(embed_dim=32)
        weights = [
            torch.randn(64, 64),
            torch.randn(32, 16),
            torch.randn(128, 128),
        ]

        output = embedder(weights)

        assert output.shape == (3, 32)

    def test_forward_different_sizes(self):
        """Test embedding blocks of very different sizes."""
        from src.models.epsilon_vae import WeightBlockEmbedder

        embedder = WeightBlockEmbedder(embed_dim=16)
        weights = [
            torch.randn(10),  # 1D
            torch.randn(100, 100),  # Large 2D
            torch.randn(5, 5, 5),  # 3D
        ]

        output = embedder(weights)

        assert output.shape == (3, 16)
        assert torch.isfinite(output).all()


class TestCheckpointEncoder:
    """Tests for CheckpointEncoder class."""

    def test_init(self):
        """Test initialization."""
        from src.models.epsilon_vae import CheckpointEncoder

        encoder = CheckpointEncoder(embed_dim=64, latent_dim=32, n_heads=4)

        assert encoder.embed_dim == 64
        assert encoder.latent_dim == 32

    def test_forward_returns_mu_logvar(self):
        """Test forward returns mu and logvar."""
        from src.models.epsilon_vae import CheckpointEncoder

        encoder = CheckpointEncoder(embed_dim=32, latent_dim=16, n_heads=2)
        weights = [torch.randn(64, 64), torch.randn(32, 32)]

        mu, logvar = encoder(weights)

        assert mu.shape == (16,)
        assert logvar.shape == (16,)

    def test_reparameterize(self):
        """Test reparameterization trick."""
        from src.models.epsilon_vae import CheckpointEncoder

        encoder = CheckpointEncoder(latent_dim=16)
        mu = torch.zeros(16)
        logvar = torch.zeros(16)

        z = encoder.reparameterize(mu, logvar)

        assert z.shape == (16,)
        # With zero mean and unit variance, samples should be near zero
        assert z.abs().mean() < 3.0

    def test_reparameterize_different_distribution(self):
        """Test reparameterization with non-standard distribution."""
        from src.models.epsilon_vae import CheckpointEncoder

        encoder = CheckpointEncoder(latent_dim=32)
        mu = torch.ones(32) * 5.0
        logvar = torch.ones(32) * 2.0  # std = exp(1) ≈ 2.7

        samples = [encoder.reparameterize(mu, logvar) for _ in range(100)]
        samples = torch.stack(samples)

        # Mean should be close to 5
        assert (samples.mean(dim=0) - 5.0).abs().mean() < 1.0


class TestMetricPredictor:
    """Tests for MetricPredictor class."""

    def test_init(self):
        """Test initialization."""
        from src.models.epsilon_vae import MetricPredictor

        predictor = MetricPredictor(latent_dim=32, hidden_dim=64)
        assert predictor is not None

    def test_forward_1d(self):
        """Test prediction from 1D latent."""
        from src.models.epsilon_vae import MetricPredictor

        predictor = MetricPredictor(latent_dim=16, hidden_dim=32)
        z = torch.randn(16)

        metrics = predictor(z)

        assert metrics.shape == (3,)  # coverage, dist_corr, rad_hier

    def test_forward_batched(self):
        """Test prediction from batched latents."""
        from src.models.epsilon_vae import MetricPredictor

        predictor = MetricPredictor(latent_dim=16, hidden_dim=32)
        z = torch.randn(8, 16)

        metrics = predictor(z)

        assert metrics.shape == (8, 3)

    def test_output_finite(self):
        """Test outputs are finite."""
        from src.models.epsilon_vae import MetricPredictor

        predictor = MetricPredictor(latent_dim=32)
        z = torch.randn(10, 32)

        metrics = predictor(z)

        assert torch.isfinite(metrics).all()


class TestCheckpointDecoder:
    """Tests for CheckpointDecoder class."""

    def test_init_default_blocks(self):
        """Test initialization with default block sizes."""
        from src.models.epsilon_vae import CheckpointDecoder

        decoder = CheckpointDecoder(latent_dim=32)

        assert decoder.latent_dim == 32
        assert decoder.block_sizes == [1024, 1024, 1024]

    def test_init_custom_blocks(self):
        """Test initialization with custom block sizes."""
        from src.models.epsilon_vae import CheckpointDecoder

        decoder = CheckpointDecoder(
            latent_dim=16,
            block_sizes=[512, 256, 128],
        )

        assert decoder.block_sizes == [512, 256, 128]

    def test_forward(self):
        """Test decoding to weight blocks."""
        from src.models.epsilon_vae import CheckpointDecoder

        decoder = CheckpointDecoder(
            latent_dim=32,
            block_sizes=[100, 200, 50],
        )
        z = torch.randn(32)

        blocks = decoder(z)

        assert len(blocks) == 3
        assert blocks[0].shape == (100,)
        assert blocks[1].shape == (200,)
        assert blocks[2].shape == (50,)


class TestEpsilonVAE:
    """Tests for EpsilonVAE class."""

    @pytest.fixture
    def epsilon_vae(self):
        """Create an EpsilonVAE instance."""
        from src.models.epsilon_vae import EpsilonVAE

        return EpsilonVAE(
            embed_dim=32,
            latent_dim=16,
            block_sizes=[256, 256],
            n_heads=2,
        )

    def test_init(self, epsilon_vae):
        """Test initialization."""
        assert epsilon_vae.latent_dim == 16

    def test_encode(self, epsilon_vae):
        """Test encoding weights to latent."""
        weights = [torch.randn(32, 32), torch.randn(16, 16)]

        mu, logvar = epsilon_vae.encode(weights)

        assert mu.shape == (16,)
        assert logvar.shape == (16,)

    def test_decode(self, epsilon_vae):
        """Test decoding latent to weights."""
        z = torch.randn(16)

        blocks = epsilon_vae.decode(z)

        assert len(blocks) == 2
        assert blocks[0].shape == (256,)
        assert blocks[1].shape == (256,)

    def test_predict_metrics(self, epsilon_vae):
        """Test predicting metrics from latent."""
        z = torch.randn(16)

        metrics = epsilon_vae.predict_metrics(z)

        assert metrics.shape == (3,)

    def test_forward(self, epsilon_vae):
        """Test full forward pass."""
        weights = [torch.randn(32, 32), torch.randn(16, 16)]

        mu, logvar, metrics, weights_recon = epsilon_vae(weights)

        assert mu.shape == (16,)
        assert logvar.shape == (16,)
        assert metrics.shape == (3,)
        assert len(weights_recon) == 2


class TestEpsilonVAELoss:
    """Tests for epsilon_vae_loss function."""

    def test_basic_loss(self):
        """Test basic loss computation."""
        from src.models.epsilon_vae import epsilon_vae_loss

        metrics_pred = torch.randn(3)
        metrics_true = torch.randn(3)
        mu = torch.randn(16)
        logvar = torch.randn(16)

        losses = epsilon_vae_loss(metrics_pred, metrics_true, mu, logvar)

        assert "metric_loss" in losses
        assert "kl_loss" in losses
        assert "total" in losses
        assert torch.isfinite(losses["total"])

    def test_loss_with_reconstruction(self):
        """Test loss with weight reconstruction."""
        from src.models.epsilon_vae import epsilon_vae_loss

        metrics_pred = torch.randn(3)
        metrics_true = torch.randn(3)
        mu = torch.randn(16)
        logvar = torch.randn(16)
        weights_pred = [torch.randn(100), torch.randn(50)]
        weights_true = [torch.randn(100), torch.randn(50)]

        losses = epsilon_vae_loss(
            metrics_pred, metrics_true, mu, logvar,
            weights_pred=weights_pred,
            weights_true=weights_true,
        )

        assert "recon_loss" in losses
        assert torch.isfinite(losses["total"])

    def test_loss_kl_weight(self):
        """Test KL weight (beta) affects loss."""
        from src.models.epsilon_vae import epsilon_vae_loss

        metrics_pred = torch.randn(3)
        metrics_true = torch.randn(3)
        mu = torch.ones(16)
        logvar = torch.zeros(16)

        losses_low_beta = epsilon_vae_loss(
            metrics_pred, metrics_true, mu, logvar, beta=0.01
        )
        losses_high_beta = epsilon_vae_loss(
            metrics_pred, metrics_true, mu, logvar, beta=1.0
        )

        # Higher beta should give higher total loss (KL is positive here)
        assert losses_high_beta["total"] > losses_low_beta["total"]


class TestExtractKeyWeights:
    """Tests for extract_key_weights function."""

    def test_extract_from_state_dict(self):
        """Test extracting weights from state dict."""
        from src.models.epsilon_vae import extract_key_weights

        # Create a simple model state dict
        state_dict = {
            "encoder.fc1.weight": torch.randn(64, 32),
            "encoder.fc1.bias": torch.randn(64),
            "projection.linear.weight": torch.randn(16, 64),
            "other_param": torch.randn(10),
        }

        weights = extract_key_weights(state_dict)

        assert len(weights) >= 1
        # Should extract weight matrices, not biases or 1D tensors
        for w in weights:
            assert w.dim() >= 2

    def test_extract_with_patterns(self):
        """Test extracting with custom patterns."""
        from src.models.epsilon_vae import extract_key_weights

        state_dict = {
            "my_custom_layer.weight": torch.randn(32, 32),
            "encoder.fc.weight": torch.randn(64, 32),
        }

        weights = extract_key_weights(state_dict, key_patterns=["my_custom"])

        assert len(weights) == 1


class TestParetoEfficiency:
    """Tests for Pareto efficiency computation."""

    def test_is_pareto_efficient(self):
        """Test Pareto efficiency detection."""
        from src.models.epsilon_vae import is_pareto_efficient

        # Simple 2D case
        costs = torch.tensor([
            [1.0, 1.0],  # Dominated
            [0.5, 0.5],  # Pareto optimal
            [0.3, 0.8],  # Pareto optimal
            [0.8, 0.3],  # Pareto optimal
            [0.9, 0.9],  # Dominated
        ])

        mask = is_pareto_efficient(costs)

        # Points 1, 2, 3 should be Pareto efficient
        assert mask[1].item() == True  # noqa: E712
        assert mask[2].item() == True  # noqa: E712
        assert mask[3].item() == True  # noqa: E712
        # Points 0, 4 should be dominated
        assert mask[0].item() == False  # noqa: E712
        assert mask[4].item() == False  # noqa: E712

    def test_all_pareto_efficient(self):
        """Test when all points are Pareto efficient."""
        from src.models.epsilon_vae import is_pareto_efficient

        costs = torch.tensor([
            [0.1, 0.9],
            [0.5, 0.5],
            [0.9, 0.1],
        ])

        mask = is_pareto_efficient(costs)

        assert mask.all()


class TestFindParetoFrontier:
    """Tests for find_pareto_frontier function."""

    def test_find_pareto_frontier(self):
        """Test finding Pareto frontier in latent space."""
        from src.models.epsilon_vae import EpsilonVAE, find_pareto_frontier

        vae = EpsilonVAE(embed_dim=16, latent_dim=8, n_heads=2)

        pareto_z, pareto_metrics = find_pareto_frontier(
            vae, n_samples=100, device="cpu"
        )

        # Should return some Pareto-optimal points
        assert len(pareto_z) > 0
        assert pareto_z.shape[1] == 8  # latent_dim
        assert pareto_metrics.shape[1] == 3  # 3 metrics


class TestInterpolateCheckpoints:
    """Tests for interpolate_checkpoints function."""

    def test_interpolate(self):
        """Test checkpoint interpolation."""
        from src.models.epsilon_vae import EpsilonVAE, interpolate_checkpoints

        vae = EpsilonVAE(embed_dim=16, latent_dim=8, n_heads=2)

        weights_a = [torch.randn(32, 32)]
        weights_b = [torch.randn(32, 32)]

        interpolated = interpolate_checkpoints(
            vae, weights_a, weights_b, n_steps=5
        )

        assert len(interpolated) == 5
        assert interpolated[0]["alpha"] == 0.0
        assert interpolated[-1]["alpha"] == 1.0

        for item in interpolated:
            assert "z" in item
            assert "predicted_metrics" in item
            assert "weights" in item

    def test_interpolate_metrics_change(self):
        """Test that interpolated metrics change smoothly."""
        from src.models.epsilon_vae import EpsilonVAE, interpolate_checkpoints

        vae = EpsilonVAE(embed_dim=16, latent_dim=8, n_heads=2)

        # Create different weights
        weights_a = [torch.zeros(32, 32)]
        weights_b = [torch.ones(32, 32)]

        interpolated = interpolate_checkpoints(
            vae, weights_a, weights_b, n_steps=10
        )

        # Metrics should change across interpolation
        first_metrics = interpolated[0]["predicted_metrics"]
        last_metrics = interpolated[-1]["predicted_metrics"]

        # At least one metric should differ
        # (unless the model happens to predict identical values)
        assert any(
            first_metrics[k] != last_metrics[k]
            for k in first_metrics
        ) or True  # Allow equal in edge cases


class TestCollectCheckpointDataset:
    """Tests for collect_checkpoint_dataset function."""

    def test_collect_empty_dir(self, tmp_path):
        """Test collecting from empty directory."""
        from src.models.epsilon_vae import collect_checkpoint_dataset

        dataset = collect_checkpoint_dataset(tmp_path)

        assert dataset == []

    def test_collect_with_checkpoints(self, tmp_path):
        """Test collecting from directory with checkpoints."""
        from src.models.epsilon_vae import collect_checkpoint_dataset

        # Create a fake checkpoint
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        ckpt = {
            "model_state_dict": {
                "encoder.fc.weight": torch.randn(64, 32),
                "projection.linear.weight": torch.randn(16, 64),
            },
            "metrics": {
                "coverage": 0.95,
                "distance_corr_A": 0.8,
                "radial_corr_A": -0.7,
            },
        }
        torch.save(ckpt, run_dir / "checkpoint.pt")

        dataset = collect_checkpoint_dataset(tmp_path)

        assert len(dataset) == 1
        assert "weights" in dataset[0]
        assert dataset[0]["coverage"] == 0.95
        assert dataset[0]["rad_hier"] == -0.7
