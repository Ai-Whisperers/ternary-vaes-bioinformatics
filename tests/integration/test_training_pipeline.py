"""Integration tests for the training pipeline.

Tests end-to-end training workflow:
1. Config loading
2. Model instantiation
3. Data loading
4. Training step execution
5. Checkpoint save/load
6. Loss computation
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.config.constants import N_TERNARY_OPERATIONS
from src.data.generation import generate_all_ternary_operations
from src.losses.dual_vae_loss import DualVAELoss
from src.models.ternary_vae import TernaryVAEV5_11


class TestModelInstantiation:
    """Test model can be instantiated with various configurations."""

    def test_default_instantiation(self):
        """Model should instantiate with default parameters."""
        model = TernaryVAEV5_11(latent_dim=16)
        assert model is not None
        assert model.latent_dim == 16

    def test_custom_latent_dim(self):
        """Model should accept custom latent dimension."""
        for dim in [8, 16, 32, 64]:
            model = TernaryVAEV5_11(latent_dim=dim)
            assert model.latent_dim == dim

    def test_model_to_device(self):
        """Model should move to CPU without error."""
        model = TernaryVAEV5_11(latent_dim=16)
        model = model.to("cpu")
        assert next(model.parameters()).device.type == "cpu"


class SimpleTernaryDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for ternary operations."""

    def __init__(self, operations: np.ndarray):
        self.operations = torch.from_numpy(operations).float()

    def __len__(self) -> int:
        return len(self.operations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.operations[idx]


class TestDataPipeline:
    """Test data generation and loading."""

    def test_generate_ternary_operations(self):
        """Should generate correct number of operations."""
        ops = generate_all_ternary_operations()
        assert len(ops) == N_TERNARY_OPERATIONS
        # Each operation is a 9-element ternary tuple with values in {-1, 0, 1}
        assert ops.shape == (N_TERNARY_OPERATIONS, 9)
        assert np.all((ops >= -1) & (ops <= 1))

    def test_ternary_dataset_creation(self):
        """Dataset should be creatable from operations."""
        ops = generate_all_ternary_operations()
        dataset = SimpleTernaryDataset(ops)
        assert len(dataset) == N_TERNARY_OPERATIONS

    def test_ternary_dataset_getitem(self):
        """Dataset should return tensor items."""
        ops = generate_all_ternary_operations()
        dataset = SimpleTernaryDataset(ops)
        item = dataset[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (9,)

    def test_dataloader_batching(self):
        """DataLoader should batch correctly."""
        ops = generate_all_ternary_operations()
        dataset = SimpleTernaryDataset(ops)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        batch = next(iter(loader))
        assert batch.shape == (256, 9)


class TestForwardPass:
    """Test model forward pass."""

    def test_forward_returns_dict(self):
        """Forward pass should return dictionary."""
        model = TernaryVAEV5_11(latent_dim=16)
        model.eval()
        x = torch.randint(0, 3, (32, 9)).float()
        with torch.no_grad():
            output = model(x)
        assert isinstance(output, dict)

    def test_forward_contains_required_keys(self):
        """Forward output should contain required keys."""
        model = TernaryVAEV5_11(latent_dim=16)
        model.eval()
        x = torch.randint(0, 3, (32, 9)).float()
        with torch.no_grad():
            output = model(x)

        required_keys = ["logits_A", "mu_A", "logvar_A"]
        for key in required_keys:
            assert key in output, f"Missing key: {key}"

    def test_forward_batch_size_preserved(self):
        """Output batch size should match input."""
        model = TernaryVAEV5_11(latent_dim=16)
        model.eval()
        batch_sizes = [1, 16, 32, 64, 128]
        for bs in batch_sizes:
            x = torch.randint(0, 3, (bs, 9)).float()
            with torch.no_grad():
                output = model(x)
            assert output["logits_A"].shape[0] == bs


class TestLossComputation:
    """Test loss function computation."""

    def test_reconstruction_loss(self):
        """Reconstruction loss should compute correctly."""
        from src.losses.dual_vae_loss import ReconstructionLoss

        loss_fn = ReconstructionLoss()
        assert loss_fn is not None

        # Create sample logits and inputs in {-1, 0, 1}
        logits = torch.randn(32, 9, 3)
        x = torch.randint(-1, 2, (32, 9)).float()

        loss = loss_fn(logits, x)
        assert loss.item() > 0

    def test_kl_divergence_loss(self):
        """KL divergence loss should compute correctly."""
        from src.losses.dual_vae_loss import KLDivergenceLoss

        loss_fn = KLDivergenceLoss(free_bits=0.0)
        assert loss_fn is not None

        mu = torch.randn(32, 16)
        logvar = torch.randn(32, 16)

        loss = loss_fn(mu, logvar)
        assert loss.item() >= 0  # KL is non-negative


class TestTrainingStep:
    """Test training step with hyperbolic projection.

    Note: TernaryVAEV5_11 uses frozen encoders/decoders from v5.5.
    Only the hyperbolic projection layer is trainable.
    Training uses geodesic/radial losses on z_hyp, not reconstruction.
    """

    def test_trainable_parameters_exist(self):
        """Model should have some trainable parameters (projection layer)."""
        model = TernaryVAEV5_11(latent_dim=16)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        # The projection layer should be trainable
        assert len(trainable_params) > 0

    def test_frozen_parameters_exist(self):
        """Model should have frozen parameters (encoders/decoders)."""
        model = TernaryVAEV5_11(latent_dim=16)
        frozen_params = [p for p in model.parameters() if not p.requires_grad]
        # The encoders and decoders should be frozen
        assert len(frozen_params) > 0

    def test_projection_forward_pass(self):
        """Projection layer should produce hyperbolic outputs."""
        model = TernaryVAEV5_11(latent_dim=16)
        model.train()
        x = torch.randint(-1, 2, (32, 9)).float()

        output = model(x)

        # Check hyperbolic outputs exist (model outputs z_A_hyp)
        assert "z_A_hyp" in output, f"Expected z_A_hyp in output, got keys: {output.keys()}"
        # Outputs should be within Poincare ball (norm < 1)
        z = output["z_A_hyp"]
        norms = torch.norm(z, dim=-1)
        assert torch.all(norms <= 1.0)


class TestCheckpointSaveLoad:
    """Test checkpoint saving and loading."""

    def test_save_and_load_state_dict(self):
        """Model state dict should be saveable and loadable."""
        model = TernaryVAEV5_11(latent_dim=16)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            # Save
            torch.save(model.state_dict(), path)

            # Create new model and load
            model2 = TernaryVAEV5_11(latent_dim=16)
            model2.load_state_dict(torch.load(path, weights_only=True))

            # Verify state dicts match
            for key in model.state_dict():
                assert key in model2.state_dict()
                assert torch.equal(model.state_dict()[key], model2.state_dict()[key])

    def test_save_checkpoint_structure(self):
        """Checkpoint should save with correct structure."""
        model = TernaryVAEV5_11(latent_dim=16)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "full_checkpoint.pt"

            # Save checkpoint
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": 5,
            }
            torch.save(checkpoint, path)

            # Load and verify
            loaded = torch.load(path, weights_only=False)
            assert "model" in loaded
            assert "optimizer" in loaded
            assert loaded["epoch"] == 5


class TestModelArchitecture:
    """Test model architecture properties."""

    def test_encoder_is_frozen(self):
        """Encoder parameters should be frozen."""
        model = TernaryVAEV5_11(latent_dim=16)
        for param in model.encoder_A.parameters():
            assert not param.requires_grad, "Encoder A should be frozen"
        for param in model.encoder_B.parameters():
            assert not param.requires_grad, "Encoder B should be frozen"

    def test_decoder_is_frozen(self):
        """Decoder parameters should be frozen."""
        model = TernaryVAEV5_11(latent_dim=16)
        for param in model.decoder_A.parameters():
            assert not param.requires_grad, "Decoder A should be frozen"

    def test_projection_is_trainable(self):
        """Projection layer parameters should be trainable."""
        model = TernaryVAEV5_11(latent_dim=16)
        # Check projection has trainable params
        projection_params = list(model.projection.parameters())
        assert len(projection_params) > 0
        assert any(p.requires_grad for p in projection_params)


class TestReproducibility:
    """Test reproducibility with seeds."""

    def test_deterministic_forward_with_seed(self):
        """Same seed should produce same forward pass results."""
        def forward_pass(seed: int):
            torch.manual_seed(seed)
            model = TernaryVAEV5_11(latent_dim=16)
            model.eval()

            x = torch.randint(-1, 2, (32, 9)).float()
            with torch.no_grad():
                output = model(x)

            return output["logits_A"].sum().item()

        result1 = forward_pass(42)
        result2 = forward_pass(42)
        result3 = forward_pass(123)

        assert result1 == result2, "Same seed should produce same result"
        assert result1 != result3, "Different seeds should produce different results"
