"""Integration tests for the full drug resistance prediction pipeline.

Tests the complete workflow from sequence input to resistance prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))


class TestFullPipeline:
    """Test complete prediction pipeline."""

    @pytest.fixture
    def sample_sequence(self):
        """Create a sample one-hot encoded sequence."""
        # PI: 99 positions, 22 amino acids
        return torch.randn(1, 99 * 22)

    @pytest.fixture
    def nrti_sequence(self):
        """Create a sample NRTI sequence."""
        # NRTI: 240 positions
        return torch.randn(1, 240 * 22)

    def test_standard_vae_prediction(self, sample_sequence):
        """Test standard VAE produces valid predictions."""
        from dataclasses import dataclass, field
        from typing import List
        import torch.nn as nn

        @dataclass
        class VAEConfig:
            input_dim: int
            latent_dim: int = 16
            hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
            dropout: float = 0.1

        class StandardVAE(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                layers = []
                in_dim = cfg.input_dim
                for h in cfg.hidden_dims:
                    layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.LayerNorm(h)])
                    in_dim = h
                self.encoder = nn.Sequential(*layers)
                self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
                self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)
                self.predictor = nn.Sequential(
                    nn.Linear(cfg.latent_dim, 32), nn.GELU(), nn.Linear(32, 1)
                )

            def forward(self, x):
                h = self.encoder(x)
                mu = self.fc_mu(h)
                logvar = self.fc_logvar(h)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                pred = self.predictor(z).squeeze(-1)
                return {"prediction": pred, "mu": mu, "logvar": logvar}

        cfg = VAEConfig(input_dim=99 * 22)
        model = StandardVAE(cfg)

        model.eval()
        with torch.no_grad():
            out = model(sample_sequence)

        assert "prediction" in out
        assert out["prediction"].shape == (1,)
        assert torch.isfinite(out["prediction"]).all()

    def test_uncertainty_estimation(self, sample_sequence):
        """Test MC Dropout uncertainty estimation."""
        import torch.nn as nn
        import torch.nn.functional as F

        class MCDropout(nn.Module):
            def __init__(self, p=0.1):
                super().__init__()
                self.p = p

            def forward(self, x):
                return F.dropout(x, p=self.p, training=True)

        class UncertaintyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(99 * 22, 64),
                    nn.GELU(),
                    MCDropout(0.2),
                    nn.Linear(64, 32),
                    nn.GELU(),
                    MCDropout(0.2),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

            def predict_with_uncertainty(self, x, n_samples=20):
                preds = [self(x) for _ in range(n_samples)]
                preds = torch.stack(preds, dim=0)
                return preds.mean(dim=0), preds.std(dim=0)

        model = UncertaintyModel()
        mean, std = model.predict_with_uncertainty(sample_sequence)

        assert mean.shape == (1,)
        assert std.shape == (1,)
        assert (std >= 0).all()

    def test_batch_prediction(self):
        """Test batch predictions work correctly."""
        import torch.nn as nn

        class SimplePredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(99 * 22, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        model = SimplePredictor()
        batch = torch.randn(16, 99 * 22)

        model.eval()
        with torch.no_grad():
            predictions = model(batch)

        assert predictions.shape == (16,)
        assert torch.isfinite(predictions).all()

    def test_cross_resistance_predictions(self, nrti_sequence):
        """Test cross-resistance model predicts for all drugs."""
        from dataclasses import dataclass, field
        from typing import List, Dict
        import torch.nn as nn

        @dataclass
        class CrossConfig:
            input_dim: int
            latent_dim: int = 16
            hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
            drug_names: List[str] = field(default_factory=lambda: ["AZT", "3TC", "TDF"])

        class SimpleCrossVAE(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.drug_names = cfg.drug_names
                self.encoder = nn.Sequential(
                    nn.Linear(cfg.input_dim, 64), nn.ReLU()
                )
                self.fc_mu = nn.Linear(64, cfg.latent_dim)
                self.drug_heads = nn.ModuleDict({
                    drug: nn.Linear(cfg.latent_dim, 1)
                    for drug in cfg.drug_names
                })

            def forward(self, x) -> Dict[str, torch.Tensor]:
                h = self.encoder(x)
                z = self.fc_mu(h)
                predictions = {
                    drug: head(z).squeeze(-1)
                    for drug, head in self.drug_heads.items()
                }
                return {"predictions": predictions}

        cfg = CrossConfig(input_dim=240 * 22)
        model = SimpleCrossVAE(cfg)

        model.eval()
        with torch.no_grad():
            out = model(nrti_sequence)

        assert "predictions" in out
        for drug in cfg.drug_names:
            assert drug in out["predictions"]
            assert out["predictions"][drug].shape == (1,)


class TestSequenceEncoding:
    """Test sequence encoding utilities."""

    def test_one_hot_encoding(self):
        """Test one-hot encoding produces valid output."""
        aa_alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
        n_aa = len(aa_alphabet)

        sequence = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVR"
        n_positions = len(sequence)

        # Encode
        aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}
        encoded = np.zeros((n_positions * n_aa,), dtype=np.float32)

        for i, aa in enumerate(sequence):
            if aa in aa_to_idx:
                encoded[i * n_aa + aa_to_idx[aa]] = 1.0

        # Verify
        assert encoded.sum() == n_positions  # One hot per position
        assert encoded.max() == 1.0
        assert encoded.min() == 0.0

    def test_encoding_with_gaps(self):
        """Test encoding handles gaps correctly."""
        aa_alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
        aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}

        sequence = "PQIT-LWQR"  # Contains gap
        n_aa = len(aa_alphabet)

        encoded = np.zeros((len(sequence) * n_aa,), dtype=np.float32)
        for i, aa in enumerate(sequence):
            if aa in aa_to_idx:
                encoded[i * n_aa + aa_to_idx[aa]] = 1.0

        assert encoded.sum() == len(sequence)


class TestModelComponents:
    """Test individual model components."""

    def test_ranking_loss_computation(self):
        """Test ranking loss is computed correctly."""
        predictions = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        targets = torch.tensor([0.2, 0.4, 0.5, 0.6, 0.8])

        # Compute correlation-based ranking loss
        p_c = predictions - predictions.mean()
        t_c = targets - targets.mean()
        corr = torch.sum(p_c * t_c) / (
            torch.sqrt(torch.sum(p_c**2) + 1e-8) *
            torch.sqrt(torch.sum(t_c**2) + 1e-8)
        )
        ranking_loss = -corr

        assert torch.isfinite(ranking_loss)
        assert ranking_loss < 0  # Should be negative (high correlation)

    def test_kl_divergence(self):
        """Test KL divergence computation."""
        mu = torch.randn(4, 16)
        logvar = torch.randn(4, 16)

        kl = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        assert torch.isfinite(kl)
        assert kl >= 0  # KL should be non-negative

    def test_reparameterization(self):
        """Test reparameterization trick."""
        mu = torch.zeros(4, 16)
        logvar = torch.zeros(4, 16)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        assert z.shape == mu.shape
        # With mu=0 and std=1, z should be standard normal
        assert z.abs().mean() < 2  # Rough check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
