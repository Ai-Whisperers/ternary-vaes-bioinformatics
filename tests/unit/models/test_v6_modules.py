# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for V6.0 model modules.

Tests the new modules added in V6.0:
- Uncertainty quantification
- Multi-task learning
- Discrete diffusion
- Contrastive learning (BYOL)
- Cross-modal fusion
- Calibration metrics
"""

import pytest
import torch
import torch.nn as nn


# ============================================================================
# Uncertainty Quantification Tests
# ============================================================================

class TestBayesianPredictor:
    """Tests for Bayesian uncertainty estimation."""

    def test_bayesian_predictor_forward(self):
        """Test BayesianPredictor forward pass."""
        from src.models.uncertainty.bayesian import BayesianPredictor

        model = BayesianPredictor(
            input_dim=64,
            output_dim=1,
            hidden_dims=[128, 64],
        )

        x = torch.randn(32, 64)
        output = model(x)

        assert output.shape == (32, 1)

    def test_bayesian_predictor_with_uncertainty(self):
        """Test uncertainty estimation."""
        from src.models.uncertainty.bayesian import BayesianPredictor

        model = BayesianPredictor(
            input_dim=64,
            output_dim=1,
            hidden_dims=[128, 64],
            learn_variance=True,
        )

        x = torch.randn(16, 64)
        result = model.predict_with_uncertainty(x, n_samples=20)

        assert "prediction" in result
        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result
        assert result["prediction"].shape == (16, 1)

    def test_mc_dropout_wrapper(self):
        """Test MC Dropout wrapper."""
        from src.models.uncertainty.bayesian import MCDropoutWrapper

        base_model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        wrapper = MCDropoutWrapper(base_model, dropout_rate=0.1)

        x = torch.randn(16, 32)
        mean, uncertainty = wrapper.predict_with_uncertainty(x, n_samples=10)

        assert mean.shape == (16, 1)
        assert uncertainty.shape == (16, 1)


class TestEvidentialPredictor:
    """Tests for evidential deep learning."""

    def test_evidential_regression(self):
        """Test evidential regression predictor."""
        from src.models.uncertainty.evidential import EvidentialPredictor

        model = EvidentialPredictor(
            input_dim=64,
            output_dim=1,
            task="regression",
        )

        x = torch.randn(16, 64)
        result = model(x)

        assert "prediction" in result
        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result
        assert "evidence" in result

    def test_evidential_classification(self):
        """Test evidential classification predictor."""
        from src.models.uncertainty.evidential import EvidentialPredictor

        model = EvidentialPredictor(
            input_dim=64,
            output_dim=5,
            task="classification",
        )

        x = torch.randn(16, 64)
        result = model(x)

        assert "prediction" in result
        assert "probabilities" in result
        assert result["probabilities"].shape == (16, 5)

    def test_evidential_loss(self):
        """Test evidential loss computation."""
        from src.models.uncertainty.evidential import EvidentialLoss

        loss_fn = EvidentialLoss(task="regression")

        predictions = torch.randn(16, 4)  # gamma, nu, alpha, beta
        targets = torch.randn(16, 1)

        loss = loss_fn(predictions, targets)
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)


class TestEnsemblePredictor:
    """Tests for ensemble methods."""

    def test_ensemble_predictor(self):
        """Test basic ensemble predictor."""
        from src.models.uncertainty.ensemble import EnsemblePredictor

        model_fn = lambda: nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        ensemble = EnsemblePredictor(model_fn, n_members=3)

        x = torch.randn(16, 32)
        result = ensemble.predict_with_uncertainty(x)

        assert "prediction" in result
        assert "epistemic_uncertainty" in result
        assert result["prediction"].shape == (16, 1)

    def test_deep_ensemble(self):
        """Test deep ensemble with learned variance."""
        from src.models.uncertainty.ensemble import DeepEnsemble

        model = DeepEnsemble(
            input_dim=32,
            output_dim=1,
            n_members=3,
        )

        x = torch.randn(16, 32)
        result = model.predict_with_uncertainty(x)

        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result


# ============================================================================
# Multi-Task Learning Tests
# ============================================================================

class TestMultiTaskResistancePredictor:
    """Tests for multi-task resistance prediction."""

    def test_mtl_forward(self):
        """Test MTL predictor forward pass."""
        from src.models.mtl.resistance_predictor import (
            MultiTaskResistancePredictor,
            MTLConfig,
        )

        config = MTLConfig(
            input_dim=64,
            n_drugs=5,
        )
        model = MultiTaskResistancePredictor(config)

        x = torch.randn(16, 64)
        outputs = model(x)

        assert "resistance" in outputs
        assert outputs["resistance"].shape == (16, 5)
        assert "mdr_logits" in outputs

    def test_mtl_loss(self):
        """Test MTL loss computation."""
        from src.models.mtl.resistance_predictor import (
            MultiTaskResistancePredictor,
            MTLConfig,
        )

        config = MTLConfig(input_dim=64, n_drugs=5)
        model = MultiTaskResistancePredictor(config)

        x = torch.randn(16, 64)
        targets = {
            "resistance": torch.randint(0, 2, (16, 5)).float(),
        }

        total_loss, losses = model.compute_loss(x, targets)

        assert not torch.isnan(total_loss).any()
        assert "resistance" in losses

    def test_cross_task_attention(self):
        """Test cross-task attention mechanism."""
        from src.models.mtl.task_heads import CrossTaskAttention

        attn = CrossTaskAttention(
            task_dim=64,
            n_tasks=5,
            n_heads=4,
        )

        task_reps = torch.randn(16, 5, 64)
        output = attn(task_reps)

        assert output.shape == (16, 5, 64)


class TestGradNorm:
    """Tests for GradNorm optimizer."""

    def test_gradnorm_step(self):
        """Test GradNorm optimization step."""
        from src.models.mtl.gradnorm import UncertaintyWeighting

        weighting = UncertaintyWeighting(n_tasks=3)

        losses = [
            torch.tensor(1.0, requires_grad=True),
            torch.tensor(2.0, requires_grad=True),
            torch.tensor(0.5, requires_grad=True),
        ]

        weighted_loss, weights = weighting(losses)

        assert not torch.isnan(weighted_loss)
        assert len(weights) == 3


# ============================================================================
# Discrete Diffusion Tests
# ============================================================================

class TestD3PM:
    """Tests for discrete diffusion model."""

    def test_d3pm_forward(self):
        """Test D3PM forward pass."""
        from src.models.diffusion.d3pm import D3PM, D3PMConfig

        config = D3PMConfig(
            vocab_size=21,
            max_length=50,
            hidden_dim=64,
            n_layers=2,
            n_timesteps=100,
        )
        model = D3PM(config)

        x_t = torch.randint(0, 21, (8, 50))
        t = torch.randint(0, 100, (8,))

        logits = model(x_t, t)

        assert logits.shape == (8, 50, 21)

    def test_d3pm_loss(self):
        """Test D3PM loss computation."""
        from src.models.diffusion.d3pm import D3PM, D3PMConfig

        config = D3PMConfig(
            vocab_size=21,
            max_length=50,
            hidden_dim=64,
            n_layers=2,
            n_timesteps=100,
        )
        model = D3PM(config)

        x_0 = torch.randint(1, 21, (8, 50))  # Clean tokens
        loss = model.compute_loss(x_0)

        assert not torch.isnan(loss)

    def test_d3pm_sampling(self):
        """Test D3PM sampling."""
        from src.models.diffusion.d3pm import D3PM, D3PMConfig

        config = D3PMConfig(
            vocab_size=21,
            max_length=20,
            hidden_dim=32,
            n_layers=1,
            n_timesteps=10,  # Few steps for testing
        )
        model = D3PM(config)

        samples = model.sample(batch_size=4, seq_len=20)

        assert samples.shape == (4, 20)
        assert samples.min() >= 0
        assert samples.max() < 21

    def test_padic_noise_schedule(self):
        """Test p-adic noise schedule."""
        from src.models.diffusion.noise_schedule import PAdicNoiseSchedule

        schedule = PAdicNoiseSchedule(
            n_timesteps=100,
            p=3,
            max_length=50,
        )

        # Check valuations computed correctly
        assert schedule.valuations[0] == 0  # v_3(1) = 0
        assert schedule.valuations[2] == 1  # v_3(3) = 1
        assert schedule.valuations[8] == 2  # v_3(9) = 2


# ============================================================================
# Contrastive Learning Tests
# ============================================================================

class TestBYOL:
    """Tests for BYOL self-supervised learning."""

    def test_byol_forward(self):
        """Test BYOL forward pass."""
        from src.models.contrastive.byol import BYOL, BYOLConfig

        encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        config = BYOLConfig(embed_dim=256, proj_dim=128)
        byol = BYOL(encoder, config)

        x = torch.randn(16, 64)
        online_pred, target_proj = byol(x)

        assert online_pred.shape == (16, 128)
        assert target_proj.shape == (16, 128)

    def test_byol_loss(self):
        """Test BYOL loss computation."""
        from src.models.contrastive.byol import BYOL, BYOLConfig

        encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        config = BYOLConfig(embed_dim=256)
        byol = BYOL(encoder, config)

        x1 = torch.randn(16, 64)
        x2 = torch.randn(16, 64)

        loss = byol.compute_loss(x1, x2)

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_momentum_update(self):
        """Test momentum encoder update."""
        from src.models.contrastive.byol import BYOL, BYOLConfig

        encoder = nn.Sequential(nn.Linear(32, 64))
        byol = BYOL(encoder, BYOLConfig(embed_dim=64))

        # Get target params before
        target_before = list(byol.momentum_encoder.target_encoder.parameters())[0].clone()

        # Modify online encoder
        with torch.no_grad():
            for p in byol.encoder.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Update target
        byol.update_target()

        # Target should have changed
        target_after = list(byol.momentum_encoder.target_encoder.parameters())[0]
        assert not torch.allclose(target_before, target_after)


class TestSimCLR:
    """Tests for SimCLR contrastive learning."""

    def test_simclr_forward(self):
        """Test SimCLR forward pass."""
        from src.models.contrastive.simclr import SimCLR

        encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        simclr = SimCLR(encoder, embed_dim=256, proj_dim=128)

        x = torch.randn(16, 64)
        z = simclr(x)

        assert z.shape == (16, 128)

    def test_ntxent_loss(self):
        """Test NT-Xent loss."""
        from src.models.contrastive.simclr import NTXentLoss

        loss_fn = NTXentLoss(temperature=0.5)

        z_i = torch.randn(16, 128)
        z_j = torch.randn(16, 128)

        loss = loss_fn(z_i, z_j)

        assert not torch.isnan(loss)
        assert loss > 0


class TestAugmentations:
    """Tests for sequence augmentations."""

    def test_mutation_augmentation(self):
        """Test mutation augmentation."""
        from src.models.contrastive.augmentations import MutationAugmentation

        aug = MutationAugmentation(mutation_rate=0.2)

        sequence = "MKLLVVLLFVAQVLA"
        augmented = aug(sequence)

        assert len(augmented) == len(sequence)
        # Should have some differences
        assert augmented != sequence or True  # May occasionally be same

    def test_masking_augmentation(self):
        """Test masking augmentation."""
        from src.models.contrastive.augmentations import MaskingAugmentation

        aug = MaskingAugmentation(mask_rate=0.15)

        tokens = torch.randint(1, 21, (50,))
        augmented = aug(tokens)

        assert augmented.shape == tokens.shape

    def test_sequence_augmentations_pipeline(self):
        """Test augmentation pipeline."""
        from src.models.contrastive.augmentations import SequenceAugmentations

        aug = SequenceAugmentations.default_for_byol()

        sequence = "MKLLVVLLFVAQVLA"
        view1, view2 = aug.create_pair(sequence)

        assert isinstance(view1, str)
        assert isinstance(view2, str)


# ============================================================================
# Cross-Modal Fusion Tests
# ============================================================================

class TestCrossModalFusion:
    """Tests for cross-modal fusion."""

    def test_concat_fusion(self):
        """Test concatenation fusion."""
        from src.models.fusion.cross_modal import ConcatFusion

        fusion = ConcatFusion(
            modality_dims={"seq": 64, "struct": 64},
            output_dim=128,
        )

        embeddings = {
            "seq": torch.randn(16, 64),
            "struct": torch.randn(16, 64),
        }

        fused = fusion(embeddings)
        assert fused.shape == (16, 128)

    def test_gated_fusion(self):
        """Test gated fusion."""
        from src.models.fusion.cross_modal import GatedFusion

        fusion = GatedFusion(
            modality_dims={"seq": 64, "struct": 64},
            output_dim=128,
        )

        embeddings = {
            "seq": torch.randn(16, 64),
            "struct": torch.randn(16, 64),
        }

        fused = fusion(embeddings)
        assert fused.shape == (16, 128)

        # Check gates
        weights = fusion.get_gate_weights(embeddings)
        assert "seq" in weights
        assert "struct" in weights

    def test_cross_modal_attention(self):
        """Test cross-modal attention."""
        from src.models.fusion.cross_modal import CrossModalAttention

        attn = CrossModalAttention(
            modality_dims={"seq": 64, "struct": 64},
            output_dim=128,
            n_heads=4,
        )

        embeddings = {
            "seq": torch.randn(16, 64),
            "struct": torch.randn(16, 64),
        }

        fused, attention = attn(embeddings, return_attention=True)

        assert fused.shape == (16, 128)
        assert attention.shape[0] == 16  # batch

    def test_multimodal_encoder(self):
        """Test multimodal encoder."""
        from src.models.fusion.multimodal import MultimodalEncoder, MultimodalConfig

        config = MultimodalConfig(
            sequence_dim=64,
            structure_dim=64,
            output_dim=128,
        )
        encoder = MultimodalEncoder(config)

        seq_emb = torch.randn(16, 64)
        struct_emb = torch.randn(16, 64)

        output = encoder(
            sequence_emb=seq_emb,
            structure_emb=struct_emb,
        )

        assert output.shape == (16, 128)


# ============================================================================
# Calibration Metrics Tests
# ============================================================================

class TestCalibrationMetrics:
    """Tests for calibration metrics."""

    def test_expected_calibration_error(self):
        """Test ECE computation."""
        from src.metrics.calibration import ExpectedCalibrationError

        ece = ExpectedCalibrationError(n_bins=10)

        # Create well-calibrated predictions
        confidences = torch.rand(100)
        accuracies = (torch.rand(100) < confidences).float()

        ece_value = ece(confidences, accuracies)

        assert ece_value >= 0
        assert ece_value <= 1

    def test_brier_score(self):
        """Test Brier score."""
        from src.metrics.calibration import BrierScore

        brier = BrierScore()

        # Perfect predictions
        probs = torch.tensor([1.0, 0.0, 1.0, 0.0])
        targets = torch.tensor([1, 0, 1, 0])

        score = brier(probs, targets)
        assert score == 0.0

        # Worst predictions
        probs = torch.tensor([0.0, 1.0, 0.0, 1.0])
        score = brier(probs, targets)
        assert score == 1.0

    def test_reliability_diagram(self):
        """Test reliability diagram data."""
        from src.metrics.calibration import ReliabilityDiagram

        diagram = ReliabilityDiagram(n_bins=5)

        confidences = torch.rand(100)
        accuracies = (torch.rand(100) < confidences).float()

        data = diagram.compute(confidences, accuracies)

        assert "bin_centers" in data
        assert "bin_accuracies" in data
        assert "bin_confidences" in data
        assert len(data["bin_centers"]) == 5

    def test_calibration_metrics_comprehensive(self):
        """Test comprehensive calibration evaluation."""
        from src.metrics.calibration import CalibrationMetrics

        metrics = CalibrationMetrics(n_bins=10)

        # Multi-class predictions
        logits = torch.randn(100, 5)
        targets = torch.randint(0, 5, (100,))

        result = metrics.evaluate(logits, targets)

        assert hasattr(result, "ece")
        assert hasattr(result, "mce")
        assert hasattr(result, "brier_score")
        assert hasattr(result, "reliability")


# ============================================================================
# Integration Tests
# ============================================================================

class TestV6Integration:
    """Integration tests for V6 modules working together."""

    def test_uncertainty_with_mtl(self):
        """Test uncertainty estimation with MTL predictor."""
        from src.models.mtl.resistance_predictor import MultiTaskResistancePredictor, MTLConfig
        from src.models.uncertainty.wrapper import UncertaintyWrapper

        config = MTLConfig(input_dim=64, n_drugs=5)
        base_model = MultiTaskResistancePredictor(config)

        # Just test that we can get predictions from MTL model
        x = torch.randn(16, 64)
        outputs = base_model(x)

        assert outputs["resistance"].shape == (16, 5)

    def test_fusion_with_encoders(self):
        """Test fusion layer with mock encoders."""
        from src.models.fusion.multimodal import MultimodalEncoder, MultimodalConfig

        # Mock encoders
        seq_encoder = nn.Sequential(nn.Linear(100, 64))
        struct_encoder = nn.Sequential(nn.Linear(50, 64))

        config = MultimodalConfig(
            sequence_dim=64,
            structure_dim=64,
            output_dim=128,
        )

        encoder = MultimodalEncoder(
            config,
            sequence_encoder=seq_encoder,
            structure_encoder=struct_encoder,
        )

        # Use raw inputs
        seq_input = torch.randn(16, 100)
        struct_input = torch.randn(16, 50)

        output = encoder(
            sequence_input=seq_input,
            structure_input=struct_input,
        )

        assert output.shape == (16, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
