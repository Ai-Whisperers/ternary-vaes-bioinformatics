"""Comprehensive Test Suite for All New Modules.

Tests for:
1. External validation framework
2. Subtype-specific models
3. Protein language model integration
4. Multi-pathogen framework
5. Clinical decision support
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))


# =============================================================================
# External Validation Tests
# =============================================================================

class TestExternalValidation:
    """Test external validation framework."""

    def test_stanford_adapter(self):
        """Test Stanford adapter creates correct structure."""
        from validation.external_validator import StanfordAdapter

        adapter = StanfordAdapter()
        assert adapter.name == "Stanford"

    def test_los_alamos_adapter(self):
        """Test Los Alamos adapter initialization."""
        from validation.external_validator import LosAlamosAdapter

        adapter = LosAlamosAdapter()
        assert adapter.name == "Los_Alamos"
        assert "Susceptible" in adapter.resistance_mapping

    def test_synthetic_data_creation(self):
        """Test synthetic data generation."""
        from validation.external_validator import create_synthetic_external_data

        dataset = create_synthetic_external_data(n_samples=100, n_positions=99)

        assert dataset.sequences.shape == (100, 99 * 22)
        assert len(dataset.resistance_scores) == 100
        assert dataset.subtypes is not None
        assert len(np.unique(dataset.subtypes)) > 1

    def test_sequence_encoding(self):
        """Test sequence encoding produces valid one-hot."""
        from validation.external_validator import StanfordAdapter

        adapter = StanfordAdapter()
        seq = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVR"
        encoded = adapter.encode_sequence(seq, n_positions=99)

        assert encoded.shape == (99 * 22,)
        assert encoded.max() == 1.0
        assert encoded.min() == 0.0

        # Check one-hot property (one 1 per position)
        reshaped = encoded.reshape(99, 22)
        assert np.allclose(reshaped[:len(seq)].sum(axis=1), 1.0)


# =============================================================================
# Subtype-Specific Model Tests
# =============================================================================

class TestSubtypeModels:
    """Test subtype-specific models."""

    @pytest.fixture
    def config(self):
        from models.subtype_specific import SubtypeConfig
        return SubtypeConfig(input_dim=99 * 22)

    def test_subtype_encoder(self):
        """Test subtype encoder produces embeddings."""
        from models.subtype_specific import SubtypeEncoder

        encoder = SubtypeEncoder(n_subtypes=7, embed_dim=16)
        idx = torch.tensor([0, 1, 2, 3])
        embed = encoder(idx)

        assert embed.shape == (4, 16)
        assert torch.isfinite(embed).all()

    def test_subtype_vae_forward(self, config):
        """Test SubtypeSpecificVAE forward pass."""
        from models.subtype_specific import SubtypeSpecificVAE

        model = SubtypeSpecificVAE(config)
        x = torch.randn(4, 99 * 22)

        out = model(x, subtype_name="C")

        assert "prediction" in out
        assert "subtype_embed" in out
        assert out["prediction"].shape == (4,)

    def test_multi_subtype_vae(self, config):
        """Test MultiSubtypeVAE predicts for all subtypes."""
        from models.subtype_specific import MultiSubtypeVAE

        model = MultiSubtypeVAE(config)
        x = torch.randn(4, 99 * 22)

        predictions = model.predict_all_subtypes(x)

        assert len(predictions) == 7  # 7 subtypes
        for subtype, pred in predictions.items():
            assert pred.shape == (4,)

    def test_transfer_learning(self, config):
        """Test transfer learning adapter."""
        from models.subtype_specific import SubtypeSpecificVAE, SubtypeTransferLearning

        base_model = SubtypeSpecificVAE(config)
        transfer = SubtypeTransferLearning(base_model)

        # Create adapter for subtype D
        transfer.create_adapter("D")
        transfer.freeze_encoder()

        x = torch.randn(4, 99 * 22)
        out = transfer(x, subtype="D", use_adapter=True)

        assert "prediction" in out
        assert "z_adapted" in out


# =============================================================================
# Protein Language Model Tests
# =============================================================================

class TestProteinLM:
    """Test protein language model integration."""

    @pytest.fixture
    def config(self):
        from models.protein_lm_integration import PLMConfig
        return PLMConfig()

    def test_mock_esm2(self):
        """Test mock ESM-2 model."""
        from models.protein_lm_integration import MockESM2

        model = MockESM2(embed_dim=320, num_layers=6)
        tokens = torch.randint(4, 24, (4, 100))

        out = model(tokens, repr_layers=[6])

        assert "representations" in out
        assert 6 in out["representations"]
        assert out["representations"][6].shape == (4, 100, 320)

    def test_plm_vae(self, config):
        """Test ProteinLMVAE forward pass."""
        from models.protein_lm_integration import ProteinLMVAE

        model = ProteinLMVAE(config)
        tokens = torch.randint(4, 24, (4, 100))

        out = model(tokens=tokens)

        assert "prediction" in out
        assert "z" in out
        assert out["prediction"].shape == (4,)
        assert out["z"].shape == (4, config.latent_dim)

    def test_tokenization(self):
        """Test sequence tokenization."""
        from models.protein_lm_integration import tokenize_sequence

        seq = "ACDEFGHIKLMNPQRSTVWY"
        tokens = tokenize_sequence(seq, max_len=32)

        assert tokens.shape == (32,)
        assert tokens[0] == 0  # CLS
        assert tokens[len(seq) + 1] == 2  # EOS
        assert (tokens[len(seq) + 2:] == 1).all()  # PAD

    def test_hybrid_vae(self, config):
        """Test hybrid VAE with both inputs."""
        from models.protein_lm_integration import HybridVAE

        model = HybridVAE(config, input_dim=99 * 22)

        x_onehot = torch.randn(4, 99 * 22)
        tokens = torch.randint(4, 24, (4, 101))

        out = model(x_onehot, tokens)

        assert "prediction" in out
        assert out["prediction"].shape == (4,)


# =============================================================================
# Multi-Pathogen Framework Tests
# =============================================================================

class TestPathogenExtension:
    """Test multi-pathogen framework."""

    def test_hcv_model(self):
        """Test HCV model for NS5A."""
        from models.pathogen_extension import UniversalDrugResistanceVAE

        model = UniversalDrugResistanceVAE(
            pathogen="HCV",
            gene="NS5A",
            input_dim=447 * 22,
        )

        x = torch.randn(4, 447 * 22)
        out = model(x)

        assert "predictions" in out
        assert len(out["predictions"]) > 0  # Has NS5A inhibitor predictions

    def test_hbv_model(self):
        """Test HBV model for Polymerase."""
        from models.pathogen_extension import UniversalDrugResistanceVAE

        model = UniversalDrugResistanceVAE(
            pathogen="HBV",
            gene="Pol",
            input_dim=845 * 22,
        )

        x = torch.randn(4, 845 * 22)
        out = model(x)

        assert "predictions" in out
        # Check HBV drugs
        expected_drugs = ["LAM", "ADV", "ETV", "TDF", "TAF"]
        for drug in expected_drugs:
            assert drug in out["predictions"]

    def test_tb_model(self):
        """Test TB model for rpoB."""
        from models.pathogen_extension import UniversalDrugResistanceVAE

        model = UniversalDrugResistanceVAE(
            pathogen="TB",
            gene="rpoB",
            input_dim=1172 * 22,
        )

        x = torch.randn(4, 1172 * 22)
        out = model(x)

        assert "predictions" in out
        # Check rifampicin drugs
        assert "RIF" in out["predictions"]

    def test_pathogen_encoder_mutations(self):
        """Test pathogen encoder returns known mutations."""
        from models.pathogen_extension import HCVEncoder

        encoder = HCVEncoder("NS5A", input_dim=447 * 22)
        mutations = encoder.get_mutation_positions()

        # NS5A should have known resistance positions
        assert len(mutations) > 0
        assert 30 in mutations  # Y93H position


# =============================================================================
# Clinical Decision Support Tests
# =============================================================================

class TestClinicalDecisionSupport:
    """Test clinical decision support module."""

    @pytest.fixture
    def sample_predictions(self):
        return {
            "LPV": 0.15,
            "DRV": 0.10,
            "TDF": 0.25,
            "ABC": 0.80,
            "3TC": 0.85,
            "DTG": 0.10,
            "BIC": 0.12,
        }

    def test_resistance_interpreter(self):
        """Test resistance level interpretation."""
        from clinical.decision_support import ResistanceInterpreter, ResistanceLevel

        interpreter = ResistanceInterpreter()

        assert interpreter.interpret(0.1) == ResistanceLevel.SUSCEPTIBLE
        assert interpreter.interpret(0.4) == ResistanceLevel.LOW
        assert interpreter.interpret(0.6) == ResistanceLevel.INTERMEDIATE
        assert interpreter.interpret(0.9) == ResistanceLevel.HIGH

    def test_report_generation(self, sample_predictions):
        """Test clinical report generation."""
        from clinical.decision_support import ClinicalDecisionSupport

        cds = ClinicalDecisionSupport()
        report = cds.generate_report(sample_predictions, patient_id="TEST-001")

        assert report.patient_id == "TEST-001"
        assert len(report.drug_results) > 0
        assert len(report.recommendations) > 0
        assert report.overall_resistance_profile is not None

    def test_cross_resistance_alerts(self, sample_predictions):
        """Test cross-resistance detection."""
        from clinical.decision_support import ClinicalDecisionSupport

        # Add high resistance to cross-resistance pair
        predictions = sample_predictions.copy()
        predictions["AZT"] = 0.8
        predictions["D4T"] = 0.85

        cds = ClinicalDecisionSupport()
        report = cds.generate_report(predictions)

        # Should have cross-resistance alert
        alert_categories = [a.category for a in report.alerts]
        assert "cross-resistance" in alert_categories

    def test_regimen_selection(self, sample_predictions):
        """Test regimen selection."""
        from clinical.decision_support import ClinicalDecisionSupport

        cds = ClinicalDecisionSupport()
        report = cds.generate_report(sample_predictions)

        # Should suggest regimens
        assert len(report.suggested_regimens) > 0

    def test_report_formatting(self, sample_predictions):
        """Test report text formatting."""
        from clinical.decision_support import ClinicalDecisionSupport

        cds = ClinicalDecisionSupport()
        report = cds.generate_report(sample_predictions)
        formatted = cds.format_report(report)

        assert "CLINICAL DECISION SUPPORT REPORT" in formatted
        assert "DISCLAIMER" in formatted
        assert "SUGGESTED REGIMENS" in formatted


# =============================================================================
# Integration Tests
# =============================================================================

class TestModuleIntegration:
    """Test integration between modules."""

    def test_plm_to_clinical(self):
        """Test PLM predictions flow to clinical decision support."""
        from models.protein_lm_integration import ProteinLMVAE, PLMConfig
        from clinical.decision_support import ClinicalDecisionSupport

        # Generate predictions with PLM
        config = PLMConfig()
        model = ProteinLMVAE(config)
        tokens = torch.randint(4, 24, (1, 100))

        with torch.no_grad():
            out = model(tokens=tokens)

        # Mock predictions for multiple drugs
        predictions = {
            "DTG": float(torch.sigmoid(out["prediction"]).item()),
            "BIC": float(torch.sigmoid(out["prediction"]).item()) + 0.1,
            "TDF": 0.2,
            "3TC": 0.15,
        }

        # Generate clinical report
        cds = ClinicalDecisionSupport()
        report = cds.generate_report(predictions)

        assert report is not None
        assert len(report.drug_results) > 0

    def test_subtype_to_validation(self):
        """Test subtype model outputs work with validation."""
        from models.subtype_specific import SubtypeSpecificVAE, SubtypeConfig
        from validation.external_validator import create_synthetic_external_data

        # Create model
        config = SubtypeConfig(input_dim=99 * 22)
        model = SubtypeSpecificVAE(config)

        # Create synthetic data
        dataset = create_synthetic_external_data(n_samples=10, n_positions=99)

        # Get predictions
        x = torch.tensor(dataset.sequences)
        with torch.no_grad():
            out = model(x, subtype_name="C")

        predictions = out["prediction"].numpy()

        # Validate outputs are reasonable
        assert len(predictions) == 10
        assert np.isfinite(predictions).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
