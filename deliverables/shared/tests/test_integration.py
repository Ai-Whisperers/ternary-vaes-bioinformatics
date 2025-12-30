# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Integration tests for bioinformatics tool pipelines.

Tests cover complete workflows for each research partner's tools,
ensuring all components work together correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add deliverables to path
deliverables_dir = Path(__file__).parent.parent.parent
project_root = deliverables_dir.parent
sys.path.insert(0, str(deliverables_dir))
sys.path.insert(0, str(project_root))


class TestHemolysisPredictor:
    """Integration tests for hemolysis prediction pipeline."""

    def test_full_prediction_workflow(self):
        """Test complete hemolysis prediction workflow."""
        from shared import HemolysisPredictor, compute_peptide_properties

        # Initialize predictor
        predictor = HemolysisPredictor()

        # Test sequence
        peptide = "GIGKFLHSAKKFGKAFVGEIMNS"  # Magainin 2

        # Get properties
        props = compute_peptide_properties(peptide)
        assert props["length"] == 23
        assert props["net_charge"] > 0

        # Predict hemolysis
        result = predictor.predict(peptide)
        assert "hc50_predicted" in result
        assert "risk_category" in result
        assert "hemolytic_probability" in result

        # HC50 should be positive
        assert result["hc50_predicted"] > 0

        # Risk should be valid category
        assert result["risk_category"] in ["Low", "Moderate", "High"]

        # Probability should be between 0 and 1
        assert 0 <= result["hemolytic_probability"] <= 1

    def test_therapeutic_index_calculation(self):
        """Test therapeutic index calculation."""
        from shared import HemolysisPredictor

        predictor = HemolysisPredictor()
        peptide = "KLWKKWKKWLK"

        ti_result = predictor.compute_therapeutic_index(peptide, mic_value=10.0)

        assert "hc50" in ti_result
        assert "mic" in ti_result
        assert "therapeutic_index" in ti_result
        assert "interpretation" in ti_result

        # TI should be HC50/MIC
        expected_ti = ti_result["hc50"] / ti_result["mic"]
        assert ti_result["therapeutic_index"] == pytest.approx(expected_ti, rel=0.01)

    def test_batch_prediction(self):
        """Test predicting hemolysis for multiple peptides."""
        from shared import HemolysisPredictor

        predictor = HemolysisPredictor()
        peptides = [
            "GIGKFLHSAKKFGKAFVGEIMNS",
            "KLWKKWKKWLK",
            "RRWWRRWWRR",
        ]

        results = [predictor.predict(p) for p in peptides]

        assert len(results) == 3
        for result in results:
            assert result["hc50_predicted"] > 0


class TestPrimerDesigner:
    """Integration tests for primer design pipeline."""

    def test_primer_design_workflow(self):
        """Test complete primer design workflow."""
        from shared import PrimerDesigner

        designer = PrimerDesigner()

        # Design primers for a peptide
        peptide = "KLWKKWKKWLK"
        primers = designer.design_for_peptide(
            peptide,
            codon_optimization="ecoli",
            add_start_codon=True,
            add_stop_codon=True,
        )

        # Check primer properties
        assert hasattr(primers, "forward")
        assert hasattr(primers, "reverse")
        assert hasattr(primers, "forward_tm")
        assert hasattr(primers, "reverse_tm")
        assert hasattr(primers, "product_size")

        # Primers should be DNA sequences
        valid_dna = set("ATCG")
        assert all(c in valid_dna for c in primers.forward)
        assert all(c in valid_dna for c in primers.reverse)

        # Tm should be in reasonable range (allowing for short sequences)
        assert 40 <= primers.forward_tm <= 80
        assert 40 <= primers.reverse_tm <= 80

    def test_codon_optimization(self):
        """Test codon optimization for different organisms."""
        from shared import PrimerDesigner

        designer = PrimerDesigner()
        peptide = "MKLW"

        for organism in ["ecoli", "yeast", "mammalian"]:
            dna = designer.peptide_to_dna(peptide, codon_optimization=organism)

            # DNA should be 3x peptide length
            assert len(dna) == len(peptide) * 3

            # Should be valid DNA
            valid_dna = set("ATCG")
            assert all(c in valid_dna for c in dna)


class TestUncertaintyQuantification:
    """Integration tests for uncertainty quantification."""

    def test_mc_dropout_sampling(self):
        """Test Monte Carlo dropout sampling."""
        try:
            from shared import UncertaintyQuantifier
        except ImportError:
            pytest.skip("UncertaintyQuantifier not exported from shared")

        uq = UncertaintyQuantifier(method="mc_dropout", n_samples=10)

        # Simulate model output (list of predictions)
        predictions = [np.random.randn(10) for _ in range(10)]
        mean, std = uq.aggregate_predictions(predictions)

        assert mean.shape == (10,)
        assert std.shape == (10,)
        assert np.all(std >= 0)

    def test_ensemble_sampling(self):
        """Test ensemble-based uncertainty."""
        try:
            from shared import UncertaintyQuantifier
        except ImportError:
            pytest.skip("UncertaintyQuantifier not exported from shared")

        uq = UncertaintyQuantifier(method="ensemble", n_samples=5)

        predictions = [np.random.randn(10) for _ in range(5)]
        mean, std = uq.aggregate_predictions(predictions)

        assert mean.shape == (10,)
        assert std.shape == (10,)

    def test_calibration_analysis(self):
        """Test uncertainty calibration."""
        try:
            from shared import UncertaintyQuantifier
        except ImportError:
            pytest.skip("UncertaintyQuantifier not exported from shared")

        uq = UncertaintyQuantifier()

        # Simulated predictions and actuals
        predictions = np.random.randn(100)
        uncertainties = np.abs(np.random.randn(100)) * 0.5
        actuals = predictions + np.random.randn(100) * 0.5

        calibration = uq.analyze_calibration(predictions, uncertainties, actuals)

        assert "coverage" in calibration
        assert "expected_coverage" in calibration
        assert "calibration_error" in calibration


class TestAMPDesignPipeline:
    """Integration tests for AMP design pipeline."""

    def test_amp_property_scoring(self):
        """Test AMP property-based scoring."""
        from shared import compute_peptide_properties, validate_sequence

        peptides = [
            "GIGKFLHSAKKFGKAFVGEIMNS",  # Magainin 2
            "KLWKKWKKWLK",  # Synthetic cationic
            "RRWWRRWWRR",  # Arg-Trp rich
        ]

        for peptide in peptides:
            # Validate
            is_valid, _ = validate_sequence(peptide)
            assert is_valid

            # Compute properties
            props = compute_peptide_properties(peptide)

            # AMPs should be cationic
            assert props["net_charge"] > 0

            # Should have some hydrophobic content
            assert props["hydrophobic_ratio"] > 0

    def test_ml_feature_generation(self):
        """Test ML feature generation for activity prediction."""
        from shared import compute_ml_features

        peptide = "GIGKFLHSAKKFGKAFVGEIMNS"
        features = compute_ml_features(peptide)

        # Should have 25 features (5 props + 20 AA composition)
        assert len(features) == 25

        # No NaN or Inf values
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))


class TestDatabaseCurators:
    """Integration tests for database curator modules."""

    def test_dramp_curator_structure(self):
        """Test DRAMP curator module structure."""
        try:
            from databases import DRAMPCurator

            curator = DRAMPCurator(cache_dir=None)  # Don't download

            # Check interface
            assert hasattr(curator, "get_sequences")
            assert hasattr(curator, "filter_by_activity")
            assert hasattr(curator, "get_pathogen_specific")
        except ImportError:
            pytest.skip("DRAMP curator not available")

    def test_protherm_curator_structure(self):
        """Test ProTherm curator module structure."""
        try:
            from databases import ProThermCurator

            curator = ProThermCurator(cache_dir=None)

            assert hasattr(curator, "get_mutations")
            assert hasattr(curator, "filter_by_protein")
        except ImportError:
            pytest.skip("ProTherm curator not available")


class TestCodonEncoder:
    """Integration tests for codon encoder."""

    def test_encoder_initialization(self):
        """Test codon encoder initializes correctly."""
        try:
            from shared import CodonEncoder

            encoder = CodonEncoder()

            assert hasattr(encoder, "encode_sequence")
            assert hasattr(encoder, "latent_dim")
        except ImportError:
            pytest.skip("CodonEncoder not available")

    def test_sequence_encoding(self):
        """Test encoding peptide sequences."""
        try:
            from shared import CodonEncoder

            encoder = CodonEncoder()
            peptide = "GIGKFLHSAKKFGKAFVGEIMNS"

            embedding = encoder.encode_sequence(peptide)

            # Check shape
            assert embedding.shape == (encoder.latent_dim,)

            # Should be finite
            assert np.all(np.isfinite(embedding))
        except ImportError:
            pytest.skip("CodonEncoder not available")


class TestEndToEndPipelines:
    """End-to-end integration tests for complete workflows."""

    def test_amp_design_to_synthesis(self):
        """Test complete AMP design to synthesis optimization workflow."""
        from shared import (
            compute_peptide_properties,
            HemolysisPredictor,
            PrimerDesigner,
            validate_sequence,
        )

        # 1. Start with candidate peptide
        candidate = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"

        # 2. Validate sequence
        is_valid, _ = validate_sequence(candidate)
        assert is_valid

        # 3. Compute biophysical properties
        props = compute_peptide_properties(candidate)
        assert props["net_charge"] > 0  # Should be cationic

        # 4. Predict hemolysis (safety check)
        predictor = HemolysisPredictor()
        hemo = predictor.predict(candidate)
        assert hemo["hc50_predicted"] > 0

        # 5. Calculate therapeutic index
        ti = predictor.compute_therapeutic_index(candidate, mic_value=5.0)
        assert ti["therapeutic_index"] > 0

        # 6. Design primers for synthesis
        designer = PrimerDesigner()
        primers = designer.design_for_peptide(
            candidate, codon_optimization="ecoli", add_start_codon=True, add_stop_codon=True
        )
        assert len(primers.forward) > 15
        assert len(primers.reverse) > 15

    def test_peptide_comparison_workflow(self):
        """Test comparing multiple peptide candidates."""
        from shared import compute_peptide_properties, HemolysisPredictor

        # Set of candidates
        candidates = {
            "Candidate_A": "KLWKKWKKWLK",
            "Candidate_B": "GIGKFLHSAKKFGKAFVGEIMNS",
            "Candidate_C": "RRWWRRWWRR",
        }

        predictor = HemolysisPredictor()
        results = {}

        for name, seq in candidates.items():
            props = compute_peptide_properties(seq)
            hemo = predictor.predict(seq)

            results[name] = {
                "length": props["length"],
                "charge": props["net_charge"],
                "hydrophobicity": props["hydrophobicity"],
                "hc50": hemo["hc50_predicted"],
                "risk": hemo["risk_category"],
            }

        # All candidates should have been processed
        assert len(results) == 3

        # Each should have all required fields
        for name, data in results.items():
            assert "length" in data
            assert "charge" in data
            assert "hc50" in data


class TestBiotoolsCLI:
    """Integration tests for biotools CLI."""

    def test_tools_dictionary_valid(self):
        """Test that all tools in TOOLS dict have valid structure."""
        sys.path.insert(0, str(deliverables_dir / "scripts"))

        from biotools import TOOLS

        required_keys = {"module", "description", "partner", "flags", "demo_args"}

        for tool_name, tool_info in TOOLS.items():
            assert isinstance(tool_name, str)
            for key in required_keys:
                assert key in tool_info, f"Tool {tool_name} missing key {key}"

    def test_list_tools_function(self):
        """Test list_tools function runs without error."""
        sys.path.insert(0, str(deliverables_dir / "scripts"))

        from biotools import list_tools

        # Should not raise
        list_tools()

    def test_analyze_peptide_function(self):
        """Test analyze_peptide function runs without error."""
        sys.path.insert(0, str(deliverables_dir / "scripts"))

        from biotools import analyze_peptide

        # Should not raise
        analyze_peptide("KLWKKWKKWLK")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
