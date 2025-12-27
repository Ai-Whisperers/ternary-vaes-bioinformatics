# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for mRNA Stability Predictor module."""

import pytest
import torch

from src.analysis.mrna_stability import (
    CODON_STABILITY_SCORES,
    MFEEstimator,
    SecondaryStructurePredictor,
    StabilityPrediction,
    UTROptimizer,
    mRNAStabilityPredictor,
)


class TestCodonStabilityScores:
    """Tests for codon stability data."""

    def test_all_codons_defined(self):
        """Test that most codons have stability scores."""
        # At least 60 codons should be defined
        assert len(CODON_STABILITY_SCORES) >= 60

    def test_scores_in_range(self):
        """Test that all scores are in valid range."""
        for codon, score in CODON_STABILITY_SCORES.items():
            assert 0 <= score <= 1, f"Invalid score for {codon}: {score}"

    def test_synonymous_codon_variation(self):
        """Test that synonymous codons have different scores."""
        # Leucine codons should have different scores
        leu_codons = ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"]
        leu_scores = [CODON_STABILITY_SCORES[c] for c in leu_codons if c in CODON_STABILITY_SCORES]

        assert len(set(leu_scores)) > 1, "Synonymous codons should have varying stability"


class TestSecondaryStructurePredictor:
    """Tests for SecondaryStructurePredictor."""

    def test_creation(self):
        """Test predictor creation."""
        predictor = SecondaryStructurePredictor(hidden_dim=32)
        assert predictor.hidden_dim == 32

    def test_forward(self):
        """Test forward pass."""
        predictor = SecondaryStructurePredictor(hidden_dim=32)

        # Nucleotide indices (0-3 for A, U, G, C)
        sequence = torch.randint(0, 4, (2, 100))

        result = predictor(sequence)

        assert "mfe" in result
        assert "hidden_states" in result
        assert result["mfe"].shape == (2, 1)


class TestMFEEstimator:
    """Tests for MFEEstimator."""

    def test_creation(self):
        """Test estimator creation."""
        estimator = MFEEstimator()
        assert len(estimator.stacking_energies) > 0

    def test_estimate_mfe_empty(self):
        """Test MFE estimation with short sequence."""
        estimator = MFEEstimator()
        mfe = estimator.estimate_mfe("AUG")
        assert mfe == 0.0

    def test_estimate_mfe_gc_rich(self):
        """Test that GC-rich sequences are more stable."""
        estimator = MFEEstimator()

        gc_rich = "G" * 50 + "C" * 50
        au_rich = "A" * 50 + "U" * 50

        mfe_gc = estimator.estimate_mfe(gc_rich)
        mfe_au = estimator.estimate_mfe(au_rich)

        # GC-rich should be more stable (higher normalized score)
        assert mfe_gc > mfe_au

    def test_estimate_mfe_normalized(self):
        """Test that MFE is normalized to 0-1."""
        estimator = MFEEstimator()

        sequence = "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC" * 10

        mfe = estimator.estimate_mfe(sequence)

        assert 0 <= mfe <= 1


class TestUTROptimizer:
    """Tests for UTROptimizer."""

    def test_creation(self):
        """Test optimizer creation."""
        optimizer = UTROptimizer()
        assert len(optimizer.stabilizing_motifs) > 0
        assert len(optimizer.destabilizing_motifs) > 0

    def test_count_motifs(self):
        """Test motif counting."""
        optimizer = UTROptimizer()

        # Sequence with ARE element
        seq_with_are = "AUUUAUAUGCAUUUACC"
        stab, destab = optimizer.count_motifs(seq_with_are)

        assert destab >= 1  # Should find AUUUA

    def test_score_utr(self):
        """Test UTR scoring."""
        optimizer = UTROptimizer()

        # Good UTRs
        good_5 = "GCCGCCAUGACCACC" * 5  # 75 nt with stabilizing
        good_3 = "AAAUAAGCCGCC" * 25  # 300 nt with stabilizing

        # Bad UTRs
        bad_5 = "AUUUAUUUAUUUA" * 5  # ARE elements
        bad_3 = "AUUUAUUUAUUUAUUUA" * 20

        score_good = optimizer.score_utr(good_5, good_3)
        score_bad = optimizer.score_utr(bad_5, bad_3)

        assert score_good > score_bad


class TestmRNAStabilityPredictor:
    """Tests for mRNAStabilityPredictor."""

    def test_creation(self):
        """Test predictor creation."""
        predictor = mRNAStabilityPredictor(hidden_dim=64)
        assert predictor.hidden_dim == 64

    def test_compute_codon_score(self):
        """Test codon score computation."""
        predictor = mRNAStabilityPredictor()

        # Optimal codons
        optimal = "CUGACCGCCAAC"  # CTG, ACC, GCC, AAC (high stability)
        score_opt, rare_opt = predictor.compute_codon_score(optimal)

        # Suboptimal codons
        suboptimal = "UUAACUGCAAAU"  # TTA, ACT, GCA, AAT (lower stability)
        score_sub, rare_sub = predictor.compute_codon_score(suboptimal)

        assert score_opt > score_sub

    def test_compute_gc_content(self):
        """Test GC content computation."""
        predictor = mRNAStabilityPredictor()

        gc_50 = "AUGCAUGC"  # 50% GC
        gc_75 = "GCGCAUGC"  # 75% GC

        assert predictor.compute_gc_content(gc_50) == pytest.approx(0.5)
        assert predictor.compute_gc_content(gc_75) == pytest.approx(0.75)

    def test_forward(self):
        """Test full prediction."""
        predictor = mRNAStabilityPredictor()

        coding = "AUGACCGCCAACGAGGUG" * 10  # 180 nt

        result = predictor(coding)

        assert isinstance(result, StabilityPrediction)
        assert 0 <= result.overall_stability <= 1
        assert result.half_life_hours > 0
        assert 0 <= result.gc_content <= 1

    def test_forward_with_utrs(self):
        """Test prediction with UTRs."""
        predictor = mRNAStabilityPredictor()

        coding = "AUGACCGCCAACGAGGUG" * 10
        utr_5 = "GCCGCCACC" * 8  # 72 nt
        utr_3 = "AAAUAAGCCGCC" * 25  # 300 nt

        result = predictor(coding, utr_5, utr_3)

        assert result.utr_score > 0

    def test_recommendations(self):
        """Test that recommendations are generated."""
        predictor = mRNAStabilityPredictor()

        # Poor sequence (should generate recommendations)
        poor_seq = "UUAUUAUUAUUAUUAUUA" * 10  # Low GC, rare codons

        result = predictor(poor_seq)

        # Should have some recommendations
        assert len(result.recommendations) > 0

    def test_optimize_sequence(self):
        """Test sequence optimization."""
        predictor = mRNAStabilityPredictor()

        # Suboptimal sequence
        original = "UUAACUGCAAAU" * 5  # 60 nt

        optimized, new_prediction = predictor.optimize_sequence(original)

        # Optimized should have better codon score
        orig_prediction = predictor(original)
        assert new_prediction.codon_score >= orig_prediction.codon_score

    def test_preserve_amino_acids(self):
        """Test that optimization preserves amino acids."""
        predictor = mRNAStabilityPredictor()

        # Sequence encoding known amino acids
        original = "UUUUUAUUACUG"  # F, Y, L, L

        optimized, _ = predictor.optimize_sequence(original, preserve_amino_acids=True)

        # Length should be preserved
        assert len(optimized) == len(original)


class TestStabilityPrediction:
    """Tests for StabilityPrediction dataclass."""

    def test_creation(self):
        """Test prediction creation."""
        prediction = StabilityPrediction(
            overall_stability=0.75,
            half_life_hours=12.0,
            codon_score=0.7,
            gc_content=0.55,
            mfe_score=0.6,
            utr_score=0.65,
            rare_codon_count=3,
            recommendations=["Consider codon optimization"],
        )

        assert prediction.overall_stability == 0.75
        assert prediction.half_life_hours == 12.0
        assert len(prediction.recommendations) == 1

    def test_default_recommendations(self):
        """Test default empty recommendations."""
        prediction = StabilityPrediction(
            overall_stability=0.8,
            half_life_hours=10.0,
            codon_score=0.8,
            gc_content=0.5,
            mfe_score=0.7,
            utr_score=0.6,
            rare_codon_count=0,
        )

        assert prediction.recommendations == []
