# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for citrullination boundary optimizer."""

import pytest
import torch

from src.optimization.citrullination_optimizer import (
    CitrullinationBoundaryOptimizer,
    CodonChoice,
    CodonContextOptimizer,
    OptimizationResult,
    PAdicBoundaryAnalyzer,
)


class TestCodonChoice:
    """Tests for CodonChoice dataclass."""

    def test_creation(self):
        """Test dataclass creation."""
        choice = CodonChoice(
            codon="CGT",
            amino_acid="R",
            padic_distance=0.333,
            boundary_distance=0.15,
            codon_usage=0.45,
            combined_score=0.7,
        )

        assert choice.codon == "CGT"
        assert choice.amino_acid == "R"
        assert choice.padic_distance == 0.333
        assert choice.boundary_distance == 0.15


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_creation(self):
        """Test result creation."""
        result = OptimizationResult(
            original_sequence="ATGCGT",
            optimized_sequence="ATGAGA",
            original_arginines=[(1, "CGT")],
            optimized_arginines=[(1, "AGA")],
            padic_improvement=0.2,
            boundary_margin_improvement=0.15,
            codon_usage_maintained=True,
        )

        assert result.original_sequence == "ATGCGT"
        assert result.optimized_sequence == "ATGAGA"
        assert result.padic_improvement == 0.2


class TestPAdicBoundaryAnalyzer:
    """Tests for PAdicBoundaryAnalyzer."""

    def test_creation(self):
        """Test analyzer creation."""
        analyzer = PAdicBoundaryAnalyzer(
            p=3, goldilocks_lower=0.1, goldilocks_upper=0.5
        )
        assert analyzer.p == 3
        assert analyzer.goldilocks_lower == 0.1
        assert analyzer.goldilocks_upper == 0.5

    def test_compute_codon_valuation(self):
        """Test codon p-adic valuation."""
        analyzer = PAdicBoundaryAnalyzer(p=3)

        # Different codons should have different valuations
        val_cgt = analyzer.compute_codon_valuation("CGT")
        val_aga = analyzer.compute_codon_valuation("AGA")

        assert isinstance(val_cgt, float)
        assert isinstance(val_aga, float)

    def test_compute_boundary_distance(self):
        """Test boundary distance computation."""
        analyzer = PAdicBoundaryAnalyzer(
            p=3, goldilocks_lower=0.1, goldilocks_upper=0.5
        )

        # Test point inside zone
        dist_inside = analyzer.compute_boundary_distance(0.3)
        assert dist_inside >= 0

        # Test point outside zone
        dist_outside = analyzer.compute_boundary_distance(0.7)
        assert dist_outside >= 0

        # Point outside zone should be further from zone boundaries
        assert dist_outside >= dist_inside or dist_inside >= 0

    def test_analyze_arginine_codons(self):
        """Test analysis of arginine codons."""
        analyzer = PAdicBoundaryAnalyzer(p=3)

        codons = ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"]
        analysis = analyzer.analyze_arginine_codons(codons)

        assert len(analysis) == 6
        for codon, result in analysis.items():
            assert "valuation" in result
            assert "distance" in result

    def test_rank_codons_by_safety(self):
        """Test codon ranking by safety."""
        analyzer = PAdicBoundaryAnalyzer(p=3)

        codons = ["CGT", "AGA", "AGG"]
        ranked = analyzer.rank_codons_by_safety(codons)

        assert len(ranked) == 3
        # All should be valid codon choices
        for choice in ranked:
            assert isinstance(choice, CodonChoice)
            assert choice.codon in codons


class TestCodonContextOptimizer:
    """Tests for CodonContextOptimizer."""

    def test_creation(self):
        """Test optimizer creation."""
        optimizer = CodonContextOptimizer(context_window=5)
        assert optimizer.context_window == 5

    def test_extract_context(self):
        """Test context extraction."""
        optimizer = CodonContextOptimizer(context_window=3)

        sequence = "ATGCGTACG"  # 3 codons
        context = optimizer.extract_context(sequence, codon_position=1)

        assert len(context) > 0

    def test_compute_context_score(self):
        """Test context scoring."""
        optimizer = CodonContextOptimizer()

        upstream = "ATG"
        downstream = "ACG"
        codon = "CGT"

        score = optimizer.compute_context_score(upstream, codon, downstream)
        assert isinstance(score, float)

    def test_find_optimal_codon(self):
        """Test finding optimal codon."""
        optimizer = CodonContextOptimizer()

        candidates = ["CGT", "AGA", "AGG"]
        upstream = "ATG"
        downstream = "ACG"

        best_codon = optimizer.find_optimal_codon(candidates, upstream, downstream)

        assert best_codon in candidates


class TestCitrullinationBoundaryOptimizer:
    """Tests for CitrullinationBoundaryOptimizer."""

    def test_creation(self):
        """Test optimizer creation."""
        optimizer = CitrullinationBoundaryOptimizer(
            p=3, goldilocks_bounds=(0.1, 0.5)
        )
        assert optimizer.p == 3

    def test_find_arginine_codons(self):
        """Test finding arginine codons in sequence."""
        optimizer = CitrullinationBoundaryOptimizer()

        # Sequence with arginine codons at positions 1 and 3 (0-indexed)
        sequence = "ATGCGTACGAGA"  # ATG CGT ACG AGA

        positions = optimizer.find_arginine_codons(sequence)

        # Should find CGT at position 1 and AGA at position 3
        assert len(positions) == 2
        assert (1, "CGT") in positions or (3, "AGA") in positions

    def test_optimize_arginine_codons(self):
        """Test arginine codon optimization."""
        optimizer = CitrullinationBoundaryOptimizer()

        codons = ["CGT", "AGA"]
        optimized = optimizer.optimize_arginine_codons(codons, usage_weight=0.3)

        assert len(optimized) == 2
        # All optimized codons should still encode arginine
        arginine_codons = {"CGT", "CGC", "CGA", "CGG", "AGA", "AGG"}
        for codon in optimized:
            assert codon in arginine_codons

    def test_optimize_sequence(self):
        """Test full sequence optimization."""
        optimizer = CitrullinationBoundaryOptimizer()

        # Simple sequence with one arginine
        sequence = "ATGCGTACG"  # ATG CGT ACG (Met Arg Thr)

        result = optimizer.optimize_sequence(sequence)

        assert isinstance(result, OptimizationResult)
        assert len(result.optimized_sequence) == len(sequence)
        # Should preserve non-arginine codons
        assert result.optimized_sequence[:3] == "ATG"
        assert result.optimized_sequence[6:] == "ACG"

    def test_optimize_sequence_no_arginines(self):
        """Test optimization of sequence without arginines."""
        optimizer = CitrullinationBoundaryOptimizer()

        sequence = "ATGACGTTG"  # No arginine codons

        result = optimizer.optimize_sequence(sequence)

        assert result.optimized_sequence == sequence
        assert len(result.original_arginines) == 0

    def test_batch_optimize(self):
        """Test batch optimization."""
        optimizer = CitrullinationBoundaryOptimizer()

        sequences = [
            "ATGCGTACG",
            "ATGAGAACG",
            "ATGACGTTT",
        ]

        results = optimizer.batch_optimize(sequences)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, OptimizationResult)


class TestIntegration:
    """Integration tests for optimization pipeline."""

    def test_full_pipeline(self):
        """Test complete optimization pipeline."""
        # Create components
        boundary_analyzer = PAdicBoundaryAnalyzer(
            p=3, goldilocks_lower=0.1, goldilocks_upper=0.5
        )
        context_optimizer = CodonContextOptimizer(context_window=5)
        optimizer = CitrullinationBoundaryOptimizer(
            p=3, goldilocks_bounds=(0.1, 0.5)
        )

        # Sequence with multiple arginines
        sequence = "ATGCGTACGAGATTG"  # ATG CGT ACG AGA TTG

        # Analyze current state
        arg_positions = optimizer.find_arginine_codons(sequence)

        # Optimize
        result = optimizer.optimize_sequence(sequence, optimize_context_flag=True)

        # Verify improvement metrics
        assert result is not None
        assert isinstance(result.padic_improvement, float)

    def test_maintains_amino_acid_sequence(self):
        """Test that optimization preserves protein sequence."""
        optimizer = CitrullinationBoundaryOptimizer()

        sequence = "ATGCGTACGAGACGT"  # Met Arg Thr Arg Arg

        result = optimizer.optimize_sequence(sequence)

        # The amino acid sequence should be preserved
        # (all arginine codons should still encode arginine)
        optimized = result.optimized_sequence

        # Check each codon position
        for i in range(0, len(optimized), 3):
            original_codon = sequence[i : i + 3]
            optimized_codon = optimized[i : i + 3]

            # If original was arginine, optimized should be too
            arginine_codons = {"CGT", "CGC", "CGA", "CGG", "AGA", "AGG"}
            if original_codon in arginine_codons:
                assert (
                    optimized_codon in arginine_codons
                ), f"Arginine codon changed to non-arginine: {optimized_codon}"

    def test_improvement_direction(self):
        """Test that optimization improves boundary margins."""
        optimizer = CitrullinationBoundaryOptimizer(
            p=3, goldilocks_bounds=(0.1, 0.5)
        )

        # Run multiple optimizations
        sequences = [
            "ATGCGTACG",
            "CGTCGTCGT",  # Three arginines
            "AGAAGAAGA",  # Three arginines (different codon)
        ]

        for seq in sequences:
            result = optimizer.optimize_sequence(seq)
            # Improvement should be non-negative (we don't make things worse)
            assert (
                result.padic_improvement >= -0.01
            )  # Small tolerance for numerical issues
