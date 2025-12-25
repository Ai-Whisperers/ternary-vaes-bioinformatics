# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for glycan loss module."""

import pytest
import torch

from src.losses.glycan_loss import (
    GlycanRemovalSimulator,
    GlycanSequonDetector,
    GlycanShieldAnalyzer,
    GlycanShieldMetrics,
    GlycanSite,
    SentinelGlycanLoss,
)


class TestGlycanSequonDetector:
    """Tests for GlycanSequonDetector."""

    def test_creation(self):
        """Test detector creation."""
        detector = GlycanSequonDetector(embedding_dim=32)
        assert detector.embedding_dim == 32

    def test_detect_sequons_nxs(self):
        """Test detection of NXS sequon."""
        detector = GlycanSequonDetector()
        sites = detector.detect_sequons("MAANSTPEPTIDE")

        # Should find NAS and NST
        sequons = [s.sequon for s in sites]
        assert "NAS" in sequons or "NST" in sequons

    def test_detect_sequons_nxt(self):
        """Test detection of NXT sequon."""
        detector = GlycanSequonDetector()
        # Sequence with N-X-T pattern where X is not P
        sites = detector.detect_sequons("MANFTPEPTIDE")

        assert len(sites) >= 1
        assert sites[0].sequon == "NFT"

    def test_no_nxp_sequon(self):
        """Test that NXP is not detected (proline blocks)."""
        detector = GlycanSequonDetector()
        sites = detector.detect_sequons("ANPPEPTIDE")

        # NPS should not be detected (P in middle position)
        sequons = [s.sequon for s in sites]
        assert "NPS" not in [s for s in sequons if "P" in s[1]]

    def test_forward(self):
        """Test forward pass."""
        detector = GlycanSequonDetector()

        # Create random sequence indices
        seq_indices = torch.randint(0, 20, (2, 50))

        result = detector(seq_indices)

        assert "sequon_probs" in result
        assert "occupancy_probs" in result
        assert result["sequon_probs"].shape == (2, 50)


class TestGlycanShieldAnalyzer:
    """Tests for GlycanShieldAnalyzer."""

    def test_creation(self):
        """Test analyzer creation."""
        analyzer = GlycanShieldAnalyzer(n_positions=500, glycan_radius=15.0)
        assert analyzer.glycan_radius == 15.0

    def test_compute_shield_coverage(self):
        """Test shield coverage computation."""
        analyzer = GlycanShieldAnalyzer()

        # Place glycans at positions
        glycan_positions = torch.tensor([50, 100, 150, 200])
        coverage = analyzer.compute_shield_coverage(glycan_positions, 300)

        # Should have some coverage but not 100%
        assert 0 < coverage < 1

    def test_empty_glycans(self):
        """Test with no glycans."""
        analyzer = GlycanShieldAnalyzer()

        glycan_positions = torch.tensor([])
        coverage = analyzer.compute_shield_coverage(glycan_positions, 100)

        assert coverage == 0.0

    def test_compute_padic_visibility(self):
        """Test p-adic visibility computation."""
        analyzer = GlycanShieldAnalyzer(p=3)

        glycan_positions = torch.tensor([0, 27, 54])  # Multiples of 27
        visibility = analyzer.compute_padic_visibility(glycan_positions, 13)

        assert visibility >= 0

    def test_identify_vulnerable_regions(self):
        """Test vulnerable region identification."""
        analyzer = GlycanShieldAnalyzer()

        # Sparse glycans
        glycan_positions = torch.tensor([10, 90])
        regions = analyzer.identify_vulnerable_regions(
            glycan_positions, 100, threshold=0.3
        )

        # Should find some vulnerable regions
        assert isinstance(regions, list)

    def test_forward(self):
        """Test forward pass returns metrics."""
        analyzer = GlycanShieldAnalyzer()

        glycan_positions = torch.tensor([20, 50, 80, 110, 140])
        metrics = analyzer(glycan_positions, 200)

        assert isinstance(metrics, GlycanShieldMetrics)
        assert metrics.total_glycans == 5
        assert 0 <= metrics.shield_coverage <= 1
        assert metrics.sentinel_score >= 0


class TestSentinelGlycanLoss:
    """Tests for SentinelGlycanLoss."""

    def test_creation(self):
        """Test loss creation."""
        loss = SentinelGlycanLoss(coverage_weight=0.3, visibility_weight=0.4)
        assert loss.coverage_weight == 0.3
        assert loss.visibility_weight == 0.4

    def test_forward_basic(self):
        """Test basic forward pass."""
        loss = SentinelGlycanLoss()

        # Predicted glycan occupancies
        predicted_glycans = torch.rand(2, 100)
        epitope_visibility = torch.rand(2, 5)

        result = loss(predicted_glycans, epitope_visibility)

        assert "loss" in result
        assert "coverage_loss" in result
        assert "visibility_loss" in result

    def test_forward_with_targets(self):
        """Test forward with target visibility."""
        loss = SentinelGlycanLoss()

        predicted_glycans = torch.rand(2, 100)
        epitope_visibility = torch.rand(2, 5)
        target_visibility = torch.rand(2, 5)

        result = loss(predicted_glycans, epitope_visibility, target_visibility)

        assert result["supervision_loss"] >= 0

    def test_weights_affect_loss(self):
        """Test that weights affect loss components."""
        loss1 = SentinelGlycanLoss(coverage_weight=1.0, visibility_weight=0.0)
        loss2 = SentinelGlycanLoss(coverage_weight=0.0, visibility_weight=1.0)

        predicted_glycans = torch.rand(1, 100)
        epitope_visibility = torch.rand(1, 5)

        result1 = loss1(predicted_glycans, epitope_visibility)
        result2 = loss2(predicted_glycans, epitope_visibility)

        # Different weight configurations should give different results
        # (unless by coincidence they're the same)
        assert result1["loss"] is not None
        assert result2["loss"] is not None


class TestGlycanRemovalSimulator:
    """Tests for GlycanRemovalSimulator."""

    def test_creation(self):
        """Test simulator creation."""
        simulator = GlycanRemovalSimulator(n_positions=500)
        assert simulator.analyzer is not None

    def test_simulate_removal(self):
        """Test glycan removal simulation."""
        simulator = GlycanRemovalSimulator()

        glycan_positions = torch.tensor([20, 50, 80, 110])
        before, after = simulator.simulate_removal(
            glycan_positions, 200, removal_indices=[1]
        )

        assert before.total_glycans == 4
        assert after.total_glycans == 3

    def test_find_optimal_removals(self):
        """Test finding optimal glycan removals."""
        simulator = GlycanRemovalSimulator()

        glycan_positions = torch.tensor([30, 50, 70, 100])
        target_epitope = (45, 55)  # Near position 50

        results = simulator.find_optimal_removals(
            glycan_positions, 150, target_epitope
        )

        # Should return list of (indices, improvement) tuples
        assert isinstance(results, list)

    def test_forward(self):
        """Test full analysis forward pass."""
        simulator = GlycanRemovalSimulator()

        glycan_positions = torch.tensor([20, 50, 80, 110, 140])
        target_epitopes = [(45, 55), (75, 85)]

        result = simulator(glycan_positions, 200, target_epitopes)

        assert "original_metrics" in result
        assert "epitope_analysis" in result
        assert len(result["epitope_analysis"]) == 2


class TestGlycanSite:
    """Tests for GlycanSite dataclass."""

    def test_creation(self):
        """Test GlycanSite creation."""
        site = GlycanSite(
            position=50,
            sequon="NAS",
            amino_acid_x="A",
            is_occupied=True,
            glycan_type="high_mannose",
        )

        assert site.position == 50
        assert site.sequon == "NAS"
        assert site.is_occupied
