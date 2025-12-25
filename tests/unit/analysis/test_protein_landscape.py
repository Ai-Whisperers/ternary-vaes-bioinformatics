# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for protein energy landscape analysis."""

import pytest
import torch

from src.analysis.protein_landscape import (
    ConformationState,
    EnergyBasin,
    FoldingFunnelAnalyzer,
    LandscapeMetrics,
    ProteinLandscapeAnalyzer,
    TransitionPath,
    TransitionStateAnalyzer,
    UltrametricDistanceMatrix,
)


class TestUltrametricDistanceMatrix:
    """Tests for UltrametricDistanceMatrix."""

    def test_creation(self):
        """Test module creation."""
        module = UltrametricDistanceMatrix(p=3, n_states=100)
        assert module.p == 3
        assert module.n_states == 100

    def test_compute_padic_distances(self):
        """Test p-adic distance computation."""
        module = UltrametricDistanceMatrix(p=3)

        indices = torch.arange(10)
        distances = module.compute_padic_distances(indices)

        assert distances.shape == (10, 10)
        # Diagonal should be zero
        assert torch.allclose(torch.diag(distances), torch.zeros(10))
        # Should be symmetric
        assert torch.allclose(distances, distances.T)

    def test_check_ultrametricity(self):
        """Test ultrametricity checking."""
        module = UltrametricDistanceMatrix(p=3)

        # p-adic distances should be perfectly ultrametric
        indices = torch.arange(10)
        padic_dist = module.compute_padic_distances(indices)
        score = module.check_ultrametricity(padic_dist)

        assert score == 1.0  # Perfect ultrametricity

    def test_forward(self):
        """Test forward pass."""
        module = UltrametricDistanceMatrix()

        states = torch.randn(2, 20, 64)  # batch, n_states, dim
        results = module(states)

        assert len(results) == 2
        assert "euclidean_distances" in results[0]
        assert "padic_distances" in results[0]
        assert "ultrametricity" in results[0]


class TestFoldingFunnelAnalyzer:
    """Tests for FoldingFunnelAnalyzer."""

    def test_creation(self):
        """Test analyzer creation."""
        analyzer = FoldingFunnelAnalyzer(input_dim=64, hidden_dim=128)
        assert analyzer.input_dim == 64
        assert analyzer.hidden_dim == 128

    def test_predict_energy(self):
        """Test energy prediction."""
        analyzer = FoldingFunnelAnalyzer(input_dim=64)

        conformations = torch.randn(10, 64)
        energies = analyzer.predict_energy(conformations)

        assert energies.shape == (10,)

    def test_classify_state(self):
        """Test state classification."""
        analyzer = FoldingFunnelAnalyzer(input_dim=64)

        conformations = torch.randn(10, 64)
        logits = analyzer.classify_state(conformations)

        assert logits.shape == (10, len(ConformationState))

    def test_detect_basins(self):
        """Test basin detection."""
        analyzer = FoldingFunnelAnalyzer(input_dim=64)

        # Create conformations with clear structure
        conformations = torch.randn(50, 64)
        energies = torch.randn(50)

        basins = analyzer.detect_basins(conformations, energies)

        assert isinstance(basins, list)
        for basin in basins:
            assert isinstance(basin, EnergyBasin)

    def test_compute_funnel_metrics(self):
        """Test funnel metrics computation."""
        analyzer = FoldingFunnelAnalyzer(input_dim=64)

        conformations = torch.randn(30, 64)
        energies = torch.randn(30)
        native = conformations[0]

        metrics = analyzer.compute_funnel_metrics(conformations, energies, native)

        assert "funnel_depth" in metrics
        assert "ruggedness" in metrics
        assert "frustration" in metrics

    def test_forward(self):
        """Test forward pass."""
        analyzer = FoldingFunnelAnalyzer(input_dim=64)

        conformations = torch.randn(2, 30, 64)
        native_states = torch.randn(2, 64)

        results = analyzer(conformations, native_states)

        assert len(results) == 2
        assert "energies" in results[0]
        assert "basins" in results[0]
        assert "funnel_metrics" in results[0]


class TestTransitionStateAnalyzer:
    """Tests for TransitionStateAnalyzer."""

    def test_creation(self):
        """Test analyzer creation."""
        analyzer = TransitionStateAnalyzer(input_dim=64)
        assert analyzer.input_dim == 64

    def test_find_transition_state(self):
        """Test transition state finding."""
        analyzer = TransitionStateAnalyzer(input_dim=64)

        state1 = torch.randn(2, 64)
        state2 = torch.randn(2, 64)

        ts = analyzer.find_transition_state(state1, state2)

        assert ts.shape == (2, 64)

    def test_estimate_barrier(self):
        """Test barrier estimation."""
        analyzer = TransitionStateAnalyzer(input_dim=64)

        state1 = torch.randn(2, 64)
        state2 = torch.randn(2, 64)

        barrier = analyzer.estimate_barrier(state1, state2)

        assert barrier.shape == (2,)
        assert (barrier >= 0).all()

    def test_compute_padic_path_length(self):
        """Test p-adic path length computation."""
        analyzer = TransitionStateAnalyzer(p=3)

        path = torch.randn(2, 10, 64)  # batch, n_steps, dim
        lengths = analyzer.compute_padic_path_length(path)

        assert lengths.shape == (2,)
        assert (lengths >= 0).all()

    def test_forward(self):
        """Test forward pass."""
        analyzer = TransitionStateAnalyzer(input_dim=64)

        state1 = torch.randn(64)
        state2 = torch.randn(64)

        result = analyzer(state1, state2)

        assert isinstance(result, TransitionPath)
        assert result.barrier_height >= 0
        assert result.path_length >= 0


class TestProteinLandscapeAnalyzer:
    """Tests for ProteinLandscapeAnalyzer."""

    def test_creation(self):
        """Test analyzer creation."""
        analyzer = ProteinLandscapeAnalyzer(input_dim=64, hidden_dim=128)
        assert analyzer.input_dim == 64

    def test_forward(self):
        """Test forward pass."""
        analyzer = ProteinLandscapeAnalyzer(input_dim=64)

        conformations = torch.randn(2, 30, 64)
        native_states = torch.randn(2, 64)

        results = analyzer(conformations, native_states)

        assert "ultrametric" in results
        assert "funnel" in results
        assert "metrics" in results

    def test_forward_without_native(self):
        """Test forward without explicit native state."""
        analyzer = ProteinLandscapeAnalyzer(input_dim=64)

        conformations = torch.randn(2, 30, 64)

        results = analyzer(conformations)

        assert len(results["metrics"]) == 2

    def test_metrics_dataclass(self):
        """Test that metrics are properly returned."""
        analyzer = ProteinLandscapeAnalyzer(input_dim=64)

        conformations = torch.randn(1, 20, 64)
        results = analyzer(conformations)

        metrics = results["metrics"][0]
        assert isinstance(metrics, LandscapeMetrics)
        assert isinstance(metrics.ruggedness, float)
        assert isinstance(metrics.n_basins, int)

    def test_compare_landscapes(self):
        """Test landscape comparison."""
        analyzer = ProteinLandscapeAnalyzer(input_dim=64)

        conformations1 = torch.randn(1, 20, 64)
        conformations2 = torch.randn(1, 20, 64)

        results1 = analyzer(conformations1)
        results2 = analyzer(conformations2)

        comparison = analyzer.compare_landscapes(results1, results2)

        assert "ruggedness_diff" in comparison
        assert "funnel_depth_diff" in comparison
        assert "stability_diff" in comparison


class TestConformationState:
    """Tests for ConformationState enum."""

    def test_all_states_exist(self):
        """Test that expected states are defined."""
        assert ConformationState.NATIVE
        assert ConformationState.MOLTEN_GLOBULE
        assert ConformationState.INTERMEDIATE
        assert ConformationState.UNFOLDED
        assert ConformationState.MISFOLDED
        assert ConformationState.AGGREGATED

    def test_state_values(self):
        """Test state values are strings."""
        assert ConformationState.NATIVE.value == "native"
        assert ConformationState.UNFOLDED.value == "unfolded"


class TestEnergyBasin:
    """Tests for EnergyBasin dataclass."""

    def test_creation(self):
        """Test energy basin creation."""
        basin = EnergyBasin(
            center=torch.randn(64),
            energy=-5.0,
            depth=2.5,
            width=1.0,
            state=ConformationState.NATIVE,
            escape_barrier=3.0,
        )

        assert basin.energy == -5.0
        assert basin.depth == 2.5
        assert basin.state == ConformationState.NATIVE


class TestTransitionPath:
    """Tests for TransitionPath dataclass."""

    def test_creation(self):
        """Test transition path creation."""
        path = TransitionPath(
            start_state=ConformationState.UNFOLDED,
            end_state=ConformationState.NATIVE,
            barrier_height=5.0,
            path_length=10.0,
            transition_state=torch.randn(64),
            rate_constant=0.01,
        )

        assert path.barrier_height == 5.0
        assert path.rate_constant == 0.01


class TestLandscapeMetrics:
    """Tests for LandscapeMetrics dataclass."""

    def test_creation(self):
        """Test metrics creation."""
        metrics = LandscapeMetrics(
            ruggedness=1.5,
            funnel_depth=10.0,
            frustration=0.2,
            ultrametricity=0.95,
            n_basins=3,
            native_stability=8.0,
        )

        assert metrics.ruggedness == 1.5
        assert metrics.n_basins == 3
        assert metrics.ultrametricity == 0.95
