# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for circadian cycle encoders."""

import math

import pytest
import torch

from src.encoders.circadian_encoder import (
    CircadianCycleEncoder,
    KaiCClockEncoder,
    ToroidalEmbedding,
)


class TestToroidalEmbedding:
    """Tests for ToroidalEmbedding."""

    def test_initialization(self):
        """Test embedding initialization."""
        emb = ToroidalEmbedding(embedding_dim=32, n_harmonics=4)
        assert emb.embedding_dim == 32
        assert emb.n_harmonics == 4

    def test_forward(self):
        """Test forward pass."""
        emb = ToroidalEmbedding(embedding_dim=32)
        theta = torch.rand(8) * 2 * math.pi
        phi = torch.rand(8) * 2 * math.pi

        output = emb(theta, phi)

        assert output.shape == (8, 32)

    def test_to_3d_torus(self):
        """Test conversion to 3D coordinates."""
        emb = ToroidalEmbedding(major_radius=1.0, minor_radius=0.5)

        # At theta=0, phi=0, should be at (R+r, 0, 0)
        coords = emb.to_3d_torus(torch.tensor([0.0]), torch.tensor([0.0]))
        assert coords[0, 0].item() == pytest.approx(1.5, abs=1e-5)
        assert coords[0, 1].item() == pytest.approx(0.0, abs=1e-5)
        assert coords[0, 2].item() == pytest.approx(0.0, abs=1e-5)

    def test_periodic_consistency(self):
        """Test that embeddings are consistent with period."""
        emb = ToroidalEmbedding(embedding_dim=32)

        # Same point with 2π offset
        theta = torch.tensor([0.5])
        phi = torch.tensor([1.0])

        emb1 = emb(theta, phi)
        emb2 = emb(theta + 2 * math.pi, phi + 2 * math.pi)

        assert torch.allclose(emb1, emb2, atol=1e-5)


class TestCircadianCycleEncoder:
    """Tests for CircadianCycleEncoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = CircadianCycleEncoder(embedding_dim=32)
        assert encoder.embedding_dim == 32
        assert encoder.n_phospho_states == 4
        assert encoder.period_hours == 24.0

    def test_forward_with_discrete_state(self):
        """Test forward pass with discrete phosphorylation state."""
        encoder = CircadianCycleEncoder(embedding_dim=32)
        time = torch.tensor([0.0, 6.0, 12.0, 18.0])
        state = torch.tensor([0, 1, 2, 3])

        output = encoder(time, phospho_state=state)

        assert output.shape == (4, 32)

    def test_forward_with_continuous_state(self):
        """Test forward pass with continuous phosphorylation."""
        encoder = CircadianCycleEncoder(embedding_dim=32)
        time = torch.tensor([0.0, 12.0])
        phospho = torch.tensor([0.0, 0.5])

        output = encoder(time, phospho_continuous=phospho)

        assert output.shape == (2, 32)

    def test_forward_without_state(self):
        """Test forward pass with only time."""
        encoder = CircadianCycleEncoder(embedding_dim=32)
        time = torch.tensor([0.0, 6.0, 12.0])

        output = encoder(time)

        assert output.shape == (3, 32)

    def test_time_to_phase(self):
        """Test time to phase conversion."""
        encoder = CircadianCycleEncoder(period_hours=24.0)

        # Midnight -> phase 0
        assert encoder.time_to_phase(torch.tensor(0.0)).item() == pytest.approx(0.0)

        # Noon -> phase π
        assert encoder.time_to_phase(torch.tensor(12.0)).item() == pytest.approx(math.pi)

        # 6 PM -> phase 3π/2
        assert encoder.time_to_phase(torch.tensor(18.0)).item() == pytest.approx(1.5 * math.pi)

    def test_phase_trajectory(self):
        """Test generating a full circadian cycle trajectory."""
        encoder = CircadianCycleEncoder(embedding_dim=32)
        embeddings, times = encoder.get_phase_trajectory(n_timepoints=48)

        assert embeddings.shape == (48, 32)
        assert times.shape == (48,)
        assert times[0].item() == pytest.approx(0.0)
        assert times[-1].item() == pytest.approx(24.0)

    def test_phase_coherence(self):
        """Test phase coherence computation."""
        encoder = CircadianCycleEncoder(embedding_dim=32)
        time = torch.tensor([0.0, 6.0, 12.0])
        state = torch.tensor([0, 1, 2])

        coherence = encoder.compute_phase_coherence(time, state)

        assert coherence.shape == (3,)
        # Coherence should be between -1 and 1
        assert (coherence >= -1.0).all()
        assert (coherence <= 1.0).all()


class TestKaiCClockEncoder:
    """Tests for KaiCClockEncoder."""

    def test_initialization(self):
        """Test KaiC encoder initialization."""
        encoder = KaiCClockEncoder(embedding_dim=64)
        assert encoder.embedding_dim == 64
        assert encoder.n_phospho_states == 4
        assert encoder.state_names == ["U", "T", "ST", "S"]

    def test_forward(self):
        """Test forward pass."""
        encoder = KaiCClockEncoder(embedding_dim=64)
        time = torch.tensor([0.0, 6.0, 12.0, 18.0])
        state = torch.tensor([0, 1, 2, 3])  # U, T, ST, S

        output = encoder(time, phospho_state=state)

        assert output.shape == (4, 64)

    def test_forward_with_kai_levels(self):
        """Test forward pass with KaiA/KaiB modulation."""
        encoder = KaiCClockEncoder(embedding_dim=64)
        time = torch.tensor([12.0])
        state = torch.tensor([2])
        kaia = torch.tensor([0.8])
        kaib = torch.tensor([0.2])

        output = encoder(time, phospho_state=state, kaia_level=kaia, kaib_level=kaib)

        assert output.shape == (1, 64)

    def test_different_states_different_embeddings(self):
        """Test that different phosphorylation states give different embeddings."""
        encoder = KaiCClockEncoder(embedding_dim=64)
        time = torch.tensor([12.0, 12.0])
        states = torch.tensor([0, 2])  # U vs ST

        output = encoder(time, phospho_state=states)

        assert not torch.allclose(output[0], output[1])

    def test_time_dependency(self):
        """Test that same state at different times gives different embeddings."""
        encoder = KaiCClockEncoder(embedding_dim=64)
        times = torch.tensor([0.0, 12.0])
        state = torch.tensor([1, 1])  # Same state

        output = encoder(times, phospho_state=state)

        assert not torch.allclose(output[0], output[1])
