# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for motor-inspired encoders."""

import math

import pytest
import torch

from src.encoders.motor_encoder import (
    ATPSynthaseEncoder,
    RotaryPositionEncoder,
    TernaryMotorEncoder,
)


class TestRotaryPositionEncoder:
    """Tests for RotaryPositionEncoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = RotaryPositionEncoder(embedding_dim=16, max_states=3)
        assert encoder.embedding_dim == 16
        assert encoder.max_states == 3

    def test_forward_1d(self):
        """Test encoding 1D phase input."""
        encoder = RotaryPositionEncoder(embedding_dim=16)
        phase = torch.tensor([0.0, math.pi / 2, math.pi])

        output = encoder(phase)

        assert output.shape == (3, 16)

    def test_forward_2d(self):
        """Test encoding 2D phase input."""
        encoder = RotaryPositionEncoder(embedding_dim=16)
        phase = torch.randn(4, 8)  # (Batch, SeqLen)

        output = encoder(phase)

        assert output.shape == (4, 8, 16)

    def test_state_to_phase(self):
        """Test state index to phase conversion."""
        encoder = RotaryPositionEncoder(embedding_dim=16, max_states=3)

        # State 0 -> phase 0
        assert encoder.state_to_phase(torch.tensor(0.0)).item() == pytest.approx(0.0)

        # State 1 -> phase 2π/3
        assert encoder.state_to_phase(torch.tensor(1.0)).item() == pytest.approx(2 * math.pi / 3)

        # State 2 -> phase 4π/3
        assert encoder.state_to_phase(torch.tensor(2.0)).item() == pytest.approx(4 * math.pi / 3)


class TestTernaryMotorEncoder:
    """Tests for TernaryMotorEncoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = TernaryMotorEncoder(embedding_dim=32)
        assert encoder.embedding_dim == 32
        assert encoder.n_subunits == 3
        assert encoder.n_states == 3

    def test_forward_single_subunit(self):
        """Test encoding single subunit states."""
        encoder = TernaryMotorEncoder(embedding_dim=32)
        states = torch.tensor([0, 1, 2])  # Three samples, single subunit each

        output = encoder(states)

        assert output.shape == (3, 32)
        # Output should be on Poincaré ball (norm < 1)
        assert (torch.norm(output, dim=-1) < 1.0).all()

    def test_forward_multi_subunit(self):
        """Test encoding multi-subunit states."""
        encoder = TernaryMotorEncoder(embedding_dim=32, n_subunits=3)
        # (Batch=4, Subunits=3)
        states = torch.randint(0, 3, (4, 3))

        output = encoder(states)

        assert output.shape == (4, 32)

    def test_forward_with_phase(self):
        """Test encoding with explicit phase."""
        encoder = TernaryMotorEncoder(embedding_dim=32)
        states = torch.randint(0, 3, (4, 3))
        phase = torch.rand(4) * 2 * math.pi

        output = encoder(states, phase=phase)

        assert output.shape == (4, 32)

    def test_state_transitions(self):
        """Test transition distances between states."""
        encoder = TernaryMotorEncoder(embedding_dim=32)
        distances = encoder.get_state_transitions()

        assert distances.shape == (3,)
        # All transitions should be positive
        assert (distances > 0).all()

    def test_rotation_cycle(self):
        """Test encoding a full rotation cycle."""
        encoder = TernaryMotorEncoder(embedding_dim=32)
        embeddings = encoder.encode_rotation_cycle(n_steps=12)

        assert embeddings.shape == (12, 32)
        # All points should be on Poincaré ball
        assert (torch.norm(embeddings, dim=-1) < 1.0).all()


class TestATPSynthaseEncoder:
    """Tests for ATPSynthaseEncoder."""

    def test_initialization(self):
        """Test ATP Synthase encoder initialization."""
        encoder = ATPSynthaseEncoder(embedding_dim=64)
        assert encoder.embedding_dim == 64
        assert encoder.n_subunits == 3
        assert encoder.n_states == 3

    def test_forward(self):
        """Test forward pass."""
        encoder = ATPSynthaseEncoder(embedding_dim=64)
        # β subunit states: Open(0), Loose(1), Tight(2)
        beta_states = torch.tensor([[0, 1, 2], [1, 2, 0]])

        output = encoder(beta_states)

        assert output.shape == (2, 64)

    def test_forward_with_gamma_angle(self):
        """Test forward pass with γ-subunit rotation."""
        encoder = ATPSynthaseEncoder(embedding_dim=64)
        beta_states = torch.tensor([[0, 1, 2]])
        gamma_angle = torch.tensor([120.0])  # degrees

        output = encoder(beta_states, gamma_angle=gamma_angle)

        assert output.shape == (1, 64)

    def test_different_states_different_embeddings(self):
        """Test that different states produce different embeddings."""
        encoder = ATPSynthaseEncoder(embedding_dim=64)

        # Truly different states (not just rotations)
        state1 = torch.tensor([[0, 0, 0]])  # All Open
        state2 = torch.tensor([[2, 2, 2]])  # All Tight

        emb1 = encoder(state1)
        emb2 = encoder(state2)

        # Different configurations should give different embeddings
        assert not torch.allclose(emb1, emb2, atol=1e-3)
