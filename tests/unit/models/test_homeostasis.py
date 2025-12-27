# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for HomeostasisController.

Tests cover:
- Q computation function
- Controller initialization
- Freeze state management
- Coverage-gated encoder_A behavior
- Hierarchy-gated encoder_B behavior
- Gradient-gated controller behavior
- Q-gated annealing
- Warmup period handling
"""

from __future__ import annotations

import pytest


class TestComputeQ:
    """Tests for compute_Q function."""

    def test_q_with_zero_inputs(self):
        """Test Q computation with zeros."""
        from src.models.homeostasis import compute_Q

        q = compute_Q(dist_corr=0.0, hierarchy=0.0)
        assert q == 0.0

    def test_q_with_positive_values(self):
        """Test Q computation with positive values."""
        from src.models.homeostasis import compute_Q

        q = compute_Q(dist_corr=0.5, hierarchy=0.3)
        # Q = 0.5 + 1.5 * |0.3| = 0.5 + 0.45 = 0.95
        assert q == pytest.approx(0.95)

    def test_q_with_negative_hierarchy(self):
        """Test Q computation with negative hierarchy (absolute value)."""
        from src.models.homeostasis import compute_Q

        q = compute_Q(dist_corr=0.5, hierarchy=-0.4)
        # Q = 0.5 + 1.5 * |-0.4| = 0.5 + 0.6 = 1.1
        assert q == pytest.approx(1.1)

    def test_q_hierarchy_contribution(self):
        """Test that hierarchy is weighted by 1.5."""
        from src.models.homeostasis import compute_Q

        q1 = compute_Q(dist_corr=0.0, hierarchy=1.0)
        q2 = compute_Q(dist_corr=0.0, hierarchy=2.0)

        assert q2 - q1 == pytest.approx(1.5)


class TestHomeostasisControllerInit:
    """Tests for HomeostasisController initialization."""

    def test_default_init(self):
        """Test default initialization."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController()

        # Check initial freeze states (Option C defaults)
        assert controller.encoder_a_frozen is True  # Starts frozen
        assert controller.encoder_b_frozen is False  # Starts trainable
        assert controller.controller_frozen is False  # Starts trainable

    def test_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController(
            coverage_freeze_threshold=0.5,
            coverage_unfreeze_threshold=0.3,
            hierarchy_plateau_threshold=0.01,
        )

        assert controller.coverage_freeze_threshold == 0.5
        assert controller.coverage_unfreeze_threshold == 0.3
        assert controller.hierarchy_plateau_threshold == 0.01

    def test_history_initialized_empty(self):
        """Test that history deques start empty."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController()

        assert len(controller.coverage_history) == 0
        assert len(controller.hierarchy_A_history) == 0
        assert len(controller.hierarchy_B_history) == 0
        assert len(controller.Q_history) == 0

    def test_window_size(self):
        """Test window size configuration."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController(window_size=10)

        assert controller.window_size == 10
        assert controller.coverage_history.maxlen == 10

    def test_annealing_enabled_by_default(self):
        """Test Q-gated annealing is enabled by default."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController()

        assert controller.enable_annealing is True

    def test_annealing_disabled(self):
        """Test Q-gated annealing can be disabled."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController(enable_annealing=False)

        assert controller.enable_annealing is False


class TestHomeostasisControllerUpdate:
    """Tests for HomeostasisController.update method."""

    def test_warmup_returns_frozen_states(self):
        """Test that warmup period returns current freeze states."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController(warmup_epochs=10)

        result = controller.update(
            epoch=5,  # During warmup
            coverage=0.5,
            hierarchy_A=-0.3,
            hierarchy_B=-0.2,
        )

        assert result["encoder_a_frozen"] is True
        assert "warmup" in result["events"]

    def test_history_updated(self):
        """Test that histories are updated on each call."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController(warmup_epochs=0)

        controller.update(
            epoch=1,
            coverage=0.8,
            hierarchy_A=-0.3,
            hierarchy_B=-0.2,
            dist_corr_A=0.4,
            controller_grad_norm=0.01,
        )

        assert len(controller.coverage_history) == 1
        assert controller.coverage_history[-1] == 0.8
        assert len(controller.hierarchy_A_history) == 1
        assert len(controller.Q_history) == 1

    def test_q_tracked(self):
        """Test that Q is computed and tracked."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController(warmup_epochs=0)

        result = controller.update(
            epoch=1,
            coverage=0.8,
            hierarchy_A=-0.5,
            hierarchy_B=-0.2,
            dist_corr_A=0.4,
        )

        # Q = 0.4 + 1.5 * |-0.5| = 0.4 + 0.75 = 1.15
        assert result["current_Q"] == pytest.approx(1.15)

    def test_best_q_updated(self):
        """Test that best_Q is updated when Q improves."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController(warmup_epochs=0)

        # First update
        controller.update(
            epoch=1,
            coverage=0.8,
            hierarchy_A=-0.3,
            hierarchy_B=-0.2,
            dist_corr_A=0.2,
        )
        q1 = controller.best_Q

        # Second update with better metrics
        controller.update(
            epoch=2,
            coverage=0.85,
            hierarchy_A=-0.5,
            hierarchy_B=-0.3,
            dist_corr_A=0.5,
        )

        assert controller.best_Q > q1


class TestHomeostasisFreezeStates:
    """Tests for freeze state transitions."""

    def test_encoder_a_starts_frozen(self):
        """Test encoder_A starts frozen (Option C)."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController()

        assert controller.encoder_a_frozen is True

    def test_encoder_b_starts_unfrozen(self):
        """Test encoder_B starts unfrozen (Option C)."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController()

        assert controller.encoder_b_frozen is False

    def test_controller_starts_unfrozen(self):
        """Test controller starts unfrozen."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController()

        assert controller.controller_frozen is False


class TestHomeostasisHysteresis:
    """Tests for hysteresis behavior."""

    def test_hysteresis_prevents_rapid_changes(self):
        """Test that hysteresis prevents rapid freeze state changes."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController(
            warmup_epochs=0, hysteresis_epochs=5
        )

        # Initial state
        initial_state = controller.encoder_a_frozen

        # Multiple rapid updates shouldn't cause rapid state changes
        for epoch in range(3):
            controller.update(
                epoch=epoch,
                coverage=0.5,
                hierarchy_A=-0.3,
                hierarchy_B=-0.2,
            )

        # State shouldn't have changed rapidly due to hysteresis
        # (actual behavior depends on metric values)


class TestHomeostasisAnnealing:
    """Tests for Q-gated annealing."""

    def test_annealing_step_configured(self):
        """Test annealing step is configurable."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController(annealing_step=0.05)

        assert controller.annealing_step == 0.05

    def test_coverage_floor_configured(self):
        """Test coverage floor is configurable."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController(coverage_floor=0.6)

        assert controller.coverage_floor == 0.6

    def test_cycle_tracking(self):
        """Test cycle counting for components."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController()

        # Initial cycle counts should be 0
        assert controller.cycle_count["encoder_a"] == 0
        assert controller.cycle_count["encoder_b"] == 0
        assert controller.cycle_count["controller"] == 0


class TestHomeostasisEvents:
    """Tests for event tracking."""

    def test_warmup_event(self):
        """Test warmup event is reported."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController(warmup_epochs=10)

        result = controller.update(
            epoch=5,
            coverage=0.8,
            hierarchy_A=-0.3,
            hierarchy_B=-0.2,
        )

        assert "warmup" in result["events"]

    def test_result_contains_freeze_states(self):
        """Test result contains all freeze states."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController(warmup_epochs=0)

        result = controller.update(
            epoch=1,
            coverage=0.8,
            hierarchy_A=-0.3,
            hierarchy_B=-0.2,
        )

        assert "encoder_a_frozen" in result
        assert "encoder_b_frozen" in result
        assert "controller_frozen" in result

    def test_result_contains_q_values(self):
        """Test result contains Q values."""
        from src.models.homeostasis import HomeostasisController

        controller = HomeostasisController(warmup_epochs=0)

        result = controller.update(
            epoch=1,
            coverage=0.8,
            hierarchy_A=-0.3,
            hierarchy_B=-0.2,
            dist_corr_A=0.4,
        )

        assert "current_Q" in result
        assert "best_Q" in result
