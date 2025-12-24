from .base import BaseFactory


class ModelConfigFactory(BaseFactory):
    """Generates configuration dictionaries for TernaryVAEV5_11."""

    @classmethod
    def build(cls, **kwargs):
        """
        Returns a config dict with sensible defaults for testing.
        Overridable via kwargs.
        """
        defaults = {
            "latent_dim": 16,
            "hidden_dim": 32,  # Small for tests
            "max_radius": 0.95,
            "curvature": 1.0,
            "use_controller": True,
            "use_dual_projection": False,
            "n_projection_layers": 1,
            "projection_dropout": 0.0,
            "learnable_curvature": False,
        }
        defaults.update(kwargs)
        return defaults

    @classmethod
    def minimal(cls):
        """Minimal config for fast unit tests."""
        return cls.build(hidden_dim=8, use_controller=False)
