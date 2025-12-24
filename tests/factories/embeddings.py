import torch
from .base import BaseFactory


class PoincareEmbeddingFactory(BaseFactory):
    """Generates valid embeddings on the Poincare disk."""

    @classmethod
    def build(cls, batch_size=32, dim=16, radius=0.9, device="cpu"):
        """
        Generates random points within the Poincare ball.
        radius < 1.0 ensures they are strictly inside.
        """
        # Generate random directions
        directions = torch.randn(batch_size, dim, device=device)
        directions = directions / directions.norm(dim=-1, keepdim=True)

        # Generate random radii uniformly in [0, radius]
        # (Technically uniform in volume would be r^d, but linear is fine for testing)
        radii = torch.rand(batch_size, 1, device=device) * radius

        return directions * radii

    @classmethod
    def create_batch(cls, size, dim=16, device="cpu"):
        return cls.build(batch_size=size, dim=dim, device=device)
