import numpy as np
import torch

from src.data.generation import (count_ternary_operations,
                                 generate_ternary_operation_by_index)

from .base import BaseFactory


class TernaryOperationFactory(BaseFactory):
    """Generates valid ternary operation tensors representing Z3^9 functions."""

    @classmethod
    def build(cls, batch_size=32, device="cpu"):
        """
        Returns a batch of mathematically valid ternary operations.
        Shape: (batch_size, 9)
        Values: {-1, 0, 1}
        """
        total_ops = count_ternary_operations()

        # Sample random indices
        indices = np.random.randint(0, total_ops, size=batch_size)

        # Generate operations
        ops = [generate_ternary_operation_by_index(idx) for idx in indices]

        return torch.tensor(ops, dtype=torch.float32).to(device)

    @classmethod
    def create_batch(cls, size, device="cpu"):
        return cls.build(batch_size=size, device=device)

    @classmethod
    def all_operations(cls, device="cpu"):
        """Returns all 19,683 operations (useful for coverage tests)."""
        from src.data.generation import generate_all_ternary_operations

        ops = generate_all_ternary_operations()
        return torch.tensor(ops, dtype=torch.float32).to(device)
