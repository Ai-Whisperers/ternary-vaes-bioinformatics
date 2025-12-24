import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pytest
import torch
from src.geometry.poincare import get_manifold


@pytest.fixture(scope="session")
def device():
    """Returns 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def poincare():
    """Returns a PoincareBall manifold instance."""
    return get_manifold(c=1.0)


@pytest.fixture(scope="session")
def ternary_ops(device):
    """Returns a batch of valid ternary operations for testing."""
    # Mock data: (Batch, 9)
    # Just a small standard batch
    return torch.randint(-1, 2, (32, 9)).float().to(device)
