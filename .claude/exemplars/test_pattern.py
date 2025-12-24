import pytest
import torch
from src.modules.exemplar import ExemplarModule


@pytest.fixture
def sample_input():
    # Deterministic input for reproducibility
    torch.manual_seed(42)
    return torch.randn(4, 10, 64)  # [Batch, Seq, Dim]


def test_forward_shape(sample_input):
    """Verify output shape matches input shape."""
    model = ExemplarModule(input_dim=64, hidden_dim=128)
    output = model(sample_input)
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any(), "Output contains NaNs"


def test_device_agnosticism():
    """Ensure model runs on CPU (default)."""
    model = ExemplarModule(input_dim=16, hidden_dim=32)
    x = torch.randn(2, 5, 16)
    _ = model(x)
