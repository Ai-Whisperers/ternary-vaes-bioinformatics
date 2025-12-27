import torch

from src.models.spectral_encoder import SpectralGraphEncoder


# Note: Uses cpu_device fixture from conftest.py to avoid geoopt device mismatch


def test_spectral_laplacian_shape(cpu_device):
    encoder = SpectralGraphEncoder(hidden_dim=4)
    encoder.to(cpu_device)
    # Batch=2, Nodes=5
    adj = torch.rand(2, 5, 5, device=cpu_device)
    # Symmetrize
    adj = (adj + adj.transpose(1, 2)) / 2

    L = encoder.compute_laplacian(adj)
    assert L.shape == (2, 5, 5)


def test_forward_pass_dims(cpu_device):
    encoder = SpectralGraphEncoder(hidden_dim=4, curvature=1.0)
    encoder.to(cpu_device)
    # Batch=2, Nodes=10
    adj = torch.rand(2, 10, 10, device=cpu_device)
    adj = (adj + adj.transpose(1, 2)) / 2

    output = encoder(adj)

    # Should perform eigendecomposition, pooling, and projection
    assert output.shape == (2, 4)

    # Check Poincaré norm constraint (default 0.95)
    # Note: geoopt/poincare projection ensures this
    assert torch.all(torch.norm(output, dim=-1) < 1.0)


def test_forward_padding(cpu_device):
    # If nodes < hidden_dim
    encoder = SpectralGraphEncoder(hidden_dim=8)
    encoder.to(cpu_device)
    # Batch=1, Nodes=4 (less than 8)
    adj = torch.rand(1, 4, 4, device=cpu_device)
    adj = (adj + adj.transpose(1, 2)) / 2

    output = encoder(adj)
    assert output.shape == (1, 8)
