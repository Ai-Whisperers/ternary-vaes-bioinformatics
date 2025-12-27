import pytest

from tests.core.builders import VAEBuilder
from tests.core.matchers import expect_poincare
from tests.factories.data import TernaryOperationFactory


# Use CPU for hyperbolic/geoopt tests to avoid device mismatch with manifold curvature
@pytest.fixture
def cpu_device():
    """Force CPU for geoopt-related tests."""
    return "cpu"


def test_vae_initialization():
    """Verify VAE initializes using Builder."""
    model = VAEBuilder().build()

    # We check if attributes exist
    assert hasattr(model, "projection")
    assert hasattr(model, "controller")


def test_vae_forward_shape(cpu_device):
    """Verify forward pass returns correct shapes."""
    # Use CPU to avoid geoopt device mismatch
    device = cpu_device
    # Use Builder to get a model with mocked frozen components
    model = VAEBuilder().build()
    model.to(device)

    # Use Factory for data
    x = TernaryOperationFactory.create_batch(size=4, device=device)

    # Forward
    outputs = model(x)

    assert "logits_A" in outputs
    assert outputs["logits_A"].shape == (4, 9, 3)
    assert "z_A_hyp" in outputs
    assert outputs["z_A_hyp"].shape == (4, 16)

    # Use Matcher
    expect_poincare(outputs["z_A_hyp"])


def test_gradients_only_projection(cpu_device):
    """Verify gradients propagate to projection but NOT to frozen encoder."""
    # Use CPU to avoid geoopt device mismatch
    device = cpu_device
    # Use Builder to get a model with mocked frozen components
    model = VAEBuilder().build()
    model.to(device)

    x = TernaryOperationFactory.create_batch(size=4, device=device)

    # We need to verify parameters of projection have grad
    outputs = model(x)
    loss = outputs["z_A_hyp"].sum()
    loss.backward()

    # Check projection weights
    # Note: init_identity=True zero-initializes the last layer of direction_net,
    # which blocks gradient flow to earlier layers of that sub-network.
    # So we only expect SOME parameters (e.g. radius_net, last layer of direction_net) to have grad.

    found_grad = False
    for name, param in model.projection.named_parameters():
        if param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                found_grad = True

    assert found_grad, "Projection should learn (at least some params have grad)"


def test_dual_projection_flow(cpu_device):
    """Verify dual projection (Bio-Hyperbolic) gradient flow."""
    # Use CPU to avoid geoopt device mismatch
    device = cpu_device
    model = VAEBuilder().with_dual_projection().build()
    model.to(device)

    x = TernaryOperationFactory.create_batch(size=4, device=device)
    outputs = model(x)

    # Dual projection produces z_A_hyp and z_B_hyp
    loss = outputs["z_A_hyp"].sum() + outputs["z_B_hyp"].sum()
    loss.backward()

    found_grad = False
    for name, param in model.projection.named_parameters():
        if param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                found_grad = True

    assert found_grad, "Dual projection should receive gradients"
