"""Comprehensive tests for src/geometry/poincare.py - Poincare ball geometry.

This module tests hyperbolic geometry operations using geoopt backend.
"""

import pytest
import torch
import torch.testing as testing
from src.geometry.poincare import (
    get_manifold,
    poincare_distance,
    poincare_distance_matrix,
    project_to_poincare,
    exp_map_zero,
    log_map_zero,
    mobius_add,
    lambda_x,
    parallel_transport,
    PoincareModule,
    create_manifold_parameter,
    create_manifold_tensor,
    get_riemannian_optimizer,
    _manifold_cache,
)
from geoopt import ManifoldParameter, ManifoldTensor


class TestGetManifold:
    """Test manifold creation and caching."""

    def test_default_curvature(self):
        """Default curvature should be 1.0."""
        manifold = get_manifold()
        # manifold.c is a tensor in geoopt
        assert float(manifold.c) == pytest.approx(1.0)

    def test_custom_curvature(self):
        """Custom curvature should be respected."""
        manifold = get_manifold(c=0.5)
        # manifold.c is a tensor in geoopt
        assert float(manifold.c) == pytest.approx(0.5)

    def test_caching(self):
        """Same curvature should return cached manifold."""
        m1 = get_manifold(c=1.0)
        m2 = get_manifold(c=1.0)
        assert m1 is m2

    def test_different_curvatures_cached_separately(self):
        """Different curvatures should create different manifolds."""
        m1 = get_manifold(c=1.0)
        m2 = get_manifold(c=2.0)
        assert m1 is not m2
        assert m1.c != m2.c


class TestPoincareDistance:
    """Test Poincare distance computation."""

    def test_distance_to_self_is_zero(self):
        """d(x, x) = 0."""
        x = torch.tensor([[0.5, 0.3]], dtype=torch.float32)
        x = project_to_poincare(x)
        d = poincare_distance(x, x)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)

    def test_distance_symmetry(self):
        """d(x, y) = d(y, x)."""
        x = project_to_poincare(torch.randn(5, 4) * 0.5)
        y = project_to_poincare(torch.randn(5, 4) * 0.5)
        d_xy = poincare_distance(x, y)
        d_yx = poincare_distance(y, x)
        assert torch.allclose(d_xy, d_yx, atol=1e-5)

    def test_distance_non_negative(self):
        """Distance should always be non-negative."""
        x = project_to_poincare(torch.randn(10, 8) * 0.5)
        y = project_to_poincare(torch.randn(10, 8) * 0.5)
        d = poincare_distance(x, y)
        assert (d >= 0).all()

    def test_distance_keepdim(self):
        """Test keepdim parameter."""
        x = project_to_poincare(torch.randn(5, 4) * 0.5)
        y = project_to_poincare(torch.randn(5, 4) * 0.5)

        d_flat = poincare_distance(x, y, keepdim=False)
        d_keep = poincare_distance(x, y, keepdim=True)

        assert d_flat.shape == (5,)
        assert d_keep.shape == (5, 1)

    def test_distance_different_curvature(self):
        """Distance with different curvature."""
        x = project_to_poincare(torch.randn(3, 2) * 0.3)
        y = project_to_poincare(torch.randn(3, 2) * 0.3)

        d1 = poincare_distance(x, y, c=1.0)
        d2 = poincare_distance(x, y, c=2.0)

        # Different curvatures should give different distances
        assert not torch.allclose(d1, d2)


class TestProjectToPoincare:
    """Test projection onto Poincare ball."""

    def test_projection_constrains_norm(self):
        """Projected points should have norm <= max_norm."""
        x = torch.randn(20, 5) * 10  # Large norms
        x_proj = project_to_poincare(x, max_norm=0.95)

        norms = x_proj.norm(dim=-1)
        assert (norms <= 0.95 + 1e-5).all()

    def test_projection_preserves_small_norms(self):
        """Small norm points should be mostly preserved."""
        x = torch.randn(10, 4) * 0.1  # Small norms
        x_proj = project_to_poincare(x, max_norm=0.95)

        # Should be close to original
        assert torch.allclose(x, x_proj, atol=0.1)

    def test_projection_different_max_norm(self):
        """Different max_norm values."""
        x = torch.randn(10, 4) * 2

        x_50 = project_to_poincare(x, max_norm=0.50)
        x_90 = project_to_poincare(x, max_norm=0.90)

        assert (x_50.norm(dim=-1) <= 0.50 + 1e-5).all()
        assert (x_90.norm(dim=-1) <= 0.90 + 1e-5).all()


class TestExpMapZero:
    """Test exponential map from origin."""

    def test_zero_vector_maps_to_origin(self):
        """exp_0(0) = 0."""
        v = torch.zeros(5, 4)
        z = exp_map_zero(v)
        assert torch.allclose(z, torch.zeros_like(z), atol=1e-6)

    def test_result_on_manifold(self):
        """Result should be on the Poincare ball (norm < 1)."""
        v = torch.randn(10, 4)
        z = exp_map_zero(v)
        norms = z.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_different_curvature(self):
        """Test with different curvature values."""
        v = torch.randn(5, 3) * 0.5
        z1 = exp_map_zero(v, c=1.0)
        z2 = exp_map_zero(v, c=2.0)

        # Results should differ
        assert not torch.allclose(z1, z2)


class TestLogMapZero:
    """Test logarithmic map to origin."""

    def test_origin_maps_to_zero(self):
        """log_0(0) = 0."""
        z = torch.zeros(5, 4)
        v = log_map_zero(z)
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)

    def test_inverse_of_exp_map(self):
        """log_0(exp_0(v)) ≈ v for small v."""
        v = torch.randn(5, 4) * 0.3
        z = exp_map_zero(v)
        v_recovered = log_map_zero(z)

        # Should approximately recover original (with some numerical error)
        assert torch.allclose(v, v_recovered, atol=1e-4)


class TestMobiusAdd:
    """Test Mobius addition."""

    def test_add_zero(self):
        """x (+) 0 = x."""
        x = project_to_poincare(torch.randn(5, 4) * 0.5)
        zero = torch.zeros_like(x)
        result = mobius_add(x, zero)
        assert torch.allclose(result, x, atol=1e-5)

    def test_zero_add_x(self):
        """0 (+) x = x."""
        x = project_to_poincare(torch.randn(5, 4) * 0.5)
        zero = torch.zeros_like(x)
        result = mobius_add(zero, x)
        assert torch.allclose(result, x, atol=1e-5)

    def test_result_on_manifold(self):
        """Result should be on the Poincare ball."""
        x = project_to_poincare(torch.randn(10, 4) * 0.5)
        y = project_to_poincare(torch.randn(10, 4) * 0.5)
        result = mobius_add(x, y)

        norms = result.norm(dim=-1)
        assert (norms < 1.0).all()


class TestLambdaX:
    """Test conformal factor computation."""

    def test_origin_conformal_factor(self):
        """λ(0) = 2 / (1 - 0) = 2."""
        x = torch.zeros(5, 4)
        lam = lambda_x(x, c=1.0)
        expected = torch.full((5, 1), 2.0)
        assert torch.allclose(lam, expected, atol=1e-5)

    def test_conformal_factor_positive(self):
        """Conformal factor should always be positive."""
        x = project_to_poincare(torch.randn(10, 4) * 0.5)
        lam = lambda_x(x)
        assert (lam > 0).all()

    def test_conformal_factor_increases_near_boundary(self):
        """Conformal factor increases as we approach boundary."""
        x_small = project_to_poincare(torch.randn(5, 4) * 0.1)
        x_large = project_to_poincare(torch.randn(5, 4) * 0.9)

        lam_small = lambda_x(x_small).mean()
        lam_large = lambda_x(x_large).mean()

        assert lam_large > lam_small

    def test_keepdim(self):
        """Test keepdim parameter."""
        x = project_to_poincare(torch.randn(5, 4) * 0.5)

        lam_keep = lambda_x(x, keepdim=True)
        lam_flat = lambda_x(x, keepdim=False)

        assert lam_keep.shape == (5, 1)
        assert lam_flat.shape == (5,)


class TestParallelTransport:
    """Test parallel transport of tangent vectors."""

    def test_transport_to_same_point(self):
        """Transport from x to x should preserve the vector."""
        x = project_to_poincare(torch.randn(5, 4) * 0.5)
        v = torch.randn(5, 4) * 0.1

        v_transported = parallel_transport(x, x, v)
        assert torch.allclose(v_transported, v, atol=1e-4)

    def test_transport_returns_valid_tensor(self):
        """Parallel transport returns a valid tensor with correct shape."""
        x = project_to_poincare(torch.randn(5, 4) * 0.3)
        y = project_to_poincare(torch.randn(5, 4) * 0.3)
        v = torch.randn(5, 4) * 0.1

        v_transported = parallel_transport(x, y, v)

        # Check output is valid and has correct shape
        assert v_transported.shape == v.shape
        assert torch.isfinite(v_transported).all()

        # In hyperbolic space, transported norms can differ significantly
        # but should still be finite and non-zero for non-zero input
        trans_norms = v_transported.norm(dim=-1)
        assert (trans_norms > 0).all()


class TestPoincareModule:
    """Test PoincareModule base class."""

    def test_initialization(self):
        """Test module initialization."""
        module = PoincareModule(c=1.5, max_norm=0.9)
        assert module.c == 1.5
        assert module.max_norm == 0.9
        assert module.manifold is not None

    def test_dist_method(self):
        """Test dist method."""
        module = PoincareModule(c=1.0)
        x = project_to_poincare(torch.randn(5, 4) * 0.5)
        y = project_to_poincare(torch.randn(5, 4) * 0.5)

        d = module.dist(x, y)
        assert d.shape == (5,)

    def test_proj_method(self):
        """Test proj method."""
        module = PoincareModule(c=1.0, max_norm=0.9)
        x = torch.randn(5, 4) * 10

        x_proj = module.proj(x)
        norms = x_proj.norm(dim=-1)
        assert (norms <= 0.9 + 1e-5).all()

    def test_expmap0_method(self):
        """Test expmap0 method."""
        module = PoincareModule()
        v = torch.randn(5, 4) * 0.5
        z = module.expmap0(v)
        assert z.shape == v.shape

    def test_logmap0_method(self):
        """Test logmap0 method."""
        module = PoincareModule()
        z = project_to_poincare(torch.randn(5, 4) * 0.5)
        v = module.logmap0(z)
        assert v.shape == z.shape

    def test_add_method(self):
        """Test add (Mobius addition) method."""
        module = PoincareModule()
        x = project_to_poincare(torch.randn(5, 4) * 0.5)
        y = project_to_poincare(torch.randn(5, 4) * 0.5)
        result = module.add(x, y)
        assert result.shape == x.shape

    def test_conformal_method(self):
        """Test conformal factor method."""
        module = PoincareModule()
        x = project_to_poincare(torch.randn(5, 4) * 0.5)
        lam = module.conformal(x)
        assert lam.shape == (5, 1)

    def test_transport_method(self):
        """Test parallel transport method."""
        module = PoincareModule()
        x = project_to_poincare(torch.randn(5, 4) * 0.3)
        y = project_to_poincare(torch.randn(5, 4) * 0.3)
        v = torch.randn(5, 4) * 0.1
        result = module.transport(x, y, v)
        assert result.shape == v.shape


class TestManifoldParameters:
    """Test manifold parameter creation."""

    def test_create_manifold_parameter(self):
        """Test creating a learnable manifold parameter."""
        data = torch.randn(10, 4)
        param = create_manifold_parameter(data, c=1.0, requires_grad=True)

        assert isinstance(param, ManifoldParameter)
        assert param.requires_grad
        # Data should be projected onto manifold
        assert (param.data.norm(dim=-1) < 1.0).all()

    def test_create_manifold_parameter_no_grad(self):
        """Test creating parameter without gradients."""
        data = torch.randn(10, 4)
        param = create_manifold_parameter(data, c=1.0, requires_grad=False)

        assert not param.requires_grad

    def test_create_manifold_tensor(self):
        """Test creating a non-learnable manifold tensor."""
        data = torch.randn(10, 4)
        tensor = create_manifold_tensor(data, c=1.0)

        assert isinstance(tensor, ManifoldTensor)
        # Data should be projected onto manifold
        assert (tensor.data.norm(dim=-1) < 1.0).all()


class TestRiemannianOptimizer:
    """Test Riemannian optimizer creation."""

    def test_create_adam_optimizer(self):
        """Test creating RiemannianAdam optimizer."""
        params = [torch.nn.Parameter(torch.randn(10, 4))]
        opt = get_riemannian_optimizer(params, lr=1e-3, optimizer_type='adam')

        assert opt is not None
        assert opt.defaults['lr'] == 1e-3

    def test_create_sgd_optimizer(self):
        """Test creating RiemannianSGD optimizer."""
        params = [torch.nn.Parameter(torch.randn(10, 4))]
        opt = get_riemannian_optimizer(params, lr=1e-2, optimizer_type='sgd')

        assert opt is not None
        assert opt.defaults['lr'] == 1e-2

    def test_invalid_optimizer_type(self):
        """Invalid optimizer type should raise error."""
        params = [torch.nn.Parameter(torch.randn(10, 4))]

        with pytest.raises(ValueError):
            get_riemannian_optimizer(params, optimizer_type='invalid')


class TestPoincareDistanceMatrix:
    """Test pairwise distance matrix computation."""

    def test_distance_matrix_shape(self):
        """Distance matrix should be n x n."""
        z = project_to_poincare(torch.randn(10, 4) * 0.5)
        D = poincare_distance_matrix(z)
        assert D.shape == (10, 10)

    def test_distance_matrix_diagonal_zero(self):
        """Diagonal should be zero (d(x, x) = 0)."""
        z = project_to_poincare(torch.randn(10, 4) * 0.5)
        D = poincare_distance_matrix(z)
        diag = D.diag()
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-5)

    def test_distance_matrix_symmetric(self):
        """Distance matrix should be symmetric."""
        z = project_to_poincare(torch.randn(10, 4) * 0.5)
        D = poincare_distance_matrix(z)
        assert torch.allclose(D, D.T, atol=1e-5)

    def test_distance_matrix_non_negative(self):
        """All distances should be non-negative."""
        z = project_to_poincare(torch.randn(10, 4) * 0.5)
        D = poincare_distance_matrix(z)
        assert (D >= -1e-6).all()

    def test_distance_matrix_curvature(self):
        """Different curvatures should give different matrices."""
        z = project_to_poincare(torch.randn(5, 4) * 0.3)
        D1 = poincare_distance_matrix(z, c=1.0)
        D2 = poincare_distance_matrix(z, c=2.0)
        assert not torch.allclose(D1, D2)


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_near_boundary_distance(self):
        """Test distance computation near ball boundary."""
        # Points very close to boundary
        x = torch.tensor([[0.99, 0.0]])
        y = torch.tensor([[0.0, 0.99]])

        d = poincare_distance(x, y)
        assert torch.isfinite(d).all()
        assert d.item() > 0

    def test_origin_operations(self):
        """Test all operations at origin."""
        origin = torch.zeros(1, 4)

        # Distance from origin to origin
        d = poincare_distance(origin, origin)
        assert d.item() == pytest.approx(0, abs=1e-6)

        # Projection of origin
        proj = project_to_poincare(origin)
        assert torch.allclose(proj, origin)

        # Conformal factor at origin
        lam = lambda_x(origin)
        assert lam.item() == pytest.approx(2.0, rel=1e-4)

    def test_batch_operations(self):
        """Test operations on larger batches."""
        batch_size = 1000
        dim = 16

        x = project_to_poincare(torch.randn(batch_size, dim) * 0.5)
        y = project_to_poincare(torch.randn(batch_size, dim) * 0.5)

        # All operations should handle large batches
        d = poincare_distance(x, y)
        assert d.shape == (batch_size,)

        lam = lambda_x(x)
        assert lam.shape == (batch_size, 1)

        result = mobius_add(x, y)
        assert result.shape == (batch_size, dim)

    def test_single_point(self):
        """Test operations on single points."""
        x = project_to_poincare(torch.randn(1, 4) * 0.5)
        y = project_to_poincare(torch.randn(1, 4) * 0.5)

        d = poincare_distance(x, y)
        assert d.shape == (1,)

        D = poincare_distance_matrix(x)
        assert D.shape == (1, 1)

    def test_high_dimensional(self):
        """Test operations in high dimensions."""
        dim = 128
        x = project_to_poincare(torch.randn(10, dim) * 0.5)
        y = project_to_poincare(torch.randn(10, dim) * 0.5)

        d = poincare_distance(x, y)
        assert d.shape == (10,)
        assert torch.isfinite(d).all()
