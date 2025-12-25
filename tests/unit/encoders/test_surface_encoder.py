# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for MaSIF-style surface encoder."""

import pytest
import torch

from src.encoders.surface_encoder import (
    GeodesicConv,
    MaSIFEncoder,
    PAdicSurfaceAttention,
    SurfaceComplementarity,
    SurfaceEncoderOutput,
    SurfaceFeatureExtractor,
    SurfaceInteractionPredictor,
    SurfacePatchEncoder,
)


class TestGeodesicConv:
    """Tests for GeodesicConv layer."""

    def test_creation(self):
        """Test geodesic convolution creation."""
        layer = GeodesicConv(in_channels=16, out_channels=32)
        assert layer.in_channels == 16
        assert layer.out_channels == 32

    def test_forward(self):
        """Test forward pass."""
        layer = GeodesicConv(in_channels=16, out_channels=32)

        features = torch.randn(2, 20, 16)
        geodesic_coords = torch.rand(2, 20, 2)  # [radial, angular]

        output = layer(features, geodesic_coords)

        assert output.shape == (2, 20, 32)

    def test_different_ring_orientations(self):
        """Test with different ring/orientation settings."""
        layer = GeodesicConv(in_channels=16, out_channels=32, n_rings=3, n_orientations=4)

        features = torch.randn(2, 20, 16)
        geodesic_coords = torch.rand(2, 20, 2)

        output = layer(features, geodesic_coords)

        assert output.shape == (2, 20, 32)


class TestSurfaceFeatureExtractor:
    """Tests for SurfaceFeatureExtractor."""

    def test_creation(self):
        """Test extractor creation."""
        extractor = SurfaceFeatureExtractor(feature_dim=16)
        assert extractor.feature_dim == 16

    def test_extract_residue_features(self):
        """Test single residue feature extraction."""
        extractor = SurfaceFeatureExtractor()

        features = extractor.extract_residue_features("A")
        assert features.shape == (4,)

        features_w = extractor.extract_residue_features("W")
        assert features_w.shape == (4,)

    def test_forward(self):
        """Test forward pass."""
        extractor = SurfaceFeatureExtractor(feature_dim=16)

        residue_features = torch.randn(2, 20, 3)  # hydro, charge, volume
        curvatures = torch.randn(2, 20)

        output = extractor(residue_features, curvatures)

        assert output.shape == (2, 20, 16)


class TestPAdicSurfaceAttention:
    """Tests for PAdicSurfaceAttention."""

    def test_creation(self):
        """Test attention creation."""
        attn = PAdicSurfaceAttention(embed_dim=64, n_heads=4, p=3)
        assert attn.embed_dim == 64
        assert attn.n_heads == 4

    def test_forward(self):
        """Test forward pass."""
        attn = PAdicSurfaceAttention(embed_dim=64, n_heads=4)

        x = torch.randn(2, 20, 64)
        output, weights = attn(x)

        assert output.shape == (2, 20, 64)
        assert weights.shape == (2, 20, 20)

    def test_padic_distances(self):
        """Test p-adic distance computation."""
        attn = PAdicSurfaceAttention(embed_dim=64, p=3)

        distances = attn.compute_padic_distances(10, torch.device("cpu"))

        assert distances.shape == (10, 10)
        # Diagonal should be zero
        assert torch.allclose(torch.diag(distances), torch.zeros(10))

    def test_with_precomputed_bias(self):
        """Test with precomputed p-adic bias."""
        attn = PAdicSurfaceAttention(embed_dim=64, n_heads=4)

        x = torch.randn(2, 20, 64)
        padic_bias = torch.rand(20, 20)

        output, weights = attn(x, padic_bias)

        assert output.shape == (2, 20, 64)


class TestSurfacePatchEncoder:
    """Tests for SurfacePatchEncoder."""

    def test_creation(self):
        """Test encoder creation."""
        encoder = SurfacePatchEncoder(input_dim=16, output_dim=32)
        assert encoder.input_dim == 16
        assert encoder.output_dim == 32

    def test_forward(self):
        """Test forward pass."""
        encoder = SurfacePatchEncoder(input_dim=16, output_dim=32)

        features = torch.randn(2, 30, 16)
        geodesic_coords = torch.rand(2, 30, 2)

        output = encoder(features, geodesic_coords)

        assert output.shape == (2, 32)


class TestMaSIFEncoder:
    """Tests for MaSIFEncoder."""

    def test_creation(self):
        """Test encoder creation."""
        encoder = MaSIFEncoder(feature_dim=16, patch_dim=64, output_dim=128)
        assert encoder.feature_dim == 16
        assert encoder.output_dim == 128

    def test_forward(self):
        """Test forward pass."""
        encoder = MaSIFEncoder(feature_dim=16, patch_dim=64, output_dim=128)

        # (batch, n_patches, n_points, feat_dim)
        patch_features = torch.randn(2, 10, 20, 4)  # hydro, charge, volume, curvature
        patch_geodesics = torch.rand(2, 10, 20, 2)

        output = encoder(patch_features, patch_geodesics)

        assert isinstance(output, SurfaceEncoderOutput)
        assert output.surface_embedding.shape == (2, 128)
        assert output.patch_embeddings.shape == (2, 10, 64)
        assert output.attention_weights.shape == (2, 10)

    def test_forward_with_curvatures(self):
        """Test forward with explicit curvatures."""
        encoder = MaSIFEncoder()

        patch_features = torch.randn(2, 5, 15, 4)
        patch_geodesics = torch.rand(2, 5, 15, 2)
        curvatures = torch.randn(2, 5, 15)

        output = encoder(patch_features, patch_geodesics, curvatures)

        assert output.surface_embedding.shape[0] == 2

    def test_without_padic(self):
        """Test encoder without p-adic attention."""
        encoder = MaSIFEncoder(use_padic=False)

        patch_features = torch.randn(2, 5, 15, 4)
        patch_geodesics = torch.rand(2, 5, 15, 2)

        output = encoder(patch_features, patch_geodesics)

        assert output.padic_structure is None


class TestSurfaceInteractionPredictor:
    """Tests for SurfaceInteractionPredictor."""

    def test_creation(self):
        """Test predictor creation."""
        predictor = SurfaceInteractionPredictor(embed_dim=128)
        assert predictor.embed_dim == 128

    def test_forward(self):
        """Test forward pass."""
        predictor = SurfaceInteractionPredictor(embed_dim=128)

        surface1 = torch.randn(2, 128)
        surface2 = torch.randn(2, 128)

        result = predictor(surface1, surface2)

        assert "interaction_score" in result
        assert "binding_probs_1" in result
        assert "binding_probs_2" in result
        assert result["interaction_score"].shape == (2,)

    def test_binding_probabilities(self):
        """Test that binding probabilities are valid."""
        predictor = SurfaceInteractionPredictor(embed_dim=64)

        surface1 = torch.randn(2, 64)
        surface2 = torch.randn(2, 64)

        result = predictor(surface1, surface2)

        # Probabilities should be in [0, 1]
        assert (result["binding_probs_1"] >= 0).all()
        assert (result["binding_probs_1"] <= 1).all()


class TestSurfaceComplementarity:
    """Tests for SurfaceComplementarity."""

    def test_creation(self):
        """Test analyzer creation."""
        analyzer = SurfaceComplementarity(p=3)
        assert analyzer.p == 3

    def test_shape_complementarity(self):
        """Test shape complementarity calculation."""
        analyzer = SurfaceComplementarity()

        # Opposite curvatures should be complementary
        curv1 = torch.tensor([1.0, 0.5, -0.5])
        curv2 = torch.tensor([-1.0, -0.5, 0.5])

        score = analyzer.shape_complementarity(curv1, curv2)

        # Negative correlation -> high complementarity
        assert 0 <= score <= 1

    def test_chemical_complementarity(self):
        """Test chemical complementarity calculation."""
        analyzer = SurfaceComplementarity()

        # Features: [hydrophobicity, charge, ...]
        feat1 = torch.tensor([[0.5, 1.0, 0.3, 0.0]])  # positive charge
        feat2 = torch.tensor([[-0.5, -1.0, 0.3, 0.0]])  # negative charge

        score = analyzer.chemical_complementarity(feat1, feat2)

        # Opposite charges -> complementary
        assert 0 <= score <= 1

    def test_forward(self):
        """Test combined analysis."""
        analyzer = SurfaceComplementarity()

        surface1 = {
            "curvatures": torch.randn(20),
            "features": torch.randn(20, 4),
        }
        surface2 = {
            "curvatures": torch.randn(20),
            "features": torch.randn(20, 4),
        }

        result = analyzer(surface1, surface2)

        assert "shape_complementarity" in result
        assert "chemical_complementarity" in result
        assert "overall_complementarity" in result
