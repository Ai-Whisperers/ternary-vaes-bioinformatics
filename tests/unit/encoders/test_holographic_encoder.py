# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for HolographicEncoder module."""

import pytest
import torch

from src.encoders.holographic_encoder import (
    GraphLaplacianEncoder,
    HierarchicalProteinEmbedding,
    HolographicEncoder,
    MultiScaleGraphFeatures,
    PPINetworkEncoder,
)


class TestGraphLaplacianEncoder:
    """Tests for GraphLaplacianEncoder."""

    def test_creation(self):
        """Test encoder creation."""
        encoder = GraphLaplacianEncoder(n_eigenvectors=8)
        assert encoder.n_eigenvectors == 8

    def test_compute_laplacian(self):
        """Test Laplacian computation."""
        encoder = GraphLaplacianEncoder()

        # Simple adjacency matrix
        adj = torch.tensor([
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        ], dtype=torch.float32)

        L = encoder.compute_laplacian(adj)

        assert L.shape == (1, 3, 3)
        # Laplacian is symmetric
        assert torch.allclose(L, L.transpose(-1, -2), atol=1e-5)

    def test_forward(self):
        """Test forward pass."""
        encoder = GraphLaplacianEncoder(n_eigenvectors=4)

        adj = torch.randn(2, 10, 10).abs()
        adj = (adj + adj.transpose(-1, -2)) / 2  # Symmetrize

        eigenvalues, eigenvectors = encoder(adj)

        assert eigenvalues.shape[0] == 2
        assert eigenvectors.shape == (2, 10, min(4, 9))


class TestMultiScaleGraphFeatures:
    """Tests for MultiScaleGraphFeatures."""

    def test_creation(self):
        """Test creation."""
        extractor = MultiScaleGraphFeatures(hidden_dim=64, n_scales=3)
        assert extractor.n_scales == 3

    def test_forward(self):
        """Test forward pass."""
        extractor = MultiScaleGraphFeatures(hidden_dim=32)

        adj = torch.randn(2, 8, 8).abs()
        adj = (adj + adj.transpose(-1, -2)) / 2

        features = extractor(adj)

        assert features.shape == (2, 32)

    def test_with_node_features(self):
        """Test with node features."""
        extractor = MultiScaleGraphFeatures(hidden_dim=32)

        adj = torch.randn(2, 8, 8).abs()
        adj = (adj + adj.transpose(-1, -2)) / 2
        node_feat = torch.randn(2, 8, 16)

        features = extractor(adj, node_feat)

        assert features.shape == (2, 32)


class TestHolographicEncoder:
    """Tests for HolographicEncoder."""

    def test_creation(self):
        """Test encoder creation."""
        encoder = HolographicEncoder(
            input_dim=32,
            hidden_dim=64,
            output_dim=16,
        )
        assert encoder.output_dim == 16

    def test_forward(self):
        """Test forward pass."""
        encoder = HolographicEncoder(
            input_dim=32,
            hidden_dim=64,
            output_dim=16,
        )

        adj = torch.randn(4, 20, 20).abs()
        adj = (adj + adj.transpose(-1, -2)) / 2

        result = encoder(adj)

        assert "z" in result
        assert "z_euclidean" in result
        assert "spectral_features" in result
        assert result["z"].shape == (4, 16)

    def test_output_on_poincare_ball(self):
        """Test that output is on Poincare ball."""
        encoder = HolographicEncoder(output_dim=16, max_norm=0.95)

        adj = torch.randn(4, 20, 20).abs()
        adj = (adj + adj.transpose(-1, -2)) / 2

        result = encoder(adj)

        norms = torch.norm(result["z"], dim=-1)
        assert (norms <= 0.95 + 1e-5).all()

    def test_with_node_features(self):
        """Test with node features."""
        encoder = HolographicEncoder(input_dim=32, output_dim=16)

        adj = torch.randn(4, 20, 20).abs()
        adj = (adj + adj.transpose(-1, -2)) / 2
        node_feat = torch.randn(4, 20, 32)

        result = encoder(adj, node_feat)

        assert result["z"].shape == (4, 16)

    def test_hierarchy_score(self):
        """Test hierarchy score computation."""
        encoder = HolographicEncoder(output_dim=8)

        # Points with different norms
        z = torch.tensor([
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Near origin (high hierarchy)
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Near boundary (low hierarchy)
        ])

        scores = encoder.compute_hierarchy_score(z)

        assert scores.shape == (2,)
        assert scores[0] > scores[1]  # Near origin = higher score


class TestPPINetworkEncoder:
    """Tests for PPINetworkEncoder."""

    def test_creation(self):
        """Test encoder creation."""
        encoder = PPINetworkEncoder(
            n_proteins=100,
            embedding_dim=16,
            output_dim=8,
        )
        assert encoder.n_proteins == 100

    def test_forward(self):
        """Test forward pass."""
        encoder = PPINetworkEncoder(
            n_proteins=100,
            embedding_dim=16,
            output_dim=8,
        )

        protein_ids = torch.randint(0, 100, (4, 20))
        adj = torch.randn(4, 20, 20).abs()
        adj = (adj + adj.transpose(-1, -2)) / 2

        result = encoder(protein_ids, adj)

        assert "z" in result
        assert result["z"].shape == (4, 8)

    def test_with_confidence(self):
        """Test with confidence scores."""
        encoder = PPINetworkEncoder(n_proteins=50, output_dim=8)

        protein_ids = torch.randint(0, 50, (2, 10))
        adj = torch.randn(2, 10, 10).abs()
        adj = (adj + adj.transpose(-1, -2)) / 2
        confidence = torch.rand(2, 10, 10)

        result = encoder(protein_ids, adj, confidence)

        assert result["z"].shape == (2, 8)


class TestHierarchicalProteinEmbedding:
    """Tests for HierarchicalProteinEmbedding."""

    def test_creation(self):
        """Test creation."""
        embedding = HierarchicalProteinEmbedding(
            hidden_dim=64,
            output_dim=16,
        )
        assert embedding.output_dim == 16

    def test_forward(self):
        """Test forward pass."""
        embedding = HierarchicalProteinEmbedding(output_dim=8)

        adj = torch.randn(4, 15, 15).abs()
        adj = (adj + adj.transpose(-1, -2)) / 2

        result = embedding(adj)

        assert "z" in result
        assert "hierarchy_scores" in result
        assert result["z"].shape == (4, 8)
        assert result["hierarchy_scores"].shape == (4,)

    def test_hierarchical_loss(self):
        """Test hierarchical loss computation."""
        embedding = HierarchicalProteinEmbedding(output_dim=8)

        z = torch.randn(4, 8) * 0.5
        hierarchy_labels = torch.tensor([0.1, 0.3, 0.7, 0.9])

        loss = embedding.compute_hierarchical_loss(z, hierarchy_labels)

        assert loss.shape == ()
        assert loss >= 0
