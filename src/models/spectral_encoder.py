# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

import torch
import torch.nn as nn
from src.geometry.poincare import project_to_poincare


class SpectralGraphEncoder(nn.Module):
    """Encodes graph structures using Spectral Graph Theory (Laplacian Eigenmaps).

    This encoder captures the global topology of a biological graph (e.g. protein contact map)
    by computing the eigenvectors of its Graph Laplacian.
    """

    def __init__(
        self, hidden_dim: int = 16, curvature: float = 1.0, max_norm: float = 0.95
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.curvature = curvature
        self.max_norm = max_norm

        # Learnable projection from spectral features to latent space
        # Eigenvectors can be variable size depending on graph, but usually we select top-k
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def compute_laplacian(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Compute Normalized Graph Laplacian (symmetric).
        L_sym = I - D^(-1/2) * A * D^(-1/2)
        """
        # adjacency: (B, N, N)
        B, N, _ = adjacency.shape

        # Degree matrix
        degree = adjacency.sum(dim=2)  # (B, N)

        # Inverse square root of degree
        # Add epsilon for numerical stability
        d_inv_sqrt = torch.pow(degree + 1e-8, -0.5)

        # Create diagonal matrix D^(-1/2)
        d_mat = torch.diag_embed(d_inv_sqrt)  # (B, N, N)

        # L_sym = I - D^(-1/2) * A * D^(-1/2)
        identity = torch.eye(N, device=adjacency.device).unsqueeze(0).expand(B, N, N)

        # D^(-1/2) * A * D^(-1/2)
        norm_adj = torch.bmm(torch.bmm(d_mat, adjacency), d_mat)

        laplacian = identity - norm_adj
        return laplacian

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            adjacency: (Batch, Nodes, Nodes) adjacency matrix.

        Returns:
            Latent vectors on Poincare ball: (Batch, Hidden_Dim)
        """
        B, N, _ = adjacency.shape

        # 1. Compute Laplacian
        L = self.compute_laplacian(adjacency)

        # 2. Eigendecomposition
        # specific to symmetric matrices (faster, stable)
        # eigenvalues: (B, N), eigenvectors: (B, N, N)
        eigvals, eigvecs = torch.linalg.eigh(L)

        # 3. Select Spectral Features
        # We need `hidden_dim` features.
        # Typically use smallest non-zero types for clustering (Fiedler vector etc.)
        # eigh returns eigenvalues in ascending order.
        # Index 0 is often constant (0 eigenvalue) for connected components.
        # We take indices 1 to hidden_dim+1

        target_dim = min(self.hidden_dim, N)

        # Take the eigenvectors corresponding to the smallest non-zero eigenvalues
        # Shape: (B, N, target_dim)
        spectral_features = eigvecs[:, :, 1 : target_dim + 1]

        # Flatten or pool?
        # For a graph embedding, we often want a fixed size vector per graph.
        # Simple pooling: mean over nodes
        graph_embedding = spectral_features.mean(dim=1)  # (B, target_dim)

        # Pad if unique nodes < hidden_dim
        if target_dim < self.hidden_dim:
            padding = torch.zeros(
                B, self.hidden_dim - target_dim, device=adjacency.device
            )
            graph_embedding = torch.cat([graph_embedding, padding], dim=1)

        # 4. Project and Hyperbolic Mapping
        linear_proj = self.projection(graph_embedding)

        # Map to Poincare Ball
        z_hyp = project_to_poincare(
            linear_proj, c=self.curvature, max_norm=self.max_norm
        )

        return z_hyp
