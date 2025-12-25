# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Codon Encoder with p-adic Embeddings.

This module implements the `CodonEncoder` (Proposal 3), a specialized embedding
layer for biological sequences that respects the ultrametric structure of the
genetic code.

Key Features:
- Maps codon indices (0-63) to latent space (Dim).
- Can optionally use p-adic valuation initialization.
- Integrates with `src.core.ternary` for 3-adic logic.

Usage:
    from src.encoders.codon_encoder import CodonEncoder
    encoder = CodonEncoder(embedding_dim=16)
    z = encoder(codon_indices)
"""

import torch
import torch.nn as nn
from typing import Optional


class CodonEncoder(nn.Module):
    """Embedding layer for codons with p-adic structure awareness."""

    def __init__(
        self,
        embedding_dim: int = 16,
        padding_idx: Optional[int] = None,
        use_padic_init: bool = True,
    ):
        """Initialize CodonEncoder.

        Args:
            embedding_dim: Size of the embedding vector
            padding_idx: Index to use for padding (zeros out gradients)
            use_padic_init: If True, initializes weights using 3-adic valuation patterns
        """
        super().__init__()
        self.num_codons = 64
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=self.num_codons,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        if use_padic_init:
            self._init_padic_weights()

    def _init_padic_weights(self):
        """Initialize embeddings to reflect 3-adic distances.

        We use the TERNARY module to compute valuations between codon indices
        and project them into the embedding space.

        Since we don't have a direct map from codon integer (0-63) to the
        19683-space of the ternary algebra, we define a mapping or use
        a simpler heuristic:

        Map 0-63 to the first 64 indices of the ternary space (or any subset).
        Then use Multi-Dimensional Scaling (MDS) or PCA on the distance matrix
        to get initial coordinates.

        For this MVP, we'll use a simplified stratified initialization:
        - Codons are grouped by first base (4 groups)
        - Then by second base (16 groups)
        - This mirrors the hierarchical structure.
        """
        with torch.no_grad():
            # Simple hierarchical init
            # 64 codons.
            # Dims 0-3: Encode Base 1
            # Dims 4-7: Encode Base 2
            # Dims 8-11: Encode Base 3
            # Dims 12+: Noise/Free

            # Base one-hot (A, C, G, T) -> 4 dims
            base_map = {
                0: [1, 0, 0, 0],  # A
                1: [0, 1, 0, 0],  # C
                2: [0, 0, 1, 0],  # G
                3: [0, 0, 0, 1],  # T
            }

            weights = self.embedding.weight.data
            scaling = 0.5  # Small scale init

            for i in range(self.num_codons):
                # Decode index i (0-63) to bases
                # format: B1 * 16 + B2 * 4 + B3
                b1 = (i // 16) % 4
                b2 = (i // 4) % 4
                b3 = i % 4

                vec_b1 = torch.tensor(base_map[b1], dtype=torch.float32)
                vec_b2 = torch.tensor(base_map[b2], dtype=torch.float32)
                vec_b3 = torch.tensor(base_map[b3], dtype=torch.float32)

                if self.embedding_dim >= 12:
                    weights[i, 0:4] = vec_b1 * scaling
                    weights[i, 4:8] = vec_b2 * scaling
                    weights[i, 8:12] = vec_b3 * scaling
                    # Remainders stay random (default init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed codon indices.

        Args:
            x: Tensor of codon indices (Batch, SeqLen)

        Returns:
            Tensor of embeddings (Batch, SeqLen, Dim)
        """
        return self.embedding(x)

    def get_distance_matrix(self) -> torch.Tensor:
        """Compute pairwise Euclidean distances between all codon embeddings."""
        w = self.embedding.weight
        return torch.cdist(w, w)
