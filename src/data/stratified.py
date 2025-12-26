# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Stratified sampling for ternary operations.

This module provides stratified sampling strategies that ensure balanced
representation of valuation levels in training batches.
"""

from __future__ import annotations

from typing import List

import torch

from src.core import TERNARY


def create_stratified_batches(
    indices: torch.Tensor,
    batch_size: int,
    device: str = "cpu",
    high_valuation_fraction: float = 0.2,
    high_valuation_threshold: int = 4,
) -> List[torch.Tensor]:
    """Create stratified batch indices ensuring all valuation levels represented.

    High-valuation points are extremely rare (v>=7 is ~9 out of 19683).
    Random sampling means most batches have NO high-valuation points.

    Solution: Stratified sampling - ensure each batch contains points from
    all valuation levels, with oversampling of rare high-valuation points.

    Args:
        indices: All operation indices
        batch_size: Target batch size
        device: Torch device
        high_valuation_fraction: Fraction of batch reserved for high-v points
        high_valuation_threshold: Valuation level considered "high"

    Returns:
        List of batch index tensors, each containing stratified samples

    Example:
        >>> indices = torch.arange(19683)
        >>> batches = create_stratified_batches(indices, batch_size=512)
        >>> for batch_idx in batches:
        ...     x_batch = x[batch_idx]
        ...     # Each batch has balanced valuation representation
    """
    n_samples = len(indices)
    valuations = TERNARY.valuation(indices).cpu().numpy()

    # Group indices by valuation level
    valuation_groups = {}
    for i, v in enumerate(valuations):
        v = int(v)
        if v not in valuation_groups:
            valuation_groups[v] = []
        valuation_groups[v].append(i)

    # Convert to tensors
    for v in valuation_groups:
        valuation_groups[v] = torch.tensor(valuation_groups[v], device=device)

    # Allocation: reserve fraction for high-v, rest proportional
    high_v_budget = int(batch_size * high_valuation_fraction)
    low_v_budget = batch_size - high_v_budget

    # Separate valuation levels
    high_v_levels = [v for v in valuation_groups if v >= high_valuation_threshold]
    low_v_levels = [v for v in valuation_groups if v < high_valuation_threshold]

    batches = []
    n_batches = (n_samples + batch_size - 1) // batch_size

    for _ in range(n_batches):
        batch_indices = []

        # Sample from high-valuation levels (with replacement if needed)
        if high_v_levels:
            per_high_v = max(1, high_v_budget // len(high_v_levels))
            for v in high_v_levels:
                group = valuation_groups[v]
                n_to_sample = min(per_high_v, len(group))
                if len(group) <= n_to_sample:
                    # Take all, with replacement if oversampling needed
                    sample_idx = torch.randint(0, len(group), (per_high_v,), device=device)
                else:
                    sample_idx = torch.randperm(len(group), device=device)[:n_to_sample]
                batch_indices.append(group[sample_idx])

        # Sample from low-valuation levels (proportional to size)
        if low_v_levels:
            total_low = sum(len(valuation_groups[v]) for v in low_v_levels)
            for v in low_v_levels:
                group = valuation_groups[v]
                n_to_sample = max(1, int(low_v_budget * len(group) / total_low))
                sample_idx = torch.randint(0, len(group), (n_to_sample,), device=device)
                batch_indices.append(group[sample_idx])

        # Combine and shuffle
        batch = torch.cat(batch_indices)

        # Trim to exact batch size or pad if needed
        if len(batch) > batch_size:
            batch = batch[torch.randperm(len(batch), device=device)[:batch_size]]
        elif len(batch) < batch_size:
            # Pad with random samples
            extra = torch.randint(0, n_samples, (batch_size - len(batch),), device=device)
            batch = torch.cat([batch, extra])

        batches.append(batch)

    return batches


def get_valuation_distribution(indices: torch.Tensor) -> dict:
    """Get distribution of valuation levels in indices.

    Args:
        indices: Tensor of operation indices

    Returns:
        Dictionary mapping valuation level to count

    Example:
        >>> indices = torch.arange(19683)
        >>> dist = get_valuation_distribution(indices)
        >>> print(dist)
        {0: 13122, 1: 4374, 2: 1458, ..., 9: 1}
    """
    valuations = TERNARY.valuation(indices).cpu().numpy()
    distribution = {}
    for v in valuations:
        v = int(v)
        distribution[v] = distribution.get(v, 0) + 1
    return dict(sorted(distribution.items()))
