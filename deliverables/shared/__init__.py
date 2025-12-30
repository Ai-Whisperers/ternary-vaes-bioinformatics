# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Shared infrastructure for all deliverables.

This module provides common utilities used by all partner implementations:
- VAE service for latent space encoding/decoding
- Configuration management
- Common utilities and constants
"""

from __future__ import annotations

from .config import Config, get_config
from .vae_service import VAEService, get_vae_service
from .constants import AMINO_ACIDS, CODON_TABLE, HYDROPHOBICITY, CHARGES, VOLUMES

__all__ = [
    "Config",
    "get_config",
    "VAEService",
    "get_vae_service",
    "AMINO_ACIDS",
    "CODON_TABLE",
    "HYDROPHOBICITY",
    "CHARGES",
    "VOLUMES",
]
