# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Bioinformatics module for DDG prediction.

This module implements a multimodal VAE architecture for protein stability
(DDG) prediction, combining:
- High-quality curated data (ProTherm)
- Benchmark data (S669)
- Large-scale diverse data (ProteinGym)

Architecture:
1. Three specialist VAEs trained on different data regimes
2. Multimodal fusion VAE combining all embeddings
3. MLP refinement for precise predictions
4. Dual output heads (fuzzy VAE + transformer)

See README.md for detailed documentation.
"""

from src.bioinformatics.data import (
    ProThermLoader,
    S669Loader,
    ProteinGymLoader,
    DatasetRegistry,
)
from src.bioinformatics.models import (
    DDGVAE,
    MultimodalDDGVAE,
    DDGMLPRefiner,
    DDGTransformer,
    DDGEnsemble,
)
from src.bioinformatics.training import (
    set_deterministic_mode,
    get_deterministic_dataloader,
)
from src.bioinformatics.evaluation import (
    DDGMetrics,
    CrossValidator,
    BenchmarkRunner,
)

__version__ = "1.0.0"
__all__ = [
    # Data
    "ProThermLoader",
    "S669Loader",
    "ProteinGymLoader",
    "DatasetRegistry",
    # Models
    "DDGVAE",
    "MultimodalDDGVAE",
    "DDGMLPRefiner",
    "DDGTransformer",
    "DDGEnsemble",
    # Training
    "set_deterministic_mode",
    "get_deterministic_dataloader",
    # Evaluation
    "DDGMetrics",
    "CrossValidator",
    "BenchmarkRunner",
]
