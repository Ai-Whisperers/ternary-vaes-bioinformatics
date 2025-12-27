# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Uncertainty quantification module.

Provides calibrated uncertainty estimates for predictions,
distinguishing between epistemic (model) and aleatoric (data) uncertainty.

Key components:
- UncertaintyWrapper: Wrap any predictor with uncertainty
- BayesianPredictor: MC Dropout-based uncertainty
- EvidentialPredictor: Evidential deep learning
- EnsemblePredictor: Ensemble disagreement
- DeepEnsemble: Heteroscedastic ensemble with aleatoric uncertainty
- CalibratedWrapper: Wrapper with isotonic calibration
"""

from src.models.uncertainty.bayesian import (
    BayesianPredictor,
    BayesianResistancePredictor,
    MCDropoutWrapper,
)
from src.models.uncertainty.evidential import (
    EvidentialEnsemble,
    EvidentialLoss,
    EvidentialPredictor,
)
from src.models.uncertainty.ensemble import (
    BatchEnsemble,
    DeepEnsemble,
    EnsemblePredictor,
    SnapshotEnsemble,
)
from src.models.uncertainty.wrapper import (
    CalibratedWrapper,
    MultiOutputWrapper,
    UncertaintyMethod,
    UncertaintyWrapper,
)

__all__ = [
    # Core wrappers
    "UncertaintyWrapper",
    "CalibratedWrapper",
    "MultiOutputWrapper",
    "UncertaintyMethod",
    # Bayesian methods
    "BayesianPredictor",
    "BayesianResistancePredictor",
    "MCDropoutWrapper",
    # Evidential methods
    "EvidentialPredictor",
    "EvidentialLoss",
    "EvidentialEnsemble",
    # Ensemble methods
    "EnsemblePredictor",
    "DeepEnsemble",
    "SnapshotEnsemble",
    "BatchEnsemble",
]
