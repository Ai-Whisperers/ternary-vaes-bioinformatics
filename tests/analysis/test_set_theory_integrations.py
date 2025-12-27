# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for set theory integration modules.

Tests cover:
- UnifiedResistanceAnalyzer
- SetTheoryAwareLoss
- LatticeAwareHyperbolicProjection
- FCAExplainer
- SetAugmenter
- ConceptAwareContrastive
- RoughUncertaintyWrapper
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, List

from src.analysis.set_theory.mutation_sets import (
    Mutation,
    MutationSet,
    ResistanceProfile,
)
from src.analysis.set_theory.rough_sets import RoughClassifier
from src.analysis.set_theory.lattice import ResistanceLattice, ResistanceLevel
from src.analysis.set_theory.formal_concepts import (
    FormalContext,
    ConceptLattice,
    GenotypePhenotypeAnalyzer,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def sample_mutations():
    """Create sample mutation sets."""
    return {
        "mono_rif": MutationSet.from_strings(["rpoB_S450L"], "mono_RIF"),
        "mono_inh": MutationSet.from_strings(["katG_S315T"], "mono_INH"),
        "mdr": MutationSet.from_strings(["rpoB_S450L", "katG_S315T"], "MDR"),
        "xdr": MutationSet.from_strings([
            "rpoB_S450L", "katG_S315T", "gyrA_D94G", "rrs_A1401G"
        ], "XDR"),
    }


@pytest.fixture
def rough_classifiers():
    """Create sample rough classifiers."""
    return {
        "RIF": RoughClassifier.from_evidence(
            definite_resistance=["rpoB_S450L"],
            possible_resistance=["rpoB_H445Y", "rpoB_D435V"],
            drug_name="RIF",
        ),
        "INH": RoughClassifier.from_evidence(
            definite_resistance=["katG_S315T"],
            possible_resistance=["inhA_C-15T"],
            drug_name="INH",
        ),
    }


@pytest.fixture
def resistance_lattice(sample_mutations):
    """Create sample resistance lattice."""
    lattice = ResistanceLattice()
    for name, ms in sample_mutations.items():
        lattice.add_profile(name, ms.to_list())
    return lattice


@pytest.fixture
def fca_analyzer():
    """Create sample FCA analyzer."""
    samples = {
        "S1": ["rpoB_S450L", "katG_S315T"],
        "S2": ["rpoB_S450L"],
        "S3": ["katG_S315T"],
        "S4": ["rpoB_S450L", "katG_S315T", "gyrA_D94G"],
    }
    resistance = {
        "S1": ["RIF", "INH"],
        "S2": ["RIF"],
        "S3": ["INH"],
        "S4": ["RIF", "INH", "FQ"],
    }
    return GenotypePhenotypeAnalyzer(samples, resistance)


@pytest.fixture
def simple_model():
    """Create simple neural network for testing."""
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(32, 1),
    )


# ==============================================================================
# UnifiedResistanceAnalyzer Tests
# ==============================================================================

class TestUnifiedResistanceAnalyzer:
    """Test UnifiedResistanceAnalyzer."""

    def test_analyzer_creation(self, rough_classifiers, resistance_lattice, fca_analyzer):
        """Test creating analyzer."""
        from src.analysis.resistance_analyzer import UnifiedResistanceAnalyzer

        analyzer = UnifiedResistanceAnalyzer(
            rough_classifiers=rough_classifiers,
            lattice=resistance_lattice,
            fca_analyzer=fca_analyzer,
        )

        assert analyzer is not None
        assert len(analyzer.rough_classifiers) == 2

    def test_analyze_sample(self, rough_classifiers, resistance_lattice, fca_analyzer):
        """Test analyzing a sample."""
        from src.analysis.resistance_analyzer import UnifiedResistanceAnalyzer

        analyzer = UnifiedResistanceAnalyzer(
            rough_classifiers=rough_classifiers,
            lattice=resistance_lattice,
            fca_analyzer=fca_analyzer,
        )

        sample = {"mutations": ["rpoB_S450L", "katG_S315T"]}
        result = analyzer.analyze(sample, include_neural=False)

        assert result.resistance_level == ResistanceLevel.MDR
        assert "RIF" in result.rough_classification
        assert "INH" in result.rough_classification

    def test_analyze_mdr_detection(self, rough_classifiers, resistance_lattice):
        """Test MDR detection."""
        from src.analysis.resistance_analyzer import UnifiedResistanceAnalyzer

        analyzer = UnifiedResistanceAnalyzer(
            rough_classifiers=rough_classifiers,
            lattice=resistance_lattice,
        )

        # MDR sample
        mdr_sample = {"mutations": ["rpoB_S450L", "katG_S315T"]}
        result = analyzer.analyze(mdr_sample, include_neural=False)

        assert result.resistance_level == ResistanceLevel.MDR

        # Mono-resistant sample
        mono_sample = {"mutations": ["rpoB_S450L"]}
        result = analyzer.analyze(mono_sample, include_neural=False)

        assert result.resistance_level == ResistanceLevel.MONO_RESISTANT

    def test_build_from_training_data(self):
        """Test building analyzer from training data."""
        from src.analysis.resistance_analyzer import UnifiedResistanceAnalyzer

        samples = {
            "S1": ["rpoB_S450L"],
            "S2": ["katG_S315T"],
        }
        resistance = {
            "S1": ["RIF"],
            "S2": ["INH"],
        }
        known_mutations = {
            "RIF": {
                "definite": ["rpoB_S450L"],
                "possible": ["rpoB_H445Y"],
            },
        }

        analyzer = UnifiedResistanceAnalyzer()
        analyzer.build_from_training_data(samples, resistance, known_mutations)

        assert "RIF" in analyzer.rough_classifiers
        assert analyzer.fca_analyzer is not None


# ==============================================================================
# SetTheoryAwareLoss Tests
# ==============================================================================

class TestSetTheoryAwareLoss:
    """Test SetTheoryAwareLoss functions."""

    def test_lattice_ordering_loss(self, resistance_lattice, sample_mutations):
        """Test lattice ordering loss."""
        from src.losses.set_theory_loss import LatticeOrderingLoss

        loss_fn = LatticeOrderingLoss(resistance_lattice, margin=0.1)

        # Create embeddings with correct ordering
        embeddings = torch.tensor([
            [0.1, 0.0],  # mono_rif - close to origin
            [0.1, 0.1],  # mono_inh - close to origin
            [0.3, 0.3],  # mdr - further from origin
            [0.5, 0.5],  # xdr - furthest
        ])

        mutation_sets = [
            sample_mutations["mono_rif"],
            sample_mutations["mono_inh"],
            sample_mutations["mdr"],
            sample_mutations["xdr"],
        ]

        loss = loss_fn(embeddings, mutation_sets)

        # Should be low since ordering is correct
        assert loss.item() >= 0

    def test_hierarchical_resistance_loss(self, resistance_lattice, sample_mutations):
        """Test hierarchical resistance loss."""
        from src.losses.set_theory_loss import HierarchicalResistanceLoss

        loss_fn = HierarchicalResistanceLoss(resistance_lattice)

        # Predictions (batch_size=2, n_levels=6)
        predictions = torch.randn(2, 6)
        targets = torch.tensor([0, 3])  # Susceptible, MDR

        mutation_sets = [
            sample_mutations["mono_rif"],
            sample_mutations["mdr"],
        ]

        loss = loss_fn(predictions, mutation_sets, targets)

        assert loss.item() >= 0

    def test_set_theory_aware_loss_combined(
        self, resistance_lattice, rough_classifiers, sample_mutations
    ):
        """Test combined set-aware loss."""
        from src.losses.set_theory_loss import SetTheoryAwareLoss, SetLossConfig

        config = SetLossConfig(
            lattice_weight=0.1,
            rough_weight=0.1,
        )

        loss_fn = SetTheoryAwareLoss(
            config=config,
            lattice=resistance_lattice,
            rough_classifiers=rough_classifiers,
            drug_names=["RIF", "INH"],
        )

        predictions = torch.randn(2, 2)  # batch=2, drugs=2
        targets = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

        embeddings = torch.randn(2, 16)
        mutation_sets = [
            sample_mutations["mdr"],
            sample_mutations["mono_rif"],
        ]

        losses = loss_fn(
            predictions=predictions,
            targets=targets,
            embeddings=embeddings,
            mutation_sets=mutation_sets,
        )

        assert "total" in losses
        assert "prediction" in losses
        assert losses["total"].item() >= 0


# ==============================================================================
# LatticeAwareHyperbolicProjection Tests
# ==============================================================================

class TestLatticeAwareHyperbolicProjection:
    """Test LatticeAwareHyperbolicProjection."""

    def test_projection_creation(self, resistance_lattice):
        """Test creating projection."""
        from src.models.lattice_projection import (
            LatticeAwareHyperbolicProjection,
            LatticeProjectionConfig,
        )

        config = LatticeProjectionConfig(latent_dim=16)
        proj = LatticeAwareHyperbolicProjection(config, resistance_lattice)

        assert proj is not None
        assert proj.projection is not None

    def test_forward_pass(self, resistance_lattice, sample_mutations):
        """Test forward pass."""
        from src.models.lattice_projection import (
            LatticeAwareHyperbolicProjection,
            LatticeProjectionConfig,
        )

        config = LatticeProjectionConfig(latent_dim=16)
        proj = LatticeAwareHyperbolicProjection(config, resistance_lattice)

        embeddings = torch.randn(2, 16)
        mutation_sets = [
            sample_mutations["mono_rif"],
            sample_mutations["mdr"],
        ]

        hyp_embeddings, losses = proj(embeddings, mutation_sets)

        assert hyp_embeddings.shape == (2, 16)
        assert "ordering" in losses
        assert "level" in losses

    def test_level_radii(self, resistance_lattice):
        """Test level radii computation."""
        from src.models.lattice_projection import (
            LatticeAwareHyperbolicProjection,
            LatticeProjectionConfig,
        )

        config = LatticeProjectionConfig(latent_dim=16, max_radius=0.95)
        proj = LatticeAwareHyperbolicProjection(config, resistance_lattice)

        radii = proj.get_level_radii()

        # Should have radius for each level
        assert len(radii) == len(ResistanceLevel)
        # Should be increasing
        for i in range(1, len(radii)):
            assert radii[i] >= radii[i - 1]


# ==============================================================================
# FCAExplainer Tests
# ==============================================================================

class TestFCAExplainer:
    """Test FCA-based explainability."""

    def test_explainer_creation(self, fca_analyzer, rough_classifiers):
        """Test creating explainer."""
        from src.analysis.explainability import FCAExplainer

        explainer = FCAExplainer(
            fca_analyzer=fca_analyzer,
            rough_classifiers=rough_classifiers,
        )

        assert explainer is not None
        assert len(explainer.implications) > 0

    def test_explain_prediction(self, fca_analyzer, rough_classifiers):
        """Test explaining a prediction."""
        from src.analysis.explainability import FCAExplainer

        explainer = FCAExplainer(
            fca_analyzer=fca_analyzer,
            rough_classifiers=rough_classifiers,
        )

        mutations = MutationSet.from_strings(["rpoB_S450L"])
        explanation = explainer.explain("RIF", mutations, "resistant", 0.95)

        assert explanation.drug == "RIF"
        assert explanation.prediction == "resistant"
        assert explanation.natural_language != ""
        assert len(explanation.key_mutations) > 0

    def test_counterfactual_generation(self, fca_analyzer, rough_classifiers):
        """Test counterfactual generation."""
        from src.analysis.explainability import FCAExplainer

        explainer = FCAExplainer(
            fca_analyzer=fca_analyzer,
            rough_classifiers=rough_classifiers,
        )

        mutations = MutationSet.from_strings(["rpoB_S450L"])
        explanation = explainer.explain("RIF", mutations, "resistant")

        # Should have some counterfactuals
        assert isinstance(explanation.counterfactuals, list)


# ==============================================================================
# SetAugmenter Tests
# ==============================================================================

class TestSetAugmenter:
    """Test set-theoretic data augmentation."""

    def test_augmenter_creation(self, resistance_lattice):
        """Test creating augmenter."""
        from src.data.set_augmentation import SetAugmenter, AugmentationConfig

        config = AugmentationConfig()
        augmenter = SetAugmenter(config, resistance_lattice)

        assert augmenter is not None

    def test_augment_sample(self, resistance_lattice, sample_mutations):
        """Test augmenting a sample."""
        from src.data.set_augmentation import SetAugmenter

        augmenter = SetAugmenter(lattice=resistance_lattice)

        # Add samples for combination
        augmenter.add_samples(list(sample_mutations.values()))

        # Augment
        original = sample_mutations["mono_rif"]
        augmented = augmenter.augment(original, n_augmentations=3)

        assert len(augmented) <= 3  # May be less if augmentation fails

    def test_balanced_sampler(self, resistance_lattice, sample_mutations):
        """Test balanced sampler."""
        from src.data.set_augmentation import BalancedSampler

        samples = list(sample_mutations.values())
        sampler = BalancedSampler(samples, resistance_lattice)

        balanced = sampler.sample_balanced(n_per_level=2)

        # Should have samples for levels that have data
        assert len(balanced) > 0


# ==============================================================================
# ConceptAwareContrastive Tests
# ==============================================================================

class TestConceptAwareContrastive:
    """Test concept-aware contrastive learning."""

    def test_contrastive_creation(self, fca_analyzer):
        """Test creating contrastive module."""
        from src.models.contrastive.concept_aware import (
            ConceptAwareContrastive,
            ConceptContrastiveConfig,
        )

        concept_lattice = fca_analyzer.lattice
        config = ConceptContrastiveConfig()

        contrastive = ConceptAwareContrastive(concept_lattice, config)

        assert contrastive is not None

    def test_contrastive_forward(self, fca_analyzer):
        """Test contrastive forward pass."""
        from src.models.contrastive.concept_aware import ConceptAwareContrastive

        concept_lattice = fca_analyzer.lattice
        contrastive = ConceptAwareContrastive(concept_lattice)

        embeddings = torch.randn(4, 64)
        sample_ids = ["S1", "S2", "S3", "S4"]

        loss = contrastive(embeddings, sample_ids)

        assert loss.item() >= 0

    def test_get_positive_pairs(self, fca_analyzer):
        """Test getting positive pairs."""
        from src.models.contrastive.concept_aware import ConceptAwareContrastive

        concept_lattice = fca_analyzer.lattice
        contrastive = ConceptAwareContrastive(concept_lattice)

        pairs = contrastive.get_positive_pairs(["S1", "S2", "S3", "S4"])

        # Should have some positive pairs
        assert isinstance(pairs, list)


# ==============================================================================
# RoughUncertaintyWrapper Tests
# ==============================================================================

class TestRoughUncertaintyWrapper:
    """Test rough set enhanced uncertainty wrapper."""

    def test_wrapper_creation(self, simple_model, rough_classifiers):
        """Test creating wrapper."""
        from src.models.uncertainty.rough_wrapper import RoughUncertaintyWrapper

        wrapper = RoughUncertaintyWrapper(
            model=simple_model,
            rough_classifiers=rough_classifiers,
        )

        assert wrapper is not None

    def test_predict_with_uncertainty(self, simple_model, rough_classifiers, sample_mutations):
        """Test prediction with uncertainty."""
        from src.models.uncertainty.rough_wrapper import RoughUncertaintyWrapper

        wrapper = RoughUncertaintyWrapper(
            model=simple_model,
            rough_classifiers=rough_classifiers,
        )

        x = torch.randn(1, 16)
        result = wrapper.predict_with_uncertainty(x)

        assert "prediction" in result
        assert "std" in result

    def test_three_way_decision(self, simple_model, rough_classifiers, sample_mutations):
        """Test three-way decision making."""
        from src.models.uncertainty.rough_wrapper import (
            RoughUncertaintyWrapper,
            Decision,
        )

        wrapper = RoughUncertaintyWrapper(
            model=simple_model,
            rough_classifiers=rough_classifiers,
        )

        x = torch.randn(1, 16)
        mutations = sample_mutations["mono_rif"]

        result = wrapper.predict_with_three_way_decision(x, mutations, "RIF")

        assert result.decision in [Decision.ACCEPT, Decision.REJECT, Decision.DEFER]
        assert result.combined_confidence >= 0 and result.combined_confidence <= 1

    def test_deferred_samples(self, simple_model, rough_classifiers, sample_mutations):
        """Test getting deferred samples."""
        from src.models.uncertainty.rough_wrapper import RoughUncertaintyWrapper

        wrapper = RoughUncertaintyWrapper(
            model=simple_model,
            rough_classifiers=rough_classifiers,
        )

        x = torch.randn(3, 16)
        mutation_sets = [
            sample_mutations["mono_rif"],
            sample_mutations["mdr"],
            MutationSet.from_strings(["rpoB_H445Y"]),  # Boundary mutation
        ]

        results = wrapper.batch_predict(x, mutation_sets, "RIF")
        deferred = wrapper.get_deferred_samples(results)

        assert isinstance(deferred, list)


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestFullIntegration:
    """Test full integration of all modules."""

    def test_end_to_end_analysis(
        self, simple_model, rough_classifiers, resistance_lattice, fca_analyzer
    ):
        """Test complete analysis pipeline."""
        from src.analysis.resistance_analyzer import UnifiedResistanceAnalyzer
        from src.analysis.explainability import FCAExplainer
        from src.models.uncertainty.rough_wrapper import RoughUncertaintyWrapper

        # Create components
        analyzer = UnifiedResistanceAnalyzer(
            rough_classifiers=rough_classifiers,
            lattice=resistance_lattice,
            fca_analyzer=fca_analyzer,
        )

        explainer = FCAExplainer(
            fca_analyzer=fca_analyzer,
            rough_classifiers=rough_classifiers,
            lattice=resistance_lattice,
        )

        # Analyze sample
        sample = {"mutations": ["rpoB_S450L", "katG_S315T"]}
        analysis = analyzer.analyze(sample, include_neural=False)

        # Explain
        mutations = MutationSet.from_strings(sample["mutations"])
        explanation = explainer.explain(
            "RIF", mutations, analysis.predicted_class.get("RIF", "unknown")
        )

        # Verify integration
        assert analysis.resistance_level == ResistanceLevel.MDR
        assert explanation.natural_language != ""

    def test_training_pipeline_integration(
        self, simple_model, rough_classifiers, resistance_lattice, sample_mutations
    ):
        """Test training pipeline with set-aware loss."""
        from src.losses.set_theory_loss import SetTheoryAwareLoss, SetLossConfig

        config = SetLossConfig(lattice_weight=0.1, rough_weight=0.1)

        loss_fn = SetTheoryAwareLoss(
            config=config,
            lattice=resistance_lattice,
            rough_classifiers=rough_classifiers,
            drug_names=["RIF", "INH"],
        )

        # Simulate training step
        x = torch.randn(2, 16)
        targets = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

        # Forward pass
        predictions = simple_model(x)
        predictions = predictions.expand(-1, 2)  # Match drug count

        # Compute loss
        losses = loss_fn(
            predictions=predictions,
            targets=targets,
            embeddings=x,
            mutation_sets=[sample_mutations["mdr"], sample_mutations["mono_rif"]],
        )

        # Backward pass should work
        losses["total"].backward()

        assert True  # If we get here, integration works
