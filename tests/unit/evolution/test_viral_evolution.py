# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for ViralEvolutionPredictor module."""

import pytest
import torch

from src.analysis.evolution import (
    AMINO_ACID_PROPERTIES,
    EscapeMutation,
    EvolutionaryPressure,
    MutationHotspot,
    SelectionType,
    ViralEvolutionPredictor,
)


class TestViralEvolutionPredictor:
    """Tests for ViralEvolutionPredictor."""

    def test_creation(self):
        """Test predictor creation."""
        predictor = ViralEvolutionPredictor(p=3, latent_dim=16)
        assert predictor.p == 3
        assert predictor.latent_dim == 16

    def test_padic_distance_same(self):
        """Test p-adic distance between same codons."""
        predictor = ViralEvolutionPredictor()
        dist = predictor.compute_padic_distance(10, 10)
        assert dist == 0.0

    def test_padic_distance_different(self):
        """Test p-adic distance between different codons."""
        predictor = ViralEvolutionPredictor(p=3)
        dist = predictor.compute_padic_distance(0, 3)
        assert dist > 0
        # Difference of 3 should have valuation 1
        assert dist == pytest.approx(1.0 / 3)

    def test_mutation_accessibility(self):
        """Test mutation accessibility computation."""
        predictor = ViralEvolutionPredictor()
        accessibility = predictor.compute_mutation_accessibility(0)

        assert len(accessibility) == 63  # All other codons
        assert 0 not in accessibility
        assert all(0 <= v <= 1 for v in accessibility.values())

    def test_encode_epitope_pressure(self):
        """Test epitope pressure encoding."""
        predictor = ViralEvolutionPredictor()

        b_cell = torch.randn(2, 20)
        cd4 = torch.randn(2, 20)
        cd8 = torch.randn(2, 20)

        pressure = predictor.encode_epitope_pressure(b_cell, cd4, cd8)

        assert pressure.shape == (2, 20, predictor.latent_dim)

    def test_predict_mutation_probabilities(self):
        """Test mutation probability prediction."""
        predictor = ViralEvolutionPredictor()

        codon_indices = torch.randint(0, 64, (2, 30))
        epitope_pressure = torch.randn(2, 30, predictor.latent_dim)

        probs = predictor.predict_mutation_probabilities(codon_indices, epitope_pressure)

        assert probs.shape == (2, 30, 20)
        # Should sum to 1 (probabilities)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 30), atol=1e-5)

    def test_compute_escape_scores(self):
        """Test escape score computation."""
        predictor = ViralEvolutionPredictor()

        codon_indices = torch.randint(0, 64, (2, 30))
        epitope_pressure = torch.randn(2, 30, predictor.latent_dim)

        scores = predictor.compute_escape_scores(codon_indices, epitope_pressure)

        assert scores.shape == (2, 30)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_forward(self):
        """Test full forward pass."""
        predictor = ViralEvolutionPredictor()

        codon_indices = torch.randint(0, 64, (2, 50))

        results = predictor(codon_indices)

        assert "mutation_probabilities" in results
        assert "escape_scores" in results
        assert "fitness_scores" in results
        assert "combined_scores" in results

    def test_forward_with_epitopes(self):
        """Test forward pass with epitope annotations."""
        predictor = ViralEvolutionPredictor()

        codon_indices = torch.randint(0, 64, (2, 50))
        epitope_annotations = {
            "b_cell": torch.rand(2, 50),
            "cd4": torch.rand(2, 50),
            "cd8": torch.rand(2, 50),
        }

        results = predictor(codon_indices, epitope_annotations)

        assert results["escape_scores"].shape == (2, 50)

    def test_identify_hotspots(self):
        """Test hotspot identification."""
        predictor = ViralEvolutionPredictor()

        # Create scores with a clear hotspot
        scores = torch.tensor([0.1, 0.2, 0.7, 0.8, 0.9, 0.8, 0.2, 0.1])

        hotspots = predictor.identify_hotspots(scores, threshold=0.5)

        assert len(hotspots) >= 1
        assert hotspots[0].start == 2
        assert hotspots[0].end == 6

    def test_predict_escape_mutations(self):
        """Test complete escape prediction."""
        predictor = ViralEvolutionPredictor()

        codon_indices = torch.randint(0, 64, (50,))

        prediction = predictor.predict_escape_mutations(codon_indices, top_k=5)

        assert len(prediction.mutations) <= 5
        assert isinstance(prediction.overall_escape_risk, float)
        assert 0 <= prediction.overall_escape_risk <= 1

    def test_evolutionary_pressure(self):
        """Test evolutionary pressure computation."""
        predictor = ViralEvolutionPredictor()

        codon_indices = torch.randint(0, 64, (20,))
        epitope_pressure = torch.randn(20, predictor.latent_dim)

        pressures = predictor.compute_evolutionary_pressure(codon_indices, epitope_pressure)

        assert len(pressures) == 20
        assert all(isinstance(p, EvolutionaryPressure) for p in pressures)


class TestEscapeMutation:
    """Tests for EscapeMutation dataclass."""

    def test_creation(self):
        """Test mutation creation."""
        mutation = EscapeMutation(
            position=10,
            original_aa="A",
            mutant_aa="V",
            escape_score=0.8,
            fitness_cost=0.2,
            padic_distance=0.33,
        )

        assert mutation.position == 10
        assert mutation.original_aa == "A"
        assert mutation.selection_type == SelectionType.POSITIVE


class TestMutationHotspot:
    """Tests for MutationHotspot dataclass."""

    def test_creation(self):
        """Test hotspot creation."""
        hotspot = MutationHotspot(
            start=100,
            end=150,
            mutation_rate=0.7,
            dominant_selection=SelectionType.POSITIVE,
            epitope_overlap=True,
        )

        assert hotspot.start == 100
        assert hotspot.end == 150
        assert hotspot.epitope_overlap is True


class TestAminoAcidProperties:
    """Tests for amino acid property data."""

    def test_all_20_amino_acids(self):
        """Test that all 20 standard amino acids are defined."""
        assert len(AMINO_ACID_PROPERTIES) == 20

    def test_property_keys(self):
        """Test that each AA has required properties."""
        required = {"hydrophobicity", "volume", "charge", "polarity"}

        for aa, props in AMINO_ACID_PROPERTIES.items():
            assert required.issubset(props.keys()), f"Missing properties for {aa}"


class TestSelectionType:
    """Tests for SelectionType enum."""

    def test_values(self):
        """Test enum values."""
        assert SelectionType.POSITIVE.value == "positive"
        assert SelectionType.NEGATIVE.value == "negative"
        assert SelectionType.NEUTRAL.value == "neutral"
        assert SelectionType.BALANCING.value == "balancing"
