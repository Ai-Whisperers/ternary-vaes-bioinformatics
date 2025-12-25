# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for Long COVID and spike protein analyzer."""

import pytest
import numpy as np

from src.diseases.long_covid import (
    KNOWN_VARIANTS,
    SPIKE_GLYCOSYLATION_SITES,
    LongCOVIDAnalyzer,
    LongCOVIDRiskProfile,
    PTMAnalysisResult,
    PTMSite,
    PTMType,
    SpikeRegion,
    SpikeVariant,
    SpikeVariantComparator,
)


class TestPTMSite:
    """Tests for PTMSite dataclass."""

    def test_basic_creation(self):
        """Test basic PTM site creation."""
        site = PTMSite(
            position=614,
            ptm_type=PTMType.MUTATION,
            original="D",
            modified="G",
            region=SpikeRegion.SD2,
        )
        assert site.position == 614
        assert site.ptm_type == PTMType.MUTATION
        assert site.original == "D"
        assert site.modified == "G"
        assert site.region == SpikeRegion.SD2

    def test_optional_fields(self):
        """Test optional fields default values."""
        site = PTMSite(position=100, ptm_type=PTMType.GLYCOSYLATION, original="N", modified="N-glycan")
        assert site.region is None
        assert site.conservation_score == 0.0
        assert site.known_variant is None


class TestSpikeVariant:
    """Tests for SpikeVariant dataclass."""

    def test_wuhan_variant(self):
        """Test Wuhan reference variant."""
        wuhan = KNOWN_VARIANTS["wuhan"]
        assert wuhan.name == "Wuhan-Hu-1"
        assert wuhan.lineage == "A"
        assert len(wuhan.mutations) == 0
        assert wuhan.transmissibility_factor == 1.0

    def test_delta_variant(self):
        """Test Delta variant has expected mutations."""
        delta = KNOWN_VARIANTS["delta"]
        assert delta.name == "Delta"
        assert len(delta.mutations) > 0
        assert delta.transmissibility_factor > 1.0

        # Check for D614G mutation
        d614g = [m for m in delta.mutations if m.position == 614]
        assert len(d614g) == 1
        assert d614g[0].original == "D"
        assert d614g[0].modified == "G"

    def test_omicron_variant(self):
        """Test Omicron variant has many mutations."""
        omicron = KNOWN_VARIANTS["omicron_ba1"]
        assert omicron.name == "Omicron BA.1"
        # Omicron has many more mutations than earlier variants
        assert len(omicron.mutations) > len(KNOWN_VARIANTS["delta"].mutations)
        assert omicron.immune_escape_score > 0.5


class TestLongCOVIDAnalyzer:
    """Tests for LongCOVIDAnalyzer."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = LongCOVIDAnalyzer()
        assert analyzer.p == 3
        assert analyzer.goldilocks_center == 0.5
        assert analyzer.spike_length == 1273

    def test_get_region(self):
        """Test region detection for positions."""
        analyzer = LongCOVIDAnalyzer()

        # RBD region
        assert analyzer._get_region(400) == SpikeRegion.RBD

        # RBM is within RBD - function returns first match (RBD)
        # Position 480 is in both RBD (319-541) and RBM (437-508)
        region = analyzer._get_region(480)
        assert region in [SpikeRegion.RBD, SpikeRegion.RBM]

        # NTD region
        assert analyzer._get_region(100) == SpikeRegion.NTD

        # Fusion peptide
        assert analyzer._get_region(800) == SpikeRegion.FP

        # Outside defined regions
        assert analyzer._get_region(700) is None

    def test_padic_valuation(self):
        """Test p-adic valuation computation."""
        analyzer = LongCOVIDAnalyzer(p=3)

        assert analyzer._compute_padic_valuation(9) == 2
        assert analyzer._compute_padic_valuation(27) == 3
        assert analyzer._compute_padic_valuation(10) == 0

    def test_analyze_single_ptm(self):
        """Test analysis of single PTM site."""
        analyzer = LongCOVIDAnalyzer()

        site = PTMSite(
            position=614,
            ptm_type=PTMType.MUTATION,
            original="D",
            modified="G",
            region=SpikeRegion.SD2,
        )

        result = analyzer.analyze_ptm(site)

        assert isinstance(result, PTMAnalysisResult)
        assert result.site == site
        assert 0 <= result.padic_distance <= 1
        assert 0 <= result.goldilocks_score <= 1
        assert 0 <= result.immunogenicity_score <= 1
        assert result.structural_impact in ["low", "medium", "high"]

    def test_analyze_rbd_mutation_higher_immunogenicity(self):
        """Test that RBD mutations have higher immunogenicity."""
        analyzer = LongCOVIDAnalyzer()

        rbd_site = PTMSite(position=484, ptm_type=PTMType.MUTATION, original="E", modified="K", region=SpikeRegion.RBM)

        non_rbd_site = PTMSite(
            position=950, ptm_type=PTMType.MUTATION, original="D", modified="N", region=SpikeRegion.HR1
        )

        rbd_result = analyzer.analyze_ptm(rbd_site)
        non_rbd_result = analyzer.analyze_ptm(non_rbd_site)

        # RBM mutations should have higher immunogenicity
        assert rbd_result.immunogenicity_score > non_rbd_result.immunogenicity_score

    def test_analyze_spike_ptms_empty(self):
        """Test analysis with no PTM sites."""
        analyzer = LongCOVIDAnalyzer()
        profile = analyzer.analyze_spike_ptms(ptm_sites=[])

        assert isinstance(profile, LongCOVIDRiskProfile)
        assert len(profile.ptm_results) == 0
        assert profile.overall_risk_score == 0.0
        assert profile.chronic_activation_probability == 0.0

    def test_analyze_spike_ptms_multiple_sites(self):
        """Test analysis with multiple PTM sites."""
        analyzer = LongCOVIDAnalyzer()

        sites = [
            PTMSite(position=484, ptm_type=PTMType.MUTATION, original="E", modified="K", region=SpikeRegion.RBM),
            PTMSite(position=501, ptm_type=PTMType.MUTATION, original="N", modified="Y", region=SpikeRegion.RBM),
            PTMSite(position=614, ptm_type=PTMType.MUTATION, original="D", modified="G", region=SpikeRegion.SD2),
        ]

        profile = analyzer.analyze_spike_ptms(ptm_sites=sites)

        assert len(profile.ptm_results) == 3
        assert profile.overall_risk_score > 0
        assert profile.microclot_propensity > 0  # RBM mutations present

    def test_compare_variants(self):
        """Test variant comparison."""
        analyzer = LongCOVIDAnalyzer()
        profiles = analyzer.compare_variants(["wuhan", "delta", "omicron_ba1"])

        assert "wuhan" in profiles
        assert "delta" in profiles
        assert "omicron_ba1" in profiles

        # Omicron should have higher risk due to more mutations
        assert profiles["omicron_ba1"].overall_risk_score >= profiles["wuhan"].overall_risk_score

    def test_predict_chronic_activation_base(self):
        """Test chronic activation prediction."""
        analyzer = LongCOVIDAnalyzer()

        sites = [
            PTMSite(position=484, ptm_type=PTMType.MUTATION, original="E", modified="K", region=SpikeRegion.RBM),
        ]
        profile = analyzer.analyze_spike_ptms(ptm_sites=sites)

        predictions = analyzer.predict_chronic_immune_activation(profile)

        assert "base_risk" in predictions
        assert "final_risk" in predictions
        assert "long_covid_probability" in predictions
        assert 0 <= predictions["final_risk"] <= 1

    def test_predict_chronic_activation_with_age(self):
        """Test age adjustment in chronic activation prediction."""
        analyzer = LongCOVIDAnalyzer()

        sites = [
            PTMSite(position=484, ptm_type=PTMType.MUTATION, original="E", modified="K", region=SpikeRegion.RBM),
        ]
        profile = analyzer.analyze_spike_ptms(ptm_sites=sites)

        young_predictions = analyzer.predict_chronic_immune_activation(profile, patient_age=25)
        old_predictions = analyzer.predict_chronic_immune_activation(profile, patient_age=70)

        # Older patients should have higher risk
        assert old_predictions["final_risk"] > young_predictions["final_risk"]

    def test_predict_chronic_activation_with_comorbidities(self):
        """Test comorbidity adjustment."""
        analyzer = LongCOVIDAnalyzer()

        sites = [
            PTMSite(position=484, ptm_type=PTMType.MUTATION, original="E", modified="K", region=SpikeRegion.RBM),
        ]
        profile = analyzer.analyze_spike_ptms(ptm_sites=sites)

        no_comorbidity = analyzer.predict_chronic_immune_activation(profile)
        with_comorbidity = analyzer.predict_chronic_immune_activation(
            profile, comorbidities=["diabetes", "obesity"]
        )

        assert with_comorbidity["final_risk"] > no_comorbidity["final_risk"]

    def test_deletion_structural_impact(self):
        """Test that deletions have high structural impact."""
        analyzer = LongCOVIDAnalyzer()

        deletion = PTMSite(position=69, ptm_type=PTMType.DELETION, original="HV", modified="-", region=SpikeRegion.NTD)

        result = analyzer.analyze_ptm(deletion)
        assert result.structural_impact == "high"


class TestSpikeVariantComparator:
    """Tests for SpikeVariantComparator."""

    def test_initialization(self):
        """Test comparator initialization."""
        comparator = SpikeVariantComparator()
        assert comparator.p == 3

    def test_compute_variant_distance_same(self):
        """Test distance between same variant is zero."""
        comparator = SpikeVariantComparator()
        dist = comparator.compute_variant_distance("wuhan", "wuhan")
        assert dist == 0.0

    def test_compute_variant_distance_different(self):
        """Test distance between different variants."""
        comparator = SpikeVariantComparator()
        dist = comparator.compute_variant_distance("wuhan", "delta")
        assert dist > 0

    def test_compute_variant_distance_unknown(self):
        """Test that unknown variant raises error."""
        comparator = SpikeVariantComparator()
        with pytest.raises(ValueError):
            comparator.compute_variant_distance("wuhan", "unknown_variant")

    def test_compute_distance_matrix(self):
        """Test distance matrix computation."""
        comparator = SpikeVariantComparator()
        names, matrix = comparator.compute_distance_matrix(["wuhan", "alpha", "delta"])

        assert len(names) == 3
        assert matrix.shape == (3, 3)

        # Diagonal should be zero
        assert np.allclose(np.diag(matrix), 0)

        # Matrix should be symmetric
        assert np.allclose(matrix, matrix.T)

    def test_find_closest_variant(self):
        """Test finding closest known variant."""
        comparator = SpikeVariantComparator()

        # Delta mutations should match Delta variant
        delta = KNOWN_VARIANTS["delta"]
        name, dist = comparator.find_closest_variant(delta.mutations)
        assert name == "delta"
        assert dist == 0.0

    def test_rank_variants_by_risk(self):
        """Test variant risk ranking."""
        comparator = SpikeVariantComparator()
        rankings = comparator.rank_variants_by_risk()

        assert len(rankings) == len(KNOWN_VARIANTS)

        # Should be sorted by risk (descending)
        risks = [r[1] for r in rankings]
        assert risks == sorted(risks, reverse=True)


class TestGlycosylationSites:
    """Tests for glycosylation site data."""

    def test_glycosylation_sites_exist(self):
        """Test that glycosylation sites are defined."""
        assert len(SPIKE_GLYCOSYLATION_SITES) > 0

    def test_glycosylation_sites_in_range(self):
        """Test that all sites are within spike protein length."""
        for site in SPIKE_GLYCOSYLATION_SITES:
            assert 1 <= site <= 1273


class TestSpikeRegion:
    """Tests for SpikeRegion enum."""

    def test_region_values(self):
        """Test region enum values."""
        assert SpikeRegion.RBD.value == "Receptor binding domain"
        assert SpikeRegion.NTD.value == "N-terminal domain"
        assert SpikeRegion.FP.value == "Fusion peptide"


class TestPTMType:
    """Tests for PTMType enum."""

    def test_ptm_type_values(self):
        """Test PTM type enum values."""
        assert PTMType.MUTATION.value == "mutation"
        assert PTMType.DELETION.value == "deletion"
        assert PTMType.GLYCOSYLATION.value == "glycosylation"
