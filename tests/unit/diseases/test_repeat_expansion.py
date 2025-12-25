# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for trinucleotide repeat expansion disease analyzer."""

import pytest

from src.diseases.repeat_expansion import (
    REPEAT_DISEASES,
    RepeatAnalysisResult,
    RepeatDiseaseInfo,
    RepeatExpansionAnalyzer,
    TrinucleotideRepeat,
)


class TestRepeatDiseaseInfo:
    """Tests for RepeatDiseaseInfo dataclass."""

    def test_huntington_disease_info(self):
        """Test Huntington's disease information."""
        hd = REPEAT_DISEASES["huntington"]
        assert hd.name == "Huntington's Disease"
        assert hd.repeat == TrinucleotideRepeat.CAG
        assert hd.gene == "HTT"
        assert hd.disease_threshold == 36

    def test_is_normal(self):
        """Test normal range classification."""
        hd = REPEAT_DISEASES["huntington"]
        assert hd.is_normal(20) is True
        assert hd.is_normal(26) is True
        assert hd.is_normal(30) is False
        assert hd.is_normal(40) is False

    def test_is_intermediate(self):
        """Test intermediate range classification."""
        hd = REPEAT_DISEASES["huntington"]
        assert hd.is_intermediate(20) is False
        assert hd.is_intermediate(27) is True
        assert hd.is_intermediate(35) is True
        assert hd.is_intermediate(36) is False

    def test_is_disease(self):
        """Test disease classification."""
        hd = REPEAT_DISEASES["huntington"]
        assert hd.is_disease(20) is False
        assert hd.is_disease(35) is False
        assert hd.is_disease(36) is True
        assert hd.is_disease(50) is True


class TestRepeatExpansionAnalyzer:
    """Tests for RepeatExpansionAnalyzer."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = RepeatExpansionAnalyzer()
        assert analyzer.p == 3
        assert analyzer.goldilocks_center == 0.5

    def test_initialization_custom_params(self):
        """Test analyzer with custom parameters."""
        analyzer = RepeatExpansionAnalyzer(p=5, goldilocks_center=0.6)
        assert analyzer.p == 5
        assert analyzer.goldilocks_center == 0.6

    def test_padic_valuation(self):
        """Test p-adic valuation computation."""
        analyzer = RepeatExpansionAnalyzer(p=3)

        # v_3(9) = 2 since 9 = 3^2
        assert analyzer._compute_padic_valuation(9) == 2

        # v_3(27) = 3 since 27 = 3^3
        assert analyzer._compute_padic_valuation(27) == 3

        # v_3(10) = 0 since 10 is not divisible by 3
        assert analyzer._compute_padic_valuation(10) == 0

        # v_3(0) should return large value (infinity)
        assert analyzer._compute_padic_valuation(0) >= 100

    def test_analyze_normal_repeat(self):
        """Test analysis of normal repeat count."""
        analyzer = RepeatExpansionAnalyzer()
        result = analyzer.analyze_repeat_padic_distance("huntington", 20)

        assert result.disease == "huntington"
        assert result.repeat_count == 20
        assert result.classification == "normal"
        assert result.risk_score < 0.3
        assert result.predicted_onset_age is None

    def test_analyze_intermediate_repeat(self):
        """Test analysis of intermediate repeat count."""
        analyzer = RepeatExpansionAnalyzer()
        result = analyzer.analyze_repeat_padic_distance("huntington", 30)

        assert result.classification == "intermediate"
        assert 0.3 <= result.risk_score <= 0.6
        assert result.predicted_onset_age is None

    def test_analyze_disease_repeat(self):
        """Test analysis of disease-causing repeat count."""
        analyzer = RepeatExpansionAnalyzer()
        result = analyzer.analyze_repeat_padic_distance("huntington", 42)

        assert result.classification == "disease"
        assert result.risk_score > 0.7
        assert result.predicted_onset_age is not None
        assert 1 <= result.predicted_onset_age <= 100

    def test_analyze_high_repeat_count(self):
        """Test analysis of very high repeat count."""
        analyzer = RepeatExpansionAnalyzer()
        result = analyzer.analyze_repeat_padic_distance("huntington", 70)

        assert result.classification == "disease"
        assert result.risk_score > 0.9
        assert result.aggregation_propensity > 0.5
        # Higher repeat counts should predict earlier onset
        assert result.predicted_onset_age is not None
        assert result.predicted_onset_age < 30

    def test_unknown_disease_raises_error(self):
        """Test that unknown disease raises ValueError."""
        analyzer = RepeatExpansionAnalyzer()
        with pytest.raises(ValueError, match="Unknown disease"):
            analyzer.analyze_repeat_padic_distance("unknown_disease", 40)

    def test_find_disease_boundary(self):
        """Test finding disease boundaries."""
        analyzer = RepeatExpansionAnalyzer()
        boundaries = analyzer.find_disease_boundary("huntington")

        assert "clinical_normal_end" in boundaries
        assert "intermediate_start" in boundaries
        assert "disease_threshold" in boundaries
        assert "full_penetrance" in boundaries

        assert boundaries["clinical_normal_end"] == 26
        assert boundaries["disease_threshold"] == 36
        assert boundaries["full_penetrance"] == 40

    def test_compare_diseases(self):
        """Test comparing risk across diseases."""
        analyzer = RepeatExpansionAnalyzer()
        results = analyzer.compare_diseases(40)

        assert len(results) == len(REPEAT_DISEASES)
        # Results should be sorted by risk score
        risk_scores = [r.risk_score for r in results]
        assert risk_scores == sorted(risk_scores, reverse=True)

    def test_generate_risk_trajectory(self):
        """Test generating risk trajectory."""
        analyzer = RepeatExpansionAnalyzer()
        counts, risks, goldilocks = analyzer.generate_risk_trajectory(
            "huntington", repeat_range=(20, 50)
        )

        assert len(counts) == 31  # 20 to 50 inclusive
        assert len(risks) == 31
        assert len(goldilocks) == 31

        # Risk should generally increase with repeat count
        assert risks[-1] > risks[0]

    def test_fragile_x_threshold(self):
        """Test Fragile X with its high threshold."""
        analyzer = RepeatExpansionAnalyzer()

        # Normal range for Fragile X (5-44)
        result_normal = analyzer.analyze_repeat_padic_distance("fragile_x", 30)
        assert result_normal.classification == "normal"

        # Intermediate/Gray zone (45-54)
        result_intermediate = analyzer.analyze_repeat_padic_distance("fragile_x", 50)
        assert result_intermediate.classification == "intermediate"

        # Premutation range (55-199) - classified as disease in current model
        # Note: Clinically this is "premutation" (carriers may have FXTAS/FXPOI)
        result_premutation = analyzer.analyze_repeat_padic_distance("fragile_x", 100)
        assert result_premutation.classification == "disease"  # Beyond intermediate

        # Full mutation threshold (200+)
        result_disease = analyzer.analyze_repeat_padic_distance("fragile_x", 200)
        assert result_disease.classification == "disease"

    def test_friedreich_ataxia_no_anticipation(self):
        """Test Friedreich's ataxia (no anticipation pattern)."""
        fa = REPEAT_DISEASES["friedreich_ataxia"]
        assert fa.anticipation is False

    def test_aggregation_propensity_by_repeat_type(self):
        """Test that CAG repeats have higher aggregation propensity."""
        analyzer = RepeatExpansionAnalyzer()

        # CAG repeats (polyglutamine) should have higher aggregation
        hd_result = analyzer.analyze_repeat_padic_distance("huntington", 45)

        # GAA repeats (Friedreich's) should have lower aggregation
        fa_result = analyzer.analyze_repeat_padic_distance("friedreich_ataxia", 100)

        # CAG-based diseases tend to have higher aggregation
        assert hd_result.aggregation_propensity > fa_result.aggregation_propensity


class TestTrinucleotideRepeat:
    """Tests for TrinucleotideRepeat enum."""

    def test_repeat_types(self):
        """Test repeat type values."""
        assert TrinucleotideRepeat.CAG.value == "CAG"
        assert TrinucleotideRepeat.CGG.value == "CGG"
        assert TrinucleotideRepeat.CTG.value == "CTG"
        assert TrinucleotideRepeat.GAA.value == "GAA"

    def test_all_diseases_have_valid_repeat_type(self):
        """Test that all diseases have valid repeat types."""
        for disease_name, info in REPEAT_DISEASES.items():
            assert isinstance(info.repeat, TrinucleotideRepeat)
