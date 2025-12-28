# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Comprehensive unit tests for HIV drug resistance analyzer.

Tests cover:
- Analyzer initialization and configuration
- All 3 genes (RT, PR, IN)
- All 4 drug classes (NRTI, NNRTI, PI, INSTI)
- All 25 individual drugs
- Mutation detection and scoring
- Cross-resistance analysis
- Synthetic data generation
- Sequence encoding
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from src.diseases.hiv_analyzer import (
    HIVAnalyzer,
    HIVConfig,
    HIVGene,
    HIVDrug,
    HIVDrugClass,
    DRUG_TO_CLASS,
    GENE_TO_DRUG_CLASSES,
    GENE_MUTATIONS,
    REFERENCE_SEQUENCES,
    RT_MUTATIONS,
    PR_MUTATIONS,
    IN_MUTATIONS,
    create_hiv_synthetic_dataset,
    get_hiv_drugs_for_gene,
    get_all_hiv_drugs,
    get_hiv_drug_classes,
)


class TestHIVAnalyzerInitialization:
    """Test HIVAnalyzer initialization and configuration."""

    def test_default_initialization(self):
        """Test analyzer initializes with default config."""
        analyzer = HIVAnalyzer()
        assert analyzer is not None
        assert analyzer.config is not None
        assert analyzer.config.name == "hiv"

    def test_custom_config(self):
        """Test analyzer accepts custom configuration."""
        config = HIVConfig(name="custom_hiv")
        analyzer = HIVAnalyzer(config=config)
        assert analyzer.config.name == "custom_hiv"

    def test_config_disease_type(self):
        """Test config has correct disease type."""
        analyzer = HIVAnalyzer()
        from src.diseases.base import DiseaseType
        assert analyzer.config.disease_type == DiseaseType.VIRAL

    def test_amino_acid_alphabet(self):
        """Test amino acid alphabet is properly set."""
        analyzer = HIVAnalyzer()
        assert len(analyzer.aa_alphabet) == 23
        assert "A" in analyzer.aa_alphabet
        assert "X" in analyzer.aa_alphabet
        assert "-" in analyzer.aa_alphabet


class TestHIVGenes:
    """Test HIV gene-related functionality."""

    def test_gene_enum_values(self):
        """Test HIVGene enum has correct values."""
        assert HIVGene.RT.value == "RT"
        assert HIVGene.PR.value == "PR"
        assert HIVGene.IN.value == "IN"

    def test_all_genes_have_reference_sequences(self):
        """Test all genes have reference sequences."""
        for gene in HIVGene:
            assert gene in REFERENCE_SEQUENCES
            assert len(REFERENCE_SEQUENCES[gene]) > 0

    def test_reference_sequence_lengths(self):
        """Test reference sequence lengths are reasonable."""
        assert len(REFERENCE_SEQUENCES[HIVGene.RT]) == 524  # RT is 524 AA (HXB2)
        assert len(REFERENCE_SEQUENCES[HIVGene.PR]) == 99   # PR is 99 AA
        assert len(REFERENCE_SEQUENCES[HIVGene.IN]) == 288  # IN is 288 AA

    def test_all_genes_have_mutations(self):
        """Test all genes have mutation databases."""
        for gene in HIVGene:
            assert gene in GENE_MUTATIONS
            assert len(GENE_MUTATIONS[gene]) > 0

    def test_gene_to_drug_class_mapping(self):
        """Test gene to drug class mapping."""
        assert HIVDrugClass.NRTI in GENE_TO_DRUG_CLASSES[HIVGene.RT]
        assert HIVDrugClass.NNRTI in GENE_TO_DRUG_CLASSES[HIVGene.RT]
        assert HIVDrugClass.PI in GENE_TO_DRUG_CLASSES[HIVGene.PR]
        assert HIVDrugClass.INSTI in GENE_TO_DRUG_CLASSES[HIVGene.IN]


class TestHIVDrugs:
    """Test HIV drug-related functionality."""

    def test_drug_enum_count(self):
        """Test correct number of drugs."""
        assert len(HIVDrug) == 25

    def test_nrti_drugs(self):
        """Test NRTI drugs are correct."""
        nrti_drugs = [d for d, c in DRUG_TO_CLASS.items() if c == HIVDrugClass.NRTI]
        assert len(nrti_drugs) == 7
        assert HIVDrug.LAM in nrti_drugs  # 3TC
        assert HIVDrug.TDF in nrti_drugs
        assert HIVDrug.ABC in nrti_drugs

    def test_nnrti_drugs(self):
        """Test NNRTI drugs are correct."""
        nnrti_drugs = [d for d, c in DRUG_TO_CLASS.items() if c == HIVDrugClass.NNRTI]
        assert len(nnrti_drugs) == 5
        assert HIVDrug.EFV in nnrti_drugs
        assert HIVDrug.NVP in nnrti_drugs

    def test_pi_drugs(self):
        """Test PI drugs are correct."""
        pi_drugs = [d for d, c in DRUG_TO_CLASS.items() if c == HIVDrugClass.PI]
        assert len(pi_drugs) == 8
        assert HIVDrug.DRV in pi_drugs
        assert HIVDrug.ATV in pi_drugs

    def test_insti_drugs(self):
        """Test INSTI drugs are correct."""
        insti_drugs = [d for d, c in DRUG_TO_CLASS.items() if c == HIVDrugClass.INSTI]
        assert len(insti_drugs) == 5
        assert HIVDrug.DTG in insti_drugs
        assert HIVDrug.RAL in insti_drugs

    def test_get_hiv_drugs_for_gene(self):
        """Test get_hiv_drugs_for_gene function."""
        rt_drugs = get_hiv_drugs_for_gene(HIVGene.RT)
        assert len(rt_drugs) == 12  # 7 NRTI + 5 NNRTI

        pr_drugs = get_hiv_drugs_for_gene(HIVGene.PR)
        assert len(pr_drugs) == 8

        in_drugs = get_hiv_drugs_for_gene(HIVGene.IN)
        assert len(in_drugs) == 5

    def test_get_all_hiv_drugs(self):
        """Test get_all_hiv_drugs function."""
        all_drugs = get_all_hiv_drugs()
        assert len(all_drugs) == 25

    def test_get_hiv_drug_classes(self):
        """Test get_hiv_drug_classes function."""
        classes = get_hiv_drug_classes()
        assert len(classes) == 4


class TestRTMutations:
    """Test RT gene mutation database."""

    def test_rt_mutation_count(self):
        """Test RT has sufficient mutations."""
        assert len(RT_MUTATIONS) >= 15

    def test_major_nrti_mutations(self):
        """Test major NRTI mutations are present."""
        # M184V - major 3TC/FTC resistance
        assert 184 in RT_MUTATIONS
        assert "V" in RT_MUTATIONS[184]["M"]["mutations"]

        # K65R - TDF resistance
        assert 65 in RT_MUTATIONS
        assert "R" in RT_MUTATIONS[65]["K"]["mutations"]

    def test_major_nnrti_mutations(self):
        """Test major NNRTI mutations are present."""
        # K103N - major EFV/NVP resistance
        assert 103 in RT_MUTATIONS
        assert "N" in RT_MUTATIONS[103]["K"]["mutations"]

        # Y181C - NVP resistance
        assert 181 in RT_MUTATIONS
        assert "C" in RT_MUTATIONS[181]["Y"]["mutations"]

    def test_tam_mutations(self):
        """Test TAM (Thymidine Analog Mutations) are present."""
        tam_positions = [41, 67, 70, 210, 215, 219]
        for pos in tam_positions:
            assert pos in RT_MUTATIONS

    def test_mutation_effects(self):
        """Test mutations have valid effect levels."""
        valid_effects = {"high", "moderate", "low"}
        for pos, info in RT_MUTATIONS.items():
            ref_aa = list(info.keys())[0]
            assert info[ref_aa]["effect"] in valid_effects


class TestPRMutations:
    """Test PR gene mutation database."""

    def test_pr_mutation_count(self):
        """Test PR has sufficient mutations."""
        assert len(PR_MUTATIONS) >= 10

    def test_major_pi_mutations(self):
        """Test major PI mutations are present."""
        # I84V - cross-resistance
        assert 84 in PR_MUTATIONS
        assert "V" in PR_MUTATIONS[84]["I"]["mutations"]

        # L90M - NFV/SQV resistance
        assert 90 in PR_MUTATIONS
        assert "M" in PR_MUTATIONS[90]["L"]["mutations"]

    def test_drv_mutations(self):
        """Test DRV-specific mutations."""
        drv_positions = [32, 47, 50, 54, 76, 84]
        for pos in drv_positions:
            assert pos in PR_MUTATIONS


class TestINMutations:
    """Test IN gene mutation database."""

    def test_in_mutation_count(self):
        """Test IN has sufficient mutations."""
        assert len(IN_MUTATIONS) >= 8

    def test_major_insti_mutations(self):
        """Test major INSTI mutations are present."""
        # Q148H/K/R - major resistance
        assert 148 in IN_MUTATIONS
        assert "H" in IN_MUTATIONS[148]["Q"]["mutations"]
        assert "K" in IN_MUTATIONS[148]["Q"]["mutations"]
        assert "R" in IN_MUTATIONS[148]["Q"]["mutations"]

        # N155H - RAL/EVG resistance
        assert 155 in IN_MUTATIONS
        assert "H" in IN_MUTATIONS[155]["N"]["mutations"]


class TestHIVAnalyzerAnalyze:
    """Test HIVAnalyzer.analyze() method."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return HIVAnalyzer()

    @pytest.fixture
    def wild_type_sequences(self):
        """Create wild-type sequence dict."""
        return {
            HIVGene.RT: [REFERENCE_SEQUENCES[HIVGene.RT]],
            HIVGene.PR: [REFERENCE_SEQUENCES[HIVGene.PR]],
            HIVGene.IN: [REFERENCE_SEQUENCES[HIVGene.IN]],
        }

    def test_analyze_returns_dict(self, analyzer, wild_type_sequences):
        """Test analyze returns dictionary."""
        results = analyzer.analyze(wild_type_sequences)
        assert isinstance(results, dict)

    def test_analyze_has_required_keys(self, analyzer, wild_type_sequences):
        """Test analyze returns required keys."""
        results = analyzer.analyze(wild_type_sequences)
        assert "n_sequences" in results
        assert "genes_analyzed" in results
        assert "drug_resistance" in results
        assert "mutations_detected" in results

    def test_wild_type_has_low_resistance(self, analyzer, wild_type_sequences):
        """Test wild-type sequences have low resistance scores."""
        results = analyzer.analyze(wild_type_sequences)

        for drug, data in results["drug_resistance"].items():
            if data["scores"]:
                # Wild-type should have low/zero resistance
                assert data["scores"][0] < 0.3

    def test_analyze_detects_genes(self, analyzer, wild_type_sequences):
        """Test analyze detects correct genes."""
        results = analyzer.analyze(wild_type_sequences)
        assert "RT" in results["genes_analyzed"]
        assert "PR" in results["genes_analyzed"]
        assert "IN" in results["genes_analyzed"]


class TestHIVAnalyzerMutantDetection:
    """Test mutation detection in HIVAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return HIVAnalyzer()

    def test_detect_m184v_mutation(self, analyzer):
        """Test detection of M184V (major 3TC resistance)."""
        reference = list(REFERENCE_SEQUENCES[HIVGene.RT])
        reference[183] = "V"  # M184V (0-indexed)
        mutant_seq = "".join(reference)

        results = analyzer.analyze({HIVGene.RT: [mutant_seq]})

        # Should have elevated resistance for 3TC/FTC
        ftc_scores = results["drug_resistance"].get("FTC", {}).get("scores", [])
        lam_scores = results["drug_resistance"].get("3TC", {}).get("scores", [])

        assert ftc_scores and ftc_scores[0] > 0.2
        assert lam_scores and lam_scores[0] > 0.2

    def test_detect_k103n_mutation(self, analyzer):
        """Test detection of K103N (major NNRTI resistance)."""
        reference = list(REFERENCE_SEQUENCES[HIVGene.RT])
        reference[102] = "N"  # K103N (0-indexed)
        mutant_seq = "".join(reference)

        results = analyzer.analyze({HIVGene.RT: [mutant_seq]})

        # Should have elevated resistance for EFV/NVP
        efv_scores = results["drug_resistance"].get("EFV", {}).get("scores", [])
        nvp_scores = results["drug_resistance"].get("NVP", {}).get("scores", [])

        assert efv_scores and efv_scores[0] > 0.2
        assert nvp_scores and nvp_scores[0] > 0.2

    def test_detect_q148h_mutation(self, analyzer):
        """Test detection of Q148H (major INSTI resistance)."""
        reference = list(REFERENCE_SEQUENCES[HIVGene.IN])
        reference[147] = "H"  # Q148H (0-indexed)
        mutant_seq = "".join(reference)

        results = analyzer.analyze({HIVGene.IN: [mutant_seq]})

        # Should have elevated resistance for RAL/EVG
        ral_scores = results["drug_resistance"].get("RAL", {}).get("scores", [])
        evg_scores = results["drug_resistance"].get("EVG", {}).get("scores", [])

        assert ral_scores and ral_scores[0] > 0.2
        assert evg_scores and evg_scores[0] > 0.2


class TestHIVSequenceEncoding:
    """Test sequence encoding functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return HIVAnalyzer()

    def test_encode_sequence_returns_array(self, analyzer):
        """Test encode_sequence returns numpy array."""
        encoding = analyzer.encode_sequence("MKLTVFG")
        assert isinstance(encoding, np.ndarray)

    def test_encode_sequence_shape(self, analyzer):
        """Test encoding has correct shape."""
        encoding = analyzer.encode_sequence("MKLTVFG", max_length=100)
        expected_size = 100 * 23  # max_length * alphabet_size
        assert encoding.shape == (expected_size,)

    def test_encode_sequence_one_hot(self, analyzer):
        """Test encoding is one-hot."""
        encoding = analyzer.encode_sequence("M", max_length=10)
        # Should have exactly one 1.0 per position
        reshaped = encoding.reshape(10, 23)
        assert np.allclose(reshaped[0].sum(), 1.0)

    def test_encode_empty_sequence(self, analyzer):
        """Test encoding empty sequence."""
        encoding = analyzer.encode_sequence("", max_length=10)
        assert encoding.shape == (10 * 23,)

    def test_encode_with_unknown_aa(self, analyzer):
        """Test encoding with unknown amino acid."""
        encoding = analyzer.encode_sequence("MX", max_length=10)
        # Should not raise, X should map to unknown index
        assert encoding.shape == (10 * 23,)


class TestHIVSyntheticDataset:
    """Test synthetic dataset generation."""

    def test_create_dataset_returns_tuple(self):
        """Test create_hiv_synthetic_dataset returns tuple."""
        X, y, ids = create_hiv_synthetic_dataset(min_samples=30)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(ids, list)

    def test_create_dataset_shapes(self):
        """Test dataset has correct shapes."""
        X, y, ids = create_hiv_synthetic_dataset(min_samples=50)
        assert X.shape[0] == len(y)
        assert X.shape[0] == len(ids)
        assert X.shape[0] >= 50

    def test_create_dataset_targets_range(self):
        """Test targets are in valid range."""
        X, y, ids = create_hiv_synthetic_dataset(min_samples=30)
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_create_dataset_has_wild_type(self):
        """Test dataset includes wild-type."""
        X, y, ids = create_hiv_synthetic_dataset(min_samples=30)
        assert "WT" in ids

    def test_create_dataset_correlation(self):
        """Test synthetic data has reasonable correlation."""
        X, y, ids = create_hiv_synthetic_dataset(min_samples=50)

        # Simple prediction using feature variance
        feature_var = np.var(X, axis=0)
        top_features = np.argsort(feature_var)[-50:]
        pred = X[:, top_features].sum(axis=1)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        # Should have some correlation structure
        rho, _ = spearmanr(y, pred)
        assert abs(rho) > 0.1  # At least weak correlation

    def test_create_dataset_different_genes(self):
        """Test dataset creation for different genes."""
        for gene in [HIVGene.RT, HIVGene.PR, HIVGene.IN]:
            X, y, ids = create_hiv_synthetic_dataset(gene=gene, min_samples=30)
            assert X.shape[0] >= 30

    def test_create_dataset_with_drug_class_filter(self):
        """Test dataset creation with drug class filter."""
        X, y, ids = create_hiv_synthetic_dataset(
            gene=HIVGene.RT,
            drug_class=HIVDrugClass.NRTI,
            min_samples=30
        )
        assert X.shape[0] >= 30


class TestHIVValidation:
    """Test validation functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return HIVAnalyzer()

    def test_validate_predictions(self, analyzer):
        """Test validate_predictions method."""
        # Create mock predictions
        predictions = {
            "drug_resistance": {
                "EFV": {"scores": [0.1, 0.5, 0.9]},
                "NVP": {"scores": [0.2, 0.6, 0.8]},
            }
        }

        # Create mock ground truth
        ground_truth = {
            "EFV": [0.0, 0.5, 1.0],
            "NVP": [0.1, 0.7, 0.9],
        }

        metrics = analyzer.validate_predictions(predictions, ground_truth)

        assert "EFV_spearman" in metrics
        assert "NVP_spearman" in metrics
        assert -1.0 <= metrics["EFV_spearman"] <= 1.0


class TestHIVDrugClassSummary:
    """Test drug class summary functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return HIVAnalyzer()

    def test_drug_class_summary_generated(self, analyzer):
        """Test drug class summary is generated."""
        sequences = {HIVGene.RT: [REFERENCE_SEQUENCES[HIVGene.RT]]}
        results = analyzer.analyze(sequences)

        assert "drug_class_summary" in results

    def test_drug_class_summary_structure(self, analyzer):
        """Test drug class summary structure."""
        sequences = {
            HIVGene.RT: [REFERENCE_SEQUENCES[HIVGene.RT]],
            HIVGene.PR: [REFERENCE_SEQUENCES[HIVGene.PR]],
            HIVGene.IN: [REFERENCE_SEQUENCES[HIVGene.IN]],
        }
        results = analyzer.analyze(sequences)

        for drug_class in results.get("drug_class_summary", {}):
            summary = results["drug_class_summary"][drug_class]
            assert "mean_score" in summary
            assert "max_score" in summary
            assert "n_drugs" in summary


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return HIVAnalyzer()

    def test_empty_sequences(self, analyzer):
        """Test handling of empty sequences."""
        results = analyzer.analyze({})
        assert results["n_sequences"] == 0

    def test_single_gene(self, analyzer):
        """Test analysis with single gene."""
        sequences = {HIVGene.RT: [REFERENCE_SEQUENCES[HIVGene.RT]]}
        results = analyzer.analyze(sequences)
        assert "RT" in results["genes_analyzed"]

    def test_short_sequence(self, analyzer):
        """Test handling of short sequence."""
        sequences = {HIVGene.RT: ["MKL"]}  # Very short
        results = analyzer.analyze(sequences)
        # Should not crash, but may not detect mutations
        assert "drug_resistance" in results

    def test_multiple_sequences(self, analyzer):
        """Test analysis with multiple sequences per gene."""
        ref = REFERENCE_SEQUENCES[HIVGene.RT]
        sequences = {HIVGene.RT: [ref, ref, ref]}
        results = analyzer.analyze(sequences)

        for drug, data in results["drug_resistance"].items():
            if data["scores"]:
                assert len(data["scores"]) == 3


# Integration test
class TestHIVIntegration:
    """Integration tests for HIV analyzer."""

    def test_full_pipeline(self):
        """Test full analysis pipeline."""
        # Create synthetic data
        X, y, ids = create_hiv_synthetic_dataset(min_samples=50)

        # Verify data quality
        assert X.shape[0] >= 50
        assert len(np.unique(y)) > 5  # Has variety in targets

        # Create analyzer and run
        analyzer = HIVAnalyzer()
        sequences = {HIVGene.RT: [REFERENCE_SEQUENCES[HIVGene.RT]]}
        results = analyzer.analyze(sequences)

        # Verify output
        assert results["n_sequences"] == 1
        assert len(results["drug_resistance"]) > 0

    def test_cross_gene_analysis(self):
        """Test analysis across all genes."""
        analyzer = HIVAnalyzer()

        sequences = {
            HIVGene.RT: [REFERENCE_SEQUENCES[HIVGene.RT]],
            HIVGene.PR: [REFERENCE_SEQUENCES[HIVGene.PR]],
            HIVGene.IN: [REFERENCE_SEQUENCES[HIVGene.IN]],
        }

        results = analyzer.analyze(sequences)

        # Should have results for drugs from all classes
        drug_classes_found = set()
        for drug in results["drug_resistance"]:
            for d in HIVDrug:
                if d.value == drug:
                    drug_classes_found.add(DRUG_TO_CLASS[d])

        # Should find drugs from all 4 classes
        assert len(drug_classes_found) == 4
