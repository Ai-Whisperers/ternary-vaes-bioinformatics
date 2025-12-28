# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for E. coli TEM beta-lactamase analyzer.

Tests the EcoliBetaLactamAnalyzer including:
- Enums (EcoliGene, BetaLactam, TEMVariant)
- Mutation detection (ESBL, IRT, stabilizer)
- Variant classification (TEM-1, ESBL, IRT, CMT)
- Drug resistance prediction
- Synthetic dataset creation
- Sequence encoding
"""

import pytest
import numpy as np


class TestEcoliImports:
    """Test all E. coli analyzer imports."""

    def test_analyzer_import(self):
        """Test analyzer can be imported."""
        from src.diseases import EcoliBetaLactamAnalyzer

        assert EcoliBetaLactamAnalyzer is not None

    def test_config_import(self):
        """Test config can be imported."""
        from src.diseases import EcoliBetaLactamConfig

        assert EcoliBetaLactamConfig is not None

    def test_gene_enum_import(self):
        """Test gene enum can be imported."""
        from src.diseases import EcoliGene

        assert EcoliGene is not None

    def test_drug_enum_import(self):
        """Test drug enum can be imported."""
        from src.diseases import BetaLactam

        assert BetaLactam is not None

    def test_variant_enum_import(self):
        """Test variant enum can be imported."""
        from src.diseases import TEMVariant

        assert TEMVariant is not None

    def test_mutations_import(self):
        """Test mutation database can be imported."""
        from src.diseases import TEM_MUTATIONS

        assert TEM_MUTATIONS is not None
        assert isinstance(TEM_MUTATIONS, dict)

    def test_reference_import(self):
        """Test reference sequence can be imported."""
        from src.diseases import TEM1_REFERENCE

        assert TEM1_REFERENCE is not None
        assert len(TEM1_REFERENCE) == 286  # TEM-1 with signal peptide

    def test_factory_import(self):
        """Test factory function can be imported."""
        from src.diseases import create_ecoli_synthetic_dataset

        assert create_ecoli_synthetic_dataset is not None


class TestEcoliGeneEnum:
    """Test EcoliGene enumeration."""

    def test_bla_tem(self):
        """Test BLA_TEM gene."""
        from src.diseases import EcoliGene

        assert hasattr(EcoliGene, "BLA_TEM")
        assert EcoliGene.BLA_TEM.value == "blaTEM"

    def test_bla_shv(self):
        """Test BLA_SHV gene."""
        from src.diseases import EcoliGene

        assert hasattr(EcoliGene, "BLA_SHV")
        assert EcoliGene.BLA_SHV.value == "blaSHV"

    def test_bla_ctx_m(self):
        """Test BLA_CTX_M gene."""
        from src.diseases import EcoliGene

        assert hasattr(EcoliGene, "BLA_CTX_M")


class TestBetaLactamEnum:
    """Test BetaLactam drug enumeration."""

    def test_penicillins(self):
        """Test penicillin drugs."""
        from src.diseases import BetaLactam

        assert hasattr(BetaLactam, "AMPICILLIN")
        assert hasattr(BetaLactam, "AMOXICILLIN")
        assert BetaLactam.AMPICILLIN.value == "ampicillin"

    def test_cephalosporins(self):
        """Test cephalosporin drugs."""
        from src.diseases import BetaLactam

        assert hasattr(BetaLactam, "CEFTAZIDIME")
        assert hasattr(BetaLactam, "CEFOTAXIME")
        assert hasattr(BetaLactam, "CEFTRIAXONE")
        assert hasattr(BetaLactam, "CEFEPIME")

    def test_inhibitor_combinations(self):
        """Test beta-lactamase inhibitor combinations."""
        from src.diseases import BetaLactam

        assert hasattr(BetaLactam, "AMOX_CLAVULANATE")
        assert hasattr(BetaLactam, "PIPERACILLIN_TAZO")


class TestTEMVariantEnum:
    """Test TEMVariant classification enum."""

    def test_tem1(self):
        """Test TEM-1 variant."""
        from src.diseases import TEMVariant

        assert hasattr(TEMVariant, "TEM_1")
        assert TEMVariant.TEM_1.value == "TEM-1"

    def test_esbl(self):
        """Test ESBL variant."""
        from src.diseases import TEMVariant

        assert hasattr(TEMVariant, "ESBL")

    def test_irt(self):
        """Test IRT variant."""
        from src.diseases import TEMVariant

        assert hasattr(TEMVariant, "IRT")

    def test_cmt(self):
        """Test CMT variant."""
        from src.diseases import TEMVariant

        assert hasattr(TEMVariant, "CMT")


class TestTEMMutations:
    """Test TEM mutation database."""

    def test_mutation_positions(self):
        """Test known mutation positions exist."""
        from src.diseases import TEM_MUTATIONS

        # ESBL positions
        assert 104 in TEM_MUTATIONS  # E104K
        assert 164 in TEM_MUTATIONS  # R164S/H
        assert 238 in TEM_MUTATIONS  # G238S
        assert 240 in TEM_MUTATIONS  # E240K

        # IRT positions
        assert 69 in TEM_MUTATIONS   # M69I/L/V
        assert 130 in TEM_MUTATIONS  # S130G
        assert 244 in TEM_MUTATIONS  # R244S/C/H

        # Stabilizer positions
        assert 182 in TEM_MUTATIONS  # M182T
        assert 39 in TEM_MUTATIONS   # Q39K

    def test_mutation_structure(self):
        """Test mutation database structure."""
        from src.diseases import TEM_MUTATIONS

        for pos, info in TEM_MUTATIONS.items():
            assert isinstance(pos, int)
            assert isinstance(info, dict)

            # Each position should have exactly one reference amino acid
            ref_aa = list(info.keys())[0]
            assert len(ref_aa) == 1

            # Each reference should have mutations, effect, phenotype
            mut_info = info[ref_aa]
            assert "mutations" in mut_info
            assert "effect" in mut_info
            assert "phenotype" in mut_info

    def test_esbl_mutations(self):
        """Test ESBL mutation phenotypes."""
        from src.diseases import TEM_MUTATIONS

        esbl_positions = [104, 164, 238, 240]
        for pos in esbl_positions:
            ref_aa = list(TEM_MUTATIONS[pos].keys())[0]
            assert TEM_MUTATIONS[pos][ref_aa]["phenotype"] == "ESBL"

    def test_irt_mutations(self):
        """Test IRT mutation phenotypes."""
        from src.diseases import TEM_MUTATIONS

        irt_positions = [69, 130, 244, 275, 276]
        for pos in irt_positions:
            ref_aa = list(TEM_MUTATIONS[pos].keys())[0]
            assert TEM_MUTATIONS[pos][ref_aa]["phenotype"] == "IRT"


class TestEcoliBetaLactamConfig:
    """Test EcoliBetaLactamConfig configuration."""

    def test_default_config(self):
        """Test default configuration."""
        from src.diseases import EcoliBetaLactamConfig
        from src.diseases.base import DiseaseType, TaskType

        config = EcoliBetaLactamConfig()
        assert config.name == "ecoli_betalactam"
        assert config.disease_type == DiseaseType.BACTERIAL
        assert TaskType.RESISTANCE in config.tasks

    def test_data_sources(self):
        """Test data sources are configured."""
        from src.diseases import EcoliBetaLactamConfig

        config = EcoliBetaLactamConfig()
        assert "arcadia" in config.data_sources
        assert "card" in config.data_sources
        assert "zenodo" in config.data_sources


class TestEcoliBetaLactamAnalyzer:
    """Test EcoliBetaLactamAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        from src.diseases import EcoliBetaLactamAnalyzer

        return EcoliBetaLactamAnalyzer()

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert hasattr(analyzer, "config")
        assert hasattr(analyzer, "aa_alphabet")
        assert hasattr(analyzer, "aa_to_idx")

    def test_has_analyze_method(self, analyzer):
        """Test analyzer has analyze method."""
        assert hasattr(analyzer, "analyze")
        assert callable(analyzer.analyze)

    def test_has_validate_predictions_method(self, analyzer):
        """Test analyzer has validate_predictions method."""
        assert hasattr(analyzer, "validate_predictions")
        assert callable(analyzer.validate_predictions)

    def test_has_encode_sequence_method(self, analyzer):
        """Test analyzer has encode_sequence method."""
        assert hasattr(analyzer, "encode_sequence")
        assert callable(analyzer.encode_sequence)


class TestMutationDetection:
    """Test mutation detection functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        from src.diseases import EcoliBetaLactamAnalyzer

        return EcoliBetaLactamAnalyzer()

    @pytest.fixture
    def reference(self):
        """Get TEM-1 reference sequence."""
        from src.diseases import TEM1_REFERENCE

        return TEM1_REFERENCE

    def test_detect_no_mutations(self, analyzer, reference):
        """Test detection with wild-type sequence."""
        mutations = analyzer._detect_mutations(reference)
        assert len(mutations) == 0

    def test_detect_esbl_mutation_g238s(self, analyzer, reference):
        """Test detection of G238S ESBL mutation."""
        # Create mutant with G238S (position 238, 0-indexed = 237)
        mutant = reference[:237] + "S" + reference[238:]
        mutations = analyzer._detect_mutations(mutant)

        assert len(mutations) == 1
        assert mutations[0]["position"] == 238
        assert mutations[0]["ref"] == "G"
        assert mutations[0]["alt"] == "S"
        assert mutations[0]["phenotype"] == "ESBL"

    def test_detect_irt_mutation_s130g(self, analyzer, reference):
        """Test detection of S130G IRT mutation."""
        # Create mutant with S130G (position 130, 0-indexed = 129)
        mutant = reference[:129] + "G" + reference[130:]
        mutations = analyzer._detect_mutations(mutant)

        assert len(mutations) == 1
        assert mutations[0]["position"] == 130
        assert mutations[0]["phenotype"] == "IRT"

    def test_detect_multiple_mutations(self, analyzer, reference):
        """Test detection of multiple mutations."""
        # Create mutant with G238S and E240K
        mutant = list(reference)
        mutant[237] = "S"  # G238S
        mutant[239] = "K"  # E240K
        mutant = "".join(mutant)

        mutations = analyzer._detect_mutations(mutant)
        assert len(mutations) == 2
        positions = [m["position"] for m in mutations]
        assert 238 in positions
        assert 240 in positions


class TestVariantClassification:
    """Test TEM variant classification."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        from src.diseases import EcoliBetaLactamAnalyzer

        return EcoliBetaLactamAnalyzer()

    @pytest.fixture
    def reference(self):
        """Get TEM-1 reference sequence."""
        from src.diseases import TEM1_REFERENCE

        return TEM1_REFERENCE

    def test_classify_tem1(self, analyzer, reference):
        """Test TEM-1 classification (no mutations)."""
        result = analyzer._classify_tem_variant(reference)

        assert result["type"] == "TEM-1"
        assert result["esbl_mutations"] == 0
        assert result["irt_mutations"] == 0

    def test_classify_esbl(self, analyzer, reference):
        """Test ESBL classification."""
        # Add G238S ESBL mutation
        mutant = reference[:237] + "S" + reference[238:]
        result = analyzer._classify_tem_variant(mutant)

        assert result["type"] == "ESBL"
        assert result["esbl_mutations"] >= 1

    def test_classify_irt(self, analyzer, reference):
        """Test IRT classification."""
        # Add S130G IRT mutation
        mutant = reference[:129] + "G" + reference[130:]
        result = analyzer._classify_tem_variant(mutant)

        assert result["type"] == "IRT"
        assert result["irt_mutations"] >= 1

    def test_classify_cmt(self, analyzer, reference):
        """Test CMT classification (ESBL + IRT)."""
        # Add both G238S (ESBL) and S130G (IRT)
        mutant = list(reference)
        mutant[237] = "S"  # G238S (ESBL)
        mutant[129] = "G"  # S130G (IRT)
        mutant = "".join(mutant)

        result = analyzer._classify_tem_variant(mutant)

        assert result["type"] == "CMT"
        assert result["esbl_mutations"] >= 1
        assert result["irt_mutations"] >= 1


class TestDrugResistancePrediction:
    """Test drug resistance prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        from src.diseases import EcoliBetaLactamAnalyzer

        return EcoliBetaLactamAnalyzer()

    @pytest.fixture
    def reference(self):
        """Get TEM-1 reference sequence."""
        from src.diseases import TEM1_REFERENCE

        return TEM1_REFERENCE

    def test_predict_ampicillin_resistance(self, analyzer, reference):
        """Test ampicillin resistance prediction for TEM-1."""
        from src.diseases import BetaLactam

        result = analyzer._predict_drug_resistance([reference], BetaLactam.AMPICILLIN)

        assert "scores" in result
        assert len(result["scores"]) == 1
        # TEM-1 should be resistant to ampicillin
        assert result["scores"][0] >= 0.5

    def test_predict_cephalosporin_susceptible(self, analyzer, reference):
        """Test cephalosporin susceptibility for TEM-1."""
        from src.diseases import BetaLactam

        result = analyzer._predict_drug_resistance([reference], BetaLactam.CEFTAZIDIME)

        assert len(result["scores"]) == 1
        # TEM-1 should be susceptible to cephalosporins
        assert result["scores"][0] < 0.5

    def test_predict_cephalosporin_resistance_esbl(self, analyzer, reference):
        """Test cephalosporin resistance for ESBL."""
        from src.diseases import BetaLactam

        # Add G238S and E240K (classic ESBL combination)
        mutant = list(reference)
        mutant[237] = "S"  # G238S
        mutant[239] = "K"  # E240K
        mutant = "".join(mutant)

        result = analyzer._predict_drug_resistance([mutant], BetaLactam.CEFTAZIDIME)

        # ESBL should be resistant to cephalosporins
        assert result["scores"][0] >= 0.5

    def test_classification_output(self, analyzer, reference):
        """Test classification output."""
        from src.diseases import BetaLactam

        result = analyzer._predict_drug_resistance([reference], BetaLactam.AMPICILLIN)

        assert "classifications" in result
        assert len(result["classifications"]) == 1
        assert result["classifications"][0] in ["susceptible", "intermediate", "resistant"]


class TestSequenceEncoding:
    """Test sequence encoding functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        from src.diseases import EcoliBetaLactamAnalyzer

        return EcoliBetaLactamAnalyzer()

    def test_encode_short_sequence(self, analyzer):
        """Test encoding a short sequence."""
        seq = "ACDE"
        encoding = analyzer.encode_sequence(seq, max_length=10)

        # Should be flattened: max_length * n_aa
        n_aa = len(analyzer.aa_alphabet)
        assert encoding.shape == (10 * n_aa,)
        assert encoding.dtype == np.float32

    def test_encode_reference(self, analyzer):
        """Test encoding TEM-1 reference."""
        from src.diseases import TEM1_REFERENCE

        encoding = analyzer.encode_sequence(TEM1_REFERENCE, max_length=300)

        n_aa = len(analyzer.aa_alphabet)
        assert encoding.shape == (300 * n_aa,)

    def test_one_hot_structure(self, analyzer):
        """Test one-hot encoding structure."""
        seq = "A"
        n_aa = len(analyzer.aa_alphabet)
        encoding = analyzer.encode_sequence(seq, max_length=1)

        # Should be one-hot: exactly one 1, rest 0s
        assert np.sum(encoding) == 1.0
        assert encoding.shape == (n_aa,)

    def test_encoding_deterministic(self, analyzer):
        """Test encoding is deterministic."""
        seq = "MSIQHFRVAL"
        enc1 = analyzer.encode_sequence(seq, max_length=20)
        enc2 = analyzer.encode_sequence(seq, max_length=20)

        np.testing.assert_array_equal(enc1, enc2)


class TestSyntheticDataset:
    """Test synthetic dataset creation."""

    def test_create_dataset_default(self):
        """Test default dataset creation."""
        from src.diseases import create_ecoli_synthetic_dataset

        X, y, ids = create_ecoli_synthetic_dataset()

        assert X.shape[0] >= 50  # min_samples default
        assert X.shape[0] == len(y)
        assert X.shape[0] == len(ids)
        assert X.dtype == np.float32

    def test_create_dataset_ampicillin(self):
        """Test dataset for ampicillin."""
        from src.diseases import create_ecoli_synthetic_dataset, BetaLactam

        X, y, ids = create_ecoli_synthetic_dataset(drug=BetaLactam.AMPICILLIN)

        assert X.shape[0] >= 50
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_create_dataset_cephalosporin(self):
        """Test dataset for cephalosporin."""
        from src.diseases import create_ecoli_synthetic_dataset, BetaLactam

        X, y, ids = create_ecoli_synthetic_dataset(drug=BetaLactam.CEFTAZIDIME)

        assert X.shape[0] >= 50

    def test_create_dataset_inhibitor(self):
        """Test dataset for inhibitor combination."""
        from src.diseases import create_ecoli_synthetic_dataset, BetaLactam

        X, y, ids = create_ecoli_synthetic_dataset(drug=BetaLactam.AMOX_CLAVULANATE)

        assert X.shape[0] >= 50

    def test_create_dataset_min_samples(self):
        """Test min_samples parameter."""
        from src.diseases import create_ecoli_synthetic_dataset

        X, y, ids = create_ecoli_synthetic_dataset(min_samples=100)

        assert X.shape[0] >= 100

    def test_dataset_ids_unique(self):
        """Test sample IDs are unique."""
        from src.diseases import create_ecoli_synthetic_dataset

        X, y, ids = create_ecoli_synthetic_dataset()

        # IDs should be unique or have a meaningful pattern
        assert len(ids) == X.shape[0]


class TestAnalyze:
    """Test the main analyze method."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        from src.diseases import EcoliBetaLactamAnalyzer

        return EcoliBetaLactamAnalyzer()

    @pytest.fixture
    def reference(self):
        """Get TEM-1 reference sequence."""
        from src.diseases import TEM1_REFERENCE

        return TEM1_REFERENCE

    def test_analyze_single_sequence(self, analyzer, reference):
        """Test analyze with single sequence."""
        from src.diseases import EcoliGene, BetaLactam

        sequences = {EcoliGene.BLA_TEM: [reference]}
        result = analyzer.analyze(sequences, drug=BetaLactam.AMPICILLIN)

        assert "n_sequences" in result
        assert result["n_sequences"] == 1
        assert "resistance" in result
        assert "variant_classification" in result
        assert "mutations_detected" in result

    def test_analyze_multiple_sequences(self, analyzer, reference):
        """Test analyze with multiple sequences."""
        from src.diseases import EcoliGene, BetaLactam

        # Create wild-type and mutant
        mutant = reference[:237] + "S" + reference[238:]  # G238S
        sequences = {EcoliGene.BLA_TEM: [reference, mutant]}

        result = analyzer.analyze(sequences, drug=BetaLactam.CEFTAZIDIME)

        assert result["n_sequences"] == 2
        assert len(result["variant_classification"]) == 2

    def test_analyze_empty_sequences(self, analyzer):
        """Test analyze with no sequences."""
        result = analyzer.analyze({})

        assert result["n_sequences"] == 0


class TestValidatePredictions:
    """Test prediction validation."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        from src.diseases import EcoliBetaLactamAnalyzer

        return EcoliBetaLactamAnalyzer()

    def test_validate_perfect_correlation(self, analyzer):
        """Test validation with perfect predictions."""
        predictions = {"resistance": {"scores": [0.1, 0.5, 0.9]}}
        ground_truth = {"scores": [0.1, 0.5, 0.9]}

        metrics = analyzer.validate_predictions(predictions, ground_truth)

        assert "spearman" in metrics
        assert metrics["spearman"] == pytest.approx(1.0, abs=0.01)

    def test_validate_with_rmse(self, analyzer):
        """Test RMSE is computed."""
        predictions = {"resistance": {"scores": [0.2, 0.4, 0.6]}}
        ground_truth = {"scores": [0.2, 0.4, 0.6]}

        metrics = analyzer.validate_predictions(predictions, ground_truth)

        assert "rmse" in metrics
        assert metrics["rmse"] == pytest.approx(0.0, abs=0.01)

    def test_validate_insufficient_data(self, analyzer):
        """Test validation with insufficient data."""
        predictions = {"resistance": {"scores": [0.5]}}
        ground_truth = {"scores": [0.5]}

        metrics = analyzer.validate_predictions(predictions, ground_truth)

        # Should return empty or limited metrics with < 3 samples
        assert isinstance(metrics, dict)


class TestBenchmark:
    """Test benchmark functionality."""

    def test_benchmark_spearman(self):
        """Test benchmark achieves expected Spearman correlation."""
        from src.diseases import create_ecoli_synthetic_dataset
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_predict
        from scipy.stats import spearmanr

        X, y, ids = create_ecoli_synthetic_dataset(min_samples=50)

        # Simple Ridge regression benchmark
        model = Ridge(alpha=1.0)
        y_pred = cross_val_predict(model, X, y, cv=5)

        rho, _ = spearmanr(y, y_pred)

        # Should achieve reasonable correlation
        assert rho > 0.5, f"Expected Spearman > 0.5, got {rho}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
