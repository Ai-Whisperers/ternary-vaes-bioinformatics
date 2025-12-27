# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for src/data/hiv/stanford.py - Stanford HIVDB data loaders."""

import pytest
import pandas as pd
from pathlib import Path

from src.data.hiv.stanford import (
    load_stanford_hivdb,
    get_stanford_drug_columns,
    parse_mutation_list,
    extract_stanford_positions,
)


class TestGetStanfordDrugColumns:
    """Test drug column name retrieval."""

    def test_pi_drugs(self):
        """PI drug columns are correct."""
        drugs = get_stanford_drug_columns("pi")
        assert "FPV" in drugs
        assert "DRV" in drugs
        assert len(drugs) == 8

    def test_nrti_drugs(self):
        """NRTI drug columns are correct."""
        drugs = get_stanford_drug_columns("nrti")
        assert "AZT" in drugs
        assert "TDF" in drugs
        assert len(drugs) == 7

    def test_nnrti_drugs(self):
        """NNRTI drug columns are correct."""
        drugs = get_stanford_drug_columns("nnrti")
        assert "EFV" in drugs
        assert "NVP" in drugs
        assert len(drugs) == 5

    def test_ini_drugs(self):
        """INI drug columns are correct."""
        drugs = get_stanford_drug_columns("ini")
        assert "DTG" in drugs
        assert "RAL" in drugs
        assert len(drugs) == 5

    def test_case_insensitive(self):
        """Drug class lookup is case insensitive."""
        assert get_stanford_drug_columns("PI") == get_stanford_drug_columns("pi")
        assert get_stanford_drug_columns("NRTI") == get_stanford_drug_columns("nrti")

    def test_invalid_class(self):
        """Invalid drug class returns empty list."""
        assert get_stanford_drug_columns("invalid") == []


class TestParseMutationList:
    """Test mutation list parsing."""

    def test_single_mutation(self):
        """Parse single mutation correctly."""
        mutations = parse_mutation_list("D30N")
        assert len(mutations) == 1
        assert mutations[0]["wild_type"] == "D"
        assert mutations[0]["position"] == 30
        assert mutations[0]["mutant"] == "N"

    def test_multiple_mutations(self):
        """Parse comma-separated mutations."""
        mutations = parse_mutation_list("D30N, M46I, R57G")
        assert len(mutations) == 3
        assert mutations[0]["position"] == 30
        assert mutations[1]["position"] == 46
        assert mutations[2]["position"] == 57

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert parse_mutation_list("") == []

    def test_none_input(self):
        """None input returns empty list."""
        assert parse_mutation_list(None) == []

    def test_nan_input(self):
        """NaN input returns empty list."""
        import numpy as np
        assert parse_mutation_list(np.nan) == []

    def test_stop_codon_mutation(self):
        """Parse mutation to stop codon (*)."""
        mutations = parse_mutation_list("Q151*")
        assert len(mutations) == 1
        assert mutations[0]["mutant"] == "*"

    def test_whitespace_handling(self):
        """Handle various whitespace patterns."""
        mutations = parse_mutation_list("D30N,M46I, R57G")
        assert len(mutations) == 3


class TestExtractStanfordPositions:
    """Test position extraction from DataFrames."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with position columns."""
        return pd.DataFrame({
            "SeqID": ["seq1", "seq2"],
            "P1": ["D", "D"],
            "P2": ["I", "I"],
            "P30": ["N", "D"],
            "P46": ["I", "M"],
            "FPV": [3.5, 1.0],
            "drug_class": ["PI", "PI"],
        })

    def test_extract_pr_positions(self, sample_df):
        """Extract protease positions."""
        positions = extract_stanford_positions(sample_df, "PR")
        assert "SeqID" in positions.columns
        assert "P1" in positions.columns
        assert "P30" in positions.columns
        assert "FPV" not in positions.columns

    def test_preserve_seqid(self, sample_df):
        """SeqID column is preserved."""
        positions = extract_stanford_positions(sample_df, "PR")
        assert list(positions["SeqID"]) == ["seq1", "seq2"]


class TestLoadStanfordHivdb:
    """Test data loading (may skip if data not available)."""

    def test_invalid_drug_class(self):
        """Invalid drug class raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            load_stanford_hivdb("invalid")
        assert "Invalid drug_class" in str(exc_info.value)

    def test_valid_drug_classes(self):
        """Valid drug classes are accepted."""
        for dc in ["pi", "nrti", "nnrti", "ini", "all"]:
            try:
                df = load_stanford_hivdb(dc)
                assert isinstance(df, pd.DataFrame)
                assert "drug_class" in df.columns
            except FileNotFoundError:
                pytest.skip(f"Stanford HIVDB {dc} data not available")

    def test_case_insensitive_loading(self):
        """Drug class loading is case insensitive."""
        try:
            df_lower = load_stanford_hivdb("pi")
            df_upper = load_stanford_hivdb("PI")
            assert len(df_lower) == len(df_upper)
        except FileNotFoundError:
            pytest.skip("Stanford HIVDB data not available")
