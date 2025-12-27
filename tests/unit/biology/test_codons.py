# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for src/biology/codons.py - Single Source of Truth for genetic code."""

import pytest

from src.biology.codons import (
    NUCLEOTIDES,
    BASE_TO_IDX,
    IDX_TO_BASE,
    GENETIC_CODE,
    AMINO_ACID_TO_CODONS,
    CODON_TO_INDEX,
    INDEX_TO_CODON,
    codon_to_index,
    index_to_codon,
    codon_to_amino_acid,
    triplet_to_codon_index,
    codon_index_to_triplet,
)


class TestNucleotideMappings:
    """Test nucleotide base mappings."""

    def test_nucleotides_string(self):
        """NUCLEOTIDES contains all 4 bases."""
        assert len(NUCLEOTIDES) == 4
        assert set(NUCLEOTIDES) == {"T", "C", "A", "G"}

    def test_base_to_idx_complete(self):
        """BASE_TO_IDX maps all 4 bases."""
        assert len(BASE_TO_IDX) == 4
        assert set(BASE_TO_IDX.keys()) == {"T", "C", "A", "G"}
        assert set(BASE_TO_IDX.values()) == {0, 1, 2, 3}

    def test_idx_to_base_complete(self):
        """IDX_TO_BASE is inverse of BASE_TO_IDX."""
        for base, idx in BASE_TO_IDX.items():
            assert IDX_TO_BASE[idx] == base

    def test_base_idx_roundtrip(self):
        """BASE_TO_IDX and IDX_TO_BASE are inverses."""
        for base in NUCLEOTIDES:
            assert IDX_TO_BASE[BASE_TO_IDX[base]] == base


class TestGeneticCode:
    """Test the genetic code mapping."""

    def test_genetic_code_size(self):
        """GENETIC_CODE has 64 codons."""
        assert len(GENETIC_CODE) == 64

    def test_all_codons_present(self):
        """All 64 possible codons are in GENETIC_CODE."""
        for b1 in NUCLEOTIDES:
            for b2 in NUCLEOTIDES:
                for b3 in NUCLEOTIDES:
                    codon = b1 + b2 + b3
                    assert codon in GENETIC_CODE, f"Missing codon: {codon}"

    def test_standard_start_codon(self):
        """ATG is the standard start codon (Methionine)."""
        assert GENETIC_CODE["ATG"] == "M"

    def test_stop_codons(self):
        """Three stop codons exist."""
        stop_codons = [c for c, aa in GENETIC_CODE.items() if aa == "*"]
        assert set(stop_codons) == {"TAA", "TAG", "TGA"}

    def test_amino_acids_covered(self):
        """All 20 standard amino acids + stop are represented."""
        amino_acids = set(GENETIC_CODE.values())
        # 20 amino acids + 1 stop
        assert len(amino_acids) == 21
        assert "*" in amino_acids


class TestAminoAcidToCodons:
    """Test reverse mapping from amino acid to codons."""

    def test_all_amino_acids_have_codons(self):
        """Every amino acid in genetic code has corresponding codons."""
        for aa in set(GENETIC_CODE.values()):
            assert aa in AMINO_ACID_TO_CODONS
            assert len(AMINO_ACID_TO_CODONS[aa]) > 0

    def test_methionine_single_codon(self):
        """Methionine (M) has only one codon (ATG)."""
        assert AMINO_ACID_TO_CODONS["M"] == ["ATG"]

    def test_tryptophan_single_codon(self):
        """Tryptophan (W) has only one codon (TGG)."""
        assert AMINO_ACID_TO_CODONS["W"] == ["TGG"]

    def test_leucine_six_codons(self):
        """Leucine (L) has six codons."""
        assert len(AMINO_ACID_TO_CODONS["L"]) == 6

    def test_reverse_mapping_consistency(self):
        """AMINO_ACID_TO_CODONS is consistent with GENETIC_CODE."""
        for aa, codons in AMINO_ACID_TO_CODONS.items():
            for codon in codons:
                assert GENETIC_CODE[codon] == aa


class TestCodonIndexing:
    """Test codon to index conversions."""

    def test_codon_to_index_range(self):
        """codon_to_index returns values in [0, 63]."""
        for codon in GENETIC_CODE.keys():
            idx = codon_to_index(codon)
            assert 0 <= idx <= 63

    def test_index_to_codon_range(self):
        """index_to_codon works for all indices 0-63."""
        for i in range(64):
            codon = index_to_codon(i)
            assert len(codon) == 3
            assert all(base in NUCLEOTIDES for base in codon)

    def test_codon_index_roundtrip(self):
        """codon_to_index and index_to_codon are inverses."""
        for codon in GENETIC_CODE.keys():
            assert index_to_codon(codon_to_index(codon)) == codon

    def test_index_codon_roundtrip(self):
        """index_to_codon and codon_to_index are inverses."""
        for i in range(64):
            assert codon_to_index(index_to_codon(i)) == i

    def test_ttt_is_zero(self):
        """TTT (first codon in standard order) should be index 0."""
        assert codon_to_index("TTT") == 0

    def test_ggg_is_63(self):
        """GGG (last codon in standard order) should be index 63."""
        assert codon_to_index("GGG") == 63

    def test_codon_to_index_dict(self):
        """CODON_TO_INDEX dictionary is consistent."""
        assert len(CODON_TO_INDEX) == 64
        for codon, idx in CODON_TO_INDEX.items():
            assert codon_to_index(codon) == idx

    def test_index_to_codon_dict(self):
        """INDEX_TO_CODON dictionary is consistent."""
        assert len(INDEX_TO_CODON) == 64
        for idx, codon in INDEX_TO_CODON.items():
            assert index_to_codon(idx) == codon


class TestAliases:
    """Test backward-compatible function aliases."""

    def test_triplet_to_codon_index_alias(self):
        """triplet_to_codon_index is alias for codon_to_index."""
        for codon in ["ATG", "TTT", "GGG", "TAA"]:
            assert triplet_to_codon_index(codon) == codon_to_index(codon)

    def test_codon_index_to_triplet_alias(self):
        """codon_index_to_triplet is alias for index_to_codon."""
        for i in [0, 1, 32, 63]:
            assert codon_index_to_triplet(i) == index_to_codon(i)


class TestCodonToAminoAcid:
    """Test codon to amino acid translation."""

    def test_codon_to_amino_acid_basic(self):
        """codon_to_amino_acid returns correct amino acid."""
        assert codon_to_amino_acid("ATG") == "M"
        assert codon_to_amino_acid("TTT") == "F"
        assert codon_to_amino_acid("TAA") == "*"

    def test_codon_to_amino_acid_all(self):
        """codon_to_amino_acid matches GENETIC_CODE."""
        for codon, aa in GENETIC_CODE.items():
            assert codon_to_amino_acid(codon) == aa


class TestPadicProperties:
    """Test p-adic properties of codon indexing."""

    def test_codon_order_padic(self):
        """Codon ordering follows base-4 structure."""
        # First 4 codons should be TTT, TTC, TTA, TTG (only last base varies)
        assert index_to_codon(0) == "TTT"
        assert index_to_codon(1) == "TTC"
        assert index_to_codon(2) == "TTA"
        assert index_to_codon(3) == "TTG"

    def test_synonymous_codons_adjacent(self):
        """Synonymous codons tend to have adjacent indices."""
        # Phenylalanine: TTT (0), TTC (1) - adjacent
        assert abs(codon_to_index("TTT") - codon_to_index("TTC")) == 1
        # Both code for F
        assert GENETIC_CODE["TTT"] == GENETIC_CODE["TTC"] == "F"
