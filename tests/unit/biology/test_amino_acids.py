# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for src/biology/amino_acids.py - Amino acid properties and mappings."""

import pytest

from src.biology.amino_acids import (
    STANDARD_AMINO_ACIDS,
    AMINO_ACID_3LETTER,
    AMINO_ACID_PROPERTIES,
    AA_PROPERTIES,
    MODIFIED_AMINO_ACIDS,
    get_amino_acid_property,
    get_amino_acid_charge,
    get_normalized_properties,
)


class TestAminoAcidConstants:
    """Test amino acid constant definitions."""

    def test_standard_amino_acids_count(self):
        """STANDARD_AMINO_ACIDS contains 20 standard amino acids."""
        assert len(STANDARD_AMINO_ACIDS) == 20

    def test_standard_amino_acids_single_letter(self):
        """All amino acids are single letter codes."""
        for aa in STANDARD_AMINO_ACIDS:
            assert len(aa) == 1
            assert aa.isupper()

    def test_amino_acid_3letter_complete(self):
        """AMINO_ACID_3LETTER has entries for all 20 amino acids plus stop."""
        assert len(AMINO_ACID_3LETTER) >= 20
        for aa in STANDARD_AMINO_ACIDS:
            assert aa in AMINO_ACID_3LETTER

    def test_amino_acid_3letter_correct(self):
        """Spot check 3-letter codes."""
        assert AMINO_ACID_3LETTER["A"] == "Ala"
        assert AMINO_ACID_3LETTER["M"] == "Met"
        assert AMINO_ACID_3LETTER["W"] == "Trp"
        assert AMINO_ACID_3LETTER["*"] == "Stop"


class TestAminoAcidProperties:
    """Test amino acid property dictionary."""

    def test_properties_complete(self):
        """AMINO_ACID_PROPERTIES has entries for all amino acids."""
        for aa in STANDARD_AMINO_ACIDS:
            assert aa in AMINO_ACID_PROPERTIES

    def test_properties_have_required_fields(self):
        """Each amino acid has required property fields."""
        required_fields = ["hydrophobicity", "charge", "volume", "polarity"]
        for aa in STANDARD_AMINO_ACIDS:
            props = AMINO_ACID_PROPERTIES[aa]
            for field in required_fields:
                assert field in props, f"Missing {field} for {aa}"

    def test_hydrophobicity_range(self):
        """Hydrophobicity values are in reasonable range (Kyte-Doolittle)."""
        for aa in STANDARD_AMINO_ACIDS:
            h = AMINO_ACID_PROPERTIES[aa]["hydrophobicity"]
            assert -5 < h < 5, f"Hydrophobicity out of range for {aa}"

    def test_charge_values(self):
        """Charge values are in expected range."""
        for aa in STANDARD_AMINO_ACIDS:
            charge = AMINO_ACID_PROPERTIES[aa]["charge"]
            assert -1.0 <= charge <= 1.0, f"Invalid charge for {aa}"

    def test_volume_positive(self):
        """Volume values are positive."""
        for aa in STANDARD_AMINO_ACIDS:
            vol = AMINO_ACID_PROPERTIES[aa]["volume"]
            assert vol > 0, f"Non-positive volume for {aa}"

    def test_polarity_binary(self):
        """Polarity is 0 or 1."""
        for aa in STANDARD_AMINO_ACIDS:
            pol = AMINO_ACID_PROPERTIES[aa]["polarity"]
            assert pol in [0.0, 1.0], f"Invalid polarity for {aa}"


class TestAAProperties:
    """Test AA_PROPERTIES legacy format."""

    def test_aa_properties_complete(self):
        """AA_PROPERTIES has entries for all amino acids."""
        for aa in STANDARD_AMINO_ACIDS:
            assert aa in AA_PROPERTIES

    def test_aa_properties_format(self):
        """AA_PROPERTIES values are tuples of 4 floats."""
        for aa, props in AA_PROPERTIES.items():
            assert isinstance(props, tuple)
            assert len(props) == 4
            assert all(isinstance(v, float) for v in props)


class TestPropertyAccessors:
    """Test property accessor functions."""

    def test_get_amino_acid_property_basic(self):
        """get_amino_acid_property returns correct values."""
        # Charged amino acids
        assert get_amino_acid_property("K", "charge") == 1.0
        assert get_amino_acid_property("D", "charge") == -1.0
        assert get_amino_acid_property("A", "charge") == 0.0

    def test_get_amino_acid_property_normalized(self):
        """get_amino_acid_property returns normalized values when requested."""
        norm = get_amino_acid_property("A", "hydrophobicity", normalized=True)
        raw = get_amino_acid_property("A", "hydrophobicity", normalized=False)
        assert norm != raw  # Should be different

    def test_get_amino_acid_property_default(self):
        """get_amino_acid_property uses default for unknown amino acids."""
        result = get_amino_acid_property("Z", "charge", default=-99.0)
        # Either uses default or falls back to X
        assert result in [-99.0, 0.0]

    def test_get_amino_acid_charge(self):
        """get_amino_acid_charge returns correct charges."""
        # Positive: K, R
        assert get_amino_acid_charge("K") == 1.0
        assert get_amino_acid_charge("R") == 1.0
        # Negative: D, E
        assert get_amino_acid_charge("D") == -1.0
        assert get_amino_acid_charge("E") == -1.0
        # Neutral: A, G, etc.
        assert get_amino_acid_charge("A") == 0.0
        assert get_amino_acid_charge("G") == 0.0

    def test_get_normalized_properties(self):
        """get_normalized_properties returns tuple of 4 floats."""
        props = get_normalized_properties("A")
        assert isinstance(props, tuple)
        assert len(props) == 4
        assert all(isinstance(v, float) for v in props)


class TestModifiedAminoAcids:
    """Test modified amino acid definitions."""

    def test_modified_amino_acids_exist(self):
        """MODIFIED_AMINO_ACIDS has entries."""
        assert len(MODIFIED_AMINO_ACIDS) > 0

    def test_modified_amino_acids_have_properties(self):
        """Modified amino acids have required properties."""
        for name, props in MODIFIED_AMINO_ACIDS.items():
            assert "hydrophobicity" in props
            assert "charge" in props
            assert "polarity" in props


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_stop_codon_properties(self):
        """Stop codon (*) has properties defined."""
        assert "*" in AMINO_ACID_PROPERTIES
        props = AMINO_ACID_PROPERTIES["*"]
        assert props["charge"] == 0.0

    def test_unknown_amino_acid(self):
        """Unknown amino acid (X) has properties defined."""
        assert "X" in AMINO_ACID_PROPERTIES

    def test_lowercase_handling(self):
        """Functions handle lowercase input."""
        # Should convert to uppercase
        result = get_amino_acid_charge("a")
        assert result == get_amino_acid_charge("A")
