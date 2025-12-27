# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for src/data/hiv/position_mapper.py - HXB2 position mapping utilities."""

import pytest

from src.data.hiv.position_mapper import (
    PositionMapper,
    HXB2_REGIONS,
    hxb2_to_protein_position,
    protein_position_to_hxb2,
    codon_position_to_hxb2,
    get_v3_positions,
    get_drug_target_positions,
)


class TestHXB2Regions:
    """Test HXB2 region definitions."""

    def test_regions_exist(self):
        """Key regions are defined."""
        assert "pr" in HXB2_REGIONS
        assert "rt" in HXB2_REGIONS
        assert "in" in HXB2_REGIONS
        assert "gp120" in HXB2_REGIONS
        assert "v3" in HXB2_REGIONS

    def test_region_attributes(self):
        """Regions have required attributes."""
        for name, region in HXB2_REGIONS.items():
            assert hasattr(region, "nt_start")
            assert hasattr(region, "nt_end")
            assert hasattr(region, "aa_start")
            assert hasattr(region, "aa_end")
            assert region.nt_end > region.nt_start
            assert region.aa_end >= region.aa_start

    def test_protease_region(self):
        """Protease region is correctly defined."""
        pr = HXB2_REGIONS["pr"]
        assert pr.aa_start == 1
        assert pr.aa_end == 99
        # PR is 99 amino acids long
        assert pr.aa_end - pr.aa_start + 1 == 99


class TestPositionMapper:
    """Test PositionMapper class."""

    @pytest.fixture
    def mapper(self):
        """Create a PositionMapper instance."""
        return PositionMapper()

    def test_list_regions(self, mapper):
        """list_regions returns all region names."""
        regions = mapper.list_regions()
        assert "pr" in regions
        assert "rt" in regions
        assert len(regions) > 5

    def test_get_region_info(self, mapper):
        """get_region_info returns region details."""
        pr = mapper.get_region_info("pr")
        assert pr.name == "Protease"
        assert pr.aa_end == 99

    def test_get_region_info_case_insensitive(self, mapper):
        """Region lookup is case insensitive."""
        pr_lower = mapper.get_region_info("pr")
        pr_upper = mapper.get_region_info("PR")
        assert pr_lower.nt_start == pr_upper.nt_start

    def test_invalid_region(self, mapper):
        """Invalid region raises ValueError."""
        with pytest.raises(ValueError):
            mapper.get_region_info("invalid")

    def test_protein_to_hxb2(self, mapper):
        """protein_to_hxb2 converts correctly."""
        # First position of protease
        hxb2 = mapper.protein_to_hxb2("pr", 1)
        assert hxb2 == HXB2_REGIONS["pr"].nt_start

    def test_protein_to_hxb2_offset(self, mapper):
        """protein_to_hxb2 handles position offsets."""
        pr_start = HXB2_REGIONS["pr"].nt_start
        # Position 2 should be 3 nucleotides after position 1
        hxb2_pos2 = mapper.protein_to_hxb2("pr", 2)
        assert hxb2_pos2 == pr_start + 3

    def test_protein_to_hxb2_out_of_range(self, mapper):
        """Out of range position raises ValueError."""
        with pytest.raises(ValueError):
            mapper.protein_to_hxb2("pr", 100)  # PR only has 99 positions

    def test_hxb2_to_protein(self, mapper):
        """hxb2_to_protein converts correctly."""
        # Use NEF which has no overlap with other regions (8797-9417)
        nef_start = HXB2_REGIONS["nef"].nt_start
        protein, pos = mapper.hxb2_to_protein(nef_start)
        assert protein.lower() == "nef"
        assert pos == 1

    def test_roundtrip_conversion(self, mapper):
        """protein_to_hxb2 and hxb2_to_protein are inverses."""
        # Use NEF which has no overlap with other regions
        for pos in [1, 30, 50, 99]:
            hxb2 = mapper.protein_to_hxb2("nef", pos)
            protein, recovered_pos = mapper.hxb2_to_protein(hxb2)
            assert protein.lower() == "nef"
            assert recovered_pos == pos


class TestModuleFunctions:
    """Test module-level convenience functions."""

    def test_hxb2_to_protein_position(self):
        """Module function works correctly."""
        # Use NEF which has no overlap with other regions (8797-9417)
        nef_start = HXB2_REGIONS["nef"].nt_start
        protein, pos = hxb2_to_protein_position(nef_start)
        assert protein.lower() == "nef"
        assert pos == 1

    def test_protein_position_to_hxb2(self):
        """Module function works correctly."""
        hxb2 = protein_position_to_hxb2("pr", 1)
        assert hxb2 == HXB2_REGIONS["pr"].nt_start

    def test_codon_position_to_hxb2(self):
        """Codon position conversion works."""
        # Codon 0 of PR is the first codon
        hxb2 = codon_position_to_hxb2("pr", 0)
        assert hxb2 == HXB2_REGIONS["pr"].nt_start


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_v3_positions(self):
        """get_v3_positions returns V3 loop coordinates."""
        start, end = get_v3_positions()
        assert start < end
        assert start == HXB2_REGIONS["v3"].nt_start
        assert end == HXB2_REGIONS["v3"].nt_end

    def test_get_drug_target_positions_pi(self):
        """get_drug_target_positions for PI."""
        protein, start, end = get_drug_target_positions("PI")
        assert protein == "pr"
        assert start == 1
        assert end == 99

    def test_get_drug_target_positions_nrti(self):
        """get_drug_target_positions for NRTI."""
        protein, start, end = get_drug_target_positions("NRTI")
        assert protein == "rt"
        assert start >= 1

    def test_get_drug_target_positions_invalid(self):
        """Invalid drug class raises ValueError."""
        with pytest.raises(ValueError):
            get_drug_target_positions("invalid")
