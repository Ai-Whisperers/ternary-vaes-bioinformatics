# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for p-adic shift operations."""

import pytest
import torch

from src.utils.padic_shift import (
    PAdicCodonAnalyzer,
    PAdicSequenceEncoder,
    PAdicShiftResult,
    batch_padic_distance,
    codon_padic_distance,
    codon_to_index,
    index_to_codon,
    padic_digits,
    padic_distance,
    padic_distance_matrix,
    padic_norm,
    padic_shift,
    padic_valuation,
    sequence_padic_encoding,
)


class TestPAdicValuation:
    """Tests for p-adic valuation computation."""

    def test_valuation_of_zero(self):
        """Test valuation of zero is large (infinity)."""
        v = padic_valuation(0, p=3)
        assert v >= 10  # Represents infinity

    def test_valuation_of_one(self):
        """Test valuation of 1 is 0."""
        v = padic_valuation(1, p=3)
        assert v == 0

    def test_valuation_of_three(self):
        """Test valuation of 3 is 1."""
        v = padic_valuation(3, p=3)
        assert v == 1

    def test_valuation_of_nine(self):
        """Test valuation of 9 is 2."""
        v = padic_valuation(9, p=3)
        assert v == 2

    def test_valuation_of_27(self):
        """Test valuation of 27 is 3."""
        v = padic_valuation(27, p=3)
        assert v == 3

    def test_valuation_of_non_divisible(self):
        """Test valuation of number not divisible by p."""
        v = padic_valuation(7, p=3)
        assert v == 0

    def test_valuation_different_prime(self):
        """Test valuation with p=2."""
        v = padic_valuation(8, p=2)
        assert v == 3


class TestPAdicNorm:
    """Tests for p-adic norm computation."""

    def test_norm_of_zero(self):
        """Test norm of zero is 0."""
        n = padic_norm(0, p=3)
        assert n == 0.0

    def test_norm_of_one(self):
        """Test norm of 1 is 1."""
        n = padic_norm(1, p=3)
        assert n == 1.0

    def test_norm_of_three(self):
        """Test norm of 3 is 1/3."""
        n = padic_norm(3, p=3)
        assert n == pytest.approx(1/3)

    def test_norm_of_nine(self):
        """Test norm of 9 is 1/9."""
        n = padic_norm(9, p=3)
        assert n == pytest.approx(1/9)


class TestPAdicDistance:
    """Tests for p-adic distance computation."""

    def test_distance_to_self(self):
        """Test distance to self is 0."""
        d = padic_distance(5, 5, p=3)
        assert d == 0.0

    def test_distance_difference_of_one(self):
        """Test distance when difference is 1."""
        d = padic_distance(0, 1, p=3)
        assert d == 1.0

    def test_distance_difference_of_three(self):
        """Test distance when difference is 3."""
        d = padic_distance(0, 3, p=3)
        assert d == pytest.approx(1/3)

    def test_ultrametric_property(self):
        """Test that p-adic distance satisfies ultrametric inequality."""
        # d(a,c) <= max(d(a,b), d(b,c))
        a, b, c = 0, 3, 9
        d_ab = padic_distance(a, b, p=3)
        d_bc = padic_distance(b, c, p=3)
        d_ac = padic_distance(a, c, p=3)

        assert d_ac <= max(d_ab, d_bc) + 1e-10


class TestPAdicDigits:
    """Tests for p-adic digit expansion."""

    def test_digits_of_zero(self):
        """Test digits of 0."""
        digits = padic_digits(0, p=3, n_digits=4)
        assert digits == [0, 0, 0, 0]

    def test_digits_of_one(self):
        """Test digits of 1."""
        digits = padic_digits(1, p=3, n_digits=4)
        assert digits == [1, 0, 0, 0]

    def test_digits_of_four(self):
        """Test digits of 4 = 1 + 1*3."""
        digits = padic_digits(4, p=3, n_digits=4)
        assert digits == [1, 1, 0, 0]

    def test_digits_of_ten(self):
        """Test digits of 10 = 1 + 0*3 + 1*9."""
        digits = padic_digits(10, p=3, n_digits=4)
        assert digits == [1, 0, 1, 0]


class TestPAdicShift:
    """Tests for p-adic shift operation."""

    def test_right_shift_by_one(self):
        """Test right shift by 1 (divide by 3)."""
        result = padic_shift(9, shift_amount=1, p=3)
        assert result.shift_value == 3.0

    def test_right_shift_by_two(self):
        """Test right shift by 2 (divide by 9)."""
        result = padic_shift(27, shift_amount=2, p=3)
        assert result.shift_value == 3.0

    def test_left_shift_by_one(self):
        """Test left shift by 1 (multiply by 3)."""
        result = padic_shift(5, shift_amount=-1, p=3)
        assert result.shift_value == 15.0

    def test_shift_result_contains_digits(self):
        """Test that shift result contains digit expansion."""
        result = padic_shift(10, shift_amount=0, p=3)
        assert isinstance(result.digits, list)
        assert len(result.digits) == 4


class TestCodonConversion:
    """Tests for codon index conversion."""

    def test_codon_to_index_uuu(self):
        """Test UUU maps to 0."""
        idx = codon_to_index("UUU")
        assert idx == 0

    def test_codon_to_index_uuc(self):
        """Test UUC maps to 1."""
        idx = codon_to_index("UUC")
        assert idx == 1

    def test_codon_to_index_ggg(self):
        """Test GGG maps to 63."""
        idx = codon_to_index("GGG")
        assert idx == 63

    def test_codon_to_index_aug(self):
        """Test AUG (start codon)."""
        idx = codon_to_index("AUG")
        # A=2, U=0, G=3 -> 2*16 + 0*4 + 3 = 35
        assert idx == 35

    def test_index_to_codon_round_trip(self):
        """Test round-trip conversion."""
        for i in range(64):
            codon = index_to_codon(i)
            idx = codon_to_index(codon)
            assert idx == i

    def test_t_to_u_conversion(self):
        """Test that T is converted to U."""
        idx_t = codon_to_index("ATG")
        idx_u = codon_to_index("AUG")
        assert idx_t == idx_u


class TestCodonPAdicDistance:
    """Tests for codon p-adic distance."""

    def test_distance_identical_codons(self):
        """Test distance between identical codons is 0."""
        d = codon_padic_distance("AUG", "AUG")
        assert d == 0.0

    def test_distance_adjacent_codons(self):
        """Test distance between adjacent codons."""
        d = codon_padic_distance("UUU", "UUC")
        assert d == 1.0

    def test_distance_wobble_position(self):
        """Test that wobble position differences are captured."""
        # UUU and UUC differ in 3rd position
        d1 = codon_padic_distance("UUU", "UUC")
        # UUU and UCU differ in 2nd position
        d2 = codon_padic_distance("UUU", "UCU")

        # Both should be non-zero
        assert d1 > 0
        assert d2 > 0


class TestSequencePAdicEncoding:
    """Tests for sequence encoding."""

    def test_encode_single_codon(self):
        """Test encoding a single codon."""
        encoding = sequence_padic_encoding("AUG", n_digits=4)
        assert encoding.shape == (1, 4)

    def test_encode_multiple_codons(self):
        """Test encoding multiple codons."""
        encoding = sequence_padic_encoding("AUGUUU", n_digits=4)
        assert encoding.shape == (2, 4)

    def test_padding_handled(self):
        """Test that sequences not divisible by 3 are padded."""
        # 5 nucleotides -> pads to 6 -> 2 codons
        encoding = sequence_padic_encoding("AUGUU", n_digits=4)
        assert encoding.shape[0] == 2


class TestBatchPAdicDistance:
    """Tests for batch p-adic distance."""

    def test_batch_distance(self):
        """Test batch distance computation."""
        idx1 = torch.tensor([0, 1, 2, 3])
        idx2 = torch.tensor([0, 2, 5, 9])

        distances = batch_padic_distance(idx1, idx2, p=3)

        assert distances.shape == (4,)
        assert distances[0] == 0.0  # Same index
        assert distances[1] == 1.0  # Diff of 1

    def test_batch_matches_scalar(self):
        """Test batch results match scalar computation."""
        for i in range(10):
            for j in range(10):
                scalar = padic_distance(i, j, p=3)
                batch = batch_padic_distance(
                    torch.tensor([i]), torch.tensor([j]), p=3
                )
                assert abs(scalar - batch[0].item()) < 1e-6


class TestPAdicDistanceMatrix:
    """Tests for p-adic distance matrix."""

    def test_matrix_shape(self):
        """Test matrix has correct shape."""
        indices = torch.arange(10)
        matrix = padic_distance_matrix(indices, p=3)
        assert matrix.shape == (10, 10)

    def test_matrix_diagonal(self):
        """Test diagonal is zero."""
        indices = torch.arange(10)
        matrix = padic_distance_matrix(indices, p=3)
        diag = torch.diag(matrix)
        assert torch.allclose(diag, torch.zeros(10))

    def test_matrix_symmetric(self):
        """Test matrix is symmetric."""
        indices = torch.arange(10)
        matrix = padic_distance_matrix(indices, p=3)
        assert torch.allclose(matrix, matrix.T)


class TestPAdicSequenceEncoder:
    """Tests for PAdicSequenceEncoder class."""

    def test_creation(self):
        """Test encoder creation."""
        encoder = PAdicSequenceEncoder(p=3, n_digits=4)
        assert encoder.p == 3
        assert encoder.n_digits == 4

    def test_encode_sequence(self):
        """Test sequence encoding."""
        encoder = PAdicSequenceEncoder()
        encoding = encoder.encode_sequence("AUGUUUAAA")
        assert encoding.shape == (3, 4)

    def test_encode_codons(self):
        """Test codon encoding."""
        encoder = PAdicSequenceEncoder()
        indices = encoder.encode_codons(["AUG", "UUU", "AAA"])
        assert indices.shape == (3,)

    def test_get_padic_digits(self):
        """Test getting p-adic digits."""
        encoder = PAdicSequenceEncoder()
        indices = torch.tensor([[0, 1, 2], [3, 4, 5]])
        digits = encoder.get_padic_digits(indices)
        assert digits.shape == (2, 3, 4)

    def test_compute_distances(self):
        """Test distance computation."""
        encoder = PAdicSequenceEncoder()
        indices = torch.tensor([[0, 1, 3, 9]])
        distances = encoder.compute_distances(indices)
        assert distances.shape == (1, 4, 4)


class TestPAdicCodonAnalyzer:
    """Tests for PAdicCodonAnalyzer class."""

    def test_creation(self):
        """Test analyzer creation."""
        analyzer = PAdicCodonAnalyzer(p=3)
        assert analyzer.p == 3

    def test_synonymous_groups(self):
        """Test synonymous codon groups exist."""
        analyzer = PAdicCodonAnalyzer()

        # Leucine should have 6 codons
        assert len(analyzer.synonymous_groups["L"]) == 6

        # Methionine should have 1 codon
        assert len(analyzer.synonymous_groups["M"]) == 1

    def test_synonymous_padic_spread(self):
        """Test p-adic spread computation."""
        analyzer = PAdicCodonAnalyzer()

        # Spread for amino acid with multiple codons
        spread_l = analyzer.synonymous_padic_spread("L")
        assert spread_l >= 0

        # Spread for single-codon amino acid should be 0
        spread_m = analyzer.synonymous_padic_spread("M")
        assert spread_m == 0.0

    def test_codon_bias_score(self):
        """Test codon bias score computation."""
        analyzer = PAdicCodonAnalyzer()

        # Any valid sequence should give a score
        score = analyzer.codon_bias_score("AUGUUUAAA")
        assert score >= 0
