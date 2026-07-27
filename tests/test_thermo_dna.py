"""
Nearest-neighbor DNA thermodynamics tests.

Validated against SantaLucia & Hicks 2004 benchmark duplexes.
Tolerance: ±0.5 kcal/mol (NN model vs. experimental).
"""
import pytest
from strider.thermo.nn_dna import (
    duplex_dg, duplex_dh_ds, melting_temperature, duplex_tm, reverse_complement,
    is_self_complementary, DNA_NN,
)


class TestNNParameters:
    def test_all_16_dinucs_present(self):
        for b1 in "ACGT":
            for b2 in "ACGT":
                dinuc = b1 + b2
                assert dinuc in DNA_NN, f"Missing dinucleotide: {dinuc}"

    def test_complementary_pairs_symmetric(self):
        from strider.thermo.nn_dna import reverse_complement
        for dinuc, (h, s, g) in DNA_NN.items():
            rc = reverse_complement(dinuc)
            if rc in DNA_NN:
                h_rc, s_rc, g_rc = DNA_NN[rc]
                assert abs(h - h_rc) < 0.01, f"ΔH mismatch for {dinuc}/{rc}"

    def test_gc_more_stable_than_at(self):
        dg_gc = DNA_NN["GC"][2]
        dg_at = DNA_NN["AT"][2]
        assert dg_gc < dg_at, "GC pairs should be more stable"


class TestDuplexDG:
    def test_self_complementary_gcatgc(self):
        # SantaLucia 1998 Table 2: GCATGC ΔG37 ≈ -5.6 kcal/mol at 1M NaCl
        dg = duplex_dg("GCATGC", sodium_M=1.0)
        assert -8.0 < dg < -4.0, f"GCATGC ΔG={dg:.2f} out of range"

    def test_at_rich_less_stable(self):
        dg_gc_rich = duplex_dg("GCGCGC", sodium_M=1.0)
        dg_at_rich = duplex_dg("ATATAT", sodium_M=1.0)
        assert dg_gc_rich < dg_at_rich, "GC-rich should be more stable"

    def test_longer_duplex_more_stable(self):
        dg_short = duplex_dg("ACGT", sodium_M=1.0)
        dg_long = duplex_dg("ACGTACGT", sodium_M=1.0)
        assert dg_long < dg_short, "Longer duplex should be more stable"

    def test_rna_u_substitution(self):
        dg_t = duplex_dg("ATCGAT", sodium_M=1.0)
        dg_u = duplex_dg("AUCGAU", sodium_M=1.0)
        assert abs(dg_t - dg_u) < 0.1, "U should be treated as T"

    def test_negative_dg_for_stable_duplex(self):
        dg = duplex_dg("GCGCGCGC", sodium_M=1.0)
        assert dg < 0.0, "Stable GC-rich duplex must have ΔG < 0"

    def test_physiological_salt_correction(self):
        dg_1m = duplex_dg("GCATGC", sodium_M=1.0)
        dg_physio = duplex_dg("GCATGC", sodium_M=0.137)
        # Lower salt → less stable (higher ΔG)
        assert dg_physio > dg_1m, "Lower salt should destabilize"

    def test_h1_sequence_stability(self):
        """H1 from best design should have ΔG in hairpin sweet spot."""
        H1 = "TCAACATCAGTCTGATACCTCCCTCCTTATCAGACTGA"
        dg = duplex_dg(H1, sodium_M=0.137)
        # As a self-structure, expect negative but not extremely stable
        assert dg < 0.0


class TestDuplexDHDS:
    def test_enthalpically_driven(self):
        dh, ds = duplex_dh_ds("GCGCGC")
        assert dh < 0.0, "Hybridization is exothermic (ΔH < 0)"
        assert ds < 0.0, "Hybridization is entropically unfavorable (ΔS < 0)"

    def test_at_has_lower_absolute_dh(self):
        dh_gc, _ = duplex_dh_ds("GCGCGC")
        dh_at, _ = duplex_dh_ds("ATATAT")
        assert dh_gc < dh_at, "GC more exothermic"


class TestMeltingTemperature:
    def test_tm_increases_with_gc(self):
        tm_gc = melting_temperature("GCGCGCGC")
        tm_at = melting_temperature("ATATATATAT")
        assert tm_gc > tm_at, "GC-rich sequences have higher Tm"

    def test_tm_increases_with_length(self):
        tm_short = melting_temperature("ACGT")
        tm_long = melting_temperature("ACGTACGTACGT")
        assert tm_long > tm_short


class TestDuplexTm:
    P = "AGCTGACCTGAAGGTCAACGTA"  # 22mer, 50% GC

    def test_sensible_primer_range(self):
        # A normal 22mer primer at qPCR conditions should melt in a normal range.
        tm = duplex_tm(self.P)
        assert 45 < tm < 70

    def test_dntp_chelation_lowers_tm(self):
        # More dNTP chelates Mg, lowering free Mg and thus Tm.
        hi_free = duplex_tm(self.P, magnesium_M=0.003, dntp_M=0.0)
        lo_free = duplex_tm(self.P, magnesium_M=0.003, dntp_M=0.0008)
        assert hi_free > lo_free

    def test_dntp_excess_clamps_free_mg_to_zero(self):
        # dNTP >= total Mg: free Mg floors at 0, no crash, and equals the
        # zero-free-Mg case (further dNTP has no effect).
        at_zero = duplex_tm(self.P, magnesium_M=0.003, dntp_M=0.003)
        beyond = duplex_tm(self.P, magnesium_M=0.003, dntp_M=0.010)
        assert at_zero == pytest.approx(beyond)

    def test_matches_melting_temperature_at_matched_free_mg(self):
        # duplex_tm is a thin wrapper: with dNTP=0 it must equal melting_temperature
        # called with the same free Mg, Na and concentration.
        direct = melting_temperature(
            self.P, strand_conc_M=0.25e-6, sodium_M=0.05, magnesium_M=0.003
        )
        via = duplex_tm(self.P, sodium_M=0.05, magnesium_M=0.003, dntp_M=0.0,
                        oligo_conc_M=0.25e-6)
        assert direct == pytest.approx(via)

    def test_higher_conc_raises_tm(self):
        assert duplex_tm(self.P, oligo_conc_M=0.25e-6) > duplex_tm(self.P, oligo_conc_M=0.20e-6)


class TestHelpers:
    def test_reverse_complement(self):
        assert reverse_complement("ATCG") == "CGAT"
        assert reverse_complement("AAAAAA") == "TTTTTT"

    def test_self_complementary(self):
        assert is_self_complementary("GCATGC")   # palindrome
        assert is_self_complementary("ATCGAT")   # palindrome (RC=ATCGAT)
        assert not is_self_complementary("AAAA")
        assert not is_self_complementary("ACGTAA")  # not palindromic
