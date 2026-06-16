"""Two-state bimolecular duplex thermodynamics."""
from __future__ import annotations

import pytest

from strider import dimer_thermo, dimer_thermo_subopt, dimer_tm, DimerThermo
from strider.thermo.engine import ThermoEngine
from strider.thermo.hairpin import hairpin_thermo
from strider.thermo.salt import TAN_CHEN_MIN_BP
from strider.thermo.structure_thermo import parse_dimer_pairs


@pytest.fixture
def perfect_self():
    # 8-bp self-complementary blunt-end duplex.
    return "GCGCGCGC"


@pytest.fixture
def perfect_hetero():
    # 8-bp heteroduplex, exactly complementary (s2 = reverse_complement(s1)).
    return "AAATTTCC", "GGAAATTT"


def _full_duplex_structure(n: int) -> str:
    return "(" * n + ")" * n


def test_perfect_self_dimer_returns_reasonable_thermo(perfect_self):
    seq = perfect_self
    r = dimer_thermo(seq, sodium_M=1.0, magnesium_M=0.0,
                     structure=_full_duplex_structure(len(seq)))
    assert isinstance(r, DimerThermo)
    assert r.is_self_dimer
    assert r.n_pairs == 8
    assert r.dG37 < 0
    assert r.dH < r.dG37
    assert 30 < r.tm_celsius < 100


def test_perfect_hetero_dimer_returns_reasonable_thermo(perfect_hetero):
    s1, s2 = perfect_hetero
    r = dimer_thermo(s1, s2, sodium_M=1.0, magnesium_M=0.0,
                     structure=_full_duplex_structure(len(s1)))
    assert isinstance(r, DimerThermo)
    assert not r.is_self_dimer
    assert r.n_pairs == 8
    assert r.dG37 < 0
    assert r.dH < r.dG37
    assert -20 < r.tm_celsius < 80


def test_explicit_dotbracket_structure_is_scored(perfect_hetero):
    s1, s2 = perfect_hetero
    struct = _full_duplex_structure(len(s1))
    r = dimer_thermo(s1, s2, sodium_M=1.0, structure=struct)
    assert r.n_pairs == 8
    assert r.structure == struct


def test_explicit_pair_list_is_scored(perfect_hetero):
    s1, s2 = perfect_hetero
    n1 = len(s1)
    pairs = [(i, 2 * n1 - 1 - i) for i in range(n1)]
    r = dimer_thermo(s1, s2, sodium_M=1.0, structure=pairs)
    assert r.n_pairs == 8


def test_concentration_raises_tm():
    seq = "GCGCGCGC"
    struct = _full_duplex_structure(len(seq))
    lo = dimer_thermo(seq, structure=struct, strand_conc_M=1e-9).tm_celsius
    hi = dimer_thermo(seq, structure=struct, strand_conc_M=1e-6).tm_celsius
    assert lo < hi


def test_salt_per_bp_for_short_stems():
    # 4-bp duplex is below the Tan-Chen fitted range.
    seq = "GCGC"
    struct = _full_duplex_structure(len(seq))
    r = dimer_thermo(seq, sodium_M=0.05, magnesium_M=0.010, structure=struct)
    assert r.n_pairs < TAN_CHEN_MIN_BP
    assert r.salt_model == "per_bp"


def test_salt_tan_chen_for_long_stems():
    # 8-bp duplex triggers the auto Tan-Chen path.
    seq = "GCGCGCGC"
    struct = _full_duplex_structure(len(seq))
    r = dimer_thermo(seq, sodium_M=0.05, magnesium_M=0.010, structure=struct)
    assert r.n_pairs >= TAN_CHEN_MIN_BP
    assert r.salt_model == "tan_chen"


def test_magnesium_raises_tm():
    seq = "GCGCGCGC"
    struct = _full_duplex_structure(len(seq))
    base = dimer_thermo(seq, sodium_M=0.05, magnesium_M=0.0,
                        salt_model="tan_chen", structure=struct).tm_celsius
    hi = dimer_thermo(seq, sodium_M=0.05, magnesium_M=0.010,
                      salt_model="tan_chen", structure=struct).tm_celsius
    assert hi > base


def test_rejects_no_pairs():
    with pytest.raises(ValueError):
        dimer_thermo("AAAA", "TTTT", structure="....")


def test_rejects_pair_not_crossing_junction():
    # Pair (0,2) lies entirely within strand 1 and must be rejected.
    pairs = [(0, 2), (3, 8)]
    with pytest.raises(ValueError):
        dimer_thermo("ACGT", "TTTTAAAA", structure=pairs)


def test_rejects_non_nested_pairs():
    # Two separate cross-junction helices (pseudoknot-like for dimers).
    pairs = [(0, 8), (1, 7), (3, 11), (4, 10)]
    with pytest.raises(ValueError):
        dimer_thermo("AAAAAAA", "TTTTTTT", structure=pairs)


def test_parse_dimer_pairs_sorts_and_validates():
    n1 = 4
    pairs = [(2, 6), (0, 8), (1, 7), (3, 5)]
    ordered = parse_dimer_pairs(pairs, n1)
    assert ordered == [(0, 8), (1, 7), (2, 6), (3, 5)]


def test_hairpin_regression_unchanged():
    # Existing hairpin_thermo output must not drift after adding dimer code.
    HP = "CTTTCAACACTGTTGCAGTAA"
    before = hairpin_thermo(HP, sodium_M=0.05, magnesium_M=0.010)
    assert 35 < before.tm_celsius < 55


def test_engine_dimer_tm_matches_module_function():
    engine = ThermoEngine(material="dna", celsius=25.0, sodium=0.137, magnesium=0.01)
    seq = "GCGCGCGC"
    struct = _full_duplex_structure(len(seq))
    from_engine = engine.dimer_thermo(seq, structure=struct)
    from_func = dimer_thermo(seq, sodium_M=0.137, magnesium_M=0.01, structure=struct)
    assert from_engine.tm_celsius == pytest.approx(from_func.tm_celsius)


def test_self_dimer_flag_with_none_seq2():
    r = dimer_thermo("GCGCGCGC")
    assert r.is_self_dimer


def test_mfe_path_honours_minimum_loop_size():
    # The native multi-strand MFE cannot close the final base pair across the
    # nick, so an 8-mer self-dimer is predicted as a 6-bp helix plus a nicked loop.
    r = dimer_thermo("GCGCGCGC", sodium_M=1.0)
    assert r.n_pairs >= 2
    assert r.is_self_dimer
    assert r.dG37 < 0


def test_self_dimer_finds_interstrand_helix():
    # Regression: native seq+seq MFE folds as intra-strand hairpins; the
    # inter-strand DP must instead return a real cross-junction helix.
    #
    # NOTE: this only checks that *an* inter-strand duplex is found — not that it
    # is the IDT/primer3 structure.  strider's DP is Watson-Crick only, so for
    # this sequence it returns a short 4-bp helix where primer3 finds a longer
    # mismatch-containing duplex (ΔH≈-78, Tm≈19.5 °C).  Pinning that is the job
    # of ``test_self_dimer_matches_primer3_structure`` below (xfail until the
    # mismatch-aware DP lands).  Avoid a ΔG magnitude threshold here: it would
    # silently encode the wrong structure (and previously masked the missing
    # bimolecular-initiation term, which made the short helix look too stable).
    seq = "TCGCATTGAAGATGCAGT"
    r = dimer_thermo(seq, sodium_M=1.0)
    assert r.is_self_dimer
    assert r.n_pairs >= 2
    assert r.dG37 < 0
    assert -50 < r.tm_celsius < 100
    assert "&" not in r.structure
    n1 = len(seq)
    assert all(i < n1 <= j for i, j in parse_dimer_pairs(r.structure, n1))


@pytest.mark.xfail(
    reason="Issue B: WC-only dimer DP cannot form the mismatch-containing "
    "homodimer primer3/IDT report; needs the mismatch-stack-aware DP "
    "(consumes the PR-imported mismatch STACK params).",
    strict=True,
)
def test_self_dimer_matches_primer3_structure():
    # primer3 calc_homodimer for this sequence (mv=1000, dna=250 nM, 37 °C):
    #   ΔG37 = -5.57 kcal/mol, ΔH = -77.8 kcal/mol, Tm = 19.5 °C
    # via a mismatch-bridged ~8-bp duplex.  strider currently returns a clean
    # 4-bp helix (ΔH≈-26, Tm≈-21), so ΔH/Tm are far off.
    r = dimer_thermo("TCGCATTGAAGATGCAGT", sodium_M=1.0, strand_conc_M=250e-9)
    assert r.dH == pytest.approx(-77.8, abs=8.0)
    assert r.tm_celsius == pytest.approx(19.5, abs=4.0)


def test_heterodimer_regression_reasonable_thermo():
    from strider.thermo.nn_dna import reverse_complement
    seq = "GGCTAAGGAACGTAAGCA"
    r = dimer_thermo(seq, reverse_complement(seq), sodium_M=1.0)
    assert not r.is_self_dimer
    assert r.n_pairs >= 2
    assert r.dG37 < -10.0
    assert r.tm_celsius > -50
    assert r.tm_celsius < 100
    assert "&" not in r.structure
    n1 = len(seq)
    assert all(i < n1 <= j for i, j in parse_dimer_pairs(r.structure, n1))


class TestDimerThermoSubopt:
    """Sub-optimal local dimer enumeration for the primer3 benchmark."""

    PRIMER3_SEQ = "CAACAAGGTCCGTGAGCTTC"

    @pytest.fixture
    def subopt_results(self):
        return dimer_thermo_subopt(
            self.PRIMER3_SEQ,
            n=5,
            sodium_M=0.050,
            magnesium_M=0.0092,
            material="dna",
            strand_conc_M=0.25e-6,
            salt_model="auto",
        )

    def test_returns_five_results(self, subopt_results):
        assert len(subopt_results) == 5
        assert all(isinstance(r, DimerThermo) for r in subopt_results)

    def test_sorted_by_dg37_ascending(self, subopt_results):
        dg37s = [r.dG37 for r in subopt_results]
        assert dg37s == sorted(dg37s)

    def test_each_structure_is_valid_interstrand_helix(self, subopt_results):
        n1 = len(self.PRIMER3_SEQ)
        for r in subopt_results:
            assert r.n_pairs >= 2
            pairs = parse_dimer_pairs(r.structure, n1)
            assert len(pairs) == r.n_pairs
            assert all(i < n1 <= j for i, j in pairs)
            assert "&" not in r.structure

    def test_first_result_equals_dimer_thermo(self, subopt_results):
        expected = dimer_thermo(
            self.PRIMER3_SEQ,
            sodium_M=0.050,
            magnesium_M=0.0092,
            material="dna",
            strand_conc_M=0.25e-6,
            salt_model="auto",
        )
        assert subopt_results[0] == expected

    def test_self_dimer_flag_set(self, subopt_results):
        assert all(r.is_self_dimer for r in subopt_results)


class TestDimerVsReference:
    """Pin perfect-duplex thermodynamics to independent engines.

    For fully-complementary duplexes strider predicts the same structure as
    primer3/NUPACK, so its ΔG37/ΔH/Tm must match theirs within tolerance.  These
    are the cases that guard the bimolecular-initiation term: without it, Tm runs
    +7…+11 °C high and these assertions fail.

    Reference values (independently reproduced in this review):
      * primer3 v2.3.0  ``calc_homodimer`` / ``calc_heterodimer`` at temp_c=37
      * NUPACK 4.0  equilibrium melt curve (fraction-bound = 0.5)
    primer3 and the NUPACK melt agree with each other to within ~2-3 °C.
    """

    # seq1, seq2(None=self), Na_M, conc_M, forced_full, p3_dG37, p3_dH, p3_Tm, nupack_Tm
    CASES = [
        ("GCGCGCGC", None, 1.0, 50e-9, True, -13.45, -70.8, 51.2, 49.6),
        ("AAATTTCC", "GGAAATTT", 1.0, 50e-9, True, -6.05, -52.6, 9.2, 12.7),
        ("GGCTAAGGAACGTAAGCA", "__revcomp__", 1.0, 250e-9, False, -22.03, -138.8, 65.8, None),
    ]

    @pytest.mark.parametrize(
        "seq1,seq2,na,conc,forced,p3_dg,p3_dh,p3_tm,nupack_tm",
        CASES,
        ids=[c[0] for c in CASES],
    )
    def test_matches_primer3(self, seq1, seq2, na, conc, forced, p3_dg, p3_dh, p3_tm, nupack_tm):
        from strider.thermo.nn_dna import reverse_complement
        if seq2 == "__revcomp__":
            seq2 = reverse_complement(seq1)
        struct = None
        if forced:
            n = len(seq1)
            struct = "(" * n + ")" * n
        r = dimer_thermo(seq1, seq2, sodium_M=na, structure=struct, strand_conc_M=conc)

        assert r.dG37 == pytest.approx(p3_dg, abs=0.5), "ΔG37 off vs primer3"
        assert r.dH == pytest.approx(p3_dh, abs=4.0), "ΔH off vs primer3"
        # Tm must track primer3 closely; the initiation term is what keeps this
        # from drifting ~10 °C high.
        assert r.tm_celsius == pytest.approx(p3_tm, abs=3.0), "Tm off vs primer3"
        if nupack_tm is not None:
            assert r.tm_celsius == pytest.approx(nupack_tm, abs=4.0), "Tm off vs NUPACK melt"
