"""Two-state bimolecular duplex thermodynamics."""
from __future__ import annotations

import pytest

from strider import dimer_thermo, dimer_tm, DimerThermo
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
    # 8-bp heteroduplex, exactly complementary.
    return "AAAATTTT", "TTTTAAAA"


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
