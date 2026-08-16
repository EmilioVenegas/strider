"""Regression tests for the Mathews 2004 parameter set work (PR #13).

Covers the review-requested pins:

* table identity of `mathews2004-dna.json` at value level (the 1999 vs 2004
  ViennaRNA `.par` sets differ in exactly the markers asserted here);
* pinned folded energies so the 12k-row parameter file cannot silently drift
  even though the ViennaRNA verification runs only at generation time;
* paramset-by-name acceptance (historical bug: `'str' object has no attribute
  'dG'');
* the "dangles=0 is bit-identical to the pre-flag default" grid;
* `fraction_folded` zero case and that caller mistakes still raise;
* the `dangles` flag on the dimer path;
* dangles scoping (pfunc/ensemble ignores the flag);
* the strand-start flank fix at exterior stems (ext_e nick rule).
"""

from __future__ import annotations

import random

import pytest

from strider.thermo import parameters_native  # noqa: F401  (tree-validity marker)
from strider.thermo.parameters import load_parameters
from strider.thermo.hairpin import hairpin_thermo, fraction_folded
from strider.thermo.dimer_thermo import dimer_thermo, dimer_tm
from strider.thermo.engine import ThermoEngine
from strider.structure.mfe import fold_mfe
from strider.structure.complex_fold import fold_complex


# ── 1. Table identity at value level (citation provenance, reviewer item 6) ──

def test_mathews2004_stack_markers_match_the_2004_par_file():
    """The 2004 and 1999 Vienna DNA par files differ in exactly these entries."""
    ps = load_parameters("mathews2004-dna")
    stack = ps.dG["stack"]
    assert stack["ATAT"] == -0.9   # 1999 par has -0.8 here
    assert stack["GTGT"] == 1.2    # 1999 par has 1.3 here
    assert stack["GATT"] == 0.3
    assert stack["AATT"] == -1.0   # Matthews 1999-table invariant baseline
    assert stack["CGCG"] == -2.2   # strongest stack, both par files agree


# ── 2. Pinned folded energies (12k-row JSON cannot silently drift) ───────────

PINNED = {
    # (dG37 kcal/mol, tm_celsius) — captured with the reviewed implementation
    "GCGATTTTATCGC": (-3.2000, 81.1102, "(((((...)))))"),
    "CCGGAAATTCCGG": (-3.7000, 85.2166, "(((((...)))))"),
    "GAGTACTTGTACTC": (-1.2000, 53.9173, "(((((....)))))"),
}


def test_pinned_hairpin_energies_mathews2004():
    for seq, (dg, tm, struct) in PINNED.items():
        th = hairpin_thermo(seq, paramset="mathews2004-dna")
        assert th.structure == struct
        assert round(th.dG37, 4) == dg
        assert round(th.tm_celsius, 4) == tm


# ── 3. paramset-by-name acceptance (historical 'str' object has no 'dG') ─────

def test_paramset_by_name_matches_instance():
    by_name = hairpin_thermo("CCGGAAATTCCGG", paramset="mathews2004-dna")
    by_instance = hairpin_thermo(
        "CCGGAAATTCCGG", paramset=load_parameters("mathews2004-dna")
    )
    assert by_name.structure == by_instance.structure
    assert by_name.dG37 == by_instance.dG37
    assert by_name.tm_celsius == by_instance.tm_celsius
    assert by_name.dH == by_instance.dH


# ── 4. dangles=0 is bit-identical to the current default ─────────────────────


def _grid42():
    cases = [
        "ACTGGTGCTCAGGTTGT", "GCGCAAAAGCGC", "GCGCAAAAAGCGC",
        "CTGATGCATCAG", "GCACGAAACGGC", "GCAAGCAAAGCGC", "GCGGCAAAGCGC",
    ]
    rng = random.Random(42)
    alphabet = "ACGT"
    while len(cases) < 42:
        n = rng.randint(10, 18)
        cases.append("".join(rng.choices(alphabet, k=n)))
    return cases


def test_dangles_zero_bit_identical_grid():
    """Flag-vs-default must never differ on any grid entry (the '42/42' claim)."""
    for seq in _grid42():
        s0, e0, p0 = fold_mfe(seq, 37.0, "dna")
        s1, e1, p1 = fold_mfe(seq, 37.0, "dna", dangles=0)
        assert (s0, e0, p0) == (s1, e1, p1), seq


# ── 5. fraction_folded: zero case and caller mistakes still raise ────────────


def test_fraction_folded_zero_for_unfoldable_sequence():
    assert fraction_folded("AAAAAAAAAAAAAAAA", 25.0,
                           sodium_M=0.05, magnesium_M=0.0) == 0.0


def test_fraction_folded_bad_stem_for_tan_chen_raises():
    """Caller-mistake must propagate, not collapse to 0.0: the sequence folds
    (5-bp stem), so the old blanket except would mask this error; tan_chen is
    fit only for stems >= 6 bp and raises ValueError."""
    with pytest.raises(ValueError):
        fraction_folded("GCGATTTTATCGC", 25.0, salt_model="tan_chen")


# ── 6. dimer path dangles flag (reviewer item 4) ─────────────────────────────

A, B = "GGCATTACGG", "AAAGGATGCC"
_KW = dict(sodium_M=0.05, magnesium_M=0.0092, strand_conc_M=0.25e-6,
           paramset="mathews2004-dna")


def test_dimer_dangles_kwarg_exists_and_shifts_dg():
    r0 = dimer_thermo(A, B, dangles=0, **_KW)
    r2 = dimer_thermo(A, B, dangles=2, **_KW)
    # same structure and same dH (flag scores the ΔG walk only)
    assert r0.structure == r2.structure
    assert r0.dH == r2.dH
    # dangles=2 adds the flanking dangle stack(s) predicted by the tables
    assert r2.dG37 < r0.dG37
    assert round(r0.dG37 - r2.dG37, 4) == 0.8


def test_dimer_tm_accepts_dangles():
    tm = dimer_tm(A, B, dangles=2, **_KW)
    assert isinstance(tm, float)


def test_dimer_dangles_invalid_value_raises():
    with pytest.raises(ValueError):
        dimer_thermo(A, B, dangles=1, **_KW)


# ── 7. dangles is scoped to MFE/subopt (reviewer item 2) ─────────────────────


def test_pfunc_ignores_dangles():
    seq = "GCGATTTTATCGC"
    e0 = ThermoEngine(material="dna", celsius=25.0, sodium=0.05, magnesium=0.0092,
                      dangles=0).pfunc(seq)
    e2 = ThermoEngine(material="dna", celsius=25.0, sodium=0.05, magnesium=0.0092,
                      dangles=2).pfunc(seq)
    assert e0.free_energy == e2.free_energy
    assert (e0.pair_probs == e2.pair_probs).all()


def test_mfe_cache_key_tracks_dangles_but_pfunc_does_not():
    e0 = ThermoEngine(material="dna", dangles=0)
    e2 = ThermoEngine(material="dna", dangles=2)
    assert e0._cache_key("mfe", ("ACGT",)) != e2._cache_key("mfe", ("ACGT",))
    assert e0._cache_key("pfunc", ("ACGT",)) == e2._cache_key("pfunc", ("ACGT",))


# ── 8. strand-start flank rule at exterior stems (ext_e nick fix) ────────────


def test_strand_start_flank_counted_in_complex():
    """Exterior pair (6,16): flank base 5 starts strand 2. The redundant nick
    test used to exclude this dangle; its stack is -0.82 kcal/mol in the table."""
    d0 = fold_complex(["TACCT", "AGATCTAGAACC"], 37.0, "dna", dangles=0)
    d2 = fold_complex(["TACCT", "AGATCTAGAACC"], 37.0, "dna", dangles=2)
    # Pinned with the fix active.  The dangle-inclusive result is structurally
    # different and more stable than the flank-excluded default result.
    assert d0.structure == "((..........&...))"
    assert d0.pairs == [(0, 16), (1, 15)]
    assert round(d0.energy, 4) == -1.28
    assert d2.structure == "((....&...))" or d2.structure == "......((....&...))"
    assert d2.pairs == [(6, 16), (7, 15)]
    assert round(d2.energy, 4) == -1.57
    assert d2.energy < d0.energy
