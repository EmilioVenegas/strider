"""Two-state hairpin Tm: self-consistency and reference agreement."""
import pytest

from strider.thermo.hairpin import hairpin_thermo, hairpin_tm, fraction_folded
from strider.thermo.salt import tan_chen_helix_dg, TAN_CHEN_MIN_BP

# A simple 4-bp / 3-nt-loop hairpin used to anchor against an independent engine.
HP = "CTTTCAACACTGTTGCAGTAA"

# A synthetic 6-bp-stem molecular-beacon-style hairpin (≥ Tan-Chen's fitted range).
HP6 = "GCGCGCAAAAAAAGCGCGC"


def test_matches_independent_reference_at_1M():
    # seqfold (same SantaLucia params) gives ~45.5 C at 1 M Na, 0 Mg.
    tm = hairpin_tm(HP, sodium_M=1.0, magnesium_M=0.0)
    assert tm == pytest.approx(45.5, abs=1.5)


def test_dG_and_Tm_are_self_consistent():
    # ΔG(Tm) must vanish for a two-state melt: fraction folded == 0.5 at Tm.
    th = hairpin_thermo(HP, sodium_M=0.05, magnesium_M=0.010)
    assert fraction_folded(HP, th.tm_celsius, 0.05, 0.010) == pytest.approx(0.5, abs=0.02)


def test_magnesium_raises_tm_monotonically():
    base = hairpin_tm(HP, 0.05, 0.0)
    mid = hairpin_tm(HP, 0.05, 0.003)
    hi = hairpin_tm(HP, 0.05, 0.010)
    assert base < mid < hi


def test_lower_sodium_lowers_tm():
    assert hairpin_tm(HP, 1.0, 0.0) > hairpin_tm(HP, 0.05, 0.0)


def test_stronger_stem_has_higher_tm():
    weak = hairpin_tm(HP, 0.05, 0.010)
    strong = hairpin_tm("CGCGAAAAAGCGCG", 0.05, 0.010)
    assert strong > weak


def test_rejects_non_hairpin():
    with pytest.raises(ValueError):
        hairpin_tm("AAAAAAAAAAAA")  # no pairs


# ─── Tan-Chen (2007) whole-helix salt model ──────────────────────────────────

def test_tan_chen_1M_anchor_is_zero():
    # At 1 M Na+, 0 Mg2+ every Tan-Chen term vanishes for any helix length.
    for N in range(TAN_CHEN_MIN_BP, 16):
        assert tan_chen_helix_dg(N, 1.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_tan_chen_is_default_for_long_stems():
    # A ≥6 bp stem uses Tan-Chen under the default "auto" policy.
    th = hairpin_thermo(HP6, sodium_M=0.05, magnesium_M=0.010)
    assert th.n_pairs >= TAN_CHEN_MIN_BP
    assert th.salt_model == "tan_chen"


def test_short_stem_falls_back_to_per_bp():
    # HP is 4 bp — below Tan-Chen's range — so "auto" must use the per-bp model,
    # and the result must equal explicitly forcing per_bp.
    th = hairpin_thermo(HP, sodium_M=0.05, magnesium_M=0.010)
    assert th.n_pairs < TAN_CHEN_MIN_BP
    assert th.salt_model == "per_bp"
    forced = hairpin_tm(HP, 0.05, 0.010, salt_model="per_bp")
    assert hairpin_tm(HP, 0.05, 0.010) == pytest.approx(forced, abs=1e-9)


def test_tan_chen_raises_when_forced_on_short_stem():
    with pytest.raises(ValueError):
        hairpin_tm(HP, 0.05, 0.010, salt_model="tan_chen")  # N=4 outside fit


def test_tan_chen_mg_slope_matches_experiment():
    # Tan-Chen reproduces the measured DNA-beacon Mg2+ Tm slope (~0.7 °C/mM at
    # 50 mM Na+) and is steeper than the per-bp model, which under-shoots.
    def slope(model):
        hi = hairpin_tm(HP6, 0.05, 0.010, salt_model=model)
        lo = hairpin_tm(HP6, 0.05, 0.00225, salt_model=model)
        return (hi - lo) / (10.0 - 2.25)
    tc = slope("tan_chen")
    assert 0.55 < tc < 0.95           # measured ≈0.70; model ≈0.71
    assert tc > slope("per_bp")       # per-bp under-shoots Mg sensitivity


def test_tan_chen_magnesium_raises_tm_monotonically():
    base = hairpin_tm(HP6, 0.05, 0.0, salt_model="tan_chen")
    mid = hairpin_tm(HP6, 0.05, 0.003, salt_model="tan_chen")
    hi = hairpin_tm(HP6, 0.05, 0.010, salt_model="tan_chen")
    assert base < mid < hi
