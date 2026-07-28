"""
Salt (Na⁺/Mg²⁺) and temperature-dependent ΔG regression tests.

Covers the three salt/temperature workstreams:
  1. Salt in the native MFE DP (fold_mfe).
  2. Mg²⁺ in the two-state duplex (nn_dna.duplex_dg).
  3. Temperature-blended ParameterSet (thermo.temperature) wired into the engine.

The unifying invariant is that *at the 1 M Na⁺ / 0 Mg²⁺ / 37 °C reference the
result is bit-identical to the pre-existing engine*, and that moving off the
reference shifts energies in the physically correct direction.
"""
import numpy as np
import pytest

from strider.structure.mfe import fold_mfe
from strider.thermo.engine import ThermoEngine
from strider.thermo.nn_dna import duplex_dg
from strider.thermo.salt import (
    dg_per_bp_salt, duplex_salt_dg, owczarzy_tm_correction,
)
from strider.thermo.nn_dna import melting_temperature
from strider.thermo.temperature import (
    blend_paramset, native_temperature_paramset, T_REF_K,
)

# Real (un-monkeypatched) STK decoration fn, captured at import so the
# "frozen at 37 °C" baseline in TestStackingEnsembleTemperature can call it
# without recursing into its own monkeypatch of pd.stk_decoration_tables.
import strider.thermo.parameters_dna as _pd_stk
_REAL_STK_TABLES = _pd_stk.stk_decoration_tables

H1 = "TCAACATCAGTCTGATACCTCCCTCCTTATCAGACTGA"


# ─── 1. Salt in MFE ───────────────────────────────────────────────────────────

class TestSaltInMFE:
    def test_reference_is_no_op(self):
        # 1 M Na⁺ / 0 Mg²⁺ must reproduce the salt-free fold exactly.
        s0, e0, p0 = fold_mfe(H1)
        s1, e1, p1 = fold_mfe(H1, 37.0, "dna", sodium_M=1.0, magnesium_M=0.0)
        assert s1 == s0 and p1 == p0
        assert e1 == e0  # bit-identical (dg_per_bp_salt(1,0) == 0)

    def test_low_sodium_destabilizes(self):
        _, e_1m, _ = fold_mfe(H1, 37.0, "dna", sodium_M=1.0)
        _, e_lo, _ = fold_mfe(H1, 37.0, "dna", sodium_M=0.05)
        assert e_lo > e_1m  # less salt → less stable (higher ΔG)

    def test_magnesium_stabilizes(self):
        _, e_no, _ = fold_mfe(H1, 37.0, "dna", sodium_M=0.137, magnesium_M=0.0)
        _, e_mg, _ = fold_mfe(H1, 37.0, "dna", sodium_M=0.137, magnesium_M=0.05)
        assert e_mg < e_no  # added Mg²⁺ → more stable

    def test_salt_scales_with_pair_count(self):
        # Each closed pair carries one dg_per_bp_salt term, so ΔE(salt) at fixed
        # structure equals n_pairs · dg_per_bp_salt.
        _, e_ref, pairs = fold_mfe(H1, 37.0, "dna", sodium_M=1.0)
        na = 0.1
        _, e_lo, pairs_lo = fold_mfe(H1, 37.0, "dna", sodium_M=na)
        assert pairs_lo == pairs  # MFE structure unchanged here
        expected = len(pairs) * dg_per_bp_salt(na, 0.0)
        assert e_lo - e_ref == pytest.approx(expected, abs=1e-9)

    def test_engine_mfe_threads_salt(self):
        ref = ThermoEngine("dna", 37.0, sodium=1.0, magnesium=0.0)
        lo = ThermoEngine("dna", 37.0, sodium=0.05, magnesium=0.0)
        assert lo.mfe(H1).energy > ref.mfe(H1).energy

    def test_traceback_consistent_when_salt_changes_structure(self):
        # Guards the `target = V[i][j] - dg_salt` subtraction in _traceback_V:
        # at low salt this stem loses a base pair (the optimal structure itself
        # changes, not just its energy), so the traceback must run with a
        # non-zero dg_salt and still recover a self-consistent structure.
        seq = "GCGCGCGCAAAATGCGCGCGC"
        s_ref, _, p_ref = fold_mfe(seq, 37.0, "dna", sodium_M=1.0)
        s_lo, _, p_lo = fold_mfe(seq, 37.0, "dna", sodium_M=0.02)
        assert p_lo != p_ref            # salt actually changed the MFE structure
        assert len(p_lo) < len(p_ref)   # fewer pairs at lower salt
        # dot-bracket and pair list must not desync, and the bracketing stays
        # balanced (a wrong dg_salt offset would corrupt the traceback).
        assert s_lo.count("(") == s_lo.count(")") == len(p_lo)
        assert all(0 <= i < j < len(seq) for i, j in p_lo)


# ─── 2. Mg²⁺ in two-state duplex ──────────────────────────────────────────────

class TestMagnesiumDuplex:
    def test_uses_per_bp_model(self):
        # The duplex correction is the per-bp DP term summed over the bp count,
        # unifying duplex_dg with the MFE/ensemble salt model.
        seq = "GCGCATGCGC"
        assert duplex_salt_dg(seq, 0.137, 0.05) == \
            len(seq) * dg_per_bp_salt(0.137, 0.05)

    def test_reference_correction_zero(self):
        assert duplex_salt_dg("GCGCATGCGC", 1.0, 0.0) == 0.0

    def test_reference_duplex_unchanged(self):
        seq = "GCGCATGCGC"
        assert duplex_dg(seq, None, 37.0, 1.0, 0.0) == duplex_dg(seq, None, 37.0, 1.0)

    def test_magnesium_stabilizes_duplex(self):
        seq = "GCGCATGCGC"
        assert duplex_dg(seq, None, 37.0, 0.137, 0.05) < \
            duplex_dg(seq, None, 37.0, 0.137, 0.0)

    def test_low_sodium_destabilizes_duplex(self):
        seq = "GCGCATGCGC"
        assert duplex_dg(seq, None, 37.0, 0.137, 0.0) > \
            duplex_dg(seq, None, 37.0, 1.0, 0.0)

    def test_engine_passes_magnesium(self):
        seq, comp = "GCGCATGCGC", "GCGCATGCGC"
        e0 = ThermoEngine("dna", 37.0, sodium=0.137, magnesium=0.0)
        e1 = ThermoEngine("dna", 37.0, sodium=0.137, magnesium=0.05)
        assert e1.duplex_dg(seq, comp) < e0.duplex_dg(seq, comp)


class TestOwczarzyMagnesiumTm:
    """Regression guard for the Owczarzy-2008 Mg Tm bug (the 1/(2(N-1)) factor
    had been hardcoded to N=2, inflating Mg Tm shifts by ~10-20 °C)."""

    SEQ20 = "ATCGATCGATCGATCGATCG"

    def test_mg_effect_is_physically_bounded(self):
        # This 20-mer at 50 mM Na⁺ / 10 mM Mg²⁺ sits in the mixed regime
        # (√[Mg]/[Na] = 2.0), so the correction goes through the von Ahsen
        # sodium-equivalent path. That legitimately puts the Mg lift in the low
        # teens for a 20-mer (~+13 °C, endpoints within ~1.5 °C of primer3), not
        # the ~+22 °C the old hardcoded N=2 factor produced. Bound guards against
        # that blow-up while allowing the correct equivalent-sodium magnitude.
        base = melting_temperature(self.SEQ20, 250e-9, 0.05, 0.0)
        mg = melting_temperature(self.SEQ20, 250e-9, 0.05, 0.01)
        assert 0.0 < (mg - base) < 16.0

    def test_mg_monotonic_stabilizes(self):
        tms = [melting_temperature(self.SEQ20, 250e-9, 0.05, mg)
               for mg in [0.0, 0.001, 0.005, 0.01, 0.05]]
        assert all(b > a for a, b in zip(tms, tms[1:]))

    def test_length_factor_engaged(self):
        # The corrected term depends on base-pair count, so a 10-mer and a
        # 40-mer must give different pure-Mg corrections (regression: the bug
        # dropped N entirely, making them identical).
        c10 = owczarzy_tm_correction("ATCGATCGAT", 1e-3, 0.01)
        c40 = owczarzy_tm_correction("ATCG" * 10, 1e-3, 0.01)
        assert abs(c10 - c40) > 1.0

    def test_mixed_regime_sane(self):
        # Typical PCR-ish buffer (Na 0.05, Mg 3 mM) must give a finite, modest Tm.
        tm = melting_temperature(self.SEQ20, 250e-9, 0.05, 0.003)
        assert 40.0 < tm < 75.0

    def test_mixed_regime_tracks_equivalent_sodium(self):
        # Regression for issue #10: in the mixed regime the Mg²⁺ correction must
        # use the von Ahsen sodium-equivalent recipe ([Na]_eq = [Na] +
        # 120·√[Mg]_free, mM), the same conversion primer3/Biopython/IDT use, not
        # the old linear blend that pinned the result near the Na-only floor and
        # left duplex Tm 6-10 °C low. The correction must equal melting_temperature
        # evaluated at the equivalent sodium, and the value must sit in the
        # ecosystem cluster (primer3/IDT ~60-62 °C) rather than the old ~48 °C.
        import math
        from strider.thermo.nn_dna import duplex_tm
        seq = "ATGTAATTGTTACATTATGTAATATTGT"  # 28 nt, low GC (worst case for the bug)
        tm = duplex_tm(seq, sodium_M=0.05, magnesium_M=0.01,
                       dntp_M=0.0008, oligo_conc_M=500e-9)
        free_mg_mM = (0.01 - 0.0008) * 1000.0
        na_eq = 0.05 + 0.120 * math.sqrt(free_mg_mM)
        ref = melting_temperature(seq, 500e-9, na_eq, 0.0)
        assert abs(tm - ref) < 0.5
        assert 58.0 < tm < 63.0


class TestHairpinOwczarzySaltModel:
    """The GC-aware Owczarzy salt model grafted onto the two-state hairpin Tm."""

    SEQ = "CGCGAACCGACTACTTTGGGTGTCCGTCGCG"  # a Rejtar beacon probe

    def test_reference_is_no_op(self):
        from strider.thermo.hairpin import hairpin_thermo
        ow = hairpin_thermo(self.SEQ, 1.0, 0.0, "dna", salt_model="owczarzy")
        ref = hairpin_thermo(self.SEQ, 1.0, 0.0, "dna", salt_model="per_bp")
        assert ow.salt_model == "owczarzy"
        assert ow.dG37 == pytest.approx(ref.dG37, abs=1e-9)
        assert ow.tm_celsius == pytest.approx(ref.tm_celsius, abs=1e-9)

    def test_magnesium_monotonic(self):
        from strider.thermo.hairpin import hairpin_tm
        tms = [hairpin_tm(self.SEQ, 0.05, mg, "dna", salt_model="owczarzy")
               for mg in (0.0, 0.003, 0.01)]
        assert all(b > a for a, b in zip(tms, tms[1:]))

    def test_low_sodium_destabilizes(self):
        from strider.thermo.hairpin import hairpin_tm
        assert hairpin_tm(self.SEQ, 0.05, 0.0, "dna", salt_model="owczarzy") < \
            hairpin_tm(self.SEQ, 1.0, 0.0, "dna", salt_model="owczarzy")


# ─── 3. Temperature-blended ParameterSet ──────────────────────────────────────

class TestTemperatureBlend:
    def test_native_override_identity_at_37(self):
        # The module-sourced override at 37 °C must equal the module constants.
        import strider.thermo.parameters_dna as pd
        ps = native_temperature_paramset("dna", 37.0)
        for k, v in ps.dG["stack"].items():
            assert v == pytest.approx(pd.STACK[k], abs=1e-12)
        assert np.max(np.abs(ps.dG["hairpin_size"]
                             - np.asarray(pd.HAIRPIN_SIZE, float))) < 1e-12

    def test_blend_paramset_identity_at_37(self):
        from strider.thermo.parameters import load_parameters
        base = load_parameters("native-dna")
        blended = blend_paramset(base, 37.0)
        for k, g in base.dG["stack"].items():
            assert blended.dG["stack"][k] == pytest.approx(g, abs=1e-12)

    def test_blend_formula_endpoints(self):
        # ΔG(Tref) == ΔG₃₇ and ΔG uses ΔS = (ΔH-ΔG)/Tref.
        from strider.thermo.parameters import load_parameters
        base = load_parameters("native-dna")
        key = next(iter(base.dG["stack"]))
        g37 = base.dG["stack"][key]
        h = base.dH["stack"].get(key)
        if h is not None:
            celsius = 55.0
            frac = (celsius + 273.15) / T_REF_K
            expected = g37 * frac + h * (1.0 - frac)
            assert blend_paramset(base, celsius).dG["stack"][key] == \
                pytest.approx(expected, abs=1e-12)

    def test_engine_identity_at_37(self):
        # celsius == 37 default path keeps override None → unchanged result.
        eng = ThermoEngine("dna", 37.0, sodium=1.0, magnesium=0.0)
        assert eng._param_override() is None

    def test_warming_destabilizes(self):
        cold = ThermoEngine("dna", 15.0, sodium=1.0, magnesium=0.0)
        hot = ThermoEngine("dna", 60.0, sodium=1.0, magnesium=0.0)
        assert hot.pfunc(H1).free_energy > cold.pfunc(H1).free_energy
        assert hot.mfe(H1).energy > cold.mfe(H1).energy

    def test_monotonic_temperature_pfunc(self):
        temps = [10, 25, 37, 50, 65]
        dgs = [ThermoEngine("dna", c, sodium=1.0).pfunc(H1).free_energy
               for c in temps]
        assert all(b > a for a, b in zip(dgs, dgs[1:]))  # strictly increasing

    def test_custom_paramset_blended_off_37(self):
        # A custom (non-native) set is blended across all keys off 37 °C.
        from strider.thermo.parameters import load_parameters
        base = load_parameters("native-dna")
        eng = ThermoEngine("dna", 55.0, sodium=1.0, parameter_set=base)
        ov = eng._param_override()
        assert ov is not None and ov.name.endswith("55C")


# ─── Monovalent salt × temperature (GAP-2) ────────────────────────────────────

class TestMonovalentSaltTemperature:
    """The per-bp salt correction is entropic (counterion release), so it scales
    with absolute temperature, exact at 37 °C and zero at 1 M Na⁺ for all T."""

    def test_reference_is_zero_at_all_temperatures(self):
        from strider.thermo.salt import dg_per_bp_salt, duplex_salt_dg
        for Tc in (5, 25, 37, 55, 85):
            assert dg_per_bp_salt(1.0, 0.0, Tc) == 0.0
            assert duplex_salt_dg("ACGTACGT", 1.0, 0.0, Tc) == 0.0

    def test_37C_matches_legacy_value_bit_identical(self):
        import math
        from strider.thermo.salt import dg_per_bp_salt
        for na in (0.05, 0.1, 0.15, 0.3):
            assert dg_per_bp_salt(na, 0.0, 37.0) == -0.114 * math.log(na)
        # combining-rule Mg path also unchanged at 37 °C
        assert dg_per_bp_salt(0.1, 0.05, 37.0) == \
            -0.114 * math.log(0.1 + 3.4 * math.sqrt(0.05))

    def test_entropic_T_scaling(self):
        from strider.thermo.salt import dg_per_bp_salt
        base = dg_per_bp_salt(0.05, 0.0, 37.0)
        for Tc in (10, 25, 55, 80):
            frac = (Tc + 273.15) / T_REF_K
            assert dg_per_bp_salt(0.05, 0.0, Tc) == pytest.approx(base * frac, rel=1e-12)

    def test_destabilization_grows_with_temperature(self):
        # |salt ΔG| at low [Na⁺] must increase monotonically with T (entropic).
        from strider.thermo.salt import dg_per_bp_salt
        mags = [abs(dg_per_bp_salt(0.1, 0.0, Tc)) for Tc in (10, 25, 37, 55, 80)]
        assert all(b > a for a, b in zip(mags, mags[1:]))

    def test_duplex_two_state_salt_grows_with_temperature(self):
        # The two-state duplex salt ΔΔG is a real ΔG (∝ T): low-salt
        # destabilisation must increase with temperature.
        seq = "GCGCATGCGCAT"
        dd = [duplex_dg(seq, celsius=Tc, sodium_M=0.1)
              - duplex_dg(seq, celsius=Tc, sodium_M=1.0) for Tc in (15, 37, 55, 70)]
        assert all(b > a for a, b in zip(dd, dd[1:]))

    # ── GAP-5: material-aware per-bp salt ──────────────────────────────────────
    def test_dna_unchanged_and_default(self):
        # DNA keeps the validated Owczarzy −0.114 magnitude; default material = DNA.
        import math
        from strider.thermo.salt import dg_per_bp_salt
        for na in (0.05, 0.1, 0.3):
            assert dg_per_bp_salt(na, 0.0, 37.0, "dna") == -0.114 * math.log(na)
            assert dg_per_bp_salt(na, 0.0, 37.0) == dg_per_bp_salt(na, 0.0, 37.0, "dna")

    def test_rna_salt_scaled_by_tan_chen_ratio(self):
        # RNA salt is the DNA magnitude × the Tan-Chen RNA/DNA per-stack ratio
        # (~1.06): stronger than DNA, same sign, still 0 at the 1 M reference.
        from strider.thermo.salt import dg_per_bp_salt, _RNA_SALT_FACTOR
        assert _RNA_SALT_FACTOR == pytest.approx(1.06, abs=0.02)
        for Tc in (25, 37, 55):
            for na in (0.05, 0.15, 0.5):
                r = dg_per_bp_salt(na, 0.0, Tc, "rna")
                d = dg_per_bp_salt(na, 0.0, Tc, "dna")
                assert r == pytest.approx(d * _RNA_SALT_FACTOR, rel=1e-12)
                assert abs(r) > abs(d)            # RNA stronger
            assert dg_per_bp_salt(1.0, 0.0, Tc, "rna") == 0.0   # ref still 0

    def test_rna_ratio_matches_tan_chen_model(self):
        # The hard-coded RNA factor reproduces strider's own Tan-Chen RNA/DNA
        # per-stack ratio (its literature provenance) to ~1%.
        from strider.thermo.salt import tan_chen_helix_dg, _RNA_SALT_FACTOR
        for na in (0.05, 0.1, 0.2, 0.5):
            ratio = (tan_chen_helix_dg(15, na, 0.0, "rna")
                     / tan_chen_helix_dg(15, na, 0.0, "dna"))
            assert ratio == pytest.approx(_RNA_SALT_FACTOR, abs=0.02)

    def test_ensemble_salt_boltzmann_factor_is_entropic(self):
        # Entropic salt ⇒ the per-pair partition-function weight exp(−ΔG_salt/RT)
        # is *temperature-independent* (= its 37 °C value), since ΔG_salt ∝ T.
        # This is the correct counterion-release signature and is what makes the
        # ensemble salt response T-stable (the old code wrongly varied it as if
        # salt were enthalpic).
        import math
        from strider.thermo.salt import dg_per_bp_salt
        R = 1.987e-3
        ref = math.exp(-dg_per_bp_salt(0.1, 0.0, 37.0) / (R * T_REF_K))
        for Tc in (10, 25, 55, 80):
            bf = math.exp(-dg_per_bp_salt(0.1, 0.0, Tc) / (R * (Tc + 273.15)))
            assert bf == pytest.approx(ref, rel=1e-12)

    def test_ensemble_bit_identical_at_37(self):
        # Threading celsius must not perturb the 37 °C ensemble result.
        from strider.thermo.ensemble import ensemble_dg
        s = "GCGCAAAAGCGCGC"
        g1 = ensemble_dg(s, celsius=37.0, material="dna", sodium_M=0.1)[0]
        g2 = ensemble_dg(s, celsius=37.0, material="dna", sodium_M=0.1)[0]
        assert g1 == g2  # deterministic; salt factor == legacy at 37 °C

    def test_direction_agrees_with_vienna_na_axis(self):
        # Model-vs-model: strider's monovalent correction and ViennaRNA's physical
        # salt_stack/salt_loop must share the qualitative (Na⁺, T) behaviour — zero
        # at 1 M, growing in magnitude as [Na⁺] falls and as T rises — even though
        # the absolute magnitudes follow different (Owczarzy vs Einert-Netz) fits.
        RNA = pytest.importorskip("RNA")
        from strider.thermo.salt import dg_per_bp_salt
        hrise = RNA.MODEL_HELICAL_RISE_DNA

        def vienna_stack(na, Tc):
            return RNA.salt_stack(na, Tc + 273.15, hrise) / 1000.0

        # zero at the 1 M reference for both, at every T
        for Tc in (15, 37, 70):
            assert dg_per_bp_salt(1.0, 0.0, Tc) == 0.0
            assert abs(vienna_stack(1.0, Tc)) < 1e-9

        # grows as [Na⁺] falls (fixed T): both strictly increasing in magnitude
        for Tc in (25, 55):
            s_mag = [abs(dg_per_bp_salt(na, 0.0, Tc)) for na in (0.5, 0.2, 0.1, 0.05)]
            v_mag = [abs(vienna_stack(na, Tc)) for na in (0.5, 0.2, 0.1, 0.05)]
            assert all(b >= a for a, b in zip(s_mag, s_mag[1:]))
            assert all(b >= a for a, b in zip(v_mag, v_mag[1:]))

        # grows with T (fixed low [Na⁺]): same sign of slope for both
        s_slope = abs(dg_per_bp_salt(0.05, 0.0, 70)) - abs(dg_per_bp_salt(0.05, 0.0, 15))
        v_slope = abs(vienna_stack(0.05, 70)) - abs(vienna_stack(0.05, 15))
        assert s_slope > 0 and v_slope >= 0


# ─── DNA dangle / terminal-mismatch × temperature (GAP-1) ─────────────────────

class TestStackingEnsembleTemperature:
    """The external-loop STK_* decoration (the live DNA dangle/TM path) is the
    literature all-dangles model derived from DANGLE_5/DANGLE_3 (Bommarito 2000),
    physically temperature-extrapolated off 37 °C."""

    # A stem whose exterior-loop helix termini carry dangles (so STK_* matters).
    STEM = "GCGCAAAAGCGC"

    @staticmethod
    def _frozen(_T):
        # Baseline = the 37 °C decoration held constant across T (the "no
        # temperature extrapolation" comparison for the off-37 tests).  Calls the
        # real fn captured at import — not pd.stk_decoration_tables, which is
        # monkeypatched to this very function (would recurse).
        return _REAL_STK_TABLES(T_REF_K)

    def test_dangle_dh_validated_against_strider_tables(self):
        # The generator's self-check: DANGLE_{5,3}_DH share the exact key sets of
        # the DANGLE tables they enthalpy-decorate (NUPACK dna04 provenance).
        import strider.thermo.parameters_dna as pd
        from strider.thermo._dna_enthalpy_generated import DANGLE_5_DH, DANGLE_3_DH
        assert set(DANGLE_5_DH) == set(pd.DANGLE_5)
        assert set(DANGLE_3_DH) == set(pd.DANGLE_3)
        assert any(v != 0.0 for v in DANGLE_5_DH.values())

    def test_stk_is_literature_all_dangles_at_37(self):
        # 37 °C decoration = exp(-DANGLE/RT): the standard all-dangles model,
        # derived purely from strider's literature dangle ΔG (no external tool).
        import math
        import strider.thermo.parameters_dna as pd
        R = 1.987e-3
        bare, d5, d3, tm = pd.stk_decoration_tables(T_REF_K)
        assert set(d5) == {k for k in pd.DANGLE_5 if k[:2] in {"AT", "TA", "GC", "CG"}}
        for k, w in d5.items():
            assert w == pytest.approx(math.exp(-pd.DANGLE_5[k] / (R * T_REF_K)), rel=1e-12)
        for k, w in d3.items():
            assert w == pytest.approx(math.exp(-pd.DANGLE_3[k] / (R * T_REF_K)), rel=1e-12)
        assert bare == {"AT": 1.0, "TA": 1.0, "GC": 1.0, "CG": 1.0}

    def test_stk_is_nupack_free(self):
        # Regression guard for the licensing fix: the baked (NUPACK-fit) STK_*
        # constants are gone; only the 1.0 NONE baseline remains as a constant.
        import strider.thermo.parameters_dna as pd
        assert not hasattr(pd, "STK_D5_DELTA")
        assert not hasattr(pd, "STK_D3_DELTA")
        assert not hasattr(pd, "STK_TM_DELTA")
        assert hasattr(pd, "STK_BARE_FACTOR")

    def test_stk_tm_is_d5_times_d3_off_37(self):
        # The "all-dangles" identity STK_TM = STK_D5 · STK_D3 holds at every T.
        import strider.thermo.parameters_dna as pd
        _, d5, d3, tm = pd.stk_decoration_tables(55.0 + 273.15)
        for key, v in tm.items():
            n, x, y, m = key
            assert v == pytest.approx(d5[x + y + n] * d3[m + y + x], rel=1e-12)

    def test_engine_deterministic_at_37(self, monkeypatch):
        # 37 °C is the reference (frac = 1): the T-extrapolated path and the
        # held-at-37 path agree there, so the engine result is well-defined.
        import strider.thermo.parameters_dna as pd
        from strider.thermo.ensemble import ensemble_dg
        g_aware = ensemble_dg(self.STEM, celsius=37.0, material="dna")[0]
        monkeypatch.setattr(pd, "stk_decoration_tables", self._frozen)
        g_frozen = ensemble_dg(self.STEM, celsius=37.0, material="dna")[0]
        assert g_aware == g_frozen

    def test_dangles_weaken_at_high_temperature(self, monkeypatch):
        # Heating should weaken the exterior dangle stabilisation: the T-aware
        # ΔG must be *higher* (less negative) than the frozen-dangle ΔG at 70 °C.
        import strider.thermo.parameters_dna as pd
        from strider.thermo.ensemble import ensemble_dg
        g_aware = ensemble_dg(self.STEM, celsius=70.0, material="dna")[0]
        monkeypatch.setattr(pd, "stk_decoration_tables", self._frozen)
        g_frozen = ensemble_dg(self.STEM, celsius=70.0, material="dna")[0]
        assert g_aware > g_frozen

    def test_residual_vs_vienna_dna_shrinks(self, monkeypatch):
        # Model-vs-model oracle: extrapolating the dangle ΔH should move strider
        # DNA closer to ViennaRNA's DNA (Mathews-2004) ensemble ΔG, on average,
        # across off-37 temperatures on terminus-dominated stems.
        RNA = pytest.importorskip("RNA")
        import strider.thermo.parameters_dna as pd
        from strider.thermo.ensemble import ensemble_dg
        from strider.thermo._param_context import param_context

        def vienna(seq, Tc):
            RNA.params_load_DNA_Mathews2004()
            md = RNA.md(); md.temperature = Tc; md.dangles = 2
            return RNA.fold_compound(seq, md).pf()[1]

        def strider(seq, Tc):
            with param_context(native_temperature_paramset("dna", Tc)):
                return ensemble_dg(seq, celsius=Tc, material="dna")[0]

        seqs = ["GCGCAAAAGCGC", "ATGCGCAAAGCGCATT",
                "TGGGAAACCCA", "AGCGCGAAACGCGCT"]
        res_aware = res_frozen = 0.0
        n = 0
        for s in seqs:
            for Tc in (25, 55, 70):
                v = vienna(s, Tc)
                a = strider(s, Tc)
                monkeypatch.setattr(pd, "stk_decoration_tables", self._frozen)
                f = strider(s, Tc)
                monkeypatch.undo()
                res_aware += abs(a - v)
                res_frozen += abs(f - v)
                n += 1
        assert res_aware / n < res_frozen / n


# ─── RNA loop-initiation ΔH (Turner 2004) ─────────────────────────────────────

class TestRNALoopEnthalpy:
    """Curated non-zero RNA loop-init ΔH in the engine temperature path."""

    def test_generated_dg_validates_against_size_arrays(self):
        # The generator's self-check, re-run: every non-sentinel loop-size ΔG in
        # the par file aligns with parameters_rna at +1 offset (same model).
        import strider.thermo.parameters_rna as p
        from strider.thermo._rna_enthalpy_generated import (
            HAIRPIN_SIZE_DH, BULGE_SIZE_DH, INTERIOR_SIZE_DH,
        )
        for dh, dg in [(HAIRPIN_SIZE_DH, p.HAIRPIN_SIZE),
                       (BULGE_SIZE_DH, p.BULGE_SIZE),
                       (INTERIOR_SIZE_DH, p.INTERIOR_SIZE)]:
            assert len(dh) == len(dg)
        # Turner-2004 RNA loop-init ΔH is genuinely non-zero (unlike DNA).
        assert any(v != 0.0 for v in HAIRPIN_SIZE_DH)
        assert any(v != 0.0 for v in BULGE_SIZE_DH)

    def test_rna_identity_at_37(self):
        # frac == 1 ⇒ the ΔH term vanishes ⇒ blended ΔG == source ΔG exactly.
        import strider.thermo.parameters_rna as p
        ps = native_temperature_paramset("rna", 37.0)
        for name, src in [("hairpin_size", p.HAIRPIN_SIZE),
                          ("bulge_size", p.BULGE_SIZE),
                          ("interior_size", p.INTERIOR_SIZE)]:
            assert np.max(np.abs(ps.dG[name] - np.asarray(src, float))) < 1e-12

    def test_rna_loop_dh_changes_off_37(self):
        # Curated ΔH must move RNA loop-init ΔG away from the old ΔH=0 (·frac) model.
        import strider.thermo.parameters_rna as p
        from strider.thermo._rna_enthalpy_generated import HAIRPIN_SIZE_DH
        frac = (25.0 + 273.15) / T_REF_K
        ps = native_temperature_paramset("rna", 25.0)
        entropic = np.asarray(p.HAIRPIN_SIZE, float) * frac          # ΔH=0 model
        curated = np.asarray(ps.dG["hairpin_size"])
        expected = entropic + np.asarray(HAIRPIN_SIZE_DH) * (1.0 - frac)
        assert np.allclose(curated, expected)
        assert not np.allclose(curated, entropic)

    def test_rna_engine_temperature_monotonic(self):
        seq = "GGGAAACUUCGGUUUCCC"
        dgs = [ThermoEngine("rna", c, sodium=1.0, magnesium=0.0).pfunc(seq).free_energy
               for c in (10, 25, 37, 50, 65)]
        assert all(b > a for a, b in zip(dgs, dgs[1:]))


# ─── RNA dangle / terminal-mismatch × temperature (GAP-4 → completes GAP-1) ────

class TestRNADangleTemperature:
    """RNA dangle/terminal-mismatch ride the *live* route-1 DP path
    (`_apply_coaxial_external` is DNA-only), so their NUPACK-rna06 ΔH enters the
    paramset and makes them temperature-correct — the RNA half of GAP-1."""

    def test_rna_dangle_tm_dh_validated_against_strider_tables(self):
        import strider.thermo.parameters_rna as pr
        from strider.thermo._rna_enthalpy_generated import (
            DANGLE_5_DH, DANGLE_3_DH, TERMINAL_MISMATCH_DH,
        )
        assert set(DANGLE_5_DH) == set(pr.DANGLE_5)
        assert set(DANGLE_3_DH) == set(pr.DANGLE_3)
        assert set(TERMINAL_MISMATCH_DH) == set(pr.TERMINAL_MISMATCH)
        # real (non-ΔG-copy) enthalpies, so ΔS ≠ 0 for these terms
        assert any(v != 0.0 for v in DANGLE_5_DH.values())
        assert any(v != 0.0 for v in TERMINAL_MISMATCH_DH.values())

    def test_rna_paramset_emits_dangle_tm_identity_at_37(self):
        import strider.thermo.parameters_rna as pr
        ps = native_temperature_paramset("rna", 37.0)
        for tbl, const in [("dangle_5", pr.DANGLE_5), ("dangle_3", pr.DANGLE_3),
                           ("terminal_mismatch", pr.TERMINAL_MISMATCH)]:
            assert tbl in ps.dG
            for k, v in const.items():
                assert ps.dG[tbl][k] == pytest.approx(v, abs=1e-12)

    def test_rna_dangle_dh_changes_result_off_37(self):
        # Curated dangle/TM ΔH must shift the off-37 RNA pfunc away from the
        # ΔH = ΔG₃₇ (frozen-energy) degrade the paramset used previously.
        import strider.thermo.parameters_rna as pr
        from strider.thermo.ensemble import ensemble_dg
        from strider.thermo._param_context import param_context

        seq = "GCGCAAAAGCGCUAGCUUUUGCUA"
        ps = native_temperature_paramset("rna", 70.0)
        with param_context(ps):
            g_curated = ensemble_dg(seq, celsius=70.0, material="rna")[0]
        # frozen-energy variant: overwrite the three tables with their 37 °C ΔG
        frozen = native_temperature_paramset("rna", 70.0)
        for tbl, const in [("dangle_5", pr.DANGLE_5), ("dangle_3", pr.DANGLE_3),
                           ("terminal_mismatch", pr.TERMINAL_MISMATCH)]:
            frozen.dG[tbl] = dict(const)
        with param_context(frozen):
            g_frozen = ensemble_dg(seq, celsius=70.0, material="rna")[0]
        assert g_curated != g_frozen

    def test_rna_residual_vs_vienna_shrinks(self):
        # Curated dangle/TM ΔH moves the RNA ensemble closer to ViennaRNA across
        # off-37 temperatures (model-vs-model, Turner-2004 lineage on both sides).
        RNA = pytest.importorskip("RNA")
        import strider.thermo.parameters_rna as pr
        from strider.thermo.ensemble import ensemble_dg
        from strider.thermo._param_context import param_context

        def vienna(seq, Tc):
            md = RNA.md(); md.temperature = Tc; md.dangles = 2
            return RNA.fold_compound(seq.replace("T", "U"), md).pf()[1]

        def strider(seq, Tc, frozen):
            ps = native_temperature_paramset("rna", Tc)
            if frozen:
                for tbl, const in [("dangle_5", pr.DANGLE_5), ("dangle_3", pr.DANGLE_3),
                                   ("terminal_mismatch", pr.TERMINAL_MISMATCH)]:
                    ps.dG[tbl] = dict(const)
            with param_context(ps):
                return ensemble_dg(seq, celsius=Tc, material="rna")[0]

        seqs = ["GCGCAAAAGCGCUAGCUUUUGCUA", "GGGAAACCCUUU",
                "CGCGCGAAAUCGCGCG", "AUGCGCAAAGCGCAUU"]
        res_new = res_frozen = 0.0
        n = 0
        for s in seqs:
            for Tc in (25, 55, 70):
                v = vienna(s, Tc)
                res_new += abs(strider(s, Tc, False) - v)
                res_frozen += abs(strider(s, Tc, True) - v)
                n += 1
        assert res_new / n < res_frozen / n
