"""
Salt (Na⁺/Mg²⁺) and temperature-dependent ΔG regression tests.

Covers the three workstreams of DIMER_SALT_TEMP_PLAN.md:
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
        # Adding 10 mM Mg²⁺ at low Na⁺ should raise Tm by a single-digit amount,
        # not the ~+22 °C the buggy factor produced.
        base = melting_temperature(self.SEQ20, 250e-9, 0.05, 0.0)
        mg = melting_temperature(self.SEQ20, 250e-9, 0.05, 0.01)
        assert 0.0 < (mg - base) < 12.0

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
