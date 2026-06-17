"""Regression and parity tests for multi-strand (dimer) MFE prediction.

These tests drive the native dimer-MFE implementation.  Under the current
codebase ``ThermoEngine.mfe('AAAA', 'TTTT')`` is folded as the single
concatenated strand ``'AAAATTTT'``, so dimer-aware behaviour is expected to
fail until the production code is updated.
"""
from __future__ import annotations

import math

import pytest

from strider.equilibrium import cyclic_symmetry
from strider.structure.mfe import fold_mfe
from strider.thermo.engine import R, ThermoEngine


class TestNativeDimerMFE:
    """Native backend must return strand-aware MFE structures for dimers."""

    def test_sequence_contains_separator(self):
        r = ThermoEngine(backend="native").mfe("AAAA", "TTTT")
        assert "&" in r.sequence
        assert r.sequence == "AAAA&TTTT"

    def test_structure_contains_separator_at_break(self):
        r = ThermoEngine(backend="native").mfe("AAAA", "TTTT")
        assert "&" in r.structure
        assert r.structure.index("&") == len("AAAA")

    def test_complementary_dimer_negative_energy(self):
        r = ThermoEngine(backend="native").mfe("AAAA", "TTTT")
        assert r.energy < 0.0, "Complementary dimer should be favourable"

    def test_acgt_tgca_forms_dimer(self):
        r = ThermoEngine(backend="native").mfe("ACGT", "TGCA")
        assert "&" in r.sequence
        assert "&" in r.structure
        # ACGT+TGCA has no valid nested inter-strand stack under
        # concatenated ordering, so energy is >= 0.
        assert r.energy >= 0.0

    def test_homodimer_sigma_correction(self):
        """Homodimer energy must include the cyclic-symmetry σ correction."""
        seq = "AAAA"
        r = ThermoEngine(backend="native", celsius=37.0).mfe(seq, seq)
        sigma = cyclic_symmetry([seq, seq])
        assert sigma == 2
        # Reference: naive single-strand fold of the concatenated sequence.
        _, naive_energy, _ = fold_mfe(seq + seq)
        T = 37.0 + 273.15
        expected = naive_energy + R * T * math.log(sigma)
        # Tolerance must be tighter than the ~0.41 kcal/mol σ shift so the
        # uncorrected current implementation fails this assertion.
        assert r.energy == pytest.approx(expected, abs=0.2)


class TestNativeViennaParity:
    """Native MFE/pfunc should agree with ViennaRNA when available.

    ViennaRNA is a *cross-check* backend. The native (SantaLucia-lineage) and
    ViennaRNA (Mathews-2004 DNA) parameter sets differ by a documented mean
    ~0.9 kcal/mol ΔΔG, so parity is asserted as (a) identical base-pairing
    topology on *stable* duplexes and (b) energy agreement within ~1 kcal/mol.
    Marginal dimers whose MFE straddles 0 (e.g. ``AAAA/TTTT``, ``ACGT/TGCA``)
    are deliberately excluded — there the two lineages legitimately disagree on
    whether binding occurs at all, so they test parameter noise, not parity.
    """

    @pytest.fixture
    def vienna_engine(self):
        pytest.importorskip("RNA")
        return ThermoEngine(backend="vienna")

    @pytest.fixture
    def native_engine(self):
        return ThermoEngine(backend="native")

    # Stable, fully-pairing duplexes where both engines agree on topology.
    STABLE_DIMERS = [
        ("GCGC", "GCGC"),
        ("GGGG", "CCCC"),
        ("GCGCGC", "GCGCGC"),
        ("ACGTACGT", "ACGTACGT"),
    ]

    @pytest.mark.parametrize("a,b", STABLE_DIMERS)
    def test_stable_dimer_energy_and_pairs(self, native_engine, vienna_engine, a, b):
        native = native_engine.mfe(a, b)
        vienna = vienna_engine.mfe(a, b)
        # Identical base-pairing topology on a stable duplex.
        assert set(native.base_pairs) == set(vienna.base_pairs)
        # Energy within the documented ~0.9 kcal/mol ΔΔG envelope.
        assert native.energy == pytest.approx(vienna.energy, abs=1.0)

    def test_pfunc_pair_probs_are_well_formed(self, native_engine, vienna_engine):
        """The fixed pf_fold path must return a real bpp matrix (not the old
        ``mean_bp_distance``/``get_pr`` garbage) and a true ensemble ΔG."""
        seq = "GGGAAACCCAAAGGGAAACCC"
        vienna = vienna_engine.pfunc(seq)
        probs = vienna.pair_probs
        n = len(seq)
        assert probs.shape == (n, n)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        assert (probs == probs.T).all()  # symmetric
        # The strong terminal stem must carry high pairing probability.
        assert probs.max() > 0.5
        # Ensemble ΔG is favourable and at least as stable as the single best
        # (MFE) structure — the defining ensemble invariant ΔG_ens ≤ ΔG_mfe.
        assert vienna.free_energy < 0.0
        assert vienna.free_energy <= vienna_engine.mfe(seq).energy + 1e-6
        # Native ensemble obeys the same invariant (sanity on the fixed path).
        native = native_engine.pfunc(seq)
        assert native.free_energy <= native_engine.mfe(seq).energy + 1e-6


class TestEdgeCases:
    """Boundary conditions for multi-strand MFE input."""

    def test_empty_strand_list_raises(self):
        with pytest.raises(ValueError):
            ThermoEngine(backend="native").mfe()

    def test_mismatched_alphabet_raises_or_normalizes(self):
        # DNA/RNA mix must be rejected or handled deterministically.
        with pytest.raises(ValueError):
            ThermoEngine(backend="native", material="dna").mfe("ACGT", "ACGU")

    def test_ampersand_in_single_string_denotes_break(self):
        # "GCGC&TGCG" is parsed as two strands; inter-strand pairs ARE
        # allowed per the multi-strand MFE spec, so the structure may
        # contain pairs that span the break.
        r = ThermoEngine(backend="native").mfe("GCGC&TGCG")
        assert "&" in r.structure
        assert any(
            i < r.structure.index("&") <= j for i, j in r.base_pairs
        ), "inter-strand pairs should be allowed across the break"
