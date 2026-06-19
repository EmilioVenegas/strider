"""Order-invariance of multi-strand MFE folding (Dirks et al. 2007).

The linear nick-aware Zuker DP only represents structures that are non-crossing
for one strand concatenation, so the predicted MFE of a complex must not depend
on how the caller orders the strands.  ``engine.mfe`` now folds the distinct
arrangements and returns the global minimum over a connected structure, so the
result is invariant to any permutation of the input strands.

See :mod:`strider.structure.complex_fold`.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from strider.equilibrium import R, cyclic_symmetry
from strider.structure.complex_fold import (
    distinct_orders,
    fold_complex,
    is_connected,
)
from strider.thermo.engine import ThermoEngine


def _rc(s: str) -> str:
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(comp[x] for x in reversed(s))


# Three domains; closed three-strand triangle: strand i pairs its 3' half to the
# next strand's 5' half, cyclically — the canonical order-dependent case.
_A, _B, _C = "GGAATTCCGTAC", "ACGTGGCATTAC", "TGCATGCAAGCT"
RING3 = [_rc(_A) + _B, _rc(_B) + _C, _rc(_C) + _A]

# A heterodimer and a homodimer.
DIMER = ["GGGAAACCCAAAGGG", "CCCTTTGGGTTTCCC"]


class TestEngineOrderInvariance:
    @pytest.fixture
    def eng(self):
        return ThermoEngine(backend="native", celsius=37.0)

    @pytest.mark.parametrize("strands", [DIMER, RING3], ids=["dimer", "ring3"])
    def test_mfe_invariant_under_permutation(self, eng, strands):
        energies = [eng.mfe(*p).energy for p in itertools.permutations(strands)]
        assert max(energies) - min(energies) < 1e-9

    def test_mfe_invariant_under_random_shuffle(self, eng):
        ref = eng.mfe(*RING3).energy
        rng = random.Random(0)
        for _ in range(8):
            p = RING3[:]
            rng.shuffle(p)
            assert eng.mfe(*p).energy == pytest.approx(ref, abs=1e-9)

    def test_homotrimer_includes_sigma_and_assoc(self, eng):
        """A homotrimer is order-invariant and carries the σ=3 term plus the
        (L−1)·ΔG_assoc association penalty (and any coaxial-junction term)."""
        from strider.thermo.parameters_dna import JOIN_PENALTY
        s = RING3[0]
        assert cyclic_symmetry([s, s, s]) == 3
        from strider.structure.complex_fold import fold_complex as _fc
        cf = _fc([s, s, s], 37.0, "dna", eng.sodium, eng.magnesium)
        coax = eng._coaxial_correction(
            cf.pairs, [len(s)] * 3, "".join(s for _ in range(3)),
        )
        expected = (cf.energy + R * (37.0 + 273.15) * math.log(3)
                    + 2 * JOIN_PENALTY + coax)
        assert eng.mfe(s, s, s).energy == pytest.approx(expected, abs=1e-9)

    def test_no_worse_than_any_single_order(self, eng):
        """The order-invariant MFE is no worse than any single concatenation
        (accounting for the σ and association corrections the engine adds)."""
        from strider.structure.mfe import fold_mfe
        from strider.thermo.parameters_dna import JOIN_PENALTY
        sigma_shift = R * (37.0 + 273.15) * math.log(cyclic_symmetry(RING3))
        assoc = (len(RING3) - 1) * JOIN_PENALTY  # coaxial term is ≤ 0
        invariant = eng.mfe(*RING3).energy
        for p in itertools.permutations(RING3):
            single = fold_mfe("&".join(p), 37.0, "dna", eng.sodium, eng.magnesium)[1]
            assert invariant <= single + sigma_shift + assoc + 1e-9


class TestSuboptOrderInvariance:
    @pytest.fixture
    def eng(self):
        return ThermoEngine(backend="native", celsius=37.0)

    def test_subopt_top_is_order_invariant_structural_mfe(self, eng):
        """subopt[0] is the order-invariant *structural* MFE (no σ/assoc/coaxial),
        i.e. the raw fold_complex energy — was order-broken before the search."""
        from strider.structure.complex_fold import fold_complex
        raw = fold_complex(RING3, 37.0, "dna", eng.sodium, eng.magnesium).energy
        top = eng.subopt(*RING3, gap=1.5)[0][1]
        assert top == pytest.approx(raw, abs=1e-6)

    def test_subopt_energy_set_invariant(self, eng):
        ref = None
        for p in itertools.permutations(RING3):
            energies = tuple(round(e, 4) for _, e, _ in eng.subopt(*p, gap=1.5, max_structures=50))
            if ref is None:
                ref = energies
            assert energies == ref

    def test_subopt_sorted_and_within_gap(self, eng):
        gap = 1.0
        out = eng.subopt(*RING3, gap=gap)
        energies = [e for _, e, _ in out]
        assert energies == sorted(energies)
        assert all(e <= energies[0] + gap + 1e-6 for e in energies)

    def test_subopt_top_is_structural_mfe(self, eng):
        """subopt energies are structural (σ-free): subopt[0] == raw fold_mfe of
        the winning order == engine.mfe minus its rotational-symmetry term."""
        from strider.structure.complex_fold import fold_complex
        s = RING3[0]
        raw = fold_complex([s, s], 37.0, "dna", eng.sodium, eng.magnesium).energy
        assert eng.subopt(s, s, gap=1.0)[0][1] == pytest.approx(raw, abs=1e-6)

    def test_subopt_no_duplicate_structures(self, eng):
        out = eng.subopt(*RING3, gap=2.0)
        dbs = [db for db, _, _ in out]
        assert len(dbs) == len(set(dbs))


class TestFoldComplexUnit:
    def test_distinct_orders_dedup_homomer(self):
        # Two identical strands → only one distinct concatenation.
        assert len(distinct_orders(["AAAA", "AAAA"])) == 1
        # Three distinct strands → 6 permutations, all distinct.
        assert len(distinct_orders(["A", "C", "G"])) == 6

    def test_identity_order_wins_ties(self):
        # A symmetric duplex folds identically in both cuts → input order kept.
        cf = fold_complex(["GGGGCCCC", "GGGGCCCC"], 37.0, "dna", 1.0, 0.0)
        assert cf.order == (0, 1)

    def test_connectivity_helper(self):
        # Strand lengths 4,4,4; a pair bridging strand 0–1 and 1–2 connects all.
        assert is_connected([(0, 4), (8, 5)], [4, 4, 4])
        # Only strand 0–1 bridged; strand 2 dangles → disconnected.
        assert not is_connected([(0, 4)], [4, 4, 4])

    def test_result_connects_all_strands(self):
        cf = fold_complex(RING3, 37.0, "dna", 1.0, 0.0)
        ordered_lens = [len(RING3[i]) for i in cf.order]
        assert cf.connected
        assert is_connected(cf.pairs, ordered_lens)

    def test_non_binding_strands_fall_back(self):
        # Strands that cannot all bind → connected=False, still returns a fold.
        cf = fold_complex(["AAAA", "AAAA", "GGGG"], 37.0, "dna", 1.0, 0.0)
        assert cf.connected is False
        assert cf.structure  # a structure (likely all-unpaired) is still returned

    def test_max_strands_guard(self):
        with pytest.raises(ValueError, match="exceeds max_strands"):
            fold_complex(["A"] * 3, max_strands=2)
