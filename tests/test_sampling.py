"""Boltzmann sampling and subopt enumeration tests."""
from collections import Counter

import pytest

from strider import sample_structures, subopt_structures
from strider.thermo.engine import ThermoEngine


@pytest.fixture
def engine():
    return ThermoEngine(material="dna", celsius=37.0,
                        sodium=0.137, magnesium=0.01)


class TestSampling:
    def test_returns_requested_count(self):
        out = sample_structures("GCGCAAAAGCGC", n_samples=25, seed=0)
        assert len(out) == 25

    def test_returns_dot_bracket_and_pairs(self):
        out = sample_structures("GCGCAAAAGCGC", n_samples=5, seed=0)
        for db, pairs in out:
            assert isinstance(db, str)
            assert len(db) == 12
            assert all(0 <= i < j < 12 for i, j in pairs)

    def test_mfe_dominates_stable_hairpin(self):
        # A strong GC hairpin: MFE structure should dominate the sample.
        samples = sample_structures("GCGCGCAAAAGCGCGC", n_samples=100, seed=42)
        counts = Counter(db for db, _ in samples)
        assert counts.most_common(1)[0][0] == "((((((....))))))"
        assert counts.most_common(1)[0][1] >= 70   # > 70 %

    def test_seed_reproducibility(self):
        out1 = sample_structures("GCATGCATGC", n_samples=10, seed=7)
        out2 = sample_structures("GCATGCATGC", n_samples=10, seed=7)
        assert out1 == out2

    def test_engine_sample_wrapper(self, engine):
        out = engine.sample("GCGCAAAAGCGC", n_samples=10, seed=1)
        assert len(out) == 10


class TestSubopt:
    def test_includes_mfe(self):
        out = subopt_structures("GCGCAAAAGCGC", gap=2.0, max_structures=20)
        assert len(out) > 0
        # First result should match the MFE structure
        mfe_db, mfe_e, _ = out[0]
        assert mfe_db == "((((....))))"
        assert mfe_e < -1.0

    def test_sorted_by_energy(self):
        out = subopt_structures("GCGCAAAAGCGC", gap=3.0)
        energies = [e for _, e, _ in out]
        assert energies == sorted(energies)

    def test_all_within_gap(self):
        gap = 1.5
        out = subopt_structures("GCGCAAAAGCGC", gap=gap)
        mfe = out[0][1]
        for _, e, _ in out:
            assert e <= mfe + gap + 1e-6

    def test_max_structures_respected(self):
        out = subopt_structures("GCATGCATGCAT", gap=10.0, max_structures=5)
        assert len(out) <= 5

    def test_no_duplicates(self):
        out = subopt_structures("GCGCAAAAGCGC", gap=3.0)
        dbs = [db for db, _, _ in out]
        assert len(dbs) == len(set(dbs))

    def test_engine_subopt_wrapper(self, engine):
        out = engine.subopt("GCGCAAAAGCGC", gap=2.0, max_structures=10)
        assert any(db == "((((....))))" for db, _, _ in out)

    def test_subopt_matches_mfe_energy(self, engine):
        """Suboptimal enumeration must return the MFE with the same energy."""
        seq = "GCGCAAAAGCGC"
        mfe_result = engine.mfe(seq)
        out = engine.subopt(seq, gap=2.0, max_structures=10)
        mfe_subopt = next((e for db, e, _ in out if db == mfe_result.structure), None)
        assert mfe_subopt is not None, "MFE structure missing from subopt output"
        assert mfe_subopt == pytest.approx(float(mfe_result.energy), abs=1e-6)

    def test_subopt_respects_salt(self, engine):
        """Suboptimal energies must shift when Mg2+ changes."""
        seq = "GCGCAAAAGCGC"
        low_salt = ThermoEngine(material="dna", celsius=37.0,
                                sodium=0.05, magnesium=0.0)
        high_mg = ThermoEngine(material="dna", celsius=37.0,
                               sodium=0.05, magnesium=0.01)
        out_low = low_salt.subopt(seq, gap=2.0, max_structures=5)
        out_high = high_mg.subopt(seq, gap=2.0, max_structures=5)
        assert out_low[0][1] != pytest.approx(out_high[0][1], abs=1e-3)
