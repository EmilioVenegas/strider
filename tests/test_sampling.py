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

    @pytest.mark.parametrize("seq", [
        "GGGAAACCC",
        "GCGCAATTGCGCTTTTGCGCAATTGCGC",
        "GGGAAACCCAAAGGGAAACCCAAAGGGAAACCC",   # multiloop
    ])
    def test_subopt_top_equals_fold_mfe(self, seq):
        # The lowest-energy subopt structure must match fold_mfe exactly
        # (same energy and dot-bracket): subopt shares fold_mfe's DP matrices.
        from strider.structure.mfe import fold_mfe
        db, e, _ = subopt_structures(seq, gap=3.0, max_structures=500)[0]
        m_db, m_e, _ = fold_mfe(seq, 37.0, "dna", 1.0, 0.0)
        assert db == m_db
        assert abs(e - m_e) < 1e-6

    def test_subopt_complete_vs_bruteforce(self):
        # With a huge gap, subopt must enumerate every valid (non-crossing,
        # min-loop-3) structure — no missing, no spurious extras.
        from strider.structure.mfe import _wc_pairs
        seq = "GCGCAAAAGC"
        n, wc = len(seq), _wc_pairs("dna")
        structs = set()

        def rec(pairs, used):
            structs.add(frozenset(pairs))
            for a in range(n):
                if a in used:
                    continue
                for b in range(a + 4, n):
                    if b in used or frozenset((seq[a], seq[b])) not in wc:
                        continue
                    if any((a < c < b < d) or (c < a < d < b)
                           or (a <= c <= b) != (a <= d <= b)
                           for c, d in pairs):
                        continue
                    rec(pairs | {(a, b)}, used | {a, b})

        rec(frozenset(), frozenset())
        got = {frozenset(pl) for _, _, pl in
               subopt_structures(seq, gap=1e6, max_structures=100000)}
        assert got == structs


class TestSuboptMultiStrand:
    def test_dimer_includes_mfe(self, engine):
        from strider.structure.mfe import fold_mfe
        out = engine.subopt("GCGCAATTGCGC", "GCGCAATTGCGC", gap=3.0)
        m_db, m_e, _ = fold_mfe("GCGCAATTGCGC&GCGCAATTGCGC", 37.0, "dna",
                                0.137, 0.01)
        assert any(db == m_db for db, _, _ in out)
        assert abs(out[0][1] - m_e) < 1e-6

    def test_dimer_separator_in_dot_bracket(self, engine):
        out = engine.subopt("AAAACCCC", "GGGGTTTT", gap=3.0)
        for db, _, _ in out:
            assert "&" in db
            assert len(db) == len("AAAACCCC") + 1 + len("GGGGTTTT")

    def test_joined_string_equals_varargs(self, engine):
        a = engine.subopt("GCGCAATTGCGC", "GCGCAATTGCGC", gap=2.0)
        b = engine.subopt("GCGCAATTGCGC&GCGCAATTGCGC", gap=2.0)
        assert a == b


class TestSuboptSalt:
    def test_magnesium_shifts_energies(self):
        # Mg2+ must lower (stabilize) subopt energies — regression for the
        # bug where salt was ignored on the subopt path entirely.
        seq = "GGGAAACCCAAAGGGAAACCC"
        e0 = subopt_structures(seq, gap=2.0, sodium_M=0.137, magnesium_M=0.0)[0][1]
        e1 = subopt_structures(seq, gap=2.0, sodium_M=0.137, magnesium_M=0.05)[0][1]
        assert e1 < e0 - 1e-3

    def test_reference_salt_is_neutral(self):
        # 1 M Na+, 0 Mg2+ is the reference: correction must be exactly zero.
        from strider.structure.mfe import fold_mfe
        seq = "GCGCAATTGCGCTTTTGCGCAATTGCGC"
        sub0 = subopt_structures(seq, gap=2.0)[0][1]   # defaults = reference
        _, m_e, _ = fold_mfe(seq, 37.0, "dna", 1.0, 0.0)
        assert abs(sub0 - m_e) < 1e-9
