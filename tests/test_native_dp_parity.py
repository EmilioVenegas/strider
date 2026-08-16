"""Parity fuzz for the Rust DP: strider._native.dimer_mfe_candidates vs the
pure-Python strider.thermo.dimer_thermo._dimer_mfe_candidates.

Two layers, same style as tests/test_native_parity.py:

* DP-level: full candidate lists across randomized (seq1, seq2) pairs —
  energies within 1e-9, pair lists exactly equal.
* Integration-level: dimer_thermo + dimer_thermo_subopt end-to-end, with the
  module pinned to the Python fallback oracle (``_dimer_mfe_candidates_py``).
  Both the success path and the ``ValueError`` (<2 inter-strand pairs)
  behaviors must agree.

Skipped when ``strider._native`` is absent (pure-Python environments).
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager

import pytest

native = pytest.importorskip("strider._native", reason="rust extension not built")
if not hasattr(native, "dimer_mfe_candidates"):
    pytest.skip("native build predates the DP port", allow_module_level=True)

import strider.thermo.dimer_thermo as dt_mod

TOL = 1e-9
MV, MG, OC = 0.05, 0.0092, 0.25e-6

rng = random.Random(2024)


def rand_seq():
    """Random ACGT sequence; ~15% of calls sprinkle degenerate IUPAC bases.

    Degenerate bases (N, R, Y, S, W, K, M, B, D, H, V) must trigger the
    alphabet guard in the wrapper (fall back to Python), since the Rust
    DP would otherwise pack them as T and hit wrong table entries.
    """
    n = rng.randint(4, 40)
    s = "".join(rng.choices("ACGT", k=n))
    if rng.random() < 0.15:
        n_degenerate = rng.randint(1, 3)
        degen = "N R Y S W K M B D H V".split()
        for _ in range(n_degenerate):
            pos = rng.randint(0, len(s) - 1)
            s = s[:pos] + rng.choice(degen) + s[pos + 1:]
    return s


@contextmanager
def python_dp():
    """Pin _dimer_mfe_candidates to the Python oracle for the block."""
    saved = dt_mod._dimer_mfe_candidates
    dt_mod._dimer_mfe_candidates = dt_mod._dimer_mfe_candidates_py
    try:
        yield
    finally:
        dt_mod._dimer_mfe_candidates = saved


def test_dp_candidates_parity():
    """Full candidate-list parity over 600 random pairs."""
    bad_e = bad_p = 0
    first = None
    for _ in range(600):
        a, b = rand_seq(), rand_seq()
        py = dt_mod._dimer_mfe_candidates_py(a, b, material="dna")
        rs = [
            (float(e), [tuple(map(int, pr)) for pr in pairs])
            for e, pairs in native.dimer_mfe_candidates(a, b)
        ]
        if len(py) != len(rs):
            bad_e += 1
            if first is None:
                first = ("count", a, b, len(py), len(rs))
            continue
        for (e_py, p_py), (e_rs, p_rs) in zip(py, rs):
            if not math.isclose(e_py, e_rs, rel_tol=0.0, abs_tol=TOL):
                bad_e += 1
                if first is None:
                    first = ("energy", a, b, e_py, e_rs)
                break
            if p_py != p_rs:
                bad_p += 1
                if first is None:
                    first = ("pairs", a, b, p_py, p_rs)
                break
    assert bad_e == 0 and bad_p == 0, (
        f"energy mismatches: {bad_e}, pair mismatches: {bad_p}, first: {first}"
    )


def _run_thermo(a, b):
    try:
        r1 = dt_mod.dimer_thermo(a, b, sodium_M=MV, magnesium_M=MG,
                                 strand_conc_M=OC, salt_model="auto")
        r2 = dt_mod.dimer_thermo_subopt(a, b, n=5, sodium_M=MV, magnesium_M=MG,
                                        strand_conc_M=OC, salt_model="auto")
        return (r1.tm_celsius, r1.dG37, r1.structure,
                [(x.tm_celsius, x.dG37, x.structure) for x in r2])
    except ValueError as e:
        return ("ValueError", str(e))


def test_dimer_thermo_end_to_end_parity():
    """Public API parity incl. error behavior over 300 random pairs."""
    bad = 0
    first = None
    for _ in range(300):
        a, b = rand_seq(), rand_seq()
        native_out = _run_thermo(a, b)
        with python_dp():
            python_out = _run_thermo(a, b)
        if native_out != python_out:
            bad += 1
            if first is None:
                first = (a, b, native_out, python_out)
    assert bad == 0, f"{bad}/300 end-to-end mismatches, first: {first}"


def _same_candidates(out, py):
    if len(out) != len(py):
        return False
    return all(
        math.isclose(float(e1), float(e2), rel_tol=0.0, abs_tol=TOL) and p1 == p2
        for (e1, p1), (e2, p2) in zip(out, py)
    )


def test_fallback_guards():
    """Non-native-eligible inputs must use the Python path: RNA material."""
    a, b = "GCGCGCGC", "GCGCGCGC"
    assert _same_candidates(
        dt_mod._dimer_mfe_candidates(a, b, material="rna"),
        dt_mod._dimer_mfe_candidates_py(a, b, material="rna"),
    )


def test_fallback_guard_param_override():
    """Active param_context must force the Python path (native = baked tables)."""
    from strider.thermo._param_context import _param_override

    a, b = "GCGCGCGC", "GCGCGCGC"

    class _EmptyOverride:
        dG = {}  # empty → lookup_table returns module-constant fallback

    token = _param_override.set(_EmptyOverride())
    try:
        assert _same_candidates(
            dt_mod._dimer_mfe_candidates(a, b, material="dna"),
            dt_mod._dimer_mfe_candidates_py(a, b, material="dna"),
        )
    finally:
        _param_override.reset(token)


def test_fallback_guard_custom_engine_params():
    """Engine with custom ParameterSet must force the Python path."""
    a, b = "GCGCGCGC", "GCGCGCGC"

    class _CustomEngine:
        material = "dna"
        celsius = 37.0
        sodium = 0.05
        magnesium = 0.0092
        def _uses_custom_params(self): return True

    assert _same_candidates(
        dt_mod._dimer_mfe_candidates(a, b, engine=_CustomEngine()),
        dt_mod._dimer_mfe_candidates_py(a, b, engine=_CustomEngine()),
    )


def test_fallback_guard_degenerate_alphabet():
    """Sequences with degenerate IUPAC bases must fall back to Python."""
    # 'N' would pack as T and hit a wrong table entry on the Rust side.
    for a, b in [
        ("ACGNACGT", "ACGTACGT"),
        ("ACGTACGN", "ACGTACGT"),
        ("NCGT", "ACGT"),
        ("ACGT", "NCGT"),
        ("ACRNACGT", "ACGTACGT"),
    ]:
        assert _same_candidates(
            dt_mod._dimer_mfe_candidates(a, b, material="dna"),
            dt_mod._dimer_mfe_candidates_py(a, b, material="dna"),
        ), f"degenerate base divergence: {a!r} vs {b!r}"


def test_native_raw_degenerate_matches_python():
    """Direct strider._native call on degenerate sequences matches Python.

    The wrapper guards alphabet, but if anyone calls the raw export directly,
    the CODE_TABLE sentinel (u32::MAX) must make every table lookup miss —
    same as Python's dict.get(key, default) returning the default.
    """
    cases = [
        ("ACGNACGT", "ACGTACGT"),
        ("ACGTACGN", "ACGTACGT"),
        ("NCGT", "ACGT"),
        ("ACGT", "NCGT"),
        ("ACRNACGT", "ACGTACGT"),
        ("ACBNACDT", "ACGT", ),
    ]
    for a, b in cases:
        py = dt_mod._dimer_mfe_candidates_py(a, b, material="dna")
        rs = [
            (float(e), [tuple(map(int, pr)) for pr in pairs])
            for e, pairs in native.dimer_mfe_candidates(a, b)
        ]
        assert _same_candidates(rs, py), (
            f"raw native diverged from Python on degenerate {a!r} vs {b!r}"
        )


def test_tables_dna_regenerated_matches_committed():
    """Regenerate tables_dna.rs from parameters_dna and compare byte-for-byte.

    Orthogonal to the DP fuzz: catches silent drift when parameters_dna.py
    is edited (e.g. new salt/dangle work) but the generated Rust tables were
    not regenerated.  Does not require the native extension — codegen imports
    only strider.thermo.parameters_dna.
    """
    import pathlib
    import subprocess
    import sys
    import tempfile

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    committed = repo_root / "native" / "src" / "tables_dna.rs"
    assert committed.exists(), f"committed tables not found: {committed}"

    with tempfile.NamedTemporaryFile(suffix=".rs", mode="w", delete=False) as f:
        tmp_path = f.name
    try:
        subprocess.run(
            [sys.executable, str(repo_root / "native" / "codegen_tables.py"), tmp_path],
            check=True, capture_output=True, text=True,
        )
        regenerated = pathlib.Path(tmp_path).read_text()
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)

    expected = committed.read_text()
    if regenerated != expected:
        # Find the first differing line for a readable diff hint.
        for i, (a, b) in enumerate(zip(expected.splitlines(), regenerated.splitlines())):
            if a != b:
                pytest.fail(
                    f"tables_dna.rs is stale at line {i + 1}.\n"
                    f"  committed:   {a[:120]}\n"
                    f"  regenerated: {b[:120]}\n"
                    f"Run: python native/codegen_tables.py"
                )
        pytest.fail(
            f"tables_dna.rs length mismatch: committed {len(expected)} vs "
            f"regenerated {len(regenerated)} — run: python native/codegen_tables.py"
        )
