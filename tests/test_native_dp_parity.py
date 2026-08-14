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
    return "".join(rng.choices("ACGT", k=rng.randint(4, 40)))


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


def test_fallback_guards():
    """RNA material must keep using the Python path."""
    a, b = "GCGCGCGC", "GCGCGCGC"
    out = dt_mod._dimer_mfe_candidates(a, b, material="rna")
    py = dt_mod._dimer_mfe_candidates_py(a, b, material="rna")
    assert len(out) == len(py)
    for (e1, p1), (e2, p2) in zip(out, py):
        assert math.isclose(float(e1), float(e2), rel_tol=0.0, abs_tol=TOL)
        assert p1 == p2
