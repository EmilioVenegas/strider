"""Bit-parity fuzz test: strider._native vs its pure-Python fallbacks.

Every function the Rust accelerator replaces is re-checked here against the
Python implementation it masks.  The Python oracle is computed in a *clean
subprocess* whose first act is to meta-path-block ``strider._native`` before
any strider import (this test suite's own conftest imports strider eagerly, so
in-process blocking is unreliable).  The oracle values are returned as JSON;
this process — with native loaded — recomputes every value and both sides are
compared to ≤1e-9 across a 10k-sequence fuzz sweep covering all Owczarzy
branches (Na-only, mixed, Mg-only), deterministic boundary ratios, odd bases,
lowercase, U-containing and self-complementary strands of every length 1..60.

Failure to import ``strider._native`` skips the whole module: on machines
without a rust toolchain the fallback path is what ships, and the repo test
suite must still pass.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest

native = pytest.importorskip("strider._native", reason="rust extension not built")

TOL = 1e-9

# ── oracle worker: runs in a clean interpreter with _native blocked ──────────

WORKER = r"""
import importlib.abc, importlib.machinery, json, random, sys


class _Blocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "strider._native":
            return importlib.machinery.ModuleSpec(fullname, self)
        return None
    def create_module(self, spec):
        raise ImportError("strider._native blocked for parity oracle")
    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Blocker())
from strider.thermo import nn_dna, salt
assert nn_dna._n is None, "oracle tainted: nn_dna resolved native anyway"
assert salt._n is None, "oracle tainted: salt resolved native anyway"

rng = random.Random(0x5EED)


def rand_seq():
    n = rng.randint(1, 60)
    alphabet = "ACGT" if rng.random() > 0.2 else "ACGTUNRYSWKMBDHV"
    s = "".join(rng.choice(alphabet) for _ in range(n))
    if rng.random() < 0.3:
        s = s.lower() if rng.random() < 0.5 else "".join(
            c.lower() if rng.random() < 0.5 else c for c in s
        )
    if rng.random() < 0.1 and "U" not in alphabet.upper():
        s = s[: max(n // 3, 1)] + "U" + s[max(n // 3, 2):]
    return s


BASES = [rand_seq() for _ in range(10_000)] + [
    "GCGCGCGCGCGC", "ATATATATATAT", "GGGGCCCCGGGG",
    "ATGC" * 15, "A", "T", "G", "C", "ACGT",
    "GC" * 30, "atgcgatcgatc", "AAAUUUCCCGGG", "NNNNNN",
]

SALT_REGIMES = [
    (1.0, 0.0), (1e-3, 1e-6), (1.0, 1e-4), (1e-4, 1e-2), (1e-3, 5e-2),
    (0.05, 0.0022), (0.137, 0.004), (5e-2, 8e-2), (1e-2, 6e-2), (1e-1, 2.5e-2),
]
CONC = [1e-9, 2.5e-7, 4e-7, 1e-4]
DNTPS = [0.0, 8e-4, 2e-3]
TEMPS = [10.0, 25.0, 37.0, 60.0, 90.0]

out = {"seqs": BASES}
out["rc"] = [nn_dna.reverse_complement(s) for s in BASES]
out["sc"] = [nn_dna.is_self_complementary(s) for s in BASES]
out["dh_ds"] = [list(nn_dna.duplex_dh_ds(s)) for s in BASES]
out["mt"] = [
    nn_dna.melting_temperature(s, strand_conc_M=CONC[i % 4],
                               sodium_M=SALT_REGIMES[i % 10][0],
                               magnesium_M=SALT_REGIMES[i % 10][1])
    for i, s in enumerate(BASES[:3000])
]
out["dtm"] = [
    nn_dna.duplex_tm(s, sodium_M=SALT_REGIMES[i % 10][0],
                     magnesium_M=SALT_REGIMES[i % 10][1],
                     dntp_M=DNTPS[i % 3], oligo_conc_M=CONC[i % 4])
    for i, s in enumerate(BASES[:3000])
]
out["dg"] = [
    nn_dna.duplex_dg(s, celsius=TEMPS[i % 5],
                     sodium_M=SALT_REGIMES[i % 10][0],
                     magnesium_M=SALT_REGIMES[i % 10][1])
    for i, s in enumerate(BASES[:3000])
]
out["owc"] = [
    salt.owczarzy_tm_correction(s, SALT_REGIMES[i % 10][0], SALT_REGIMES[i % 10][1])
    for i, s in enumerate(BASES[:5000])
]
out["na_dg"] = [
    salt.na_correction_dg(s, SALT_REGIMES[i % 10][0], [25.0, 37.0, 55.0][i % 3])
    for i, s in enumerate(BASES[:2000])
]
out["duplex_salt_dg"] = [
    salt.duplex_salt_dg(s, SALT_REGIMES[i % 10][0], SALT_REGIMES[i % 10][1],
                        [25.0, 37.0][i % 2])
    for i, s in enumerate(BASES[:2000])
]

# grid-only scalars + error semantics
out["bp"] = [
    salt.dg_per_bp_salt(na, mg, t, mat)
    for na, mg in SALT_REGIMES + [(0.0, 1e-2)]
    for t in (25.0, 37.0, 50.0)
    for mat in ("dna", "rna", "DNA", "RNA")
]
out["tc"] = [
    salt.tan_chen_helix_dg(n, na, mg, mat)
    for n in range(6, 40)
    for na, mg in SALT_REGIMES
    for mat in ("dna", "rna")
]
out["tc_err"] = []
for n in (1, 5):
    try:
        salt.tan_chen_helix_dg(n, 0.1, 0.0)
        out["tc_err"].append(None)
    except ValueError:
        out["tc_err"].append("ValueError")
try:
    salt.tan_chen_helix_dg(8, 0.1, 0.0, "dna-rna")
    out["tc_err"].append(None)
except ValueError:
    out["tc_err"].append("ValueError")

json.dump(out, sys.stdout)
"""


def _load_oracle():
    proc = subprocess.run(
        [sys.executable, "-c", WORKER],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        pytest.fail(f"oracle worker failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


ORACLE = _load_oracle()
SEQ = ORACLE["seqs"]
SALT_REGIMES = [
    (1.0, 0.0), (1e-3, 1e-6), (1.0, 1e-4), (1e-4, 1e-2), (1e-3, 5e-2),
    (0.05, 0.0022), (0.137, 0.004), (5e-2, 8e-2), (1e-2, 6e-2), (1e-1, 2.5e-2),
]
CONC = [1e-9, 2.5e-7, 4e-7, 1e-4]
DNTPS = [0.0, 8e-4, 2e-3]
TEMPS = [10.0, 25.0, 37.0, 60.0, 90.0]


def _same(a, b):
    # math.isclose agrees inf==inf-same-sign; NaN is never close (treated as failure)
    return math.isclose(a, b, rel_tol=0.0, abs_tol=TOL)


def test_reverse_complement():
    bad = [s for s, want in zip(SEQ, ORACLE["rc"]) if native.reverse_complement(s) != want]
    assert not bad, f"{len(bad)} mismatches, first: {bad[:3]}"


def test_is_self_complementary():
    bad = [s for s, want in zip(SEQ, ORACLE["sc"]) if native.is_self_complementary(s) != want]
    assert not bad, f"{len(bad)} mismatches, first: {bad[:3]}"


def test_duplex_dh_ds():
    bad = [
        (s, want, got)
        for s, want in zip(SEQ, ORACLE["dh_ds"])
        for got in [native.duplex_dh_ds(s)]
        if not (_same(want[0], got[0]) and _same(want[1], got[1]))
    ]
    assert not bad, f"{len(bad)} mismatches, first: {bad[:2]}"


def test_melting_temperature():
    bad = []
    for i, s in enumerate(SEQ[:3000]):
        ct, (na, mg) = CONC[i % 4], SALT_REGIMES[i % 10]
        got = native.melting_temperature(s, strand_conc_M=ct, sodium_M=na, magnesium_M=mg)
        if not _same(ORACLE["mt"][i], got):
            bad.append((s, ORACLE["mt"][i], got))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:3]}"


def test_duplex_tm():
    bad = []
    for i, s in enumerate(SEQ[:3000]):
        na, mg = SALT_REGIMES[i % 10]
        got = native.duplex_tm(s, sodium_M=na, magnesium_M=mg,
                               dntp_M=DNTPS[i % 3], oligo_conc_M=CONC[i % 4])
        if not _same(ORACLE["dtm"][i], got):
            bad.append((s, ORACLE["dtm"][i], got))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:3]}"


def test_duplex_dg():
    bad = []
    for i, s in enumerate(SEQ[:3000]):
        na, mg = SALT_REGIMES[i % 10]
        got = native.duplex_dg(s, celsius=TEMPS[i % 5], sodium_M=na, magnesium_M=mg)
        if not _same(ORACLE["dg"][i], got):
            bad.append((s, ORACLE["dg"][i], got))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:3]}"


def test_owczarzy_tm_correction():
    bad = []
    for i, s in enumerate(SEQ[:5000]):
        na, mg = SALT_REGIMES[i % 10]
        got = native.owczarzy_tm_correction(s, na, mg)
        if not _same(ORACLE["owc"][i], got):
            bad.append((s, ORACLE["owc"][i], got))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:3]}"


def test_na_correction_dg():
    bad = []
    for i, s in enumerate(SEQ[:2000]):
        got = native.na_correction_dg(s, SALT_REGIMES[i % 10][0], [25.0, 37.0, 55.0][i % 3])
        if not _same(ORACLE["na_dg"][i], got):
            bad.append((s, ORACLE["na_dg"][i], got))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:3]}"


def test_duplex_salt_dg():
    bad = []
    for i, s in enumerate(SEQ[:2000]):
        na, mg = SALT_REGIMES[i % 10]
        got = native.duplex_salt_dg(s, na, mg, [25.0, 37.0][i % 2])
        if not _same(ORACLE["duplex_salt_dg"][i], got):
            bad.append((s, ORACLE["duplex_salt_dg"][i], got))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:3]}"


def test_dg_per_bp_salt_grid():
    k = 0
    for na, mg in SALT_REGIMES + [(0.0, 1e-2)]:
        for t in (25.0, 37.0, 50.0):
            for mat in ("dna", "rna", "DNA", "RNA"):
                got = native.dg_per_bp_salt(na, mg, t, mat)
                assert _same(ORACLE["bp"][k], got), (na, mg, t, mat, ORACLE["bp"][k], got)
                k += 1


def test_tan_chen_helix_dg_grid():
    k = 0
    for n in range(6, 40):
        for na, mg in SALT_REGIMES:
            for mat in ("dna", "rna"):
                got = native.tan_chen_helix_dg(n, na, mg, mat)
                assert _same(ORACLE["tc"][k], got), (n, na, mg, mat, ORACLE["tc"][k], got)
                k += 1


def test_tan_chen_error_parity():
    for i, n in enumerate((1, 5)):
        assert ORACLE["tc_err"][i] == "ValueError"
        import pytest as _pt
        with _pt.raises(ValueError):
            native.tan_chen_helix_dg(n, 0.1, 0.0)
    assert ORACLE["tc_err"][2] == "ValueError"
    with pytest.raises(ValueError):
        native.tan_chen_helix_dg(8, 0.1, 0.0, "dna-rna")
