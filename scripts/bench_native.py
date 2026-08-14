#!/usr/bin/env python3
"""Benchmark: strider._native (Rust) vs pure-Python fallback vs primer3-py.

Self-orchestrating: the pure-Python numbers are measured in a subprocess that
blocks ``strider._native`` before importing strider (module rebinding happens
at import time), while this process measures the native and primer3 rows.
Corpus: 3000 deterministic pseudo-random oligos (len 15-35), fixed salt
conditions mirroring a typical qPCR primer setup.

    python scripts/bench_native.py
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import timeit

N_SEQS = 3000


def corpus():
    rng = random.Random(42)
    return ["".join(rng.choices("ACGT", k=rng.randint(15, 35))) for _ in range(N_SEQS)]


WORKER_TIMING = r"""
import importlib.abc, importlib.machinery, importlib.metadata, json, random, sys, timeit


class _Blocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "strider._native":
            return importlib.machinery.ModuleSpec(fullname, self)
        return None
    def create_module(self, spec):
        raise ImportError("strider._native blocked for python-fallback timing")
    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Blocker())
try:
    primer3 = __import__("primer3")
except ImportError:
    primer3 = None
from strider.thermo import nn_dna, salt
assert nn_dna._n is None, "timing oracle tainted: native resolved anyway"

rng = random.Random(42)
seqs = ["".join(rng.choices("ACGT", k=rng.randint(15, 35))) for _ in range(%d)]

def tmil(fn, reps=7):
    return min(timeit.repeat(fn, repeat=reps, number=1))

out = {}
out["py_mt"] = tmil(lambda: [nn_dna.melting_temperature(
    s, strand_conc_M=400e-9, sodium_M=0.1958, magnesium_M=0.0) for s in seqs])
out["py_bp"] = tmil(lambda: [salt.dg_per_bp_salt(0.05, 0.0022) for _ in range(20000)])
if primer3 is not None:
    KW = dict(mv_conc=50.0, dv_conc=10.0, dntp_conc=0.8, dna_conc=400.0,
              tm_method="santalucia", salt_corrections_method="santalucia")
    out["primer3"] = tmil(lambda: [primer3.calc_tm(s, **KW) for s in seqs])
json.dump(out, sys.stdout)
""" % N_SEQS


def main() -> None:
    seqs = corpus()

    from strider.thermo import nn_dna, salt
    if nn_dna._n is None or salt._n is None:
        sys.exit("strider._native is not available — build it first "
                 "(scripts/build_native.sh or pip install .)")

    def tmil(fn, reps=7):
        return min(timeit.repeat(fn, repeat=reps, number=1))

    native_mt = tmil(lambda: [nn_dna.melting_temperature(
        s, strand_conc_M=400e-9, sodium_M=0.1958, magnesium_M=0.0) for s in seqs])
    native_bp = tmil(lambda: [salt.dg_per_bp_salt(0.05, 0.0022) for _ in range(20000)])

    proc = subprocess.run([sys.executable, "-c", WORKER_TIMING],
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        sys.exit(f"fallback timing worker failed:\n{proc.stderr[-2000:]}")
    w = json.loads(proc.stdout)

    rows = [
        ("melting_temperature ×3000", w["py_mt"], native_mt),
        ("dg_per_bp_salt  ×20000", w["py_bp"], native_bp),
    ]
    print(f"{'call':34s} {'python [ms]':>12} {'native [ms]':>12} {'speedup':>8}")
    for label, py_ms, rs_ms in rows:
        print(f"{label:34s} {py_ms*1000:12.2f} {rs_ms*1000:12.2f} {py_ms/rs_ms:7.1f}×")
    if "primer3" in w:
        print(f"{'primer3 calc_tm (C extension) ×3000':34s} {'':12s} "
              f"{w['primer3']*1000:12.2f} {w['primer3']/native_mt:7.1f}× vs native")


if __name__ == "__main__":
    main()
