"""
Frozen cross-validation suite: strider native engine vs. NUPACK 4 / ViennaRNA.

Three independent agreement axes on a shared, seeded set of sequences/tubes, eachcemitting a one-line-reproducible JSON receipt so the numbers can be *frozen andccited* rather than re-quoted by hand:

  1. ``pfunc``  — ensemble free energy ΔG from the partition function, plus the
                  pfunc wall-clock speed ratio. "Does the pure-Python McCaskill DP
                  reproduce the reference C kernel's pfunc, and at what speed cost?"
  2. ``pairs``  — base-pair probability *matrix* agreement (MAE / RMSE / max on the
                  off-diagonal P[i<j] entries). A pointwise check that the whole
                  ensemble — not just its scalar ΔG — matches.
  3. ``conc``   — multi-strand *equilibrium concentration* agreement. Builds the
                  same two-strand test tube in both tools and compares the
                  per-complex concentrations from each concentration solver.

Reference tools are optional and detected at import time:

  * **NUPACK 4** (closed-source, install separately) — exercises all three axes.
  * **ViennaRNA** — exercises ``pfunc`` and ``pairs`` (no built-in multi-strand concentration solver), via strider's own ``backend='vienna'`` adapter, i.e. a cross-check of the native DP against the Vienna C kernel through one engine API.

Because NUPACK and ViennaRNA rarely coexist in one interpreter, run the suite once per environment; each run freezes whatever reference is importable::

    # NUPACK env (all three axes)
    PYTHONPATH=strider nupack_env310/bin/python scripts/bench_vs_nupack.py \
        --suite all --material rna --json paper/receipts/xval_vs_nupack_rna.json

    # ViennaRNA env (pfunc + pairs)
    PYTHONPATH=strider .venv/bin/python scripts/bench_vs_nupack.py \
        --suite pfunc pairs --material rna --json paper/receipts/xval_vs_vienna_rna.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from typing import Callable, Iterable, Literal

import numpy as np

from strider.thermo.engine import ThermoEngine
from strider.tube import ComplexSet, SetSpec, Strand, Tube

try:
    from nupack import Model as _NupackModel
    from nupack import Strand as _NupackStrand
    from nupack import Tube as _NupackTube
    from nupack import SetSpec as _NupackSetSpec
    from nupack import pairs as _nupack_pairs
    from nupack import pfunc as _nupack_pfunc
    from nupack import tube_analysis as _nupack_tube_analysis

    _HAS_NUPACK = True
except ImportError:  # pragma: no cover - depends on external install
    _HAS_NUPACK = False

try:
    import RNA as _RNA  # noqa: F401  (presence is all we need)

    _HAS_VIENNA = True
except ImportError:  # pragma: no cover - depends on external install
    _HAS_VIENNA = False


Material = Literal["dna", "rna"]


# ─── helpers ────────────────────────────────────────────────────────────────


def random_sequence(n: int, material: Material, rng: random.Random) -> str:
    alphabet = "ACGU" if material == "rna" else "ACGT"
    return "".join(rng.choice(alphabet) for _ in range(n))


def reverse_complement(seq: str, material: Material) -> str:
    comp = {"A": "U" if material == "rna" else "T", "C": "G", "G": "C",
            "T": "A", "U": "A"}
    return "".join(comp[b] for b in reversed(seq))


def _median_ms(fn: Callable[[], object], reps: int) -> float:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def _error_stats(deltas: list[float]) -> dict:
    """MAE / RMSE / max over a list of (signed or unsigned) residuals."""
    if not deltas:
        return {"n": 0, "mae": None, "rmse": None, "max_abs": None}
    abs_d = [abs(x) for x in deltas]
    return {
        "n": len(deltas),
        "mae": statistics.mean(abs_d),
        "rmse": math.sqrt(statistics.mean(x * x for x in deltas)),
        "max_abs": max(abs_d),
    }


def _nupack_dg(seqs: list[str], model) -> float:
    # nupack.pfunc returns (partition_function, free_energy)
    return float(_nupack_pfunc(seqs, model=model)[1])


# ─── suite 1: pfunc ΔG + speed ──────────────────────────────────────────────


def run_pfunc(
    lengths: Iterable[int],
    n_seqs: int,
    reps: int,
    material: Material,
    celsius: float,
    sodium: float,
    magnesium: float,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    native = ThermoEngine(material=material, celsius=celsius, sodium=sodium, magnesium=magnesium, backend="native")
    nupack_model = (
        _NupackModel(material=material, celsius=celsius, sodium=sodium, magnesium=magnesium)
        if _HAS_NUPACK
        else None
    )
    vienna = (
        ThermoEngine(material=material, celsius=celsius, sodium=sodium, magnesium=magnesium, backend="vienna")
        if _HAS_VIENNA
        else None
    )

    rows = []
    for n in lengths:
        seqs = [random_sequence(n, material, rng) for _ in range(n_seqs)]

        native_ms = []
        nupack_ms = []
        ddg_nupack = []
        ddg_vienna = []
        for s in seqs:
            dg_native = native.pfunc(s).free_energy
            native_ms.append(_median_ms(lambda s=s: native.pfunc(s), reps))

            if _HAS_NUPACK:
                dg_nupack = _nupack_dg([s], nupack_model)
                nupack_ms.append(_median_ms(lambda s=s: _nupack_pfunc([s], model=nupack_model), reps))
                ddg_nupack.append(dg_native - dg_nupack)
            if _HAS_VIENNA:
                ddg_vienna.append(dg_native - vienna.pfunc(s).free_energy)

        row = {
            "length": n,
            "n_seqs": n_seqs,
            "native_ms_per_seq": statistics.median(native_ms),
        }
        if _HAS_NUPACK:
            row["nupack_ms_per_seq"] = statistics.median(nupack_ms)
            row["speed_ratio_native_over_nupack"] = (
                statistics.median(native_ms) / statistics.median(nupack_ms)
            )
            row["ddg_vs_nupack"] = _error_stats(ddg_nupack)
        if _HAS_VIENNA:
            row["ddg_vs_vienna"] = _error_stats(ddg_vienna)
        rows.append(row)

    return {"axis": "pfunc", "rows": rows}


# ─── suite 2: pair-probability matrix agreement ─────────────────────────────


def _offdiag_upper(P: np.ndarray) -> np.ndarray:
    """Flatten the strict upper triangle P[i, j], i < j (the pair probabilities)."""
    n = P.shape[0]
    iu = np.triu_indices(n, k=1)
    return np.asarray(P)[iu]


def run_pairs(
    lengths: Iterable[int],
    n_seqs: int,
    material: Material,
    celsius: float,
    sodium: float,
    magnesium: float,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    native = ThermoEngine(material=material, celsius=celsius, sodium=sodium, magnesium=magnesium, backend="native")
    nupack_model = (
        _NupackModel(material=material, celsius=celsius, sodium=sodium, magnesium=magnesium)
        if _HAS_NUPACK
        else None
    )
    vienna = (
        ThermoEngine(material=material, celsius=celsius, sodium=sodium, magnesium=magnesium, backend="vienna")
        if _HAS_VIENNA
        else None
    )

    rows = []
    for n in lengths:
        seqs = [random_sequence(n, material, rng) for _ in range(n_seqs)]
        d_nupack: list[float] = []
        d_vienna: list[float] = []
        for s in seqs:
            p_native = _offdiag_upper(native.pairs(s))
            if _HAS_NUPACK:
                # NUPACK PairsMatrix.to_array() is n×n with the diagonal holding the
                # unpaired probability; off-diagonal P[i<j] are the pair probs (same
                # convention as strider's off-diagonal).
                p_nu = _offdiag_upper(_nupack_pairs([s], model=nupack_model).to_array())
                d_nupack.extend((p_native - p_nu).tolist())
            if _HAS_VIENNA:
                d_vienna.extend((p_native - _offdiag_upper(vienna.pairs(s))).tolist())

        row = {"length": n, "n_seqs": n_seqs}
        if _HAS_NUPACK:
            row["bpp_vs_nupack"] = _error_stats(d_nupack)
        if _HAS_VIENNA:
            row["bpp_vs_vienna"] = _error_stats(d_vienna)
        rows.append(row)

    return {"axis": "pairs", "rows": rows}


# ─── suite 3: multi-strand equilibrium concentrations ───────────────────────


def _strider_tube_conc(
    sa: str, sb: str, total: float, engine: ThermoEngine
) -> dict[tuple[str, ...], float]:
    a = Strand("a", sa)
    b = Strand("b", sb)
    tube = Tube(
        strand_totals={a: total, b: total},
        complexes=ComplexSet([a, b], SetSpec(max_size=2)),
    )
    res = tube.analyze(engine)
    # canonical names like "a", "a_a", "a_b" → sorted strand-name tuple key.
    return {tuple(sorted(name.split("_"))): c for name, c in res.concentrations.items()}


def _nupack_tube_conc(
    sa: str, sb: str, total: float, model
) -> dict[tuple[str, ...], float]:
    a = _NupackStrand(sa, name="a")
    b = _NupackStrand(sb, name="b")
    tube = _NupackTube({a: total, b: total}, complexes=_NupackSetSpec(max_size=2), name="t")
    res = _nupack_tube_analysis([tube], model=model, compute=["pfunc"])
    out: dict[tuple[str, ...], float] = {}
    for cx, conc in res.tubes[tube].complex_concentrations.items():
        # cx.name like "(a+b)" → ("a", "b")
        key = tuple(sorted(cx.name.strip("()").split("+")))
        out[key] = float(conc)
    return out


def run_conc(
    n_pairs: int,
    length: int,
    total: float,
    floor: float,
    mismatches: int,
    material: Material,
    celsius: float,
    sodium: float,
    magnesium: float,
    seed: int,
) -> dict:
    if not _HAS_NUPACK:
        return {
            "axis": "conc",
            "skipped": "concentration cross-validation requires NUPACK",
            "rows": [],
        }

    rng = random.Random(seed)
    native = ThermoEngine(material=material, celsius=celsius, sodium=sodium, magnesium=magnesium, backend="native")
    nupack_model = _NupackModel(material=material, celsius=celsius, sodium=sodium, magnesium=magnesium)

    rel_err: list[float] = []
    dex: list[float] = []  # |log10(c_native / c_nupack)| for species above floor in both
    rows = []
    for _ in range(n_pairs):
        # Partner B is the reverse complement of A (with a chance of point mismatches),
        # so a real A+B duplex forms and the concentration solver is actually exercised
        # — random uncorrelated strands barely dimerize and only test "monomers stay
        # monomers", which is trivially in agreement.
        sa = random_sequence(length, material, rng)
        sb_list = list(reverse_complement(sa, material))
        for _m in range(mismatches):
            i = rng.randrange(length)
            sb_list[i] = rng.choice([b for b in ("ACGU" if material == "rna" else "ACGT")
                                     if b != sb_list[i]])
        sb = "".join(sb_list)
        c_native = _strider_tube_conc(sa, sb, total, native)
        c_nupack = _nupack_tube_conc(sa, sb, total, nupack_model)

        species = []
        for key in sorted(set(c_native) | set(c_nupack)):
            cn = c_native.get(key, 0.0)
            cu = c_nupack.get(key, 0.0)
            entry = {"species": "+".join(key), "native_M": cn, "nupack_M": cu}
            if cu > floor:
                entry["rel_err"] = abs(cn - cu) / cu
                rel_err.append(entry["rel_err"])
            if cn > floor and cu > floor:
                entry["dex"] = math.log10(cn / cu)
                dex.append(entry["dex"])
            species.append(entry)
        rows.append({"a": sa, "b": sb, "total_M": total, "species": species})

    return {
        "axis": "conc",
        "total_M": total,
        "floor_M": floor,
        "mismatches": mismatches,
        "partner": "reverse_complement",
        "rel_err": _error_stats(rel_err),
        "dex": _error_stats(dex),  # log10-concentration agreement (0.30 dex = 2×)
        "rows": rows,
    }


# ─── reporting ──────────────────────────────────────────────────────────────


def _fmt(x, spec: str) -> str:
    return "    n/a" if x is None else format(x, spec)


def print_pfunc(res: dict, material: str, celsius: float) -> None:
    print(f"\n[pfunc] ensemble ΔG + speed ({material} @ {celsius} C)")
    hdr = f"{'len':>5} {'native ms':>11}"
    if _HAS_NUPACK:
        hdr += f" {'nupack ms':>11} {'ratio':>8} {'mae ΔΔG':>9} {'max ΔΔG':>9}"
    if _HAS_VIENNA:
        hdr += f" {'mae(V)':>8} {'max(V)':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in res["rows"]:
        line = f"{r['length']:>5} {r['native_ms_per_seq']:>11.2f}"
        if _HAS_NUPACK:
            n = r["ddg_vs_nupack"]
            line += (
                f" {r['nupack_ms_per_seq']:>11.2f}"
                f" {r['speed_ratio_native_over_nupack']:>7.1f}x"
                f" {_fmt(n['mae'], '>9.3f')} {_fmt(n['max_abs'], '>9.3f')}"
            )
        if _HAS_VIENNA:
            v = r["ddg_vs_vienna"]
            line += f" {_fmt(v['mae'], '>8.3f')} {_fmt(v['max_abs'], '>8.3f')}"
        print(line)


def print_pairs(res: dict) -> None:
    print("\n[pairs] base-pair probability matrix agreement (off-diagonal P[i<j])")
    hdr = f"{'len':>5} {'#entries':>9}"
    if _HAS_NUPACK:
        hdr += f" {'mae(N)':>9} {'rmse(N)':>9} {'max(N)':>9}"
    if _HAS_VIENNA:
        hdr += f" {'mae(V)':>9} {'rmse(V)':>9} {'max(V)':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in res["rows"]:
        ref = r.get("bpp_vs_nupack") or r.get("bpp_vs_vienna")
        line = f"{r['length']:>5} {ref['n']:>9}"
        if _HAS_NUPACK:
            n = r["bpp_vs_nupack"]
            line += f" {_fmt(n['mae'], '>9.4f')} {_fmt(n['rmse'], '>9.4f')} {_fmt(n['max_abs'], '>9.4f')}"
        if _HAS_VIENNA:
            v = r["bpp_vs_vienna"]
            line += f" {_fmt(v['mae'], '>9.4f')} {_fmt(v['rmse'], '>9.4f')} {_fmt(v['max_abs'], '>9.4f')}"
        print(line)


def print_conc(res: dict) -> None:
    print("\n[conc] two-strand equilibrium concentration agreement vs NUPACK")
    if res.get("skipped"):
        print(f"  skipped: {res['skipped']}")
        return
    e = res["rel_err"]
    d = res["dex"]
    print(
        f"  species above {res['floor_M']:.0e} M: n={e['n']}  "
        f"mean rel.err={_fmt(e['mae'], '.3f')}  max rel.err={_fmt(e['max_abs'], '.3f')}"
    )
    print(
        f"  log10-conc (dex, both>floor): n={d['n']}  "
        f"mean|Δdex|={_fmt(d['mae'], '.3f')}  max|Δdex|={_fmt(d['max_abs'], '.3f')}"
        "   (0.30 dex = 2×)"
    )


# ─── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--suite",
        nargs="+",
        choices=["pfunc", "pairs", "conc", "all"],
        default=["all"],
        help="which agreement axes to run",
    )
    ap.add_argument("--material", choices=["rna", "dna"], default="rna")
    ap.add_argument("--celsius", type=float, default=37.0)
    # Salt is held at the nearest-neighbor reference state (1 M Na+, 0 Mg2+) so the
    # comparison isolates parameter/DP agreement from salt-model differences (which
    # are validated separately in tests/test_salt_temperature.py). Override to probe
    # physiological conditions, but then native↔reference differences mix the two.
    ap.add_argument("--sodium", type=float, default=1.0, help="Na+ molarity (default 1.0 = NN reference state)")
    ap.add_argument("--magnesium", type=float, default=0.0, help="Mg2+ molarity (default 0.0 = NN reference state)")
    ap.add_argument("--seed", type=int, default=0)
    # pfunc
    ap.add_argument("--lengths", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    # pairs
    ap.add_argument("--pairs-lengths", type=int, nargs="+", default=[20, 40, 60])
    ap.add_argument("--pairs-n-seqs", type=int, default=8)
    # conc
    ap.add_argument("--conc-pairs", type=int, default=6)
    ap.add_argument("--conc-length", type=int, default=12)
    ap.add_argument("--conc-total", type=float, default=1e-6)
    ap.add_argument("--conc-floor", type=float, default=1e-9)
    ap.add_argument("--conc-mismatches", type=int, default=1,
                    help="point mismatches in the reverse-complement partner strand")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    suites = {"pfunc", "pairs", "conc"} if "all" in args.suite else set(args.suite)

    if not (_HAS_NUPACK or _HAS_VIENNA):
        print("WARNING: neither nupack nor RNA (ViennaRNA) importable — native-only.\n")

    refs = []
    if _HAS_NUPACK:
        import nupack

        refs.append(f"nupack {nupack.__version__}")
    if _HAS_VIENNA:
        refs.append(f"viennarna {_RNA.__version__}")

    salt_note = "NN reference state" if (args.sodium == 1.0 and args.magnesium == 0.0) else "custom"
    print(f"=== strider native cross-validation ({args.material} @ {args.celsius} C) ===")
    print(f"references: {', '.join(refs) or 'none'}   seed: {args.seed}")
    print(f"salt: {args.sodium} M Na+, {args.magnesium} M Mg2+ ({salt_note})")

    result: dict = {
        "material": args.material,
        "celsius": args.celsius,
        "sodium": args.sodium,
        "magnesium": args.magnesium,
        "seed": args.seed,
        "nupack_available": _HAS_NUPACK,
        "vienna_available": _HAS_VIENNA,
        "references": refs,
        "axes": {},
    }

    if "pfunc" in suites:
        r = run_pfunc(
            args.lengths, args.n_seqs, args.reps, args.material, args.celsius,
            args.sodium, args.magnesium, args.seed,
        )
        result["axes"]["pfunc"] = r
        print_pfunc(r, args.material, args.celsius)
    if "pairs" in suites:
        r = run_pairs(
            args.pairs_lengths, args.pairs_n_seqs, args.material, args.celsius,
            args.sodium, args.magnesium, args.seed,
        )
        result["axes"]["pairs"] = r
        print_pairs(r)
    if "conc" in suites:
        r = run_conc(
            args.conc_pairs, args.conc_length, args.conc_total, args.conc_floor,
            args.conc_mismatches, args.material, args.celsius,
            args.sodium, args.magnesium, args.seed,
        )
        result["axes"]["conc"] = r
        print_conc(r)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nReceipt written to {args.json}")


if __name__ == "__main__":
    main()
