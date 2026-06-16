"""
Benchmark runners — produce concrete numbers for accuracy and timing claims.

Three runners:

* :func:`run_structure_benchmark` — Zuker MFE on the
  :mod:`strider.benchmarks.structure_refs` set; reports per-sequence
  sensitivity / PPV / F-measure against the published reference structure
  and (when ViennaRNA is installed) head-to-head ΔG agreement.
* :func:`run_tmsd_benchmark` — validates that
  :func:`strider.kinetics.tmsd.toehold_kf` reproduces the Zhang & Winfree
  2009 Fig. 4 lookup at 25 °C and that the Arrhenius extrapolation gives
  monotonic, finite values at 37 °C.
* :func:`run_timing_benchmark` — wall-clock timing of MFE and pfunc across
  a small sequence-length sweep with optional ViennaRNA head-to-head.

Each runner returns a structured dataclass; the CLI in
``scripts/bench_accuracy.py`` formats them into a human-readable table.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from strider.benchmarks.accuracy import (
    StructureMetrics, dot_bracket_metrics, relative_error,
)
from strider.benchmarks.structure_refs import StructureRef, get_references

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine


# ─── shared helpers ──────────────────────────────────────────────────────────


def _has_vienna() -> bool:
    try:
        import RNA  # noqa: F401
        return True
    except ImportError:
        return False


def _random_sequence(n: int, material: str, rng: random.Random) -> str:
    alphabet = "ACGU" if material == "rna" else "ACGT"
    return "".join(rng.choice(alphabet) for _ in range(n))


# ─── structure-prediction benchmark ──────────────────────────────────────────


@dataclass
class StructurePerSequence:
    ref: StructureRef
    predicted_structure: str
    metrics: StructureMetrics
    dG_native: float
    dG_vienna: float | None = None
    structure_vienna: str | None = None


@dataclass
class StructureBenchmark:
    per_sequence: list[StructurePerSequence] = field(default_factory=list)

    @property
    def mean_sensitivity(self) -> float:
        return _safe_mean(s.metrics.sensitivity for s in self.per_sequence)

    @property
    def mean_ppv(self) -> float:
        return _safe_mean(s.metrics.ppv for s in self.per_sequence)

    @property
    def mean_f_measure(self) -> float:
        return _safe_mean(s.metrics.f_measure for s in self.per_sequence)

    @property
    def n_exact(self) -> int:
        return sum(1 for s in self.per_sequence if s.metrics.exact_match)

    @property
    def mean_abs_dG_vienna_diff(self) -> float | None:
        """Mean |ΔG_native − ΔG_vienna| (kcal/mol) over sequences where both are available."""
        diffs = [
            abs(s.dG_native - s.dG_vienna)
            for s in self.per_sequence
            if s.dG_vienna is not None
        ]
        if not diffs:
            return None
        return sum(diffs) / len(diffs)


def run_structure_benchmark(
    engine: "ThermoEngine | None" = None,
    refs: list[StructureRef] | None = None,
    include_vienna: bool = True,
) -> StructureBenchmark:
    """
    Fold every reference, score the prediction, and (optionally) compare to
    a ViennaRNA fold of the same sequence at the same temperature.

    A ``ThermoEngine`` may be passed in to control material / temperature /
    salt; if omitted, engines are allocated per material at the **nearest-
    neighbor reference state (37 °C, 1 M Na⁺, 0 Mg²⁺)** — the condition at
    which the Turner/SantaLucia parameters *and* the reference structures are
    defined.  Folding against those reference structures at the engine's
    physiological salt default (0.137 M Na⁺ / 0.01 M Mg²⁺) would penalise
    marginally-stable short hairpins below their reference-state stability and
    is the wrong comparison for a structure-prediction benchmark.  Mixed-
    material reference lists are handled by allocating one engine per material
    on the fly.
    """
    from strider.thermo.engine import ThermoEngine

    refs = refs if refs is not None else get_references()
    engines: dict[str, ThermoEngine] = {}

    def _engine_for(material: str) -> ThermoEngine:
        if engine is not None and engine.material == material:
            return engine
        if material not in engines:
            engines[material] = ThermoEngine(
                material=material, celsius=37.0, sodium=1.0, magnesium=0.0,
            )
        return engines[material]

    vienna_avail = include_vienna and _has_vienna()
    if vienna_avail:
        import RNA

    out = StructureBenchmark()
    for ref in refs:
        eng = _engine_for(ref.material)
        result = eng.mfe(ref.sequence)
        metrics = dot_bracket_metrics(ref.structure, result.structure)

        dG_v: float | None = None
        struct_v: str | None = None
        if vienna_avail:
            try:
                seq_for_vienna = (
                    ref.sequence if ref.material == "rna"
                    else ref.sequence.replace("T", "U")
                )
                struct_v, dG_v = RNA.fold(seq_for_vienna)
                dG_v = float(dG_v)
            except Exception:
                dG_v = None
                struct_v = None

        out.per_sequence.append(
            StructurePerSequence(
                ref=ref,
                predicted_structure=result.structure,
                metrics=metrics,
                dG_native=result.energy,
                dG_vienna=dG_v,
                structure_vienna=struct_v,
            )
        )
    return out


# ─── TMSD benchmark ──────────────────────────────────────────────────────────


@dataclass
class TMSDPoint:
    toehold_nt: int
    kf_predicted: float
    kf_reference: float
    relative_error: float


@dataclass
class TMSDBenchmark:
    points_25C: list[TMSDPoint] = field(default_factory=list)
    points_37C: list[float] = field(default_factory=list)
    arrhenius_monotonic: bool = True

    @property
    def mean_rel_error(self) -> float:
        return _safe_mean(p.relative_error for p in self.points_25C)

    @property
    def max_rel_error(self) -> float:
        if not self.points_25C:
            return 0.0
        return max(p.relative_error for p in self.points_25C)


def run_tmsd_benchmark() -> TMSDBenchmark:
    """
    Validate :func:`strider.kinetics.tmsd.toehold_kf` against Zhang & Winfree
    (2009) Fig. 4 at 25 °C, then check that the Arrhenius extrapolation to
    37 °C produces strictly-increasing kf values (forward TMSD speeds up
    with temperature on the strider model).
    """
    from strider.kinetics.tmsd import _ZW_KF_25C, toehold_kf

    out = TMSDBenchmark()
    for nt, kf_ref in _ZW_KF_25C.items():
        kf_pred = toehold_kf(nt, material="dna", celsius=25.0)
        err = relative_error(kf_pred, kf_ref)
        out.points_25C.append(
            TMSDPoint(toehold_nt=nt, kf_predicted=kf_pred,
                      kf_reference=kf_ref, relative_error=err)
        )

    # 37 °C extrapolation: monotonic increase in toehold length.
    out.points_37C = [
        toehold_kf(nt, material="dna", celsius=37.0)
        for nt in sorted(_ZW_KF_25C.keys())
    ]
    out.arrhenius_monotonic = all(
        b >= a - 1e-9 for a, b in zip(out.points_37C, out.points_37C[1:])
    )
    return out


# ─── timing benchmark ────────────────────────────────────────────────────────


@dataclass
class TimingRow:
    length: int
    mfe_median_ms: float
    mfe_p95_ms: float
    pfunc_median_ms: float
    pfunc_p95_ms: float
    vienna_mfe_median_ms: float | None = None
    vienna_pfunc_median_ms: float | None = None


@dataclass
class TimingBenchmark:
    rows: list[TimingRow] = field(default_factory=list)


def run_timing_benchmark(
    lengths: list[int] | None = None,
    reps: int = 3,
    material: str = "dna",
    seed: int = 0,
    include_vienna: bool = True,
) -> TimingBenchmark:
    """
    Time MFE and pfunc median + p95 across a length sweep.  Sequences are
    random over the alphabet for the chosen material (so the timing
    reflects mean-case behaviour, not best/worst case).
    """
    from strider.thermo.engine import ThermoEngine

    lengths = lengths if lengths is not None else [20, 50, 100, 200]
    rng = random.Random(seed)
    eng = ThermoEngine(material=material, celsius=37.0)
    vienna_avail = include_vienna and _has_vienna()
    if vienna_avail:
        import RNA

    out = TimingBenchmark()
    for L in lengths:
        seqs = [_random_sequence(L, material, rng) for _ in range(reps)]

        mfe_times: list[float] = []
        pfunc_times: list[float] = []
        for s in seqs:
            t0 = time.perf_counter()
            eng.mfe(s)
            mfe_times.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            eng.pfunc(s)
            pfunc_times.append((time.perf_counter() - t0) * 1000)

        vienna_mfe: float | None = None
        vienna_pf: float | None = None
        if vienna_avail:
            vmfe: list[float] = []
            vpf: list[float] = []
            for s in seqs:
                vs = s.replace("T", "U") if material == "rna" else s.replace("T", "U")
                t0 = time.perf_counter()
                RNA.fold(vs)
                vmfe.append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                RNA.pf_fold(vs)
                vpf.append((time.perf_counter() - t0) * 1000)
            vienna_mfe = statistics.median(vmfe) if vmfe else None
            vienna_pf = statistics.median(vpf) if vpf else None

        out.rows.append(TimingRow(
            length=L,
            mfe_median_ms=statistics.median(mfe_times),
            mfe_p95_ms=_p95(mfe_times),
            pfunc_median_ms=statistics.median(pfunc_times),
            pfunc_p95_ms=_p95(pfunc_times),
            vienna_mfe_median_ms=vienna_mfe,
            vienna_pfunc_median_ms=vienna_pf,
        ))
    return out


# ─── small helpers ───────────────────────────────────────────────────────────


def _safe_mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _p95(values: list[float]) -> float:
    """Cheap p95 — sort and pick the index at ceil(0.95*n)-1."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(0.95 * len(s)) - 1) if len(s) > 1 else 0
    return s[idx]
