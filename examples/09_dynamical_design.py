"""
Example 09: Closed-loop dynamical sequence design (mantis-feedback driven)

Drives sequence optimization from a *kinetic* cost — an ODE simulation —
instead of a static equilibrium defect.  Every score evaluation:

    1.  Take the candidate sequence
    2.  Rebuild a strider CircuitBridge (recomputes ΔΔG per reaction)
    3.  Convert ΔΔG → rates via Zhang & Winfree (2009)
    4.  Hand the rate dict to mantis as a CRNetwork
    5.  Run an ODE simulation (deterministic Radau)
    6.  Score the trajectory and feed the scalar back to the SA optimizer

The closed loop is honest: the rate constants reflect the *real* binding
affinity of the latest sequence to its partners, not a frozen rate table.

── Demonstrated objective ──────────────────────────────────────────────────
This example showcases `DesignObjective.kinetic_trajectory`, the canonical
use case from outperform_nupack.md item 1:  "Optimize sequences to match a
target step-response."  We specify a desired [AB](t) curve and let the
optimizer mutate the toehold of A until the simulated trajectory matches.

Other dynamical factories — `maximize_kcat`, `minimize_leak`,
`bistable_threshold`, `from_simulation` — follow the same pattern (closure
over `bridge.to_crnetwork`, ODE inside the score function); see the unit
tests in `tests/test_design_dynamical.py` for self-contained examples of
each.

── Circuit ─────────────────────────────────────────────────────────────────
A single bimolecular hybridization step:

    A + B  <->  AB

Strand A is just the 7-nt designed toehold (no flanking tail), so ΔΔG of
A·B is fully sequence-dependent.  Against the fixed 18-nt partner B, a 7-mer
spans ΔΔG ≈ −1 kcal/mol (no complementary register) down to ≈ −5.8 kcal/mol
(the strongest 7-nt window of B, reverse-complemented).  Because B is long,
*partial* complements still bind in their best register, so the cost
landscape is graded rather than a needle-in-a-haystack — the optimizer has a
gradient to climb.

── What the design lever actually is (read this — it was a bug magnet) ──────
With [B] in large excess the reaction is pseudo-first-order: the apparent
rate is k_obs = kf·[B] + kr and the equilibrium fraction bound is
    [AB]/A₀ = Keq·[B] / (1 + Keq·[B]),     Keq = exp(−ΔΔG / RT).
Two consequences drive every choice below:

  • **kf is sequence-independent.**  Zhang–Winfree kf depends on toehold
    *length*, not sequence — only kr (hence Keq, hence the *plateau*) varies
    with ΔΔG.  So the tunable knob is the **plateau height (response
    amplitude)**, not the rise time.  At [B] = 300 µM the system reaches
    steady state in milliseconds regardless of sequence.

  • **The plateau is sub-saturating, so the target must be reachable.**  The
    strongest 7-mer reaches only Keq·[B] ≈ 9 at [B] = 300 µM → ~90 % bound.
    An earlier version of this example targeted *full* saturation (10 nM) —
    physically unreachable for a 7-mer, so the MSE never fell and the
    optimizer looked broken.  Here the target is pinned to a fraction of the
    *measured* achievable maximum (the perfect complement's own plateau), so
    it is reachable by construction.  We aim for 50 % of that maximum, an
    **intermediate** affinity: both too-weak and too-strong toeholds are
    penalized, making this a genuine two-sided match rather than "maximize
    binding."

Requires the optional `mantis-delta` dependency (`pip install strider-dna[mantis]`).
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
import matplotlib.pyplot as plt
import numpy as np

from strider import (
    CircuitBridge,
    DesignObjective,
    DomainSpec,
    HardConstraint,
    SequenceDesigner,
    ThermoEngine,
)

_here = pathlib.Path(__file__).parent

# ── Mantis availability check (fail clearly, not with a wall of +inf) ───────
# kinetic_trajectory swallows factory exceptions and returns +inf, so a
# missing mantis silently turns every candidate into +inf and the optimizer
# "succeeds" on noise.  Catch it up front instead.
try:
    import mantis  # noqa: F401
    from mantis import CRNetwork  # noqa: F401
except Exception:
    print(
        "This example requires the optional 'mantis-delta' dependency.\n"
        "Install it with:  pip install strider-dna[mantis]\n"
        "or from source:   cd hairpin/mantis && pip install -e ."
    )
    sys.exit(0)


def revcomp(seq: str) -> str:
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(comp[b] for b in reversed(seq))


# ── Engine + fixed strand ──────────────────────────────────────────────────
engine = ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)

# Fixed 18-nt partner.  A is *just* the 7-nt toehold (no flanking tail), so
# A·B affinity is dominated by the toehold sequence.
B_SEQ = "GCAGTGAGACGAGCTGCT"
TOEHOLD_LEN = 7


def make_factory(domain_name: str = "toehold_A"):
    """Closure: ``(seqs) -> mantis.CRNetwork``."""
    reactions = ["A + B <-> AB"]

    def factory(seqs: dict[str, str]):
        toehold = seqs[domain_name]
        bridge = CircuitBridge(
            reactions=reactions,
            sequences={"A": toehold, "B": B_SEQ},
            engine=engine,
            toehold_map={"A + B <-> AB": len(toehold)},
        )
        return bridge.to_crnetwork()

    return factory


factory = make_factory()

# ── Reaction conditions ─────────────────────────────────────────────────────
# 10 nM A + 300 µM B (B in 3×10⁴ excess so [B] stays ~constant).  The high
# excess lifts Keq·[B] into the discriminating range so a 7-mer can span from
# ~0 to ~90 % bound — without it the whole 7-mer family plateaus below 10 %
# bound and there is nothing to optimize.
A0_CONC = 1e-8          # M, total A
B_CONC = 3e-4           # M, fixed partner (pseudo-first-order)
IC = {"A": A0_CONC, "B": B_CONC, "AB": 0.0}
t_window = (0.0, 1.0)
t_eval = np.linspace(*t_window, 121)


def plateau_of(toehold: str) -> float:
    """Steady-state [AB] (in M) for a given toehold, via the closed loop."""
    sim = factory({"toehold_A": toehold}).simulate(
        IC, t_span=t_window, t_eval=t_eval
    )
    return float(sim.concentrations["AB"][-1])


# ── Physically-anchored target ──────────────────────────────────────────────
# The strongest possible 7-mer is the reverse complement of B's most stable
# 7-nt window.  Measure ITS plateau — that is the achievable ceiling — and aim
# for half of it (an intermediate-affinity design, reachable by construction).
windows = [B_SEQ[i : i + TOEHOLD_LEN] for i in range(len(B_SEQ) - TOEHOLD_LEN + 1)]
ideal_toehold = max((revcomp(w) for w in windows), key=plateau_of)
plateau_max = plateau_of(ideal_toehold)

TARGET_FRACTION = 0.5
A_plateau = TARGET_FRACTION * plateau_max     # M
tau_target = 0.05                              # s (rise faster than this is fine)
target_AB = A_plateau * (1.0 - np.exp(-t_eval / tau_target))
target_curve = {"AB": target_AB}

print("── Reachable design window (measured via the closed loop) ─────────")
print(f"  strongest 7-mer = {ideal_toehold}  →  plateau {plateau_max * 1e9:.2f} nM")
print(f"  target          = {TARGET_FRACTION:.0%} of max  →  {A_plateau * 1e9:.2f} nM")

# ── Composed objective ─────────────────────────────────────────────────────
objective = (
    # Primary: match the target [AB](t) step response.
    1.0 * DesignObjective.kinetic_trajectory(
        factory, IC, target_curve, t_eval, label="trajectory_MSE",
    )
    # Mild static regularizer so the toehold stays synthesizable.
    + 0.05 * DesignObjective.gc_content("toehold_A", target_gc=0.55)
)

# ── Baseline: deliberately poor (low-affinity, GC-imbalanced) toehold ─────
baseline_seqs = {"toehold_A": "ATATATA"}    # weak hybridization, low GC
baseline_breakdown = objective.evaluate_breakdown(baseline_seqs)
baseline_score = sum(baseline_breakdown.values())

print("\n── Baseline (deliberately weak toehold) ──────────────────")
print(f"  toehold_A = {baseline_seqs['toehold_A']}")
for label, val in baseline_breakdown.items():
    print(f"    {label:<42}  {val:+.4f}")
print(f"  total = {baseline_score:+.4f}")

# ── Optimize ───────────────────────────────────────────────────────────────
print("\n── Optimizing (closed-loop mantis feedback) ──────────────")
print("  Each SA step rebuilds the CRN and reruns the ODE against the target curve.")
designer = SequenceDesigner(engine=engine, seed=3)
result = designer.design(
    domains={"toehold_A": DomainSpec(length=TOEHOLD_LEN, material="dna")},
    objective=objective,
    hard_constraints=[
        HardConstraint.gc_content(min_gc=0.4, max_gc=0.8),
        HardConstraint.max_run(max_run_length=3),
    ],
    n_trials=4,
    max_iterations=200,
    verbose=False,
)

best_toehold = result.sequences["toehold_A"]
best_breakdown = result.objective_breakdown
print(f"\n  best toehold_A = {best_toehold}")
for label, val in best_breakdown.items():
    print(f"    {label:<42}  {val:+.4f}")
print(f"  total = {result.objective_value:+.4f}")
print(f"  improvement = {baseline_score - result.objective_value:+.4f}  (positive = better)")

# ── Visualize: target vs. baseline vs. optimized (vs. strongest possible) ──
sim_base = factory(baseline_seqs).simulate(IC, t_span=t_window, t_eval=t_eval)
sim_best = factory(result.sequences).simulate(IC, t_span=t_window, t_eval=t_eval)
sim_ideal = factory({"toehold_A": ideal_toehold}).simulate(
    IC, t_span=t_window, t_eval=t_eval
)


def plateau_nm(arr: np.ndarray) -> float:
    return float(arr[-1]) * 1e9


plateau_target = plateau_nm(target_AB)
plateau_base = plateau_nm(sim_base.concentrations["AB"])
plateau_best = plateau_nm(sim_best.concentrations["AB"])
plateau_ideal = plateau_nm(sim_ideal.concentrations["AB"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle(
    "Closed-loop dynamical design: kinetic-trajectory matching",
    fontsize=12, fontweight="bold",
)

# Left panel: target + baseline + optimized + achievable ceiling
ax = axes[0]
ax.plot(t_eval, target_AB * 1e9, ":", color="#222", lw=2.0,
        label=f"target  ({plateau_target:.2f} nM)")
ax.plot(t_eval, sim_ideal.concentrations["AB"] * 1e9, "-", color="#cc7a00",
        lw=1.5, alpha=0.7,
        label=f"strongest 7-mer {ideal_toehold}  ({plateau_ideal:.2f} nM)")
ax.plot(t_eval, sim_base.concentrations["AB"] * 1e9, "--",
        color="#aaaaaa", lw=2.0,
        label=f"baseline {baseline_seqs['toehold_A']}  ({plateau_base:.2f} nM)")
ax.plot(t_eval, sim_best.concentrations["AB"] * 1e9, "-",
        color="#22882f", lw=2.5,
        label=f"optimized {best_toehold}  ({plateau_best:.2f} nM)")
ax.axhline(plateau_target, color="#222", linestyle=":", lw=0.5, alpha=0.4)
ax.set_xlabel("Time (s)")
ax.set_ylabel("[AB] (nM)")
ax.set_title(
    "Trajectory match — design lever is the plateau\n"
    "(kf is length-set, so kinetics are ~instant at 300 µM B)"
)
ax.legend(fontsize=8, loc="center right")
ax.grid(alpha=0.3)

# Right panel: per-trial SA convergence — each bar is the best objective
# score reached on that trial, with the global best highlighted.
ax = axes[1]
trial_scores = result.trial_scores
n_trials = len(trial_scores)
colors = ["#22882f" if s == result.objective_value else "#7799cc"
          for s in trial_scores]
bars = ax.bar(range(1, n_trials + 1), trial_scores, color=colors,
              edgecolor="#222", linewidth=0.5)
ax.axhline(baseline_score, color="#aaaaaa", linestyle="--", lw=1.5,
           label=f"baseline ({baseline_seqs['toehold_A']}) = {baseline_score:.3f}")
ax.axhline(result.objective_value, color="#22882f", linestyle=":", lw=1.5,
           label=f"best ({best_toehold}) = {result.objective_value:.3f}")
for i, s in enumerate(trial_scores):
    ax.text(i + 1, s + max(trial_scores) * 0.02, f"{s:.3f}",
            ha="center", fontsize=8)
ax.set_xticks(range(1, n_trials + 1))
ax.set_xlabel("SA trial")
ax.set_ylabel("Objective (lower = better)")
ax.set_title("Closed-loop SA convergence\n(each bar = best of one annealing run)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out = _here / "dynamical_design.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out.name}")
print("\nDone.")
