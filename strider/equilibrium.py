"""
Equilibrium concentration solver for nucleic acid complex mixtures.

Given a set of strands at known total concentrations and a list of complexes
with associated partition functions, this module solves for the equilibrium
concentration of every complex via mass-action and mass-balance constraints.

Algorithm
---------
We solve the standard convex equilibrium problem (Dirks, Bois, Schaeffer,
Winfree & Pierce, 2007, SIAM Review 49:65-88).  Let

    A_{c,s} = number of copies of strand s in complex c
    log Q_c = -ΔG_c° / RT   (dimensionless log partition function)
    b_s     = total concentration of strand s   (mol / L)

The equilibrium concentrations x_c minimize

    f(x) = Σ_c x_c (log(x_c / Q_c) - 1)     s.t.  A^T x = b,  x ≥ 0

which is convex.  We solve the dual: find chemical potentials μ_s such that

    x_c(μ) = Q_c · exp(Σ_s A_{c,s} μ_s)
    A^T x(μ) = b

via damped Newton iteration on the residual r(μ) = A^T x(μ) - b.  The Hessian
H = A^T diag(x) A is positive definite at the optimum so Newton converges
quadratically once close enough.  A simple backtracking line search guarantees
monotone reduction of |r|.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine

R = 1.987e-3   # kcal / (mol · K)


@dataclass
class EquilibriumResult:
    """
    Equilibrium concentrations and diagnostics.

    Attributes
    ----------
    concentrations : complex_name → concentration (M)
    strand_free    : strand_name → free (monomer) concentration (M)
    converged      : True if final ‖A^T x − b‖_∞ / b < tol
    iterations     : Newton steps taken
    residual       : final relative mass-balance residual
    """
    concentrations: dict[str, float]
    strand_free: dict[str, float] = field(default_factory=dict)
    converged: bool = False
    iterations: int = 0
    residual: float = 0.0


def cyclic_symmetry(strand_list: list[str]) -> int:
    """
    Cyclic rotational symmetry σ of a strand list.

    For an ordered complex of N strands, σ is the number of cyclic rotations
    of the list that leave it unchanged.  Examples::

        [A, B]          → 1   (heterodimer)
        [A, A]          → 2   (homodimer)
        [A, A, A]       → 3   (homotrimer)
        [A, B, A, B]    → 2
        [A, A, B]       → 1   (the three rotations are distinct)

    The partition function of an ordered complex must be divided by σ to
    avoid over-counting indistinguishable cyclic rearrangements (see Dirks
    et al. 2007, eq. 11).
    """
    n = len(strand_list)
    if n <= 1:
        return 1
    count = 0
    for k in range(n):
        rotated = strand_list[k:] + strand_list[:k]
        if rotated == strand_list:
            count += 1
    return count


def water_molarity(celsius: float = 37.0) -> float:
    """Molar concentration of pure water at a given temperature (mol/L).

    Polynomial fit from Lide (2008) CRC Handbook of Chemistry and Physics:
        ρ = 0.99987 + 6.69e-5·T − 8.96e-6·T²  (g/mL)
    divided by molar mass 18.015 g/mol.  At 25 °C → 55.34, at 37 °C → 55.13.
    """
    T = celsius
    rho = 0.99987 + 6.69e-5 * T - 8.96e-6 * T * T   # g/mL
    return rho * 1000.0 / 18.015


def solve_equilibrium(
    complexes: dict[str, tuple[list[str], float]],
    totals: dict[str, float],
    celsius: float = 37.0,
    max_iter: int = 200,
    tol: float = 1e-9,
    standard_state_M: float = 1.0,
) -> EquilibriumResult:
    """
    Solve equilibrium complex concentrations for a multi-strand system.

    Parameters
    ----------
    complexes : mapping ``name → (strand_list, ΔG_kcal_per_mol)``.
        Every strand species that appears in ``totals`` must also appear as a
        single-strand complex (the "monomer" entry) so the free-strand state
        is represented in the equilibrium ensemble.  ΔG values are the
        ensemble free energies of each complex (kcal/mol) at the chosen
        temperature, expressed at the 1 M standard state — i.e., the same
        convention as :meth:`ThermoEngine.pfunc`.
    totals : strand species → total strand concentration (M).
    celsius : temperature (used only to convert ΔG → log Q).
    max_iter, tol : Newton stopping criteria.
    """
    strand_names = list(totals.keys())
    s_idx = {s: i for i, s in enumerate(strand_names)}
    cx_names = list(complexes.keys())
    n_s = len(strand_names)
    n_c = len(cx_names)

    if n_s == 0 or n_c == 0:
        return EquilibriumResult(concentrations={}, strand_free={}, converged=True)

    A = np.zeros((n_c, n_s))
    logQ = np.zeros(n_c)
    RT = R * (celsius + 273.15)

    # Standard-state conversion: if input ΔG is at c0 (e.g. a water-molarity
    # reference), shift to a 1 M reference for the solver, then scale back at
    # the end.  The shift is (N − 1) · ln(c0_M) per complex of N strands.
    log_c0 = math.log(standard_state_M) if standard_state_M != 1.0 else 0.0

    # Symmetry handling lives at the pfunc layer (the engine.pfunc method is
    # responsible for σ-correcting homomeric multi-strand complexes), so this
    # solver assumes every input ΔG already corresponds to the "species"
    # partition function for that complex (Dirks et al. 2007 §4).

    for i, name in enumerate(cx_names):
        strands, dG = complexes[name]
        for s in strands:
            if s not in s_idx:
                raise ValueError(
                    f"complex {name!r} references strand {s!r} not in totals"
                )
            A[i, s_idx[s]] += 1
        n_strands_in = len(strands)
        # Q at 1 M = Q at c0 / c0^(N-1)  ⇒  add (N-1)·ln(c0) to logQ_input
        # ΔG_1M = ΔG_c0 + (N-1)·RT·ln(c0); logQ_1M = -ΔG_1M/RT
        logQ[i] = -dG / RT - (n_strands_in - 1) * log_c0

    b = np.array([float(totals[s]) for s in strand_names])

    # Initialize μ so that monomer x ≈ totals.
    monomer_logQ = {}
    for i in range(n_c):
        if A[i].sum() == 1 and (A[i] > 0).sum() == 1:
            s = int(A[i].argmax())
            monomer_logQ.setdefault(s, logQ[i])

    mu = np.zeros(n_s)
    for s in range(n_s):
        lq = monomer_logQ.get(s, 0.0)
        mu[s] = math.log(max(b[s], 1e-30)) - lq

    def _x_and_residual(mu_vec):
        log_x = logQ + A @ mu_vec
        log_x = np.clip(log_x, -200, 200)
        x = np.exp(log_x)
        r = A.T @ x - b
        return x, r

    iteration = 0
    residual_norm = float("inf")
    for iteration in range(1, max_iter + 1):
        x, r = _x_and_residual(mu)
        rel = np.abs(r) / np.maximum(b, 1e-30)
        residual_norm = float(rel.max())
        if residual_norm < tol:
            break

        H = A.T @ (x[:, None] * A) + 1e-12 * np.eye(n_s)
        try:
            delta = np.linalg.solve(H, r)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(H, r, rcond=None)

        # Backtracking line search on max relative residual.
        step = 1.0
        for _ in range(40):
            mu_new = mu - step * delta
            _, r_new = _x_and_residual(mu_new)
            rel_new = float((np.abs(r_new) / np.maximum(b, 1e-30)).max())
            if rel_new < residual_norm:
                mu = mu_new
                break
            step *= 0.5
        else:
            break

    x, r = _x_and_residual(mu)
    rel = np.abs(r) / np.maximum(b, 1e-30)
    residual_norm = float(rel.max())

    conc = {name: float(x[i]) for i, name in enumerate(cx_names)}
    free = {}
    for s, name in enumerate(strand_names):
        free[name] = sum(
            float(x[i]) for i in range(n_c)
            if A[i, s] == 1 and A[i].sum() == 1
        )

    return EquilibriumResult(
        concentrations=conc,
        strand_free=free,
        converged=residual_norm < tol,
        iterations=iteration,
        residual=residual_norm,
    )


def equilibrium_from_engine(
    engine: "ThermoEngine",
    strands: dict[str, str],
    totals: dict[str, float],
    complexes: Iterable[tuple[str, list[str]]] | None = None,
    max_size: int = 2,
    tol: float = 1e-9,
) -> EquilibriumResult:
    """
    Convenience wrapper: enumerate complexes up to ``max_size`` strands,
    compute their partition functions with ``engine``, then solve the
    equilibrium.

    **API status: stable compatibility wrapper** — part of the 1.0 stable
    surface and *not* deprecated; kept indefinitely as a thin shim. For new
    code, prefer the higher-level :class:`strider.tube.Tube` API, which is what
    this function delegates to internally and which also exposes pair
    probabilities, ensemble defect, and richer complex specification.

    Parameters
    ----------
    engine     : ThermoEngine — provides pfunc() for ΔG calculation.
    strands    : strand_name → sequence (must include every name in ``totals``).
    totals     : strand_name → total concentration (M).
    complexes  : explicit iterable of ``(name, [strand_list])`` to include.
                 If ``None`` (default), all complexes up to ``max_size`` strands
                 are enumerated automatically.  Monomers are always included.
    max_size   : maximum number of strands per complex when auto-enumerating.
    """
    from strider.tube import ComplexSet, Complex, SetSpec, Strand, Tube

    material = engine.material
    strand_objs = {n: Strand(n, seq, material=material) for n, seq in strands.items()}

    if complexes is None:
        spec = SetSpec(max_size=max_size)
    else:
        explicit = [
            Complex(strands=tuple(strand_objs[s] for s in sl), name=cname)
            for cname, sl in complexes
        ]
        spec = SetSpec(max_size=0, include=explicit)

    tube = Tube(
        strand_totals={strand_objs[n]: float(c) for n, c in totals.items()},
        complexes=ComplexSet(list(strand_objs.values()), spec=spec),
    )
    res = tube.analyze(engine, tol=tol)
    return EquilibriumResult(
        concentrations=dict(res.concentrations),
        strand_free=dict(res.strand_free),
        converged=res.converged,
        iterations=res.iterations,
        residual=res.residual,
    )
