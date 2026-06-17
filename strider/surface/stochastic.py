"""
[EXPERIMENTAL API — outside the semver contract until stabilized; see README
"API stability and versioning".]

Low-copy stochastic surface capture — shot-noise-limited detection (Frontier §6).

The deterministic :class:`~strider.surface.transducer.SurfaceModel` returns the
*mean* number of captured molecules and calls an assay "detectable" when that
mean signal clears the read-out floor.  At the concentrations that matter for a
real limit of detection (fM–aM), that is the wrong question: the analyte aliquot
holds only ~10¹–10⁴ molecules, the captured count is a small integer, and its
**Poisson shot noise is the dominant noise source** — not the electronics.  No
amount of amplifier gain beats counting statistics.

This module adds that physics:

  * **Capture is a counting process.**  The number of captured molecules ``N`` is
    Poisson with mean μ = the deterministic ``captured_count`` (compound Poisson:
    a Poisson-distributed aliquot population, binomially thinned by the capture
    efficiency).  Its standard deviation is √μ.

  * **Detection limit via Currie (1968).**  With a blank mean ``μ_b`` (non-specific
    binding) and a Gaussian read-out noise ``σ_read`` (the ADC/TIA floor expressed
    in equivalent label events), the **critical level** (decision threshold) is
    ``L_C = k·σ₀`` with ``σ₀ = √(μ_b + σ_read²)``, and the **detection limit** is

        L_D = 2·L_C + k²            (net counts above blank)

    — the canonical Currie result for heteroscedastic Poisson counting at
    k = 1.645 (the z₀.₉₅ quantile, α = β = 0.05), i.e. ``L_D = 3.29·σ₀ + 2.71``.
    The ``+k²`` term is the shot-noise floor: even with perfect electronics
    (σ_read→0) and a zero blank, you still need ``L_D = k² ≈ 2.71`` captured
    molecules — the textbook zero-background counting limit — to call a
    detection at 95 %/95 % confidence.  The
    :meth:`StochasticSurfaceModel.shot_noise_lod` it produces is therefore higher
    — and more honest — than the deterministic ``SurfaceModel.lod``.

  * **mantis SSA driver.**  :meth:`StochasticSurfaceModel.simulate_capture` builds
    a one-reaction capture CRN (``Analyte → Captured``, pseudo-first-order at the
    diffusion-limited efficiency) and runs the mantis Gillespie SSA over an
    ensemble of seeds, returning the empirical captured-count distribution.  It
    reproduces the Poisson model and demonstrates the
    sequences → mantis → surface pipeline at single-molecule resolution.

NUPACK has no surface, no kinetics, and no stochastic mode at all; this is
several layers past what it can express.  All SI units; concentrations in molar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from strider.surface.labels import N_A
from strider.surface.transducer import SurfaceModel, SurfaceParams, captured_count

# Currie coverage factor k = z_0.95 (one-sided normal quantile) for α = β = 0.05.
# Gives the canonical L_D = 2·L_C + k² = 3.29·σ₀ + 2.71 counts.
K_CURRIE = 1.6449


@dataclass
class CurrieLevels:
    """Decision / detection thresholds for a counting measurement (counts)."""
    blank_mean: float       # μ_b, expected blank counts
    sigma_read: float       # σ_read, read-out noise in equivalent counts
    sigma_blank: float      # σ₀ = √(μ_b + σ_read²)
    critical_level: float   # L_C = k·σ₀  (net counts to *decide* "detected")
    detection_limit: float  # L_D = 2·L_C + k²  (net counts reliably detected)
    k: float


def readout_sigma_counts(params: SurfaceParams) -> float:
    """Read-out noise floor expressed in equivalent captured-label *counts*.

    The :class:`ReadoutChain` charge floor is one resolvable charge quantum; one
    captured event contributes ``label.signal_per_event()`` of charge, so the
    floor corresponds to ``charge_floor / signal_per_event`` events.  This is the
    Gaussian (electronics) noise that sits alongside the Poisson shot noise.
    """
    q_event = params.label.signal_per_event()
    if q_event <= 0:
        return float("inf")
    return params.readout.charge_floor_C() / q_event


def currie_levels(blank_mean: float, sigma_read: float, k: float = K_CURRIE) -> CurrieLevels:
    """Currie (1968) critical level and detection limit, in net counts.

    ``L_C = k·σ₀`` and ``L_D = 2·L_C + k²`` with ``σ₀ = √(μ_b + σ_read²)`` —
    the closed-form solution of ``L_D = L_C + k·√(μ_b + L_D + σ_read²)`` for
    Poisson counting on top of a Gaussian read-out floor.
    """
    blank_mean = max(blank_mean, 0.0)
    sigma_read = max(sigma_read, 0.0)
    sigma_blank = math.sqrt(blank_mean + sigma_read ** 2)
    l_c = k * sigma_blank
    l_d = 2.0 * l_c + k ** 2
    return CurrieLevels(blank_mean, sigma_read, sigma_blank, l_c, l_d, k)


def detection_probability(
    mean_signal: float,
    levels: CurrieLevels,
) -> float:
    """P(declare detection) for a true net signal of ``mean_signal`` counts.

    Detection fires when the measured net count exceeds the critical level
    ``L_C``.  The measurement has variance ``μ_b + mean_signal + σ_read²``
    (Poisson shot noise + read-out), so by a normal approximation

        P = Φ( (mean_signal − L_C) / √(μ_b + mean_signal + σ_read²) ).
    """
    mean_signal = max(mean_signal, 0.0)
    var = levels.blank_mean + mean_signal + levels.sigma_read ** 2
    if var <= 0:
        return 1.0 if mean_signal > levels.critical_level else 0.0
    z = (mean_signal - levels.critical_level) / math.sqrt(var)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class CaptureSamples:
    """Empirical captured-count distribution from a mantis SSA ensemble."""
    counts: np.ndarray          # captured count per trajectory
    mean_signal: float          # analytic Poisson mean μ (deterministic capture)
    n_total: float              # expected molecules in the aliquot
    p_capture: float            # per-molecule capture efficiency

    @property
    def empirical_mean(self) -> float:
        return float(np.mean(self.counts))

    @property
    def empirical_var(self) -> float:
        return float(np.var(self.counts))

    def detection_rate(self, levels: CurrieLevels) -> float:
        """Fraction of trajectories whose net count clears the critical level."""
        net = self.counts - levels.blank_mean
        return float(np.mean(net > levels.critical_level))


class StochasticSurfaceModel:
    """Shot-noise-aware wrapper around a :class:`SurfaceModel`.

    Shares the deterministic capture physics (diffusion-limited flux, ULOQ) for
    the *mean*, then layers Poisson counting statistics + the Currie detection
    framework on top.  Optionally drives the capture as a mantis Gillespie SSA.
    """

    def __init__(
        self,
        params: SurfaceParams | None = None,
        *,
        blank_mean_counts: float = 0.0,
        k: float = K_CURRIE,
    ) -> None:
        self.params = params or SurfaceParams()
        self.surface = SurfaceModel(self.params)
        self.blank_mean_counts = float(blank_mean_counts)
        self.k = float(k)

    # ── counting levels ──────────────────────────────────────────────────────
    def levels(self) -> CurrieLevels:
        """Currie decision/detection thresholds for this read-out chain."""
        return currie_levels(self.blank_mean_counts,
                             readout_sigma_counts(self.params), self.k)

    def capture_mean(self, times: np.ndarray, species_M: np.ndarray) -> float:
        """Mean captured count μ, respecting the finite analyte budget.

        The deterministic :func:`captured_count` integrates a diffusion-limited
        flux against a bulk concentration that is *never depleted* — an
        infinite-reservoir assumption that, at the fM–aM concentrations where the
        LOD lives, lets it "capture" far more molecules than the small aliquot
        actually holds.  Here we cap it physically: capture depletes the analyte
        pool, so the mean follows ``μ = N_total·(1 − exp(−N_unsat/N_total))`` (and
        is still bounded by the finite probe-site count ``N_max``).  When
        ``N_unsat ≪ N_total`` this reduces to the linear ``μ ≈ N_unsat``; when
        ``N_unsat ≫ N_total`` it saturates at ``N_total`` (the whole aliquot is
        eventually caught) instead of exceeding it.
        """
        n_total, n_unsat = self._flux_budget(times, species_M)
        if n_total <= 0:
            return 0.0
        mu = n_total * (1.0 - math.exp(-min(n_unsat / n_total, 700.0)))
        return min(mu, self.params.max_capture_sites())

    def detection_probability(self, times: np.ndarray, species_M: np.ndarray) -> float:
        """P(detect) for a trace, including Poisson shot noise."""
        return detection_probability(self.capture_mean(times, species_M), self.levels())

    # ── shot-noise-limited LOD ───────────────────────────────────────────────
    def shot_noise_lod(self, make_trace, triggers) -> "float | None":
        """Lowest trigger whose mean capture reaches the Currie detection limit.

        ``make_trace(c) -> (times, species_M)`` as for ``SurfaceModel.lod``.
        Returns the first trigger with ``μ(c) ≥ L_D`` (which guarantees the
        false-negative rate is ≤ β by construction), or ``None``.  This LOD is
        shot-noise-limited: it never drops below the ``L_D ≈ k²`` counting floor
        no matter how quiet the electronics are.
        """
        l_d = self.levels().detection_limit
        for c in np.asarray(triggers, dtype=float):
            times, species_M = make_trace(float(c))
            if self.capture_mean(times, species_M) >= l_d:
                return float(c)
        return None

    # ── capture efficiency / molecule budget (for the SSA) ───────────────────
    def _flux_budget(self, times: np.ndarray, species_M: np.ndarray) -> tuple[float, float]:
        """Return (N_total molecules in the aliquot, N_unsat infinite-reservoir flux).

        ``N_unsat = ∫ 4·D·a·f(τ)·C(t) dt`` is the would-be capture if the bulk
        never depleted; the caller turns it into a depletion-corrected mean /
        per-molecule probability.
        """
        p = self.params
        D, a = p.d_species_m2_s, p.electrode_radius_m
        from strider.surface.transducer import shoup_szabo_f
        t = np.asarray(times, dtype=float)
        c_num = np.asarray(species_M, dtype=float) * N_A * 1000.0   # /m³
        tau = 4.0 * D * np.maximum(t, 0.0) / a ** 2
        flux = 4.0 * D * a * shoup_szabo_f(tau) * c_num
        n_unsat = float(np.trapezoid(flux, t))
        c_final = float(np.asarray(species_M)[-1])
        n_total = c_final * N_A * 1000.0 * (p.sample_volume_L * 1e-3)  # in m³
        return n_total, n_unsat

    def _capture_budget(self, times: np.ndarray, species_M: np.ndarray) -> tuple[float, float]:
        """Return (N_total molecules in the aliquot, per-molecule capture prob).

        ``p_cap = 1 − exp(−N_unsat/N_total)`` is the depletion-corrected
        probability that any given molecule is caught during the incubation, so
        captured ~ Binomial(N_total, p_cap) ≈ Poisson(μ) in the low-copy regime.
        """
        n_total, n_unsat = self._flux_budget(times, species_M)
        if n_total <= 0:
            return 0.0, 0.0
        p_cap = 1.0 - math.exp(-min(n_unsat / n_total, 700.0))
        return n_total, p_cap

    def simulate_capture(
        self,
        times: np.ndarray,
        species_M: np.ndarray,
        *,
        n_trajectories: int = 200,
        seed: int = 0,
        max_molecules: int = 200_000,
    ) -> CaptureSamples:
        """Sample the captured-count distribution with the mantis Gillespie SSA.

        Builds a one-reaction capture CRN ``Analyte → Captured`` whose
        pseudo-first-order rate reproduces the diffusion-limited capture
        efficiency over the incubation, then runs ``n_trajectories`` independent
        SSA realizations.  Intended for the low-copy (LOD) regime; raises if the
        molecule budget exceeds ``max_molecules`` (SSA would be needlessly slow).

        Requires the optional ``mantis-delta`` dependency.
        """
        try:
            from mantis import CRNetwork
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "simulate_capture requires mantis-delta (pip install mantis-delta)"
            ) from exc

        n_total, p_cap = self._capture_budget(times, species_M)
        if n_total > max_molecules:
            raise ValueError(
                f"aliquot holds ~{n_total:.0f} molecules (> max_molecules={max_molecules}); "
                "the SSA driver targets the low-copy regime — lower the concentration "
                "or raise max_molecules")
        duration = max(float(times[-1]) - float(times[0]), 1e-30)
        # first-order rate that gives capture probability p_cap over the window
        k_cap = -math.log(max(1.0 - p_cap, 1e-300)) / duration
        rn = CRNetwork.from_string(["Analyte -> Captured"],
                                   rates={"Analyte -> Captured": k_cap})

        rng = np.random.default_rng(seed)
        mean_signal = n_total * p_cap
        out = np.empty(n_trajectories)
        for i in range(n_trajectories):
            # the aliquot population itself is Poisson — the dominant shot noise
            n_start = int(rng.poisson(n_total))
            if n_start == 0:
                out[i] = 0.0
                continue
            res = rn.stochastic_simulate(
                {"Analyte": n_start, "Captured": 0},
                (0.0, duration), volume_L=self.params.sample_volume_L,
                initial_as="count", seed=int(rng.integers(0, 2**31 - 1)),
            )
            out[i] = res.counts["Captured"][-1]
        return CaptureSamples(out, mean_signal, n_total, p_cap)
