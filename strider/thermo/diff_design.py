"""
[EXPERIMENTAL API — outside the semver contract until stabilized; see README
"API stability and versioning".]

Differentiable design objectives for gradient-based sequence design.

These are the design-time counterparts of the discrete
:class:`strider.design.objective.DesignObjective` terms, but they consume a
*soft* sequence — a ``(B, N, 4)`` tensor of per-position base distributions over
(A, C, G, U/T) — and return a per-batch loss tensor that is differentiable w.r.t.
that sequence.  They are built on the autodiff base-pair-probability matrix in
:mod:`strider.thermo.differentiable`, so structural objectives (ensemble defect,
toehold accessibility, pairing entropy) and energetic objectives (ΔG / ΔΔG
targets) can all be minimized by gradient descent.

A :class:`DiffObjective` composes terms exactly like ``DesignObjective``
(weighted sum, ``+`` and ``*`` operators, ``evaluate_breakdown``), so a design
spec reads the same on the differentiable side as on the simulated-annealing
side.  Each evaluation folds the sequence once and shares the resulting free
energy / BPP across every term via a small per-call cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

from strider.structure.dot_bracket import parse_pairs
from strider.thermo.differentiable import (
    ThermoParameters,
    BatchedPartitionFunction,
)

# Per-position base order used everywhere in the differentiable engine.
_BASE_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
TermFn = Callable[[torch.Tensor, "EvalCache"], torch.Tensor]


class EvalCache:
    """Lazily-computed, shared per-evaluation fold of a soft sequence.

    The free energy and the base-pair-probability matrix both come from a single
    DP pass (:meth:`BatchedPartitionFunction.soft_free_energy_and_bpp`); the cache
    makes sure an objective with several BPP-based terms folds only once.
    """

    def __init__(self, model: BatchedPartitionFunction, probs: torch.Tensor,
                 nicks: list[int] | None = None):
        self.model = model
        self.probs = probs
        self.nicks = nicks
        self._fe: torch.Tensor | None = None
        self._bpp: torch.Tensor | None = None

    def _ensure(self) -> None:
        if self._fe is None:
            self._fe, self._bpp = self.model.soft_free_energy_and_bpp(
                self.probs, nicks=self.nicks)

    @property
    def free_energy(self) -> torch.Tensor:
        self._ensure()
        return self._fe

    @property
    def bpp(self) -> torch.Tensor:
        self._ensure()
        return self._bpp

    def paired_prob(self) -> torch.Tensor:
        """Per-position total pairing probability ``q_i = Σ_j P(i, j)`` (B, N)."""
        return self.bpp.sum(dim=-1).clamp(max=1.0)


def _model(params: ThermoParameters | None, material: str) -> BatchedPartitionFunction:
    if params is None:
        params = ThermoParameters(material=material)
    return BatchedPartitionFunction(params)


@dataclass
class DiffObjective:
    """
    A weighted, composable *differentiable* design objective.

    Terms are summed: ``loss = Σ weight_i · term_i(probs, cache)`` and the result
    is a ``(B,)`` tensor (one loss per parallel design in the batch).  Lower is
    better.  Mirrors :class:`strider.design.objective.DesignObjective` so the two
    designers share a mental model.
    """

    _terms: list[tuple[float, TermFn]] = field(default_factory=list, repr=False)
    _labels: list[str] = field(default_factory=list, repr=False)
    _model: BatchedPartitionFunction | None = field(default=None, repr=False)

    def bind(self, model: BatchedPartitionFunction) -> "DiffObjective":
        """Attach the fold engine used to build the per-call cache."""
        self._model = model
        return self

    def __call__(self, probs: torch.Tensor,
                 model: BatchedPartitionFunction | None = None,
                 nicks: list[int] | None = None) -> torch.Tensor:
        mdl = model or self._model
        if mdl is None:
            raise ValueError("DiffObjective needs a bound model; call .bind(model).")
        cache = EvalCache(mdl, probs, nicks=nicks)
        total = torch.zeros(probs.shape[0], dtype=probs.dtype, device=probs.device)
        for w, fn in self._terms:
            total = total + w * fn(probs, cache)
        return total

    def evaluate_breakdown(self, probs: torch.Tensor,
                           model: BatchedPartitionFunction | None = None,
                           nicks: list[int] | None = None
                           ) -> dict[str, float]:
        """Per-term mean score (over the batch) keyed by label, for diagnostics."""
        mdl = model or self._model
        cache = EvalCache(mdl, probs, nicks=nicks)
        out: dict[str, float] = {}
        for label, (w, fn) in zip(self._labels, self._terms):
            out[label] = float((w * fn(probs, cache)).mean().detach())
        return out

    def __add__(self, other: "DiffObjective") -> "DiffObjective":
        obj = DiffObjective()
        obj._terms = self._terms + other._terms
        obj._labels = self._labels + other._labels
        obj._model = self._model or other._model
        return obj

    def __mul__(self, weight: float) -> "DiffObjective":
        obj = DiffObjective()
        obj._terms = [(w * weight, fn) for w, fn in self._terms]
        obj._labels = self._labels[:]
        obj._model = self._model
        return obj

    __rmul__ = __mul__

    def _single(self, fn: TermFn, weight: float, label: str) -> "DiffObjective":
        self._terms = [(weight, fn)]
        self._labels = [label]
        return self

    # ─── structural objectives (built on the autodiff BPP) ───────────────────

    @classmethod
    def ensemble_defect(cls, target_structure: str, weight: float = 1.0,
                        normalize: bool = True, label: str | None = None,
                        model: BatchedPartitionFunction | None = None) -> "DiffObjective":
        """
        Differentiable normalized ensemble defect of a target dot-bracket.

        Same definition as :meth:`strider.thermo.engine.ThermoEngine.ensemble_defect`
        — ``Σ_i (1 − P_correct(i))`` with ``P_correct(i) = P(i, j*)`` for a target
        pair ``(i, j*)`` and ``1 − Σ_j P(i, j)`` for a target-unpaired position —
        evaluated on the soft BPP, so its gradient flows to the sequence.  This is
        the canonical inverse-folding loss.
        """
        n = len(target_structure.replace("&", "").replace("+", ""))
        partner = list(range(n))            # partner[i] = j (self if unpaired in target)
        is_paired = [False] * n
        for i, j in parse_pairs(target_structure):
            partner[i], partner[j] = j, i
            is_paired[i] = is_paired[j] = True
        lbl = label or "ensemble_defect"

        def fn(probs: torch.Tensor, cache: EvalCache) -> torch.Tensor:
            bpp = cache.bpp                               # (B, N, N)
            B, N, _ = bpp.shape
            q = bpp.sum(dim=-1).clamp(max=1.0)            # (B, N) total pairing prob
            # P(i, partner_i) for every position, via a single gather (no in-place).
            idx = torch.tensor(partner[:N], device=bpp.device).view(1, N, 1).expand(B, N, 1)
            partner_prob = bpp.gather(2, idx).squeeze(-1)  # (B, N)
            paired = torch.tensor(is_paired[:N], device=bpp.device, dtype=torch.bool)
            p_correct = torch.where(paired.unsqueeze(0), partner_prob, 1.0 - q)
            defect = (1.0 - p_correct).sum(dim=-1)
            return defect / n if normalize else defect

        return cls()._bind(model)._single(fn, weight, lbl)

    @classmethod
    def accessibility(cls, positions: list[int], min_prob: float = 0.8,
                      weight: float = 1.0, label: str | None = None,
                      model: BatchedPartitionFunction | None = None) -> "DiffObjective":
        """
        Penalize low accessibility of a window (e.g. a toehold).

        Accessibility = ``Π_{i∈positions} (1 − Σ_j P(i, j))`` (the independent-site
        unpaired probability used by
        :meth:`ThermoEngine.toehold_accessibility`).  Score = ``max(0, min_prob −
        accessibility)²`` so the optimizer is rewarded only up to the target.
        """
        lbl = label or f"accessibility(pos={positions[:3]}...)"
        pos = list(positions)

        def fn(probs: torch.Tensor, cache: EvalCache) -> torch.Tensor:
            q = cache.paired_prob()                       # (B, N)
            unpaired = (1.0 - q[:, pos]).clamp(min=1e-9)
            access = unpaired.prod(dim=-1)                # (B,)
            shortfall = (min_prob - access).clamp(min=0.0)
            return shortfall ** 2

        return cls()._bind(model)._single(fn, weight, lbl)

    @classmethod
    def pairing_entropy(cls, positions: list[int] | None = None,
                        maximize: bool = False, weight: float = 1.0,
                        label: str | None = None,
                        model: BatchedPartitionFunction | None = None) -> "DiffObjective":
        """
        Positional Shannon entropy of the pairing distribution.

        For each position the distribution is ``{P(i, j)}_j ∪ {1 − Σ_j P(i, j)}``
        (pair-with-each-partner plus stay-unpaired).  ``maximize=False`` (default)
        *minimizes* entropy — i.e. drives positions toward a single well-defined
        state (a crisp, low-ambiguity structure); ``maximize=True`` negates the
        sign to reward disorder.  Averaged over ``positions`` (all if ``None``).
        """
        lbl = label or ("pairing_entropy" + ("(max)" if maximize else "(min)"))
        sign = -1.0 if maximize else 1.0

        def fn(probs: torch.Tensor, cache: EvalCache) -> torch.Tensor:
            bpp = cache.bpp                               # (B, N, N)
            B, N, _ = bpp.shape
            q = bpp.sum(dim=-1, keepdim=True).clamp(max=1.0)
            dist = torch.cat([bpp, (1.0 - q).clamp(min=0.0)], dim=-1)  # (B, N, N+1)
            ent = -(dist * (dist + 1e-12).log()).sum(dim=-1)          # (B, N)
            idx = list(range(N)) if positions is None else [p for p in positions if p < N]
            ent = ent[:, idx]
            return sign * ent.mean(dim=-1)

        return cls()._bind(model)._single(fn, weight, lbl)

    # ─── energetic objectives ────────────────────────────────────────────────

    @classmethod
    def free_energy_target(cls, target_dg: float, weight: float = 1.0,
                           label: str | None = None,
                           model: BatchedPartitionFunction | None = None) -> "DiffObjective":
        """Penalize ``(ΔG_ens − target_dg)²`` (kcal/mol).  Hairpin / stem stability."""
        lbl = label or f"free_energy_target({target_dg:.1f})"

        def fn(probs: torch.Tensor, cache: EvalCache) -> torch.Tensor:
            return (cache.free_energy - target_dg) ** 2

        return cls()._bind(model)._single(fn, weight, lbl)

    @classmethod
    def free_energy_range(cls, min_dg: float, max_dg: float, weight: float = 1.0,
                          label: str | None = None,
                          model: BatchedPartitionFunction | None = None) -> "DiffObjective":
        """Penalize ΔG_ens outside ``[min_dg, max_dg]`` (zero inside the band)."""
        lbl = label or f"free_energy_range({min_dg:.1f},{max_dg:.1f})"

        def fn(probs: torch.Tensor, cache: EvalCache) -> torch.Tensor:
            fe = cache.free_energy
            below = (min_dg - fe).clamp(min=0.0)
            above = (fe - max_dg).clamp(min=0.0)
            return below ** 2 + above ** 2

        return cls()._bind(model)._single(fn, weight, lbl)

    # ─── composition / sequence constraints ──────────────────────────────────

    @classmethod
    def gc_content(cls, target_gc: float = 0.5, weight: float = 1.0,
                   label: str | None = None) -> "DiffObjective":
        """Penalize ``(expected_GC − target_gc)²`` (no fold needed)."""
        lbl = label or f"gc_content({target_gc:.0%})"

        def fn(probs: torch.Tensor, cache: EvalCache) -> torch.Tensor:
            gc = probs[:, :, 1] + probs[:, :, 2]          # P(C) + P(G) per position
            return (gc.mean(dim=-1) - target_gc) ** 2

        return cls()._single(fn, weight, lbl)

    @classmethod
    def gc_band(cls, lo: float, hi: float, weight: float = 1.0,
                label: str | None = None) -> "DiffObjective":
        """Penalize expected GC content outside ``[lo, hi]`` (mirrors gc_band)."""
        lbl = label or f"gc_band({lo:.0%},{hi:.0%})"

        def fn(probs: torch.Tensor, cache: EvalCache) -> torch.Tensor:
            gc = (probs[:, :, 1] + probs[:, :, 2]).mean(dim=-1)
            return (lo - gc).clamp(min=0.0) ** 2 + (gc - hi).clamp(min=0.0) ** 2

        return cls()._single(fn, weight, lbl)

    @classmethod
    def forbidden_motifs(cls, motifs: list[str], weight: float = 1.0,
                         label: str | None = None) -> "DiffObjective":
        """
        Penalize the expected count of forbidden subsequences.

        For each motif and each start offset the expected indicator is the product
        of per-position base probabilities (independent-site model); summing over
        offsets gives the expected number of occurrences, which the optimizer is
        pushed toward zero.  Useful for runs like ``GGGG`` or restriction sites.
        """
        lbl = label or f"forbidden_motifs({len(motifs)})"
        coded = [[_BASE_IDX[b] for b in m.upper()] for m in motifs]

        def fn(probs: torch.Tensor, cache: EvalCache) -> torch.Tensor:
            B, N, _ = probs.shape
            total = torch.zeros(B, dtype=probs.dtype, device=probs.device)
            for code in coded:
                L = len(code)
                for s in range(0, N - L + 1):
                    p = torch.ones(B, dtype=probs.dtype, device=probs.device)
                    for k, c in enumerate(code):
                        p = p * probs[:, s + k, c]
                    total = total + p
            return total

        return cls()._single(fn, weight, lbl)

    @classmethod
    def from_term(cls, fn: TermFn, weight: float = 1.0,
                  label: str = "custom") -> "DiffObjective":
        """Wrap any ``(probs, cache) -> (B,)`` callable as a term."""
        return cls()._single(fn, weight, label)

    def _bind(self, model: BatchedPartitionFunction | None) -> "DiffObjective":
        if model is not None:
            self._model = model
        return self

    def __repr__(self) -> str:
        return f"DiffObjective({', '.join(self._labels)})"
