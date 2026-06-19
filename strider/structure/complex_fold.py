"""
Order-invariant minimum-free-energy folding of a multi-strand complex.

The linear nick-aware Zuker DP in :mod:`strider.structure.mfe` folds the
strands *in the concatenation order the caller supplied*.  The set of
structures it can represent is exactly those that are non-crossing for that
one linear cut of the strand circle — so the predicted MFE of a complex
**changes if you relabel the strand order**, even though the real equilibrium
of a complex depends only on which strands are present, not how a human writes
them down (Dirks, Bois, Schaeffer, Winfree & Pierce, "Thermodynamic Analysis
of Interacting Nucleic Acid Strands," SIAM Review 49(1):65-88, 2007).

This module removes that artifact for the MFE: it folds every distinct cyclic
arrangement (every distinct concatenation of the strand multiset) with the
existing linear kernel and returns the global minimum over a structure that
**connects all strands** (a defined ordered complex must be connected; an
arrangement that leaves a strand unpaired describes a different, lower-order
species — complex + free monomer — and is rejected).

The rotational-symmetry correction σ for homomeric complexes is *not* applied
here; it is a per-complex additive constant (``R·T·ln σ``) handled once at the
engine layer (:meth:`strider.thermo.engine.ThermoEngine._mfe_sigma_correction`),
so it composes with this search without double counting.

References
----------
Dirks R.M., Bois J.S., Schaeffer J.M., Winfree E., Pierce N.A. (2007).
    Thermodynamic analysis of interacting nucleic acid strands. SIAM Review
    49: 65-88.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from strider.structure.mfe import fold_mfe

# Default cap on strand count.  The number of distinct concatenations grows as
# N! (with deduplication for repeated strands), so an exhaustive search becomes
# expensive fast: N=6 → 720 folds, N=7 → 5040.  Above ``DEFAULT_MAX_EXHAUSTIVE``
# distinct orders we switch from exhaustive enumeration to a crossing-
# minimisation heuristic (a handful of folds); ``max_strands`` only refuses
# truly unbounded inputs.
DEFAULT_MAX_STRANDS = 12
DEFAULT_MAX_EXHAUSTIVE = 24  # ≤ 4 distinct strands → fold every cut exactly

# Exhaustive search folds every distinct cut, so its cost is
# ``n_orders × O(total_len³)``.  To keep a deliberate multi-strand MFE call
# bounded (~15-20 s worst case) we only enumerate exhaustively when the order
# count fits a length-scaled budget; larger complexes use the heuristic.  The
# budget is calibrated as "≈ this many folds at 100 nt" and scales as len³.
EXHAUSTIVE_FOLD_BUDGET = 30.0

INF = float("inf")


def _use_exhaustive(n_orders: int, total_len: int, max_exhaustive: int) -> bool:
    """Whether to fold every cut (exact, order-invariant) vs. the heuristic.

    Exhaustive only when the order count is within both the hard cap
    ``max_exhaustive`` and a length-scaled fold budget, so the exact path stays
    fast on small complexes (dimers, trimers, small 4-strand) and large complexes
    fall back to the cheap heuristic.
    """
    per_fold = max((total_len / 100.0) ** 3, 1e-3)
    budget = max(2, int(EXHAUSTIVE_FOLD_BUDGET / per_fold))
    return n_orders <= min(max_exhaustive, budget)


@dataclass
class ComplexFold:
    """Result of an order-invariant complex fold.

    Attributes
    ----------
    energy    : MFE in kcal/mol (without the σ correction, which the engine adds).
    structure : dot-bracket string with ``'&'`` separators, in ``order``.
    pairs     : 0-indexed base pairs over the concatenated sequence in ``order``.
    order     : the strand permutation (indices into the *input* strand list)
                that achieves the MFE.  Identity if the input order was already
                optimal.
    connected : True if the returned structure connects all strands.  False only
                when *no* arrangement connects every strand (the strands do not
                all bind); the lowest-energy arrangement is returned anyway.
    n_orders  : number of distinct concatenations evaluated.
    evaluated : ``(order, per-order MFE)`` for every arrangement folded.  Exposed
                so an order-invariant suboptimal enumeration can reuse exactly the
                arrangements the MFE search considered (keeping ``subopt`` and
                ``mfe`` consistent).
    """

    energy: float
    structure: str
    pairs: list[tuple[int, int]]
    order: tuple[int, ...]
    connected: bool
    n_orders: int
    evaluated: tuple[tuple[tuple[int, ...], float], ...] = ()


def _nicks_for(strand_lens: list[int]) -> list[int]:
    """Concatenation positions immediately after each strand (excluding the last)."""
    nicks: list[int] = []
    pos = 0
    for length in strand_lens[:-1]:
        pos += length
        nicks.append(pos)
    return nicks


def _strand_of(pos: int, nicks: list[int]) -> int:
    """Strand index (0-based) of concatenated position ``pos`` given ``nicks``."""
    s = 0
    for nk in nicks:
        if pos >= nk:
            s += 1
        else:
            break
    return s


def is_connected(
    pairs: list[tuple[int, int]], strand_lens: list[int]
) -> bool:
    """True if the base-pair graph connects every strand into one component.

    ``strand_lens`` are the strand lengths in the *folded* (concatenation)
    order; ``pairs`` are 0-indexed over that concatenation.  A single strand is
    trivially connected.
    """
    n_strands = len(strand_lens)
    if n_strands <= 1:
        return True
    nicks = _nicks_for(strand_lens)
    parent = list(range(n_strands))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        si, sj = _strand_of(i, nicks), _strand_of(j, nicks)
        if si != sj:
            parent[find(si)] = find(sj)

    root = find(0)
    return all(find(s) == root for s in range(n_strands))


def distinct_orders(strands: list[str]) -> list[tuple[int, ...]]:
    """Distinct strand permutations, deduplicated by resulting concatenation.

    Each permutation is a tuple of indices into ``strands`` (new slot ``k`` holds
    input strand ``perm[k]``).  Two permutations that produce the same
    concatenated string (because some strands are identical) are collapsed to
    the first one seen, so a homomeric complex is not folded redundantly.

    Every distinct *cut* is kept (we do not deduplicate by rotation): the linear
    kernel is not cut-invariant — different cuts of the same strand circle expose
    different non-crossing structure sets — so each must be folded to find the
    true minimum.
    """
    seen: set[str] = set()
    out: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(strands))):
        concat = "\x00".join(strands[i] for i in perm)
        if concat in seen:
            continue
        seen.add(concat)
        out.append(perm)
    return out


def _fold_one(
    strands: list[str],
    order: tuple[int, ...],
    celsius: float,
    material: str,
    sodium_M: float,
    magnesium_M: float,
) -> ComplexFold:
    """Fold a single arrangement ``order`` of ``strands`` with the linear kernel."""
    ordered = [strands[i] for i in order]
    structure, energy, pairs = fold_mfe(
        "&".join(ordered), celsius, material, sodium_M, magnesium_M
    )
    connected = is_connected(pairs, [len(s) for s in ordered])
    return ComplexFold(energy, structure, pairs, order, connected, 0)


def _select(folds: list[ComplexFold], n_orders: int) -> ComplexFold:
    """Pick the lowest-energy *connected* fold, falling back to the global min.

    A defined ordered complex must be connected; if no arrangement connects all
    strands (the strands do not all bind) the lowest-energy arrangement is
    returned with ``connected=False``.  Ties keep the earliest fold (so the
    caller's input order wins when it already achieves the minimum).
    """
    best = None
    best_any = None
    for cf in folds:
        if best_any is None or cf.energy < best_any.energy:
            best_any = cf
        if cf.connected and (best is None or cf.energy < best.energy):
            best = cf
    chosen = best if best is not None else best_any
    assert chosen is not None
    evaluated = tuple((cf.order, cf.energy) for cf in folds)
    return ComplexFold(
        chosen.energy, chosen.structure, chosen.pairs, chosen.order,
        chosen.connected, n_orders, evaluated,
    )


def _affinity_matrix(strands: list[str], k: int = 6) -> list[list[int]]:
    """Pairwise inter-strand complementarity score (reverse-complement k-mer hits).

    ``W[i][j]`` counts the distinct length-``k`` substrings of strand ``i`` whose
    reverse complement occurs in strand ``j`` — a cheap, fold-free proxy for how
    strongly two strands hybridise.  Used to seed the heuristic order search with
    a high-affinity arrangement (neighbours that bind sit adjacent), which
    recovers the intended order of a closed network from *any* input order
    without first needing a good fold.
    """
    comp = str.maketrans("ACGT", "TGCA")
    norm = [s.upper().replace("U", "T") for s in strands]
    kmers = [
        {s[p:p + k] for p in range(len(s) - k + 1)} for s in norm
    ]
    n = len(strands)
    W = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cnt = sum(1 for km in kmers[i] if km.translate(comp)[::-1] in kmers[j])
            W[i][j] = W[j][i] = cnt
    return W


def _affinity_order(strands: list[str]) -> tuple[int, ...]:
    """Greedy max-affinity chain over :func:`_affinity_matrix` (a candidate order).

    Seeds with the most strongly hybridising strand pair and extends whichever
    chain end gains the most affinity — the same greedy chain heuristic used by
    :func:`strider.viz.geometry._greedy_strand_order`, but driven by sequence
    complementarity instead of a fold's base pairs.
    """
    from collections import deque

    n = len(strands)
    W = _affinity_matrix(strands)

    si, sj, best = 0, 1, -1
    for i in range(n):
        for j in range(i + 1, n):
            if W[i][j] > best:
                si, sj, best = i, j, W[i][j]
    if best <= 0:
        return tuple(range(n))

    placed = [False] * n
    chain = deque([si, sj])
    placed[si] = placed[sj] = True
    while len(chain) < n:
        left, right = chain[0], chain[-1]
        pick, pw, end = None, -1, "right"
        for kk in range(n):
            if placed[kk]:
                continue
            if W[left][kk] > pw:
                pick, pw, end = kk, W[left][kk], "left"
            if W[right][kk] > pw:
                pick, pw, end = kk, W[right][kk], "right"
        if pick is None:
            pick = next(kk for kk in range(n) if not placed[kk])
            end = "right"
        (chain.appendleft if end == "left" else chain.append)(pick)
        placed[pick] = True
    return tuple(chain)


def fold_complex(
    strands: list[str],
    celsius: float = 37.0,
    material: str = "dna",
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    max_strands: int = DEFAULT_MAX_STRANDS,
    max_exhaustive: int = DEFAULT_MAX_EXHAUSTIVE,
) -> ComplexFold:
    """Order-invariant MFE fold of a multi-strand complex.

    Returns the global minimum over strand arrangements of the linear nick-aware
    kernel (:func:`strider.structure.mfe.fold_mfe`), restricted to arrangements
    whose structure connects all strands.

    Two regimes:

    * **Exhaustive** (order count within ``max_exhaustive`` *and* a length-scaled
      fold budget — see :func:`_use_exhaustive`): every cut is folded, so the
      result is the exact order-invariant MFE, invariant to any permutation of
      ``strands``.  Covers the common cases (dimers, trimers, small 4-strand).
    * **Heuristic** (more orders, e.g. a large 5-/6-strand network): folds two
      seeds — the input order and a sequence-complementarity seed
      (:func:`_affinity_order`, which places strands that hybridise adjacent) —
      then follows :func:`strider.viz.geometry.best_strand_order` (minimise
      pseudoknot crossings) from each to a fixpoint.  A handful of folds instead
      of N!.  The affinity seed recovers a closed network's intended ring from
      *any* input order, so the result is order-invariant in practice for the
      networks this targets (e.g. the dendrimer), though not guaranteed exact.

    The returned ``structure``/``pairs`` are expressed in the winning ``order`` —
    a structure that is nested in that order may be pseudoknotted in another, so
    it cannot in general be rendered as a dot-bracket in the caller's order.

    Raises
    ------
    ValueError
        If ``len(strands)`` exceeds ``max_strands``.
    """
    if not strands:
        raise ValueError("fold_complex requires at least one strand")
    n = len(strands)
    if n > max_strands:
        raise ValueError(
            f"complex has {n} strands; exceeds max_strands={max_strands}."
        )

    if n == 1:
        structure, energy, pairs = fold_mfe(
            strands[0], celsius, material, sodium_M, magnesium_M
        )
        return ComplexFold(energy, structure, pairs, (0,), True, 1, (((0,), energy),))

    orders = distinct_orders(strands)
    total_len = sum(len(s) for s in strands)

    if _use_exhaustive(len(orders), total_len, max_exhaustive):
        folds = [
            _fold_one(strands, o, celsius, material, sodium_M, magnesium_M)
            for o in orders
        ]
        return _select(folds, len(orders))

    # Heuristic: from each seed (input order + sequence-affinity order), follow
    # crossing minimisation to a fixpoint.
    from strider.viz.geometry import best_strand_order

    folds: list[ComplexFold] = []
    seen: set[tuple[int, ...]] = set()
    seeds = [tuple(range(n)), _affinity_order(strands)]
    for seed in seeds:
        current = seed
        for _ in range(n + 2):  # bounded; converges well before this
            if current in seen:
                break
            seen.add(current)
            cf = _fold_one(strands, current, celsius, material, sodium_M, magnesium_M)
            folds.append(cf)
            strand_lens = [len(strands[i]) for i in current]
            q = best_strand_order(cf.pairs, strand_lens)
            nxt = tuple(current[q[k]] for k in range(n))
            if nxt == current:
                break
            current = nxt

    return _select(folds, len(folds))
