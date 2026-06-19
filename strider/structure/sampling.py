"""
Boltzmann sampling and suboptimal-structure enumeration.

Both procedures build on the McCaskill partition-function DP in
``strider.thermo.ensemble`` and the MFE DP in ``strider.structure.mfe``.

* ``sample_structures(seq, n, …)`` — Ding-Lawrence (2003) stochastic
  traceback over the Qb / Q matrices, yielding N structures distributed
  according to the equilibrium Boltzmann weights.

* ``subopt_structures(seq, gap, …)`` — Wuchty-style enumeration of all
  structures within ``gap`` kcal/mol of the MFE.  Uses a worklist of partial
  decompositions over the V / W matrices, pruned by a lower-bound energy
  estimate.

References
----------
Wuchty S., Fontana W., Hofacker I.L., Schuster P. (1999) Biopolymers
49:145-165.

Ding Y. & Lawrence C.E. (2003) Nucleic Acids Res. 31:7280-7301.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np

from strider.thermo.ensemble import (
    R, _MAX_IL, _wc_pairs, _can_pair_nicks,
    _hairpin_loop_energy, _stack_energy, _interior_bulge_energy,
    _terminal_pair_penalty, _boltzmann, _fill_dp_nicks,
    _apply_coaxial_external,
)
from strider.structure.dot_bracket import to_dot_bracket

if TYPE_CHECKING:
    pass


INF = float("inf")


# ─── Boltzmann sampling ───────────────────────────────────────────────────────

def sample_structures(
    sequence: str,
    n_samples: int,
    celsius: float = 37.0,
    material: str = "dna",
    seed: int | None = None,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
) -> list[tuple[str, list[tuple[int, int]]]]:
    """
    Draw ``n_samples`` structures from the equilibrium ensemble.

    Returns a list of ``(dot_bracket, pair_list)`` tuples, each entry sampled
    independently with probability ∝ exp(−E / RT).

    Salt correction: the per-closed-base-pair Boltzmann factor of
    :func:`strider.thermo.salt.dg_per_bp_salt` is folded into ``Qb`` (exactly as
    :func:`strider.thermo.ensemble.ensemble_dg` does), so the sampling weights
    track [Na⁺]/[Mg²⁺].  At 1 M Na⁺, 0 Mg²⁺ the factor is 1.0 and the
    distribution is unchanged.
    """
    from strider.thermo.salt import dg_per_bp_salt
    rng = random.Random(seed)
    seq = sequence.upper().replace("U", "T") if material == "dna" else sequence.upper().replace("T", "U")
    n = len(seq)
    T = celsius + 273.15
    pairs = _wc_pairs(material)
    bp_salt_factor = _boltzmann(dg_per_bp_salt(sodium_M, magnesium_M, celsius, material), T)

    Q  = np.zeros((n, n))
    Qb = np.zeros((n, n))
    QM = np.zeros((n, n))
    QM1 = np.zeros((n, n))
    for i in range(n):
        Q[i][i] = 1.0
    for i in range(n - 1):
        Q[i][i + 1] = 1.0

    _fill_dp_nicks(seq, Q, Qb, QM, QM1, n, T, pairs, material, nicks=[],
                   bp_salt_factor=bp_salt_factor)
    _apply_coaxial_external(seq, Q, Qb, n, T, material)

    results = []
    for _ in range(n_samples):
        pair_list: list[tuple[int, int]] = []
        _sample_Q(seq, 0, n - 1, Q, Qb, QM, QM1, T, pairs, material, pair_list, rng)
        pair_list.sort()
        results.append((to_dot_bracket(pair_list, n), pair_list))
    return results


def _sample_Q(seq, i, j, Q, Qb, QM, QM1, T, pairs, material, out_pairs, rng):
    """Stochastic traceback through Q[i][j] (external context)."""
    while i <= j:
        # Compute contributions: (j unpaired) + Σ_k (stem (k,j) with various dangles)
        total = Q[i][j]
        if total <= 0:
            return
        target = rng.random() * total
        cum = 0.0

        # Option A: j unpaired → recurse on [i, j-1]
        contrib = Q[i][j - 1] if j > i else 1.0
        cum += contrib
        if target < cum:
            j -= 1
            continue

        # Option B: close (k, j) with no dangles
        chosen = False
        for k in range(i, j + 1):
            if Qb[k][j] <= 0:
                continue
            left = Q[i][k - 1] if k > i else 1.0
            contrib = left * Qb[k][j]
            cum += contrib
            if target < cum:
                out_pairs.append((k, j))
                _sample_Qb(seq, k, j, Q, Qb, QM, QM1, T, pairs, material, out_pairs, rng)
                # Continue with left subproblem [i..k-1]
                _sample_Q(seq, i, k - 1, Q, Qb, QM, QM1, T, pairs, material, out_pairs, rng)
                chosen = True
                break
        if chosen:
            return
        # Numerical edge case: fall through with j unpaired
        j -= 1


def _ml_factors(material, T):
    from strider.thermo._param_context import lookup_scalar
    if material == "dna":
        from strider.thermo.parameters_dna import ML_INIT, ML_PAIR, ML_BASE
    else:
        from strider.thermo.parameters_rna import ML_INIT, ML_PAIR, ML_BASE
    ML_INIT = lookup_scalar("multiloop_init", float(ML_INIT))
    ML_PAIR = lookup_scalar("multiloop_pair", float(ML_PAIR))
    ML_BASE = lookup_scalar("multiloop_base", float(ML_BASE))
    return (_boltzmann(ML_INIT + ML_PAIR, T), _boltzmann(ML_PAIR, T),
            _boltzmann(ML_BASE, T))


def _sample_Qb(seq, i, j, Q, Qb, QM, QM1, T, pairs, material, out_pairs, rng):
    """Stochastic traceback through Qb[i][j] (forced pair at i, j)."""
    if Qb[i][j] <= 0:
        return
    target = rng.random() * Qb[i][j]
    cum = 0.0

    # 1. Hairpin
    cum += _boltzmann(_hairpin_loop_energy(seq, i, j, material, T), T)
    if target < cum:
        return

    # 2. Stack
    if _can_pair_nicks(seq, i + 1, j - 1, pairs, []) and Qb[i + 1][j - 1] > 0:
        cum += _boltzmann(_stack_energy(seq, i, j, material), T) * Qb[i + 1][j - 1]
        if target < cum:
            out_pairs.append((i + 1, j - 1))
            _sample_Qb(seq, i + 1, j - 1, Q, Qb, QM, QM1, T, pairs, material, out_pairs, rng)
            return

    # 3. Interior loop / bulge
    for nl in range(_MAX_IL + 1):
        ip = i + nl + 1
        if ip > j - 2:
            break
        for nr in range(_MAX_IL - nl + 1):
            if nl == 0 and nr == 0:
                continue
            jp = j - nr - 1
            if jp <= ip:
                break
            if not _can_pair_nicks(seq, ip, jp, pairs, []):
                continue
            if Qb[ip][jp] <= 0:
                continue
            dG = _interior_bulge_energy(seq, i, j, ip, jp, nl, nr, material)
            cum += Qb[ip][jp] * _boltzmann(dG, T)
            if target < cum:
                out_pairs.append((ip, jp))
                _sample_Qb(seq, ip, jp, Q, Qb, QM, QM1, T, pairs, material, out_pairs, rng)
                return

    # 4. Multiloop closed by (i, j) — ≥2 branches: QM[i+1][k-1]·QM1[k][j-1]
    bm_ml_init_pair, _, _ = _ml_factors(material, T)
    if j - i > 2:
        for k in range(i + 2, j):
            qm_left = QM[i + 1][k - 1]
            if qm_left <= 0:
                continue
            q1 = QM1[k][j - 1]
            if q1 <= 0:
                continue
            cum += bm_ml_init_pair * qm_left * q1
            if target < cum:
                _sample_QM(seq, i + 1, k - 1, Q, Qb, QM, QM1, T, pairs, material, out_pairs, rng)
                _sample_QM1(seq, k, j - 1, Q, Qb, QM, QM1, T, pairs, material, out_pairs, rng)
                return


def _sample_QM1(seq, i, j, Q, Qb, QM, QM1, T, pairs, material, out_pairs, rng):
    """Traceback through QM1[i][j]: one branch block — stem at i + trailing unpaired."""
    _, bm_ml_pair, bm_ml_base = _ml_factors(material, T)
    while i <= j:
        total = QM1[i][j]
        if total <= 0:
            return
        target = rng.random() * total
        cum = 0.0
        # stem ends exactly at j
        if Qb[i][j] > 0:
            cum += Qb[i][j] * bm_ml_pair
            if target < cum:
                out_pairs.append((i, j))
                _sample_Qb(seq, i, j, Q, Qb, QM, QM1, T, pairs, material, out_pairs, rng)
                return
        # j unpaired (trailing)
        if j > i and QM1[i][j - 1] > 0:
            j -= 1
            continue
        return


def _sample_QM(seq, i, j, Q, Qb, QM, QM1, T, pairs, material, out_pairs, rng):
    """Traceback through QM[i][j]: ≥1 branch block, leading unpaired allowed."""
    _, _, bm_ml_base = _ml_factors(material, T)
    total = QM[i][j]
    if total <= 0:
        return
    target = rng.random() * total
    cum = 0.0
    # QM[i][j] = Σ_k (b_base^(k-i) + QM[i][k-1]) · QM1[k][j]
    for k in range(i, j):
        q1 = QM1[k][j]
        if q1 <= 0:
            continue
        lead = bm_ml_base ** (k - i)
        prev = QM[i][k - 1] if k > i else 0.0
        contrib = (lead + prev) * q1
        cum += contrib
        if target < cum:
            # decide: leading-unpaired-only (i..k-1 unpaired) vs earlier branches
            if k > i and rng.random() * (lead + prev) >= lead:
                _sample_QM(seq, i, k - 1, Q, Qb, QM, QM1, T, pairs, material, out_pairs, rng)
            _sample_QM1(seq, k, j, Q, Qb, QM, QM1, T, pairs, material, out_pairs, rng)
            return


# ─── Suboptimal structures ────────────────────────────────────────────────────

def subopt_structures(
    sequence: str,
    gap: float = 1.0,
    celsius: float = 37.0,
    material: str = "dna",
    max_structures: int = 200,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
) -> list[tuple[str, float, list[tuple[int, int]]]]:
    """
    Enumerate suboptimal structures within ``gap`` kcal/mol of the MFE.

    Single-strand or multi-strand: pass ``'&'`` / ``'+'`` separated strands
    (e.g. ``"AAAA&TTTT"``) to enumerate dimer / complex structures.

    Returns ``(dot_bracket, energy, pair_list)`` sorted by energy, capped at
    ``max_structures`` results.  ``dot_bracket`` carries strand separators;
    ``pair_list`` is 0-indexed over the concatenated sequence (separators
    removed), matching :func:`strider.structure.mfe.fold_mfe`.

    Implementation: a Wuchty-style enumeration over the *same* nick-aware
    Zuker–Stiegler V/W/WM/WM1 matrices that :func:`fold_mfe` uses (built via
    :func:`strider.structure.mfe._build_mfe_matrices`), so the lowest-energy
    structure returned here is identical to ``fold_mfe`` — hairpins, internal
    loops/bulges, multiloops, inter-strand pairs and the salt correction are all
    accounted for.  Decompositions are pruned against the matrix lower bounds
    and the ``mfe + gap`` window; total enumeration work is capped so a large
    ``gap`` on a long sequence cannot hang.

    Salt correction: each closed base pair contributes the per-base-pair ΔG
    shift of :func:`strider.thermo.salt.dg_per_bp_salt` (baked into the shared
    matrices), so subopt energies and the enumeration window track
    [Na⁺]/[Mg²⁺].  At 1 M Na⁺, 0 Mg²⁺ the correction is exactly 0.
    """
    from strider.structure.mfe import (
        _parse_strands, _normalize, _build_mfe_matrices, _to_dot_bracket,
        MIN_HAIRPIN_LOOP, _MAX_IL,
    )
    from strider.thermo.salt import dg_per_bp_salt

    raw_seq, nicks, sep_char = _parse_strands(sequence, material)
    seq = _normalize(raw_seq, material)
    n = len(seq)
    if n == 0:
        return []

    T = celsius + 273.15
    dg_salt = dg_per_bp_salt(sodium_M, magnesium_M, celsius, material)
    V, W, WM, WM1, energy_fns = _build_mfe_matrices(seq, T, material, nicks, dg_salt)
    (can, spans, inter, hairpin_e, stack_e, il_e, terminal_e,
     ml_a, ml_b, ml_c) = energy_fns

    if n == 1:
        return [(_to_dot_bracket([], n, nicks, sep_char), 0.0, [])]

    mfe = float(W[0][n - 1])
    bound = mfe + gap
    EPS = 1e-7
    # Cap total enumerated structures (incl. grammar-ambiguous duplicates,
    # which are deduplicated below) so a wide gap on a long sequence is bounded.
    hard_cap = max(max_structures * 200, 20000)

    def sub_V(i, j, cap):
        """Structures on [i..j] with (i,j) paired and total energy ≤ ``cap``.

        Mirrors the V recurrence of :func:`fold_mfe`; ``dg_salt`` is added once
        for the pair ``(i, j)`` created here (inner pairs carry their own)."""
        if V[i][j] >= INF or V[i][j] > cap + EPS:
            return
        inner = cap - dg_salt          # budget left after this pair's salt term
        pair = (i, j)

        # Hairpin
        if not spans(i, j):
            e = hairpin_e(seq, i, j, T)
            if e <= inner + EPS:
                yield frozenset((pair,)), e + dg_salt

        # Inter-strand terminal (blunt) pair
        if inter(i, j) and not can(i + 1, j - 1):
            e = terminal_e(seq, i, j)
            if e <= inner + EPS:
                yield frozenset((pair,)), e + dg_salt

        # Stack on (i+1, j-1)
        if i + 1 < j - 1 and can(i + 1, j - 1):
            st = stack_e(seq, i, j)
            for p, e in sub_V(i + 1, j - 1, inner - st):
                yield p | {pair}, st + e + dg_salt

        # Internal loop / bulge to inner pair (ip, jp)
        max_ip = min(i + _MAX_IL + 1, j - MIN_HAIRPIN_LOOP - 2)
        for ip in range(i + 1, max_ip + 1):
            min_jp = max(ip + MIN_HAIRPIN_LOOP + 1, j - _MAX_IL - 1)
            for jp in range(min_jp, j):
                if ip == i + 1 and jp == j - 1:
                    continue  # covered by the stack case
                nl = ip - i - 1
                nr = j - jp - 1
                if nl + nr == 0 or nl + nr > _MAX_IL:
                    continue
                if not can(ip, jp) or V[ip][jp] >= INF:
                    continue
                if spans(i + 1, ip - 1) or spans(jp + 1, j - 1):
                    continue
                il = il_e(seq, i, j, ip, jp, nl, nr)
                if il + V[ip][jp] > inner + EPS:
                    continue
                for p, e in sub_V(ip, jp, inner - il):
                    yield p | {pair}, il + e + dg_salt

        # Multi-loop closed by (i, j): ≥2 branches inside
        base = ml_a + ml_b
        for k in range(i + 2, j - 1):
            if WM[i + 1][k] >= INF or WM1[k + 1][j - 1] >= INF:
                continue
            if base + WM[i + 1][k] + WM1[k + 1][j - 1] > inner + EPS:
                continue
            for lp, le in sub_WM(i + 1, k, inner - base - WM1[k + 1][j - 1]):
                for rp, re in sub_WM1(k + 1, j - 1, inner - base - le):
                    yield lp | rp | {pair}, base + le + re + dg_salt

    def sub_WM1(i, j, cap):
        """One-branch multi-loop fragment WM1[i..j], energy ≤ ``cap``."""
        if WM1[i][j] >= INF or WM1[i][j] > cap + EPS:
            return
        if V[i][j] < INF and V[i][j] + ml_b <= cap + EPS:
            for p, e in sub_V(i, j, cap - ml_b):
                yield p, e + ml_b
        if j > i and WM1[i][j - 1] < INF and WM1[i][j - 1] + ml_c <= cap + EPS:
            for p, e in sub_WM1(i, j - 1, cap - ml_c):
                yield p, e + ml_c

    def sub_WM(i, j, cap):
        """Multi-loop fragment WM[i..j] with ≥1 branch, energy ≤ ``cap``."""
        if WM[i][j] >= INF or WM[i][j] > cap + EPS:
            return
        if WM1[i][j] < INF and WM1[i][j] <= cap + EPS:
            yield from sub_WM1(i, j, cap)
        if j > i and WM[i][j - 1] < INF and WM[i][j - 1] + ml_c <= cap + EPS:
            for p, e in sub_WM(i, j - 1, cap - ml_c):
                yield p, e + ml_c
        for k in range(i, j):
            if WM[i][k] >= INF or WM1[k + 1][j] >= INF:
                continue
            if WM[i][k] + WM1[k + 1][j] > cap + EPS:
                continue
            for lp, le in sub_WM(i, k, cap - WM1[k + 1][j]):
                for rp, re in sub_WM1(k + 1, j, cap - le):
                    yield lp | rp, le + re

    def sub_W(i, j, cap):
        """Exterior-context structures on [i..j], energy ≤ ``cap``.

        Unambiguous by leftmost base: ``i`` is either unpaired or the 5' end of
        exactly one stem ``(i, k)`` (intra- or inter-strand)."""
        if i > j:
            yield frozenset(), 0.0
            return
        if W[i][j] > cap + EPS:
            return
        # i unpaired
        yield from sub_W(i + 1, j, cap)
        # i is the 5' partner of a stem closing at k; remainder is exterior.
        # Prune on V[i][k] + rest lower bound (not V[i][k] alone — a weak stem
        # can still belong to a very stable structure via the remainder).
        for k in range(i + 1, j + 1):
            if V[i][k] >= INF:
                continue
            rest_lb = W[k + 1][j] if k + 1 <= j else 0.0
            if V[i][k] + rest_lb > cap + EPS:
                continue
            for vp, ve in sub_V(i, k, cap - rest_lb):
                for rp, re in sub_W(k + 1, j, cap - ve):
                    yield vp | rp, ve + re

    # Drive the enumeration; deduplicate grammar-ambiguous repeats (keep min E).
    seen: dict[frozenset, float] = {}
    count = 0
    for p, e in sub_W(0, n - 1, bound):
        count += 1
        if e <= bound + EPS:
            cur = seen.get(p)
            if cur is None or e < cur:
                seen[p] = e
        if count >= hard_cap:
            break

    items = sorted(seen.items(), key=lambda kv: kv[1])
    out: list[tuple[str, float, list[tuple[int, int]]]] = []
    for pset, e in items[:max_structures]:
        plist = sorted(pset)
        out.append((_to_dot_bracket(plist, n, nicks, sep_char), e, plist))
    return out


# ─── order-invariant multi-strand suboptimals ──────────────────────────────────

def _render_pseudoknot(
    pairs: list[tuple[int, int]], n: int, nicks: list[int], sep_char: str
) -> str:
    """Dot-bracket for ``pairs`` over ``[0, n)``, layering crossing pairs.

    Pairs that nest are drawn with ``()``; pairs that cross the nested layer use
    ``[]`` (and a second crossing layer ``{}``).  Needed because an order-invariant
    suboptimal set is reported in one common (the MFE-winning) strand order, in
    which a structure nested under a *different* order may cross.
    """
    from strider.structure.mfe import _insert_separators
    from strider.viz.geometry import classify_pairs

    db = ["."] * n
    remaining = [(min(i, j), max(i, j)) for i, j in pairs]
    for opener, closer in (("(", ")"), ("[", "]"), ("{", "}")):
        if not remaining:
            break
        nested, crossing = classify_pairs(remaining)
        for i, j in nested:
            db[i], db[j] = opener, closer
        remaining = crossing
    # Any residual (≥3 crossing layers) falls back to the level-2 bracket; rare.
    for i, j in remaining:
        db[i], db[j] = "{", "}"
    return _insert_separators("".join(db), nicks, sep_char)


def _slot_offsets(lens: list[int]) -> list[int]:
    off = [0]
    for length in lens:
        off.append(off[-1] + length)
    return off


def _pos_remapper(from_order, to_order, lens_input):
    """Return ``pos -> pos`` mapping concat positions between two strand orders.

    ``lens_input`` are strand lengths in *input* order; ``from_order``/``to_order``
    are permutations of input strand indices.  Maps a position in the
    ``from_order`` concatenation to the same nucleotide's position in the
    ``to_order`` concatenation.
    """
    off_from = _slot_offsets([lens_input[i] for i in from_order])
    off_to = _slot_offsets([lens_input[i] for i in to_order])
    slot_in_to = {s: k for k, s in enumerate(to_order)}

    def remap(pos: int) -> int:
        k = 0
        while k + 1 < len(off_from) and pos >= off_from[k + 1]:
            k += 1
        local = pos - off_from[k]
        s = from_order[k]
        return off_to[slot_in_to[s]] + local

    return remap


def subopt_complex(
    strands: list[str],
    gap: float = 1.0,
    celsius: float = 37.0,
    material: str = "dna",
    max_structures: int = 200,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
) -> list[tuple[str, float, list[tuple[int, int]]]]:
    """Order-invariant suboptimal enumeration for a multi-strand complex.

    The linear DP only represents structures non-crossing for one strand
    concatenation, so a single-order ``subopt_structures`` both uses the wrong
    (order-dependent) MFE baseline and misses structures nested only under a
    different cut.  This enumerates suboptimals across the same strand
    arrangements the order-invariant MFE search considers
    (:func:`strider.structure.complex_fold.fold_complex`), measures the gap from
    the *global* MFE, deduplicates structures across orders, and reports them in
    one common order — the MFE-winning order — using pseudoknot brackets for any
    structure that crosses in that order.

    Energies are *structural* (per-structure loop energy), so ``subopt[0]``
    equals the order-invariant raw MFE (``fold_mfe`` of the winning order) — the
    rotational-symmetry term σ is a complex-level ensemble correction that lives
    in ``engine.mfe``/``pfunc``, not in a per-structure energy.  Returns
    ``(dot_bracket, energy, pair_list)`` like :func:`subopt_structures`, sorted
    ascending, capped at ``max_structures``.
    """
    from strider.structure.complex_fold import fold_complex

    lens = [len(s) for s in strands]
    win = fold_complex(strands, celsius, material, sodium_M, magnesium_M)
    win_order = win.order
    global_mfe = win.energy
    bound = global_mfe + gap
    EPS = 1e-7

    # Winning-order nicks (the common reporting frame).
    win_nicks: list[int] = []
    pos = 0
    for i in win_order[:-1]:
        pos += lens[i]
        win_nicks.append(pos)
    n = sum(lens)

    seen: dict[frozenset, float] = {}
    for order, e_order in win.evaluated:
        per_order_gap = bound - e_order
        if per_order_gap < -EPS:
            continue
        concat = "&".join(strands[i] for i in order)
        remap = _pos_remapper(order, win_order, lens)
        for _db, e, plist in subopt_structures(
            concat, max(0.0, per_order_gap), celsius, material,
            max_structures * 4, sodium_M, magnesium_M,
        ):
            if e > bound + EPS:
                continue
            canon = frozenset(
                (lambda a, b: (a, b) if a < b else (b, a))(remap(i), remap(j))
                for i, j in plist
            )
            cur = seen.get(canon)
            if cur is None or e < cur:
                seen[canon] = e

    items = sorted(seen.items(), key=lambda kv: kv[1])[:max_structures]
    out: list[tuple[str, float, list[tuple[int, int]]]] = []
    for pset, e in items:
        plist = sorted(pset)
        db = _render_pseudoknot(plist, n, win_nicks, "&")
        out.append((db, e, plist))
    return out
