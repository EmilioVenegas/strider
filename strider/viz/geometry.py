"""
Native 2D layout geometry for nucleic-acid secondary structures.

Pure math (numpy only, no matplotlib) so it is fast and unit-testable without
rendering.  Produces RNAplot/NAView-style radial coordinates from a base-pair
list, with no dependency on ViennaRNA.

Algorithm
---------
1. ``classify_pairs`` splits a pair list into a maximal *nested* set and the
   leftover *crossing* (pseudoknot) pairs.  Only nested pairs drive the layout;
   crossing pairs are drawn afterwards as straight chords by the caller.
2. ``radial_layout`` walks the exterior loop along a line; every helix is laid
   out as a straight parallel ladder, and every loop places its boundary bases
   evenly around a circle whose closing chord is the helix's inner pair.
3. ``relax`` optionally applies a deterministic spring/repulsion pass to reduce
   residual base overlaps (no RNG, fixed iterations → reproducible).

All coordinates are over the *concatenated* index space (separators already
stripped).  ``nicks`` do not affect geometry — strand breaks only change which
backbone segments the renderer draws — but are accepted for a uniform signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Pair classification
# --------------------------------------------------------------------------- #
def _crosses(p: tuple[int, int], q: tuple[int, int]) -> bool:
    """True if pairs ``p`` and ``q`` interleave (pseudoknot crossing)."""
    a, b = p
    c, d = q
    # exactly one of c, d lies strictly inside (a, b)
    return (a < c < b < d) or (c < a < d < b)


def classify_pairs(
    pairs: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Partition ``pairs`` into ``(nested, crossing)``.

    Greedy: accept pairs in canonical order, keeping a pair only if it does not
    cross any already-accepted pair.  Deterministic; the rejected (crossing)
    pairs are returned so the caller can draw them as chords.
    """
    nested: list[tuple[int, int]] = []
    crossing: list[tuple[int, int]] = []
    for pair in sorted((min(i, j), max(i, j)) for i, j in pairs):
        if any(_crosses(pair, acc) for acc in nested):
            crossing.append(pair)
        else:
            nested.append(pair)
    return nested, crossing


# --------------------------------------------------------------------------- #
# Strand ordering (multi-strand pseudoknot minimisation)
# --------------------------------------------------------------------------- #
def crossing_count(pairs: list[tuple[int, int]]) -> int:
    """Count interleaving pair-of-pairs (the number of pseudoknot crossings).

    Cheaper than :func:`classify_pairs` when only the crossing *count* is needed
    (e.g. comparing strand orders). Pairs are sorted by start, so once a later
    pair starts past the current pair's close, no further pair can interleave it.
    """
    ps = sorted((min(i, j), max(i, j)) for i, j in pairs)
    count = 0
    for a, (ai, aj) in enumerate(ps):
        for bi, bj in ps[a + 1:]:
            if bi >= aj:               # sorted by start → nothing later interleaves a
                break
            if ai < bi < aj < bj:      # bi already > ai (sorted); only this case crosses
                count += 1
    return count


def _strand_offsets(strand_lens: list[int]) -> list[int]:
    off = [0]
    for length in strand_lens:
        off.append(off[-1] + length)
    return off


def remap_pairs(
    pairs: list[tuple[int, int]], strand_lens: list[int], order: list[int]
) -> list[tuple[int, int]]:
    """Remap concatenated-index pairs when strands are reordered.

    ``order`` is a permutation of strand indices: new slot ``k`` holds original
    strand ``order[k]``. Returns pairs over the *new* concatenated index space.
    """
    n = len(strand_lens)
    off = _strand_offsets(strand_lens)

    def split(g: int) -> tuple[int, int]:
        for s in range(n):
            if off[s] <= g < off[s + 1]:
                return s, g - off[s]
        return n - 1, g - off[n - 1]

    new_off: dict[int, int] = {}
    cur = 0
    for s in order:
        new_off[s] = cur
        cur += strand_lens[s]

    out = []
    for a, b in pairs:
        sa, la = split(a)
        sb, lb = split(b)
        out.append((new_off[sa] + la, new_off[sb] + lb))
    return out


def _greedy_strand_order(
    pairs: list[tuple[int, int]], strand_lens: list[int]
) -> list[int]:
    """Greedy adjacency heuristic: keep heavily base-paired strands neighbours.

    Fallback for complexes with too many strands to brute-force.  Seeds with the
    most strongly paired strand pair and extends the chain at whichever end gains
    the most base pairs.  Not optimal, but a reasonable non-crossing ordering.
    """
    from collections import deque

    n = len(strand_lens)
    off = _strand_offsets(strand_lens)

    def strand_of(g: int) -> int:
        for s in range(n):
            if off[s] <= g < off[s + 1]:
                return s
        return n - 1

    w = [[0] * n for _ in range(n)]
    for a, b in pairs:
        sa, sb = strand_of(a), strand_of(b)
        if sa != sb:
            w[sa][sb] += 1
            w[sb][sa] += 1

    si, sj, best = 0, 1, -1
    for i in range(n):
        for j in range(i + 1, n):
            if w[i][j] > best:
                si, sj, best = i, j, w[i][j]
    if best <= 0:
        return list(range(n))

    placed = [False] * n
    chain = deque([si, sj])
    placed[si] = placed[sj] = True
    while len(chain) < n:
        left, right = chain[0], chain[-1]
        pick, pw, end = None, -1, "right"
        for k in range(n):
            if placed[k]:
                continue
            if w[left][k] > pw:
                pick, pw, end = k, w[left][k], "left"
            if w[right][k] > pw:
                pick, pw, end = k, w[right][k], "right"
        if pick is None:                       # disconnected remainder
            pick = next(k for k in range(n) if not placed[k])
            end = "right"
        (chain.appendleft if end == "left" else chain.append)(pick)
        placed[pick] = True
    return list(chain)


def best_strand_order(
    pairs: list[tuple[int, int]], strand_lens: list[int], max_brute: int = 7
) -> list[int]:
    """Find a strand concatenation order that minimises pseudoknot crossings.

    Multi-strand pseudoknot-freeness is *order-dependent*: the same base pairs
    can be fully nested under one strand order and heavily crossing under
    another (a closed network such as a dendrimer is pseudoknotted in the
    "obvious" order yet flat in an alternating one).  Returns a permutation of
    strand indices (original index per new slot) minimising
    :func:`crossing_count`.  The identity order is always evaluated, so the
    result never *increases* crossings.

    Brute-forces all permutations for ``<= max_brute`` strands (early-exiting on
    the first crossing-free order); above that, uses :func:`_greedy_strand_order`.
    """
    import itertools

    n = len(strand_lens)
    identity = list(range(n))
    if n <= 2 or not pairs:
        return identity

    def cost(order: list[int]) -> int:
        return crossing_count(remap_pairs(pairs, strand_lens, order))

    best, best_c = identity, cost(identity)
    if best_c == 0:
        return identity

    if n <= max_brute:
        for order in itertools.permutations(range(n)):
            c = cost(list(order))
            if c < best_c:
                best, best_c = list(order), c
                if best_c == 0:
                    break
    else:
        cand = _greedy_strand_order(pairs, strand_lens)
        if cost(cand) < best_c:
            best = cand
    return best


# --------------------------------------------------------------------------- #
# Structure tree (for inspection / tests)
# --------------------------------------------------------------------------- #
@dataclass
class StructureNode:
    """A loop or helix in the secondary-structure decomposition."""

    kind: str  # "external" | "helix" | "loop"
    pairs: list[tuple[int, int]] = field(default_factory=list)
    unpaired: list[int] = field(default_factory=list)
    children: list["StructureNode"] = field(default_factory=list)


def _partner_array(pairs: list[tuple[int, int]], n: int) -> list[int]:
    partner = [-1] * n
    for i, j in pairs:
        partner[i] = j
        partner[j] = i
    return partner


def _stem_length(partner: list[int], c: int, d: int) -> int:
    """Number of stacked pairs in the helix opening at ``(c, d)``."""
    length = 0
    while c + length < d - length and partner[c + length] == d - length:
        length += 1
    return length


def build_structure_tree(
    pairs: list[tuple[int, int]], n: int, nicks: list[int] | None = None
) -> StructureNode:
    """
    Build a nested loop/helix tree from a *nested* pair list.

    Crossing pairs must be removed first (see :func:`classify_pairs`).  Returned
    mainly for inspection and testing; :func:`radial_layout` does not require it.
    """
    partner = _partner_array(pairs, n)

    def build_loop(lo: int, hi: int, kind: str) -> StructureNode:
        node = StructureNode(kind=kind)
        i = lo
        while i < hi:
            j = partner[i]
            if j == -1 or j < i:
                node.unpaired.append(i)
                i += 1
            else:
                length = _stem_length(partner, i, j)
                helix = StructureNode(
                    kind="helix",
                    pairs=[(i + t, j - t) for t in range(length)],
                )
                inner_lo, inner_hi = i + length, j - length + 1
                helix.children.append(build_loop(inner_lo, inner_hi, "loop"))
                node.children.append(helix)
                i = j + 1
        return node

    return build_loop(0, n, "external")


def element_types(
    pairs: list[tuple[int, int]], n: int, nicks: list[int] | None = None
) -> list[str]:
    """
    Classify each base by its secondary-structure element.

    Returns a list of ``n`` strings drawn from ``{"stem", "hairpin", "interior",
    "multiloop", "exterior"}``: paired bases are ``"stem"``; unpaired bases take
    the type of their enclosing loop (a loop with 0 child helices is a hairpin,
    1 is an interior loop / bulge, >=2 is a multiloop; the open exterior loop is
    ``"exterior"``).
    """
    nested, _ = classify_pairs(pairs)
    tree = build_structure_tree(nested, n, nicks)
    types = ["exterior"] * n

    def visit(node: StructureNode, is_external: bool) -> None:
        if node.kind in ("external", "loop"):
            if is_external:
                kind = "exterior"
            else:
                n_helix = sum(1 for c in node.children if c.kind == "helix")
                kind = ("hairpin" if n_helix == 0
                        else "interior" if n_helix == 1 else "multiloop")
            for u in node.unpaired:
                types[u] = kind
            for c in node.children:
                visit(c, False)
        elif node.kind == "helix":
            for a, b in node.pairs:
                types[a] = types[b] = "stem"
            for c in node.children:
                visit(c, False)

    visit(tree, True)
    return types


def is_branched(nested: list[tuple[int, int]], n: int,
                nicks: list[int] | None = None) -> bool:
    """True if the structure has a loop (incl. the exterior) with >= 2 child
    helices — a multiloop or several exterior stems.  That branching is exactly
    where the space-aware :func:`tree_layout` beats the simple radial one, so it
    is the selector for ``layout_structure(method="auto")``."""
    tree = build_structure_tree(nested, n, nicks)

    def scan(node: StructureNode) -> bool:
        if node.kind in ("external", "loop"):
            if sum(1 for c in node.children if c.kind == "helix") >= 2:
                return True
        return any(scan(c) for c in node.children)

    return scan(tree)


def has_segment_crossing(
    coords: np.ndarray, segments: list[tuple[int, int]]
) -> bool:
    """True as soon as two of the ``segments`` properly cross (shared endpoints
    excluded).  Used by ``layout_structure(method="auto")`` to keep the simple
    radial layout when it is already planar and only fall back to the heavier
    :func:`tree_layout` when radial would overlap.  Early-exits on the first
    crossing, so a tangled layout is rejected cheaply."""
    def side(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    m = len(segments)
    for i in range(m):
        a, b = segments[i]
        p1, p2 = coords[a], coords[b]
        for j in range(i + 1, m):
            c, d = segments[j]
            if a == c or a == d or b == c or b == d:
                continue
            p3, p4 = coords[c], coords[d]
            if ((side(p3, p4, p1) > 0) != (side(p3, p4, p2) > 0)
                    and (side(p1, p2, p3) > 0) != (side(p1, p2, p4) > 0)):
                return True
    return False


# --------------------------------------------------------------------------- #
# Radial coordinate assignment
# --------------------------------------------------------------------------- #
def radial_layout(
    pairs: list[tuple[int, int]],
    n: int,
    nicks: list[int] | None = None,
    spacing: float = 1.0,
    rise: float = 1.0,
) -> np.ndarray:
    """
    Assign 2D coordinates to every base from a *nested* pair list.

    Parameters
    ----------
    pairs   : nested base pairs (0-indexed, over the concatenated sequence)
    n       : number of bases
    spacing : chord length between adjacent bases on a loop circle and the
              ladder width of a base pair
    rise    : stacking rise between consecutive pairs of a helix

    Returns
    -------
    ndarray of shape ``(n, 2)``.
    """
    coords = np.zeros((n, 2), dtype=float)
    if n == 0:
        return coords
    partner = _partner_array(pairs, n)
    pd = float(spacing)

    def place_stem(c: int, d: int, direction: np.ndarray) -> None:
        """Lay helix opening at ``(c, d)`` as a ladder growing along ``direction``."""
        u = direction / (np.linalg.norm(direction) or 1.0)
        length = _stem_length(partner, c, d)
        for t in range(1, length):
            coords[c + t] = coords[c] + t * rise * u
            coords[d - t] = coords[d] + t * rise * u
        a, b = c + length - 1, d - length + 1  # innermost pair
        place_loop(a, b, u)

    def place_loop(a: int, b: int, inward: np.ndarray) -> None:
        """Place the loop closed by pair ``(a, b)`` around a circle."""
        boundary = [a]
        i = a + 1
        while i < b:
            j = partner[i]
            if j == -1 or j < i:
                boundary.append(i)
                i += 1
            else:
                boundary.append(i)
                boundary.append(j)
                i = j + 1
        boundary.append(b)

        k = len(boundary)
        if k < 3:
            return  # blunt end / empty loop: a and b already placed

        Pa, Pb = coords[a], coords[b]
        mid = (Pa + Pb) / 2.0
        d_ab = float(np.linalg.norm(Pa - Pb)) or pd
        u = inward / (np.linalg.norm(inward) or 1.0)
        r = (d_ab / 2.0) / np.sin(np.pi / k)
        h = np.sqrt(max(r * r - (d_ab / 2.0) ** 2, 0.0))
        center = mid + h * u

        ang_a = float(np.arctan2(*(Pa - center)[::-1]))
        ang_b = float(np.arctan2(*(Pb - center)[::-1]))
        ccw = (ang_b - ang_a) % (2 * np.pi)
        # the stem side is the short arc (< pi); traverse the long arc for items
        step = ccw / (k - 1) if ccw >= np.pi else -((2 * np.pi - ccw) / (k - 1))

        for s, base in enumerate(boundary):
            ang = ang_a + s * step
            coords[base] = center + r * np.array([np.cos(ang), np.sin(ang)])

        # recurse into child helices on this loop
        for s, base in enumerate(boundary):
            j = partner[base]
            if j != -1 and j > base and base != a:
                Pc, Pd = coords[base], coords[j]
                out = (Pc + Pd) / 2.0 - center
                place_stem(base, j, out)

    nick_set = set(nicks or [])

    def spans_nick(c: int, d: int) -> bool:
        """True if pair (c, d) is inter-strand (a nick lies between its bases)."""
        return any(c < k <= d for k in nick_set)

    # collect the exterior elements in 5'->3' order
    items: list[tuple] = []
    i = 0
    while i < n:
        j = partner[i]
        if j == -1 or j < i:
            items.append(("base", i))
            i += 1
        else:
            items.append(("helix", i, j))
            i = j + 1

    helices = [it for it in items if it[0] == "helix"]
    bases = [it for it in items if it[0] == "base"]

    def _is_linear_duplex() -> bool:
        """True if the whole molecule is one (possibly nicked/coaxial) duplex —
        no exterior unpaired bases and the helices form one coaxial chain."""
        if bases or not helices:
            return False
        for a, b in zip(helices, helices[1:]):
            if not (b[1] == a[2] + 1 and b[1] not in nick_set
                    and spans_nick(*a[1:]) and spans_nick(*b[1:])):
                return False
        return True

    if not helices:
        # fully unpaired: lay the strand(s) along a line
        for idx in range(n):
            coords[idx] = [idx * spacing, 0.0]
        return coords

    if _is_linear_duplex():
        # straight coaxial stack (a single linear duplex reads cleanest as a line)
        prev = None
        for _, ci, cj in helices:
            if prev is not None:
                axial, close_pt, rung = prev
                direction = -axial
                coords[ci] = close_pt + direction * rise
                coords[cj] = coords[ci] + rung
            else:
                direction = np.array([0.0, 1.0])
                coords[ci] = [0.0, 0.0]
                coords[cj] = [pd, 0.0]
            place_stem(ci, cj, direction)
            prev = (direction.copy(), coords[cj].copy(), coords[ci] - coords[cj])
        return coords

    # radial exterior loop: place every exterior element around a circle so the
    # structure radiates (RNAplot / forna style) instead of standing on a line.
    circle_pts: list[int] = []
    helix_slots: list[tuple[int, int, int]] = []
    for it in items:
        if it[0] == "base":
            circle_pts.append(it[1])
        else:
            _, ci, cj = it
            circle_pts.append(ci)
            helix_slots.append((ci, cj, len(circle_pts) - 1))
            circle_pts.append(cj)

    m = len(circle_pts)
    k = m + 1  # one extra slot for the open 5'->3' gap
    r = pd / (2.0 * np.sin(np.pi / k))
    # gap centred at the bottom (-y); points run counter-clockwise from there
    base_ang = -np.pi / 2
    for slot, base in enumerate(circle_pts):
        ang = base_ang + (slot + 1) * (2 * np.pi / k)
        coords[base] = r * np.array([np.cos(ang), np.sin(ang)])

    # each exterior helix radiates outward (away from the loop centre)
    for ci, cj, _ in helix_slots:
        mid = (coords[ci] + coords[cj]) / 2.0
        outward = mid / (np.linalg.norm(mid) or 1.0)
        place_stem(ci, cj, outward)

    return coords


# --------------------------------------------------------------------------- #
# Junction-graph (loop-tree) layout
# --------------------------------------------------------------------------- #
# angular-reservation blend levels tried by :func:`tree_layout`, from the most
# compact (``0.0`` = reserve only each sub-tree's lateral *width*) to the safest
# (``1.0`` = reserve its full radial *reach*, the old always-planar behaviour).
_TREE_BLEND_LEVELS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0)


def _min_pair_distance(coords: np.ndarray) -> float:
    """Smallest distance between any two points (``inf`` for <2 points)."""
    if len(coords) < 2:
        return float("inf")
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = (diff * diff).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min()))


def tree_layout(
    pairs: list[tuple[int, int]],
    n: int,
    nicks: list[int] | None = None,
    spacing: float = 1.0,
    rise: float = 1.0,
    safety: float = 1.5,
) -> np.ndarray:
    """
    Space-aware loop-tree layout (RNApuzzler-style passes 1 & 2).

    A pseudoknot-free structure is a *tree* of loops joined by helices, so the
    only reason :func:`radial_layout` lets big sub-structures collide is that it
    gives every helix on a loop the same angular slot regardless of how large the
    sub-tree hanging off it is.  This layout fixes that with two passes:

    1. **measure** (post-order) — each helix sub-tree's radial *reach* ``R`` (its
       farthest point) and lateral *width* ``W`` (its widest loop) are computed
       bottom-up;
    2. **place** (top-down) — every loop is drawn as a circle whose radius is
       *solved* (smallest radius whose total angular demand fits 2π) so each child
       helix gets a wedge of ``safety·atan(reach/dist)``.

    Picking the wedge ``reach`` is a trade-off: reserving each arm's full radial
    extent ``R`` guarantees no overlap but, where several long arms meet a loop
    joined by only a base or two (e.g. a dendrimer core), it blows that loop up
    into a huge circle whose short connecting backbone is then stretched across a
    big arc — long, ugly bonds.  An arm that radiates outward really only needs a
    wedge set by its lateral *width* ``W``, not its length, so this layout
    **escalates**: it lays the structure out at increasing reservation
    ``W + t·(R − W)`` for ``t`` in :data:`_TREE_BLEND_LEVELS` and returns the
    first result that is planar (no segment crossing) *and* overlap-free
    (min base distance > spacing).  Compact folds settle at ``t=0`` (tight, like
    forna); only a genuinely crowded junction climbs toward ``t=1`` — and ``t=1``
    is the old full-reach behaviour, so the layout is **never worse than before**.

    Helices stay rigid ladders and loops stay circular (the clean "diagram"
    look); the result is deterministic (no RNG).  Same signature as
    :func:`radial_layout` so it is a drop-in alternative.  ``safety`` (>=1) is the
    margin multiplier on each sub-tree's angular wedge.
    """
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    partner = _partner_array(pairs, n)
    pd = float(spacing)

    def boundary(a: int, b: int) -> list[tuple]:
        """Items strictly between closing pair (a, b), 5'->3': bases and stems."""
        items: list[tuple] = []
        i = a + 1
        while i < b:
            j = partner[i]
            if j == -1 or j < i:
                items.append(("base", i))
                i += 1
            else:
                items.append(("stem", i, j))
                i = j + 1
        return items

    def base_angle(r: float) -> float:
        """Angle a one-spacing chord subtends at radius ``r`` (loop base step)."""
        return 2.0 * np.arcsin(min(1.0, (pd / 2.0) / r))

    def metas_of(items: list[tuple], ends: int) -> list[tuple]:
        """Boundary points around the loop as ``(kind, point, stem)`` triples:
        ``point`` is the base index to place; ``stem`` is the owning stem's opener
        (for clearance / rigidity lookups).  Includes the closing pair ``a``/``b``
        as ``end`` points for a closed loop, every unpaired base, and each child
        stem's two strands (opener ``c`` and closer ``partner[c]``)."""
        m: list[tuple] = [("end", -1, -1)] if ends == 2 else []
        for it in items:
            if it[0] == "base":
                m.append(("base", it[1], it[1]))
            else:
                c, d = it[1], it[2]
                m.append(("open", c, c))
                m.append(("close", d, c))
        if ends == 2:
            m.append(("end", -1, -1))
        return m

    def _layout(blend: float) -> np.ndarray:
        """Lay the whole structure out reserving ``W + blend·(R − W)`` per arm."""
        coords = np.zeros((n, 2), dtype=float)
        R_of: dict[int, float] = {}   # stem opener c -> sub-tree radial reach
        W_of: dict[int, float] = {}   # stem opener c -> sub-tree lateral width
        sl_of: dict[int, int] = {}    # stem opener c -> stem length (stacked pairs)
        r_of: dict[int, float] = {}   # loop key (closing 'a', or -1) -> radius

        def reach_of(c: int) -> float:
            return W_of[c] + blend * (R_of[c] - W_of[c])

        def gaps_of(metas: list[tuple], r: float) -> tuple[list[float], list[bool]]:
            """Angular gap between each pair of consecutive boundary points, with a
            rigid flag.  Every gap is one base step; a flexible gap flanking a child
            stem also reserves ``safety·atan(reach/dist)`` so the sub-tree clears
            its neighbour.  The gap *inside* a stem (its two strands) is **rigid** —
            fixed at one chord so the helix ladder stays one spacing wide."""
            ba = base_angle(r)
            gaps, rigid = [], []
            for s in range(len(metas) - 1):
                cur, nxt = metas[s], metas[s + 1]
                is_rigid = cur[0] == "open" and nxt[0] == "close" and cur[2] == nxt[2]
                g = ba
                if not is_rigid:
                    if nxt[0] == "open":               # entering a stem (5' flank)
                        c = nxt[2]
                        g += safety * np.arctan(reach_of(c) / (r + sl_of[c] * rise))
                    if cur[0] == "close":              # leaving a stem (3' flank)
                        c = cur[2]
                        g += safety * np.arctan(reach_of(c) / (r + sl_of[c] * rise))
                gaps.append(g)
                rigid.append(is_rigid)
            return gaps, rigid

        def demand(items: list[tuple], ends: int, r: float) -> float:
            """Total angle the loop needs at radius ``r``: all gaps + the reserved
            closing chord (closed loop) or open-ends gap (exterior)."""
            return sum(gaps_of(metas_of(items, ends), r)[0]) + base_angle(r)

        def solve_radius(items: list[tuple], ends: int) -> float:
            """Smallest radius whose angular demand fits 2π.  A child stem's demand
            is ``atan(reach/dist)`` with ``dist`` growing as the loop inflates, so
            the demand *shrinks* with ``r`` and a bisection on the critical radius
            always converges — and unlike an arc-length rule it never blows the loop
            up to seat one large arm.  Bisection (vs a geometric ramp) lands on the
            tight critical radius, so no slack is left to over-stretch backbone."""
            n_pts = ends + sum(1 if it[0] == "base" else 2 for it in items)
            r = (pd / 2.0) / np.sin(np.pi / max(n_pts, 3))   # equal-seat floor
            if demand(items, ends, r) <= 2.0 * np.pi:
                return r
            lo, hi = r, r
            for _ in range(200):                              # grow until it fits
                hi *= 1.5
                if demand(items, ends, hi) <= 2.0 * np.pi:
                    break
            for _ in range(60):                               # bisect to critical r
                mid = 0.5 * (lo + hi)
                if demand(items, ends, mid) <= 2.0 * np.pi:
                    hi = mid
                else:
                    lo = mid
            return hi

        # ---- Pass 1: bottom-up reach + width + per-loop radius (post-order)
        def measure_stem(c: int, d: int) -> float:
            sl = _stem_length(partner, c, d)
            sl_of[c] = sl
            a2, b2 = c + sl - 1, d - sl + 1        # innermost pair of this helix
            items = boundary(a2, b2)
            child_reach = 0.0
            child_width = 0.0
            for it in items:
                if it[0] == "stem":
                    r_in = measure_stem(it[1], it[2])
                    child_reach = max(child_reach, sl_of[it[1]] * rise + r_in)
                    child_width = max(child_width, W_of[it[1]])
            r_loop = solve_radius(items, ends=2)
            r_of[a2] = r_loop
            R_of[c] = r_loop + child_reach         # farthest point from loop centre
            W_of[c] = max(r_loop, child_width)     # widest loop in the sub-tree
            return R_of[c]

        ext_items = boundary(-1, n)                # whole sequence as one open loop
        for it in ext_items:
            if it[0] == "stem":
                measure_stem(it[1], it[2])

        # ---- Pass 2: top-down placement
        def place_children(items, ends, center, r, ang0, direction) -> None:
            """Walk the boundary points around the circle from ``ang0``, advancing
            by each angular gap (slack spread evenly so they fill 2π); stems then
            radiate outward and recurse."""
            metas = metas_of(items, ends)
            gaps, rigid = gaps_of(metas, r)
            free = [i for i, rg in enumerate(rigid) if not rg]   # not stem-internal
            slack = 2.0 * np.pi - (sum(gaps) + base_angle(r))    # minus reserved gap
            if free and slack > 0:
                for i in free:
                    gaps[i] += slack / len(free)
            cum = 0.0
            for idx, m in enumerate(metas):
                if m[0] != "end":                      # closing pair is pre-placed
                    ang = ang0 + direction * cum
                    coords[m[1]] = center + r * np.array([np.cos(ang), np.sin(ang)])
                if idx < len(gaps):
                    cum += gaps[idx]
            for it in items:
                if it[0] == "stem":
                    c, d = it[1], it[2]
                    outward = (coords[c] + coords[d]) / 2.0 - center
                    place_stem(c, d, outward)

        def place_stem(c: int, d: int, direction: np.ndarray) -> None:
            u = direction / (np.linalg.norm(direction) or 1.0)
            sl = sl_of[c]
            for t in range(1, sl):
                coords[c + t] = coords[c] + t * rise * u
                coords[d - t] = coords[d] + t * rise * u
            place_loop(c + sl - 1, d - sl + 1, u)

        def place_loop(a: int, b: int, inward: np.ndarray) -> None:
            items = boundary(a, b)
            if not items:
                return                         # blunt hairpin: a, b already placed
            r = r_of[a]
            Pa, Pb = coords[a], coords[b]
            mid = (Pa + Pb) / 2.0
            u = inward / (np.linalg.norm(inward) or 1.0)
            h = np.sqrt(max(r * r - (pd / 2.0) ** 2, 0.0))
            center = mid + h * u
            ang_a = float(np.arctan2(*(Pa - center)[::-1]))
            ang_b = float(np.arctan2(*(Pb - center)[::-1]))
            ccw = (ang_b - ang_a) % (2 * np.pi)
            direction = 1.0 if ccw >= np.pi else -1.0    # traverse the long arc
            place_children(items, 2, center, r, ang_a, direction)

        if not any(it[0] == "stem" for it in ext_items):
            for idx in range(n):                         # fully unpaired -> a line
                coords[idx] = [idx * pd, 0.0]
            return coords

        # exterior loop: centre at origin, open gap at the bottom for 5'/3' ends
        r_ext = solve_radius(ext_items, ends=0)
        place_children(ext_items, 0, np.zeros(2), r_ext,
                       ang0=-np.pi / 2 + base_angle(r_ext), direction=1.0)
        return coords

    # Escalate reservation from compact to full reach; first planar + overlap-free
    # wins.  Full reach (the last level) is the old always-planar layout, so the
    # loop always returns something no worse than before.
    nick_set = set(nicks or [])
    accept_segs = ([(i, i + 1) for i in range(n - 1) if (i + 1) not in nick_set]
                   + list(pairs))
    coords = None
    for blend in _TREE_BLEND_LEVELS:
        coords = _layout(blend)
        if (not has_segment_crossing(coords, accept_segs)
                and _min_pair_distance(coords) > 0.99 * pd):
            return coords
    return coords


# --------------------------------------------------------------------------- #
# Optional relaxation
# --------------------------------------------------------------------------- #
def relax(
    coords: np.ndarray,
    pairs: list[tuple[int, int]],
    nicks: list[int] | None = None,
    iterations: int = 60,
    spacing: float = 1.0,
    repel_range: float = 0.85,
) -> np.ndarray:
    """
    Deterministic spring/repulsion smoothing to reduce base overlaps.

    Backbone and base-pair springs restore ideal bond lengths; a repulsion term
    pushes apart bases within ``repel_range * spacing`` of each other.  No
    randomness, fixed iteration count → reproducible coordinates (safe for image
    baselines).

    ``repel_range`` is the repulsion cutoff in units of ``spacing``.  The default
    ``0.85`` only separates bases that are actually overlapping; larger values
    (~2-3) give a longer-range repulsion that spreads whole sub-structures apart,
    which de-congests big multi-branch complexes (many junctions) where distinct
    arms would otherwise pack together near the centre.
    """
    if iterations <= 0:
        return coords
    nick_set = set(nicks or [])
    n = len(coords)
    if n < 3:
        return coords
    pts = coords.astype(float).copy()
    pair_set = [(i, j) for i, j in pairs]
    cutoff = repel_range * spacing
    step = 0.05

    for _ in range(iterations):
        disp = np.zeros_like(pts)
        # backbone springs (skip strand breaks)
        for i in range(n - 1):
            if (i + 1) in nick_set:
                continue
            d = pts[i + 1] - pts[i]
            L = np.linalg.norm(d) or 1.0
            f = (L - spacing) / L * d
            disp[i] += 0.5 * f
            disp[i + 1] -= 0.5 * f
        # base-pair springs
        for i, j in pair_set:
            d = pts[j] - pts[i]
            L = np.linalg.norm(d) or 1.0
            f = (L - spacing) / L * d
            disp[i] += 0.5 * f
            disp[j] -= 0.5 * f
        # short-range repulsion (O(n^2), fine for typical sizes)
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        close = dist < cutoff
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = np.where(close, (cutoff - dist) / np.where(dist == 0, 1.0, dist), 0.0)
        rep = (diff * mag[:, :, None]).sum(axis=1)
        disp += 0.5 * rep
        pts += step * disp

    return pts
