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
# Optional relaxation
# --------------------------------------------------------------------------- #
def relax(
    coords: np.ndarray,
    pairs: list[tuple[int, int]],
    nicks: list[int] | None = None,
    iterations: int = 60,
    spacing: float = 1.0,
) -> np.ndarray:
    """
    Deterministic spring/repulsion smoothing to reduce base overlaps.

    Backbone and base-pair springs restore ideal bond lengths; a short-range
    repulsion pushes apart bases that have collided.  No randomness, fixed
    iteration count → reproducible coordinates (safe for image baselines).
    """
    if iterations <= 0:
        return coords
    nick_set = set(nicks or [])
    n = len(coords)
    if n < 3:
        return coords
    pts = coords.astype(float).copy()
    pair_set = [(i, j) for i, j in pairs]
    cutoff = 0.85 * spacing
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
