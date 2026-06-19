"""
2D secondary-structure rendering (single- and multi-strand).

``draw_structure`` lays out a structure with the native engine in
:mod:`strider.viz.layout` and draws it as a proper fold: backbone, base-pair
rungs, colored base circles, 5'/3' and strand labels, strand-nick markers, and
dashed pseudoknot chords.  ``draw_complex`` is the multi-strand convenience
wrapper that folds a set of strands and colors them per strand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strider.viz import style
from strider.viz.layout import Layout2D, layout_structure

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _base_colors(
    seq: str,
    layout: Layout2D,
    color: str,
    accessibility: dict[int, float] | None,
    strand_names: list[str] | None = None,
    strand_colors=None,
):
    """Per-base fill colors and an optional ScalarMappable (for a colorbar)."""
    n = len(seq)
    if color == "structure":
        from strider.viz import geometry

        types = geometry.element_types(layout.rungs + layout.crossing, n, layout.nicks)
        return [style.element_color(t) for t in types], None
    if color == "strand":
        def col_for(s: int) -> str:
            if strand_colors is not None:
                if isinstance(strand_colors, dict):
                    nm = strand_names[s] if strand_names and s < len(strand_names) else None
                    if nm in strand_colors:
                        return strand_colors[nm]
                elif s < len(strand_colors):
                    return strand_colors[s]
            return style.strand_color(s)
        return [col_for(int(layout.strand_of[i])) for i in range(n)], None
    if color == "accessibility":
        import matplotlib as mpl

        cmap = style.accessibility_cmap()
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        acc = accessibility or {}
        cols = [cmap(norm(acc.get(i, 0.0))) for i in range(n)]
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        return cols, sm
    # default: by nucleotide
    return [style.nt_color(b) for b in seq], None


def draw_structure(
    sequence: str,
    structure: str | None = None,
    *,
    pairs: list[tuple[int, int]] | None = None,
    nicks: list[int] | None = None,
    engine=None,
    ax: "Axes | None" = None,
    layout: Layout2D | None = None,
    color: str = "auto",
    accessibility: dict[int, float] | None = None,
    highlight: list[tuple[int, int]] | None = None,
    domains: dict | None = None,
    strand_names: list[str] | None = None,
    strand_colors=None,
    base_text: bool = True,
    base_radius: float = 0.42,
    labels: bool = True,
    nick_markers: bool = False,
    relax_iters: int = 0,
    repel_range: float = 0.85,
    method: str = "auto",
    loop_tightness: float = 1.5,
    title: str | None = None,
):
    """
    Draw one secondary structure in 2D.

    Parameters
    ----------
    sequence      : nucleotide sequence; may contain ``&``/``+`` separators
    structure     : dot-bracket; folded via ``engine`` if None and ``pairs`` is None
    pairs         : explicit base pairs (overrides ``structure``)
    color         : ``"auto"`` (structure for single strands, strand for
                    complexes), ``"structure"`` (by element), ``"nt"``,
                    ``"strand"``, or ``"accessibility"``
    accessibility : ``{position: unpaired_prob}`` for ``color="accessibility"``
    highlight     : list of ``(start, end)`` index ranges to outline (e.g. toeholds)
    domains       : ``{name: positions}`` regions to outline + label
    strand_colors : identity-based strand colors — a ``{name: color}`` map (with
                    ``strand_names``) or a list aligned to strand order. Keeps a
                    given strand the *same* color across panels.
    relax_iters   : passed through to the layout (0 disables relaxation)
    loop_tightness : margin on each arm's angular wedge in the ``tree`` layout
                    (default 1.5); lower to shrink multiloop radii / shorten
                    junction linkers (too low → arms collide). No effect on radial.

    Returns
    -------
    matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    seq = sequence.replace("&", "").replace("+", "").upper()

    if structure is None and pairs is None and engine is not None:
        result = engine.mfe(*sequence.split("&")) if "&" in sequence else engine.mfe(sequence)
        structure = result.structure
    if layout is None:
        layout = layout_structure(
            sequence, structure, pairs=pairs, nicks=nicks,
            relax_iters=relax_iters, repel_range=repel_range, method=method,
            loop_tightness=loop_tightness,
        )

    if color == "auto":
        color = "strand" if layout.n_strands > 1 else "structure"

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    xy = layout.coords

    # backbone (nick gaps already omitted by the layout)
    for i, j in layout.backbone:
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=style.BACKBONE, lw=1.3, zorder=1)
    # base-pair rungs
    for i, j in layout.rungs:
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=style.RUNG, lw=1.0, zorder=1)
    # pseudoknot chords
    for i, j in layout.crossing:
        ax.plot(
            xy[[i, j], 0], xy[[i, j], 1],
            color=style.C_ACCENT, lw=1.0, ls="--", alpha=0.8, zorder=1,
        )

    # bases — drawn as circles in DATA units so they scale with the layout and
    # never visually merge regardless of figure / panel size
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle

    cols, mappable = _base_colors(seq, layout, color, accessibility,
                                  strand_names=strand_names, strand_colors=strand_colors)
    # every circle gets the SAME thin outline; strands are told apart by fill color
    patches = [Circle((xy[i, 0], xy[i, 1]), base_radius) for i in range(len(seq))]
    ax.add_collection(PatchCollection(
        patches, facecolors=cols, edgecolors="#333333",
        linewidths=0.6, zorder=3, match_original=False,
    ))
    # base letters drawn as data-unit glyphs so the letter:ball ratio is constant
    if base_text:
        _draw_base_letters(ax, xy, seq, base_radius)

    # strand names + per-strand position numbers, placed by ONE collision
    # resolver so names and numbers also avoid each other (not just bases)
    if labels:
        names = strand_names or (
            [f"S{k + 1}" for k in range(layout.n_strands)]
            if layout.n_strands > 1 else [None]
        )
        name_specs = _label_specs(xy, layout.strand_lens, names, base_radius,
                                  layout.rungs + layout.crossing)
        num_specs = _number_specs(xy, layout.rungs, base_radius,
                                  layout.strand_lens)
        label_pos = _place_all_labels(ax, name_specs, num_specs, xy, base_radius)
        if nick_markers:
            for k in layout.nicks:
                mid = (xy[k - 1] + xy[k]) / 2
                ax.plot(*mid, marker="x", color=style.NICK, ms=7, mew=1.6, zorder=5)

    # region overlays
    if highlight:
        _outline_positions(ax, xy, [i for s, e in highlight for i in range(s, e)],
                           radius=base_radius * 1.35)
    if domains:
        from strider.viz.annotate import overlay_domains

        overlay_domains(ax, layout, domains, radius=base_radius * 1.35)

    if mappable is not None:
        cb = ax.figure.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("unpaired probability")

    # explicit limits with padding; include resolved label positions so nothing
    # is clipped (patches/text do not grow the data limits on their own)
    if len(xy):
        import numpy as np

        pts = xy
        if labels and len(label_pos):
            pts = np.vstack([xy, label_pos])
        pad = base_radius + 0.8
        ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
        ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, family="sans-serif")
    return ax


def _draw_base_letters(ax, xy, seq, radius):
    """Draw nucleotide letters as data-unit glyphs (constant letter:ball ratio)."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    fp = FontProperties(family="sans-serif", weight="bold")
    for i, base in enumerate(seq):
        tp = TextPath((0, 0), base, size=1, prop=fp)
        bb = tp.get_extents()
        w, h = bb.width or 1.0, bb.height or 1.0
        scale = min(1.15 * radius / h, 1.5 * radius / w)
        tr = (Affine2D()
              .translate(-(bb.x0 + w / 2), -(bb.y0 + h / 2))
              .scale(scale)
              .translate(xy[i, 0], xy[i, 1]))
        ax.add_patch(PathPatch(tp.transformed(tr), color="white", lw=0, zorder=4))


def _label_specs(xy, strand_lens, names, radius, rungs):
    """Build collision-resolved label specs — one bold name per strand, anchored
    beside that strand's own backbone. The offset direction is the base-pair axis
    (away from the partner) at a mid-strand anchor, so on a thin coaxial duplex
    the name sits *beside* its column instead of collapsing onto it (a
    ``mean − centroid`` offset vanishes when a strand straddles the centroid).
    Returns ``(anchor, direction, text, fontsize, bold, colour)`` specs; the
    final placement (jointly with the position numbers) is done by
    :func:`_place_all_labels`."""
    import numpy as np

    centroid = xy.mean(axis=0) if len(xy) else np.zeros(2)
    partner: dict[int, int] = {}
    for a, b in rungs:
        partner[a] = b
        partner[b] = a

    specs = []
    off = 0
    for sid, L in enumerate(strand_lens):
        if names[sid] and L > 0:
            # Anchor mid-strand, but nudged off multiples of 10 and the strand
            # ends so the name doesn't sit on a position number or an endpoint.
            anchor_local = max(1, min(L, L // 2))
            if L > 6 and anchor_local % 10 in (0, 1, 2, 9):
                anchor_local = min(L - 2, anchor_local + 3)
            gi = off + anchor_local - 1
            d = xy[gi] - xy[partner[gi]] if gi in partner else xy[gi] - centroid
            nrm = float(np.linalg.norm(d))
            d = d / nrm if nrm > 1e-6 else np.array([1.0, 0.0])
            specs.append((xy[gi], d, names[sid], 9, True, "#222222"))
        off += L
    return specs


def _number_specs(xy, rungs, radius, strand_lens=None, step=10):
    """Compute per-strand position-number label records (no drawing).

    Returns a list of ``(anchor_xy, init_pos, half_extent, text)`` tuples. The
    leader direction comes from an angular openness search around each base; the
    final overlap resolution is done jointly with the strand-name tags in
    :func:`_place_all_labels`, so numbers and names avoid each other.

    Numbering is **per strand** (each strand restarts at 1) and always marks each
    strand's first and last residue, so where one strand starts and ends is clear
    from the count + colour alone (no 5'/3' or nick markers needed)."""
    import numpy as np

    if not len(xy):
        return []
    centroid = xy.mean(axis=0)
    partner: dict[int, int] = {}
    for a, b in rungs:
        partner[a] = b
        partner[b] = a

    # global base index -> per-strand display number
    if not strand_lens:
        strand_lens = [len(xy)]
    targets: dict[int, int] = {}
    bounds_of: dict[int, tuple[int, int]] = {}       # gi -> (strand start, end)
    off = 0
    for L in strand_lens:
        if L <= 0:
            continue
        locals_to_mark = {1, L}                      # always show start & end
        for m in range(step, L + 1, step):
            if 2 < m < L - 1:                        # skip multiples next to ends
                locals_to_mark.add(m)
        for t in locals_to_mark:
            gi = off + t - 1
            targets[gi] = t
            bounds_of[gi] = (off, off + L - 1)
        off += L

    def _nearest_other(c, exclude):
        """Distance from point ``c`` to the closest base other than ``exclude``."""
        dd = xy - c
        dist = np.hypot(dd[:, 0], dd[:, 1])
        for j in exclude:
            dist[j] = np.inf
        return float(dist.min())

    # The "natural" outward direction for a base: away from its pairing partner
    # (paired) or perpendicular to the local backbone (unpaired).
    def _natural_dir(i):
        if i in partner:
            d = xy[i] - xy[partner[i]]
        else:
            lo, hi = bounds_of.get(i, (0, len(xy) - 1))
            prev_i, next_i = max(lo, i - 1), min(hi, i + 1)
            t = xy[next_i] - xy[prev_i]
            d = np.array([-t[1], t[0]]) if float(np.linalg.norm(t)) > 1e-6 \
                else (xy[i] - centroid)
        nrm = float(np.linalg.norm(d))
        return d / nrm if nrm > 1e-6 else np.array([0.0, -1.0])

    probes = (radius + 0.6, radius + 1.0, radius + 1.6)
    records = []
    for i in sorted(targets):
        # Choose the leader direction by a local openness search rather than the
        # raw H-bond axis: in a tight corner/junction *both* sides of that axis
        # are crowded, so we sweep all angles and take the one whose whole leader
        # stays clearest, biased toward the natural direction so open helices and
        # loops keep their tidy radial/perpendicular look.
        nat = _natural_dir(i)
        d, best = nat, -1e18
        for k in range(48):
            ang = 2.0 * np.pi * k / 48
            cand = np.array([np.cos(ang), np.sin(ang)])
            clear = min(_nearest_other(xy[i] + pr * cand, (i,)) for pr in probes)
            score = clear + 0.5 * float(cand @ nat)
            if score > best:
                best, d = score, cand
        text = str(targets[i])
        init = xy[i] + (radius + 0.95) * d
        half = np.array([0.11 * len(text) / 2 + 0.13, 0.22])
        records.append((xy[i], init, half, text))
    return records


def _place_all_labels(ax, name_specs, num_specs, coords, radius):
    """Place strand-name tags *and* per-strand position numbers with a single
    overlap resolver, so the two kinds of label avoid each other (not only the
    bases). Each starts at its anchor-offset spot, then is iteratively pushed
    apart from every other label and away from the nearest base until nothing
    overlaps. Position-number labels also get a leader tick to their base."""
    import numpy as np

    # unified label list: (anchor, init_pos, half, text, fontsize, bold, colour,
    # has_leader). Names have no leader and a larger box; numbers carry a leader.
    entries = []
    for a, d, text, fs, bold, col in name_specs:
        a = np.asarray(a, float)
        entries.append((a, a + (radius + 0.7) * np.asarray(d, float),
                        np.array([0.26 * len(text) / 2 + 0.16, 0.30]),
                        text, fs, bold, col, False))
    for anchor, init, half, text in num_specs:
        entries.append((np.asarray(anchor, float), np.asarray(init, float),
                        np.asarray(half, float), text, 7, False, "#555555", True))

    if not entries:
        return np.empty((0, 2))
    pos = np.array([e[1] for e in entries])
    half = np.array([e[2] for e in entries])

    for _ in range(120):
        moved = False
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                delta = pos[a] - pos[b]
                ox = (half[a][0] + half[b][0]) - abs(delta[0])
                oy = (half[a][1] + half[b][1]) - abs(delta[1])
                if ox > 0 and oy > 0:  # AABB overlap → separate on the shallow axis
                    if ox <= oy:
                        s = (ox / 2 + 0.03) * (1.0 if delta[0] >= 0 else -1.0)
                        pos[a][0] += s
                        pos[b][0] -= s
                    else:
                        s = (oy / 2 + 0.03) * (1.0 if delta[1] >= 0 else -1.0)
                        pos[a][1] += s
                        pos[b][1] -= s
                    moved = True
        for a in range(len(pos)):
            dd = coords - pos[a]
            dist = np.hypot(dd[:, 0], dd[:, 1])
            jm = int(dist.argmin())
            clear = radius + float(half[a].max()) + 0.05
            if dist[jm] < clear:
                push = pos[a] - coords[jm]
                pn = float(np.linalg.norm(push)) or 1.0
                pos[a] += push / pn * (clear - dist[jm])
                moved = True
        if not moved:
            break

    for (anchor, _, _, text, fs, bold, col, leader), p, h in zip(entries, pos, half):
        if leader:                                       # leader tick to the base
            seg = p - anchor
            sn = float(np.linalg.norm(seg)) or 1.0
            u = seg / sn
            start = anchor + radius * u
            elbow = p - (float(h.max()) + 0.1) * u
            if float(np.dot(elbow - start, u)) > 0:      # only if there's room
                ax.plot([start[0], elbow[0]], [start[1], elbow[1]],
                        color="#9aa0a6", lw=0.8, zorder=2)
        ax.text(p[0], p[1], text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=col,
                family="sans-serif", zorder=6, clip_on=False)
    return pos


def _outline_positions(ax, xy, positions, color=None, radius=0.55):
    """Draw a data-unit ring around a set of base positions."""
    from matplotlib.patches import Circle

    color = color or style.C_WARN
    for i in positions:
        ax.add_patch(Circle((xy[i, 0], xy[i, 1]), radius, fill=False,
                            edgecolor=color, lw=2.2, zorder=2.5, alpha=0.9))


def draw_complex(
    complex_or_seqs,
    structure: str | None = None,
    *,
    engine=None,
    ax: "Axes | None" = None,
    names: list[str] | None = None,
    color: str = "strand",
    reorder: bool = True,
    relax_iters: int = 0,
    repel_range: float = 0.85,
    method: str = "auto",
    loop_tightness: float = 1.5,
    **kw,
):
    """
    Draw a multi-strand complex.

    ``complex_or_seqs`` may be a :class:`strider.tube.Complex` (uses ``.sequences``
    / ``.strand_names``), a list of sequence strings, or a ``&``-joined string.
    The complex is folded via ``engine.mfe(*seqs)`` when ``structure`` is None.

    Multi-strand pseudoknot-freeness is *order-dependent*, so by default
    (``reorder=True``) the strands are reordered to a concatenation order that
    minimises base-pair crossings before layout — a closed network (e.g. a
    dendrimer) that is heavily pseudoknotted in the input order then draws flat.
    Strand colours and names follow their strand through the reorder, so a given
    strand keeps its identity.  De-overlap of multi-junction networks is handled
    by the layout itself (``method="auto"`` falls back to the space-aware tree
    layout only when the compact radial one would overlap), so the spring
    relaxation is off by default; ``relax_iters``/``repel_range`` enable it for
    ``method="radial"`` if wanted.  ``loop_tightness`` (default 1.5) shrinks the
    tree layout's multiloop radii — lower it (e.g. 0.8) to pull crowded junction
    linkers in; too low and the arms start to collide.
    """
    from strider.structure.dot_bracket import parse_pairs
    from strider.viz import geometry as _geom

    # resolve sequences + names
    if hasattr(complex_or_seqs, "sequences"):
        seqs = list(complex_or_seqs.sequences)
        names = names or list(getattr(complex_or_seqs, "strand_names", []))
    elif isinstance(complex_or_seqs, str):
        seqs = complex_or_seqs.split("&")
    else:
        seqs = list(complex_or_seqs)

    if structure is None and engine is not None:
        result = engine.mfe(*seqs)
        structure = result.structure
        # engine.mfe folds a multi-strand complex order-invariantly and returns
        # the structure in the winning strand order; adopt that order here so the
        # pairs, names and colours stay aligned to the sequence being drawn.
        order = list(result.strand_order)
        if order and order != list(range(len(seqs))):
            seqs = [seqs[i] for i in order]
            if names and len(names) == len(order):
                names = [names[i] for i in order]
            sc = kw.get("strand_colors")
            if isinstance(sc, list) and len(sc) == len(order):
                kw["strand_colors"] = [sc[i] for i in order]

    # work over an explicit pair list so a reorder only has to remap indices
    # (no dot-bracket round-trip, no second fold)
    pairs = parse_pairs(structure) if structure else None

    if reorder and pairs and len(seqs) > 2:
        strand_lens = [len(s) for s in seqs]
        order = _geom.best_strand_order(pairs, strand_lens)
        if order != list(range(len(seqs))):
            seqs = [seqs[i] for i in order]
            if names and len(names) == len(order):
                names = [names[i] for i in order]
            sc = kw.get("strand_colors")
            if isinstance(sc, list) and len(sc) == len(order):
                kw["strand_colors"] = [sc[i] for i in order]
            pairs = _geom.remap_pairs(pairs, strand_lens, order)

    joined = "&".join(seqs)
    return draw_structure(
        joined, pairs=pairs, engine=engine, ax=ax, color=color,
        strand_names=names, relax_iters=relax_iters, repel_range=repel_range,
        method=method, loop_tightness=loop_tightness, **kw,
    )
