"""
Arc diagram visualization for secondary structures.

Draws sequence along a horizontal axis with semicircular arcs connecting
base-paired positions. Arcs are colored by pair type or probability; bases by
nucleotide or strand. Multi-strand inputs (``&``/``+`` separated) are supported:
strand breaks are marked and the backbone is split at each nick.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from strider.viz import style

if TYPE_CHECKING:
    import numpy as np


def arc_diagram(
    sequence: str,
    structure: str | None = None,
    pair_probs: "np.ndarray | None" = None,
    engine=None,
    ax=None,
    color_by: str = "type",
    title: str | None = None,
    min_prob: float = 0.1,
    nicks: list[int] | None = None,
):
    """
    Draw an arc diagram for a secondary structure.

    Parameters
    ----------
    sequence   : nucleotide sequence; may contain ``&``/``+`` strand separators
    structure  : dot-bracket string (computed via engine if None)
    pair_probs : (n,n) probability matrix (overrides structure if given)
    engine     : ThermoEngine (used only if structure and pair_probs are None)
    color_by   : 'type' (GC/AT/GU), 'probability', or 'strand'
    min_prob   : minimum pair probability to draw (when using pair_probs)
    nicks      : strand-break indices; inferred from separators otherwise
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from strider.structure.mfe import _parse_strands

    # resolve strand breaks
    if nicks is None:
        _, nicks, _ = _parse_strands(sequence, "dna")
    seq = sequence.replace("&", "").replace("+", "").upper()
    n = len(seq)
    strand_of = np.zeros(n, dtype=int)
    for k in nicks:
        strand_of[k:] += 1

    if ax is None:
        fig_w = max(8, n // 3)
        _, ax = plt.subplots(figsize=(fig_w, 4))

    # Resolve pairs / arcs
    arcs: list[tuple[int, int, float]] = []  # (i, j, weight)
    if pair_probs is not None:
        for i in range(n):
            for j in range(i + 4, n):
                p = float(pair_probs[i, j])
                if p >= min_prob:
                    arcs.append((i, j, p))
    else:
        if structure is None and engine is not None:
            seqs = sequence.split("&") if "&" in sequence else [sequence]
            structure = engine.mfe(*seqs).structure
        if structure:
            from strider.structure.dot_bracket import parse_pairs

            arcs = [(i, j, 1.0) for i, j in parse_pairs(structure)]

    # Backbone (split at nicks)
    seg_start = 0
    for k in sorted(nicks) + [n]:
        if k - 1 > seg_start:
            ax.hlines(0, seg_start, k - 1, color=style.BACKBONE, linewidth=1.5, zorder=1)
        seg_start = k

    # Nucleotides
    for i, base in enumerate(seq):
        c = style.strand_color(int(strand_of[i])) if color_by == "strand" else style.nt_color(base)
        ax.plot(i, 0, "o", color=c, markersize=8, zorder=3)
        ax.text(i, -0.15, base, ha="center", va="top", fontsize=7, color=c)

    # Nick markers
    for k in nicks:
        ax.plot(k - 0.5, 0, marker="x", color=style.NICK, ms=8, mew=1.8, zorder=4)

    # Arcs
    cmap = plt.cm.Blues if color_by == "probability" else None
    for i, j, weight in arcs:
        cx = (i + j) / 2
        radius = (j - i) / 2
        theta = [math.pi * k / 100 for k in range(101)]
        xs = [cx + radius * math.cos(t) for t in theta]
        ys = [radius * math.sin(t) for t in theta]

        if color_by == "probability":
            color = cmap(weight)
        elif color_by == "strand":
            color = "#777" if strand_of[i] == strand_of[j] else style.C_ACCENT
        else:
            color = style.pair_color(seq[i], seq[j])

        ax.plot(xs, ys, color=color, linewidth=1.0 + weight, alpha=0.7, zorder=2)

    ax.set_xlim(-1, n)
    ax.set_ylim(-0.4, max((j - i) / 2 for _, j, _ in arcs) * 1.1 if arcs else 1)
    ax.set_xlabel("Position")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    if title:
        ax.set_title(title)
    elif structure:
        ax.set_title(f"Arc diagram  |  {structure[:60]}{'...' if len(structure) > 60 else ''}")

    return ax
