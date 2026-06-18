"""
Domain and toehold-accessibility annotations.

Per-base accessibility is the unpaired probability ``1 - Σ_j P(i, j)`` taken from
the engine's base-pair probability matrix (the engine's
``toehold_accessibility`` returns only a single joint scalar, so the per-position
layer is computed here).  Helpers draw a 1-D accessibility track and outline
named domain / toehold regions on an existing 2-D structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from strider.viz import style

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from strider.viz.layout import Layout2D


def per_position_accessibility(
    sequence: str, engine, *, positions: list[int] | None = None
) -> dict[int, float]:
    """
    Per-base unpaired probability from ``engine.pairs(sequence)``.

    Returns ``{position: prob}``.  ``positions`` restricts the result to a subset.
    """
    probs = engine.pairs(sequence)
    n = probs.shape[0]
    idx = positions if positions is not None else range(n)
    return {i: float(1.0 - probs[i].sum()) for i in idx}


def domain_positions(domain_map: dict) -> dict[str, list[int]]:
    """
    Normalize a domain map to ``{name: [positions]}``.

    Accepts ``{name: (start, end)}`` half-open spans or ``{name: [positions]}``.
    """
    out: dict[str, list[int]] = {}
    for name, spec in domain_map.items():
        if isinstance(spec, tuple) and len(spec) == 2 and all(isinstance(v, int) for v in spec):
            out[name] = list(range(spec[0], spec[1]))
        else:
            out[name] = list(spec)
    return out


def overlay_domains(
    ax: "Axes",
    layout: "Layout2D",
    domains: dict,
    *,
    outline: bool = True,
    label: bool = True,
    radius: float = 0.55,
) -> None:
    """Outline and label named domain regions on a 2-D structure axes."""
    from matplotlib.patches import Circle

    xy = layout.coords
    norm = domain_positions(domains)
    for d_idx, (name, positions) in enumerate(norm.items()):
        positions = [p for p in positions if 0 <= p < len(xy)]
        if not positions:
            continue
        col = style.strand_color(d_idx)
        if outline:
            for p in positions:
                ax.add_patch(Circle((xy[p, 0], xy[p, 1]), radius, fill=False,
                                    edgecolor=col, lw=2.4, zorder=2.5, alpha=0.9))
        if label:
            centroid = xy[positions].mean(axis=0)
            ax.annotate(name, centroid, textcoords="offset points", xytext=(0, 14),
                        ha="center", fontsize=8, color=col, fontweight="bold")


def draw_accessibility_track(
    sequence: str,
    engine=None,
    accessibility: dict[int, float] | None = None,
    domains: dict | None = None,
    ax: "Axes | None" = None,
    title: str | None = None,
):
    """
    Draw a 1-D accessibility strip: position on x, unpaired-probability color.

    Optional ``domains`` are drawn as labeled brackets beneath the strip.
    Returns the Axes.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    seq = sequence.replace("&", "").replace("+", "").upper()
    n = len(seq)
    if accessibility is None:
        if engine is None:
            raise ValueError("provide either accessibility or an engine")
        accessibility = per_position_accessibility(sequence, engine)
    values = np.array([accessibility.get(i, 0.0) for i in range(n)])

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, n / 6), 1.8))

    cmap = style.accessibility_cmap()
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    ax.imshow(values[None, :], aspect="auto", cmap=cmap, norm=norm,
              extent=(-0.5, n - 0.5, 0, 1))
    ax.set_yticks([])
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel("position")

    if domains:
        for d_idx, (name, positions) in enumerate(domain_positions(domains).items()):
            positions = [p for p in positions if 0 <= p < n]
            if not positions:
                continue
            lo, hi = min(positions), max(positions)
            col = style.strand_color(d_idx)
            ax.plot([lo, hi], [-0.18, -0.18], color=col, lw=3,
                    clip_on=False, solid_capstyle="butt")
            ax.annotate(name, ((lo + hi) / 2, -0.32), ha="center", va="top",
                        fontsize=8, color=col, annotation_clip=False)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("unpaired prob.")

    ax.set_title(title or "Accessibility")
    return ax
