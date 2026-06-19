"""
Layout orchestration: turn a sequence/structure into drawable 2D geometry.

:func:`layout_structure` resolves base pairs and strand breaks, runs the native
radial layout in :mod:`strider.viz.geometry`, and packages everything a renderer
needs into a :class:`Layout2D` — coordinates, backbone segments (with nick gaps
omitted), base-pair rungs, pseudoknot chords, and per-base strand identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from strider.structure.dot_bracket import parse_pairs
from strider.structure.mfe import _parse_strands
from strider.viz import geometry as _geom


@dataclass
class Layout2D:
    """Drawable geometry for a (possibly multi-strand) secondary structure."""

    coords: np.ndarray                       # (n, 2)
    backbone: list[tuple[int, int]]          # (i, i+1) segments, nick gaps omitted
    rungs: list[tuple[int, int]]             # nested base pairs
    crossing: list[tuple[int, int]]          # pseudoknot pairs (draw as chords)
    nicks: list[int]                         # strand-break indices
    strand_of: np.ndarray                    # (n,) strand id per base
    strand_lens: list[int] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.coords)

    @property
    def n_strands(self) -> int:
        return len(self.strand_lens)


def _strand_of(n: int, nicks: list[int]) -> tuple[np.ndarray, list[int]]:
    """Per-base strand id and per-strand lengths from nick positions."""
    strand_of = np.zeros(n, dtype=int)
    bounds = [0, *nicks, n]
    lens: list[int] = []
    for sid in range(len(bounds) - 1):
        lo, hi = bounds[sid], bounds[sid + 1]
        strand_of[lo:hi] = sid
        lens.append(hi - lo)
    return strand_of, lens


def layout_structure(
    sequence: str,
    structure: str | None = None,
    *,
    pairs: list[tuple[int, int]] | None = None,
    nicks: list[int] | None = None,
    relax_iters: int = 0,
    repel_range: float = 0.85,
    method: str = "auto",
    spacing: float = 1.0,
    rise: float = 1.0,
    loop_tightness: float = 1.5,
) -> Layout2D:
    """
    Compute drawable 2D geometry for ``sequence``.

    Parameters
    ----------
    sequence    : nucleotide sequence; may contain ``&``/``+`` strand separators
    structure   : dot-bracket (may also contain separators). If both ``structure``
                  and ``pairs`` are None the caller must pass ``pairs``.
    pairs       : explicit base pairs over the concatenated index space
                  (overrides ``structure``)
    nicks       : explicit strand breaks; inferred from separators otherwise
    relax_iters : spring-relaxation iterations (auto-skipped for tiny structures;
                  pass 0 to disable). Only applied to the ``radial`` method — the
                  ``tree`` method is overlap-free by construction.
    method      : ``"radial"`` (simple RNAplot-style), ``"tree"`` (space-aware
                  loop-tree, spreads multi-junction networks), or ``"auto"``
                  (default: ``tree`` for branched / >=3-strand structures, else
                  ``radial`` so simple folds keep their compact look)
    loop_tightness : margin multiplier on each sub-tree's angular wedge in the
                  ``tree`` layout (default 1.5); lower it to shrink multiloop
                  radii and pull long junction linkers in (too low → arms collide).
                  No effect on the ``radial`` layout.
    """
    # resolve strand breaks + clean concatenated sequence
    if nicks is None:
        clean_seq, nicks, _ = _parse_strands(sequence, "dna")
    else:
        clean_seq = sequence.replace("&", "").replace("+", "")
    n = len(clean_seq)

    # resolve pairs
    if pairs is None:
        pairs = parse_pairs(structure) if structure else []
    # drop any pairs that fall outside the sequence (length mismatch safety)
    pairs = [(i, j) for i, j in pairs if 0 <= i < n and 0 <= j < n]

    nested, crossing = _geom.classify_pairs(pairs)
    nick_set = set(nicks)
    if method == "auto":
        # Prefer the simple, compact radial layout; only fall back to the
        # space-aware tree layout when radial would actually overlap (a branched
        # structure whose arms collide). This keeps already-clean radial drawings
        # untouched and reserves the tree layout for the cases that need it.
        coords = _geom.radial_layout(nested, n, nicks, spacing=spacing, rise=rise)
        if _geom.is_branched(nested, n, nicks):
            segs = ([(i, i + 1) for i in range(n - 1) if (i + 1) not in nick_set]
                    + nested)
            if _geom.has_segment_crossing(coords, segs):
                coords = _geom.tree_layout(nested, n, nicks, spacing=spacing,
                                           rise=rise, safety=loop_tightness)
    elif method == "tree":
        coords = _geom.tree_layout(nested, n, nicks, spacing=spacing, rise=rise,
                                   safety=loop_tightness)
    else:
        coords = _geom.radial_layout(nested, n, nicks, spacing=spacing, rise=rise)
        if relax_iters and n > 40:
            coords = _geom.relax(coords, nested, nicks, iterations=relax_iters,
                                 spacing=spacing, repel_range=repel_range)

    backbone = [(i, i + 1) for i in range(n - 1) if (i + 1) not in nick_set]
    strand_of, strand_lens = _strand_of(n, nicks)

    return Layout2D(
        coords=coords,
        backbone=backbone,
        rungs=nested,
        crossing=crossing,
        nicks=list(nicks),
        strand_of=strand_of,
        strand_lens=strand_lens,
    )
