"""
Centralized colors and matplotlib styling for strider visualization.

This module defines the shared palette, nucleotide / base-pair / per-strand
color maps, and a :func:`style_context` helper that applies the strider figure
look (fonts, dpi, tight bbox) **without** mutating global ``rcParams`` or forcing
a backend at import time. Importing :mod:`strider.viz.style` therefore has no
side effects, so it is safe to import from anywhere in the package.
"""

from __future__ import annotations

import contextlib


# --- Qualitative palette (aligned with paper/figures/_style.py) --------------
C_NATIVE = "#4C78A8"   # strider native backend
C_NUPACK = "#E45756"   # NUPACK / Vienna comparison
C_TORCH = "#54A24B"    # torch differentiable
C_ACCENT = "#B279A2"
C_WARN = "#F58518"

# --- Nucleotide fill colors (vivid pastel) -----------------------------------
NT_COLORS: dict[str, str] = {
    "A": "#F2A65A",   # warm orange
    "T": "#6FA8DC",   # blue
    "U": "#6FA8DC",
    "C": "#89C997",   # green
    "G": "#E8786F",   # coral
}

# --- Base-pair colors by pair type (extracted from arc.py) -------------------
PAIR_COLORS: dict[str, str] = {
    "GC": "#d44", "CG": "#d44",
    "AT": "#44d", "TA": "#44d",
    "AU": "#44d", "UA": "#44d",
    "GU": "#da4", "UG": "#da4",
    "GT": "#da4", "TG": "#da4",
}

# --- Per-strand qualitative cycle (vivid pastel, distinct) --------------------
STRAND_CYCLE: list[str] = [
    "#6FA8DC", "#F2A65A", "#89C997", "#C39BD3",
    "#F4A6C0", "#6FD4C8", "#F2D06B", "#A89BE0",
]

# --- Structural element colors -----------------------------------------------
BACKBONE = "#8c8c8c"
RUNG = "#cccccc"
NICK = "#b22222"

# --- Secondary-structure element colors (vivid pastel; for color="structure") -
ELEMENT_COLORS: dict[str, str] = {
    "stem": "#6FA8DC",       # helix / stacked pair  (blue)
    "hairpin": "#F2A65A",    # hairpin loop          (orange)
    "interior": "#89C997",   # interior loop / bulge (green)
    "multiloop": "#C39BD3",  # multiloop             (purple)
    "exterior": "#6FD4C8",   # exterior / dangling   (teal)
}


def element_color(kind: str) -> str:
    """Color for a secondary-structure element type."""
    return ELEMENT_COLORS.get(kind, "gray")

# --- Figure rcParams applied via style_context (never at import) -------------
_RC: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "mathtext.fontset": "dejavusans",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


def nt_color(base: str) -> str:
    """Fill color for a nucleotide (gray for unknown letters)."""
    return NT_COLORS.get(base.upper(), "gray")


def strand_color(idx: int) -> str:
    """Color for the ``idx``-th strand (cycles through :data:`STRAND_CYCLE`)."""
    return STRAND_CYCLE[idx % len(STRAND_CYCLE)]


def pair_color(a: str, b: str) -> str:
    """Color for a base pair given its two bases (gray for non-canonical)."""
    return PAIR_COLORS.get((a + b).upper(), "#888")


def accessibility_cmap():
    """
    Colormap for per-base accessibility / unpaired probability in ``[0, 1]``.

    Low (paired / occluded) → cool gray; high (open / accessible) → warm orange.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "strider_access", ["#2c3e50", "#bdc3c7", C_WARN]
    )


@contextlib.contextmanager
def style_context(**overrides):
    """
    Temporarily apply the strider figure rcParams.

    Wraps :func:`matplotlib.rc_context`, so global rcParams are restored on exit
    and the active backend is never changed. STIX fonts gracefully fall back to
    matplotlib's default when unavailable.

    Parameters
    ----------
    **overrides
        rcParam keys to override on top of the strider defaults.
    """
    import matplotlib as mpl

    rc = dict(_RC)
    rc.update(overrides)
    with mpl.rc_context(rc):
        yield
