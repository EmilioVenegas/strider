"""Gallery: ``minimal`` thumbnail mode for ``draw_complex``.

The default ``draw_complex`` renders every base as a labeled ball — great at full
page size, unreadable once the figure is shrunk into a small inset (the balls
collide into noise).  ``minimal=True`` collapses each strand to a single outlined
ribbon (a plain perimeter) tracing its fold, labeled only by its strand ID.  The
layout geometry is identical to the full view, so the fold still reads, but it
stays legible at small sizes.

This script renders, for a four-strand junction complex:
  - top: the full per-base view next to the minimal ribbon view, and
  - bottom: a strip of the minimal view re-saved at shrinking sizes, to show it
    survives down to thumbnail scale where the full view would smear.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strider import ThermoEngine
from strider.viz.structure2d import draw_complex

_here = pathlib.Path(__file__).parent


def rc(s: str) -> str:
    """Reverse complement of a DNA strand."""
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


# ---------------------------------------------------------------------------
# A four-arm junction: four strands, each pairing half-and-half with its two
# neighbours around the ring, so the MFE fold is a closed four-way junction.
# ---------------------------------------------------------------------------
a = "GACCGATTCAGT"
b = "TGCAACGGTACC"
c = "ATGGCTTAGGCA"
d = "CCAGTATCGAAT"

S1 = a + b
S2 = rc(b) + c
S3 = rc(c) + d
S4 = rc(d) + rc(a)

strands = [S1, S2, S3, S4]
names = ["S1", "S2", "S3", "S4"]
colors = {"S1": "#4A90E2", "S2": "#E94A3F", "S3": "#50E3C2", "S4": "#F5A623"}

eng = ThermoEngine(material="dna", celsius=37, backend="native")

# --- 1. full vs minimal, side by side --------------------------------------
fig, (ax_full, ax_min) = plt.subplots(1, 2, figsize=(13, 6.5))

draw_complex(
    strands, engine=eng, ax=ax_full, names=names, strand_colors=colors,
    title="full — per-base balls + labels",
)
draw_complex(
    strands, engine=eng, ax=ax_min, names=names, strand_colors=colors,
    minimal=True, title="minimal — strand ribbons + ID labels",
)

out_main = _here / "gallery_23_minimal.png"
fig.savefig(out_main, dpi=150, bbox_inches="tight")
print(f"Wrote {out_main}")

# --- 2. the minimal view rendered at genuinely shrinking sizes -------------
# The panels have real, different physical widths (width_ratios), so the same
# fold is *rendered* at decreasing scale. The ribbon width is in data units, so
# it scales down with the structure and stays proportional — never looking fat
# in the small panel — which is the whole point of the mode.
sizes_in = [3.0, 1.8, 1.1]
fig2, axes = plt.subplots(
    1, len(sizes_in), figsize=(sum(sizes_in) + 1.0, 3.4),
    gridspec_kw={"width_ratios": sizes_in},
)
for ax, side in zip(axes, sizes_in):
    draw_complex(
        strands, engine=eng, ax=ax, names=names, strand_colors=colors,
        minimal=True, title=f'{side}" wide',
    )

out_insets = _here / "gallery_23_minimal_insets.png"
fig2.savefig(out_insets, dpi=150, bbox_inches="tight")
print(f"Wrote {out_insets}")
