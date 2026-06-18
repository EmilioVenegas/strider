"""Example 19: a four-strand immobile 4-way (Holliday-style) junction.

Stress-tests label placement: four arms radiate from one crowded central
multiloop, so every strand's end (per-strand 1 / 24) and the four position-10
labels next to the hinge all compete for the small gaps around the junction.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strider import ThermoEngine
from strider.viz.structure2d import draw_complex

_here = pathlib.Path(__file__).parent


def _rc(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


# four distinct, GC-balanced arm sequences; each strand spans two adjacent arms
# joined by a short single-stranded hinge so the junction opens into four arms.
X1, X2, X3, X4 = "CGATCAGGTCA", "TGCACTGAACG", "ACGTGTCATGT", "GTAGCTGACCT"
H = "TT"
strands = [_rc(X4) + H + X1, _rc(X1) + H + X2, _rc(X2) + H + X3, _rc(X3) + H + X4]
names = ["S1", "S2", "S3", "S4"]

eng = ThermoEngine(material="dna", celsius=37, backend="native")
fig, ax = plt.subplots(figsize=(8, 8))
draw_complex(strands, engine=eng, ax=ax, names=names,
             strand_colors=["#6FA8DC", "#F2A65A", "#89C997", "#C39BD3"],
             title="Four-way junction (4 strands)")
fig.savefig(_here / "gallery_19_fourway.png", dpi=130, bbox_inches="tight")
print("wrote gallery_19_fourway.png")
