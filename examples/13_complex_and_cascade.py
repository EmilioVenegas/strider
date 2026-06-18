"""
Example 13: Multi-strand complexes, toehold accessibility, and a whole cascade

Showcases the native (no-ViennaRNA) visualization layer:
- draw_structure / draw_complex : 2D folds of single strands and bound complexes
- draw_accessibility_track      : per-base toehold accessibility with a domain mark
- draw_cascade                  : the entire CHA reaction as a single figure

All panels are produced from strider results only (native DNA backend, 37 °C).
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "sans-serif", "mathtext.fontset": "dejavusans"})
import matplotlib.pyplot as plt

from strider import ThermoEngine
from strider.bridge.mantis_bridge import CHABridge
from strider.viz.annotate import draw_accessibility_track
from strider.viz.reaction import draw_cascade
from strider.viz.structure2d import draw_complex, draw_structure

_here = pathlib.Path(__file__).parent

# Order-ready CHA design for hsa-miR-21-5p
# (urotrace/results/hsa-miR-21-5p/sequences.txt; D1=6, D2=11, capture handle K=22 nt).
# H1 carries the 22-nt capture handle K at its 3' end and CP = K*, so in the
# H1·H2·CP complex the capture probe is fully bound to H1 (22 bp), not dangling.
SEQS = {
    "mirna": "TAGCTTATCAGACTGATGTTGA",
    "H1": "TCAACATCAGTCTGATAACTTAATTAAGTTATCAGACTGACAAGTCGTTTACGACTAGAGGG",
    "H2": "TCAGTCTGATAACTTAATTAAGTTATCAGACTGATGTTGA",
    "CP": "CCCTCTAGTCGTAAACGACTTG",
}

engine = ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01,
                      backend="native")
bridge = CHABridge(SEQS, engine=engine)

# ── 1. Single strands + multi-strand complexes ───────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("strider — native 2D structures & complexes (DNA, 37 °C)", fontsize=15)

draw_structure(SEQS["H1"], engine=engine, ax=axes[0][0], title="H1 hairpin")
draw_structure(SEQS["H2"], engine=engine, ax=axes[0][1], title="H2 hairpin")
draw_accessibility_track(SEQS["H1"], engine=engine, ax=axes[0][2],
                         domains={"D1 toehold": (0, 6)})
axes[0][2].set_title("H1 toehold accessibility")

# identity-based strand colors: a given strand keeps its color across all panels
STRAND_COL = {"mirna": "#4C78A8", "H1": "#F58518", "H2": "#54A24B", "CP": "#B279A2"}
cols = lambda *keys: [STRAND_COL[k] for k in keys]

draw_complex([SEQS["mirna"], SEQS["H1"]], engine=engine, ax=axes[1][0],
             names=["miRNA", "H1"], strand_colors=cols("mirna", "H1"),
             title="miRNA · H1  (initiation)")
draw_complex([SEQS["H1"], SEQS["H2"]], engine=engine, ax=axes[1][1],
             names=["H1", "H2"], strand_colors=cols("H1", "H2"),
             title="H1 · H2  (signal dimer)")
draw_complex([SEQS["H2"], SEQS["H1"], SEQS["CP"]], engine=engine, ax=axes[1][2],
             names=["H2", "H1", "CP"], strand_colors=cols("H2", "H1", "CP"),
             title="H1 · H2 · CP  (captured)")

fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(_here / "complex_and_cascade.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: complex_and_cascade.png")

# ── 2. The whole reaction cascade ────────────────────────────────────────────
fig_casc = draw_cascade(bridge, engine=engine, show_rates=True,
                        title="CHA reaction cascade (miR-21)")
fig_casc.savefig(_here / "cha_cascade.png", dpi=150, bbox_inches="tight")
plt.close(fig_casc)
print("Saved: cha_cascade.png")

print("Done.")
