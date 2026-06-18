"""Example 16: a two-strand toehold complex — strand identity colors + side tags."""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strider import ThermoEngine
from strider.viz.structure2d import draw_complex

_here = pathlib.Path(__file__).parent
eng = ThermoEngine(material="dna", celsius=37, backend="native")

# invader (toehold + branch-migration domain) binding an output strand
INVADER = "TCTCTAGGGAAATTTCCCTGAAG"
OUTPUT = "CTTCAGGGAAATTTCCC"
fig, ax = plt.subplots(figsize=(5, 6))
draw_complex([INVADER, OUTPUT], engine=eng, ax=ax, names=["Invader", "Output"],
             title="Toehold complex (Invader : Output)")
fig.savefig(_here / "gallery_16_toehold.png", dpi=130, bbox_inches="tight")
print("wrote gallery_16_toehold.png")
