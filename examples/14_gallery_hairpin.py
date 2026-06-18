"""Example 14: single hairpin — structure-element coloring + position numbers."""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strider import ThermoEngine
from strider.viz.structure2d import draw_structure

_here = pathlib.Path(__file__).parent
eng = ThermoEngine(material="dna", celsius=37, backend="native")

# a molecular-beacon-style hairpin
SEQ = "GCGAGCTTCAGGGAAATTTCCCTGAAGCTCGC"
fig, ax = plt.subplots(figsize=(5, 5))
draw_structure(SEQ, engine=eng, ax=ax, title="Molecular beacon (hairpin)")
fig.savefig(_here / "gallery_14_hairpin.png", dpi=130, bbox_inches="tight")
print("wrote gallery_14_hairpin.png")
