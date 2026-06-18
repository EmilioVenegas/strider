"""Example 17: the three-strand CHA captured complex (coaxial duplex + side tags)."""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strider import ThermoEngine
from strider.viz.structure2d import draw_complex

_here = pathlib.Path(__file__).parent
eng = ThermoEngine(material="dna", celsius=37, backend="native")

# order-ready hsa-miR-21-5p design (urotrace)
H1 = "TCAACATCAGTCTGATAACTTAATTAAGTTATCAGACTGACAAGTCGTTTACGACTAGAGGG"
H2 = "TCAGTCTGATAACTTAATTAAGTTATCAGACTGATGTTGA"
CP = "CCCTCTAGTCGTAAACGACTTG"
fig, ax = plt.subplots(figsize=(5, 11))
draw_complex([H2, H1, CP], engine=eng, ax=ax, names=["H2", "H1", "CP"],
             strand_colors=["#89C997", "#F2A65A", "#C39BD3"],
             title="Captured complex (H1 : H2 : CP)")
fig.savefig(_here / "gallery_17_captured.png", dpi=130, bbox_inches="tight")
print("wrote gallery_17_captured.png")
