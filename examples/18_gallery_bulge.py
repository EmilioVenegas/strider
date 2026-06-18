"""Example 18: a hairpin with an internal loop and a bulge — shows interior color."""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strider.viz.structure2d import draw_structure

_here = pathlib.Path(__file__).parent

# stem - interior loop - stem - bulge - stem - hairpin loop
STRUCT = "((((..((((.((((....))))))))..))))"
SEQ = ("GCGCAAGCGCAGCGCAAAAGCGCGCGCAAGCGC")[: len(STRUCT)]
fig, ax = plt.subplots(figsize=(5, 6))
draw_structure(SEQ, STRUCT, ax=ax, title="Interior loop + bulge")
fig.savefig(_here / "gallery_18_bulge.png", dpi=130, bbox_inches="tight")
print("wrote gallery_18_bulge.png")
