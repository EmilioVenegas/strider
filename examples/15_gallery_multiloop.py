"""Example 15: a three-way junction (single strand) — shows the multiloop color."""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strider.viz.structure2d import draw_structure

_here = pathlib.Path(__file__).parent

# explicit cloverleaf-like multiloop (sequence content is illustrative)
STRUCT = "(((....(((....)))....(((....)))....)))"
SEQ = ("GCAUGCA" * 6)[: len(STRUCT)]
fig, ax = plt.subplots(figsize=(5.5, 5.5))
draw_structure(SEQ, STRUCT, ax=ax, title="Three-way junction (multiloop)")
fig.savefig(_here / "gallery_15_multiloop.png", dpi=130, bbox_inches="tight")
print("wrote gallery_15_multiloop.png")
