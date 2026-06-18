"""
Visualization subsystem for strider (static matplotlib).

Public API
----------
- :func:`~strider.viz.structure2d.draw_structure` / ``draw_complex`` — native
  2-D secondary-structure and multi-strand complex drawing (no ViennaRNA).
- :func:`~strider.viz.reaction.draw_cascade` / ``draw_reaction_step`` — generic
  reaction / toehold-cascade panels.
- :func:`~strider.viz.annotate.draw_accessibility_track` — toehold accessibility.
- :func:`~strider.viz.arc.arc_diagram`, :func:`~strider.viz.mountain_plot.mountain_plot`,
  :func:`~strider.viz.mountain_plot.energy_landscape`,
  :func:`~strider.viz.circuit_diagram.cha_circuit`.
- :func:`~strider.viz.layout.layout_structure` — coordinate engine (advanced).
- :mod:`strider.viz.style` — shared palette and ``style_context``.

Importing a name pulls in matplotlib only at that point.
"""

from __future__ import annotations

from strider.viz.annotate import draw_accessibility_track, per_position_accessibility
from strider.viz.arc import arc_diagram
from strider.viz.circuit_diagram import cha_circuit
from strider.viz.layout import Layout2D, layout_structure
from strider.viz.mountain_plot import energy_landscape, mountain_plot
from strider.viz.reaction import draw_cascade, draw_reaction_step
from strider.viz.structure2d import draw_complex, draw_structure

__all__ = [
    "draw_structure",
    "draw_complex",
    "draw_cascade",
    "draw_reaction_step",
    "draw_accessibility_track",
    "per_position_accessibility",
    "arc_diagram",
    "mountain_plot",
    "energy_landscape",
    "cha_circuit",
    "layout_structure",
    "Layout2D",
]
