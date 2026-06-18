"""
Image-baseline tests (pytest-mpl), behind the ``slow`` marker.

Run / compare:   pytest -m slow --mpl
(Re)generate:    pytest -m slow --mpl-generate-path=tests/baseline

The native layout is deterministic (no RNG, fixed relaxation iterations), so
baselines are stable.  Tolerance is generous to absorb matplotlib/font patch
differences across versions.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from strider.viz.structure2d import draw_complex, draw_structure

pytestmark = pytest.mark.slow


@pytest.mark.mpl_image_compare(tolerance=25, remove_text=True)
def test_hairpin_structure_baseline():
    # sequence and dot-bracket must be the same length (28 nt here)
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_structure("GGGGGGGGGGGGAAAACCCCCCCCCCCC", "((((((((((((....))))))))))))",
                   ax=ax, base_text=False)
    return fig


@pytest.mark.mpl_image_compare(tolerance=25, remove_text=True)
def test_duplex_complex_baseline(engine):
    fig, ax = plt.subplots(figsize=(5, 6))
    draw_complex(["GGGGAAAACCCC", "GGGGTTTTCCCC"], engine=engine, ax=ax,
                 names=["A", "B"], base_text=False)
    return fig
