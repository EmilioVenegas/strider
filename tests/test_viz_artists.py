"""Artist-level tests for 2-D structure rendering (no pixel comparison)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from strider.viz.annotate import draw_accessibility_track, per_position_accessibility
from strider.viz.structure2d import draw_complex, draw_structure


def _base_circle_count(ax):
    from matplotlib.collections import PatchCollection
    return sum(len(c.get_paths()) for c in ax.collections
               if isinstance(c, PatchCollection))


def _rings(ax):
    # overlay halos / domain rings are unfilled Circle patches
    return [p for p in ax.patches if not p.get_fill()]


class TestDrawStructure:
    def test_one_circle_per_base(self):
        ax = draw_structure("GGGGAAAACCCC", "((((....))))")
        assert _base_circle_count(ax) == 12
        plt.close(ax.figure)

    def test_title_and_equal_aspect(self):
        ax = draw_structure("GGGGAAAACCCC", "((((....))))", title="hp")
        assert ax.get_title() == "hp"
        assert ax.get_aspect() == 1.0
        plt.close(ax.figure)

    def test_accessibility_adds_colorbar(self, engine):
        seq = "GGGGAAAACCCCTTTTTTGGGGAAAACCCC"
        acc = per_position_accessibility(seq, engine)
        ax = draw_structure(seq, engine=engine, color="accessibility", accessibility=acc)
        # a colorbar adds a second Axes to the figure
        assert len(ax.figure.axes) >= 2
        plt.close(ax.figure)

    def test_highlight_adds_halos(self):
        ax = draw_structure("GGGGAAAACCCC", "((((....))))", highlight=[(0, 3)])
        assert len(_rings(ax)) == 3
        plt.close(ax.figure)


class TestDrawComplex:
    def test_nick_marker_off_by_default(self, engine):
        # strands are told apart by colour + per-strand count, so no nick 'x' marker
        ax = draw_complex(["GGGGAAAACCCC", "GGGGTTTTCCCC"], engine=engine, names=["A", "B"])
        assert [ln for ln in ax.lines if ln.get_marker() == "x"] == []
        plt.close(ax.figure)

    def test_nick_marker_opt_in(self, engine):
        ax = draw_complex(["GGGGAAAACCCC", "GGGGTTTTCCCC"], engine=engine,
                          names=["A", "B"], nick_markers=True)
        nick = [ln for ln in ax.lines if ln.get_marker() == "x"]
        assert len(nick) == 1
        plt.close(ax.figure)

    def test_accepts_joined_string(self, engine):
        ax = draw_complex("GGGGAAAACCCC&GGGGTTTTCCCC", engine=engine)
        assert ax.get_aspect() == 1.0
        plt.close(ax.figure)

    def test_reorder_flattens_pseudoknotted_complex(self):
        # 4 strands; pairs join strand 0<->2 and 1<->3, which interleave in the
        # input order (drawn as dashed pseudoknot chords) but are flat once the
        # strands are reordered.  No engine needed: structure is explicit.
        strands = ["GC", "GC", "GC", "GC"]
        struct = "((&[[&))&]]"

        def n_chords(ax):
            return sum(1 for ln in ax.lines if ln.get_linestyle() == "--")

        kept = draw_complex(strands, structure=struct, reorder=False, relax_iters=0)
        flat = draw_complex(strands, structure=struct, reorder=True, relax_iters=0)
        assert n_chords(kept) > 0    # crossings remain in input order
        assert n_chords(flat) == 0   # reorder removed them
        plt.close(kept.figure)
        plt.close(flat.figure)


class TestAccessibilityTrack:
    def test_track_renders_with_domains(self, engine):
        seq = "GGGGAAAACCCCTTTTTTGGGGAAAACCCC"
        ax = draw_accessibility_track(seq, engine=engine, domains={"toehold": (0, 6)})
        assert ax.images  # imshow strip present
        plt.close(ax.figure)

    def test_requires_engine_or_values(self):
        with pytest.raises(ValueError):
            draw_accessibility_track("ACGT")
