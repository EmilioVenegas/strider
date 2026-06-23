"""Tests for the generic reaction / cascade renderer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from strider.bridge.mantis_bridge import CHABridge
from strider.viz import reaction


class TestNormalization:
    def test_species_label_variants(self):
        assert reaction._species_label(("H1", "ACGT")) == "H1"
        assert reaction._species_label("ACGT") == "ACGT"
        assert reaction._species_label([("a", "AC"), ("b", "GT")]) == "a·b"

    def test_species_payload(self):
        assert reaction._species_payload(("H1", ["AC", "GT"])) == ["AC", "GT"]
        assert reaction._species_payload("ACGT") == ["ACGT"]
        assert reaction._species_payload(("x", 5)) is None


class TestDrawCascade:
    def test_cha_bridge_cascade(self, engine, cha_seqs):
        bridge = CHABridge(cha_seqs, engine=engine)
        fig = reaction.draw_cascade(bridge, engine=engine, show_rates=True)
        # 4 steps (R1, R2, R3, leak); a populated multi-axes figure
        assert len(fig.axes) > 4
        # ΔΔG annotations appear somewhere in the figure text
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any("ΔΔG" in t for t in texts)
        assert any(t in ("R1", "R2", "R3", "leak") for t in texts)
        plt.close(fig)

    def test_explicit_steps(self, engine):
        steps = [
            ([("A", "GGGGCCCC")], [("B", "GGGGCCCC")], {"ddg": -3.0, "label": "step1"}),
        ]
        fig = reaction.draw_cascade(steps, engine=engine)
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_empty_raises(self, engine):
        with pytest.raises(ValueError):
            reaction.draw_cascade([], engine=engine)


class TestAssemblyLandscape:
    def test_cha_bridge_landscape(self, engine, cha_seqs):
        bridge = CHABridge(cha_seqs, engine=engine)
        ax_e, ax_v = reaction.draw_assembly_landscape(bridge, engine=engine)
        fig = ax_e.figure
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        # energy column carries per-step ΔΔG (mathtext), net ΔG and step labels
        assert any("Delta" in t for t in texts)
        assert any("net" in t for t in texts)
        assert any("R1" in t for t in texts)
        # the catalyst recycle annotation is drawn by default for a bridge
        assert any("recycled" in t for t in texts)
        # one horizontal energy level per macrostate (4 for a CHA with CP)
        from matplotlib.collections import LineCollection
        levels = [c for c in ax_e.collections if isinstance(c, LineCollection)]
        assert sum(len(c.get_segments()) for c in levels) == 4
        plt.close(fig)

    def test_explicit_states(self, engine):
        states = [
            {"energy": 0.0, "components": [("A", "GGGGCCCC"), ("B", "GGGGCCCC")]},
            {"energy": -6.0, "components": [("A·B", ["GGGGCCCC", "GGGGCCCC"])],
             "caption": "bound", "scale": 1.4},
        ]
        ax_e, ax_v = reaction.draw_assembly_landscape(
            states, engine=engine, recycle=False, step_labels=["k1"])
        texts = [t.get_text() for t in ax_e.texts]
        assert any("k1" in t for t in texts)
        # auto-captioned first state joins component labels; explicit one kept
        viz_texts = [t.get_text() for t in ax_v.texts]
        assert "bound" in viz_texts
        plt.close(ax_e.figure)

    def test_into_provided_axes(self, engine):
        fig, (a, b) = plt.subplots(1, 2)
        states = [
            {"energy": 0.0, "components": [("A", "GGGGCCCC")]},
            {"energy": -3.0, "components": [("A", "GGGGCCCC")]},
        ]
        out_e, out_v = reaction.draw_assembly_landscape(
            states, engine=engine, axes=(a, b), recycle=False)
        assert out_e is a and out_v is b
        plt.close(fig)

    def test_empty_raises(self, engine):
        with pytest.raises(ValueError):
            reaction.draw_assembly_landscape([], engine=engine)


class TestDrawReactionStep:
    def test_single_step(self, engine):
        fig = reaction.draw_reaction_step(
            [("miRNA", "TAGCTTATCAGACTGATGTTGA")],
            [("dup", ["GGGGCCCC", "GGGGCCCC"])],
            engine=engine, ddg=-5.0, label="R1",
        )
        assert len(fig.axes) >= 2
        plt.close(fig)
