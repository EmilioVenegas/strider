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


class TestDrawReactionStep:
    def test_single_step(self, engine):
        fig = reaction.draw_reaction_step(
            [("miRNA", "TAGCTTATCAGACTGATGTTGA")],
            [("dup", ["GGGGCCCC", "GGGGCCCC"])],
            engine=engine, ddg=-5.0, label="R1",
        )
        assert len(fig.axes) >= 2
        plt.close(fig)
