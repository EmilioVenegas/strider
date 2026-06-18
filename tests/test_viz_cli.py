"""Smoke tests for the `strider draw` CLI subcommands."""

from __future__ import annotations

import json

from strider.cli import main

H1 = "TCAACATCAGTCTGATACCTCCCTCCTTATCAGACTGA"


def _out(tmp_path, name):
    return str(tmp_path / name)


class TestDrawCLI:
    def test_structure(self, tmp_path):
        out = _out(tmp_path, "s.png")
        assert main(["draw", "structure", "GGGGAAAACCCC", "--out", out]) == 0
        assert (tmp_path / "s.png").stat().st_size > 0

    def test_structure_accessibility_color(self, tmp_path):
        out = _out(tmp_path, "s.png")
        rc = main(["draw", "structure", H1, "--color", "accessibility", "--out", out])
        assert rc == 0 and (tmp_path / "s.png").stat().st_size > 0

    def test_complex(self, tmp_path):
        out = _out(tmp_path, "c.png")
        rc = main(["draw", "complex", "GGGGAAAACCCC", "GGGGTTTTCCCC",
                   "--names", "A", "B", "--out", out])
        assert rc == 0 and (tmp_path / "c.png").stat().st_size > 0

    def test_accessibility_with_domains(self, tmp_path):
        out = _out(tmp_path, "a.png")
        rc = main(["draw", "accessibility", H1,
                   "--domains", '{"toehold":[0,6]}', "--out", out])
        assert rc == 0 and (tmp_path / "a.png").stat().st_size > 0

    def test_arc_strand_multistrand(self, tmp_path):
        out = _out(tmp_path, "arc.png")
        rc = main(["draw", "arc", "GGGGAAAACCCC&GGGGTTTTCCCC",
                   "--color-by", "strand", "--out", out])
        assert rc == 0 and (tmp_path / "arc.png").stat().st_size > 0

    def test_reaction(self, tmp_path, cha_seqs):
        spec = tmp_path / "spec.json"
        spec.write_text(json.dumps(cha_seqs))
        out = _out(tmp_path, "rxn.png")
        rc = main(["draw", "reaction", "--spec", str(spec), "--rates", "--out", out])
        assert rc == 0 and (tmp_path / "rxn.png").stat().st_size > 0

    def test_svg_output(self, tmp_path):
        out = _out(tmp_path, "s.svg")
        assert main(["draw", "structure", "GGGGAAAACCCC", "--out", out]) == 0
        assert (tmp_path / "s.svg").exists()
