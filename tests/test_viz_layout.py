"""Tests for the Layout2D orchestrator (nick handling, strand bookkeeping)."""

from __future__ import annotations

import numpy as np

from strider.structure.mfe import _parse_strands
from strider.viz.layout import layout_structure


class TestSingleStrand:
    def test_hairpin_backbone_is_complete(self):
        lo = layout_structure("GGGGAAAACCCC", "((((....))))")
        assert lo.n == 12
        assert lo.backbone == [(i, i + 1) for i in range(11)]
        assert lo.nicks == []
        assert lo.n_strands == 1
        assert len(lo.rungs) == 4

    def test_pseudoknot_pairs_go_to_crossing(self):
        lo = layout_structure("A" * 12, "((([[[)))]]]")
        assert len(lo.crossing) == 3
        assert len(lo.rungs) == 3


class TestMultiStrand:
    def test_backbone_omits_exactly_the_nick(self):
        seq = "GGGGAAAACCCC&GGGGTTTTCCCC"
        clean, nicks, _ = _parse_strands(seq, "dna")
        struct = "((((((((((((&))))))))))))"
        lo = layout_structure(seq, struct)
        assert lo.nicks == nicks == [12]
        # nick segment (11,12) must be absent; all others present
        assert (11, 12) not in lo.backbone
        assert len(lo.backbone) == (lo.n - 1) - len(nicks)

    def test_strand_bookkeeping_matches_parse_strands(self):
        seq = "AAAA&TTTTTT&CCC"
        clean, nicks, _ = _parse_strands(seq, "dna")
        lo = layout_structure(seq, "." * len(clean), nicks=nicks)
        assert lo.strand_lens == [4, 6, 3]
        assert lo.n_strands == 3
        assert list(lo.strand_of) == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2]

    def test_explicit_pairs_override_structure(self):
        lo = layout_structure("AAAA", pairs=[(0, 3)])
        assert lo.rungs == [(0, 3)]
        assert np.isfinite(lo.coords).all()
