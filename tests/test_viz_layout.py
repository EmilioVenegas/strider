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


class TestMethodSelection:
    def _mindist(self, lo):
        from scipy.spatial.distance import pdist
        return pdist(lo.coords).min()

    def test_auto_keeps_radial_when_planar(self):
        # A small multiloop whose radial layout is already crossing-free: auto
        # must NOT switch it to the tree layout (the rule falls back only when
        # radial is branched AND actually overlaps).  Coords must match radial.
        seq, db = "G" * 24, "((..((...))..((...))..))"
        auto = layout_structure(seq, db, method="auto")
        radial = layout_structure(seq, db, method="radial")
        assert np.allclose(auto.coords, radial.coords)

    def test_auto_falls_back_to_tree_when_radial_crosses(self):
        # A branched fold whose radial layout self-intersects: auto must detect
        # the crossing and fall back to the tree layout.  The tree result is
        # crossing-free; its tightest spacing (~0.73, two helices meeting at a
        # junction) is well clear of the radial collision (~0.05 overlap).
        from strider.viz import geometry as g
        from strider.structure.dot_bracket import parse_pairs

        db = "..()()(((.)..)..)(((.).()())...)"
        n = len(db)            # 32
        seq = "G" * n
        nested, _ = g.classify_pairs(parse_pairs(db))
        segs = [(i, i + 1) for i in range(n - 1)] + nested

        radial = layout_structure(seq, db, method="radial")
        tree = layout_structure(seq, db, method="tree")
        auto = layout_structure(seq, db, method="auto")

        # radial genuinely overlaps + crosses; tree is clean
        assert g.has_segment_crossing(radial.coords, segs) is True
        assert g.has_segment_crossing(tree.coords, segs) is False
        # auto fell back to tree: crossing-free, no base collisions, == tree
        assert g.has_segment_crossing(auto.coords, segs) is False
        assert self._mindist(auto) > 0.5
        assert np.allclose(auto.coords, tree.coords)

    def test_loop_tightness_shrinks_multiloops(self):
        # lower loop_tightness reserves a smaller wedge per arm -> smaller loop
        # radii -> shorter junction linkers (the longest backbone bond drops).
        hp = "((((((((((....))))))))))"
        db = "((((" + "." + hp + "." + hp + "." + hp + "." + "))))"
        seq = "G" * len(db)

        def max_bond(lo):
            xy = lo.coords
            return max(np.linalg.norm(xy[i] - xy[i + 1]) for i in range(lo.n - 1))

        loose = layout_structure(seq, db, method="tree", loop_tightness=1.5)
        tight = layout_structure(seq, db, method="tree", loop_tightness=0.6)
        assert max_bond(tight) < max_bond(loose)
        # default matches an explicit 1.5
        assert np.allclose(
            layout_structure(seq, db, method="tree").coords, loose.coords
        )

    def test_explicit_method_is_honoured(self):
        seq, db = "G" * 24, "((..((...))..((...))..))"
        tree = layout_structure(seq, db, method="tree")
        radial = layout_structure(seq, db, method="radial")
        # same pairs/topology, different coordinates
        assert tree.rungs == radial.rungs
        assert not np.allclose(tree.coords, radial.coords)

    def test_simple_hairpin_stays_radial(self):
        # auto must not switch a plain hairpin away from the radial path
        seq, db = "G" * 12, "((((....))))"
        assert np.allclose(
            layout_structure(seq, db).coords,
            layout_structure(seq, db, method="radial").coords,
        )
