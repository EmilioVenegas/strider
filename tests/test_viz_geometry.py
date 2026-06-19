"""Invariant tests for the native 2D layout geometry (pure math, no matplotlib)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from strider.structure.dot_bracket import parse_pairs
from strider.viz import geometry as g


CASES = {
    "hairpin": ("((((....))))", 12),
    "multiloop": ("((..((...))..((...))..))", 24),
    "duplex": ("(((((((((((())))))))))))", 24),
    "pseudoknot": ("((([[[)))]]]", 12),
    "empty": ("............", 12),
    "bulge": ("(((..((....))))).", 17),
}


class TestClassifyPairs:
    def test_nested_kept_crossing_split(self):
        nested, crossing = g.classify_pairs(parse_pairs("((([[[)))]]]"))
        assert len(nested) == 3
        assert len(crossing) == 3
        # every crossing pair interleaves some nested pair
        assert all(any(g._crosses(c, npair) for npair in nested) for c in crossing)

    def test_pure_nested_has_no_crossing(self):
        nested, crossing = g.classify_pairs(parse_pairs("((((....))))"))
        assert crossing == []
        assert len(nested) == 4

    def test_deterministic(self):
        pairs = parse_pairs("((([[)))]]")
        assert g.classify_pairs(pairs) == g.classify_pairs(pairs)


class TestStrandOrder:
    # 4 strands of 2 nt each (indices A=[0,1] B=[2,3] C=[4,5] D=[6,7]); the
    # pairs join A-C and B-D, which interleave in the input order A,B,C,D but
    # are disjoint (flat) in the order A,C,B,D.
    PAIRS = [(0, 5), (1, 4), (2, 7), (3, 6)]
    LENS = [2, 2, 2, 2]

    def test_crossing_count_zero_iff_fully_nested(self):
        # crossing_count tallies interleaving pair-of-pairs; it is 0 exactly when
        # classify_pairs leaves nothing in the crossing set (a flat structure).
        for db in ("((((....))))", "((..((...))..((...))..))", "(((..((....))))).", "."*8):
            pairs = parse_pairs(db)
            assert g.crossing_count(pairs) == 0
            assert g.classify_pairs(pairs)[1] == []
        for db in ("((([[[)))]]]", "((([[)))]]"):
            pairs = parse_pairs(db)
            assert g.crossing_count(pairs) > 0
            assert len(g.classify_pairs(pairs)[1]) > 0

    def test_order_dependent_crossings(self):
        assert g.crossing_count(self.PAIRS) > 0                      # input order crosses
        flat = g.remap_pairs(self.PAIRS, self.LENS, [0, 2, 1, 3])    # A,C,B,D
        assert g.crossing_count(flat) == 0

    def test_best_order_finds_flat_arrangement(self):
        order = g.best_strand_order(self.PAIRS, self.LENS)
        assert g.crossing_count(g.remap_pairs(self.PAIRS, self.LENS, order)) == 0

    def test_best_order_never_worsens(self):
        # an already-flat structure must keep the identity order
        flat = g.remap_pairs(self.PAIRS, self.LENS, [0, 2, 1, 3])
        assert g.best_strand_order(flat, [2, 2, 2, 2]) == [0, 1, 2, 3]

    def test_remap_pairs_roundtrip(self):
        order = [0, 2, 1, 3]
        once = g.remap_pairs(self.PAIRS, self.LENS, order)
        lens2 = [self.LENS[i] for i in order]
        back = g.remap_pairs(once, lens2, [0, 2, 1, 3])  # this order is its own inverse
        assert sorted((min(a, b), max(a, b)) for a, b in back) == \
            sorted((min(a, b), max(a, b)) for a, b in self.PAIRS)

    def test_trivial_cases(self):
        assert g.best_strand_order([], [3, 3]) == [0, 1]
        assert g.best_strand_order([(0, 3)], [2, 2]) == [0, 1]


class TestRadialLayout:
    @pytest.mark.parametrize("name", list(CASES))
    def test_finite_coordinates(self, name):
        db, n = CASES[name]
        nested, _ = g.classify_pairs(parse_pairs(db))
        xy = g.radial_layout(nested, n)
        assert xy.shape == (n, 2)
        assert np.isfinite(xy).all()

    @pytest.mark.parametrize("name", list(CASES))
    def test_no_base_overlap(self, name):
        db, n = CASES[name]
        nested, _ = g.classify_pairs(parse_pairs(db))
        xy = g.radial_layout(nested, n)
        if n > 1:
            assert pdist(xy).min() > 1e-3

    @pytest.mark.parametrize("name", list(CASES))
    def test_paired_bases_at_ladder_distance(self, name):
        db, n = CASES[name]
        nested, _ = g.classify_pairs(parse_pairs(db))
        xy = g.radial_layout(nested, n)
        for i, j in nested:
            assert np.linalg.norm(xy[i] - xy[j]) == pytest.approx(1.0, abs=1e-6)

    def test_empty_structure_is_a_line(self):
        xy = g.radial_layout([], 8)
        assert np.allclose(xy[:, 1], 0.0)
        assert xy[1, 0] > xy[0, 0]

    def test_zero_length(self):
        assert g.radial_layout([], 0).shape == (0, 2)


def _nested(db, n):
    nested, _ = g.classify_pairs(parse_pairs(db))
    return nested, n


def _segment_crossings(xy, segs):
    """Count properly-intersecting segment pairs (shared endpoints excluded)."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    crossings = 0
    for i in range(len(segs)):
        a, b = segs[i]
        for j in range(i + 1, len(segs)):
            c, d = segs[j]
            if a in (c, d) or b in (c, d):
                continue
            p1, p2, p3, p4 = xy[a], xy[b], xy[c], xy[d]
            if ((ccw(p3, p4, p1) > 0) != (ccw(p3, p4, p2) > 0) and
                    (ccw(p1, p2, p3) > 0) != (ccw(p1, p2, p4) > 0)):
                crossings += 1
    return crossings


class TestTreeLayout:
    # branched / multi-junction structures (where tree layout earns its keep)
    BRANCHED = {
        "multiloop": ("((..((...))..((...))..))", 24),
        "4-way": ("((((...))((...))((...))))", 25),
        "bulge": ("(((..((....))))).", 17),
        "tRNA": ("(((((((..((((........)))).((((.........))))"
                 ".....(((((.......))))))))))))..", 73),
    }

    @pytest.mark.parametrize("name", list(BRANCHED))
    def test_overlap_free_and_no_crossings(self, name):
        db, n = self.BRANCHED[name]
        nested, _ = g.classify_pairs(parse_pairs(db))
        xy = g.tree_layout(nested, n)
        assert np.isfinite(xy).all()
        # no two bases collide ...
        assert pdist(xy).min() > 0.99
        # ... and no backbone/base-pair segments cross: a clean planar diagram
        segs = [(i, i + 1) for i in range(n - 1)] + nested
        assert _segment_crossings(xy, segs) == 0

    @pytest.mark.parametrize("name", list(BRANCHED))
    def test_paired_bases_at_ladder_distance(self, name):
        db, n = self.BRANCHED[name]
        nested, _ = g.classify_pairs(parse_pairs(db))
        xy = g.tree_layout(nested, n)
        for i, j in nested:
            assert np.linalg.norm(xy[i] - xy[j]) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("name", list(BRANCHED))
    def test_backbone_bonds_stay_short(self, name):
        # The reservation escalation keeps loops from over-inflating, so a
        # backbone step is never stretched far beyond one spacing.  (Full-reach
        # reservation could blow a crowded loop up and stretch its short linkers.)
        db, n = self.BRANCHED[name]
        nested, _ = g.classify_pairs(parse_pairs(db))
        xy = g.tree_layout(nested, n)
        bonds = [np.linalg.norm(xy[i] - xy[i + 1]) for i in range(n - 1)]
        assert max(bonds) < 3.0

    def test_crowded_junction_stays_compact(self):
        # A multiloop closing three long (10-bp) hairpins joined by single-nt
        # linkers — a "dendrimer in miniature".  Reserving each arm's full radial
        # reach would inflate the central loop and stretch the 1-nt linkers into
        # long bonds; the width-based escalation keeps it tight and planar.
        hp = "((((((((((....))))))))))"
        db = "((((" + "." + hp + "." + hp + "." + hp + "." + "))))"
        n = len(db)
        nested, _ = g.classify_pairs(parse_pairs(db))
        xy = g.tree_layout(nested, n)
        segs = [(i, i + 1) for i in range(n - 1)] + nested
        assert _segment_crossings(xy, segs) == 0           # planar
        assert pdist(xy).min() > 0.99                      # no collisions
        bonds = [np.linalg.norm(xy[i] - xy[i + 1]) for i in range(n - 1)]
        assert max(bonds) < 2.0                            # linkers not stretched

    def test_multi_strand_complex_is_clean(self):
        # a 3-way junction split across 3 strands (a nicked multiloop)
        seq = "GGGGAAAA&TTTTCCCCAAAA&TTTTGGGG"
        # pairs: strand0 5' with strand2 3', etc. — build a closed 3-strand ring
        nested = [(0, 27), (1, 26), (2, 25), (3, 24),
                  (8, 19), (9, 18), (10, 17), (11, 16)]
        n = 28
        nicks = [8, 20]
        xy = g.tree_layout(nested, n, nicks)
        assert np.isfinite(xy).all()
        assert pdist(xy).min() > 0.99

    def test_deterministic(self):
        nested, _ = g.classify_pairs(parse_pairs("((((...))((...))((...))))"))
        a = g.tree_layout(nested, 25)
        b = g.tree_layout(nested, 25)
        assert np.array_equal(a, b)

    def test_is_branched_detects_multiloop(self):
        assert g.is_branched(*_nested("((..((...))..((...))..))", 24))   # multiloop
        assert g.is_branched(*_nested("((((...))((...))((...))))", 25))  # 3-way
        assert not g.is_branched(*_nested("((((....))))", 12))           # plain hairpin
        assert not g.is_branched(*_nested("(((..((....))))).", 17))      # single bulge chain


class TestHasSegmentCrossing:
    def test_detects_proper_crossing(self):
        # two segments forming an X must be flagged as crossing
        coords = np.array([[0.0, 0.0], [2.0, 2.0],   # seg 0: (0,1) diagonal up
                           [0.0, 2.0], [2.0, 0.0]])  # seg 2: (2,3) diagonal down
        assert g.has_segment_crossing(coords, [(0, 1), (2, 3)]) is True

    def test_parallel_segments_do_not_cross(self):
        # two horizontal, vertically separated segments never intersect
        coords = np.array([[0.0, 0.0], [2.0, 0.0],
                           [0.0, 1.0], [2.0, 1.0]])
        assert g.has_segment_crossing(coords, [(0, 1), (2, 3)]) is False

    def test_shared_endpoint_is_not_a_crossing(self):
        # segments meeting at a common vertex (a backbone corner) are excluded
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        assert g.has_segment_crossing(coords, [(0, 1), (1, 2)]) is False

    def test_disjoint_far_segments_do_not_cross(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0],
                           [5.0, 5.0], [6.0, 5.0]])
        assert g.has_segment_crossing(coords, [(0, 1), (2, 3)]) is False


class TestStructureTree:
    def test_hairpin_decomposition(self):
        n = 12
        tree = g.build_structure_tree(parse_pairs("((((....))))"), n)
        assert tree.kind == "external"
        assert len(tree.children) == 1
        helix = tree.children[0]
        assert helix.kind == "helix"
        assert len(helix.pairs) == 4
        loop = helix.children[0]
        assert loop.unpaired == [4, 5, 6, 7]

    def test_multiloop_has_multiple_child_helices(self):
        tree = g.build_structure_tree(parse_pairs("((..((...))..((...))..))"), 24)
        outer = tree.children[0]  # external -> outer helix
        loop = outer.children[0]
        assert sum(c.kind == "helix" for c in loop.children) == 2


class TestRelax:
    def test_relax_is_deterministic(self):
        nested, _ = g.classify_pairs(parse_pairs("((((....))))"))
        xy = g.radial_layout(nested, 12)
        a = g.relax(xy, nested, iterations=40)
        b = g.relax(xy, nested, iterations=40)
        assert np.allclose(a, b)

    def test_relax_preserves_finiteness(self):
        nested, _ = g.classify_pairs(parse_pairs("((..((...))..((...))..))"))
        xy = g.radial_layout(nested, 24)
        out = g.relax(xy, nested, iterations=60)
        assert np.isfinite(out).all()

    def test_zero_iterations_noop(self):
        nested, _ = g.classify_pairs(parse_pairs("((((....))))"))
        xy = g.radial_layout(nested, 12)
        assert np.array_equal(g.relax(xy, nested, iterations=0), xy)

    def test_longer_repel_range_spreads_more(self):
        # a wider repulsion cutoff must push bases at least as far apart overall
        nested, _ = g.classify_pairs(parse_pairs("((..((...))..((...))..))"))
        xy = g.radial_layout(nested, 24)
        near = g.relax(xy, nested, iterations=80, repel_range=0.85)
        far = g.relax(xy, nested, iterations=80, repel_range=2.5)
        assert np.isfinite(far).all()
        assert pdist(far).max() > pdist(near).max()
