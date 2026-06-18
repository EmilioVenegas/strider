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
