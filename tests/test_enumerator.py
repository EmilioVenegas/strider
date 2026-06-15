"""
Tests for the template-free domain-level reaction enumerator (Peppercorn paradigm).

The central validation is that, given only the strand topology of a toehold-mediated
strand-displacement (TMSD) reaction, the enumerator *derives* the canonical
bind → branch-migration → release network with no hand-written reactions, assigns
physically-signed detailed-balance rates, and emits a simulable mantis CRNetwork.
"""
import importlib.util
import math

import pytest

from strider.thermo.engine import ThermoEngine
from strider.kinetics.enumerator import (
    DomainReactionEnumerator,
    EnumerationResult,
    Complex,
    _is_complementary,
)

RT = 1.987204e-3 * (37.0 + 273.15)


def make_engine():
    return ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)


# ── domain TMSD fixtures ──────────────────────────────────────────────────────
TMSD_DOMAINS = {"t": "CCCT", "b": "ACGTACGTACGT"}  # 4-nt toehold, 12-nt branch
TMSD_STRANDS = {"Invader": ["t", "b"], "Output": ["b"], "Base": ["b*", "t*"]}


def tmsd_result(**kw) -> EnumerationResult:
    enum = DomainReactionEnumerator(domains=TMSD_DOMAINS, engine=make_engine(), **kw)
    return enum.enumerate(strands=TMSD_STRANDS, initial_complexes=[["Output", "Base"]])


class TestComplementarity:
    def test_token_complement(self):
        assert _is_complementary("a", "a*")
        assert _is_complementary("a*", "a")
        assert not _is_complementary("a", "b*")
        assert not _is_complementary("a", "a")


class TestCanonicalForm:
    def test_single_strand_identity(self):
        c1 = Complex(strands=(("S", ("a", "b")),), bonds=frozenset())
        c2 = Complex(strands=(("S", ("a", "b")),), bonds=frozenset())
        assert c1 == c2 and hash(c1) == hash(c2)

    def test_strand_order_invariance(self):
        # Same physical 2-strand complex, listed in opposite strand order, with
        # the single bond re-indexed accordingly, must canonicalise equal.
        a = Complex(
            strands=(("A", ("x",)), ("B", ("x*",))),
            bonds=frozenset({frozenset({(0, 0), (1, 0)})}),
        )
        b = Complex(
            strands=(("B", ("x*",)), ("A", ("x",))),
            bonds=frozenset({frozenset({(0, 0), (1, 0)})}),
        )
        assert a == b and hash(a) == hash(b)

    def test_different_structure_distinct(self):
        bound = Complex(
            strands=(("A", ("x",)), ("B", ("x*",))),
            bonds=frozenset({frozenset({(0, 0), (1, 0)})}),
        )
        unbound = Complex(
            strands=(("A", ("x",)), ("B", ("x*",))),
            bonds=frozenset(),
        )
        assert bound != unbound


class TestTMSDNetwork:
    def test_enumerates_canonical_network(self):
        res = tmsd_result()
        assert not res.truncated
        # Invader, Output, substrate(Base·Output), ternary intermediate,
        # product(Base·Invader) = 5 complexes.
        assert len(res.complexes) == 5
        mechanisms = sorted(r.mechanism for r in res.reactions)
        assert mechanisms == ["bind", "migrate"]

    def test_toehold_bind_rate(self):
        res = tmsd_result()
        bind = next(r for r in res.reactions if r.mechanism == "bind")
        # Zhang–Winfree 4-nt toehold kf is ~1e5 M^-1 s^-1.
        assert 1e4 < bind.kf < 1e6

    def test_branch_migration_releases_output(self):
        res = tmsd_result()
        migrate = next(r for r in res.reactions if r.mechanism == "migrate")
        product_names = {c.name for c in migrate.products}
        # The freed Output strand must appear as a standalone product complex.
        assert "Output" in product_names
        # ... and the migration product set has two complexes (split happened).
        assert len(migrate.products) == 2

    def test_detailed_balance_consistency(self):
        # For every reaction kf/kr == exp(-ΔG/RT) with ΔG = G(products)-G(reactants)
        # computed from the same helix energies the enumerator uses.
        enum = DomainReactionEnumerator(domains=TMSD_DOMAINS, engine=make_engine())
        res = enum.enumerate(strands=TMSD_STRANDS, initial_complexes=[["Output", "Base"]])
        for r in res.reactions:
            dg = enum._reaction_dg(list(r.reactants), list(r.products))
            keq = r.kf / r.kr
            assert keq == pytest.approx(math.exp(-dg / RT), rel=1e-6)

    def test_all_rates_positive_finite(self):
        res = tmsd_result()
        for r in res.reactions:
            assert r.kf > 0 and math.isfinite(r.kf)
            assert r.kr > 0 and math.isfinite(r.kr)


@pytest.mark.skipif(
    importlib.util.find_spec("mantis") is None,
    reason="mantis is an optional peer dependency",
)
class TestCRNetworkIntegration:
    def test_to_crnetwork(self):
        res = tmsd_result()
        crn = res.to_crnetwork()
        assert crn.n_species == 5
        # two reversible reactions → four directed reactions
        assert crn.n_reactions == 4

    def test_simulate_releases_output(self):
        res = tmsd_result()
        crn = res.to_crnetwork()
        substrate = next(s for s in crn.species if s.startswith("Base_Output"))
        ic = {s: 0.0 for s in crn.species}
        ic["Invader"] = 1e-7
        ic[substrate] = 1e-7
        out = crn.simulate(ic, (0, 3600))
        assert out.success
        # The displacement should liberate a substantial fraction of the Output.
        assert out.final()["Output"] > 2e-8


class TestPseudoknotAndTermination:
    def test_pseudoknot_rejected(self):
        # Two strands whose only way to pair on two domains each would cross
        # (a*…b* against a…b in the same direction) must not form a pseudoknot.
        enum = DomainReactionEnumerator(
            domains={"a": "CCCT", "b": "GGGA"}, engine=make_engine()
        )
        res = enum.enumerate(strands={"X": ["a", "b"], "Y": ["a*", "b*"]})
        # Whatever forms, no enumerated complex may contain crossing bonds.
        for c in res.complexes:
            assert not enum._is_pseudoknotted(c)

    def test_max_complexes_cap_terminates(self):
        # A self-complementary polymerising motif could blow up; the cap must
        # stop it and flag truncation rather than hang.
        enum = DomainReactionEnumerator(
            domains={"a": "CCCT", "b": "AGGG"},
            engine=make_engine(),
            max_complexes=8,
        )
        res = enum.enumerate(strands={"M": ["a", "b", "a*", "b*"]})
        assert res.truncated
        # overshoot is bounded to the ≤2 products of the reaction that tripped it
        assert len(res.complexes) <= 8 + 2

    def test_strict_raises_on_overflow(self):
        enum = DomainReactionEnumerator(
            domains={"a": "CCCT", "b": "AGGG"},
            engine=make_engine(),
            max_complexes=4,
            strict=True,
        )
        with pytest.raises(RuntimeError):
            enum.enumerate(strands={"M": ["a", "b", "a*", "b*"]})


class TestValidation:
    def test_unknown_domain_raises(self):
        enum = DomainReactionEnumerator(domains={"a": "CCCT"}, engine=make_engine())
        with pytest.raises(KeyError):
            enum.enumerate(strands={"S": ["a", "z"]})
