"""
Template-free domain-level reaction enumerator (Peppercorn paradigm).

Where :class:`strider.dsd.DSDCompiler` keeps the *sequence* layer in sync with a
circuit whose reactions the user still writes by hand, this module *derives* the
reactions from the strand topology — the job Visual DSD / Peppercorn do.  Given a
set of strands written in domain space, it enumerates the reachable complexes and
the transitions between them, assigns detailed-balance rate constants from the
active :class:`~strider.thermo.engine.ThermoEngine`, and emits a ready-to-simulate
``mantis.CRNetwork``.

Pipeline (Frontier §5 / §2 of ``STRIDER_STATUS.md``)::

    strands + (concentrations) ─▶ enumerate complexes
                               ─▶ bind / 3-way branch-migration / open transitions
                               ─▶ detailed-balance rates (Zhang–Winfree kf, kr via ΔG)
                               ─▶ mantis.CRNetwork

Scope (v1, documented and deliberate)
-------------------------------------
The enumerator handles the **non-pseudoknotted, 3-way** core that is sufficient for
toehold-mediated strand-displacement cascades (TMSD, HCR-style polymerisation, the
displacement steps of CHA/seesaw circuits):

* **bind**     — a complementary *toehold* (short) domain pair, one in each
  complex, hybridises and merges the two complexes (bimolecular).  Long-domain and
  remote-toehold binding are off by default to keep the network finite and the
  initiation physical; ``include_leak=True`` adds slow blunt-end (zero-toehold)
  binding for leak-pathway analysis.
* **migrate**  — 3-way branch migration: an unbound domain adjacent to a junction
  displaces an identical incumbent domain bound across that junction.
* **open**     — a short (toehold-length) helix spontaneously dissociates; long
  helices are treated as effectively irreversible (they are never opened), which is
  both physical at 37 °C and the lever that keeps enumeration terminating.

4-way branch migration and intramolecular long-domain re-closure (hairpin folding)
are **not** modelled in v1; pseudoknotted bonds are never formed.

Example
-------
>>> from strider import ThermoEngine
>>> from strider.kinetics.enumerator import DomainReactionEnumerator
>>> enum = DomainReactionEnumerator(
...     domains={"t": "CCCT", "b": "ACGTACGTACGT"},   # t = 4-nt toehold
...     engine=ThermoEngine(material="dna"),
... )
>>> result = enum.enumerate(strands={
...     "Invader": ["t", "b"],
...     "Output":  ["b"],
...     "Base":    ["b*", "t*"],
... }, initial_complexes=[["Output", "Base"]])
>>> crn = result.to_crnetwork()           # mantis.CRNetwork
>>> result.summary()                      # doctest: +SKIP
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import TYPE_CHECKING

from strider.thermo.nn_dna import reverse_complement
from strider.kinetics.tmsd import toehold_kf, displacement_kf

if TYPE_CHECKING:  # pragma: no cover
    from strider.thermo.engine import ThermoEngine

R = 1.987204e-3  # kcal / (mol · K)

# A position inside a complex: (strand_index, domain_index_within_strand).
Position = tuple[int, int]


def _complement_token(tok: str) -> str:
    """Complement of a domain token: ``a`` ⇄ ``a*``."""
    return tok[:-1] if tok.endswith("*") else tok + "*"


def _is_complementary(a: str, b: str) -> bool:
    """True iff two domain tokens are complementary (same base, opposite star)."""
    return a == _complement_token(b)


@dataclass(frozen=True)
class Complex:
    """
    A resting (or transient) complex: an ordered tuple of strands plus a
    domain-level pairing.

    ``strands`` is a tuple of ``(strand_name, (token, token, ...))`` pairs.
    ``bonds`` is a frozenset of frozenset-pairs of :data:`Position`; each
    position appears in at most one bond.  Equality/hashing use the
    rotation/permutation-canonical form so the same physical complex is one node.
    """

    strands: tuple[tuple[str, tuple[str, ...]], ...]
    bonds: frozenset[frozenset[Position]]
    name: str = field(default="", compare=False)

    # ── canonicalisation ──────────────────────────────────────────────────────
    def _canonical_key(self) -> tuple:
        """
        Order-independent signature.  Two complexes are identical iff some
        relabelling of strand indices maps one onto the other; we take the
        lexicographically smallest signature over all strand permutations.
        """
        n = len(self.strands)
        best = None
        for perm in permutations(range(n)):
            # global offset of each strand under this ordering
            offset, off = [], 0
            order = list(perm)
            for si in order:
                offset.append(off)
                off += len(self.strands[si][1])
            inv = {si: rank for rank, si in enumerate(order)}

            names = tuple(self.strands[si][0] for si in order)
            toks = tuple(self.strands[si][1] for si in order)

            gbonds = []
            for bond in self.bonds:
                (sa, da), (sb, db) = tuple(bond)
                ga = offset[inv[sa]] + da
                gb = offset[inv[sb]] + db
                gbonds.append((min(ga, gb), max(ga, gb)))
            sig = (names, toks, tuple(sorted(gbonds)))
            if best is None or sig < best:
                best = sig
        return best

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Complex) and self._canonical_key() == other._canonical_key()

    def __hash__(self) -> int:
        return hash(self._canonical_key())

    # ── derived properties ──────────────────────────────────────────────────
    @property
    def strand_names(self) -> tuple[str, ...]:
        return tuple(s[0] for s in self.strands)

    @property
    def n_strands(self) -> int:
        return len(self.strands)

    def token(self, pos: Position) -> str:
        si, di = pos
        return self.strands[si][1][di]

    def partner(self, pos: Position) -> Position | None:
        for bond in self.bonds:
            if pos in bond:
                other = (bond - {pos})
                return next(iter(other))
        return None

    def is_bound(self, pos: Position) -> bool:
        return self.partner(pos) is not None

    def positions(self) -> list[Position]:
        out: list[Position] = []
        for si, (_, toks) in enumerate(self.strands):
            for di in range(len(toks)):
                out.append((si, di))
        return out


@dataclass(frozen=True)
class DomainReaction:
    """A reversible transition between multisets of complexes."""

    reactants: tuple[Complex, ...]
    products: tuple[Complex, ...]
    kf: float
    kr: float
    mechanism: str  # "bind" | "migrate" | "open"

    def reaction_string(self) -> str:
        lhs = " + ".join(c.name for c in self.reactants)
        rhs = " + ".join(c.name for c in self.products)
        return f"{lhs} <-> {rhs}"


@dataclass
class EnumerationResult:
    complexes: list[Complex]
    reactions: list[DomainReaction]
    truncated: bool = False

    def reaction_strings(self) -> list[str]:
        return [r.reaction_string() for r in self.reactions]

    def rates(self) -> dict[str, float]:
        """Map ``"A -> B"`` / ``"B -> A"`` directed reaction strings to rate constants."""
        out: dict[str, float] = {}
        for r in self.reactions:
            lhs = " + ".join(c.name for c in r.reactants)
            rhs = " + ".join(c.name for c in r.products)
            out[f"{lhs} -> {rhs}"] = r.kf
            out[f"{rhs} -> {lhs}"] = r.kr
        return out

    def to_crnetwork(self):
        """Build a ``mantis.CRNetwork`` from the enumerated reactions and rates."""
        from mantis import CRNetwork  # local import: mantis is an optional peer

        return CRNetwork.from_string(self.reaction_strings(), rates=self.rates())

    def summary(self) -> str:
        lines = [
            f"Enumerated {len(self.complexes)} complexes, "
            f"{len(self.reactions)} reactions"
            + (" (truncated)" if self.truncated else "")
            + ":",
        ]
        for c in self.complexes:
            lines.append(f"  {c.name:<16} <{' '.join(c.strand_names)}>")
        lines.append("")
        for r in self.reactions:
            lines.append(
                f"  {r.reaction_string():<44} "
                f"kf={r.kf:.2e}  kr={r.kr:.2e}  ({r.mechanism})"
            )
        return "\n".join(lines)


class DomainReactionEnumerator:
    """
    Enumerate the reaction network reachable from a set of domain-level strands.

    Parameters
    ----------
    domains          : base domain name → nucleotide sequence (DNA).  Star
                       complements (``a*``) are derived as reverse complements.
    engine           : ``ThermoEngine`` for helix ΔG (detailed-balance reverse rates).
                       Defaults to ``ThermoEngine(material="dna")``.
    toehold_max_len  : a helix this long or shorter is a *toehold* — it may
                       nucleate a bimolecular ``bind`` and may spontaneously
                       ``open``.  Longer helices are treated as irreversible.
    max_complexes    : hard cap on the number of distinct complexes enumerated
                       (guards against runaway polymerisation; raises if exceeded
                       only when ``strict=True``).
    max_strands      : skip any transition that would build a complex with more
                       strands than this.
    include_leak     : also enumerate slow blunt-end (zero-toehold) binding on
                       long complementary domains, for leak-pathway analysis.
    celsius          : temperature for the rate models (defaults to engine.celsius).
    """

    def __init__(
        self,
        domains: dict[str, str],
        engine: "ThermoEngine | None" = None,
        *,
        toehold_max_len: int = 8,
        max_complexes: int = 200,
        max_strands: int = 6,
        include_leak: bool = False,
        celsius: float | None = None,
        strict: bool = False,
    ) -> None:
        if engine is None:
            from strider.thermo.engine import ThermoEngine

            engine = ThermoEngine(material="dna")
        self.engine = engine
        self.material = engine.material
        self.celsius = celsius if celsius is not None else engine.celsius
        self.toehold_max_len = toehold_max_len
        self.max_complexes = max_complexes
        self.max_strands = max_strands
        self.include_leak = include_leak
        self.strict = strict

        self.domains = {n: s.upper().replace("U", "T") for n, s in domains.items()}
        self._strand_defs: dict[str, tuple[str, ...]] = {}
        self._energy_cache: dict[int, float] = {}
        self._RT = R * (self.celsius + 273.15)

    # ── sequence resolution ──────────────────────────────────────────────────
    def _domain_seq(self, tok: str) -> str:
        if tok in self.domains:
            return self.domains[tok]
        if tok.endswith("*") and tok[:-1] in self.domains:
            return reverse_complement(self.domains[tok[:-1]])
        raise KeyError(f"unknown domain {tok!r}; register {tok.rstrip('*')!r}")

    def _domain_len(self, tok: str) -> int:
        return len(self._domain_seq(tok))

    # ── public API ───────────────────────────────────────────────────────────
    def enumerate(
        self,
        strands: dict[str, list[str]],
        initial_complexes: list[list[str]] | None = None,
    ) -> EnumerationResult:
        """
        Enumerate the network reachable from a set of initial species.

        strands            : strand name → ordered list of domain tokens (5'→3').
        initial_complexes  : optional list of pre-formed complexes, each a list of
                             strand names that are perfectly hybridised on every
                             complementary domain pair (the usual "substrate" /
                             "fuel" duplexes).  Strands not named here start free.

        Returns an :class:`EnumerationResult`.
        """
        self._strand_defs = {n: tuple(ds) for n, ds in strands.items()}
        for toks in self._strand_defs.values():
            for tok in toks:
                self._domain_seq(tok)  # validate early

        named = set(initial_complexes and {n for grp in initial_complexes for n in grp} or [])
        initial: list[Complex] = []

        # free single strands
        for n, toks in self._strand_defs.items():
            if n not in named:
                initial.append(self._single_strand_complex(n))
        # pre-formed initial complexes
        for grp in initial_complexes or []:
            initial.append(self._assemble_initial_complex(grp))

        # Worklist BFS.  Each complex is *processed* exactly once: on processing
        # we apply its unimolecular moves and pair it with every already-registered
        # complex (itself included).  Because every unordered pair has a
        # later-registered member, and that member pairs against all earlier ones
        # when processed, every pair is covered at least once (reactions dedup by
        # key, so double-coverage is harmless).
        seen: dict[Complex, Complex] = {}
        order: list[Complex] = []
        worklist: list[Complex] = []
        for c in initial:
            cc, is_new = self._register(c, seen, order)
            if is_new:
                worklist.append(cc)

        reactions: dict[tuple, DomainReaction] = {}
        truncated = False
        head = 0
        while head < len(worklist):
            c = worklist[head]
            head += 1
            for rxn in self._unimolecular(c):
                self._add_reaction(rxn, reactions, seen, order, worklist)
                if self._over_cap(order):
                    truncated = True
                    break
            if truncated:
                break
            for other in list(order):
                for rxn in self._bimolecular(c, other):
                    self._add_reaction(rxn, reactions, seen, order, worklist)
                    if self._over_cap(order):
                        truncated = True
                        break
                if truncated:
                    break
            if truncated:
                break

        result = EnumerationResult(complexes=order, reactions=list(reactions.values()))
        result.truncated = truncated
        return result

    def _over_cap(self, order: list) -> bool:
        if len(order) > self.max_complexes:
            if self.strict:
                raise RuntimeError(
                    f"enumeration exceeded max_complexes={self.max_complexes}; "
                    "raise the cap or tighten toehold_max_len/max_strands"
                )
            return True
        return False

    # ── complex construction ─────────────────────────────────────────────────
    def _single_strand_complex(self, name: str) -> Complex:
        c = Complex(strands=((name, self._strand_defs[name]),), bonds=frozenset())
        return self._rename(c)

    def _assemble_initial_complex(self, group: list[str]) -> Complex:
        """Hybridise every complementary unbound domain pair across the named strands."""
        strands = tuple((n, self._strand_defs[n]) for n in group)
        positions = [
            (si, di)
            for si, (_, toks) in enumerate(strands)
            for di in range(len(toks))
        ]
        bonds: set[frozenset[Position]] = set()
        used: set[Position] = set()
        # greedy nearest complementary pairing between *different* strands,
        # antiparallel — enough for the standard fully-paired substrate duplex.
        for i, p in enumerate(positions):
            if p in used:
                continue
            tp = strands[p[0]][1][p[1]]
            for q in reversed(positions):
                if q in used or q[0] == p[0]:
                    continue
                tq = strands[q[0]][1][q[1]]
                if _is_complementary(tp, tq):
                    bonds.add(frozenset({p, q}))
                    used.add(p)
                    used.add(q)
                    break
        return self._rename(Complex(strands=strands, bonds=frozenset(bonds)))

    # ── reaction enumeration ─────────────────────────────────────────────────
    def _unimolecular(self, c: Complex):
        yield from self._open_reactions(c)
        yield from self._migration_reactions(c)

    def _open_reactions(self, c: Complex):
        """Dissociate a maximal helix if it is toehold-length (reversible)."""
        for helix in self._helices(c):
            length = sum(self._domain_len(c.token(p)) for p, _ in helix)
            if length > self.toehold_max_len:
                continue  # long helix: effectively irreversible, never opened
            new_bonds = set(c.bonds) - {frozenset({p, q}) for p, q in helix}
            products = self._split(Complex(c.strands, frozenset(new_bonds)))
            if products is None:
                continue
            products = tuple(self._rename(p) for p in products)
            dg = self._reaction_dg([c], products)
            kf = self._open_kf(length)            # s^-1 (dissociation)
            kr = self._reverse_rate(kf, -dg)      # reverse = bind; ΔG(reverse)=-ΔG(open)
            yield DomainReaction((c,), products, kf, kr, "open")

    def _migration_reactions(self, c: Complex):
        """
        3-way branch migration.  For each anchor bond A=(s1,i)↔B=(s2,j), look one
        domain *past the junction* on s1 for an unbound migrating domain D, and on
        s2 (antiparallel direction) for an incumbent domain C bound to F.  If D and
        F are the same identity (both complementary to C), D displaces F.
        """
        seen_moves: set = set()
        for bond in c.bonds:
            (sa, da), (sb, db) = sorted(tuple(bond))
            for (s1, i), (s2, j) in (((sa, da), (sb, db)), ((sb, db), (sa, da))):
                for dir1 in (+1, -1):
                    Dpos = (s1, i + dir1)
                    Cpos = (s2, j - dir1)  # antiparallel
                    if not self._valid_pos(c, Dpos) or not self._valid_pos(c, Cpos):
                        continue
                    if c.is_bound(Dpos):
                        continue
                    Fpos = c.partner(Cpos)
                    if Fpos is None:
                        continue
                    # D must be able to take C's place: D complementary to C
                    if not _is_complementary(c.token(Dpos), c.token(Cpos)):
                        continue
                    move = frozenset({Dpos, Cpos, Fpos})
                    if move in seen_moves:
                        continue
                    seen_moves.add(move)

                    new_bonds = set(c.bonds)
                    new_bonds.discard(frozenset({Cpos, Fpos}))
                    new_bonds.add(frozenset({Dpos, Cpos}))
                    products = self._split(Complex(c.strands, frozenset(new_bonds)))
                    if products is None:
                        continue
                    products = tuple(self._rename(p) for p in products)
                    nt = self._domain_len(c.token(Dpos))
                    dg = self._reaction_dg([c], products)
                    kf = displacement_kf(nt, self.material, self.celsius)
                    kr = self._reverse_rate(kf, dg)
                    yield DomainReaction((c,), products, kf, kr, "migrate")

    def _bimolecular(self, ca: Complex, cb: Complex):
        """Toehold (or, with include_leak, blunt-end) binding between two complexes."""
        for pa in ca.positions():
            if ca.is_bound(pa):
                continue
            ta = ca.token(pa)
            la = self._domain_len(ta)
            for pb in cb.positions():
                if cb.is_bound(pb):
                    continue
                if not _is_complementary(ta, cb.token(pb)):
                    continue
                lb = self._domain_len(cb.token(pb))  # == la (complementary)
                is_toehold = la <= self.toehold_max_len
                if not is_toehold and not self.include_leak:
                    continue
                merged = self._merge(ca, cb, pa, pb)
                if merged is None:
                    continue
                merged = self._rename(merged)
                dg = self._reaction_dg([ca, cb], [merged])
                if is_toehold:
                    kf = toehold_kf(la, self.material, self.celsius)
                else:
                    # blunt-end leak nucleation: Boltzmann-suppressed
                    kf = toehold_kf(0, self.material, self.celsius)
                kr = self._reverse_rate(kf, dg)
                yield DomainReaction((ca, cb), (merged,), kf, kr, "bind")

    # ── structure helpers ────────────────────────────────────────────────────
    @staticmethod
    def _valid_pos(c: Complex, pos: Position) -> bool:
        si, di = pos
        return 0 <= si < len(c.strands) and 0 <= di < len(c.strands[si][1])

    def _helices(self, c: Complex) -> list[list[tuple[Position, Position]]]:
        """Group bonds into maximal stacked (coaxial, antiparallel) helices."""
        bonds = [tuple(sorted(b)) for b in c.bonds]
        bondset = {frozenset(b) for b in c.bonds}
        helices: list[list] = []
        assigned: set = set()
        for b in bonds:
            if frozenset(b) in assigned:
                continue
            (p, q) = b
            helix = [(p, q)]
            assigned.add(frozenset(b))
            # extend in both stacking directions
            for sign in (+1, -1):
                pp, qq = p, q
                while True:
                    np_ = (pp[0], pp[1] + sign)
                    nq_ = (qq[0], qq[1] - sign)
                    nb = frozenset({np_, nq_})
                    if nb in bondset and nb not in assigned:
                        helix.append((np_, nq_))
                        assigned.add(nb)
                        pp, qq = np_, nq_
                    else:
                        break
            helices.append(helix)
        return helices

    def _split(self, c: Complex) -> list[Complex] | None:
        """
        Partition a (possibly disconnected) complex into its connected components.
        Returns the list of component complexes, or None if it would create a
        complex larger than ``max_strands``.
        """
        n = len(c.strands)
        # union-find over strands connected by a bond
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for bond in c.bonds:
            (sa, _), (sb, _) = tuple(bond)
            union(sa, sb)

        groups: dict[int, list[int]] = {}
        for si in range(n):
            groups.setdefault(find(si), []).append(si)

        if len(groups) == 1:
            if n > self.max_strands:
                return None
            return [c]

        comps: list[Complex] = []
        for members in groups.values():
            if len(members) > self.max_strands:
                return None
            remap = {old: new for new, old in enumerate(members)}
            strands = tuple(c.strands[old] for old in members)
            bonds = frozenset(
                frozenset({(remap[p[0]], p[1]), (remap[q[0]], q[1])})
                for bond in c.bonds
                for p, q in [tuple(bond)]
                if p[0] in remap and q[0] in remap
            )
            comps.append(Complex(strands, bonds))
        return comps

    def _merge(self, ca: Complex, cb: Complex, pa: Position, pb: Position) -> Complex | None:
        """Join two complexes with a new bond pa(in ca)–pb(in cb)."""
        if ca.n_strands + cb.n_strands > self.max_strands:
            return None
        shift = ca.n_strands
        strands = ca.strands + cb.strands
        bonds = set(ca.bonds)
        for bond in cb.bonds:
            (p, q) = tuple(bond)
            bonds.add(frozenset({(p[0] + shift, p[1]), (q[0] + shift, q[1])}))
        bonds.add(frozenset({pa, (pb[0] + shift, pb[1])}))
        merged = Complex(strands, frozenset(bonds))
        if self._is_pseudoknotted(merged):
            return None
        return merged

    def _is_pseudoknotted(self, c: Complex) -> bool:
        """
        Reject bond sets that cannot be drawn without crossings in *any* strand
        ordering — i.e. genuinely pseudoknotted complexes (v1 does not model them).
        """
        n = len(c.strands)
        for perm in permutations(range(n)):
            offset, off = {}, 0
            for si in perm:
                offset[si] = off
                off += len(c.strands[si][1])
            arcs = []
            for bond in c.bonds:
                (sa, da), (sb, db) = tuple(bond)
                a = offset[sa] + da
                b = offset[sb] + db
                arcs.append((min(a, b), max(a, b)))
            if not _has_crossing(arcs):
                return False
        return True

    # ── energy / rates ───────────────────────────────────────────────────────
    def _complex_energy(self, c: Complex) -> float:
        key = hash(c)
        if key in self._energy_cache:
            return self._energy_cache[key]
        dg = 0.0
        for helix in self._helices(c):
            seq = "".join(self._domain_seq(c.token(p)) for p, _ in helix)
            dg += self.engine.duplex_dg(seq, reverse_complement(seq))
        self._energy_cache[key] = dg
        return dg

    def _reaction_dg(self, reactants: list[Complex], products: list[Complex]) -> float:
        gr = sum(self._complex_energy(c) for c in reactants)
        gp = sum(self._complex_energy(c) for c in products)
        return gp - gr

    def _reverse_rate(self, kf: float, dg_forward: float) -> float:
        """kr = kf · exp(ΔG_forward / RT) so that Keq = kf/kr = exp(-ΔG/RT)."""
        kr = kf * math.exp(dg_forward / self._RT)
        return min(max(kr, 1e-30), 1e30)

    def _open_kf(self, helix_len: int) -> float:
        """First-order dissociation prefactor for a short helix (s^-1)."""
        # Use the bimolecular nucleation rate as the attempt frequency; detailed
        # balance against helix ΔG then sets the actual dissociation rate.  Here we
        # parameterise the *forward* (open) rate directly via the helix stability.
        return toehold_kf(helix_len, self.material, self.celsius)

    # ── bookkeeping ──────────────────────────────────────────────────────────
    def _register(self, c: Complex, seen: dict, order: list) -> tuple[Complex, bool]:
        if c in seen:
            return seen[c], False
        seen[c] = c
        order.append(c)
        return c, True

    def _add_reaction(self, rxn: DomainReaction, reactions: dict, seen, order, worklist):
        # canonicalise reactant/product multisets for dedup
        rk = tuple(_multiset(rxn.reactants))
        pk = tuple(_multiset(rxn.products))
        if rk == pk:
            return
        key = frozenset({rk, pk})
        if key in reactions:
            return
        # register any new complexes (and enqueue them for processing)
        new_reactants = []
        new_products = []
        for c in rxn.reactants:
            cc, is_new = self._register(c, seen, order)
            new_reactants.append(cc)
            if is_new:
                worklist.append(cc)
        for c in rxn.products:
            cc, is_new = self._register(c, seen, order)
            new_products.append(cc)
            if is_new:
                worklist.append(cc)
        reactions[key] = DomainReaction(
            tuple(new_reactants), tuple(new_products), rxn.kf, rxn.kr, rxn.mechanism
        )

    # ── naming ───────────────────────────────────────────────────────────────
    def _rename(self, c: Complex) -> Complex:
        """Assign a stable, readable name based on the canonical signature."""
        if c.n_strands == 1 and not c.bonds:
            return Complex(c.strands, c.bonds, name=c.strands[0][0])
        names = "_".join(sorted(c.strand_names))
        # disambiguate different structures of the same strand multiset; the
        # suffix is derived from the canonical signature so it is deterministic.
        suffix = format(hash(c) & 0xFFF, "03x")
        return Complex(c.strands, c.bonds, name=f"{names}_{suffix}")


# ── module-level helpers ──────────────────────────────────────────────────────
def _multiset(complexes) -> list:
    return sorted((c._canonical_key() for c in complexes), key=repr)


def _has_crossing(arcs: list[tuple[int, int]]) -> bool:
    """True if any two arcs (i,j),(k,l) cross: i<k<j<l."""
    for (a, b), (c, d) in combinations(arcs, 2):
        if a < c < b < d or c < a < d < b:
            return True
    return False
