"""
[EXPERIMENTAL API — outside the semver contract until stabilized; see README
"API stability and versioning".]

G-quadruplex (G4) folding and aptamer thermodynamics — Frontier §3.

NUPACK (and every secondary-structure partition-function engine) hardcodes
pseudoknots **off** and represents only Watson–Crick / wobble base pairs.  A
G-quadruplex is neither: four guanine tracts associate into stacked G-tetrads
held together by Hoogsteen hydrogen bonds and a central column of monovalent
cations (K⁺ ≫ Na⁺).  It therefore lives entirely outside the model NUPACK can
express.  This module adds it as a *competing macrostate* layered on top of the
existing McCaskill ensemble, so that for a sequence that can fold either into a
duplex/hairpin **or** a G4, the equilibrium probability of each emerges — the
core quantity for a K⁺-gated aptamer biosensor.

Two pieces:

1. **Motif recognition** (:func:`find_g4_motifs`): the standard putative
   quadruplex sequence (PQS) pattern G≥m·L·G≥m·L·G≥m·L·G≥m, m tetrads, with
   loop-length limits.  Pure sequence pattern — no thermodynamics.

2. **Two-state thermodynamics** (:func:`g4_thermodynamics`): an empirical
   ΔH/ΔS model with a tract-association nucleation term, a per-tetrad-stack
   term (so more tetrads ⇒ more stable), a loop-length entropic penalty
   (Guédin, Gros & Mergny 2010, NAR 38:7858 — longer loops destabilize), and a
   monovalent-cation stabilization that distinguishes K⁺ from Na⁺ (Bhattacharyya
   et al. 2016 review; K⁺ stabilizes G4 far more than Na⁺).  Reference
   parameters were fit to canonical melting data (c-myc Pu22, human-telomere
   22AG, thrombin-binding aptamer) at 100 mM K⁺; see ``scratch/fit_g4_params.py``.
   The absolute numbers are empirical (G4 thermodynamics are genuinely
   construct-dependent and polymorphic); the model is built so the *trends* are
   correct: 3-tetrad > 2-tetrad, short loops > long loops, K⁺ > Na⁺, [cation]↑ ⇒
   stabilize.

3. **Partition competition** (:func:`quadruplex_ensemble`): combines the
   secondary-structure partition function with the G4 macrostate(s).  The G4 is
   added by re-using the rigorous constrained ``dangle_free_partition(blocked=…)``
   machinery — the G-tract positions are forced unpaired (engaged in tetrads),
   the rest of the strand folds around them, and the whole G4 carries a Boltzmann
   weight exp(−ΔG_G4/RT).  No edit to the core DP.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

R = 1.987e-3      # kcal / (mol·K)
TREF = 310.15     # 37 °C, the reference temperature for the ΔG fit

# ── reference (100 mM K⁺) two-state parameters, fit to canonical melts ─────────
#    dH = H_NUC + n_stack·H_STACK                 (kcal/mol)
#    dS = S_NUC + n_stack·S_STACK + loopnt·S_LOOP (kcal/mol/K)
#    n_stack = n_tetrads − 1 inter-tetrad stacking steps
# See scratch/fit_g4_params.py (anchors: c-myc Pu22, telomere 22AG, TBA).
H_NUC = -25.0       # tract-association nucleation enthalpy
S_NUC = -0.065055   # nucleation entropy cost
H_STACK = -8.7013   # per inter-tetrad stack (negative ⇒ more tetrads more stable)
S_STACK = -0.021687
S_LOOP = -0.002237  # per loop nucleotide (negative ⇒ longer loops destabilize)

# ── monovalent-cation stabilization (relative to the 100 mM K⁺ reference) ──────
KD_CATION = 0.02       # M, effective per-site half-saturation
A_REF = 0.10           # M, K⁺ activity at which the reference params were fit
G_ION_SITE = 2.5       # kcal/mol per cation site — sized so that with no
                       # stabilizing cation the G4 is essentially unfolded
                       # (telomere folded fraction → ~0.06), and so K⁺ ≫ Na⁺.
REL_AFFINITY = {"K": 1.0, "Na": 0.15, "Li": 0.0, "none": 0.0}


@dataclass
class G4Motif:
    """A putative intramolecular G-quadruplex on a single strand."""
    tracts: list[tuple[int, int]]   # 4 (start, end) inclusive G-run indices
    n_tetrads: int                  # tetrad layers = min G-run length (capped)
    loops: list[int]                # the 3 loop lengths between consecutive tracts
    span: tuple[int, int]           # (start, end) inclusive of the whole motif

    @property
    def loop_nt(self) -> int:
        return sum(self.loops)

    def engaged_positions(self) -> set[int]:
        """Indices forced into tetrads (the first ``n_tetrads`` G's of each tract)."""
        pos: set[int] = set()
        for s, e in self.tracts:
            for k in range(self.n_tetrads):
                pos.add(s + k)
        return pos


@dataclass
class G4Fold:
    """Best-G4 folding result for a sequence under given conditions."""
    motif: G4Motif | None
    dH: float            # kcal/mol (folding, U→F)
    dS: float            # kcal/mol/K
    dG: float            # kcal/mol at the requested temperature
    tm_celsius: float    # two-state melting temperature
    folded_fraction: float
    structure: str       # extended dot-bracket: '+' marks tetrad G's


# ── 1. motif recognition ──────────────────────────────────────────────────────

def find_g4_motifs(
    sequence: str,
    min_tetrads: int = 2,
    max_tetrads: int = 4,
    min_loop: int = 1,
    max_loop: int = 7,
) -> list[G4Motif]:
    """Find putative intramolecular G4 motifs (PQS pattern).

    Pattern: ``G{m,} L{min_loop,max_loop} G{m,} L G{m,} L G{m,}`` for the largest
    ``m`` in ``[min_tetrads, max_tetrads]`` that matches at each start — guanines
    may be RNA (treated as G).  Returns motifs sorted by descending tetrad count
    then ascending loop length (most stable first).
    """
    seq = sequence.upper().replace("U", "T")
    motifs: list[G4Motif] = []
    seen: set[tuple] = set()

    for m in range(max_tetrads, min_tetrads - 1, -1):
        g = "G" * m
        loop = f"[ACGT]{{{min_loop},{max_loop}}}"
        # capture the four G-tracts and the spans between them
        pat = re.compile(f"(G{{{m},}})({loop})(G{{{m},}})({loop})(G{{{m},}})({loop})(G{{{m},}})")
        for mo in _overlapping_finditer(pat, seq):
            tracts_raw = [mo.span(i) for i in (1, 3, 5, 7)]
            # clamp each tract to exactly m tetrad rows (anchored at its 5' end)
            tracts = [(s, s + m - 1) for (s, e) in tracts_raw]
            loops = [mo.span(i)[1] - mo.span(i)[0] for i in (2, 4, 6)]
            span = (tracts[0][0], tracts[3][0] + m - 1)
            key = (span, m)
            if key in seen:
                continue
            seen.add(key)
            motifs.append(G4Motif(tracts=tracts, n_tetrads=m, loops=loops, span=span))

    motifs.sort(key=lambda x: (-x.n_tetrads, x.loop_nt, x.span[0]))
    return motifs


def _overlapping_finditer(pat: re.Pattern, seq: str):
    """Yield matches allowing overlaps (advance one position past each start)."""
    pos = 0
    while pos < len(seq):
        mo = pat.search(seq, pos)
        if mo is None:
            return
        yield mo
        pos = mo.start() + 1


# ── 2. two-state thermodynamics ────────────────────────────────────────────────

def _cation_activity(potassium: float, sodium: float, lithium: float = 0.0) -> float:
    """Effective G4-stabilizing monovalent-cation activity (M)."""
    return (REL_AFFINITY["K"] * max(potassium, 0.0)
            + REL_AFFINITY["Na"] * max(sodium, 0.0)
            + REL_AFFINITY["Li"] * max(lithium, 0.0))


def _ion_entropy_shift(n_cation: int, potassium: float, sodium: float) -> float:
    """ΔS shift (kcal/mol/K) from monovalent cations, vs the 100 mM K⁺ reference.

    Modelled as a Langmuir site-occupancy stabilization treated as purely
    entropic, so the reference (a = A_REF) gives zero shift, higher cation
    activity stabilizes (raises Tm), and K⁺ ≫ Na⁺.
    """
    a = _cation_activity(potassium, sodium)
    theta = a / (KD_CATION + a) if a > 0 else 0.0
    theta_ref = A_REF / (KD_CATION + A_REF)
    ddG = -G_ION_SITE * n_cation * (theta - theta_ref)   # >0 ⇒ destabilizing
    return -ddG / TREF                                   # entropic: dG(Tref)=ddG


def g4_thermodynamics(
    motif: G4Motif,
    celsius: float = 37.0,
    potassium: float = 0.1,
    sodium: float = 0.0,
) -> tuple[float, float, float]:
    """Two-state (ΔH, ΔS, ΔG(T)) of folding into ``motif``.

    Returns ``(dH, dS, dG)`` in kcal/mol (ΔS in kcal/mol/K).  ΔG < 0 favours the
    folded G4.
    """
    n_stack = max(motif.n_tetrads - 1, 0)
    n_cation = max(motif.n_tetrads - 1, 0)   # cations stack between tetrad layers
    dH = H_NUC + n_stack * H_STACK
    dS = (S_NUC + n_stack * S_STACK + motif.loop_nt * S_LOOP
          + _ion_entropy_shift(n_cation, potassium, sodium))
    T = celsius + 273.15
    dG = dH - T * dS
    return dH, dS, dG


def _folded_fraction(dG: float, celsius: float) -> float:
    """Unimolecular two-state folded fraction = 1/(1+exp(ΔG/RT))."""
    T = celsius + 273.15
    x = dG / (R * T)
    if x > 700:        # avoid overflow; fully unfolded
        return 0.0
    return 1.0 / (1.0 + math.exp(x))


def fold_quadruplex(
    sequence: str,
    celsius: float = 37.0,
    potassium: float = 0.1,
    sodium: float = 0.0,
    min_tetrads: int = 2,
    max_loop: int = 7,
) -> G4Fold:
    """Best (lowest-ΔG) G4 fold for ``sequence`` under the given cation conditions.

    If no PQS motif is present, returns a :class:`G4Fold` with ``motif=None`` and
    zero folded fraction.
    """
    motifs = find_g4_motifs(sequence, min_tetrads=min_tetrads, max_loop=max_loop)
    n = len(sequence)
    if not motifs:
        return G4Fold(None, 0.0, 0.0, 0.0, float("nan"), 0.0, "." * n)

    best = None
    best_dG = float("inf")
    for mo in motifs:
        dH, dS, dG = g4_thermodynamics(mo, celsius, potassium, sodium)
        if dG < best_dG:
            best_dG = dG
            best = (mo, dH, dS, dG)

    mo, dH, dS, dG = best
    tm = (dH / dS - 273.15) if dS != 0 else float("nan")
    frac = _folded_fraction(dG, celsius)
    struct = list("." * n)
    for p in mo.engaged_positions():
        struct[p] = "+"
    return G4Fold(mo, dH, dS, dG, tm, frac, "".join(struct))


# ── 3. partition competition (duplex/hairpin vs G4) ────────────────────────────

@dataclass
class QuadruplexEnsemble:
    """Equilibrium partition between secondary structure and G4 macrostates."""
    z_secondary: float                       # partition over all WC/wobble structures
    z_total: float                           # + G4 macrostates
    p_g4: float                              # total probability in any G4
    p_g4_by_motif: list[tuple[G4Motif, float]] = field(default_factory=list)
    best: G4Fold | None = None

    @property
    def p_secondary(self) -> float:
        return self.z_secondary / self.z_total if self.z_total > 0 else 0.0


def quadruplex_ensemble(
    sequence: str,
    celsius: float = 37.0,
    material: str = "dna",
    sodium: float = 0.0,
    magnesium: float = 0.0,
    potassium: float = 0.1,
    min_tetrads: int = 2,
    max_loop: int = 7,
) -> QuadruplexEnsemble:
    """Equilibrium competition between the WC/wobble ensemble and G4 folding.

    ::

        Z_total = Z_secondary + Σ_g exp(−ΔG_g/RT)·Z_flank(g)

    where ``Z_secondary`` is the ordinary McCaskill partition function and, for
    each G4 motif *g*, ``Z_flank(g)`` is the partition function of the strand
    with *g*'s tetrad guanines forced unpaired (they are engaged in tetrads) —
    computed with the same DP via the ``blocked`` constraint, so duplex-vs-G4
    competition is captured exactly within the model.  Both Z's share the
    fully-unfolded strand as their common reference, so the weights are
    commensurate.  Monovalent cations (K⁺ + Na⁺) act as the duplex salt for the
    flank partition.

    Returns a :class:`QuadruplexEnsemble` with per-motif occupancies.
    """
    from strider.thermo.ensemble import dangle_free_partition

    # K⁺ and Na⁺ both screen the duplex backbone; combine as effective monovalent.
    na_eff = max(sodium, 0.0) + max(potassium, 0.0)
    na_eff = na_eff if na_eff > 0 else 1.0   # avoid log(0) in salt model

    z_secondary = dangle_free_partition(
        sequence, celsius, material, sodium_M=na_eff, magnesium_M=magnesium
    )

    T = celsius + 273.15
    motifs = find_g4_motifs(sequence, min_tetrads=min_tetrads, max_loop=max_loop)
    z_total = z_secondary
    by_motif: list[tuple[G4Motif, float]] = []
    best_fold = fold_quadruplex(sequence, celsius, potassium, sodium,
                                min_tetrads=min_tetrads, max_loop=max_loop)
    for mo in motifs:
        _, _, dG = g4_thermodynamics(mo, celsius, potassium, sodium)
        z_flank = dangle_free_partition(
            sequence, celsius, material, sodium_M=na_eff, magnesium_M=magnesium,
            blocked=mo.engaged_positions(),
        )
        weight = math.exp(-dG / (R * T)) * z_flank
        z_total += weight
        by_motif.append((mo, weight))

    if z_total <= 0:
        return QuadruplexEnsemble(z_secondary, z_secondary, 0.0, [], best_fold)

    p_by_motif = [(mo, w / z_total) for mo, w in by_motif]
    p_g4 = sum(p for _, p in p_by_motif)
    return QuadruplexEnsemble(
        z_secondary=z_secondary,
        z_total=z_total,
        p_g4=p_g4,
        p_g4_by_motif=p_by_motif,
        best=best_fold,
    )
