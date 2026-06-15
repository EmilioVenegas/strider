"""Two-state thermodynamics for a bimolecular DNA duplex.

A duplex melts in a two-state reaction

    A + B ⇌ AB        (hetero)
    2 A ⇌ A2          (self)

so its melting temperature depends on strand concentration.  This module walks
a given inter-strand helix with the same per-element engine used for hairpins
(``_stack_energy``, ``_interior_bulge_energy``) but replaces the hairpin-loop
term with ``JOIN_PENALTY`` plus terminal-pair / dangling-end contributions at
both helix ends.  ΔH is obtained by running the identical walk against the ΔH
tables, so

    ΔS = (ΔH − ΔG₃₇) / T_ref

is exact at the table reference temperature 310.15 K (37 °C).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

T_REF = 310.15  # K, reference temperature of the ΔG tables
R = 1.987e-3   # kcal / (mol · K)


@dataclass(frozen=True)
class DimerThermo:
    """Two-state dimer thermodynamics at the requested salt concentration."""
    tm_celsius: float
    dH: float          # kcal/mol
    dS: float          # cal/mol/K
    dG37: float        # kcal/mol, salt-corrected closed-state free energy
    n_pairs: int
    structure: str
    salt_model: str = "per_bp"   # "tan_chen" / "per_bp"
    is_self_dimer: bool = False


def _dimer_mfe(
    seq1: str,
    seq2: str | None = None,
    *,
    engine=None,
    material: str = "dna",
    celsius: float = 37.0,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
):
    """
    Minimum free energy inter-strand duplex for two strands.

    The dynamic program enumerates all antiparallel alignments, including
    blunt-end stacks and 3'/5' staggered overlaps, and allows single-base
    bulges on either strand.  Energy terms are taken from the same nearest-
    neighbour helpers used by the hairpin / ensemble DP plus sign-gated dangle
    contributions, matching the bookkeeping in
    :func:`strider.thermo.structure_thermo._sum_dimer_elements`.

    The bimolecular ``JOIN_PENALTY`` is intentionally omitted from the returned
    energy; it belongs in the concentration-dependent Tm calculation, not in
    the closed-state duplex free energy.

    The search uses the 37 °C ΔG tables (``T_REF = 310.15 K``).  Inter-strand
    helices contain no T-dependent hairpin-loop term, so the ``celsius``
    argument is accepted for API consistency but does not change the predicted
    structure.

    Returns an :class:`strider.thermo.engine.MFEResult` whose ``structure`` is
    a dot-bracket string on ``seq1 + seq2`` (no ``&`` separator) and whose
    ``base_pairs`` are sorted, 0-based, and satisfy ``i < len(seq1) <= j``.
    """
    from strider.thermo._param_context import lookup_table, param_context
    from strider.thermo.ensemble import (
        _interior_bulge_energy,
        _stack_energy,
        _terminal_pair_penalty,
        _wc_pairs,
    )
    from strider.thermo.engine import MFEResult

    if engine is not None:
        material = engine.material
        celsius = engine.celsius
        sodium_M = engine.sodium
        magnesium_M = engine.magnesium

    material = material.lower()
    seq1 = seq1.upper().replace("U", "T")
    if seq2 is None:
        seq2 = seq1
    else:
        seq2 = seq2.upper().replace("U", "T")

    n1 = len(seq1)
    n2 = len(seq2)
    seq = seq1 + seq2
    n = n1 + n2

    if n1 == 0 or n2 == 0:
        return MFEResult(energy=0.0, structure="." * n, base_pairs=[], sequence=seq)

    pairs_set = _wc_pairs(material)
    can_pair = lambda i, j_loc: frozenset([seq1[i], seq2[j_loc]]) in pairs_set

    if material == "dna":
        from strider.thermo.parameters_dna import DANGLE_3, DANGLE_5
    else:
        from strider.thermo.parameters_rna import DANGLE_3, DANGLE_5

    dangle_5 = lookup_table("dangle_5", DANGLE_5)
    dangle_3 = lookup_table("dangle_3", DANGLE_3)

    INF = float("inf")
    inner = np.full((n1, n2), INF)
    trace: list[list[tuple[int, int] | None]] = [[None] * n2 for _ in range(n1)]

    def _inner_dangles(i: int, j_loc: int) -> float:
        """Dangles adjacent to the inner terminus of pair (i, n1+j_loc)."""
        j_concat = n1 + j_loc
        total = 0.0
        if j_concat - 1 >= n1:
            d5 = dangle_5.get(seq[j_concat] + seq[i] + seq[j_concat - 1])
            if d5 is not None and d5 < 0:
                total += d5
        if i - 1 >= 0 and i + 1 < n1:
            d3 = dangle_3.get(seq[i - 1] + seq[i] + seq[i + 1])
            if d3 is not None and d3 < 0:
                total += d3
        return total

    def _outer_dangles(i: int, j_loc: int) -> float:
        """Dangles adjacent to the outer terminus of pair (i, n1+j_loc)."""
        j_concat = n1 + j_loc
        total = 0.0
        if i - 1 >= 0:
            d5 = dangle_5.get(seq[i] + seq[j_concat] + seq[i - 1])
            if d5 is not None and d5 < 0:
                total += d5
        if j_concat + 1 < n:
            d3 = dangle_3.get(seq[j_concat - 1] + seq[j_concat] + seq[j_concat + 1])
            if d3 is not None and d3 < 0:
                total += d3
        return total

    def _run_dp() -> tuple[float, list[tuple[int, int]]]:
        # Fill inner[i][j_loc] from inside out (i decreasing, j_loc increasing).
        for i in range(n1 - 1, -1, -1):
            for j_loc in range(n2):
                if not can_pair(i, j_loc):
                    continue
                j_concat = n1 + j_loc

                stop_val = (
                    _terminal_pair_penalty(seq, i, j_concat, material)
                    + _inner_dangles(i, j_loc)
                )

                best_continue = INF
                best_next: tuple[int, int] | None = None
                transitions = []
                if i + 1 < n1 and j_loc - 1 >= 0 and can_pair(i + 1, j_loc - 1):
                    transitions.append((i + 1, j_loc - 1, 0, 0))
                if i + 2 < n1 and j_loc - 1 >= 0 and can_pair(i + 2, j_loc - 1):
                    transitions.append((i + 2, j_loc - 1, 1, 0))
                if i + 1 < n1 and j_loc - 2 >= 0 and can_pair(i + 1, j_loc - 2):
                    transitions.append((i + 1, j_loc - 2, 0, 1))

                for ip, jp_loc, nl, nr in transitions:
                    jp_concat = n1 + jp_loc
                    if inner[ip][jp_loc] == INF:
                        continue
                    if nl == 0 and nr == 0:
                        e = _stack_energy(seq, i, j_concat, material)
                    else:
                        e = _interior_bulge_energy(
                            seq, i, j_concat, ip, jp_concat, nl, nr, material
                        )
                    cand = e + inner[ip][jp_loc]
                    if cand < best_continue:
                        best_continue = cand
                        best_next = (ip, jp_loc)

                if best_next is None or stop_val <= best_continue:
                    inner[i][j_loc] = stop_val
                    trace[i][j_loc] = None
                else:
                    inner[i][j_loc] = best_continue
                    trace[i][j_loc] = best_next

        best_energy = INF
        best_outer: tuple[int, int] | None = None
        for i in range(n1):
            for j_loc in range(n2):
                if not can_pair(i, j_loc):
                    continue
                j_concat = n1 + j_loc
                outer_val = (
                    _terminal_pair_penalty(seq, i, j_concat, material)
                    + _outer_dangles(i, j_loc)
                    + inner[i][j_loc]
                )
                total = outer_val
                if total < best_energy:
                    best_energy = total
                    best_outer = (i, j_loc)

        if best_outer is None:
            return 0.0, []

        pairs: list[tuple[int, int]] = []
        i, j_loc = best_outer
        while True:
            pairs.append((i, n1 + j_loc))
            nxt = trace[i][j_loc]
            if nxt is None:
                break
            i, j_loc = nxt
        return best_energy, pairs

    override = None
    if engine is not None and getattr(engine, "_uses_custom_params", lambda: False)():
        override = engine.params

    with param_context(override):
        energy, pairs = _run_dp()

    if not pairs:
        return MFEResult(energy=0.0, structure="." * n, base_pairs=[], sequence=seq)

    pairs.sort()
    structure = _dotbracket(seq, pairs)
    return MFEResult(
        energy=energy,
        structure=structure,
        base_pairs=pairs,
        sequence=seq,
    )


def dimer_thermo(
    seq1: str,
    seq2: str | None = None,
    *,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    material: str = "dna",
    structure: str | list[tuple[int, int]] | None = None,
    strand_conc_M: float = 250e-9,
    salt_model: str = "auto",
) -> DimerThermo:
    """
    Two-state thermodynamics (Tm, ΔH, ΔS, ΔG₃₇) for a bimolecular duplex.

    Parameters
    ----------
    seq1, seq2 : strand sequences.  If ``seq2`` is omitted, ``seq1`` is folded
        against itself as a self-dimer.
    sodium_M, magnesium_M : ion concentrations (1 M Na⁺ / 0 Mg²⁺ = reference).
    material : ``"dna"`` or ``"rna"``.
    structure : optional structure to score; either a dot-bracket string on the
        concatenated sequence ``seq1 + seq2`` or a list of ``(i, j)`` pairs.
        If omitted, the MFE inter-strand structure is predicted.
    strand_conc_M : total strand concentration in molar.
    salt_model : closed-state salt correction; see :func:`strider.thermo.hairpin.hairpin_thermo`.

    Raises
    ------
    ValueError : if the structure is not a single nested helix crossing the
        strand junction, or has fewer than two base pairs.
    """
    from strider.thermo.engine import ThermoEngine
    from strider.thermo.salt import dg_per_bp_salt, tan_chen_helix_dg, TAN_CHEN_MIN_BP
    from strider.thermo.structure_thermo import (
        parse_dimer_pairs,
        structure_enthalpy_dimer,
        structure_free_energy_dimer,
    )

    is_self_dimer = seq2 is None or seq1.upper().replace("U", "T") == seq2.upper().replace("U", "T")

    seq1 = seq1.upper().replace("U", "T")
    if seq2 is None:
        seq2 = seq1
    else:
        seq2 = seq2.upper().replace("U", "T")

    n1 = len(seq1)
    seq = seq1 + seq2
    pairs = None

    if structure is None:
        engine = ThermoEngine(material=material, celsius=25.0, sodium=sodium_M, magnesium=magnesium_M)
        mfe = _dimer_mfe(seq1, seq2, engine=engine)
        struct = mfe.structure
    elif isinstance(structure, str):
        struct = structure
        if len(struct) != len(seq):
            raise ValueError("structure length does not match concatenated sequence length")
    else:
        pairs = parse_dimer_pairs(structure, n1)
        struct = _dotbracket(seq, pairs)

    if pairs is None:
        pairs = parse_dimer_pairs(struct, n1)
    n = len(pairs)

    dG37_1M = structure_free_energy_dimer(seq, n1, struct, material)
    dH = structure_enthalpy_dimer(seq, n1, struct, material)

    use_tc = salt_model == "tan_chen" or (
        salt_model == "auto" and material.lower() in ("dna", "rna") and n >= TAN_CHEN_MIN_BP
    )
    if use_tc:
        salt_dg = tan_chen_helix_dg(n, sodium_M, magnesium_M, material)
        applied = "tan_chen"
    else:
        salt_dg = n * dg_per_bp_salt(sodium_M, magnesium_M)
        applied = "per_bp"

    dG37 = dG37_1M + salt_dg
    dS_kcal = (dH - dG37) / T_REF
    if dS_kcal == 0:
        raise ValueError("degenerate entropy — cannot define a melting point")

    # Concentration term: homodimer (seq1 == seq2) uses ln(CT); heterodimer
    # uses ln(CT/4) because the two strands are distinguishable.
    ln_term = math.log(strand_conc_M) if is_self_dimer else math.log(strand_conc_M / 4.0)
    tm_K = dH / (dS_kcal + R * ln_term)
    return DimerThermo(
        tm_celsius=tm_K - 273.15,
        dH=dH,
        dS=dS_kcal * 1000.0,
        dG37=dG37,
        n_pairs=n,
        structure=struct,
        salt_model=applied,
        is_self_dimer=is_self_dimer,
    )


def dimer_tm(
    seq1: str,
    seq2: str | None = None,
    *,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    material: str = "dna",
    strand_conc_M: float = 250e-9,
    salt_model: str = "auto",
) -> float:
    """Melting temperature (°C) of the predicted duplex. See :func:`dimer_thermo`."""
    return dimer_thermo(
        seq1, seq2,
        sodium_M=sodium_M,
        magnesium_M=magnesium_M,
        material=material,
        strand_conc_M=strand_conc_M,
        salt_model=salt_model,
    ).tm_celsius


def _dotbracket(seq: str, pairs: list[tuple[int, int]]) -> str:
    s = ["."] * len(seq)
    for i, j in pairs:
        s[i], s[j] = "(", ")"
    return "".join(s)
