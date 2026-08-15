"""Two-state thermodynamics for a bimolecular DNA duplex.

A duplex melts in a two-state reaction

    A + B ⇌ AB        (hetero)
    2 A ⇌ A2          (self)

so its melting temperature depends on strand concentration.  This module walks
a given inter-strand helix with the same per-element engine used for hairpins
(``_stack_energy``, ``_interior_bulge_energy``) but replaces the hairpin-loop
term with terminal-pair / dangling-end contributions at both helix ends, and
adds the bimolecular duplex *initiation* (nucleation) term once per duplex.  ΔH
is obtained by running the identical walk against the ΔH tables, so

    ΔS = (ΔH − ΔG₃₇) / T_ref

is exact at the table reference temperature 310.15 K (37 °C).  The two-state Tm then follows from ΔH, ΔS and the concentration term R·ln(C_T) see :data:`DUPLEX_INIT_DG37` for why initiation and the concentration term are both needed and are not interchangeable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

T_REF = 310.15  # K, reference temperature of the ΔG tables
R = 1.987e-3   # kcal / (mol · K)

# SantaLucia & Hicks (2004) unified bimolecular duplex *initiation* (nucleation).
# Applied once per duplex, independent of length/sequence and of the terminal-AT
# penalty (which the per-element walk already adds at both helix ends).  This is
# the duplex-formation nucleation free energy and is PHYSICALLY DISTINCT from the
# concentration term R·ln(C_T): the latter is translational entropy of bringing
# two strands together, the former is the enthalpic/entropic cost of nucleating
# the first base pair.  Both are required for a correct two-state Tm.
#
#   ΔG°37_init = +1.96 kcal/mol,  ΔH°_init = +0.2 kcal/mol
#   ⇒ ΔS°_init = (ΔH − ΔG37)/T_ref = (0.2 − 1.96)/310.15·1000 ≈ −5.7 cal/mol/K
#
# Omitting it (the state of this branch before the fix) left ΔH correct but ΔS
# ~6 cal/mol/K too small in magnitude, inflating dimer Tm by ~7–11 °C versus
# primer3 and NUPACK equilibrium melts.
DUPLEX_INIT_DG37 = 1.96  # kcal/mol
DUPLEX_INIT_DH = 0.2     # kcal/mol


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


def _dimer_mfe_candidates(
    seq1: str,
    seq2: str | None = None,
    *,
    engine=None,
    material: str = "dna",
    celsius: float = 37.0,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
) -> list[tuple[float, list[tuple[int, int]]]]:
    """
    Rank every antiparallel inter-strand helix start state by closed-state
    free energy.

    The dynamic program is identical to :func:`_dimer_mfe` but, instead of
    returning only the best start, returns all candidate helices sorted by
    total energy ascending and then by base-pair count descending (more pairs
    wins ties).  Distinct start pairs are reported individually; overlapping
    alignments are not collapsed.

    The bimolecular ``JOIN_PENALTY`` is intentionally omitted from the returned
    energy; it belongs in the concentration-dependent Tm calculation, not in
    the closed-state duplex free energy.

    The search uses the 37 °C ΔG tables (``T_REF = 310.15 K``).  Inter-strand
    helices contain no T-dependent hairpin-loop term, so the ``celsius``
    argument is accepted for API consistency but does not change the predicted
    structure.
    """
    from strider.thermo._param_context import lookup_table
    from strider.thermo.ensemble import (
        _interior_bulge_energy,
        _stack_energy,
        _terminal_pair_penalty,
        _wc_pairs,
    )

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
        return []

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

    # Maximum total unpaired bases in an interior loop / bulge considered inside
    # a dimer helix.  Limiting to 4 captures the 1×1, 1×2, 2×1 and 2×2 exact
    # tables and is enough for the small loops primer3 scores in homodimers.
    MAX_LOOP = 4

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
            for nl in range(MAX_LOOP + 1):
                for nr in range(MAX_LOOP + 1 - nl):
                    ip = i + 1 + nl
                    jp_loc = j_loc - 1 - nr
                    if ip < n1 and jp_loc >= 0 and can_pair(ip, jp_loc):
                        transitions.append((ip, jp_loc, nl, nr))

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

    candidates = []
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
            if not math.isfinite(outer_val):
                continue

            pairs: list[tuple[int, int]] = []
            ci, cj_loc = i, j_loc
            while True:
                pairs.append((ci, n1 + cj_loc))
                nxt = trace[ci][cj_loc]
                if nxt is None:
                    break
                ci, cj_loc = nxt

            candidates.append((outer_val, pairs))

    candidates.sort(key=lambda x: (x[0], -len(x[1])))
    return candidates


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

    Returns an :class:`strider.thermo.engine.MFEResult` whose ``structure`` is
    a dot-bracket string on ``seq1 + seq2`` (no ``&`` separator) and whose
    ``base_pairs`` are sorted, 0-based, and satisfy ``i < len(seq1) <= j``.

    This function remains a thin wrapper over :func:`_dimer_mfe_candidates` so
    the DP fill is shared with :func:`dimer_thermo_subopt`.
    """
    from strider.thermo._param_context import param_context
    from strider.thermo.engine import MFEResult

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

    override = None
    if engine is not None and getattr(engine, "_uses_custom_params", lambda: False)():
        override = engine.params

    with param_context(override):
        candidates = _dimer_mfe_candidates(
            seq1,
            seq2,
            engine=engine,
            material=material,
            celsius=celsius,
            sodium_M=sodium_M,
            magnesium_M=magnesium_M,
        )

    if not candidates:
        return MFEResult(energy=0.0, structure="." * n, base_pairs=[], sequence=seq)

    energy, pairs = candidates[0]
    pairs.sort()
    structure = _dotbracket(seq, pairs)
    return MFEResult(
        energy=energy,
        structure=structure,
        base_pairs=pairs,
        sequence=seq,
    )


def _dotbracket(seq: str, pairs: list[tuple[int, int]]) -> str:
    s = ["."] * len(seq)
    for i, j in pairs:
        s[i], s[j] = "(", ")"
    return "".join(s)


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
    paramset=None,
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
    paramset : optional :class:`~strider.thermo.parameters.ParameterSet` (or a
        name resolved via :func:`~strider.thermo.parameters.load_parameters`)
        used both to predict the MFE helix and to score it.  When ``None``
        (default), the native parameter set for ``material`` is used.

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
        engine = ThermoEngine(material=material, celsius=25.0, sodium=sodium_M,
                              magnesium=magnesium_M, parameter_set=paramset)
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

    dG37_1M = structure_free_energy_dimer(seq, n1, struct, material, paramset=paramset)
    dH = structure_enthalpy_dimer(seq, n1, struct, material, paramset=paramset)

    # Bimolecular duplex initiation (nucleation), once per duplex.  Added to both
    # ΔG37 and ΔH so that ΔS = (ΔH − ΔG37)/T_ref picks up ΔS_init ≈ −5.7 cal/mol/K,
    # which is what sets the correct melting temperature.  The initiation term is
    # a constant offset and therefore does not affect MFE structure selection,
    # so the predicting DP (``_dimer_mfe_candidates``) deliberately omits it.
    dG37_1M += DUPLEX_INIT_DG37
    dH += DUPLEX_INIT_DH

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


def dimer_thermo_subopt(
    seq1: str,
    seq2: str | None = None,
    *,
    n: int = 5,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    material: str = "dna",
    strand_conc_M: float = 250e-9,
    salt_model: str = "auto",
    paramset=None,
) -> list[DimerThermo]:
    """
    Return the top ``n`` sub-optimal antiparallel dimer alignments.

    Each alignment is scored with the same two-state model as
    :func:`dimer_thermo`.  Alignments are ranked by the closed-state DP energy
    (``_outer_dangles + _terminal_pair_penalty + inner``) and ties are broken
    by base-pair count (more pairs first).  Distinct start/end pairs are
    reported individually; overlapping alignments are not collapsed.

    Parameters
    ----------
    seq1, seq2 : strand sequences.  If ``seq2`` is omitted, ``seq1`` is folded
        against itself as a self-dimer.
    n : number of distinct alignments to return.
    sodium_M, magnesium_M : ion concentrations (1 M Na⁺ / 0 Mg²⁺ = reference).
    material : ``"dna"`` or ``"rna"``.
    strand_conc_M : total strand concentration in molar.
    salt_model : closed-state salt correction; see :func:`dimer_thermo`.
    paramset : optional :class:`~strider.thermo.parameters.ParameterSet` (or a
        name) used both to predict the alignments and to score them — see
        :func:`dimer_thermo`.
    """
    from strider.thermo._param_context import param_context
    from strider.thermo.engine import ThermoEngine

    seq1_clean = seq1.upper().replace("U", "T")
    seq2_clean = seq2.upper().replace("U", "T") if seq2 is not None else seq1_clean
    seq = seq1_clean + seq2_clean

    engine = ThermoEngine(
        material=material, celsius=25.0, sodium=sodium_M, magnesium=magnesium_M,
        parameter_set=paramset,
    )
    override = engine.params if engine._uses_custom_params() else None

    with param_context(override):
        candidates = _dimer_mfe_candidates(seq1_clean, seq2_clean, engine=engine)

    results: list[DimerThermo] = []
    seen_outer: set[tuple[int, int]] = set()
    for energy, pairs in candidates:
        if len(results) >= n:
            break
        if len(pairs) < 2:
            continue
        outer = pairs[0]
        if outer in seen_outer:
            continue
        seen_outer.add(outer)
        pairs_sorted = sorted(pairs)
        structure = _dotbracket(seq, pairs_sorted)
        dt = dimer_thermo(
            seq1_clean,
            seq2_clean,
            sodium_M=sodium_M,
            magnesium_M=magnesium_M,
            material=material,
            structure=structure,
            strand_conc_M=strand_conc_M,
            salt_model=salt_model,
            paramset=paramset,
        )
        results.append(dt)

    results.sort(key=lambda dt: dt.dG37)
    return results


def dimer_tm(
    seq1: str,
    seq2: str | None = None,
    *,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    material: str = "dna",
    strand_conc_M: float = 250e-9,
    salt_model: str = "auto",
    paramset=None,
) -> float:
    """Melting temperature (°C) of the predicted duplex. See :func:`dimer_thermo`."""
    return dimer_thermo(
        seq1, seq2,
        sodium_M=sodium_M,
        magnesium_M=magnesium_M,
        material=material,
        strand_conc_M=strand_conc_M,
        salt_model=salt_model,
        paramset=paramset,
    ).tm_celsius


# ── optional Rust accelerator ──────────────────────────────────────────────
# ``strider._native.dimer_mfe_candidates`` ports this module's DP fill (DNA)
# to Rust — same recurrence, same tables (generated from
# ``strider.thermo.parameters_dna``, see ``native/codegen_tables.py``), 0/600
# mismatches across the parity fuzz.  The Python implementation above stays
# as the fallback for RNA / param overrides / no extension, and as the test
# oracle (``_dimer_mfe_candidates_py``).
_dimer_mfe_candidates_py = _dimer_mfe_candidates

try:
    from strider import _native as _n

    if hasattr(_n, "dimer_mfe_candidates"):
        def _dimer_mfe_candidates(
            seq1: str,
            seq2: str | None = None,
            *,
            engine=None,
            material: str = "dna",
            celsius: float = 37.0,
            sodium_M: float = 1.0,
            magnesium_M: float = 0.0,
        ) -> list[tuple[float, list[tuple[int, int]]]]:
            """Native DP when DNA and no parameter override are in play; the
            Python _dimer_mfe_candidates handles every other case.

            kwarg semantics mirror the original: an attached ``engine``
            overrides material/celsius/salts (which only select the table
            set — the concrete energies come from the 37 °C tables either
            way), and custom ParameterSets active via ``param_context`` force
            the Python path since the native one uses the baked-in tables.
            """
            from strider.thermo._param_context import _param_override
            if engine is not None:
                material = engine.material
            if (engine is None or not getattr(engine, "_uses_custom_params", lambda: False)()) \
                    and str(material or "dna").lower() == "dna" \
                    and _param_override.get() is None:
                s1 = seq1.upper().replace("U", "T")
                s2 = (seq2 if seq2 is not None else seq1).upper().replace("U", "T")
                # Alphabet guard: non-ACGT bases (N, R, Y, …) would pack as T
                # on the Rust side and hit wrong table entries where Python's
                # dict.get(key, default) misses. Fall back to Python — the
                # native DP is only exact for pure ACGT + U sequences.
                if s1 and s2 and set(s1) <= set("ACGT") and set(s2) <= set("ACGT"):
                    return [
                        (float(e), [tuple(map(int, pr)) for pr in pairs])
                        for e, pairs in _n.dimer_mfe_candidates(s1, s2)
                    ]
            return _dimer_mfe_candidates_py(
                seq1, seq2, engine=engine, material=material, celsius=celsius,
                sodium_M=sodium_M, magnesium_M=magnesium_M,
            )
except ImportError:  # pragma: no cover - extension absent (pure-Python env)
    _n = None
