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
        mfe = engine.mfe(seq1, seq2)
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
