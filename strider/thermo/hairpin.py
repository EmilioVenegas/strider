"""
Self-consistent hairpin melting temperature.

A unimolecular hairpin melts in a two-state fashion (folded ⇌ open), so its
melting temperature is the temperature at which the folding free energy
vanishes:

    Tm = ΔH / ΔS         (concentration-independent, unlike a duplex)

Strider's folding model (``structure.mfe`` / ``thermo.ensemble``) is built on
ΔG₃₇ stack/loop parameters, so on its own it has no enthalpy and its partition
function never melts.  We recover a consistent enthalpy from the native DNA
parameter set, which now carries real loop ΔH tables (stack ΔH plus
mismatch / triloop / tetraloop ΔH; loop *initiation* ΔH is purely entropic and
set to zero — see :func:`strider.thermo.parameters_native.build_native_paramset`).

ΔG and ΔH are obtained from the *same* structure walk
(:func:`strider.thermo.structure_thermo.structure_free_energy` /
:func:`~strider.thermo.structure_thermo.structure_enthalpy`), so the derived

    ΔS = (ΔH − ΔG₃₇) / T_ref

is exact at the table reference temperature.  This reproduces independent
reference engines (e.g. seqfold) to <1 °C at 1 M Na⁺, and — unlike a stack-only
ΔH sum — it also counts the loop-closing terminal-mismatch enthalpy, which
matters for GC-rich stems.

Caveats
-------
* Single, unbranched hairpin only (one stem, optionally with bulges/internal
  loops, plus one hairpin loop).  Multiloops / pseudoknots raise ``ValueError``
  — use the full ensemble for those.
* Tm is *hypersensitive* to the ΔH/ΔS bookkeeping: a ~few-% shift in either
  moves Tm by tens of °C.  Treat absolute Tm as calibratable, not exact, and
  anchor it against experimental (e.g. qPCR) melts.
* Salt/Mg²⁺ enters through the per-base-pair ΔG model
  (:func:`strider.thermo.salt.dg_per_bp_salt`), folded into the closed-state ΔG
  before ΔS is derived — the same correction the ensemble DP uses, not the
  (oversized, duplex-calibrated) Owczarzy Tm shift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

T_REF = 310.15  # K, the 37 °C reference at which ΔG₃₇ tables are defined


@dataclass(frozen=True)
class HairpinThermo:
    """Two-state hairpin thermodynamics at the requested salt conditions."""
    tm_celsius: float
    dH: float          # kcal/mol
    dS: float          # cal/mol/K
    dG37: float        # kcal/mol, salt-corrected closed-state free energy
    n_pairs: int
    structure: str


def hairpin_thermo(
    seq: str,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    material: str = "dna",
    structure: list[tuple[int, int]] | str | None = None,
) -> HairpinThermo:
    """
    Two-state thermodynamics (Tm, ΔH, ΔS, ΔG₃₇) for a hairpin.

    Parameters
    ----------
    seq : strand sequence.
    sodium_M, magnesium_M : ion concentrations (1 M Na⁺ / 0 Mg²⁺ = reference).
    structure : optional structure to score; either a list of ``(i, j)`` base
        pairs or a dot-bracket string.  If omitted the MFE hairpin is predicted
        with :func:`strider.structure.mfe.fold_mfe`.

    Raises
    ------
    ValueError : if the structure has no base pairs or is not a single
        unbranched hairpin (e.g. a multiloop).
    """
    from strider.structure.mfe import fold_mfe
    from strider.thermo.salt import dg_per_bp_salt
    from strider.thermo.structure_thermo import (
        parse_hairpin_pairs,
        structure_enthalpy,
        structure_free_energy,
    )

    seq = seq.upper().replace("U", "T")

    if structure is None:
        struct_str, _, _ = fold_mfe(seq, 37.0, material)
    elif isinstance(structure, str):
        struct_str = structure
    else:
        struct_str = _dotbracket(seq, sorted(structure))

    pairs = parse_hairpin_pairs(struct_str)
    if pairs is None:
        raise ValueError("structure is not a single unbranched hairpin")

    # ΔG and ΔH from the same per-element walk → a consistent ΔS at T_ref.
    dG37_1M = structure_free_energy(seq, struct_str, material)
    dH = structure_enthalpy(seq, struct_str, material)

    # Fold salt into the closed-state ΔG₃₇ (per closed base pair), then derive ΔS.
    dG37 = dG37_1M + len(pairs) * dg_per_bp_salt(sodium_M, magnesium_M)
    dS_kcal = (dH - dG37) / T_REF               # kcal/mol/K
    if dS_kcal == 0:
        raise ValueError("degenerate entropy — cannot define a melting point")
    tm_K = dH / dS_kcal
    return HairpinThermo(
        tm_celsius=tm_K - 273.15,
        dH=dH,
        dS=dS_kcal * 1000.0,
        dG37=dG37,
        n_pairs=len(pairs),
        structure=struct_str,
    )


def hairpin_tm(
    seq: str,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    material: str = "dna",
) -> float:
    """Melting temperature (°C) of the predicted hairpin. See :func:`hairpin_thermo`."""
    return hairpin_thermo(seq, sodium_M, magnesium_M, material).tm_celsius


def fraction_folded(
    seq: str,
    celsius: float,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    material: str = "dna",
) -> float:
    """
    Two-state folded fraction at ``celsius`` — the quantity a beacon melt
    (fluorophore dequenching) actually traces out.
    """
    th = hairpin_thermo(seq, sodium_M, magnesium_M, material)
    T = celsius + 273.15
    dG_T = th.dH - T * th.dS / 1000.0           # ΔG(T) of the closed state
    R = 1.987e-3
    return 1.0 / (1.0 + math.exp(dG_T / (R * T)))


# ─── internals ────────────────────────────────────────────────────────────────

def _dotbracket(seq: str, pairs: list[tuple[int, int]]) -> str:
    s = ["."] * len(seq)
    for i, j in pairs:
        s[i], s[j] = "(", ")"
    return "".join(s)
