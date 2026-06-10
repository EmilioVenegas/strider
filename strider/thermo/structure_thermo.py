"""
Structure-resolved thermodynamics for a *folded* nucleic-acid structure:
ΔG, ΔH, ΔS and a unimolecular melting temperature.

Motivation
----------
``ThermoEngine.melting_temperature`` solves the *bimolecular* duplex equation
``Tm = ΔH / (ΔS + R·ln(CT/4))``.  A hairpin (or any intramolecular fold) melts as
a *unimolecular* two-state transition whose Tm is concentration-independent:

    Tm = ΔH / ΔS

This module computes ΔG and ΔH for a specific folded structure by walking the
structure and summing the engine's own per-element energies
(:func:`strider.thermo.ensemble._stack_energy`, ``_hairpin_loop_energy``,
``_interior_bulge_energy``).  The walk is validated to reproduce
``strider.structure.mfe.fold_mfe`` energy exactly for single hairpins
(see ``scripts/validate_loop_decomposition.py``).  ΔH is obtained by running the
identical walk under a :func:`~strider.thermo._param_context.param_context`
override whose tables are the ΔH parameters, then

    ΔS = (ΔH − ΔG) / T_ref        (T_ref = 310.15 K, the ΔG-table reference)
    Tm = ΔH / ΔS − 273.15         (+ Owczarzy salt correction)

Accuracy of ΔH (hence Tm) depends on the completeness of the parameter set's ΔH
tables; see :func:`strider.thermo.parameters_native.build_native_paramset`.
"""
from __future__ import annotations

from strider.thermo._param_context import param_context
from strider.thermo.ensemble import (
    _stack_energy,
    _hairpin_loop_energy,
    _interior_bulge_energy,
)

T_REF_K = 310.15  # 37 °C — reference temperature of the ΔG parameter tables


class _TableView:
    """Minimal object exposing a ``.dG`` attribute so :func:`param_context` will
    route energy lookups through an arbitrary table dict (e.g. the ΔH tables)."""

    __slots__ = ("dG",)

    def __init__(self, tables: dict):
        self.dG = tables


def parse_hairpin_pairs(structure: str):
    """Return base pairs ``[(i, j), ...]`` outermost-first for a single unbranched
    hairpin, or ``None`` for multiloops / pseudoknots / unpaired structures."""
    stack, pairs = [], []
    for k, c in enumerate(structure):
        if c == "(":
            stack.append(k)
        elif c == ")":
            if not stack:
                return None
            pairs.append((stack.pop(), k))
        elif c != ".":
            return None
    if stack or not pairs:
        return None
    pairs.sort()
    for a in range(1, len(pairs)):
        if not (pairs[a][0] > pairs[a - 1][0] and pairs[a][1] < pairs[a - 1][1]):
            return None  # branched (multiloop) — not a single hairpin
    return pairs


def _sum_elements(seq: str, pairs, material: str, T: float) -> float:
    """Sum per-element energy for a single hairpin using the engine's own
    decomposition.  Whichever tables are active (via ``param_context``) decide
    whether this returns ΔG or ΔH."""
    total = 0.0
    for k in range(len(pairs) - 1):
        i, j = pairs[k]
        ip, jp = pairs[k + 1]
        nl, nr = ip - i - 1, j - jp - 1
        if nl == 0 and nr == 0:
            total += _stack_energy(seq, i, j, material)
        else:
            total += _interior_bulge_energy(seq, i, j, ip, jp, nl, nr, material)
    il, jl = pairs[-1]
    total += _hairpin_loop_energy(seq, il, jl, material, T)
    return total


def structure_free_energy(seq: str, structure: str, material: str = "dna",
                          paramset=None) -> float | None:
    """ΔG (kcal/mol) of a folded hairpin from the ΔG tables.  Reproduces
    ``fold_mfe`` energy exactly for single hairpins."""
    pairs = parse_hairpin_pairs(structure)
    if pairs is None:
        return None
    if paramset is not None:
        with param_context(paramset):
            return _sum_elements(seq, pairs, material, T_REF_K)
    return _sum_elements(seq, pairs, material, T_REF_K)


def structure_enthalpy(seq: str, structure: str, material: str = "dna",
                       paramset=None) -> float | None:
    """ΔH (kcal/mol) of a folded hairpin, via the same walk run against the ΔH
    tables.  ``paramset`` defaults to the native set for ``material``."""
    pairs = parse_hairpin_pairs(structure)
    if pairs is None:
        return None
    if paramset is None:
        from strider.thermo.parameters import load_parameters
        paramset = load_parameters("native") if material == "dna" \
            else load_parameters("native-rna")
    with param_context(_TableView(paramset.dH)):
        return _sum_elements(seq, pairs, material, T_REF_K)


def hairpin_thermo(seq: str, structure: str | None = None, *,
                   sodium_M: float = 1.0, magnesium_M: float = 0.0,
                   celsius: float = 37.0, material: str = "dna",
                   paramset=None) -> dict | None:
    """ΔG, ΔH, ΔS and unimolecular Tm for a hairpin.

    If ``structure`` is omitted the MFE structure is folded at ``celsius``.
    Returns ``None`` when the (predicted) structure is not a single hairpin or
    has no base pairs.  Tm is salt-corrected with Owczarzy relative to 1 M Na+.
    """
    if structure is None:
        from strider.structure.mfe import fold_mfe
        structure, _, _ = fold_mfe(seq, celsius, material)

    pairs = parse_hairpin_pairs(structure)
    if pairs is None:
        return None

    dG = structure_free_energy(seq, structure, material, paramset)
    dH = structure_enthalpy(seq, structure, material, paramset)
    if dG is None or dH is None:
        return None

    dS = (dH - dG) / T_REF_K  # kcal/mol/K
    tm = None
    if dS < 0 and dH < 0:
        tm = dH / dS - 273.15  # unimolecular two-state, 1 M Na+ reference
        if sodium_M != 1.0 or magnesium_M > 0:
            from strider.thermo.salt import owczarzy_tm_correction
            stem = "".join(seq[i] for i, _ in pairs)  # GC content of the stem
            tm += owczarzy_tm_correction(stem, sodium_M, magnesium_M)

    return {
        "structure": structure,
        "dG": dG,                    # kcal/mol at 37 °C
        "dH": dH,                    # kcal/mol
        "dS": dS * 1000.0,           # cal/mol/K
        "Tm": tm,                    # °C, or None if no stable fold
    }
