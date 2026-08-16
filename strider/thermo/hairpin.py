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
* Salt/Mg²⁺ enters by default through the Tan-Chen (2007) whole-helix TBI model
  (:func:`strider.thermo.salt.tan_chen_helix_dg`), folded into the closed-state
  ΔG₃₇ before ΔS is derived.  Tan-Chen reproduces the experimental Mg²⁺ Tm slope
  (~0.7 °C/mM on a DNA molecular-beacon qPCR panel) where the per-base-pair model
  under-shoots (~0.4) and the duplex-calibrated Owczarzy correction over-shoots
  (~2.2).  For stems below 6 bp (outside Tan-Chen's fitted range) it falls back to
  the per-base-pair :func:`dg_per_bp_salt` — the same correction the ensemble DP
  uses.  Select via ``salt_model={"auto","tan_chen","per_bp"}``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

T_REF = 310.15  # K, the 37 °C reference at which ΔG₃₇ tables are defined

# Below this |ΔS| (cal/mol/K) the two-state Tm = ΔH/ΔS is numerically
# unstable: a tiny perturbation in dG37 swings Tm by hundreds of degrees.
# 0.5 cal/mol/K is the smallest dG37 shift (~0.15 kcal/mol) that still
# produces a Tm within the physical range for DNA hairpins.
MIN_DS_CAL = 0.5        # cal/mol/K

# DNA hairpins in aqueous solution melt well below 200 °C (IDT's max
# observed value is ~106 °C).  Tm above this ceiling is a numerical
# artifact of near-degenerate dH/dS and the two-state model does not apply.
MAX_TM_CELSIUS = 200.0  # °C

# Fold temperature for the internal MFE: 25 °C, not the 37 °C dG-table
# reference.  At 37 °C, marginally stable hairpins (Tm 25-37 °C) unfold
# and are missed, yielding NA Tm for sequences that do form hairpins at
# typical assay temperatures.
FOLD_CELSIUS = 25.0


@dataclass(frozen=True)
class HairpinThermo:
    """Two-state hairpin thermodynamics at the requested salt conditions."""
    tm_celsius: float
    dH: float          # kcal/mol
    dS: float          # cal/mol/K
    dG37: float        # kcal/mol, salt-corrected closed-state free energy
    n_pairs: int
    structure: str
    salt_model: str = "per_bp"   # which salt correction was applied ("tan_chen"/"per_bp")


def hairpin_thermo(
    seq: str,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    material: str = "dna",
    structure: list[tuple[int, int]] | str | None = None,
    salt_model: str = "auto",
    paramset=None,
    dangles: int = 0,
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
    salt_model : how Na⁺/Mg²⁺ enters the closed-state ΔG₃₇.
        ``"auto"`` (default) uses the Tan-Chen (2007) whole-helix TBI model
        (:func:`strider.thermo.salt.tan_chen_helix_dg`) for stems ≥
        ``TAN_CHEN_MIN_BP`` bp and falls back to the per-base-pair model below
        that; ``"tan_chen"`` forces Tan-Chen (raises on too-short stems);
        ``"per_bp"`` forces the per-base-pair :func:`dg_per_bp_salt`;
        ``"owczarzy"`` grafts the GC-aware Owczarzy 2004/2008 ΔTm of the *stem*
        onto the hairpin (see :func:`_owczarzy_salt_dg`).  Tan-Chen reproduces the
        experimental Mg²⁺ Tm slope (~0.7 °C/mM) where the per-bp model under-shoots
        (~0.4); the Owczarzy path is GC-aware and uses the corrected Mg term — see
        the README salt benchmark.
    paramset : optional :class:`~strider.thermo.parameters.ParameterSet` to use
        for ΔG and ΔH lookups instead of the default native parameter set.  Pass
        ``load_parameters("mathews2004-dna")`` to use Mathews 2004 DNA parameters
        (matching IDT/ViennaRNA).  When ``None`` (default), uses the native
        parameter set for ``material``.

    Raises
    ------
    ValueError : if the structure has no base pairs or cannot be decomposed
        into hairpin stem-loops (e.g. a pseudoknot).  Multiloops are handled
        by splitting into individual stem-loops and scoring the most stable.
    """
    from strider.thermo.salt import dg_per_bp_salt, tan_chen_helix_dg, TAN_CHEN_MIN_BP
    from strider.thermo.structure_thermo import (
        parse_hairpin_pairs,
        structure_enthalpy,
        structure_free_energy,
    )

    seq = seq.upper().replace("U", "T")

    if structure is None:
        # Fold at 25C (not 37C) via ThermoEngine.  At 37C (the dG-table
        # reference), marginally stable hairpins (Tm 25-37C) unfold and are
        # missed, yielding NA Tm for sequences that do form hairpins at typical
        # assay temperatures.  ThermoEngine applies dG(T) = dH - T*dS, which
        # at 25C makes weak structures stable enough to find.  Salt is NOT
        # passed: the MFE structure must be the same regardless of the
        # caller's salt conditions (the Owczarzy salt model expects a fixed
        # structure and only adjusts Tm).  Salt corrections are applied later
        # via dG37 = dG37_1M + salt_dg.
        from strider import ThermoEngine
        eng = ThermoEngine(
            material=material, celsius=FOLD_CELSIUS,
            parameter_set=paramset if paramset is not None else None,
            dangles=dangles,
        )
        mfe_result = eng.mfe(seq)
        struct_str = mfe_result.structure
    elif isinstance(structure, str):
        struct_str = structure
    else:
        struct_str = _dotbracket(seq, sorted(structure))

    pairs = parse_hairpin_pairs(struct_str)
    if pairs is None:
        # Multiloop or branched structure: try splitting into individual
        # stem-loops and recurse on the most stable one.  This recovers Tm
        # for sequences whose MFE is a multiloop but contain a meaningful
        # hairpin stem (common in GC-rich primers).
        stem_groups = _split_stem_groups(seq, struct_str)
        if stem_groups is None:
            raise ValueError("structure is not a single unbranched hairpin")

        best_result = None
        best_dG = math.inf
        for sub_seq, sub_struct in stem_groups:
            try:
                result = hairpin_thermo(
                    sub_seq, sodium_M, magnesium_M, material,
                    structure=sub_struct, salt_model=salt_model,
                    paramset=paramset, dangles=dangles,
                )
                if result.dG37 < best_dG:
                    best_dG = result.dG37
                    best_result = result
            except Exception:
                continue
        if best_result is None:
            raise ValueError("no valid stem-loop found in multiloop")
        return best_result

    # ΔG and ΔH from the same per-element walk → a consistent ΔS at T_ref.
    dG37_1M = structure_free_energy(seq, struct_str, material, paramset=paramset, dangles=dangles)
    dH = structure_enthalpy(seq, struct_str, material, paramset=paramset, dangles=dangles)

    # Fold salt into the closed-state ΔG₃₇, then derive ΔS at T_ref.  The Tan-Chen
    # whole-helix model needs the stem length and is only fit for ≥6 bp; below that
    # (or when forced off) we use the per-base-pair correction.
    n = len(pairs)
    if salt_model == "owczarzy":
        # Graft the GC-aware, experimentally-calibrated Owczarzy Tm shift onto the
        # hairpin: find the closed-state ΔG offset that reproduces exactly the
        # Owczarzy ΔTm relative to this model's own 1 M-Na⁺ Tm.  This corrects the
        # *salt slope* without disturbing the absolute ΔH/ΔS offset.  The relevant
        # duplex for Owczarzy is the stem, so fGC / N come from the stem arm, not
        # the loop-diluted full probe.
        salt_dg = _owczarzy_salt_dg(seq, pairs, dG37_1M, dH, sodium_M, magnesium_M)
        applied = "owczarzy"
    else:
        use_tc = salt_model == "tan_chen" or (
            salt_model == "auto" and material.lower() in ("dna", "rna")
            and n >= TAN_CHEN_MIN_BP
        )
        if use_tc:
            salt_dg = tan_chen_helix_dg(n, sodium_M, magnesium_M, material)
            applied = "tan_chen"
        else:
            salt_dg = n * dg_per_bp_salt(sodium_M, magnesium_M, material=material)
            applied = "per_bp"
    dG37 = dG37_1M + salt_dg
    dS_kcal = (dH - dG37) / T_REF               # kcal/mol/K
    if abs(dS_kcal) < MIN_DS_CAL / 1000.0:
        raise ValueError("degenerate entropy — cannot define a melting point")
    tm_K = dH / dS_kcal
    tm_C = tm_K - 273.15
    if tm_C > MAX_TM_CELSIUS:
        raise ValueError("Tm exceeds 200C — two-state model not applicable")
    return HairpinThermo(
        tm_celsius=tm_C,
        dH=dH,
        dS=dS_kcal * 1000.0,
        dG37=dG37,
        n_pairs=n,
        structure=struct_str,
        salt_model=applied,
    )


def hairpin_tm(
    seq: str,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    material: str = "dna",
    salt_model: str = "auto",
    paramset=None,
    dangles: int = 0,
) -> float:
    """Melting temperature (°C) of the predicted hairpin. See :func:`hairpin_thermo`
    (raises ``ValueError`` when the sequence does not fold into a hairpin)."""
    return hairpin_thermo(seq, sodium_M, magnesium_M, material,
                          salt_model=salt_model, paramset=paramset,
                          dangles=dangles).tm_celsius


def fraction_folded(
    seq: str,
    celsius: float,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    material: str = "dna",
    salt_model: str = "auto",
    paramset=None,
    dangles: int = 0,
) -> float:
    """
    Two-state folded fraction at ``celsius`` — the quantity a beacon melt
    (fluorophore dequenching) actually traces out.

    Sequences with no stable hairpin fold at all → their melt is a flat zero;
    ``fraction_folded`` returns ``0.0`` for them (it used to raise ``ValueError``,
    which made every non-folding primer crash melt-curve analysis).

    Only the no-base-pairs case maps to zero.  Caller-mistake conditions
    (unknown ``salt_model``, bad ``material``, degenerate ΔS, multiloop MFE)
    still raise instead of being silently flattened.
    """
    from strider import ThermoEngine

    norm = seq.upper().replace("U", "T")
    eng = ThermoEngine(
        material=material, celsius=FOLD_CELSIUS,
        parameter_set=paramset if paramset is not None else None,
        dangles=dangles,
    )
    mfe_result = eng.mfe(norm)
    struct_str = mfe_result.structure
    if "(" not in struct_str:
        return 0.0  # no stable fold — flat melt curve
    th = hairpin_thermo(seq, sodium_M, magnesium_M, material,
                        salt_model=salt_model, paramset=paramset,
                        dangles=dangles, structure=struct_str)
    T = celsius + 273.15
    dG_T = th.dH - T * th.dS / 1000.0           # ΔG(T) of the closed state
    R = 1.987e-3
    return 1.0 / (1.0 + math.exp(dG_T / (R * T)))


# ─── internals ────────────────────────────────────────────────────────────────

def _owczarzy_salt_dg(seq, pairs, dG37_1M, dH, sodium_M, magnesium_M):
    """Closed-state ΔG₃₇ offset reproducing the Owczarzy ΔTm for the stem.

    The two-state hairpin obeys ``Tm = ΔH/ΔS`` with ``ΔS = (ΔH − ΔG₃₇)/T_ref``.
    Adding ``salt_dg`` to ΔG₃₇ moves Tm; solving for the ``salt_dg`` that yields
    exactly ``Tm₁ₘ + ΔTm_owczarzy`` gives

        salt_dg = T_ref · ΔH · (1/Tm₁ₘ − 1/(Tm₁ₘ + ΔTm)),

    with Tm in kelvin.  ΔTm comes from :func:`strider.thermo.salt.owczarzy_tm_correction`
    evaluated on the *stem* (its 5' arm: one base per base pair), so fGC and the
    base-pair count match the duplex the Owczarzy fit was calibrated on.  Returns
    0 at the 1 M Na⁺ / 0 Mg²⁺ reference (ΔTm = 0).
    """
    from strider.thermo.salt import owczarzy_tm_correction
    dS_kcal_1M = (dH - dG37_1M) / T_REF
    if dS_kcal_1M == 0:
        return 0.0
    tm1_K = dH / dS_kcal_1M
    stem_arm = "".join(seq[i] for i, _ in pairs)  # 5' strand of the stem
    dTm = owczarzy_tm_correction(stem_arm, sodium_M, magnesium_M)
    denom = tm1_K + dTm
    if denom == 0 or tm1_K == 0:
        return 0.0
    return T_REF * dH * (1.0 / tm1_K - 1.0 / denom)


def _dotbracket(seq: str, pairs: list[tuple[int, int]]) -> str:
    s = ["."] * len(seq)
    for i, j in pairs:
        s[i], s[j] = "(", ")"
    return "".join(s)


def _split_stem_groups(
    seq: str, struct: str,
) -> list[tuple[str, str]] | None:
    """Split a multiloop dot-bracket into individual stem-loops.

    Walks the base-pair list and groups contiguous nested pairs into stems.
    Each stem (with its closing hairpin loop) becomes a standalone hairpin
    sub-structure.  Returns ``None`` if the structure is a single hairpin
    (no split needed) or malformed (unbalanced / pseudoknot).

    Mirrors the ``splitStemGroups`` algorithm in Oligool's HairpinSVG.tsx.
    """
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for k, c in enumerate(struct):
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

    groups: list[list[tuple[int, int]]] = []
    cur: list[tuple[int, int]] = [pairs[0]]
    prev_right = pairs[0][1]

    for k in range(1, len(pairs)):
        if pairs[k][1] < prev_right:
            cur.append(pairs[k])
            prev_right = pairs[k][1]
        else:
            groups.append(cur)
            cur = [pairs[k]]
            prev_right = pairs[k][1]
    groups.append(cur)

    if len(groups) <= 1:
        return None

    result: list[tuple[str, str]] = []
    for g in groups:
        start = g[0][0]
        end = g[0][1]  # outermost pair has the largest closing position
        result.append((seq[start:end + 1], struct[start:end + 1]))
    return result
