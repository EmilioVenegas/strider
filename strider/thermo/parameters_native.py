"""
Built-in MIT-licensed parameter set.

Assembled from primary-literature constants — SantaLucia 2004 (DNA) and
Mathews 1999 / Turner 2004 (RNA) — covering the full set of tables consumed
by strider's native MFE / pfunc engines: stack, hairpin loop sizes, bulge
loop sizes, interior loop sizes, terminal penalty, multiloop coefficients,
dangles, terminal/hairpin/interior mismatches, 1×1 / 1×2 / 2×2 interior
loops, sequence-specific triloop / tetraloop bonuses, and coaxial stacking
(the DNA-specific advanced tables are populated for DNA and absent for
RNA — RNA only ships dangle and terminal-mismatch tables on top of the
common loop tables).
"""

from __future__ import annotations

import numpy as np

from strider.thermo.parameters import ParameterSet


# ─── stack tables ─────────────────────────────────────────────────────────────

# Build the Turner-format stack dict from the dinucleotide NN ΔG37 table.
# Stack key is top5+top3+bot5+bot3, where the bottom strand is read 5'→3'.
# For a Watson–Crick stack, bot5+bot3 = reverse-complement(top5+top3).

_COMPL_DNA = str.maketrans("ACGT", "TGCA")
_COMPL_RNA = str.maketrans("ACGU", "UGCA")


def _stack_key(top: str, complement_map) -> str:
    """Return the 4-char Turner-format stack key for the Watson-Crick stack on top dinuc ``top``."""
    bot = top.translate(complement_map)[::-1]
    return top + bot


def _stack_dict_dna() -> dict[str, float]:
    """16-entry DNA stack table from SantaLucia 2004 NN parameters (ΔG37 kcal/mol)."""
    from strider.thermo.nn_dna import DNA_NN
    out: dict[str, float] = {}
    for dinuc, (_h, _s, g) in DNA_NN.items():
        key = _stack_key(dinuc, _COMPL_DNA)
        out[key] = float(g)
    return out


def _stack_dict_rna() -> dict[str, float]:
    """16-entry RNA stack table from Mathews 1999 NN parameters (ΔG37 kcal/mol)."""
    from strider.thermo.nn_rna import RNA_NN
    out: dict[str, float] = {}
    for dinuc, (_h, _s, g) in RNA_NN.items():
        # Keys use T (not U) so DNA and RNA stack tables share the same alphabet.
        key = _stack_key(dinuc, _COMPL_RNA).replace("U", "T")
        out[key] = float(g)
    return out


def _stack_dh_dna() -> dict[str, float]:
    """16-entry DNA stack ΔH (kcal/mol) for temperature extrapolation."""
    from strider.thermo.nn_dna import DNA_NN
    return {_stack_key(d, _COMPL_DNA): float(h) for d, (h, _s, _g) in DNA_NN.items()}


def _stack_dh_rna() -> dict[str, float]:
    """16-entry RNA stack ΔH (kcal/mol) for temperature extrapolation."""
    from strider.thermo.nn_rna import RNA_NN
    return {_stack_key(d, _COMPL_RNA).replace("U", "T"): float(h)
            for d, (h, _s, _g) in RNA_NN.items()}


# ─── loop size tables ─────────────────────────────────────────────────────────

# Source: Turner 2004 / Mathews 1999 loop initiation tables.  Index = unpaired
# nucleotides in the loop; entries before MIN_HAIRPIN_LOOP=3 are sentinels.

_INF = 30.0  # large positive number used in place of float('inf')

# DNA hairpin loop initiation (SantaLucia / Mathews compromise; same numbers
# already inlined in strider.structure.mfe._hairpin_energy).
_HAIRPIN_DNA = np.array(
    [_INF, _INF, _INF, 4.1, 4.3, 4.9, 4.4, 4.3, 4.1, 4.0,
     4.7, 4.9, 5.0, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6,
     5.6, 5.7, 5.7, 5.8, 5.8, 5.9, 5.9, 6.0, 6.0, 6.1]
)

_HAIRPIN_RNA = np.array(
    [_INF, _INF, _INF, 5.4, 5.6, 5.7, 5.4, 6.0, 5.5, 6.4,
     6.5, 6.6, 6.7, 6.8, 6.9, 6.9, 7.0, 7.1, 7.1, 7.2,
     7.2, 7.3, 7.3, 7.4, 7.4, 7.5, 7.5, 7.6, 7.6, 7.7]
)

# Bulge loop initiation (Turner 2004 short-bulge table extended logarithmically).
_BULGE = np.array(
    [_INF, 3.8, 2.8, 3.2, 3.6, 4.0, 4.4, 4.6, 4.7, 4.8,
     4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.4, 5.5, 5.5, 5.6,
     5.7, 5.7, 5.8, 5.8, 5.8, 5.9, 5.9, 6.0, 6.0, 6.0]
)

# Interior loop initiation (Turner 2004).  Index 1 is unused (no length-1 interior).
_INTERIOR = np.array(
    [_INF, _INF, 4.1, 5.1, 4.9, 5.3, 5.7, 5.9, 6.1, 6.3,
     6.5, 6.6, 6.8, 6.9, 7.0, 7.1, 7.2, 7.2, 7.3, 7.4,
     7.5, 7.5, 7.6, 7.6, 7.7, 7.7, 7.8, 7.8, 7.9, 7.9]
)

# Asymmetric loop penalty (Ninio, 1979); five-term lookup matching the Turner schema.
_ASYM_NINIO = np.array([0.4, 0.3, 0.2, 0.1, 3.0])


# ─── builder ──────────────────────────────────────────────────────────────────

def build_native_paramset(material: str = "DNA") -> ParameterSet:
    """
    Construct a :class:`ParameterSet` from strider's built-in NN tables.

    The DNA path populates the full advanced table set (dangles, terminal
    mismatch, hairpin mismatch, interior mismatch, interior_1_1 / 1_2 /
    2_2, hairpin triloop / tetraloop, coaxial stacking) by referencing the
    module-level constants in :mod:`strider.thermo.parameters_dna`.  The
    RNA path includes the subset that primary literature defines for RNA
    (dangles, terminal mismatch, triloop / tetraloop) — `interior_1_1`
    and friends are DNA-only in the bundled tables.

    Parameters
    ----------
    material : ``"DNA"`` or ``"RNA"``
    """
    material = material.upper()
    if material == "DNA":
        stack = _stack_dict_dna()
        stack_dH = _stack_dh_dna()
        hairpin = _HAIRPIN_DNA
        wobble = False
        terminal_penalty = {"AT": 0.45, "TA": 0.45}    # SantaLucia INIT_AT penalty
        advanced = _advanced_dna_tables()
    elif material == "RNA":
        stack = _stack_dict_rna()
        stack_dH = _stack_dh_rna()
        hairpin = _HAIRPIN_RNA
        wobble = True
        terminal_penalty = {"AT": 0.45, "TA": 0.45, "GT": 0.45}  # AU/GU penalty
        advanced = _advanced_rna_tables()
    else:
        raise ValueError(f"material must be 'DNA' or 'RNA', got {material!r}")

    dG = {
        "stack": stack,
        "hairpin_size": hairpin,
        "bulge_size": _BULGE,
        "interior_size": _INTERIOR,
        "asymmetry_ninio": _ASYM_NINIO,
        "terminal_penalty": terminal_penalty,
        # Multiloop coefficients: Turner 2004 / SantaLucia defaults.
        "multiloop_init": 3.4,
        "multiloop_pair": 0.4,
        "multiloop_base": 0.0,
        "join_penalty": 1.96,
        "log_loop_penalty": 1.07,
        **advanced,
    }
    if material == "DNA":
        # Full DNA loop ΔH from SantaLucia 2004 / Mathews 1999 (UNAFold `.dh`
        # tables), generated by scripts/generate_dna_enthalpy_tables.py.  Loop
        # *initiation* ΔH is 0 (purely entropic, per the UNAFold convention),
        # so the hairpin/bulge/interior size arrays are zeroed; sequence- and
        # context-specific ΔH come from the stack / mismatch / tri-/tetraloop
        # tables.  Enables temperature extrapolation and a unimolecular Tm.
        from strider.thermo._dna_enthalpy_generated import (
            STACK_DH, HAIRPIN_MISMATCH_DH, INTERIOR_MISMATCH_DH,
            HAIRPIN_TRILOOP_DH, HAIRPIN_TETRALOOP_DH,
        )
        dH = {
            "stack": dict(STACK_DH),
            "hairpin_size": np.zeros(len(hairpin)),
            "bulge_size": np.zeros(len(_BULGE)),
            "interior_size": np.zeros(len(_INTERIOR)),
            "hairpin_mismatch": dict(HAIRPIN_MISMATCH_DH),
            "interior_mismatch": dict(INTERIOR_MISMATCH_DH),
            "hairpin_triloop": dict(HAIRPIN_TRILOOP_DH),
            "hairpin_tetraloop": dict(HAIRPIN_TETRALOOP_DH),
        }
    else:
        # RNA loop ΔH not yet curated — keep the coarse ΔG-copy approximation.
        dH = {
            "stack": stack_dH,
            "hairpin_size": hairpin,
            "bulge_size": _BULGE,
            "interior_size": _INTERIOR,
        }

    return ParameterSet(
        name=f"native-{material.lower()}",
        material=material,
        default_wobble_pairing=wobble,
        dG=dG,
        dH=dH,
        source_path=None,
        comment="strider built-in: SantaLucia 2004 (DNA) / Mathews 1999 (RNA)",
    )


# ─── advanced-table population ────────────────────────────────────────────────


def _advanced_dna_tables() -> dict[str, dict[str, float]]:
    """Re-export the DNA advanced ΔG tables verbatim from the module constants."""
    from strider.thermo.parameters_dna import (
        DANGLE_3, DANGLE_5,
        TERMINAL_MISMATCH, HAIRPIN_MISMATCH, INTERIOR_MISMATCH,
        INTERIOR_1_1, INTERIOR_1_2, INTERIOR_2_2,
        HAIRPIN_TRILOOP, HAIRPIN_TETRALOOP,
        COAXIAL_STACK,
    )
    return {
        "dangle_3": dict(DANGLE_3),
        "dangle_5": dict(DANGLE_5),
        "terminal_mismatch": dict(TERMINAL_MISMATCH),
        "hairpin_mismatch": dict(HAIRPIN_MISMATCH),
        "interior_mismatch": dict(INTERIOR_MISMATCH),
        "interior_1_1": dict(INTERIOR_1_1),
        "interior_1_2": dict(INTERIOR_1_2),
        "interior_2_2": dict(INTERIOR_2_2),
        "hairpin_triloop": dict(HAIRPIN_TRILOOP),
        "hairpin_tetraloop": dict(HAIRPIN_TETRALOOP),
        "coaxial_stack": dict(COAXIAL_STACK),
    }


def _advanced_rna_tables() -> dict[str, dict[str, float]]:
    """
    Re-export the RNA advanced ΔG tables.

    RNA bundles only the subset that strider's RNA tables actually carry —
    dangles, terminal mismatch, triloop / tetraloop.  Interior_1_1/2 and
    hairpin / interior mismatch are DNA-only in the current data set.
    """
    from strider.thermo.parameters_rna import (
        DANGLE_3, DANGLE_5,
        TERMINAL_MISMATCH, HAIRPIN_TRILOOP, HAIRPIN_TETRALOOP,
    )
    return {
        "dangle_3": dict(DANGLE_3),
        "dangle_5": dict(DANGLE_5),
        "terminal_mismatch": dict(TERMINAL_MISMATCH),
        "hairpin_triloop": dict(HAIRPIN_TRILOOP),
        "hairpin_tetraloop": dict(HAIRPIN_TETRALOOP),
    }
