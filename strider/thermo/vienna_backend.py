"""
Optional ViennaRNA backend for ThermoEngine.

ViennaRNA is GPL-licensed and must be installed separately:
    pip install ViennaRNA   (or via conda)

This module is imported lazily — strider works without it. It is a *cross-check*
backend (never auto-selected); request it explicitly with ``backend='vienna'``.

Material handling
-----------------
ViennaRNA's process-global default parameters are RNA (Turner 2004). To honour the
engine's ``material``, DNA folds load ViennaRNA's bundled Mathews-2004 DNA set just
before constructing the fold_compound, then restore the RNA default — so a DNA fold
never leaks DNA parameters into other in-process ViennaRNA users. The bundled param
sets ship with the user's own ViennaRNA install; strider redistributes nothing.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def is_available() -> bool:
    """Return True if ViennaRNA (RNA module) is importable."""
    try:
        import RNA  # noqa: F401
        return True
    except ImportError:
        return False


def _fold_compound(sequence: str, celsius: float, material: str):
    """Build a ViennaRNA fold_compound with material-appropriate parameters.

    The param set must be loaded *before* the model details and fold_compound are
    constructed (ViennaRNA snapshots the global parameters at that point), then the
    RNA default is restored so DNA parameters never persist past the call.
    """
    import RNA
    if material == "dna":
        RNA.params_load_DNA_Mathews2004()
        try:
            md = RNA.md()
            md.temperature = celsius
            return RNA.fold_compound(sequence, md)
        finally:
            RNA.params_load_RNA_Turner2004()
    md = RNA.md()
    md.temperature = celsius
    return RNA.fold_compound(sequence, md)


def fold(sequence: str, celsius: float = 37.0, material: str = "rna") -> tuple[str, float]:
    """MFE structure and energy via ViennaRNA."""
    fc = _fold_compound(sequence, celsius, material)
    structure, mfe = fc.mfe()
    return structure, mfe


def pf_fold(
    sequence: str, celsius: float = 37.0, material: str = "rna"
) -> tuple[float, "np.ndarray"]:
    """Ensemble free energy (kcal/mol) and pair-probability matrix via ViennaRNA.

    ``fold_compound.pf()`` returns ``(propensity_structure, ensemble_free_energy)``
    — the second element is the true ensemble ΔG = −RT·ln Z. Pair probabilities
    come from ``fold_compound.bpp()`` (a 1-indexed ``(n+1)×(n+1)`` table); the
    legacy ``get_pr`` accessor no longer exists in the SWIG bindings.
    """
    import numpy as np
    fc = _fold_compound(sequence, celsius, material)
    _, dG_ens = fc.pf()  # ensemble free energy, kcal/mol
    bpp = fc.bpp()       # 1-indexed lower/upper-triangular pair probabilities
    n = len(sequence)
    bppm = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            p = bpp[i][j]
            bppm[i - 1][j - 1] = p
            bppm[j - 1][i - 1] = p
    return float(dG_ens), bppm


def co_fold(
    seq1: str, seq2: str, celsius: float = 37.0, material: str = "rna"
) -> tuple[str, float]:
    """Co-folding of two strands via ViennaRNA (dimer MFE)."""
    fc = _fold_compound(seq1 + "&" + seq2, celsius, material)
    structure, mfe = fc.mfe_dimer()
    return structure, mfe
