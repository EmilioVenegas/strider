"""
Structure-resolved thermodynamics for a *folded* nucleic-acid structure: the
ΔG and ΔH of a specific fold.

This is the low-level engine behind the unimolecular hairpin melting temperature
in :mod:`strider.thermo.hairpin` (``Tm = ΔH / ΔS``).  It computes ΔG and ΔH by
walking the structure and summing the engine's own per-element energies
(:func:`strider.thermo.ensemble._stack_energy`, ``_hairpin_loop_energy``,
``_interior_bulge_energy``).  The walk is validated to reproduce
``strider.structure.mfe.fold_mfe`` energy exactly for single hairpins
(see ``scripts/validate_loop_decomposition.py``).  ΔH is obtained by running the
identical walk under a :func:`~strider.thermo._param_context.param_context`
override whose tables are the ΔH parameters, so that

    ΔS = (ΔH − ΔG) / T_ref        (T_ref = 310.15 K, the ΔG-table reference)

is exact at the reference temperature.  Accuracy of ΔH (hence Tm) depends on the
completeness of the parameter set's ΔH tables; see
:func:`strider.thermo.parameters_native.build_native_paramset`.
"""
from __future__ import annotations

from strider.thermo._param_context import param_context, lookup_table
from strider.thermo.ensemble import (
    _stack_energy,
    _hairpin_loop_energy,
    _interior_bulge_energy,
    _terminal_pair_penalty,
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


def parse_dimer_pairs(structure: str | list[tuple[int, int]], n1: int):
    """Return base pairs ``[(i, j), ...]`` outermost-first for a single nested
    dimer helix where every pair crosses the strand junction (``i < n1 <= j``).

    Raises ``ValueError`` if the structure is not a valid single nested dimer
    helix or has fewer than two base pairs.
    """
    if isinstance(structure, str):
        stack, raw = [], []
        for k, c in enumerate(structure):
            if c == "(":
                stack.append(k)
            elif c == ")":
                if not stack:
                    raise ValueError("unbalanced dot-bracket structure")
                raw.append((stack.pop(), k))
            elif c != ".":
                raise ValueError("invalid character in dot-bracket structure")
        if stack:
            raise ValueError("unbalanced dot-bracket structure")
        pairs = raw
    else:
        pairs = list(structure)

    # The MFE cofold may also contain intra-strand hairpins; for dimer Tm we
    # only score the inter-strand helix.
    pairs = [(i, j) for i, j in pairs if i < n1 <= j]

    if len(pairs) < 2:
        raise ValueError("dimer helix must contain at least two inter-strand base pairs")

    pairs.sort()
    for a in range(1, len(pairs)):
        prev_i, prev_j = pairs[a - 1]
        i, j = pairs[a]
        if not (prev_i < i and j < prev_j):
            raise ValueError("dimer structure must be a single nested helix")
    return pairs


def _sum_dimer_elements(seq: str, seq1_len: int, pairs, material: str, T: float) -> float:
    """Sum per-element ΔG/ΔH for a single nested bimolecular duplex.

    The walk uses the same stack/interior/bulge decomposition as the hairpin
    walk, but replaces the hairpin-loop term with terminal-pair / dangle
    contributions at both helix ends.  Whichever tables are active via
    :func:`param_context` decide whether the returned value is ΔG or ΔH.

    The bimolecular association ``JOIN_PENALTY`` is intentionally omitted here;
    it belongs in the concentration-dependent Tm calculation (via the ``ln(C)``
    term), not in the closed-state duplex free energy.  This matches the
    reporting convention used by primer3 and IDT OligoAnalyzer.
    """
    n = len(seq)
    paired = set()
    for i, j in pairs:
        paired.add(i)
        paired.add(j)

    total = 0.0
    for k in range(len(pairs) - 1):
        i, j = pairs[k]
        ip, jp = pairs[k + 1]
        nl, nr = ip - i - 1, j - jp - 1
        if nl == 0 and nr == 0:
            total += _stack_energy(seq, i, j, material)
        else:
            total += _interior_bulge_energy(seq, i, j, ip, jp, nl, nr, material)

    if material == "dna":
        from strider.thermo.parameters_dna import (
            DANGLE_3, DANGLE_5,
        )
    else:
        from strider.thermo.parameters_rna import (
            DANGLE_3, DANGLE_5,
        )
    dangle_5 = lookup_table("dangle_5", DANGLE_5)
    dangle_3 = lookup_table("dangle_3", DANGLE_3)

    i_out, j_out = pairs[0]
    total += _terminal_pair_penalty(seq, i_out, j_out, material)
    if i_out - 1 >= 0 and (i_out - 1) not in paired:
        d5 = dangle_5.get(seq[i_out] + seq[j_out] + seq[i_out - 1])
        if d5 is not None and d5 < 0:
            total += d5
    if j_out + 1 < n and (j_out + 1) not in paired:
        d3 = dangle_3.get(seq[j_out - 1] + seq[j_out] + seq[j_out + 1])
        if d3 is not None and d3 < 0:
            total += d3

    i_in, j_in = pairs[-1]
    total += _terminal_pair_penalty(seq, i_in, j_in, material)
    if j_in - 1 >= seq1_len and (j_in - 1) not in paired:
        d5 = dangle_5.get(seq[j_in] + seq[i_in] + seq[j_in - 1])
        if d5 is not None and d5 < 0:
            total += d5
    if i_in + 1 < seq1_len and (i_in + 1) not in paired:
        d3 = dangle_3.get(seq[i_in - 1] + seq[i_in] + seq[i_in + 1])
        if d3 is not None and d3 < 0:
            total += d3

    return total


def structure_free_energy_dimer(
    seq: str,
    seq1_len: int,
    structure: str | list[tuple[int, int]],
    material: str = "dna",
    paramset=None,
) -> float:
    """ΔG (kcal/mol) of a bimolecular duplex from the ΔG tables.

    ``structure`` is either a dot-bracket string on the concatenated sequence
    ``seq1 + seq2`` or a list of 0-based ``(i, j)`` pairs.  The structure is
    required to be a single nested helix with every pair crossing the strand
    junction at ``seq1_len``.
    """
    pairs = parse_dimer_pairs(structure, seq1_len)
    if paramset is not None:
        with param_context(paramset):
            return _sum_dimer_elements(seq, seq1_len, pairs, material, T_REF_K)
    return _sum_dimer_elements(seq, seq1_len, pairs, material, T_REF_K)


def structure_enthalpy_dimer(
    seq: str,
    seq1_len: int,
    structure: str | list[tuple[int, int]],
    material: str = "dna",
    paramset=None,
) -> float:
    """ΔH (kcal/mol) of a bimolecular duplex, via the same walk run against the
    ΔH tables.  ``paramset`` defaults to the native set for ``material``."""
    pairs = parse_dimer_pairs(structure, seq1_len)
    if paramset is None:
        from strider.thermo.parameters import load_parameters
        paramset = load_parameters("native") if material == "dna" \
            else load_parameters("native-rna")
    with param_context(_TableView(paramset.dH)):
        return _sum_dimer_elements(seq, seq1_len, pairs, material, T_REF_K)
