"""
Minimum free energy (MFE) secondary structure prediction.

Implements the Zuker–Stiegler dynamic program with the full nearest-neighbor
energy decomposition (hairpin, stack, internal loop, bulge, multi-branch
loop) shared with the McCaskill partition-function DP in
:mod:`strider.thermo.ensemble`.  Both engines consume the same loop-energy
functions, so MFE energies and ensemble ΔG are mutually consistent.

References
----------
Zuker M. & Stiegler P. (1981) Nucleic Acids Res. 9: 133-148.
SantaLucia J. & Hicks D. (2004) Annu. Rev. Biophys. Biomol. Struct. 33: 415-440.
Mathews D.H., Sabina J., Zuker M., Turner D.H. (1999) J. Mol. Biol. 288: 911-940.
Lu Z.J., Turner D.H., Mathews D.H. (2006) Nucleic Acids Res. 34: 4912-4924.
"""

from __future__ import annotations
import numpy as np

INF = float("inf")
MIN_HAIRPIN_LOOP = 3   # minimum unpaired bases inside a hairpin loop
_MAX_IL = 30           # maximum internal-loop / bulge size (matches ensemble.py)


def fold_mfe(
    sequence: str,
    celsius: float = 37.0,
    material: str = "dna",
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
) -> tuple[str, float, list[tuple[int, int]]]:
    """
    Predict the MFE secondary structure for a single strand or strand complex.

    Multi-strand input may be given with ``'&'`` or ``'+'`` strand separators
    inside ``sequence`` (e.g. ``"AAAA&TTTT"``).  The dynamic program runs on
    the concatenated sequence (separators removed); ``'&'`` is re-inserted in
    the returned dot-bracket string.  Hairpin and internal-loop moves whose
    enclosed unpaired region would span a strand break are forbidden, while
    inter-strand base pairs and exterior-loop decompositions at breaks are
    allowed with no backbone-adjacency penalty.

    Parameters
    ----------
    sequence : nucleotide sequence (case-insensitive, U/T interchangeable).
               Use ``'&'`` or ``'+'`` between strands for multi-strand folding.
    celsius  : temperature in °C (default 37).
    material : ``"dna"`` or ``"rna"``.
    sodium_M : [Na⁺] in molar (default 1.0 = SantaLucia/Turner reference).
    magnesium_M : [Mg²⁺] in molar (default 0.0).

    Salt correction: each closed base pair contributes the per-base-pair ΔG
    shift of :func:`strider.thermo.salt.dg_per_bp_salt` — the same term the
    McCaskill ensemble DP folds into ``Qb`` — so MFE energies track [Na⁺]/[Mg²⁺]
    consistently with :func:`strider.thermo.ensemble.ensemble_dg`.  At 1 M Na⁺,
    0 Mg²⁺ the correction is exactly 0 and the result is unchanged.

    Returns
    -------
    structure : dot-bracket string with ``'&'`` separators preserved.
    energy    : MFE in kcal/mol (negative = stable).
    pairs     : list of ``(i, j)`` 0-indexed base-pair positions over the
                concatenated sequence (without separators).
    """
    raw_seq, nicks, sep_char = _parse_strands(sequence, material)
    seq = _normalize(raw_seq, material)
    n = len(seq)
    if n == 0:
        return _insert_separators("", nicks, sep_char), 0.0, []

    T = celsius + 273.15
    # Per-closed-base-pair salt ΔG (0 at the 1 M Na⁺ reference).  Added to V[i,j]
    # once per pair, mirroring the bp_salt_factor applied once per Qb pair in the
    # ensemble DP; recursion makes the total scale with the number of pairs.
    from strider.thermo.salt import dg_per_bp_salt
    dg_salt = dg_per_bp_salt(sodium_M, magnesium_M, celsius, material)
    V, W, WM, WM1, energy_fns = _build_mfe_matrices(seq, T, material, nicks, dg_salt)
    (can, spans, inter, hairpin_e, stack_e, il_e, terminal_e,
     ml_a, ml_b, ml_c) = energy_fns

    # Traceback
    pairs: list[tuple[int, int]] = []
    _traceback_W(W, V, WM, WM1, seq, 0, n - 1, T, pairs,
                 hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
    pairs.sort()

    energy = float(W[0][n - 1]) if n > 1 else 0.0
    structure = _to_dot_bracket(pairs, n, nicks, sep_char)
    return structure, energy, pairs


def _build_mfe_matrices(seq, T, material, nicks, dg_salt):
    """Fill and return the Zuker–Stiegler DP matrices for ``seq``.

    Returns ``(V, W, WM, WM1, energy_fns)`` where ``energy_fns`` is the tuple
    ``(can, spans, inter, hairpin_e, stack_e, il_e, terminal_e, ml_a, ml_b,
    ml_c)``.  Factored out of :func:`fold_mfe` so the suboptimal enumerator in
    :mod:`strider.structure.sampling` shares the exact same matrices and energy
    closures — guaranteeing that the subopt MFE equals :func:`fold_mfe`.
    """
    n = len(seq)
    pairs_set = _wc_pairs(material)
    ml_a, ml_b, ml_c = _multiloop_params(material)

    # DP tables
    V   = np.full((n, n), INF)   # V[i,j]: min energy with (i,j) paired
    W   = np.zeros((n, n))       # W[i,j]: min energy on [i..j], any topology
    WM  = np.full((n, n), INF)   # WM[i,j]: multi-loop fragment, ≥1 branch
    WM1 = np.full((n, n), INF)   # WM1[i,j]: multi-loop fragment, branch ends at j

    can = lambda i, j: _can_pair_nicks(seq, i, j, pairs_set, nicks)
    spans = lambda i, j: _spans_nick(i, j, nicks)
    inter = lambda i, j: _is_inter_strand(i, j, nicks)
    hairpin_e = _hairpin_energy_fn(material)
    stack_e   = _stack_energy_fn(material)
    il_e      = _internal_bulge_energy_fn(material)
    terminal_e = _terminal_pair_penalty_fn(material)

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            # V[i,j]: structures with (i,j) paired
            if can(i, j):
                v_best = INF

                if not spans(i, j):
                    v_best = hairpin_e(seq, i, j, T)

                if i + 1 < j - 1 and can(i + 1, j - 1):
                    cand = stack_e(seq, i, j) + V[i + 1][j - 1]
                    if cand < v_best:
                        v_best = cand

                if inter(i, j) and not can(i + 1, j - 1):
                    cand = terminal_e(seq, i, j)
                    if cand < v_best:
                        v_best = cand

                max_ip = min(i + _MAX_IL + 1, j - MIN_HAIRPIN_LOOP - 2)
                for ip in range(i + 1, max_ip + 1):
                    min_jp = max(ip + MIN_HAIRPIN_LOOP + 1, j - _MAX_IL - 1)
                    for jp in range(min_jp, j):
                        if ip == i + 1 and jp == j - 1:
                            continue  # covered by stack case
                        nl = ip - i - 1
                        nr = j - jp - 1
                        if nl + nr == 0 or nl + nr > _MAX_IL:
                            continue
                        if not can(ip, jp):
                            continue
                        if spans(i + 1, ip - 1) or spans(jp + 1, j - 1):
                            continue
                        cand = il_e(seq, i, j, ip, jp, nl, nr) + V[ip][jp]
                        if cand < v_best:
                            v_best = cand

                # Multi-loop: (i,j) closes a multi-branch loop with ≥2 stems inside
                for k in range(i + 2, j - 1):
                    if WM[i + 1][k] < INF and WM1[k + 1][j - 1] < INF:
                        cand = ml_a + ml_b + WM[i + 1][k] + WM1[k + 1][j - 1]
                        if cand < v_best:
                            v_best = cand

                V[i][j] = v_best + dg_salt

            # WM1[i,j]: multi-loop fragment containing exactly one branch ending at j
            wm1_best = INF
            if V[i][j] < INF:
                wm1_best = V[i][j] + ml_b
            if j > i and WM1[i][j - 1] < INF:
                cand = WM1[i][j - 1] + ml_c
                if cand < wm1_best:
                    wm1_best = cand
            WM1[i][j] = wm1_best

            # WM[i,j]: multi-loop fragment with ≥1 branch
            wm_best = WM1[i][j]
            if j > i and WM[i][j - 1] < INF:
                cand = WM[i][j - 1] + ml_c
                if cand < wm_best:
                    wm_best = cand
            for k in range(i, j):
                if WM[i][k] < INF and WM1[k + 1][j] < INF:
                    cand = WM[i][k] + WM1[k + 1][j]
                    if cand < wm_best:
                        wm_best = cand
            WM[i][j] = wm_best

            # W[i,j]: any-topology min energy on [i..j]
            w_best = W[i][j - 1] if j > i else 0.0   # j unpaired
            if V[i][j] < w_best:
                w_best = V[i][j]
            for k in range(i, j):
                left = W[i][k] if k > i else 0.0
                if V[k + 1][j] < INF:
                    cand = left + V[k + 1][j]
                    if cand < w_best:
                        w_best = cand
                if (k + 1) in nicks and W[k + 1][j] < w_best:
                    cand = left + W[k + 1][j]
                    if cand < w_best:
                        w_best = cand
            W[i][j] = w_best

    energy_fns = (can, spans, inter, hairpin_e, stack_e, il_e, terminal_e,
                  ml_a, ml_b, ml_c)
    return V, W, WM, WM1, energy_fns


# ─── traceback ────────────────────────────────────────────────────────────────

def _traceback_W(W, V, WM, WM1, seq, i, j, T, out, hairpin_e, stack_e, il_e,
                 terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt):
    """Recover the base-pair list achieving W[i..j] by following the recurrences."""
    if j <= i:
        return
    target = W[i][j]

    left_w = W[i][j - 1] if j > i else 0.0
    if abs(target - left_w) < 1e-9:
        _traceback_W(W, V, WM, WM1, seq, i, j - 1, T, out,
                     hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
        return

    if abs(target - V[i][j]) < 1e-9:
        out.append((i, j))
        _traceback_V(V, WM, WM1, seq, i, j, T, out,
                     hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
        return

    for k in range(i, j):
        left = W[i][k] if k > i else 0.0
        if V[k + 1][j] < INF and abs(target - (left + V[k + 1][j])) < 1e-9:
            if k > i:
                _traceback_W(W, V, WM, WM1, seq, i, k, T, out,
                             hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
            out.append((k + 1, j))
            _traceback_V(V, WM, WM1, seq, k + 1, j, T, out,
                         hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
            return
        if (k + 1) in nicks and W[k + 1][j] < INF \
                and abs(target - (left + W[k + 1][j])) < 1e-9:
            if k > i:
                _traceback_W(W, V, WM, WM1, seq, i, k, T, out,
                             hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
            _traceback_W(W, V, WM, WM1, seq, k + 1, j, T, out,
                         hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
            return


def _traceback_V(V, WM, WM1, seq, i, j, T, out, hairpin_e, stack_e, il_e,
                 terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt):
    """Recover the base-pair decomposition of V[i,j] (forced pair at (i,j))."""
    # V[i,j] carries one ``dg_salt`` for the (i,j) pair itself; strip it so the
    # candidate energies below (which reference inner V values that already
    # include their own salt terms) compare exactly.
    target = V[i][j] - dg_salt

    if not spans(i, j) and abs(target - hairpin_e(seq, i, j, T)) < 1e-9:
        return

    if inter(i, j) and not can(i + 1, j - 1):
        if abs(target - terminal_e(seq, i, j)) < 1e-9:
            return

    if i + 1 < j - 1 and can(i + 1, j - 1):
        cand = stack_e(seq, i, j) + V[i + 1][j - 1]
        if abs(target - cand) < 1e-9:
            out.append((i + 1, j - 1))
            _traceback_V(V, WM, WM1, seq, i + 1, j - 1, T, out,
                         hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
            return

    max_ip = min(i + _MAX_IL + 1, j - MIN_HAIRPIN_LOOP - 2)
    for ip in range(i + 1, max_ip + 1):
        min_jp = max(ip + MIN_HAIRPIN_LOOP + 1, j - _MAX_IL - 1)
        for jp in range(min_jp, j):
            if ip == i + 1 and jp == j - 1:
                continue
            nl = ip - i - 1
            nr = j - jp - 1
            if nl + nr == 0 or nl + nr > _MAX_IL:
                continue
            if not can(ip, jp):
                continue
            if spans(i + 1, ip - 1) or spans(jp + 1, j - 1):
                continue
            cand = il_e(seq, i, j, ip, jp, nl, nr) + V[ip][jp]
            if abs(target - cand) < 1e-9:
                out.append((ip, jp))
                _traceback_V(V, WM, WM1, seq, ip, jp, T, out,
                             hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
                return

    for k in range(i + 2, j - 1):
        if WM[i + 1][k] < INF and WM1[k + 1][j - 1] < INF:
            cand = ml_a + ml_b + WM[i + 1][k] + WM1[k + 1][j - 1]
            if abs(target - cand) < 1e-9:
                _traceback_WM(V, WM, WM1, seq, i + 1, k, T, out,
                              hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
                _traceback_WM1(V, WM, WM1, seq, k + 1, j - 1, T, out,
                               hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
                return


def _traceback_WM1(V, WM, WM1, seq, i, j, T, out, hairpin_e, stack_e, il_e,
                   terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt):
    """Recover the single-branch multi-loop fragment WM1[i,j]."""
    target = WM1[i][j]
    if V[i][j] < INF and abs(target - (V[i][j] + ml_b)) < 1e-9:
        out.append((i, j))
        _traceback_V(V, WM, WM1, seq, i, j, T, out,
                     hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
        return
    if j > i and WM1[i][j - 1] < INF:
        if abs(target - (WM1[i][j - 1] + ml_c)) < 1e-9:
            _traceback_WM1(V, WM, WM1, seq, i, j - 1, T, out,
                           hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
            return


def _traceback_WM(V, WM, WM1, seq, i, j, T, out, hairpin_e, stack_e, il_e,
                  terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt):
    """Recover the multi-loop fragment WM[i,j] (≥1 branch)."""
    target = WM[i][j]
    if abs(target - WM1[i][j]) < 1e-9:
        _traceback_WM1(V, WM, WM1, seq, i, j, T, out,
                       hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
        return
    if j > i and WM[i][j - 1] < INF and abs(target - (WM[i][j - 1] + ml_c)) < 1e-9:
        _traceback_WM(V, WM, WM1, seq, i, j - 1, T, out,
                      hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
        return
    for k in range(i, j):
        if WM[i][k] < INF and WM1[k + 1][j] < INF:
            cand = WM[i][k] + WM1[k + 1][j]
            if abs(target - cand) < 1e-9:
                _traceback_WM(V, WM, WM1, seq, i, k, T, out,
                              hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
                _traceback_WM1(V, WM, WM1, seq, k + 1, j, T, out,
                               hairpin_e, stack_e, il_e, terminal_e, can, spans, inter, nicks, ml_a, ml_b, ml_c, dg_salt)
                return


# ─── helpers ──────────────────────────────────────────────────────────────────

def _wc_pairs(material: str) -> set[frozenset[str]]:
    """Watson–Crick (plus G·U wobble for RNA) allowed base pairs."""
    if material == "rna":
        return {frozenset("AU"), frozenset("UA"), frozenset("GC"), frozenset("CG"),
                frozenset("GU"), frozenset("UG")}
    return {frozenset("AT"), frozenset("TA"), frozenset("GC"), frozenset("CG")}


def _parse_strands(sequence: str, material: str) -> tuple[str, list[int], str]:
    """
    Split ``'&'`` or ``'+'`` separated strands.

    Returns ``(raw_concatenated, nicks, sep_char)`` where ``nicks`` are the
    positions in the concatenated sequence immediately after each strand,
    matching the convention used by :func:`strider.thermo.ensemble.multistrand_pairs`.
    """
    if "&" in sequence and "+" in sequence:
        raise ValueError("sequence cannot mix '&' and '+' strand separators")
    if "&" in sequence:
        parts = sequence.split("&")
        sep_char = "&"
    elif "+" in sequence:
        parts = sequence.split("+")
        sep_char = "+"
    else:
        return sequence, [], "&"
    if any(len(p) == 0 for p in parts):
        raise ValueError("empty strand in multi-strand input")
    nicks: list[int] = []
    pos = 0
    for p in parts[:-1]:
        pos += len(p)
        nicks.append(pos)
    return "".join(parts), nicks, sep_char


def _can_pair_nicks(seq: str, i: int, j: int, pairs: set, nicks: list[int]) -> bool:
    """True if ``(i,j)`` is pairable, allowing inter-strand pairs across nicks."""
    if j <= i:
        return False
    if frozenset([seq[i], seq[j]]) not in pairs:
        return False
    if any(i < k <= j for k in nicks):
        return True
    return (j - i) > MIN_HAIRPIN_LOOP


def _spans_nick(i: int, j: int, nicks: list[int]) -> bool:
    """True if the open interval ``(i, j)`` contains a nick position."""
    return any(i < k < j for k in nicks)


def _is_inter_strand(i: int, j: int, nicks: list[int]) -> bool:
    """True if positions ``i`` and ``j`` lie on opposite sides of a nick."""
    return any(i < k <= j for k in nicks)


def _normalize(seq: str, material: str) -> str:
    """Uppercase and unify T/U letters for the chosen ``material`` alphabet."""
    seq = seq.upper()
    if material == "dna":
        return seq.replace("U", "T")
    return seq.replace("T", "U")


def _insert_separators(structure: str, nicks: list[int], sep_char: str) -> str:
    """Insert strand separators into a dot-bracket string at nick positions."""
    if not nicks:
        return structure
    chars = list(structure)
    for pos in reversed(nicks):
        chars.insert(pos, sep_char)
    return "".join(chars)


def _to_dot_bracket(
    pairs: list[tuple[int, int]],
    n: int,
    nicks: list[int],
    sep_char: str,
) -> str:
    """Render a sorted pair list as a dot-bracket string with strand separators."""
    db = ["."] * n
    for i, j in pairs:
        db[i] = "("
        db[j] = ")"
    return _insert_separators("".join(db), nicks, sep_char)


# ─── shared energy adapters ───────────────────────────────────────────────────
#
# These thin wrappers route MFE energy lookups through the same functions that
# `strider.thermo.ensemble` uses for the partition function.  Keeping a single
# energy source means MFE energies and Boltzmann-weighted ensemble ΔG cannot
# drift apart.

def _hairpin_energy_fn(material: str):
    """Return ``hairpin(seq, i, j, T)`` reading from the canonical loop tables."""
    from strider.thermo.ensemble import _hairpin_loop_energy

    def hairpin(seq: str, i: int, j: int, T: float) -> float:
        return _hairpin_loop_energy(seq, i, j, material, T)
    return hairpin


def _stack_energy_fn(material: str):
    """Return ``stack(seq, i, j)`` for the closing pair (i,j) over inner pair (i+1,j-1)."""
    from strider.thermo.ensemble import _stack_energy

    def stack(seq: str, i: int, j: int) -> float:
        return _stack_energy(seq, i, j, material)
    return stack


def _internal_bulge_energy_fn(material: str):
    """Return ``il(seq, i, j, ip, jp, nl, nr)`` for internal-loop / bulge contributions."""
    from strider.thermo.ensemble import _interior_bulge_energy

    def il(seq, i, j, ip, jp, nl, nr):
        return _interior_bulge_energy(seq, i, j, ip, jp, nl, nr, material)
    return il


def _terminal_pair_penalty_fn(material: str):
    """Return ``terminal(seq, i, j)`` for an inter-strand helix terminus."""
    from strider.thermo.ensemble import _terminal_pair_penalty

    def terminal(seq: str, i: int, j: int) -> float:
        return _terminal_pair_penalty(seq, i, j, material)
    return terminal


def _multiloop_params(material: str) -> tuple[float, float, float]:
    """Multi-loop linear coefficients (a, b, c): a + b·branches + c·unpaired."""
    from strider.thermo._param_context import lookup_scalar
    if material == "dna":
        from strider.thermo.parameters_dna import ML_INIT, ML_PAIR, ML_BASE
    else:
        from strider.thermo.parameters_rna import ML_INIT, ML_PAIR, ML_BASE
    return (
        lookup_scalar("multiloop_init", float(ML_INIT)),
        lookup_scalar("multiloop_pair", float(ML_PAIR)),
        lookup_scalar("multiloop_base", float(ML_BASE)),
    )


# ─── backward-compatible private API ──────────────────────────────────────────
# These are imported by strider.structure.pseudoknot.  Names preserved.

def _can_pair_fn(material: str):
    """Return ``can(seq, i, j) -> bool`` using the material's WC (+ wobble) pair set."""
    pairs = _wc_pairs(material)

    def can(seq: str, i: int, j: int) -> bool:
        return _can_pair_nicks(seq, i, j, pairs, [])
    return can


def _stack_fn(material: str):
    """Return ``stack(seq, i, j) -> float`` — same function used by the MFE DP."""
    return _stack_energy_fn(material)


def _hairpin_energy(seq: str, i: int, j: int, celsius: float, material: str) -> float:
    """Hairpin loop ΔG (kcal/mol).  Same function used by the MFE DP."""
    from strider.thermo.ensemble import _hairpin_loop_energy
    return _hairpin_loop_energy(seq, i, j, material, celsius + 273.15)
