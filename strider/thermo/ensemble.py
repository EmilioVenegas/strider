"""
Partition function and ensemble free energy for nucleic acid secondary structures.

Single-strand: McCaskill (1990) O(n^3) DP with the SantaLucia/Turner nearest-
               neighbor energy model.
Multi-strand:  nick-aware McCaskill DP on the concatenated sequence.
               A nick at strand boundary position k prevents any hairpin loop
               from spanning that position.

References:
    McCaskill J.S. (1990). The equilibrium partition function and base pair
        binding probabilities for RNA secondary structure. Biopolymers 29:
        1105-1119.
    Dirks R.M. & Pierce N.A. (2003). A partition function algorithm for
        nucleic acid secondary structure including pseudoknots. J. Comput.
        Chem. 24: 1664-1677.
    Dirks R.M., Bois J.S., Schaeffer J.M., Winfree E., Pierce N.A. (2007).
        Thermodynamic analysis of interacting nucleic acid strands. SIAM
        Review 49: 65-88. (multi-strand partition function with nicks)
    SantaLucia J. & Hicks D. (2004). The thermodynamics of DNA structural
        motifs. Annu. Rev. Biophys. Biomol. Struct. 33: 415-440.
    Mathews D.H., Sabina J., Zuker M., Turner D.H. (1999). Expanded sequence
        dependence of thermodynamic parameters improves prediction of RNA
        secondary structure. J. Mol. Biol. 288: 911-940.
    Lu Z.J., Turner D.H., Mathews D.H. (2006). A set of nearest neighbor
        parameters for predicting the enthalpy change of RNA secondary
        structure formation. Nucleic Acids Res. 34: 4912-4924.
        (stacking-ensemble decoration model: NONE / D5 / D3 / TM at each
         external-loop helix terminus pair)
"""

from __future__ import annotations
import math
import numpy as np

from strider.thermo import parameters_dna as _pdna
from strider.thermo import parameters_rna as _prna
from strider.thermo._param_context import lookup_scalar, lookup_table

# Hoisted from the per-call ``from ... import ...`` statements that used to run
# inside the DP inner loops (Python re-executes the import statement on every
# function call — ~µs each, tens of thousands of times per fold).

R = 1.987e-3        # kcal / (mol · K)
INF = float("inf")

_MAX_IL = 30        # maximum interior loop / bulge size to enumerate


def _wc_pairs(material: str) -> set[frozenset[str]]:
    # All internal sequence handling is done in T-form (U normalized to T at the
    # entry points), so RNA pair sets are expressed in T-form with the GU wobble
    # encoded as GT/TG.  This keeps every downstream table lookup against the
    # T-keyed Turner / SantaLucia tables consistent for both materials.
    if material == "rna":
        return {frozenset("AT"), frozenset("TA"), frozenset("GC"), frozenset("CG"),
                frozenset("GT"), frozenset("TG")}
    return {frozenset("AT"), frozenset("TA"), frozenset("GC"), frozenset("CG")}


def ensemble_dg(
    sequence: str,
    celsius: float = 37.0,
    material: str = "dna",
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    blocked: "set | None" = None,
    pair_probs: bool = True,
) -> tuple[float, np.ndarray]:
    """
    Ensemble free energy and base-pair probability matrix.

    Returns:
        dG_ens (kcal/mol): ensemble free energy = -RT ln(Q)
        pair_probs (ndarray, shape (n, n)): prob that positions i-j are paired

    Salt correction: ``sodium_M`` and ``magnesium_M`` apply an Owczarzy-style
    per-base-pair ΔG shift (see :func:`strider.thermo.salt.dg_per_bp_salt`).
    Defaults to 1 M Na⁺ / 0 Mg²⁺ — the SantaLucia/Turner reference state, at
    which no correction is applied.

    ``blocked`` optionally forbids a set of 0-based positions from pairing
    (hard structural constraint; also used to compute unpaired marginals).

    ``pair_probs=False`` skips the outside recurrence
    (:func:`_external_Q_nodangle` + :func:`_pair_probs_outside`) and returns a
    zero matrix. ``dG_ens`` is byte-identical either way: the inside partition
    (``_fill_dp_nicks`` + ``_apply_coaxial_external``) fully determines
    ``Q[0][n-1]`` before the outside pass runs, and the outside pass writes
    only to local arrays. Use this when only ``dG_ens`` is needed.
    """
    # Always work in T-form internally.  RNA-specific tables (TERMINAL_PENALTY,
    # TERMINAL_MISMATCH, DANGLE_*, INTERIOR_MISMATCH, STACK) all use T-keyed
    # entries; only HAIRPIN_TRILOOP / HAIRPIN_TETRALOOP for RNA are U-keyed,
    # and we convert locally at those lookup sites.
    seq = sequence.upper().replace("U", "T")
    n = len(seq)
    T = celsius + 273.15
    pairs = _wc_pairs(material)

    Q  = np.zeros((n, n))
    Qb = np.zeros((n, n))
    QM = np.zeros((n, n))
    QM1 = np.zeros((n, n))

    for i in range(n):
        Q[i][i] = 1.0
    for i in range(n - 1):
        Q[i][i + 1] = 1.0

    from strider.thermo.salt import dg_per_bp_salt
    bp_salt_factor = _boltzmann(dg_per_bp_salt(sodium_M, magnesium_M, celsius, material), T)

    _fill_dp_nicks(seq, Q, Qb, QM, QM1, n, T, pairs, material, nicks=[],
                   bp_salt_factor=bp_salt_factor, blocked=blocked)
    _apply_coaxial_external(seq, Q, Qb, n, T, material)

    Z = Q[0][n - 1]
    if Z <= 0:
        Z = 1.0
    dG_ens = -R * T * math.log(Z)
    if not pair_probs:
        return dG_ens, np.zeros((n, n))
    # Pair probabilities use a dangle-free external loop that is exactly
    # consistent with the (dangle-free) Qb/QM/QM1 inside recurrence, so the
    # outside recurrence is self-adjoint (satisfies the unpaired-marginal
    # identity).  ΔG above keeps the full dangle/coaxial model.
    Qnd = _external_Q_nodangle(Qb, n)
    Znd = Qnd[0][n - 1] if Qnd[0][n - 1] > 0 else 1.0
    pair_probs = _pair_probs_outside(seq, Qnd, Qb, QM, QM1, n, Znd, T, pairs,
                                     material, nicks=[], blocked=blocked,
                                     bp_salt_factor=bp_salt_factor)
    return dG_ens, pair_probs


# ─── energy functions ─────────────────────────────────────────────────────────

def _hairpin_loop_energy(seq: str, i: int, j: int, material: str, T: float) -> float:
    """Hairpin loop free energy under the SantaLucia/Turner nearest-neighbor model.

    Implements the standard Mathews-Turner hairpin-loop decomposition
    (Mathews et al. 1999, J. Mol. Biol. 288:911-940; SantaLucia & Hicks 2004):
      base = hairpin_size[len(seq) - 3]  (len includes closing pair)
      triloop:  base + tp(back,front) + triloop_table[seq] + tp(front,back)
      tetraloop: base + tetraloop_table[seq] + mismatch[mm_key]  + tp(front,back)
      general:  base + mismatch[mm_key] + tp(front,back)
    where mm_key = seq[j-1]+seq[j]+seq[i]+seq[i+1]  (canonical NN-mismatch key).

    The tp(front,back) term applies SantaLucia's terminal-pair penalty at the
    external/multi-loop interface of the closing pair.
    """
    loop_size = j - i - 1
    if loop_size < 3:
        return INF

    if material == "dna":
        mismatch_const = _pdna.HAIRPIN_MISMATCH
    else:
        mismatch_const = _prna.TERMINAL_MISMATCH
    P = _pdna if material == "dna" else _prna

    HAIRPIN_SIZE = lookup_table("hairpin_size", P.HAIRPIN_SIZE)
    LOG_LOOP_PENALTY = lookup_scalar("log_loop_penalty", P.LOG_LOOP_PENALTY)
    TERMINAL_PENALTY = lookup_table("terminal_penalty", P.TERMINAL_PENALTY)
    HAIRPIN_TRILOOP = lookup_table("hairpin_triloop", P.HAIRPIN_TRILOOP)
    HAIRPIN_TETRALOOP = lookup_table("hairpin_tetraloop", P.HAIRPIN_TETRALOOP)
    # First-mismatch table follows the original DNA/RNA branch.
    if material == "dna":
        mismatch_table = lookup_table("hairpin_mismatch", mismatch_const)
    else:
        mismatch_table = lookup_table("terminal_mismatch", mismatch_const)

    # Base size: hairpin_size table is indexed by loop_size - 1.
    size_idx = loop_size - 1
    if size_idx < len(HAIRPIN_SIZE):
        dG = HAIRPIN_SIZE[size_idx]
    else:
        dG = HAIRPIN_SIZE[-1] + LOG_LOOP_PENALTY * math.log(loop_size / 30.0)

    if loop_size == 3:  # triloop
        key = seq[i:j + 1]
        if material == "rna":
            key = key.replace("T", "U")
        # Mathews-Turner: triloops apply the terminal-pair penalty on both sides.
        dG += TERMINAL_PENALTY.get(seq[j] + seq[i], 0.0)
        dG += HAIRPIN_TRILOOP.get(key, 0.0)
        dG += TERMINAL_PENALTY.get(seq[i] + seq[j], 0.0)  # external loop TP
        return dG

    if loop_size == 4:  # tetraloop: add bonus then fall through to mismatch
        key = seq[i:j + 1]
        if material == "rna":
            key = key.replace("T", "U")
        dG += HAIRPIN_TETRALOOP.get(key, 0.0)

    # Mismatch key order = seq[j-1]+seq[j]+seq[i]+seq[i+1]  (canonical NN order).
    # seq is normalized to T-form at the entry point and both RNA and DNA
    # TERMINAL_MISMATCH / HAIRPIN_MISMATCH tables are T-keyed, so no further
    # normalization is needed here.
    mm_key = seq[j - 1] + seq[j] + seq[i] + seq[i + 1]
    dG += mismatch_table.get(mm_key, 0.0)

    # Terminal penalty for closing pair (external/multi-loop context)
    dG += TERMINAL_PENALTY.get(seq[i] + seq[j], 0.0)

    return dG


def _stack_energy(seq: str, i: int, j: int, material: str) -> float:
    """
    Stacking energy for closing pair (i,j) stacked on inner pair (i+1,j-1).
    Key: seq[i] + seq[i+1] + seq[j-1] + seq[j]
        (canonical NN-stack convention, SantaLucia 1998 PNAS 95:1460-1465).
    """
    key = seq[i] + seq[i + 1] + seq[j - 1] + seq[j]
    if material == "dna":
        return lookup_table("stack", _pdna.STACK).get(key, -1.5)
    else:
        return lookup_table("stack", _prna.STACK).get(key, -2.0)


def _interior_bulge_energy(
    seq: str, i: int, j: int, ip: int, jp: int, nl: int, nr: int, material: str
) -> float:
    """
    Free energy for an interior loop or bulge.
    Outer pair (i,j), inner pair (ip,jp).
    nl = unpaired bases on left (between i and ip), nr on right (between jp and j).

    Implements the Mathews-Turner interior-loop / bulge model (Mathews et al.
    1999; SantaLucia & Hicks 2004) with a TP adjustment for strider's
    architecture, where TP(inner) is already embedded in Qb[ip][jp] via
    _hairpin_loop_energy. The universal correction is:
        dG_strider = TP_outer + dG_nn - TP_inner
    where dG_nn is the standard nearest-neighbor interior-loop energy.
    """
    P = _pdna if material == "dna" else _prna
    BULGE_SIZE = lookup_table("bulge_size", P.BULGE_SIZE)
    INTERIOR_SIZE = lookup_table("interior_size", P.INTERIOR_SIZE)
    LOG_LOOP_PENALTY = lookup_scalar("log_loop_penalty", P.LOG_LOOP_PENALTY)
    ASYMMETRY_NINIO = lookup_table("asymmetry_ninio", P.ASYMMETRY_NINIO)
    TERMINAL_PENALTY = lookup_table("terminal_penalty", P.TERMINAL_PENALTY)
    STACK = lookup_table("stack", P.STACK)
    if material == "dna":
        INTERIOR_MISMATCH = lookup_table("interior_mismatch", P.INTERIOR_MISMATCH)
        INTERIOR_1_1 = lookup_table("interior_1_1", P.INTERIOR_1_1)
        INTERIOR_1_2 = lookup_table("interior_1_2", P.INTERIOR_1_2)
        INTERIOR_2_2 = lookup_table("interior_2_2", P.INTERIOR_2_2)

    TP_outer = TERMINAL_PENALTY.get(seq[i] + seq[j], 0.0)
    TP_inner = TERMINAL_PENALTY.get(seq[ip] + seq[jp], 0.0)

    if nl == 0 or nr == 0:
        # Bulge loop: sz = total unpaired bases
        n = nl + nr
        if n <= 30:
            dG = BULGE_SIZE[n - 1]  # bulge-size table indexed by total unpaired - 1
        else:
            dG = BULGE_SIZE[-1] + LOG_LOOP_PENALTY * math.log(n / 30.0)

        if n == 1:
            # Single-base bulge: stacking across the bulge, no terminal penalty.
            # NN stack key: stack(outer_5', inner_5', inner_3', outer_3').
            stk_key = seq[i] + seq[ip] + seq[jp] + seq[j]
            dG += STACK.get(stk_key, 0.0)
            # Universal correction: TP_outer + dG_nn - TP_inner
            return dG + TP_outer - TP_inner
        else:
            # Multi-base bulge: the NN reference adds TP_outer + TP_inner inside
            # the interior energy; after our universal correction this becomes
            # 2 * TP_outer.
            return dG + 2.0 * TP_outer

    # Interior loops — exact tables for small sizes (DNA only)
    if material == "dna":
        if nl == 1 and nr == 1:
            # 1×1: key = outer_5'+left_unp+inner_5'+inner_3'+right_unp+outer_3'
            key = seq[i] + seq[i + 1] + seq[ip] + seq[jp] + seq[j - 1] + seq[j]
            val = INTERIOR_1_1.get(key)
            if val is not None:
                return TP_outer + val - TP_inner

        if nl == 1 and nr == 2:
            # 1×2: seq1=[outer_5',left_unp,inner_5'], seq2=[inner_3',r1,r2,outer_3']
            key = seq[i] + seq[i + 1] + seq[ip] + seq[jp] + seq[jp + 1] + seq[j - 1] + seq[j]
            val = INTERIOR_1_2.get(key)
            if val is not None:
                return TP_outer + val - TP_inner

        if nl == 2 and nr == 1:
            # 2×1: uses interior_1_2 table with seq2 as first argument
            key = seq[jp] + seq[j - 1] + seq[j] + seq[i] + seq[i + 1] + seq[i + 2] + seq[ip]
            val = INTERIOR_1_2.get(key)
            if val is not None:
                return TP_outer + val - TP_inner

        if nl == 2 and nr == 2:
            # 2×2: seq1=[outer_5',l1,l2,inner_5'], seq2=[inner_3',r1,r2,outer_3']
            key = seq[i] + seq[i + 1] + seq[i + 2] + seq[ip] + seq[jp] + seq[jp + 1] + seq[j - 1] + seq[j]
            val = INTERIOR_2_2.get(key)
            if val is not None:
                return TP_outer + val - TP_inner

    # General interior loop: size + Ninio asymmetry + interior mismatch
    n = nl + nr
    if n <= 30:
        dG = INTERIOR_SIZE[n - 1]  # interior-size table indexed by total unpaired - 1
    else:
        dG = INTERIOR_SIZE[-1] + LOG_LOOP_PENALTY * math.log(n / 30.0)

    # Ninio asymmetry correction
    asym = abs(nl - nr)
    ninio_number = min(min(nl, nr), 4) - 1
    dG += min(ASYMMETRY_NINIO[4], asym * ASYMMETRY_NINIO[ninio_number])

    # Interior mismatch at both closing junctions
    if material == "dna":
        if (nl == 1 and nr > 2) or (nl > 2 and nr == 1):
            # 1×large interior loop: use 'A' placeholder on the far side
            # (Mathews-Turner convention for 1×N with N > 2).
            # IM(A, seq[j], seq[i], A) and IM(A, seq[ip], seq[jp], A)
            outer_key = 'A' + seq[j] + seq[i] + 'A'
            inner_key = 'A' + seq[ip] + seq[jp] + 'A'
        else:
            # General: outer = seq[j-1]+seq[j]+seq[i]+seq[i+1]
            #          inner = seq[ip-1]+seq[ip]+seq[jp]+seq[jp+1]
            outer_key = seq[j - 1] + seq[j] + seq[i] + seq[i + 1]
            inner_key = seq[ip - 1] + seq[ip] + seq[jp] + seq[jp + 1]
        dG += INTERIOR_MISMATCH.get(outer_key, 0.0)
        dG += INTERIOR_MISMATCH.get(inner_key, 0.0)
    else:
        dG += TERMINAL_PENALTY.get(seq[i] + seq[j], 0.0)
        dG += TERMINAL_PENALTY.get(seq[ip] + seq[jp], 0.0)

    # Universal correction: TP_outer + dG_nn - TP_inner
    return dG + TP_outer - TP_inner


def _boltzmann(dG: float, T: float) -> float:
    if dG == INF:
        return 0.0
    return math.exp(-dG / (R * T))


def _can_pair(seq: str, i: int, j: int, pairs: set) -> bool:
    return frozenset([seq[i], seq[j]]) in pairs and (j - i) > 3


def _can_pair_nicks(seq: str, i: int, j: int, pairs: set, nicks: list,
                    blocked: "set | frozenset | None" = None) -> bool:
    if j <= i:
        return False
    if blocked is not None and (i in blocked or j in blocked):
        return False
    if frozenset([seq[i], seq[j]]) not in pairs:
        return False
    if any(i < k <= j for k in nicks):
        return True
    return (j - i) > 3


def _terminal_pair_penalty(seq: str, i: int, j: int, material: str) -> float:
    """Terminal base-pair penalty at a helix terminus (SantaLucia 1998)."""
    P = _pdna if material == "dna" else _prna
    return lookup_table("terminal_penalty", P.TERMINAL_PENALTY).get(seq[i] + seq[j], 0.0)


# ─── DP fill ──────────────────────────────────────────────────────────────────

def _fill_dp(seq, Q, Qb, QM, n, T, pairs, material, QM1=None):
    if QM1 is None:
        QM1 = np.zeros((n, n))
    _fill_dp_nicks(seq, Q, Qb, QM, QM1, n, T, pairs, material, nicks=[])


def _fill_dp_nicks(seq, Q, Qb, QM, QM1, n, T, pairs, material, nicks: list,
                   bp_salt_factor: float = 1.0, blocked: "set | None" = None):
    """
    Nick-aware McCaskill DP.

    Four matrices:
      Q[i][j]   — external loop partition function on [i..j]
      Qb[i][j]  — same, with i and j forced to be paired
      QM1[i][j] — multi-loop region [i..j] that is exactly ONE branch block: a
                  stem starting at i (pays ML_PAIR) plus trailing unpaired bases
                  (each pays ML_BASE)
      QM[i][j]  — multi-loop region [i..j] with ≥ 1 branch block, allowing
                  leading/internal/trailing unpaired bases

    The multiloop closure (in :func:`_qb_val`) requires ≥ 2 branch blocks, so a
    stacked/interior helix is never mis-scored as a degenerate 1-branch
    "multiloop" (which would double-count it against the stack/interior terms).

    ``bp_salt_factor`` is the per-base-pair Boltzmann factor for the salt
    correction (= exp(-ΔG_per_bp/RT)).  Multiplied into Qb[i][j] once per
    closed pair so the correction is automatically ensemble-weighted by the
    pair probability.  Defaults to 1.0 (no correction, 1 M Na⁺ reference).

    ``blocked`` is an optional set of positions forbidden from pairing (hard
    structural constraint / used to compute unpaired marginals).
    """
    P = _pdna if material == "dna" else _prna
    ML_INIT = lookup_scalar("multiloop_init", P.ML_INIT)
    ML_PAIR = lookup_scalar("multiloop_pair", P.ML_PAIR)
    ML_BASE = lookup_scalar("multiloop_base", P.ML_BASE)
    DANGLE_5 = lookup_table("dangle_5", P.DANGLE_5)
    DANGLE_3 = lookup_table("dangle_3", P.DANGLE_3)

    bm_ml_pair        = _boltzmann(ML_PAIR, T)
    bm_ml_base        = _boltzmann(ML_BASE, T)
    bm_ml_init_pair   = _boltzmann(ML_INIT + ML_PAIR, T)  # outer pair of multiloop

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            # ── Qb[i][j] ──────────────────────────────────────────────────────
            if _can_pair_nicks(seq, i, j, pairs, nicks, blocked):
                Qb[i][j] = _qb_val(
                    seq, i, j, Q, Qb, QM, QM1, T, pairs, material, nicks,
                    bm_ml_init_pair, blocked
                ) * bp_salt_factor

            # ── QM1[i][j] — region [i..j] is ONE branch block ─────────────────
            # a stem starting at i (pays ML_PAIR), then trailing unpaired bases.
            if j > i and QM1[i][j - 1] > 0:
                QM1[i][j] = QM1[i][j - 1] * bm_ml_base
            if Qb[i][j] > 0:
                QM1[i][j] += Qb[i][j] * bm_ml_pair

            # ── QM[i][j] — region [i..j] with ≥1 branch block ─────────────────
            # k = start of the first branch block; (b_base^lead | earlier
            # branches QM[i][k-1]) · QM1[k][j].  Leading unpaired is now modeled.
            qm_acc = 0.0
            for k in range(i, j):
                q1 = QM1[k][j]
                if q1 <= 0:
                    continue
                lead = bm_ml_base ** (k - i)          # unpaired i..k-1
                prev = QM[i][k - 1] if k > i else 0.0
                qm_acc += (lead + prev) * q1
            QM[i][j] = qm_acc

            # ── Q[i][j] ───────────────────────────────────────────────────────
            Q[i][j] = Q[i][j - 1]  # j unpaired
            for k in range(i, j):
                left = Q[i][k - 1] if k > i else 1.0

                # Stem (k, j) — no 5' dangle
                if Qb[k][j] > 0:
                    Q[i][j] += left * Qb[k][j]
                    # 5' dangle at k-1 on stem (k, j)
                    if k > i:
                        d5_key = seq[k] + seq[j] + seq[k - 1]
                        d5 = DANGLE_5.get(d5_key, 0.0)
                        if d5 < 0:
                            left5 = Q[i][k - 2] if k > i + 1 else 1.0
                            Q[i][j] += left5 * _boltzmann(d5, T) * Qb[k][j]

                # Stem (k, j-1) with 3' dangle at j
                if j >= k + 5 and Qb[k][j - 1] > 0:
                    d3_key = seq[j] + seq[j - 1] + seq[k]
                    d3 = DANGLE_3.get(d3_key, 0.0)
                    if d3 < 0:
                        Q[i][j] += left * Qb[k][j - 1] * _boltzmann(d3, T)
                        # Both 5' and 3' dangle
                        if k > i:
                            d5_53_key = seq[k] + seq[j - 1] + seq[k - 1]
                            d5_53 = DANGLE_5.get(d5_53_key, 0.0)
                            if d5_53 < 0:
                                left5 = Q[i][k - 2] if k > i + 1 else 1.0
                                Q[i][j] += left5 * _boltzmann(d5_53, T) * Qb[k][j - 1] * _boltzmann(d3, T)


def _apply_coaxial_external(seq: str, Q: np.ndarray, Qb: np.ndarray, n: int, T: float, material: str) -> None:
    """
    Recompute Q[0][j] under the standard "all dangles" external-loop
    decoration model (Mathews et al. 1999 §4; Lu, Turner, Mathews 2006).

    Each end of each external-loop stem decorates *independently*: an unpaired
    flanking base contributes (1 + boltz(dangle)) on its end, so a stem with
    both flanks unpaired has decoration factor
        Z_dec = (1 + boltz(d5)) * (1 + boltz(d3))
            = 1 + boltz(d5) + boltz(d3) + boltz(d5)*boltz(d3)
    The cross term boltz(d5)*boltz(d3) is exactly what STK_TM_DELTA stores
    (verified: every entry of STK_TM_DELTA equals STK_D5_DELTA * STK_D3_DELTA
    at the same key), so the multiplicative-Z form and the per-state-sum form
    are algebraically identical.

    Single-pass DP Q_stk[j] where each stem (k,j) contributes:
      NONE:  left   * STK_BARE_FACTOR[xy] * Qb[k][j]
      D5:    left5  * STK_D5_DELTA[xyn5]  * Qb[k][j]
      D3:    left   * STK_D3_DELTA[myx]   * Qb[k][j-1]   (stem is (k,j-1))
      TM:    left5  * STK_TM_DELTA[n5xym] * Qb[k][j-1]   (= D5*D3 cross term)
      COAX:  QRIGHT[k-1][b] * (bm_coax - bare_left*bare_right) * Qb[k][j]
        (flush coaxial stacking; Walter et al. 1994 PNAS 91:9218-9222)

    All STK_* values are raw Boltzmann factors (dimensionless), not ΔG.
    DNA only; RNA coaxial / TM parameters not yet wired.
    """
    if material != "dna":
        return
    from strider.thermo._param_context import lookup_table
    from strider.thermo.parameters_dna import COAXIAL_STACK, stk_decoration_tables
    # STK_* are the 37 °C stacking-ensemble decoration factors (see their provenance
    # header in parameters_dna); stk_decoration_tables re-extrapolates D5/D3 toward
    # the Bommarito-2000 dangle ΔH (TM = D5·D3) at T, returning the baked dicts
    # verbatim at 37 °C.  They are not exposed via the ParameterSet schema, so an
    # override is permitted only for COAXIAL_STACK.
    STK_BARE_FACTOR, STK_D5_DELTA, STK_D3_DELTA, STK_TM_DELTA = stk_decoration_tables(T)
    COAXIAL_STACK = lookup_table("coaxial_stack", COAXIAL_STACK)

    BASE_LIST = ['A', 'T', 'G', 'C']
    BASE_IDX  = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

    Q_stk = np.zeros(n)
    # QRIGHT[p][b]: Σ_{m: seq[m]=BASE_LIST[b]} Q_stk[m-1] * Qb[m][p]
    # Used to compute the NONE-NONE baseline for coaxial stacking corrections.
    QRIGHT = np.zeros((n, 4))

    for j in range(n):
        Q_stk[j] = Q_stk[j - 1] if j > 0 else 1.0  # j unpaired

        for k in range(j):
            left  = Q_stk[k - 1] if k > 0 else 1.0
            left5 = Q_stk[k - 2] if k > 1 else 1.0

            # Stem (k, j) — 3' face at segment end (no 3' dangle from j+1).
            if Qb[k][j] > 0:
                bare_f = STK_BARE_FACTOR.get(seq[k] + seq[j], 1.0)
                Q_stk[j] += left * bare_f * Qb[k][j]   # NONE
                if k > 0:
                    d5_delta = STK_D5_DELTA.get(seq[k] + seq[j] + seq[k - 1], 0.0)
                    if d5_delta != 0.0:
                        Q_stk[j] += left5 * d5_delta * Qb[k][j]  # D5

            # Stem (k, j-1) — position j is the 3'-dangle base.
            if j >= k + 5 and Qb[k][j - 1] > 0:
                d3_delta = STK_D3_DELTA.get(seq[j] + seq[j - 1] + seq[k], 0.0)
                if d3_delta != 0.0:
                    Q_stk[j] += left * d3_delta * Qb[k][j - 1]   # D3
                if k > 0:
                    tm_delta = STK_TM_DELTA.get(seq[k - 1] + seq[k] + seq[j - 1] + seq[j], 0.0)
                    if tm_delta != 0.0:
                        Q_stk[j] += left5 * tm_delta * Qb[k][j - 1]  # D5*D3 cross term

        # Update QRIGHT for downstream coaxial corrections.
        for m in range(j + 1):
            if Qb[m][j] > 0:
                b_idx = BASE_IDX.get(seq[m], -1)
                if b_idx >= 0:
                    q_m1 = Q_stk[m - 1] if m > 0 else 1.0
                    QRIGHT[j][b_idx] += q_m1 * Qb[m][j]

        # Flush coaxial extra: (bm_coax − bare_left·bare_right) over NONE-NONE.
        for k in range(1, j):
            if Qb[k][j] <= 0:
                continue
            bare_right = STK_BARE_FACTOR.get(seq[k] + seq[j], 1.0)
            for b_idx in range(4):
                cx_key = seq[k - 1] + seq[k] + seq[j] + BASE_LIST[b_idx]
                cx_val = COAXIAL_STACK.get(cx_key, 0.0)
                if cx_val < 0:
                    bare_left = STK_BARE_FACTOR.get(BASE_LIST[b_idx] + seq[k - 1], 1.0)
                    extra = _boltzmann(cx_val, T) - bare_left * bare_right
                    if extra != 0.0:
                        Q_stk[j] += QRIGHT[k - 1][b_idx] * extra * Qb[k][j]

    Q[0, :] = Q_stk


def _qb_val(seq, i, j, Q, Qb, QM, QM1, T, pairs, material, nicks,
            bm_ml_init_pair, blocked=None):
    """Partition function contribution for the forced pair (i,j)."""
    val = 0.0
    spans_nick = any(i < k < j for k in nicks)
    is_inter   = any(i < k <= j for k in nicks)

    # ── Hairpin (intra-strand only) ────────────────────────────────────────────
    if not spans_nick:
        dG_hp = _hairpin_loop_energy(seq, i, j, material, T)
        val += _boltzmann(dG_hp, T)

    # ── Terminal pair penalty for outermost inter-strand pair ─────────────────
    # Applied once per helix terminus (SantaLucia 1998 terminal-pair penalty).
    # The gate is a *model-shape* decision — "is the immediate inner (i+1, j-1)
    # pairable by sequence?" — so it must NOT consult ``blocked``.  ``blocked`` is
    # an external marginal constraint; letting it flip this gate would activate a
    # terminal-penalty leaf for an enclosing inter-strand pair whose inner became
    # unpairable only because of the constraint, fabricating structures that carry
    # zero weight in the true ensemble (over-counting the blocked partition near a
    # nick).  Sequence-only gating keeps the constrained partition a faithful
    # restriction of the unconstrained one, so the unpaired-marginal identity
    # holds to numerical precision at the nick junction too.
    if is_inter and not _can_pair_nicks(seq, i + 1, j - 1, pairs, nicks):
        val += _boltzmann(_terminal_pair_penalty(seq, i, j, material), T)

    # ── Stack: adjacent inner pair (i+1, j-1) ─────────────────────────────────
    if _can_pair_nicks(seq, i + 1, j - 1, pairs, nicks, blocked):
        qb_inner = Qb[i + 1][j - 1]
        if qb_inner > 0:
            val += _boltzmann(_stack_energy(seq, i, j, material), T) * qb_inner

    # ── Interior loops and bulges (one inner pair, up to MAX_IL bases each side) ──
    for nl in range(0, _MAX_IL + 1):
        ip = i + nl + 1
        if ip > j - 2:
            break
        for nr in range(0, _MAX_IL - nl + 1):
            if nl == 0 and nr == 0:
                continue  # stack: handled above
            jp = j - nr - 1
            if jp <= ip:
                break
            if not _can_pair_nicks(seq, ip, jp, pairs, nicks, blocked):
                continue
            qb_inner = Qb[ip][jp]
            if qb_inner == 0.0:
                continue
            dG = _interior_bulge_energy(seq, i, j, ip, jp, nl, nr, material)
            val += qb_inner * _boltzmann(dG, T)

    # ── Multi-loop: outer pair (i,j) closes a loop with ≥ 2 inner branches ────
    # interior [i+1, j-1] = (≥1 branch block: QM[i+1][k-1]) + (final branch
    # block: QM1[k][j-1]).  Requiring both factors enforces ≥2 branches, so a
    # single inner helix stays an interior loop / stack, never a "multiloop".
    if j - i > 2:
        ml = 0.0
        for k in range(i + 2, j):           # k = start of the last branch block
            qm_left = QM[i + 1][k - 1]
            if qm_left > 0.0:
                q1 = QM1[k][j - 1]
                if q1 > 0.0:
                    ml += qm_left * q1
        if ml > 0.0:
            val += bm_ml_init_pair * ml

    return val


def _external_Q_nodangle(Qb, n):
    """Dangle-free external-loop partition function from Qb.

    The dangle/coaxial decorations live only in the dangle-decorated ``Q`` used
    for ΔG; ``Qb``/``QM``/``QM1`` are already dangle-free.  Recomputing the
    external loop without dangles gives an external context that is *exactly*
    consistent with the inside ``Qb`` recurrence, so the outside recurrence built
    on it satisfies the unpaired-marginal identity to numerical precision.
    """
    Qnd = np.zeros((n, n))
    for i in range(n):
        Qnd[i][i] = 1.0
    for i in range(n - 1):
        Qnd[i][i + 1] = 1.0
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            val = Qnd[i][j - 1]                       # j unpaired
            for k in range(i, j + 1):
                if Qb[k][j] > 0:
                    left = Qnd[i][k - 1] if k > i else 1.0
                    val += left * Qb[k][j]
            Qnd[i][j] = val
    return Qnd


def _pairs_dp(seq, T, material, pairs, nicks, bp_salt_factor, blocked):
    """Fill the dangle-free DP used for pair probabilities; return (Qnd, Qb, QM, QM1, Znd)."""
    n = len(seq)
    Q  = np.zeros((n, n))
    Qb = np.zeros((n, n))
    QM = np.zeros((n, n))
    QM1 = np.zeros((n, n))
    for i in range(n):
        Q[i][i] = 1.0
    for i in range(n - 1):
        Q[i][i + 1] = 1.0
    _fill_dp_nicks(seq, Q, Qb, QM, QM1, n, T, pairs, material, nicks,
                   bp_salt_factor=bp_salt_factor, blocked=blocked)
    Qnd = _external_Q_nodangle(Qb, n)
    Znd = Qnd[0][n - 1] if n > 0 and Qnd[0][n - 1] > 0 else 1.0
    return Qnd, Qb, QM, QM1, Znd


def dangle_free_partition(sequence=None, celsius=37.0, material="dna",
                          sodium_M=1.0, magnesium_M=0.0, blocked=None,
                          sequences=None):
    """Dangle-free partition function Z (single- or multi-strand).

    Exposed for validation (unpaired marginals) and as the consistent reference
    behind :func:`ensemble_dg`'s pair probabilities.
    """
    from strider.thermo.salt import dg_per_bp_salt
    if sequences is not None:
        seq = "".join(sequences).upper().replace("U", "T")
        nicks, pos = [], 0
        for s in sequences[:-1]:
            pos += len(s); nicks.append(pos)
    else:
        seq = sequence.upper().replace("U", "T")
        nicks = []
    T = celsius + 273.15
    pairs = _wc_pairs(material)
    bp = _boltzmann(dg_per_bp_salt(sodium_M, magnesium_M, celsius, material), T)
    _, _, _, _, Znd = _pairs_dp(seq, T, material, pairs, nicks, bp, blocked)
    return Znd


def _pair_probs(Q, Qb, n, Z):
    """External-context-only pair probabilities (kept for backward compatibility)."""
    probs = np.zeros((n, n))
    if Z == 0:
        return probs
    for i in range(n):
        for j in range(i + 4, n):
            if Qb[i][j] > 0:
                left  = Q[0][i - 1] if i > 0 else 1.0
                right = Q[j + 1][n - 1] if j < n - 1 else 1.0
                probs[i][j] = (left * Qb[i][j] * right) / Z
                probs[j][i] = probs[i][j]
    return probs


def _pair_probs_outside(
    seq: str,
    Q: np.ndarray,
    Qb: np.ndarray,
    QM: np.ndarray,
    QM1: np.ndarray,
    n: int,
    Z: float,
    T: float,
    pairs: set,
    material: str,
    nicks: list,
    blocked: "set | None" = None,
    bp_salt_factor: float = 1.0,
) -> np.ndarray:
    """
    Full pair probabilities via the McCaskill outside recurrence.

    Implements the external context, stack propagation, interior-loop / bulge
    propagation, **and** the multiloop outside contribution — so pairs inside a
    multiloop are no longer underestimated.  The result is the exact adjoint of the
    inside recurrence — for single-strand **and** multi-strand complexes alike — and
    satisfies the unpaired-marginal identity Σ_j P(i,j) = 1 − P_unpaired(i) to
    numerical precision at every position, including the immediate nick-junction
    pair (i, i+1) straddling a strand boundary (validated in
    tests/test_pair_probs.py against an exact brute-force enumeration of the model).
    """
    Pp = np.zeros((n, n))
    if Z <= 0:
        return Pp

    # Multiloop Boltzmann factors (mirror _fill_dp_nicks).
    from strider.thermo._param_context import lookup_scalar
    if material == "dna":
        from strider.thermo.parameters_dna import ML_INIT, ML_PAIR, ML_BASE
    else:
        from strider.thermo.parameters_rna import ML_INIT, ML_PAIR, ML_BASE
    bm_ml_pair      = _boltzmann(lookup_scalar("multiloop_pair", ML_PAIR), T)
    bm_ml_base      = _boltzmann(lookup_scalar("multiloop_base", ML_BASE), T)
    bm_ml_init_pair = _boltzmann(lookup_scalar("multiloop_init", ML_INIT)
                                 + lookup_scalar("multiloop_pair", ML_PAIR), T)

    # 1. External context: (i,j) is an outermost stem in the external loop.
    #    Loop from i+1 (not i+4): an inter-strand pair spanning a nick may be
    #    short (j-i<4).  Qb>0 already excludes invalid intra-strand short pairs.
    for i in range(n):
        for j in range(i + 1, n):
            if Qb[i][j] > 0:
                left  = Q[0][i - 1] if i > 0 else 1.0
                right = Q[j + 1][n - 1] if j < n - 1 else 1.0
                Pp[i][j] += left * Qb[i][j] * right / Z

    # 2. Enclosed contributions: iterate outer pair (I, J) from largest to
    #    smallest.  Down to length 2: an inter-strand pair spanning a nick can be
    #    short and still enclose another (coaxial) pair across the junction.
    for length in range(n, 1, -1):
        for I in range(n - length + 1):
            J = I + length - 1
            if Qb[I][J] <= 0 or Pp[I][J] <= 0:
                continue
            # Every inside contribution to Qb[I][J] carries one bp_salt_factor
            # (applied to the whole _qb_val), so the outside must re-apply it when
            # distributing the (I,J) probability to the pairs it encloses.
            base = (Pp[I][J] / Qb[I][J]) * bp_salt_factor

            # 2a. Stack: inner pair (I+1, J-1)
            i, j = I + 1, J - 1
            if j > i and _can_pair_nicks(seq, i, j, pairs, nicks) and Qb[i][j] > 0:
                stack_e = _stack_energy(seq, I, J, material)
                Pp[i][j] += base * Qb[i][j] * _boltzmann(stack_e, T)

            # 2b. Interior loop / bulge: inner pair (ip, jp) with (nl, nr) unpaired
            for nl in range(_MAX_IL + 1):
                ip = I + nl + 1
                if ip > J - 2:
                    break
                for nr in range(_MAX_IL - nl + 1):
                    if nl == 0 and nr == 0:
                        continue  # stack handled above
                    jp = J - nr - 1
                    if jp <= ip:
                        break
                    if not _can_pair_nicks(seq, ip, jp, pairs, nicks):
                        continue
                    if Qb[ip][jp] <= 0:
                        continue
                    il_e = _interior_bulge_energy(seq, I, J, ip, jp, nl, nr, material)
                    Pp[ip][jp] += base * Qb[ip][jp] * _boltzmann(il_e, T)

            # 2c. Multiloop: (I,J) closes a multiloop; (i,j) is one inner branch.
            #     left/right over [I+1,i-1] / [j+1,J-1] are each (all-unpaired) or
            #     (≥1 branch: QM); subtracting the both-unpaired product enforces
            #     that (i,j) is never the sole branch (≥2 branches required).
            ml_coeff = base * bm_ml_init_pair * bm_ml_pair
            if ml_coeff > 0.0:
                for i in range(I + 1, J):
                    lead_left = bm_ml_base ** (i - 1 - I)         # I+1..i-1 unpaired
                    qm_left = QM[I + 1][i - 1] if (i - 1) >= (I + 1) else 0.0
                    for j in range(i + 1, J):
                        if Qb[i][j] <= 0.0:
                            continue
                        lead_right = bm_ml_base ** (J - 1 - j)    # j+1..J-1 unpaired
                        qm_right = QM[j + 1][J - 1] if (J - 1) >= (j + 1) else 0.0
                        # ≥2 branches ⇒ at least one of the left/right regions has
                        # a branch.  Expanded (cancellation-free) form of
                        #   (lead_left+qm_left)(lead_right+qm_right) − lead_left·lead_right
                        valid = (lead_left * qm_right + qm_left * lead_right
                                 + qm_left * qm_right)
                        if valid > 0.0:
                            Pp[i][j] += ml_coeff * Qb[i][j] * valid

    # Symmetrize
    for i in range(n):
        for j in range(i + 1, n):
            if Pp[i][j] > 0:
                Pp[j][i] = Pp[i][j]
    return Pp


# ─── multi-strand ─────────────────────────────────────────────────────────────

def multistrand_pairs(
    sequences: list[str],
    celsius: float = 37.0,
    material: str = "dna",
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    pair_probs: bool = True,
) -> tuple[float, np.ndarray]:
    """
    Ensemble free energy and pair-probability matrix for a multi-strand complex.

    Same DP as :func:`_multistrand_dg` but also returns the pair probabilities
    over the concatenated sequence.  Strand boundaries are tracked internally as
    nicks so no hairpin can span a junction.

    ``pair_probs=False`` skips the outside recurrence and returns a zero matrix;
    ``dG`` is byte-identical (see :func:`ensemble_dg`).
    """
    seq = "".join(sequences).upper().replace("U", "T")
    n = len(seq)
    if n == 0:
        return 0.0, np.zeros((0, 0))
    T = celsius + 273.15
    pairs = _wc_pairs(material)

    nicks: list[int] = []
    pos = 0
    for s in sequences[:-1]:
        pos += len(s)
        nicks.append(pos)

    Q  = np.zeros((n, n))
    Qb = np.zeros((n, n))
    QM = np.zeros((n, n))
    QM1 = np.zeros((n, n))
    for i in range(n):
        Q[i][i] = 1.0
    for i in range(n - 1):
        Q[i][i + 1] = 1.0

    from strider.thermo.salt import dg_per_bp_salt
    bp_salt_factor = _boltzmann(dg_per_bp_salt(sodium_M, magnesium_M, celsius, material), T)

    _fill_dp_nicks(seq, Q, Qb, QM, QM1, n, T, pairs, material, nicks,
                   bp_salt_factor=bp_salt_factor)
    _apply_coaxial_external(seq, Q, Qb, n, T, material)

    Z = Q[0][n - 1]
    if Z <= 0:
        Z = 1.0
    dG = -R * T * math.log(Z)
    if not pair_probs:
        return dG, np.zeros((n, n))
    Qnd = _external_Q_nodangle(Qb, n)
    Znd = Qnd[0][n - 1] if Qnd[0][n - 1] > 0 else 1.0
    probs = _pair_probs_outside(seq, Qnd, Qb, QM, QM1, n, Znd, T, pairs, material,
                                nicks, bp_salt_factor=bp_salt_factor)
    return dG, probs


def bimolecular_dg(
    seq1: str,
    seq2: str,
    celsius: float = 37.0,
    material: str = "dna",
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
) -> float:
    """Binding ΔG° for a two-strand complex: ΔG(complex) − ΔG(seq1) − ΔG(seq2)."""
    dG1, _ = ensemble_dg(seq1, celsius, material, sodium_M, magnesium_M)
    dG2, _ = ensemble_dg(seq2, celsius, material, sodium_M, magnesium_M)
    dG_complex = _multistrand_dg([seq1, seq2], celsius, material, sodium_M, magnesium_M)
    return dG_complex - dG1 - dG2


def _multistrand_dg(
    sequences: list[str],
    celsius: float,
    material: str,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
) -> float:
    """
    Ensemble ΔG for the concatenated multi-strand complex (kcal/mol).
    Includes the bimolecular JOIN_PENALTY for each inter-strand association.
    """
    seq = "".join(sequences).upper().replace("U", "T")
    n = len(seq)
    if n == 0:
        return 0.0
    T = celsius + 273.15
    pairs = _wc_pairs(material)

    nicks: list[int] = []
    pos = 0
    for s in sequences[:-1]:
        pos += len(s)
        nicks.append(pos)

    Q  = np.zeros((n, n))
    Qb = np.zeros((n, n))
    QM = np.zeros((n, n))
    QM1 = np.zeros((n, n))
    for i in range(n):
        Q[i][i] = 1.0
    for i in range(n - 1):
        Q[i][i + 1] = 1.0

    from strider.thermo.salt import dg_per_bp_salt
    bp_salt_factor = _boltzmann(dg_per_bp_salt(sodium_M, magnesium_M, celsius, material), T)

    _fill_dp_nicks(seq, Q, Qb, QM, QM1, n, T, pairs, material, nicks,
                   bp_salt_factor=bp_salt_factor)
    _apply_coaxial_external(seq, Q, Qb, n, T, material)

    Z = Q[0][n - 1]
    if Z <= 0:
        Z = 1.0
    return -R * T * math.log(Z)
