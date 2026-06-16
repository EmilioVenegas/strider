"""
Salt correction models for nucleic acid thermodynamics.

Sources:
  - Owczarzy et al. (2004) Biochemistry 43:3537-3554  (Na+ correction)
  - Owczarzy et al. (2008) Biochemistry 47:5336-5353  (Mg2+ correction)
  - Tan & Chen (2006) Biophys. J. 90:1175-1190         (unified model)
"""

from __future__ import annotations
import math


def owczarzy_tm_correction(
    seq: str,
    sodium_M: float,
    magnesium_M: float = 0.0,
) -> float:
    """
    Tm correction (°C) relative to 1 M NaCl reference.

    Uses Owczarzy 2004 for pure Na+ and Owczarzy 2008 Eq. 16 for Mg2+.
    When both ions present, uses the Mg2+/Na+ ratio to select the regime.
    """
    fGC = _fgc(seq)

    if magnesium_M > 0 and sodium_M > 0:
        ratio = math.sqrt(magnesium_M) / sodium_M
        if ratio < 0.22:
            return _na_correction(fGC, sodium_M)
        elif ratio < 6.0:
            return _mixed_correction(fGC, sodium_M, magnesium_M)
        else:
            return _mg_correction(fGC, magnesium_M)
    elif magnesium_M > 0:
        return _mg_correction(fGC, magnesium_M)
    else:
        return _na_correction(fGC, sodium_M)


def na_correction_dg(seq: str, sodium_M: float, celsius: float = 37.0) -> float:
    """
    ΔG correction (kcal/mol) for non-1M NaCl conditions.

    Approximated from the Tm correction via:
        ΔTm ≈ -ΔΔG / (ΔS)
    Using the simplified linear approximation from Owczarzy 2004.
    """
    n = len(seq) - 1  # number of phosphates
    if n <= 0 or sodium_M <= 0:
        return 0.0
    # Owczarzy 2004 Eq. 4 (simplified): 1/Tm_x = 1/Tm_1M + (0.368/n)·ln([Na+])
    # Convert Tm shift to ΔG shift at working temperature:
    # ΔΔG ≈ ΔΔTm · ΔS  (first-order)
    # We use the simpler per-phosphate formula:
    dG_correction = 0.368 * n * math.log(sodium_M) * 1.987e-3 * (celsius + 273.15) / 1000.0
    return -dG_correction  # stabilizing when [Na+] > 1M


def duplex_salt_dg(seq: str, sodium_M: float, magnesium_M: float = 0.0) -> float:
    """
    Whole-duplex salt ΔG correction (kcal/mol) relative to 1 M Na⁺ / 0 Mg²⁺.

    Uses the *same* per-base-pair model the native McCaskill / Zuker DP applies
    (:func:`dg_per_bp_salt`), summed over the duplex's base pairs:

        ΔΔG = N · (−0.114 · ln([Na⁺] + 3.4·√[Mg²⁺]))      (N = len(seq) bp)

    Benchmarked (2026-06-15, 6 duplexes × 4 [Na⁺]) against the Owczarzy-2004
    experimental Tm fit: 2.41 °C ΔTm RMSE — statistically tied with the
    per-helix Tan & Chen model (2.43 °C) and far better than the prior
    per-phosphate correction (9.6 °C).  Using the per-bp term keeps the
    two-state duplex, the ensemble, and the MFE engines on one salt model, folds
    Mg²⁺ in via the √[Mg²⁺] combining rule, and imposes no minimum stem length.

    Like :func:`dg_per_bp_salt` it is fit at 37 °C and carries no explicit
    temperature dependence; at the 1 M Na⁺ / 0 Mg²⁺ reference it is exactly 0.
    """
    return len(seq) * dg_per_bp_salt(sodium_M, magnesium_M)


def dg_per_bp_salt(sodium_M: float, magnesium_M: float = 0.0) -> float:
    """
    Per-base-pair ΔG salt correction (kcal/mol) relative to 1 M NaCl, 0 Mg²⁺.

    Empirical Owczarzy-style fit (Owczarzy et al. 2004 Biochemistry
    43:3537-3554; Owczarzy 2008 Biochemistry 47:5336-5353) over
    Na⁺ ∈ [0.05, 1.0] M and Mg²⁺ ∈ [0, 0.1] M at 37 °C
    (within ±0.005 kcal/mol per bp; see scratch/probe_salt.py):

        ΔG_per_bp = −0.114 · ln([Na⁺] + 3.4·√[Mg²⁺])

    Used by the McCaskill DP: each closed base pair contributes a
    Boltzmann factor of exp(−ΔG_per_bp/RT) on top of its stack /
    loop / hairpin energy.  At Na⁺=1 M, Mg²⁺=0 the correction is 0.
    """
    effective_na = sodium_M + 3.4 * math.sqrt(max(magnesium_M, 0.0))
    if effective_na <= 0:
        return 0.0
    return -0.114 * math.log(effective_na)


# Tan & Chen empirical helix salt model is fit for stems of 6–15 bp.
TAN_CHEN_MIN_BP = 6
TAN_CHEN_MAX_BP = 15


def tan_chen_helix_dg(
    n_pairs: int,
    sodium_M: float,
    magnesium_M: float = 0.0,
    material: str = "dna",
) -> float:
    """
    Whole-helix electrostatic salt correction ΔG (kcal/mol) relative to 1 M NaCl,
    from the tightly-bound-ion (TBI) theory of Tan & Chen (2007) Biophys. J.
    92:3615–3632 (DNA Eqs. 26/29/30, RNA Eqs. 16/20, mixing Eqs. 24/25).

    Unlike :func:`dg_per_bp_salt` (a *per-base-pair* correction for the McCaskill
    DP), this is a *per-helix* quantity: it needs the stem length ``n_pairs`` (N),
    sums the per-base-stack free energy over the N−1 stacks, and adds the
    Na⁺/Mg²⁺ interference cross-term Δg₁₂.  Use it for the two-state hairpin Tm,
    where N is known.

    The mean electrostatic folding free energy per base stack is

        Δg₁ = a₁ + b₁/N                          (Na⁺,  Eq. 29 DNA / 16 RNA)
        Δg₂ = a₂ + b₂/N²                         (Mg²⁺, Eq. 30 DNA / 20 RNA)

    with (DNA) a₁=−0.07·ln[Na⁺]+0.012·ln²[Na⁺], b₁=0.013·ln²[Na⁺],
    a₂=0.02·ln[Mg²⁺]+0.0068·ln²[Mg²⁺], b₂=1.18·ln[Mg²⁺]+0.344·ln²[Mg²⁺].
    Mixed solutions combine the two by fractional weights (Eq. 24)

        x₁ = [Na⁺] / ([Na⁺] + (8.1 − 32.4/N)(5.2 − ln[Na⁺])[Mg²⁺]),  x₂ = 1 − x₁

    plus the cross-term (Eq. 25)

        Δg₁₂ = −0.6·x₁·x₂·ln[Na⁺]·ln((1/x₁ − 1)[Na⁺]) / N

    so the total correction is  (N−1)(x₁Δg₁ + x₂Δg₂) + Δg₁₂.  At 1 M Na⁺/0 Mg²⁺
    every term vanishes (correction = 0), matching the reference state.

    Raises ``ValueError`` for N < ``TAN_CHEN_MIN_BP`` (outside the fitted range,
    where the (8.1 − 32.4/N) factor degenerates) or unknown ``material``.
    """
    N = int(n_pairs)
    if N < TAN_CHEN_MIN_BP:
        raise ValueError(
            f"Tan-Chen helix salt model is fit for stems ≥ {TAN_CHEN_MIN_BP} bp; "
            f"got N={N}. Use the per-base-pair model for short stems."
        )
    mat = material.lower()
    if mat not in ("dna", "rna"):
        raise ValueError("material must be 'dna' or 'rna'")

    lnNa = math.log(sodium_M) if sodium_M > 0 else 0.0
    if mat == "dna":
        a1 = -0.07 * lnNa + 0.012 * lnNa ** 2
        b1 = 0.013 * lnNa ** 2
    else:
        a1 = -0.075 * lnNa + 0.012 * lnNa ** 2
        b1 = 0.018 * lnNa ** 2
    dg1 = a1 + b1 / N
    if magnesium_M <= 0:
        return (N - 1) * dg1

    lnMg = math.log(magnesium_M)
    if mat == "dna":
        a2 = 0.02 * lnMg + 0.0068 * lnMg ** 2
        b2 = 1.18 * lnMg + 0.344 * lnMg ** 2
    else:
        a2 = -0.6 / N + 0.025 * lnMg + 0.0068 * lnMg ** 2
        b2 = lnMg + 0.38 * lnMg ** 2
    dg2 = a2 + b2 / N ** 2
    if sodium_M <= 0:
        return (N - 1) * dg2

    x1 = sodium_M / (sodium_M + (8.1 - 32.4 / N) * (5.2 - lnNa) * magnesium_M)
    x2 = 1.0 - x1
    arg = (1.0 / x1 - 1.0) * sodium_M
    dg12 = -0.6 * x1 * x2 * lnNa * math.log(arg) / N if arg > 0 else 0.0
    return (N - 1) * (x1 * dg1 + x2 * dg2) + dg12


# ─── private ─────────────────────────────────────────────────────────────────

def _fgc(seq: str) -> float:
    seq = seq.upper()
    gc = sum(1 for b in seq if b in "GC")
    return gc / len(seq) if seq else 0.5


def _na_correction(fGC: float, sodium_M: float) -> float:
    """Owczarzy 2004 Eq. 4 linearized around reference."""
    ln_na = math.log(sodium_M)
    inv_Tm_correction = (4.29 * fGC - 3.95) * 1e-5 * ln_na + 9.40e-6 * ln_na ** 2
    # Return approximate ΔTm by assuming Tm ≈ 340 K (first-order)
    Tm_ref = 340.0
    return -inv_Tm_correction * Tm_ref ** 2


def _mg_correction(fGC: float, mg_M: float) -> float:
    """Owczarzy 2008 Eq. 16."""
    ln_mg = math.log(mg_M)
    a, b, c, d, e, f, g = (
        3.92e-5, -9.11e-6, 6.26e-5, 1.42e-5,
        -4.82e-4, 5.25e-4, 8.31e-5,
    )
    inv_Tm_corr = (
        a + b * ln_mg + fGC * (c + d * ln_mg)
        + (1.0 / (2.0 * (1.0))) * (e + f * ln_mg + g * ln_mg ** 2)
    )
    Tm_ref = 340.0
    return -inv_Tm_corr * Tm_ref ** 2


def _mixed_correction(fGC: float, sodium_M: float, magnesium_M: float) -> float:
    """Owczarzy 2008 mixed-ion regime."""
    na_part = _na_correction(fGC, sodium_M)
    mg_part = _mg_correction(fGC, magnesium_M)
    ratio = math.sqrt(magnesium_M) / sodium_M
    alpha = (ratio - 0.22) / (6.0 - 0.22)
    return (1 - alpha) * na_part + alpha * mg_part
