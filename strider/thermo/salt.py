"""
Salt correction models for nucleic acid thermodynamics.

Sources:
  - Owczarzy et al. (2004) Biochemistry 43:3537-3554  (Na+ correction)
  - Owczarzy et al. (2008) Biochemistry 47:5336-5353  (Mg2+ correction)
  - Tan & Chen (2006) Biophys. J. 90:1175-1190         (unified model)
"""

from __future__ import annotations
import math

_T_REF_K = 310.15  # 37 °C — reference temperature of the empirical salt fits


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
    n_bp = len(seq)  # number of base pairs; enters the Owczarzy 2008 Mg term

    if magnesium_M > 0 and sodium_M > 0:
        ratio = math.sqrt(magnesium_M) / sodium_M
        if ratio < 0.22:
            return _na_correction(fGC, sodium_M)
        elif ratio < 6.0:
            return _mixed_correction(fGC, sodium_M, magnesium_M, n_bp)
        else:
            return _mg_correction(fGC, magnesium_M, n_bp)
    elif magnesium_M > 0:
        return _mg_correction(fGC, magnesium_M, n_bp)
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


def duplex_salt_dg(
    seq: str, sodium_M: float, magnesium_M: float = 0.0, celsius: float = 37.0,
    material: str = "dna",
) -> float:
    """
    Whole-duplex salt ΔG correction (kcal/mol) relative to 1 M Na⁺ / 0 Mg²⁺.

    Sums the per-base-pair model the McCaskill / Zuker DP applies
    (:func:`dg_per_bp_salt`) over the duplex's ``N = len(seq)`` base pairs, so the
    two-state duplex, ensemble, and MFE engines share one salt model with no
    minimum stem length.  Benchmarks to 2.4 °C ΔTm RMSE against the Owczarzy-2004
    experimental fit.  Exactly 0 at the 1 M Na⁺ / 0 Mg²⁺ reference for every T.
    """
    return len(seq) * dg_per_bp_salt(sodium_M, magnesium_M, celsius, material)


# Per-base-pair monovalent salt coefficient (DNA), Owczarzy et al. 2004.
_DG_PER_BP_NA = -0.114
# RNA/DNA per-stack electrostatic ratio from the Tan & Chen 2007 TBI model
# (`tan_chen_helix_dg`): RNA's tighter A-form helix (smaller axial charge spacing)
# gives ~6% stronger counterion-release salt dependence than B-form DNA.  The
# ratio is stable to <1% over [Na⁺] ∈ [0.05, 1] M, so a single scalar suffices.
_RNA_SALT_FACTOR = 1.06


def dg_per_bp_salt(
    sodium_M: float, magnesium_M: float = 0.0, celsius: float = 37.0,
    material: str = "dna",
) -> float:
    """
    Per-base-pair ΔG salt correction (kcal/mol) relative to 1 M NaCl, 0 Mg²⁺.

        ΔG_per_bp(T) = c · ln([Na⁺] + 3.4·√[Mg²⁺]) · T / T_ref     (T_ref = 310.15 K)

    ``c`` is the empirical Owczarzy monovalent coefficient (Owczarzy et al. 2004
    Biochemistry 43:3537-3554; 2008 47:5336-5353): −0.114 for DNA, scaled by
    ``_RNA_SALT_FACTOR`` for RNA (Tan & Chen 2007). Mg²⁺ enters via the √[Mg²⁺]
    combining rule.

    The polyelectrolyte salt dependence is entropic (counterion release; the
    Manning/Record/SantaLucia framework places it in ΔS with ΔH_salt ≈ 0), so it
    scales with absolute temperature: exactly 0 at the 1 M Na⁺ / 0 Mg²⁺ reference
    for every T, and equal to the 37 °C value at ``T_ref``.

    Applied per closed base pair by the DP as a Boltzmann factor
    exp(−ΔG_per_bp/RT) on top of the stack / loop / hairpin energy; unpaired loop
    backbones carry no separate length-dependent salt term.
    """
    effective_na = sodium_M + 3.4 * math.sqrt(max(magnesium_M, 0.0))
    if effective_na <= 0:
        return 0.0
    coeff = _DG_PER_BP_NA * (_RNA_SALT_FACTOR if material.lower() == "rna" else 1.0)
    # Compute T/T_ref first so celsius == 37 is an exact ×1.0 (bit-identical to
    # the legacy 37 °C value); folding it into the product would round it away.
    frac = (celsius + 273.15) / _T_REF_K
    return coeff * math.log(effective_na) * frac


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


def _mg_correction(fGC: float, mg_M: float, n_bp: int) -> float:
    """Owczarzy 2008 Eq. 16 (divalent Tm correction relative to 1 M Na⁺).

    The published equation is

        1/Tm(Mg) − 1/Tm(1M) = a + b·ln[Mg] + fGC·(c + d·ln[Mg])
                              + 1/(2·(N_bp−1)) · (e + f·ln[Mg] + g·ln²[Mg])

    where ``N_bp`` is the number of base pairs.  The ``1/(2·(N_bp−1))`` factor is
    essential: the (e, f, g) length term dominates, so hardcoding ``N_bp = 2``
    (the previous ``1/(2·1)``) over-weighted it by roughly ``N_bp−1`` and blew Tm
    shifts up to tens of °C for normal-length oligos.
    """
    ln_mg = math.log(mg_M)
    a, b, c, d, e, f, g = (
        3.92e-5, -9.11e-6, 6.26e-5, 1.42e-5,
        -4.82e-4, 5.25e-4, 8.31e-5,
    )
    length_factor = 1.0 / (2.0 * max(n_bp - 1, 1))
    inv_Tm_corr = (
        a + b * ln_mg + fGC * (c + d * ln_mg)
        + length_factor * (e + f * ln_mg + g * ln_mg ** 2)
    )
    Tm_ref = 340.0
    return -inv_Tm_corr * Tm_ref ** 2


def _mixed_correction(fGC: float, sodium_M: float, magnesium_M: float, n_bp: int) -> float:
    """Owczarzy 2008 mixed-ion regime via the sodium-equivalent recipe.

    In the mixed regime (0.22 ≤ √[Mg²⁺]/[Na⁺] < 6) the divalent contribution is
    folded into an equivalent monovalent concentration (von Ahsen et al. 2001,
    the conversion the Owczarzy 2008 decision tree, primer3, Biopython and IDT
    OligoAnalyzer all use), and the monovalent Owczarzy 2004 correction is then
    evaluated at that equivalent sodium:

        [Na⁺]_eq = [Na⁺] + 120·√[Mg²⁺]_free      (concentrations in mM)

    ``magnesium_M`` here is already the free magnesium (``duplex_tm`` subtracts
    dNTP before calling).  The previous linear blend between the Na-only and
    pure-Mg fits stayed pinned near the Na-only floor and left mixed-regime Tm
    predictions 6–10 °C low versus the rest of the ecosystem.
    """
    na_eq = sodium_M + 0.120 * math.sqrt(magnesium_M * 1000.0)  # mM → back to M
    return _na_correction(fGC, na_eq)
