"""
DNA nearest-neighbor thermodynamic parameters.

Source: SantaLucia & Hicks (2004) Annu. Rev. Biophys. Biomol. Struct. 33:415-440
        SantaLucia (1998) PNAS 95:1460-1465

Conditions: 1 M NaCl, pH 7, standard state.
Units: ΔH in kcal/mol, ΔS in cal/mol/K, ΔG37 in kcal/mol.

Keys are 5'→3' dinucleotide on the top strand (e.g. 'AA' pairs with 'TT' on
the bottom strand read 3'→5', which is equivalent to bottom 5'→3' = 'TT').
"""

from __future__ import annotations

# NN parameters: {5'XY3': (ΔH kcal/mol, ΔS cal/mol/K, ΔG37 kcal/mol)}
DNA_NN: dict[str, tuple[float, float, float]] = {
    "AA": (-7.9,  -22.2, -1.00),
    "TT": (-7.9,  -22.2, -1.00),   # complement of AA/TT (same by symmetry)
    "AT": (-7.2,  -20.4, -0.88),
    "TA": (-7.2,  -21.3, -0.58),
    "CA": (-8.5,  -22.7, -1.45),
    "TG": (-8.5,  -22.7, -1.45),   # complement of CA/GT
    "GT": (-8.4,  -22.4, -1.44),
    "AC": (-8.4,  -22.4, -1.44),   # complement of GT/CA
    "CT": (-7.8,  -21.0, -1.28),
    "AG": (-7.8,  -21.0, -1.28),   # complement of CT/GA
    "GA": (-8.2,  -22.2, -1.30),
    "TC": (-8.2,  -22.2, -1.30),   # complement of GA/CT
    "CG": (-10.6, -27.2, -2.17),
    "GC": (-9.8,  -24.4, -2.24),
    "GG": (-8.0,  -19.9, -1.84),
    "CC": (-8.0,  -19.9, -1.84),   # complement of GG/CC
}

# Initiation parameters
INIT_GC = (0.1,  -2.8,  0.98)   # terminal G-C or C-G pair
INIT_AT = (2.3,   4.1,  1.03)   # terminal A-T or T-A pair

# Symmetry correction (add when sequence is self-complementary)
SYMMETRY = (0.0, -1.4, 0.43)

R = 1.987e-3  # kcal / (mol · K)

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence (5'→3')."""
    return seq.upper().translate(COMPLEMENT)[::-1]


def is_self_complementary(seq: str) -> bool:
    """Return True if the sequence equals its own reverse complement."""
    return seq.upper() == reverse_complement(seq)


def duplex_dg(
    seq: str,
    complement: str | None = None,
    celsius: float = 37.0,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
) -> float:
    """
    ΔG (kcal/mol) of duplex formation at given temperature, [Na+], and [Mg2+].

    seq         : top strand 5'→3'
    complement  : bottom strand 5'→3' (default: perfect complement of seq)
    celsius     : temperature in Celsius
    sodium_M    : [Na+] in molar
    magnesium_M : [Mg2+] in molar (folded in via the √[Mg2+] combining rule of
                  :func:`strider.thermo.salt.dg_per_bp_salt`)

    The salt correction is applied whenever ``sodium_M != 1.0`` or
    ``magnesium_M > 0`` via :func:`strider.thermo.salt.duplex_salt_dg`
    (``N · dg_per_bp_salt``) — the same per-base-pair model the native McCaskill
    / Zuker DP uses, so this two-state path stays consistent with the folding
    engine; at the 1 M Na+ / 0 Mg2+ reference it is 0.

    Returns the bimolecular association free energy (negative = stable duplex).
    Reference: SantaLucia & Hicks 2004 (NN ΔH/ΔS); the per-bp salt term uses the
    3.4·√[Mg2+] mono-/divalent combining rule (see ``salt.dg_per_bp_salt``).
    """
    seq = seq.upper().replace("U", "T")
    if complement is None:
        complement = reverse_complement(seq)
    else:
        complement = complement.upper().replace("U", "T")

    T = celsius + 273.15

    dH, dS = _sum_nn(seq)
    dH_init, dS_init = _initiation(seq)
    dH += dH_init
    dS += dS_init

    if is_self_complementary(seq):
        dS += SYMMETRY[1]

    dG = dH - T * (dS / 1000.0)  # dS: cal → kcal

    if sodium_M != 1.0 or magnesium_M > 0:
        from strider.thermo.salt import duplex_salt_dg
        dG += duplex_salt_dg(seq, sodium_M, magnesium_M, celsius)

    return dG


def duplex_dh_ds(seq: str, complement: str | None = None) -> tuple[float, float]:
    """
    Return (ΔH kcal/mol, ΔS cal/mol/K) at standard conditions (1 M NaCl).

    Sums nearest-neighbor stacking contributions and adds initiation terms.
    """
    seq = seq.upper().replace("U", "T")
    dH, dS = _sum_nn(seq)
    dH_i, dS_i = _initiation(seq)
    if is_self_complementary(seq):
        dS += SYMMETRY[1]
    return dH + dH_i, dS + dS_i


def melting_temperature(
    seq: str,
    strand_conc_M: float = 250e-9,
    sodium_M: float = 0.137,
    magnesium_M: float = 0.0,
) -> float:
    """
    Melting temperature (°C) for the given duplex.

    Uses the full NN ΔH/ΔS and solves:
        Tm = ΔH / (ΔS + R·ln(CT/4)) − 273.15

    for non-self-complementary (CT = total strand conc).
    Salt corrections applied via Owczarzy 2004.
    """
    from strider.thermo.salt import owczarzy_tm_correction

    seq = seq.upper().replace("U", "T")
    dH, dS = duplex_dh_ds(seq)
    self_comp = is_self_complementary(seq)

    # Standard two-state formula. Symmetry is already folded into dS by
    # duplex_dh_ds() for self-complementary strands; _ln_CT picks CT vs CT/4.
    CT = strand_conc_M
    tm = (dH * 1000.0) / (dS + R * 1000.0 * _ln_CT(CT, self_comp)) - 273.15

    if sodium_M != 1.0 or magnesium_M > 0:
        tm += owczarzy_tm_correction(seq, sodium_M, magnesium_M)

    return tm


# ─── internals ───────────────────────────────────────────────────────────────

def _sum_nn(seq: str) -> tuple[float, float]:
    """Sum nearest-neighbor ΔH (kcal/mol) and ΔS (cal/mol/K) for a DNA sequence."""
    dH = dS = 0.0
    for i in range(len(seq) - 1):
        dinuc = seq[i : i + 2]
        if dinuc in DNA_NN:
            h, s, _ = DNA_NN[dinuc]
        else:
            # fallback: complement pair
            rc = reverse_complement(dinuc)
            if rc in DNA_NN:
                h, s, _ = DNA_NN[rc]
            else:
                h, s = -8.0, -22.0  # average
        dH += h
        dS += s
    return dH, dS


def _initiation(seq: str) -> tuple[float, float]:
    """Return initiation ΔH and ΔS contributions for the 5' and 3' terminal bases."""
    dH = dS = 0.0
    for end_base in (seq[0], seq[-1]):
        if end_base in ("G", "C"):
            dH += INIT_GC[0]
            dS += INIT_GC[1]
        else:
            dH += INIT_AT[0]
            dS += INIT_AT[1]
    return dH, dS


def _ln_CT(CT: float, self_comp: bool) -> float:
    """Return ln(CT) or ln(CT/4) depending on self-complementarity, per standard Tm formula."""
    import math
    if self_comp:
        return math.log(CT)
    return math.log(CT / 4.0)
