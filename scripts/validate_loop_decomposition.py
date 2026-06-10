"""
Validation harness: reconstruct a hairpin's folding energy by walking its
structure and calling strider's OWN per-element energy functions, then check the
sum reproduces the MFE DP's energy.

If reconstruction == fold_mfe energy across many sequences, the structure-walk is
faithful, so the identical walk run under a ΔH `param_context` yields a trustworthy
ΔH (the basis for the unimolecular hairpin Tm = ΔH/ΔS).

Run: python scripts/validate_loop_decomposition.py
"""
from __future__ import annotations

import random

from strider.structure.mfe import fold_mfe
from strider.thermo.ensemble import (
    _stack_energy,
    _hairpin_loop_energy,
    _interior_bulge_energy,
)


def parse_pairs(db: str):
    """Return base pairs [(i,j)] sorted outermost-first, or None if not a single
    unbranched hairpin (multiloop / pseudoknot / unpaired)."""
    stack, pairs = [], []
    for k, c in enumerate(db):
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
            return None  # multiloop: not a single nested stem
    return pairs


def reconstruct_energy(seq: str, pairs, T: float) -> float:
    """Sum per-element energy for a single hairpin using strider's own functions.

    Elements between consecutive pairs (i,j) -> (ip,jp):
      nl==0 and nr==0  -> stacked pair          : _stack_energy(seq,i,j)
      otherwise        -> bulge / interior loop  : _interior_bulge_energy(...)
    Terminal loop closed by the innermost pair   : _hairpin_loop_energy(...)
    """
    total = 0.0
    n = len(pairs)
    for k in range(n - 1):
        i, j = pairs[k]
        ip, jp = pairs[k + 1]
        nl = ip - i - 1
        nr = j - jp - 1
        if nl == 0 and nr == 0:
            total += _stack_energy(seq, i, j, "dna")
        else:
            total += _interior_bulge_energy(seq, i, j, ip, jp, nl, nr, "dna")
    il, jl = pairs[-1]
    total += _hairpin_loop_energy(seq, il, jl, "dna", T)
    return total


def random_seq(n: int) -> str:
    return "".join(random.choice("ACGT") for _ in range(n))


def main() -> None:
    random.seed(7)
    T = 310.15  # 37 °C reference (fold_mfe default)
    tol = 1e-6
    tested = matched = skipped = 0
    worst = 0.0
    mism = []
    for _ in range(4000):
        seq = random_seq(random.randint(12, 34))
        struct, energy, _ = fold_mfe(seq, 37.0, "dna")
        pairs = parse_pairs(struct)
        if pairs is None:
            skipped += 1
            continue
        tested += 1
        recon = reconstruct_energy(seq, pairs, T)
        diff = abs(recon - energy)
        worst = max(worst, diff)
        if diff <= tol:
            matched += 1
        elif len(mism) < 12:
            mism.append((seq, struct, round(energy, 3), round(recon, 3), round(diff, 3)))

    print(f"single-hairpin structures tested : {tested}")
    print(f"  exact reconstruction matches   : {matched}")
    print(f"  skipped (multiloop/unpaired)   : {skipped}")
    print(f"  worst |Δ|                       : {worst:.6f} kcal/mol")
    if mism:
        print("\nmismatches (seq, struct, fold_mfe, recon, |Δ|):")
        for row in mism:
            print("  ", row)


if __name__ == "__main__":
    main()
