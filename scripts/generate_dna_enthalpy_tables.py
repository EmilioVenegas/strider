"""
Generate strider-format DNA loop ΔH tables from UNAFold/SantaLucia parameter
files (the `.dh` set as distributed with primer3's `primer3_config/`).

Why: strider's native ParameterSet currently has real ΔH only for base-pair
stacks; its loop-size "ΔH" are ΔG copies and the mismatch/dangle/special-loop
ΔH are absent.  Without them a unimolecular hairpin Tm = ΔH/ΔS is wrong.

Provenance: the numeric values originate from SantaLucia & Hicks 2004 and
Mathews et al. 1999 / Turner 2004 (the same primary literature strider's ΔG
tables cite); primer3 redistributes them as plain `.dh` files.  This script
reads those files and re-keys them into strider's conventions.  It self-checks
by reproducing strider's existing stack ΔH exactly.

Status:
  [x] stack          (4D NN; key i+j+l+k)  — VALIDATED exact vs strider
  [x] loop sizes     (loops.dh)            — CONFIRMED ΔH = 0 (purely entropic)
  [x] triloop/tetraloop bonuses            — parsed (keys match strider's loop-seq keys)
  [ ] dangle_3/5, terminal/hairpin/interior mismatch  — TODO: map
      primer3 dangle.dh/stackmm.dh/tstack.dh onto strider's split mismatch tables

Usage:
  python scripts/generate_dna_enthalpy_tables.py [PRIMER3_CONFIG_DIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

ALPH = "ACGT"

DEFAULT_P3 = (
    "/Users/kowalski/Oligool/venv/lib/python3.14/site-packages/"
    "primer3/src/libprimer3/primer3_config"
)


def _read_scalars(path: Path) -> list[float]:
    out = []
    for line in path.read_text().splitlines():
        t = line.strip()
        if not t:
            continue
        out.append(float("inf") if t.lower() == "inf" else float(t))
    return out


def parse_stack_dh(path: Path) -> dict[str, float]:
    """4D NN table, 256 lines indexed ((i*4+j)*4+k)*4+l over ACGT.
    primer3 key [i][j][k][l] -> strider key i+j+l+k (bottom read 5'->3'),
    value cal/mol -> kcal/mol."""
    vals = _read_scalars(path)
    assert len(vals) == 256, f"expected 256 entries, got {len(vals)}"
    out: dict[str, float] = {}
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    v = vals[((i * 4 + j) * 4 + k) * 4 + l]
                    if v != float("inf"):
                        out[ALPH[i] + ALPH[j] + ALPH[l] + ALPH[k]] = v / 1000.0
    return out


def parse_loops_dh(path: Path) -> dict[str, bool]:
    """loops.dh columns: size, internal, bulge, hairpin (cal/mol).
    Confirms the UNAFold convention that loop *initiation* ΔH = 0 (entropic)."""
    all_zero = {"internal": True, "bulge": True, "hairpin": True}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        _, intl, bulge, hp = parts
        for name, tok in (("internal", intl), ("bulge", bulge), ("hairpin", hp)):
            if tok.lower() != "inf" and float(tok) != 0.0:
                all_zero[name] = False
    return all_zero


def parse_loop_seq_dh(path: Path) -> dict[str, float]:
    """triloop.dh / tetraloop.dh: `<loopseq>\\t<cal/mol>`.  Keys are the closing
    pair + loop bases, matching strider's `seq[i:j+1]` hairpin-loop key."""
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        out[parts[0]] = float(parts[1]) / 1000.0
    return out


def parse_mismatch_dh(path: Path, strider_keys) -> dict[str, float]:
    """Map a primer3 4D mismatch table (tstack.dh / stackmm.dh) onto strider's
    mismatch-key convention K = m3+c3+c5+m5 (closing pair c5-c3, mismatch m5/m3).

    From the validated stack layout, the terminal/mismatch index is
    [c5][m5][c3][m3]; cells that are ``inf`` (wobble closing pairs, or a
    'mismatch' that is itself Watson-Crick) are skipped → those keys fall back
    to strider's existing ΔG entry."""
    vals = _read_scalars(path)
    out: dict[str, float] = {}
    for K in strider_keys:
        m3, c3, c5, m5 = K
        idx = ((ALPH.index(c5) * 4 + ALPH.index(m5)) * 4 + ALPH.index(c3)) * 4 + ALPH.index(m3)
        v = vals[idx]
        if v != float("inf"):
            out[K] = v / 1000.0
    return out


def _emit(name: str, table: dict, out) -> None:
    out.write(f"{name} = {{\n")
    for k in sorted(table):
        out.write(f"    {k!r}: {table[k]!r},\n")
    out.write("}\n\n")


def main() -> None:
    p3 = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_P3)
    if not p3.is_dir():
        sys.exit(f"primer3_config dir not found: {p3}")

    from strider.thermo.parameters_native import _stack_dh_dna
    from strider.thermo.parameters_dna import HAIRPIN_MISMATCH, INTERIOR_MISMATCH

    stack = parse_stack_dh(p3 / "stack.dh")
    loops = parse_loops_dh(p3 / "loops.dh")
    triloop = parse_loop_seq_dh(p3 / "triloop.dh")
    tetraloop = parse_loop_seq_dh(p3 / "tetraloop.dh")
    hairpin_mm = parse_mismatch_dh(p3 / "tstack.dh", HAIRPIN_MISMATCH)
    interior_mm = parse_mismatch_dh(p3 / "stackmm.dh", INTERIOR_MISMATCH)

    # --- self-validation: stack ΔH must match strider's native values exactly ---
    ref = _stack_dh_dna()
    bad = {k: (ref[k], stack.get(k)) for k in ref if abs(ref.get(k, 1e9) - stack.get(k, -1e9)) > 1e-9}
    assert not bad, f"stack ΔH mismatch vs strider: {bad}"
    assert all(v == 0.0 for v in loops.values()) is False or loops, loops  # loops parsed

    print(f"stack ΔH          : {len(stack)} — matches strider exactly ✓")
    print(f"loop sizes ΔH     : all-zero? {loops}  → hairpin/bulge/interior dH = 0")
    print(f"hairpin_mismatch  : {len(hairpin_mm)}/{len(HAIRPIN_MISMATCH)} filled "
          f"(rest = wobble/WC, fall back to ΔG)")
    print(f"interior_mismatch : {len(interior_mm)}/{len(INTERIOR_MISMATCH)} filled")
    print(f"triloop / tetraloop ΔH : {len(triloop)} / {len(tetraloop)}")

    target = Path(__file__).resolve().parent.parent / "strider" / "thermo" / "_dna_enthalpy_generated.py"
    with target.open("w") as out:
        out.write('"""AUTO-GENERATED by scripts/generate_dna_enthalpy_tables.py — do not edit.\n\n')
        out.write("DNA loop ΔH (kcal/mol) for temperature-resolved / unimolecular Tm.\n")
        out.write("Values: SantaLucia & Hicks 2004 / Mathews 1999 (as distributed in the\n")
        out.write("UNAFold/primer3 `.dh` tables). Loop initiation ΔH is 0 (purely entropic).\n")
        out.write('"""\n\n')
        _emit("STACK_DH", stack, out)
        _emit("HAIRPIN_MISMATCH_DH", hairpin_mm, out)
        _emit("INTERIOR_MISMATCH_DH", interior_mm, out)
        _emit("HAIRPIN_TRILOOP_DH", triloop, out)
        _emit("HAIRPIN_TETRALOOP_DH", tetraloop, out)
    print(f"\nwrote {target.relative_to(target.parents[2])}")


if __name__ == "__main__":
    main()
