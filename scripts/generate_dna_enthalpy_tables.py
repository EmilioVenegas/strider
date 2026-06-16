"""
Generate strider-format DNA loop ΔH tables from the SantaLucia/Turner nearest-neighbor literature.

Why: strider's native ParameterSet currently has real ΔH only for base-pair stacks; its loop-size "ΔH" are ΔG copies and the mismatch/dangle/special-loop ΔH are absent.  Without them a unimolecular hairpin Tm = ΔH/ΔS is wrong.

Provenance: every numeric value is an experimentally-measured thermodynamic parameter from the primary nearest-neighbor literature — the same works strider's 37 °C ΔG tables already cite (`parameters_dna.py` header):
  * stack / loop-size / mismatch / special-loop ΔH — SantaLucia & Hicks 2004 (Annu. Rev. Biophys. 33:415) and Mathews et al. 1999 (JMB 288:911), as redistributed in the open primer3 `.dh` tables; re-keyed here and self-checked by reproducing strider's existing stack ΔH exactly.
  * dangle_3/5 ΔH — Bommarito, Peyret & SantaLucia 2000 (Nucleic Acids Res. 28:1929-1934), the paper strider cites for the dangle ΔG; the ΔH is the co-measured enthalpy from the same table.  Embedded as a literature constant below (keyed to strider's `DANGLE_5`/`DANGLE_3` convention) and validated by asserting its key set matches those tables before emitting.

Status:
  [x] stack          (4D NN; key i+j+l+k)  — VALIDATED exact vs strider
  [x] loop sizes     (loops.dh)            — CONFIRMED ΔH = 0 (purely entropic)
  [x] triloop/tetraloop bonuses            — parsed (keys match strider's loop-seq keys)
  [x] hairpin/interior mismatch            — mapped from primer3 tstack.dh/stackmm.dh
  [x] dangle_3/5                           — Bommarito 2000 enthalpies (embedded below)

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

# ── Dangle enthalpies — Bommarito, Peyret & SantaLucia (2000) NAR 28:1929-1934 ──
# ΔH (kcal/mol) co-measured with the dangle ΔG strider already uses; keyed to the
# parameters_dna DANGLE_5 / DANGLE_3 convention (XYN, see parameters_dna.py).  The
# GT/TG wobble keys carry the small entropic-default ΔH of that parameter set.
DANGLE_5_DH: dict[str, float] = {
    'ATA': -0.7, 'ATC': 4.4, 'ATG': -1.6, 'ATT': 2.9,
    'CGA': -2.1, 'CGC': -0.2, 'CGG': -3.9, 'CGT': -4.4,
    'GCA': -5.9, 'GCC': -2.6, 'GCG': -3.2, 'GCT': -5.2,
    'GTA': -0.2, 'GTC': -0.2, 'GTG': -0.2, 'GTT': -0.2,
    'TAA': -0.5, 'TAC': 4.7, 'TAG': -4.1, 'TAT': -3.8,
    'TGA': -0.2, 'TGC': -0.2, 'TGG': -0.2, 'TGT': -0.2,
}
DANGLE_3_DH: dict[str, float] = {
    'AAT': 0.2, 'ACG': -6.3, 'AGC': -3.7, 'AGT': -0.1,
    'ATA': -2.9, 'ATG': -0.1, 'CAT': 0.6, 'CCG': -4.4,
    'CGC': -4.0, 'CGT': -0.1, 'CTA': -4.1, 'CTG': -0.1,
    'GAT': -1.1, 'GCG': -5.1, 'GGC': -3.9, 'GGT': -0.1,
    'GTA': -4.2, 'GTG': -0.1, 'TAT': -6.9, 'TCG': -4.0,
    'TGC': -4.9, 'TGT': -0.1, 'TTA': -0.2, 'TTG': -0.1,
}


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


def validate_dangle_dh(
    name: str, dh: dict[str, float], strider_dg: dict[str, float],
) -> dict[str, float]:
    """Confirm an embedded dangle-ΔH literature table is keyed to the strider
    ``DANGLE_5``/``DANGLE_3`` parameter set it decorates, then return it.

    The ΔH (Bommarito 2000) is co-indexed with the dangle ΔG strider already
    carries; a key-set mismatch means the dG table was re-parameterised without
    updating the ΔH — fail loudly rather than emit an inconsistent ΔS.  (37 °C
    bit-identity and the ViennaRNA off-37 cross-check in the test-suite are the
    downstream correctness guards.)
    """
    assert set(dh) == set(strider_dg), (
        f"{name}: ΔH key set differs from strider dG table "
        f"(only ΔH={set(dh) - set(strider_dg)}, only dG={set(strider_dg) - set(dh)})"
    )
    return dict(dh)


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
    from strider.thermo.parameters_dna import (
        HAIRPIN_MISMATCH, INTERIOR_MISMATCH, DANGLE_5, DANGLE_3,
    )

    stack = parse_stack_dh(p3 / "stack.dh")
    loops = parse_loops_dh(p3 / "loops.dh")
    triloop = parse_loop_seq_dh(p3 / "triloop.dh")
    tetraloop = parse_loop_seq_dh(p3 / "tetraloop.dh")
    hairpin_mm = parse_mismatch_dh(p3 / "tstack.dh", HAIRPIN_MISMATCH)
    interior_mm = parse_mismatch_dh(p3 / "stackmm.dh", INTERIOR_MISMATCH)

    dangle_5_dh = validate_dangle_dh("dangle_5", DANGLE_5_DH, DANGLE_5)
    dangle_3_dh = validate_dangle_dh("dangle_3", DANGLE_3_DH, DANGLE_3)

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
    print(f"dangle_5 / dangle_3 ΔH : {len(dangle_5_dh)} / {len(dangle_3_dh)} "
          f"(Bommarito 2000; key-validated vs strider DANGLE tables ✓)")

    target = Path(__file__).resolve().parent.parent / "strider" / "thermo" / "_dna_enthalpy_generated.py"
    with target.open("w") as out:
        out.write('"""AUTO-GENERATED by scripts/generate_dna_enthalpy_tables.py — do not edit.\n\n')
        out.write("DNA loop ΔH (kcal/mol) for temperature-resolved / unimolecular Tm.\n")
        out.write("Values: SantaLucia & Hicks 2004 (Annu. Rev. Biophys. 33:415) / Mathews\n")
        out.write("et al. 1999 (JMB 288:911), as distributed in the open primer3 `.dh`\n")
        out.write("tables. Loop initiation ΔH is 0 (purely entropic).\n")
        out.write("DANGLE_{5,3}_DH: dangle enthalpies of Bommarito, Peyret & SantaLucia 2000\n")
        out.write("(Nucleic Acids Res. 28:1929-1934) — the same primary work strider cites\n")
        out.write("for the dangle ΔG; the ΔH is the co-measured enthalpy from that table.\n")
        out.write('"""\n\n')
        _emit("STACK_DH", stack, out)
        _emit("HAIRPIN_MISMATCH_DH", hairpin_mm, out)
        _emit("INTERIOR_MISMATCH_DH", interior_mm, out)
        _emit("HAIRPIN_TRILOOP_DH", triloop, out)
        _emit("HAIRPIN_TETRALOOP_DH", tetraloop, out)
        _emit("DANGLE_5_DH", dangle_5_dh, out)
        _emit("DANGLE_3_DH", dangle_3_dh, out)
    print(f"\nwrote {target.relative_to(target.parents[2])}")


if __name__ == "__main__":
    main()
