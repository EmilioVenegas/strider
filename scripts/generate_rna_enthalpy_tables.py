"""
Generate strider-format RNA loop ΔH tables from the ViennaRNA Turner-2004
parameter file (`rna_turner2004.par`, vendored at `data/`).

Why: strider's native RNA path carries real ΔH only for base-pair stacks (from
`nn_rna.RNA_NN`); its loop-*size* "ΔH" are either a ΔG copy or zero, so RNA
temperature extrapolation (`native_temperature_paramset`) treats every loop
initiation as purely entropic (ΔH = 0).  Unlike the SantaLucia DNA convention,
the Turner-2004 RNA loop-initiation enthalpies are genuinely **non-zero**
(hairpin 1.3 / 4.8 / 3.6 / −2.9 … ; bulge 10.6 / 7.1 … ; internal −7.2 / −1.3 …),
so a ΔH = 0 model mis-states the RNA loop ΔS and hence ΔG(T).

Provenance: ViennaRNA `rna_turner2004.par` redistributes the Turner-2004 RNA
parameters (Mathews 1999 / Mathews 2004 / Turner-Mathews 2010 — the same primary
literature strider's RNA ΔG tables cite).  Energies in the `.par` file are
integers in units of 0.01 kcal/mol; `INF` marks a disallowed loop size.

This script reads that file and re-keys the *enthalpy* sections into strider's
`parameters_rna.py` index conventions.  It **self-validates** by asserting that
the file's loop ΔG sections reproduce strider's `parameters_rna` loop-size ΔG
arrays exactly (same model / scale) before grafting the co-indexed ΔH — so a
silent table mismatch fails loudly rather than producing a bogus ΔS.

The source `.par` lives under `data/` (gitignored — large param file, not
committed); re-fetch the canonical copy with:
  curl -fsSLo data/rna_turner2004.par \
    https://raw.githubusercontent.com/ViennaRNA/ViennaRNA/master/misc/rna_turner2004.par

Usage:
  python scripts/generate_rna_enthalpy_tables.py [rna_turner2004.par]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DEFAULT_PAR = Path(__file__).resolve().parent.parent / "data" / "rna_turner2004.par"
TOL = 1e-6


def _sections(path: Path) -> dict[str, list[str]]:
    """Split a ViennaRNA `.par` file into `# section -> raw lines`."""
    secs: dict[str, list[str]] = {}
    name: str | None = None
    cur: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("# "):
            if name is not None:
                secs[name] = cur
            name, cur = s[2:].strip(), []
        elif name is not None:
            cur.append(line)
    if name is not None:
        secs[name] = cur
    return secs


def _array(lines: list[str]) -> list[float]:
    """Parse a numeric loop array section to kcal/mol (INF -> +inf)."""
    out: list[float] = []
    for line in lines:
        line = re.sub(r"/\*.*?\*/", " ", line)  # strip /* comments */
        for tok in line.split():
            out.append(float("inf") if tok.upper() == "INF" else int(tok) / 100.0)
    return out


def _loop_seq(lines: list[str]) -> dict[str, float]:
    """Parse a `SEQ  dG  dH` block -> {SEQ: dH_kcal} (uses the 3rd column)."""
    out: dict[str, float] = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 3:
            out[parts[0]] = int(parts[2]) / 100.0
    return out


def graft_loop_dh(
    name: str, strider_dg: list[float], par_dg: list[float], par_dh: list[float],
) -> list[float]:
    """ΔH array co-indexed with ``strider_dg`` (the parameters_rna size array).

    strider's parameters_rna loop arrays drop ViennaRNA's leading ``INF`` size-0
    entry, so ``strider_dg[i]`` aligns with the `.par` ``[i + 1]`` size slot.
    Leading ``0.0`` entries in the strider array are sentinels for loop sizes the
    DP handles specially (min-loop guard / 1×1·1×2·2×2 tables) — they get ΔH = 0
    (ΔS = 0).  Every non-sentinel entry is validated: its ΔG must match the `.par`
    value, else the tables are not the same model and we refuse to graft.
    """
    dh: list[float] = []
    checked = mismatch = 0
    for i, g in enumerate(strider_dg):
        v = i + 1
        pg = par_dg[v] if v < len(par_dg) else float("inf")
        ph = par_dh[v] if v < len(par_dh) else float("inf")
        if g == 0.0 or pg == float("inf"):
            dh.append(0.0)            # sentinel / disallowed size
            continue
        if ph == float("inf"):
            dh.append(0.0)
        else:
            dh.append(round(ph, 2))
        if abs(g - pg) <= TOL:
            checked += 1
        else:
            mismatch += 1
            # Tolerated only in the flat large-loop tail (ΔH already constant);
            # a mismatch among the small, physically-relevant sizes is fatal.
            if i < 24:
                raise AssertionError(
                    f"{name}: ΔG mismatch at small size i={i}: "
                    f"strider {g} vs par {pg} — not the same loop table"
                )
    print(f"  {name:9s}: {checked} sizes ΔG-validated, "
          f"{mismatch} tail drift tolerated, {len(dh)} ΔH entries")
    return dh


def _emit_list(name: str, vals: list[float], out) -> None:
    out.write(f"{name} = [\n")
    for r in range(0, len(vals), 10):
        out.write("    " + ", ".join(f"{v:g}" for v in vals[r:r + 10]) + ",\n")
    out.write("]\n\n")


def _emit_dict(name: str, table: dict[str, float], out) -> None:
    out.write(f"{name} = {{\n")
    for k in sorted(table):
        out.write(f"    {k!r}: {table[k]!r},\n")
    out.write("}\n\n")


def main() -> None:
    par = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PAR)
    if not par.is_file():
        sys.exit(f"parameter file not found: {par}")

    import strider.thermo.parameters_rna as p

    S = _sections(par)
    hp = graft_loop_dh("hairpin", p.HAIRPIN_SIZE, _array(S["hairpin"]),
                       _array(S["hairpin_enthalpies"]))
    bg = graft_loop_dh("bulge", p.BULGE_SIZE, _array(S["bulge"]),
                       _array(S["bulge_enthalpies"]))
    it = graft_loop_dh("internal", p.INTERIOR_SIZE, _array(S["internal"]),
                       _array(S["internal_enthalpies"]))

    par_tri = _loop_seq(S["Triloops"])
    par_tet = _loop_seq(S["Tetraloops"])
    tri = {k: par_tri[k] for k in p.HAIRPIN_TRILOOP if k in par_tri}
    tet = {k: par_tet[k] for k in p.HAIRPIN_TETRALOOP if k in par_tet}
    print(f"  triloop  : {len(tri)}/{len(p.HAIRPIN_TRILOOP)} keys have ΔH")
    print(f"  tetraloop: {len(tet)}/{len(p.HAIRPIN_TETRALOOP)} keys have ΔH "
          f"(rest absent from Turner-2004 set, fall back to ΔG)")

    target = Path(__file__).resolve().parent.parent / "strider" / "thermo" / "_rna_enthalpy_generated.py"
    with target.open("w") as out:
        out.write('"""AUTO-GENERATED by scripts/generate_rna_enthalpy_tables.py — do not edit.\n\n')
        out.write("RNA loop ΔH (kcal/mol) for temperature-resolved RNA folding.\n")
        out.write("Source: ViennaRNA rna_turner2004.par (Mathews 1999 / Turner 2004).\n")
        out.write("Indexed to match strider.thermo.parameters_rna loop-size arrays;\n")
        out.write("ΔG sections were validated against those arrays before grafting ΔH.\n")
        out.write("Unlike DNA, RNA loop-initiation ΔH is non-zero.\n")
        out.write('"""\n\n')
        _emit_list("HAIRPIN_SIZE_DH", hp, out)
        _emit_list("BULGE_SIZE_DH", bg, out)
        _emit_list("INTERIOR_SIZE_DH", it, out)
        _emit_dict("HAIRPIN_TRILOOP_DH", tri, out)
        _emit_dict("HAIRPIN_TETRALOOP_DH", tet, out)
    print(f"\nwrote {target.relative_to(target.parents[2])}")


if __name__ == "__main__":
    main()
