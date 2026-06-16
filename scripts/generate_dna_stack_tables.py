"""
Generate strider's DNA STACK (ΔG at 37 °C) and STACK_DH (ΔH) tables from
primer3/UNAFold `stack.{dh,ds}` and `stackmm.{dh,ds}` parameter files.

The mismatch-stack tables are the internal single-mismatch nearest-neighbor
parameters from Allawi & SantaLucia (Biochemistry 1997-1999) as redistributed by
primer3.  WC stacks are taken from primer3's `stack.{dh,ds}` for the ΔH table and
preserved from strider's existing `STACK` for the ΔG table (the two sources agree
to ~0.02 kcal/mol; keeping the existing rounded values avoids regressing tests).

Usage:
    python scripts/generate_dna_stack_tables.py [PRIMER3_CONFIG_DIR]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ALPH = "ACGT"
T = 310.15  # K (37 °C)

DEFAULT_P3 = (
    "/Users/kowalski/Oligool/venv/lib/python3.14/site-packages/"
    "primer3/src/libprimer3/primer3_config"
)

REPO_ROOT = Path(__file__).resolve().parent.parent

WC_PAIRS = {"AT", "TA", "CG", "GC"}


def _is_wc(key: str) -> bool:
    """A stack has two WC pairs: top5-bottom5 and top3-bottom3."""
    assert len(key) == 4
    return (key[0] + key[3] in WC_PAIRS) and (key[1] + key[2] in WC_PAIRS)


def _read_matrix(path: Path) -> np.ndarray:
    vals: list[float] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()[1]
            vals.append(float("inf") if tok.lower() == "inf" else float(tok))
    return np.array(vals, dtype=float).reshape(4, 4, 4, 4)


def _build_tables(p3_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Return (dg, dh) dicts keyed by strider's top5-top3-bottom3-bottom5 key."""
    dg: dict[str, float] = {}
    dh: dict[str, float] = {}
    for ds_path, dh_path in (
        (p3_dir / "stack.ds", p3_dir / "stack.dh"),
        (p3_dir / "stackmm.ds", p3_dir / "stackmm.dh"),
    ):
        arr_ds = _read_matrix(ds_path)
        arr_dh = _read_matrix(dh_path)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        vds = arr_ds[i, j, k, l]
                        vdh = arr_dh[i, j, k, l]
                        if not np.isfinite(vds):
                            continue
                        # primer3 file index [i][j][k][l] = top5 top3 bottom5 bottom3
                        # strider key             = top5 top3 bottom3 bottom5
                        key = ALPH[i] + ALPH[j] + ALPH[l] + ALPH[k]
                        dg[key] = (vdh - T * vds) / 1000.0
                        dh[key] = vdh / 1000.0
    return dg, dh


def _clean(v: float) -> float:
    return 0.0 if abs(v) < 5e-4 else v


def _replace_block(path: Path, pattern: str, repl: str) -> None:
    text = path.read_text()
    new_text, n = re.subn(pattern, repl, text, flags=re.DOTALL | re.MULTILINE)
    if n != 1:
        raise RuntimeError(f"Expected 1 match, got {n} in {path} for pattern {pattern[:80]}...")
    path.write_text(new_text)


def main() -> None:
    p3 = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_P3)
    if not p3.is_dir():
        sys.exit(f"primer3_config dir not found: {p3}")

    sys.path.insert(0, str(REPO_ROOT))
    from strider.thermo.parameters_dna import STACK as old_stack

    p3_dg, p3_dh = _build_tables(p3)

    wc_count = sum(1 for k in p3_dg if _is_wc(k))
    mm_count = len(p3_dg) - wc_count
    print(f"Primer3 finite stacks: {len(p3_dg)} ({wc_count} WC + {mm_count} mismatch)")

    # Use existing rounded ΔG values for WC stacks to avoid regressing current
    # tests/snapshots; mismatches come from primer3.  ΔH for all stacks comes
    # from primer3.
    new_stack: dict[str, float] = {}
    new_dh: dict[str, float] = {}
    mismatched: dict[str, float] = {}
    for key in p3_dg:
        if _is_wc(key):
            if key in old_stack:
                new_stack[key] = _clean(old_stack[key])
            else:
                # Fallback to primer3-derived value if not present somehow.
                new_stack[key] = _clean(round(p3_dg[key], 3))
        else:
            mismatched[key] = _clean(round(p3_dg[key], 3))
            new_stack[key] = mismatched[key]
        new_dh[key] = _clean(round(p3_dh[key], 1))

    keys = sorted(new_stack)

    # --- Update parameters_dna.py ------------------------------------------------
    target = REPO_ROOT / "strider" / "thermo" / "parameters_dna.py"
    stack_lines = []
    for k in keys:
        if _is_wc(k):
            # Keep strider's existing rounded WC ΔG values unchanged.
            stack_lines.append(f'    "{k}": {old_stack[k]!r},')
        else:
            stack_lines.append(f'    "{k}": {new_stack[k]:.3f},')
    stack_body = "\n".join(stack_lines)
    # Replace the whole STACK block including the type-annotated assignment.
    _replace_block(
        target,
        r'^STACK: dict\[str, float\] = \{.*?\n\}\n',
        f'STACK: dict[str, float] = {{\n{stack_body}\n}}\n',
    )
    # Update the comment above the table to match the new size.
    _replace_block(
        target,
        r'# \d+ entries: \d+ Watson-Crick \+ \d+ mismatch stacks\.\n',
        f'# {len(new_stack)} entries: {wc_count} Watson-Crick + {mm_count} mismatch stacks.\n',
    )

    # --- Update _dna_enthalpy_generated.py --------------------------------------
    target = REPO_ROOT / "strider" / "thermo" / "_dna_enthalpy_generated.py"
    dh_body = "\n".join(f"    '{k}': {new_dh[k]:.1f}," for k in keys)
    _replace_block(
        target,
        r'^STACK_DH = \{.*?\n\}\n',
        f'STACK_DH = {{\n{dh_body}\n}}\n',
    )

    print(f"Wrote {len(new_stack)} STACK entries ({wc_count} WC + {mm_count} mismatch).")
    # Sanity check: the generated values should agree with primer3-derived
    # mismatch literature values for the original 20 strider mismatch keys.
    bad = {
        k: (old_stack.get(k), mismatched.get(k))
        for k in old_stack
        if not _is_wc(k) and k in mismatched and abs(old_stack[k] - mismatched[k]) > 0.05
    }
    if bad:
        print("WARNING: large drift for original mismatch keys:", bad)


if __name__ == "__main__":
    main()
