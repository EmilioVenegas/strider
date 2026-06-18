"""
Example 02: Hairpin folding — structure prediction and visualization

Covers:
- MFE folding with dot-bracket output
- Partition function and base-pair probabilities
- Arc diagram and mountain plot visualization
- H-type pseudoknot detection
- AT-stem vs GC-stem molecular beacon: stability tradeoff

Two molecular beacon sequences are analysed:
  BEACON_AT — 4-nt AT stem (lower stability, faster opening)
  BEACON_GC — 4-nt GC stem (higher stability, better selectivity)
This reflects a real design tradeoff in molecular beacon engineering:
a stable stem improves background rejection but slows target opening.
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
import matplotlib.pyplot as plt
from strider import ThermoEngine
from strider.structure.dot_bracket import parse_pairs, stem_regions, unpaired_positions
from strider.structure.mfe import fold_mfe
from strider.structure.pseudoknot import fold_pseudoknot
from strider.viz.arc import arc_diagram
from strider.viz.mountain_plot import mountain_plot
from strider.viz.structure2d import draw_structure

_here = pathlib.Path(__file__).parent

engine = ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)
engine_native = ThermoEngine(material="dna", celsius=37.0, sodium=0.137,
                              magnesium=0.01, backend="native")

# Molecular beacon sequences with contrasting stem stability
BEACON_AT = "ATATTTTTTTTTTTTATAT"   # 19 nt: 4-nt AT stem (palindrome) + 11-T loop
BEACON_GC = "CGCGTTTTTTTTTTTTCGCG"  # 20 nt: 4-nt GC stem (palindrome) + 12-T loop

# ── 1. MFE folding ───────────────────────────────────────────────────────────
print("── MFE Folding ──────────────────────────────────────────────")
for name, seq in [("BEACON_AT", BEACON_AT), ("BEACON_GC", BEACON_GC)]:
    structure, energy, pairs = fold_mfe(seq, celsius=37.0, material="dna")
    stems    = stem_regions(structure)
    unpaired = unpaired_positions(structure)
    print(f"\n  {name} ({len(seq)} nt)")
    print(f"    Sequence:  {seq}")
    print(f"    Structure: {structure}")
    print(f"    ΔG_MFE  = {energy:+.2f} kcal/mol")
    print(f"    Stems   : {stems}")
    print(f"    Unpaired: {len(unpaired)} positions")

# ── 2. Base-pair probabilities from partition function ───────────────────────
print("\n── Base-pair probability profiles (native DP) ───────────────")
for name, seq in [("BEACON_AT", BEACON_AT), ("BEACON_GC", BEACON_GC)]:
    result_nupack = engine.pfunc(seq)
    result_native = engine_native.pfunc(seq)
    probs = result_native.pair_probs
    p_paired = probs.sum(axis=1)
    stem_p = p_paired[:4].mean()    # 4-nt stem region
    loop_p = p_paired[4:-4].mean()  # T-rich loop
    print(f"\n  {name}: ΔG (nupack) = {result_nupack.free_energy:+.2f} kcal/mol  "
          f"ΔG (native) = {result_native.free_energy:+.2f}")
    print(f"    Mean pairing prob (stem [0:4]):  {stem_p:.3f}  ← high = stable stem")
    print(f"    Mean pairing prob (loop [4:-4]): {loop_p:.3f}  ← low = open loop")

# ── 3. Pseudoknot search ────────────────────────────────────────────────────
# Synthetic sequence with a known H-type pseudoknot for demonstration
print("\n── Pseudoknot detection ─────────────────────────────────────")
PK_SEQ = "AAAACCCCAAAATTTTGGGG"  # simplified H-type test sequence
struct_pk, energy_pk, pairs_pk = fold_pseudoknot(PK_SEQ, celsius=37.0)
print(f"  Sequence:  {PK_SEQ}")
print(f"  Structure: {struct_pk}  ([] = pseudoknot pairs)")
print(f"  Energy:    {energy_pk:+.2f} kcal/mol")
if "[" in struct_pk:
    print("  → H-type pseudoknot detected")
else:
    print("  → No pseudoknot found (simple stem-loop)")

# ── 4. Visualization ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("strider — Molecular Beacon Folding Analysis\n"
             "(AT-stem vs GC-stem stability comparison)", fontsize=13)

arc_diagram(BEACON_AT, engine=engine_native, ax=axes[0][0],
            title="BEACON_AT — arc diagram (4-nt AT stem)")
arc_diagram(BEACON_GC, engine=engine_native, ax=axes[0][1],
            title="BEACON_GC — arc diagram (4-nt GC stem)")

mountain_plot(BEACON_AT, engine=engine_native, ax=axes[1][0],
              title="BEACON_AT — mountain plot")
mountain_plot(BEACON_GC, engine=engine_native, ax=axes[1][1],
              title="BEACON_GC — mountain plot")

plt.tight_layout()
fig.savefig(_here / "hairpin_folding.png", dpi=150, bbox_inches="tight")
print("\nSaved: hairpin_folding.png")

# Native 2D folds (no ViennaRNA): the two beacons drawn as proper stem-loops
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
fig2.suptitle("strider — native 2D secondary structures", fontsize=13)
draw_structure(BEACON_AT, engine=engine_native, ax=axes2[0], title="BEACON_AT (AT stem)")
draw_structure(BEACON_GC, engine=engine_native, ax=axes2[1], title="BEACON_GC (GC stem)")
fig2.tight_layout()
fig2.savefig(_here / "hairpin_structure2d.png", dpi=150, bbox_inches="tight")
print("Saved: hairpin_structure2d.png")

# ── 5. Temperature scan: stem stability vs temperature ───────────────────────
# BEACON_GC (stronger stem) is expected to remain folded to higher temperatures.
print("\n── BEACON_GC stability across temperatures ──────────────────")
print(f"  {'T (°C)':<10} {'ΔG (kcal/mol)':<18} {'Stem access.'}")
for celsius in [20, 30, 37, 45, 55, 65]:
    eng_t = ThermoEngine(material="dna", celsius=celsius, sodium=0.137, magnesium=0.01)
    eng_t_nat = ThermoEngine(material="dna", celsius=celsius, sodium=0.137,
                              magnesium=0.01, backend="native")
    pf  = eng_t.pfunc(BEACON_GC)
    acc = eng_t_nat.toehold_accessibility(BEACON_GC, list(range(4)))
    print(f"  {celsius:<10} {pf.free_energy:<18.2f} {acc:.3f}")

print("\nDone.")
