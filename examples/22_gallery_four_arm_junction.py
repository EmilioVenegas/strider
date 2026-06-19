"""Gallery 22 — Four-arm DNA junction: MFE structure + suboptimal ensemble.

Goes beyond a single draw_complex render (that's gallery-19's job) to showcase
strider's analytical toolkit on the same motif:

  Left   MFE structure of the four-arm junction.
  Right  Suboptimal ensemble: the top-4 low-energy conformers drawn side by
         side so a designer can see which alternative folds compete with the
         intended junction.

Construction: four 12-nt arm domains (d0..d3) with deliberate structural
features — a 2-nt bulge on arm 1, a T·T mismatch on arm 2, and a dangling
3'-tail on strand S3 — so the suboptimal ensemble actually has something
interesting to reveal (a perfect junction is thermodynamically boring).
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from strider import ThermoEngine                          # noqa: E402
from strider.viz.structure2d import draw_complex, draw_structure  # noqa: E402

_here = pathlib.Path(__file__).parent


def rc(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


# ── Arm domains ──────────────────────────────────────────────────────────────
# Four 12-nt domains; arms are paired L / R across adjacent strands.
D0 = "CGATCAGGTCAA"
D1 = "TGCACTGAACGT"
D2 = "ACGTGTCATGTA"
D3 = "GTAGCTGACCTG"

# Paired halves with deliberate defects:
L_D0, R_D0 = D0, rc(D0)                                  # arm 0: perfect
L_D1, R_D1 = D1, rc(D1[:5]) + "AA" + rc(D1[7:])          # arm 1: 2-nt bulge (AA insert on R)
L_D2 = D2[:6] + "T" + D2[7:]                              # arm 2: T·T mismatch at pos 6
R_D2 = rc(D2)
L_D3, R_D3 = D3, rc(D3)                                  # arm 3: perfect

# Hinge nucleotides at the central junction.
H = "TT"

# Strand assembly: S_i = R(arm i-1) + hinge + L(arm i).
# S3 gets a 5-nt dangling 3' tail for accessibility contrast.
TAIL = "ACGTC"
strands = [
    R_D3 + H + L_D0,            # S0
    R_D0 + H + L_D1,            # S1
    R_D1 + H + L_D2,            # S2
    R_D2 + H + L_D3 + TAIL,     # S3 (tail)
]
names = ["S0", "S1", "S2", "S3"]
colors = {
    "S0": "#4A90E2",
    "S1": "#E94A3F",
    "S2": "#50E3C2",
    "S3": "#F5A623",
}

eng = ThermoEngine(material="dna", celsius=37, backend="native")

# ── Fold and print summary ───────────────────────────────────────────────────
mfe = eng.mfe(*strands)
total_nt = sum(len(s) for s in strands)
print(f"Four-arm junction ({len(strands)} strands, {total_nt} nt)")
print(f"  MFE: {mfe.energy:.2f} kcal/mol, {len(mfe.base_pairs)} base pairs")
print(f"  Structure: {mfe.structure}")

# Ensemble defect
edef = eng.ensemble_defect(tuple(strands), mfe.structure)
print(f"  Ensemble defect (normalized): {edef:.4f}")

# Suboptimals within 3 kcal/mol
# subopt() returns raw structural energies (no σ / association / coaxial
# corrections).  Shift them by the same offset so they match mfe.energy.
subs = eng.subopt(*strands, gap=3.0, max_structures=20)
correction = mfe.energy - subs[0][1]  # corrected MFE − raw structural MFE
print(f"  Suboptimals within 3.0 kcal/mol: {len(subs)} structures")
for i, (ss, en, _) in enumerate(subs[:6]):
    tag = " ← MFE" if i == 0 else ""
    print(f"    [{i}] {en + correction:+.2f} kcal/mol  {ss[:40]}...{tag}")

# ── Figure: 1×2 layout (MFE + suboptimals) ──────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.suptitle(
    f"Four-Arm Junction — Structural Analysis  ({total_nt} nt, "
    f"MFE = {mfe.energy:.1f} kcal/mol, "
    f"ens. defect = {edef:.3f})",
    fontsize=14, fontweight="bold", y=0.97,
)

# Left panel: MFE structure (no domain overlay)
ax_a = fig.add_subplot(1, 2, 1)
draw_complex(
    strands, engine=eng, ax=ax_a, names=names,
    strand_colors=colors,
    title="MFE fold",
)

# Right panel: Top-4 suboptimal structures
n_sub = min(4, len(subs))
ax_d = fig.add_subplot(1, 2, 2)
ax_d.set_axis_off()
ax_d.set_title(f"Top-{n_sub} suboptimal conformers", fontsize=11)

# Inset axes within the right half
bbox = ax_d.get_position()
sub_w = bbox.width / 2 * 0.92
sub_h = bbox.height / 2 * 0.88
for idx in range(n_sub):
    row, col = divmod(idx, 2)
    left = bbox.x0 + col * (bbox.width / 2) + 0.01
    bottom = bbox.y0 + (1 - row) * (bbox.height / 2) - 0.02
    ax_sub = fig.add_axes([left, bottom, sub_w, sub_h])
    ss_db, ss_en, ss_pairs = subs[idx]
    joined_seq = "&".join(strands)
    draw_structure(
        joined_seq, ss_db, engine=eng, ax=ax_sub,
        strand_names=names, strand_colors=colors,
        color="strand", base_text=False, labels=False,
        title=f"#{idx}: {ss_en + correction:+.1f} kcal/mol",
    )

fig.subplots_adjust(wspace=0.15, top=0.91)
output_file = _here / "gallery_22_four_arm_junction.png"
fig.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"\nWrote analysis to {output_file}")
