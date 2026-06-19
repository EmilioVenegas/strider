"""Gallery 22 — Four-arm DNA junction, folded order-invariantly.

A four-arm (Holliday-style) junction is the canonical branched DNA motif: four
strands meet at a central point, each contributing one duplex arm with its two
neighbours.  It is a *closed* complex, so — like the gallery-21 dendrimer — its
minimum-free-energy structure is **strand-order-dependent** under a naive linear
fold: the four arms are only simultaneously pseudoknot-free for a cyclic strand
order that walks the junction perimeter, and a linear DP folded on the "wrong"
concatenation leaves arms open.

Unlike the 6-strand dendrimer (too large for an exhaustive search), this 4-strand
complex is small enough that ``engine.mfe`` folds **every** distinct strand cut and
returns the global minimum — so the fold is *exactly* order-invariant: you get the
same −38 kcal/mol four-arm junction no matter how you list the strands.

Construction: four orthogonal 9-nt domains d0..d3; strand i carries the reverse
complement of d_i followed by d_(i+1), so strand i's 3' half pairs strand (i+1)'s
5' half all the way around the ring — a clean closed 4-cycle.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from strider import ThermoEngine  # noqa: E402
from strider.viz.structure2d import draw_complex  # noqa: E402

_here = pathlib.Path(__file__).parent


def rc(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


# Four mutually orthogonal 9-nt arm domains.
DOMAINS = ["GGAATTCGC", "ACGTGCATT", "TGCATGAAG", "CAGTCCGTA"]

# Strand i = rc(d_i) + d_(i+1): its 5' half closes arm i with strand i-1, its 3'
# half opens arm i+1 with strand i+1.  Four strands → a closed four-arm junction.
strands = [rc(DOMAINS[i]) + DOMAINS[(i + 1) % 4] for i in range(4)]
names = [f"S{i}" for i in range(4)]
colors = {
    "S0": "#4A90E2", "S1": "#E94A3F", "S2": "#50E3C2", "S3": "#F5A623",
}

eng = ThermoEngine(material="dna", celsius=37, backend="native")

# Order-invariance: fold a deliberately "scrambled" listing and the canonical
# ring listing — the exhaustive order search returns the identical MFE.
ring = eng.mfe(*strands)
scrambled = eng.mfe(strands[0], strands[2], strands[1], strands[3])
print(f"Four-arm junction MFE: {ring.energy:.1f} kcal/mol, {len(ring.base_pairs)} pairs")
print(f"Scrambled-order MFE:   {scrambled.energy:.1f} kcal/mol  "
      f"(Δ = {abs(ring.energy - scrambled.energy):.2e} — exactly order-invariant)")

fig, ax = plt.subplots(figsize=(10, 10))
draw_complex(
    strands,
    engine=eng,
    ax=ax,
    names=names,
    strand_colors=colors,
    title="Four-Arm DNA Junction (4 strands, 72 nt — order-invariant fold)",
)

output_file = _here / "gallery_22_four_arm_junction.png"
fig.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Wrote four-arm junction figure to {output_file}")
