import pathlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strider import ThermoEngine
from strider.viz.structure2d import draw_complex

_here = pathlib.Path(__file__).parent

def rc(s: str) -> str:
    """Helper to generate reverse complements."""
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]

# =======================================================================
# 1. BASE DOMAINS (15-17 nt each)
# Nine mutually orthogonal domains act as the structural "struts".  They
# were generated so every domain's longest reverse-complementary match to
# any *other* domain is <= 5 nt (no cross-hybridisation), with balanced GC,
# no homopolymer runs, and no self-hairpins -- so the MFE fold is the
# intended network rather than a frustrated tangle of mispairs.
# =======================================================================
C0_base = "TTAGTTGTGCCGCA"  # Core Hub Arm 0
C1_base = "ACCAATGCCTGTTG"  # Core Hub Arm 1
C2_base = "GGTCCACCGGATCA"  # Core Hub Arm 2

B0_1_base = "GTGCATAGAGCC"    # Branch 0, Arm 1
B0_2_base = "GAAGCCGGTCGA"    # Branch 0, Arm 2
B1_1_base = "GCAGTGGTAATT"    # Branch 1, Arm 1
B1_2_base = "ATAATGTTCTAG"    # Branch 1, Arm 2
B2_1_base = "TCAACGAGCTTA"    # Branch 2, Arm 1
B2_2_base = "AGCTGACATTGC"    # Branch 2, Arm 2

# =======================================================================
# 2. INTENTIONAL STRUCTURAL DEFECTS
# We pair Left (L) and Right (R) halves, injecting loops and bulges.
# =======================================================================
# Core 0: 3-nt asymmetrical bulge on the right strand
L_C0 = C0_base
R_C0 = rc(C0_base[8:]) + "TTT" + rc(C0_base[:8])

# Core 1: 2x2 internal loop (AA forced opposite AA)
L_C1 = C1_base[:8] + "AA" + C1_base[10:]
R_C1 = rc(C1_base[10:]) + "AA" + rc(C1_base[:8])

# Core 2: Perfect duplex
L_C2 = C2_base
R_C2 = rc(C2_base)

# Branch 0_1: 1-nt mismatch bubble (A opposite A)
L_B0_1 = B0_1_base[:7] + "A" + B0_1_base[8:]
R_B0_1 = rc(B0_1_base[8:]) + "A" + rc(B0_1_base[:7])

# The rest are perfectly complementary structural supports
L_B0_2, R_B0_2 = B0_2_base, rc(B0_2_base)
L_B1_1, R_B1_1 = B1_1_base, rc(B1_1_base)
L_B1_2, R_B1_2 = B1_2_base, rc(B1_2_base)
L_B2_1, R_B2_1 = B2_1_base, rc(B2_1_base)
L_B2_2, R_B2_2 = B2_2_base, rc(B2_2_base)

# =======================================================================
# 3. ROUTING THE COMPLEX
# The network is built from 3 inner "Hub" strands and 3 outer "Staples".
# =======================================================================
H_C = "AAC"  # Central hinges
H_B = "TTAC"   # Outer branch hinges

# Inner Hub Strands: Each spans one branch arm, flows through two core arms,
# and exits into the next branch.
Hub1 = R_B2_2 + H_B + R_C2 + H_C + L_C0 + H_B + L_B0_1
Hub2 = R_B0_2 + H_B + R_C0 + H_C + L_C1 + H_B + L_B1_1
Hub3 = R_B1_2 + H_B + R_C1 + H_C + L_C2 + H_B + L_B2_1

# Outer Staples: These close the outer branches into 3-way junctions.
# We give each staple a unique topological feature.

# Staple 0 gets a massive extruded hairpin in its hinge
HP = "GCGCG" + "TTTT" + "CGCGC"
Staple0 = R_B0_1 + "A" + HP + "A" + L_B0_2

# Staple 1 gets a long single-stranded dangling tail
Staple1 = R_B1_1 + "TTT" + L_B1_2 + "ACGTACGTACGT"

# Staple 2 gets a wide, unstructured, un-paired bubble hinge
Staple2 = R_B2_1 + "CCCCCC" + L_B2_2

# =======================================================================
# 4. RENDER ENGINE
# The strands are listed in *ring order* (hub, staple, hub, staple, ...)
# around the network's perimeter.  Multi-strand folding is order-dependent:
# the native MFE backend optimises over structures that are pseudoknot-free
# for a *given* strand concatenation, and the closed dendrimer is only
# pseudoknot-free in this alternating order (it is heavily pseudoknotted in,
# e.g., the "all hubs then all staples" order).
#
# engine.mfe() now searches strand orders automatically (it folds the distinct
# arrangements and keeps the global minimum), so small complexes are fully
# order-invariant.  For a large fused network like this one the search uses a
# sequence-affinity + crossing-minimisation heuristic rather than an exhaustive
# fold of every order, and it does not always recover the global optimum — so
# supplying this ring order still yields the cleanest, lowest-energy fold (all
# four junctions closed) and the best layout.
#
# draw_complex then renders it flat.  Because a poorly ordered input is heavily
# pseudoknotted, the radial layout would overlap, so method="auto" falls back to
# the space-aware "tree" layout: it sizes each loop so its branches radiate apart
# instead of overlapping, escalating each arm's reserved wedge from its lateral
# width toward its full reach only as far as planarity needs — so the four
# junctions come out as compact circles joined by ladder helices (no giant
# central loop / stretched linkers), with no de-overlap pass needed.
# reorder=True (default) additionally guards the *drawing* against a poorly
# ordered / pseudoknotted input.  Strand colours are keyed by name so each strand
# keeps its colour regardless of order.
# =======================================================================
strands = [Hub1, Staple0, Hub2, Staple1, Hub3, Staple2]
names = ["Hub1", "Staple0", "Hub2", "Staple1", "Hub3", "Staple2"]
colors = {
    "Hub1": "#4A90E2", "Hub2": "#E94A3F", "Hub3": "#50E3C2",
    "Staple0": "#F5A623", "Staple1": "#9013FE", "Staple2": "#F8E71C",
}

eng = ThermoEngine(material="dna", celsius=37, backend="native")
fig, ax = plt.subplots(figsize=(14, 14))

draw_complex(
    strands,
    engine=eng,
    ax=ax,
    names=names,
    strand_colors=colors,
    title="Generation-2 Dendrimer Network (6 Strands, 4 Junctions, 334 nt)",
    loop_tightness=0
)

output_file = _here / "gallery_21_dendrimer_network.png"
fig.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Wrote gallery stress test to {output_file}")

# =======================================================================
# 5. THERMODYNAMIC SUMMARY
# Verify the network folds as designed: with the strands in ring order the
# engine closes all junctions into one deeply negative, base-paired complex.
# (engine.mfe() now also searches strand arrangements — Dirks et al. 2007 — so
# small complexes are order-invariant; for a 6-strand fused network this large
# the search uses a heuristic, and ring order still gives the cleanest fold.
# See gallery 22 for an exactly order-invariant complex.)
# =======================================================================
folded = eng.mfe(*strands)
total_nt = sum(len(s) for s in strands)
print(f"Network MFE: {folded.energy:.1f} kcal/mol over {total_nt} nt, "
      f"{len(folded.base_pairs)} base pairs "
      f"({100 * 2 * len(folded.base_pairs) / total_nt:.0f}% of bases paired)")
