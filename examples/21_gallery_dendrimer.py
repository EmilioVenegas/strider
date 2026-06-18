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
# 1. BASE DOMAINS (~15-17 nt each)
# We define 9 orthogonal domains to act as our structural "struts".
# =======================================================================
C0_base = "GATCGACTGCTAGCGTA"  # Core Hub Arm 0
C1_base = "TACGTACGATCGATCAA"  # Core Hub Arm 1
C2_base = "GCATCGCGCGATACGCC"  # Core Hub Arm 2

B0_1_base = "ATCGATCGAGCTAGC"    # Branch 0, Arm 1
B0_2_base = "CGATCGTACGATCGA"    # Branch 0, Arm 2
B1_1_base = "GCTAGCTAGCATCGA"    # Branch 1, Arm 1
B1_2_base = "TACGATCGCGTACGT"    # Branch 1, Arm 2
B2_1_base = "CGTACGTACGATCGA"    # Branch 2, Arm 1
B2_2_base = "CGTAGCTAGCATCGA"    # Branch 2, Arm 2

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
H_C = "AA"  # Central hinges
H_B = "T"   # Outer branch hinges

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
# =======================================================================
strands = [Hub1, Hub2, Hub3, Staple0, Staple1, Staple2]
names = ["Hub1", "Hub2", "Hub3", "Staple0", "Staple1", "Staple2"]
colors = ["#4A90E2", "#E94A3F", "#50E3C2", "#F5A623", "#9013FE", "#F8E71C"]

eng = ThermoEngine(material="dna", celsius=37, backend="native")
fig, ax = plt.subplots(figsize=(14, 14))

draw_complex(
    strands, 
    engine=eng, 
    ax=ax, 
    names=names, 
    strand_colors=colors,
    title="Generation-2 Dendrimer Network (6 Strands, 4 Junctions, 334 nt)"
)

output_file = _here / "gallery_21_dendrimer_network.png"
fig.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Wrote gallery stress test to {output_file}")