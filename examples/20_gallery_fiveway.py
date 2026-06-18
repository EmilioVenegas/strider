import pathlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strider import ThermoEngine
from strider.viz.structure2d import draw_complex

_here = pathlib.Path(__file__).parent

# Define the pairing arms explicitly to introduce structural complexity
# like bulges, internal loops, mismatches, and varying lengths.
# Format: (left_half, right_half) where left_half is on strand i (3' end)
# and right_half is on strand i+1 (5' end).
ARMS = [
    # Arm 0: Long perfect duplex (19 bp)
    ("CGATCAGGTCAACGTTAGG", "CCTAACGTTGACCTGATCG"),
    
    # Arm 1: 3-nt bulge on the left half (AAA)
    ("TGCACAAATGAACGTGA", "TCACGTTCAGTGCA"),
    
    # Arm 2: 2x2 internal loop (TT opposite AA)
    ("ACGTGTTTCATGTCAG", "CTGACATGAAACACGT"),
    
    # Arm 3: Straight perfect duplex (18 bp)
    ("GTAGCTGACCTACGATAC", "GTATCGTAGGTCAGCTAC"),
    
    # Arm 4: 1-nt A-C mismatch in the middle
    ("ATCCGTACAGGTTC", "GAACCTGATACGGAT"),
]

# Asymmetric single-stranded hinges connecting the arms at the central junction
HINGES = ["T", "CCA", "G", "TTCT", "AA"]

# Assemble the five strands: S_i = right_half(Arm i-1) + Hinge_i + left_half(Arm i)
strands = [
    ARMS[4][1] + HINGES[0] + ARMS[0][0],  # S1 (35 nt)
    ARMS[0][1] + HINGES[1] + ARMS[1][0],  # S2 (39 nt)
    ARMS[1][1] + HINGES[2] + ARMS[2][0],  # S3 (31 nt)
    ARMS[2][1] + HINGES[3] + ARMS[3][0],  # S4 (38 nt)
    ARMS[3][1] + HINGES[4] + ARMS[4][0],  # S5 (34 nt)
]

names = ["S1", "S2", "S3", "S4", "S5"]
colors = ["#6FA8DC", "#F2A65A", "#89C997", "#C39BD3", "#E8859B"]

eng = ThermoEngine(material="dna", celsius=37, backend="native")
fig, ax = plt.subplots(figsize=(10, 10))

draw_complex(
    strands, 
    engine=eng, 
    ax=ax, 
    names=names, 
    strand_colors=colors,
    title="Complex Five-way Multiloop (Asymmetric arms, bulges, loops, ~177 nt)"
)

fig.savefig(_here / "gallery_20_complex_fiveway.png", dpi=130, bbox_inches="tight")
print("wrote gallery_20_complex_fiveway.png")