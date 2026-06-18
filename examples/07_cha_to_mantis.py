"""
Example 07: Full pipeline — sequences → thermodynamics → CRNetwork → simulate

Demonstrates the complete strider↔mantis integration:
1. Compute CHA thermodynamic pathway with ThermoEngine
2. Verify all 7 design checks with CHABridge
3. Export CRNetwork to mantis
4. Simulate CHA dynamics
5. Plot signal accumulation
"""

import pathlib
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
import matplotlib.pyplot as plt
from strider import ThermoEngine, CHABridge
from strider.kinetics.tmsd import toehold_kf

_here = pathlib.Path(__file__).parent

# ── Sequences (from verify_codesign.py matching 04_cha_cascade.py) ──────────
MIR21_SEQ = "TAGCTTATCAGACTGATGTTGA"
H1_SEQ    = "TCAACATCAGTCTGATAAGCTAACTTAATTAAGTTAGCTTATCAGACTG"
H2_SEQ    = "CAGTCTGATAAGCTAACTTAATTAAGTTAGCTTATCAGACTGATGTTGACCCAACAT"
CP_SEQ    = "ATGTTGGG"

# ── 1. Engine ────────────────────────────────────────────────────────────────
engine = ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01, backend="native")
print(f"Backend: {engine.backend_name}")

# ── 2. CHABridge ─────────────────────────────────────────────────────────────
bridge = CHABridge(
    sequences={"mirna": MIR21_SEQ, "H1": H1_SEQ, "H2": H2_SEQ, "CP": CP_SEQ},
    engine=engine,
    toehold_d1=7,
    toehold_d2=15,
    tail_cp=8,
)

# ── 3. Thermodynamic pathway ─────────────────────────────────────────────────
ddg = bridge.ddg_pathway
print("\n── ΔΔG Pathway ─────────────────────────────────────────")
print(f"  G(H1)           = {ddg['g_H1']:.2f} kcal/mol")
print(f"  G(H2)           = {ddg['g_H2']:.2f} kcal/mol")
print(f"  ΔΔG(R1, init)   = {ddg['R1']:.2f} kcal/mol")
print(f"  ΔΔG(R2, prop)   = {ddg['R2']:.2f} kcal/mol")
print(f"  ΔΔG(R3, detect) = {ddg['R3']:.2f} kcal/mol")
print(f"  ΔΔG(spont)      = {ddg['leakage']:.2f} kcal/mol")
print(f"  ΔΔG(CP leakage) = {ddg['cp_leakage']:.2f} kcal/mol")

# ── 4. Verification ──────────────────────────────────────────────────────────
print("\n── Verification ─────────────────────────────────────────")
report = bridge.verify()
print(report)

# ── 5. Kinetic rates ─────────────────────────────────────────────────────────
print("\n── Kinetic rates (M⁻¹s⁻¹ or s⁻¹) ──────────────────────")
for key, val in bridge.rates.items():
    print(f"  {val:10.3e}  {key}")

# ── 6. Mantis CRNetwork ──────────────────────────────────────────────────────
try:
    rn = bridge.to_crnetwork()
    print(f"\n── mantis CRNetwork ────────────────────────────────────")
    print(f"  Species:   {rn.n_species}")
    print(f"  Reactions: {rn.n_reactions}")
    print(f"  Deficiency: δ = {rn.deficiency}")
    print(f"  Weakly reversible: {rn.is_weakly_reversible}")

    # Initial conditions
    ic = bridge._default_ic()
    print(f"\n  Initial conditions: {ic}")

    # Simulate CHA cascade
    print("\n  Simulating CHA dynamics (0 → 7200 s)...")
    result = rn.simulate(ic, t_span=(0, 7200))
    if result.success:
        final = result.final()
        h1h2_cp = [k for k in final if "CP" in k and "_" in k]
        if h1h2_cp:
            signal_key = h1h2_cp[0]
            print(f"  Final [{signal_key}] = {final[signal_key]:.3e} M")
            print(f"  Predicted signal fraction: {report.signal_fraction_predicted:.1%}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Signal accumulation
        ax = axes[0]
        for species, concs in result.concentrations.items():
            if any(k in species for k in ["H1H2", "H1h2"]):
                ax.plot(result.times / 60, concs * 1e9, label=species, linewidth=2)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Concentration (nM)")
        ax.set_title("CHA Signal Accumulation")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Energy landscape
        from strider.viz.mountain_plot import energy_landscape
        pathway_states = {
            "H1+H2+CP+miR": 0.0,
            "miR·H1+H2+CP": ddg["R1"],
            "H1·H2+CP+miR": ddg["R1"] + ddg["R2"],
            "H1·H2·CP+miR": ddg["R1"] + ddg["R2"] + ddg["R3"],
        }
        energy_landscape(pathway_states, ax=axes[1], title="CHA Energy Landscape")

        plt.tight_layout()
        fig.savefig(_here / "cha_pipeline.png", dpi=150, bbox_inches="tight")
        print("\n  Saved: cha_pipeline.png")
    else:
        print("  Simulation failed to converge.")

except ImportError as e:
    print(f"\n  mantis not available: {e}")
    print("  Install with: pip install mantis-delta")
    print("  (Thermodynamic analysis above still works without mantis)")

# ── 7. TMSD toehold sweep ────────────────────────────────────────────────────
print("\n── Toehold kf sweep ─────────────────────────────────────")
for nt in range(4, 13):
    kf = toehold_kf(nt, material="dna", celsius=37.0)
    print(f"  {nt:2d} nt → kf = {kf:.2e} M⁻¹s⁻¹")

# ── 8. Whole-cascade figure (generic reaction renderer) ──────────────────────
import matplotlib.pyplot as plt  # noqa: E402
from strider.viz.reaction import draw_cascade  # noqa: E402

fig_casc = draw_cascade(bridge, engine=engine, show_rates=True,
                        title="CHA reaction cascade (miR-21)")
fig_casc.savefig(_here / "cha_cascade.png", dpi=150, bbox_inches="tight")
plt.close(fig_casc)
print("\n  Saved: cha_cascade.png")
