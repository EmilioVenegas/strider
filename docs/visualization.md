# Visualization

strider ships a self-contained, static-matplotlib visualization layer. The 2-D
secondary-structure layout is **native** — it does not depend on ViennaRNA — so
single strands, multi-strand complexes, whole reaction cascades, and toehold
accessibility can all be drawn directly from strider results.

All drawing functions accept an existing matplotlib `ax`/`fig` for composition
and return the Axes (or Figure for multi-panel cascades), so they slot into
larger figures. Colors come from a shared palette in `strider.viz.style`.

## Quick start

```python
import matplotlib.pyplot as plt
from strider import ThermoEngine, draw_structure, draw_complex, draw_cascade

eng = ThermoEngine(material="dna", celsius=37)

# 1) a single hairpin (folded automatically)
draw_structure("TCAACATCAGTCTGATACCTCCCTCCTTATCAGACTGA", engine=eng)

# 2) a two-strand complex, colored per strand, with the nick marked
draw_complex(["GGGGAAAACCCC", "GGGGTTTTCCCC"], engine=eng, names=["A", "B"])
plt.show()
```

## Structures and complexes

`draw_structure` renders one (possibly multi-strand) structure: backbone,
base-pair rungs, colored bases, 5'/3' and strand labels, strand-nick markers,
and dashed chords for any pseudoknot/crossing pairs.

```python
draw_structure(
    sequence,                # may contain '&' / '+' strand separators
    structure=None,          # dot-bracket; folded via engine if omitted
    color="nt",              # "nt" | "strand" | "accessibility"
    accessibility=acc,       # {position: unpaired_prob} for color="accessibility"
    domains={"toehold": (0, 6)},   # outline + label regions
    engine=eng,
)
```

`draw_complex` is the multi-strand convenience wrapper. It accepts a
`strider.tube.Complex`, a list of sequences, or a `&`-joined string, folds the
complex with strider's native (nick-aware) MFE, and colors strands distinctly.

## Toehold accessibility

Per-base accessibility is the unpaired probability `1 − Σⱼ P(i, j)` taken from
the engine's pair-probability matrix.

```python
from strider import draw_accessibility_track
from strider.viz.annotate import per_position_accessibility

acc = per_position_accessibility(H1, eng)          # {pos: prob}
draw_structure(H1, engine=eng, color="accessibility", accessibility=acc,
               domains={"toehold": (0, 6)})

# or a compact 1-D strip with domain brackets
draw_accessibility_track(H1, engine=eng, domains={"toehold": (0, 6)})
```

## Reactions and cascades

`draw_cascade` renders a whole pathway as stacked reactant → product panels with
ΔΔG / rate-annotated arrows. It is generic: pass an explicit step list, a
`CHABridge`, a `CHA` template, or an enumerator `EnumerationResult`.

```python
from strider import ThermoEngine, draw_cascade
from strider.bridge.mantis_bridge import CHABridge

bridge = CHABridge({"mirna": MIR21, "H1": h1, "H2": h2, "CP": cp}, engine=eng)
fig = draw_cascade(bridge, engine=eng, show_rates=True, title="CHA cascade")
fig.savefig("cascade.png", dpi=150, bbox_inches="tight")
```

Each species is drawn as its folded structure; the leakage step is shown with a
dashed arrow. An explicit step is `(reactants, products, meta)` where each
species is a sequence, a list of sequences, a `(label, payload)` tuple, or a
`Complex`, and `meta` may carry `ddg`, `rate`, `label`, and `leak`.

## Assembly free-energy landscape

`draw_assembly_landscape` renders a pathway as an **energy staircase beside its
assembled complexes**: a classic energy-level diagram (each macrostate a level at
its free energy, with downhill arrows and per-step ΔΔG) next to a column of
minimalist native-viz cartoons of the complex(es) present at each level, tied
together by faint leaders. An optional curved arrow shows a recycled catalyst.

```python
from strider import ThermoEngine, draw_assembly_landscape
from strider.bridge.mantis_bridge import CHABridge

bridge = CHABridge({"mirna": MIR21, "H1": h1, "H2": h2, "CP": cp}, engine=eng)
# pass a bridge to auto-build the CHA macrostate ladder + miR-21 recycle loop
ax_energy, ax_viz = draw_assembly_landscape(bridge, engine=eng)
ax_energy.figure.savefig("landscape.png", dpi=150, bbox_inches="tight")
```

Pass `axes=(ax_energy, ax_viz)` to draw into an existing multi-panel figure.
States can also be given explicitly as dicts `{energy, components, caption?,
scale?}` (or tuples); each `component` is a species (a `(label, sequences)` pair,
`Complex`, or sequence string/list) drawn side by side. Use `recycle={"src": i,
"dst": j, "label": ..., "color": ...}` to draw a catalyst loop (or `recycle=False`
to omit), `step_labels` to name the ΔΔG steps, and `scale` to enlarge a state's
cartoon.

## Arc and mountain plots

`arc_diagram` draws pairs as semicircular arcs (colored by pair type,
probability, or strand) and is multi-strand aware. `mountain_plot` and
`energy_landscape` cover nesting-depth and reaction-coordinate views.

## Command line

Every renderer is exposed under `strider draw` (writes to `--out`, format from
the file extension):

```bash
strider draw structure TCAACATCAGTCTGATACCTCCCTCCTTATCAGACTGA --out hp.png
strider draw complex GGGGAAAACCCC GGGGTTTTCCCC --names A B --out duplex.png
strider draw accessibility <H1> --domains '{"toehold":[0,6]}' --out acc.png
strider draw arc "<H1>&<H2>" --color-by strand --out arc.svg
strider draw reaction --spec cha.json --rates --out cascade.png
strider draw landscape --spec cha.json --out landscape.png
```
