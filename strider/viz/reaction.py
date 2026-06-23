"""
Reaction- and cascade-level visualization.

``draw_cascade`` renders a whole reaction pathway as a stack of
reactant → product panels with ΔΔG / rate-annotated arrows.  It is *generic*:
``steps`` may be an explicit list, an
:class:`~strider.kinetics.enumerator.EnumerationResult`, a
:class:`~strider.bridge.mantis_bridge.CHABridge`, or a ``circuits.CHA`` template.
``draw_reaction_step`` renders a single step.

A *species* is normalized to ``(label, payload)`` where ``payload`` is a list of
sequences (drawn as a folded structure) or ``None`` (drawn as a labeled box when
sequences are not resolvable, e.g. abstract domain complexes).
"""

from __future__ import annotations

from strider.viz import style
from strider.viz.structure2d import draw_structure


# --------------------------------------------------------------------------- #
# Species / step normalization
# --------------------------------------------------------------------------- #
def _species_label(obj) -> str:
    if isinstance(obj, tuple) and len(obj) == 2:
        return str(obj[0])
    if hasattr(obj, "name") and obj.name:
        return str(obj.name)
    if hasattr(obj, "canonical_name"):
        return str(obj.canonical_name)
    if isinstance(obj, str):
        return obj if len(obj) <= 10 else obj[:8] + "…"
    if isinstance(obj, (list, tuple)):
        return "·".join(_species_label(s) for s in obj)
    return str(obj)


def _species_payload(obj) -> list[str] | None:
    """Return a list of nucleotide sequences for ``obj``, or None if abstract."""
    if isinstance(obj, tuple) and len(obj) == 2:
        payload = obj[1]
    else:
        payload = obj
    if hasattr(payload, "sequences"):
        return list(payload.sequences)
    if isinstance(payload, str):
        return [payload] if set(payload.upper()) <= set("ACGTUN&+") else None
    if isinstance(payload, (list, tuple)) and all(isinstance(s, str) for s in payload):
        return list(payload)
    return None


# --------------------------------------------------------------------------- #
# Drawing primitives
# --------------------------------------------------------------------------- #
def _draw_species(ax, species, engine, seq_color=None):
    label = _species_label(species)
    seqs = _species_payload(species)
    if seqs and engine is not None:
        try:
            structure = engine.mfe(*seqs).structure
            # identity-based strand colors: each strand keeps its color across panels
            strand_colors = None
            if seq_color is not None and len(seqs) > 1:
                strand_colors = [seq_color.get(s, style.strand_color(k))
                                 for k, s in enumerate(seqs)]
            draw_structure(
                "&".join(seqs), structure, engine=engine, ax=ax,
                color="strand" if len(seqs) > 1 else "structure",
                strand_colors=strand_colors, base_text=False, labels=False,
            )
            ax.set_title(label, fontsize=9)
            return
        except Exception:
            pass
    # abstract / unresolved species: a labeled box
    ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round", fc="#eef3f8", ec=style.C_NATIVE))
    ax.set_axis_off()


def _draw_arrow(ax, ddg=None, rate=None, label=None, leak=False):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    col = style.NICK if leak else "#455a64"
    ls = "dashed" if leak else "solid"
    ax.annotate("", xy=(0.92, 0.5), xytext=(0.08, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, linestyle=ls))
    if label:
        ax.text(0.5, 0.78, label, ha="center", fontsize=9, color=col, fontweight="bold")
    if ddg is not None:
        ax.text(0.5, 0.60, f"ΔΔG={ddg:.1f}", ha="center", fontsize=8, color=col)
    if rate is not None:
        ax.text(0.5, 0.24, f"k={rate:.1e}", ha="center", fontsize=8, color="gray",
                style="italic")


def _draw_row(fig, gs, row, ncols, reactants, products, meta, engine, seq_color=None):
    cells: list = []
    for k, sp in enumerate(reactants):
        if k:
            cells.append(("plus", None))
        cells.append(("species", sp))
    cells.append(("arrow", meta))
    for k, sp in enumerate(products):
        if k:
            cells.append(("plus", None))
        cells.append(("species", sp))

    # left-align cells within the row's ncols
    for col, (kind, payload) in enumerate(cells[:ncols]):
        ax = fig.add_subplot(gs[row, col])
        if kind == "species":
            _draw_species(ax, payload, engine, seq_color=seq_color)
        elif kind == "plus":
            ax.text(0.5, 0.5, "+", ha="center", va="center", fontsize=16)
            ax.set_axis_off()
        else:  # arrow
            _draw_arrow(ax, ddg=payload.get("ddg"), rate=payload.get("rate"),
                        label=payload.get("label"), leak=payload.get("leak", False))
    # blank out unused trailing cells
    for col in range(len(cells), ncols):
        ax = fig.add_subplot(gs[row, col])
        ax.set_axis_off()


# --------------------------------------------------------------------------- #
# Adapters: normalize various inputs to a list of (reactants, products, meta)
# --------------------------------------------------------------------------- #
def _steps_from_cha_bridge(bridge) -> list[tuple]:
    s = bridge.sequences
    m = s.get("mirna", s.get("target", ""))
    H1, H2, CP = s.get("H1", ""), s.get("H2", ""), s.get("CP", "")
    ddg = bridge.ddg_pathway
    fwd = [v for k, v in bridge.rates.items() if "->" in k][0::2]  # forward kf per step

    def rate(i):
        return fwd[i] if i < len(fwd) else None

    steps = [
        ([("miRNA", m), ("H1", H1)], [("m·H1", [m, H1])],
         {"ddg": ddg["R1"], "rate": rate(0), "label": "R1"}),
        ([("m·H1", [m, H1]), ("H2", H2)], [("H1·H2", [H1, H2]), ("miRNA", m)],
         {"ddg": ddg["R2"], "rate": rate(1), "label": "R2"}),
    ]
    if CP:
        steps.append(
            ([("H1·H2", [H1, H2]), ("CP", CP)], [("H1·H2·CP", [H2, H1, CP])],
             {"ddg": ddg["R3"], "rate": rate(2), "label": "R3"})
        )
    steps.append(
        ([("H1", H1), ("H2", H2)], [("H1·H2", [H1, H2])],
         {"ddg": ddg["leakage"], "rate": rate(3), "label": "leak", "leak": True})
    )
    return steps


def _steps_from_enumeration(result, engine) -> list[tuple]:
    steps = []
    for rxn in result.reactions:
        reactants = list(rxn.reactants)
        products = list(rxn.products)
        meta = {"rate": getattr(rxn, "kf", None), "label": getattr(rxn, "mechanism", "")}
        steps.append((reactants, products, meta))
    return steps


def _build_seq_color(norm: list[tuple]) -> dict[str, str]:
    """Assign a stable color to each distinct strand sequence (first-seen order)."""
    seq_color: dict[str, str] = {}
    for reactants, products, _ in norm:
        for sp in list(reactants) + list(products):
            for s in (_species_payload(sp) or []):
                if s not in seq_color:
                    seq_color[s] = style.strand_color(len(seq_color))
    return seq_color


def _normalize_steps(steps, engine) -> list[tuple]:
    # CHABridge / CHA template
    if hasattr(steps, "ddg_pathway"):
        return _steps_from_cha_bridge(steps)
    if hasattr(steps, "bridge") and hasattr(steps.bridge, "ddg_pathway"):
        return _steps_from_cha_bridge(steps.bridge)
    # EnumerationResult
    if hasattr(steps, "reactions"):
        return _steps_from_enumeration(steps, engine)
    # already a list of (reactants, products, meta)
    return list(steps)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def draw_reaction_step(
    reactants, products, *, engine=None, fig=None, ddg=None, rate=None,
    label=None, leak=False, title=None,
):
    """Render a single ``reactants → products`` step. Returns the Figure."""
    import matplotlib.pyplot as plt

    reactants = reactants if isinstance(reactants, list) else [reactants]
    products = products if isinstance(products, list) else [products]
    meta = {"ddg": ddg, "rate": rate, "label": label, "leak": leak}
    ncols = len(reactants) + len(products) + 1 + (len(reactants) - 1) + (len(products) - 1)
    if fig is None:
        fig = plt.figure(figsize=(2.6 * ncols, 3.0))
    gs = fig.add_gridspec(1, ncols)
    seq_color = _build_seq_color([(reactants, products, meta)])
    _draw_row(fig, gs, 0, ncols, reactants, products, meta, engine, seq_color=seq_color)
    if title:
        fig.suptitle(title)
    return fig


# --------------------------------------------------------------------------- #
# Assembly free-energy landscape (energy staircase + assembled-complex column)
# --------------------------------------------------------------------------- #
def _assembly_seq_color(norm) -> dict[str, str]:
    """Stable first-seen color per distinct strand sequence across all states."""
    seq_color: dict[str, str] = {}
    for st in norm:
        for comp in st["components"]:
            for s in (_species_payload(comp) or []):
                if s not in seq_color:
                    seq_color[s] = style.strand_color(len(seq_color))
    return seq_color


def _draw_assembly_component(ax, species, engine, seq_color):
    """Draw one component (a strand or a sub-complex) as a minimal thumbnail."""
    from strider.viz.structure2d import draw_complex

    seqs = _species_payload(species)
    if not seqs:
        ax.set_axis_off()
        ax.text(0.5, 0.5, _species_label(species), ha="center", va="center",
                fontsize=8)
        return
    cols = [seq_color.get(s, style.strand_color(k)) for k, s in enumerate(seqs)]
    if len(seqs) == 1:
        draw_structure(seqs[0], engine=engine, ax=ax, color="strand",
                       strand_colors=cols, minimal=True, labels=False)
    else:
        draw_complex(seqs, engine=engine, ax=ax, strand_colors=cols,
                     minimal=True, labels=False)


def _draw_assembly_ensemble(inset, components, engine, seq_color):
    """Lay several components side by side inside one state's inset."""
    inset.axis("off")
    n = len(components)
    for c, comp in enumerate(components):
        sub = inset.inset_axes([c / n, 0.0, 1.0 / n, 1.0])
        _draw_assembly_component(sub, comp, engine, seq_color)


def _states_from_cha_bridge_assembly(bridge):
    """Build the CHA macrostate ladder (reactants → reporter) from a bridge."""
    s = bridge.sequences
    m = s.get("mirna", s.get("target", ""))
    H1, H2, CP = s.get("H1", ""), s.get("H2", ""), s.get("CP", "")
    ddg = bridge.ddg_pathway

    lv = [0.0, ddg["R1"], ddg["R1"] + ddg["R2"]]
    states = [
        # H2 only enters at R2, so the reactant level carries just miR-21 + H1.
        {"energy": lv[0], "components": [("miR-21", m), ("H1", H1)],
         "caption": "miR-21\n+ H1"},
        {"energy": lv[1], "components": [("miR·H1", [m, H1]), ("H2", H2)],
         "caption": "miR·H1\n+ H2"},
        # miR-21 is displaced here, so it reappears as a free product strand.
        {"energy": lv[2], "components": [("H1·H2", [H1, H2]), ("miR-21", m)],
         "caption": "H1·H2\n+ miR-21"},
    ]
    steps = ["R1", "R2"]
    recycle = None
    if CP:
        states.append(
            {"energy": lv[2] + ddg["R3"],
             "components": [("H1·H2·CP", [H1, H2, CP])],
             "caption": "H1·H2·CP\n(reporter)", "scale": (1.15, 1.7)})
        steps.append("R3")
    # fixed per-strand colors so each strand keeps its identity across cartoons
    seq_color = {m: style.C_WARN, H1: style.C_NATIVE,
                 H2: style.C_TORCH, CP: style.C_ACCENT}
    # miR-21 is regenerated: recycle it from the H1·H2 product state back to the
    # reactant pool (the catalyst loop), colored like the miR-21 strand.
    recycle = {"src": 2, "dst": 0, "label": "miR-21\nrecycled", "color": style.C_WARN}
    return states, recycle, steps, seq_color


def _normalize_states(states, engine):
    """Return ``(states, recycle, step_labels, seq_color)``.

    Each normalized state is a dict ``{energy, components, caption, scale}`` where
    ``scale`` is a ``(width, height)`` multiplier on the cartoon box.
    """
    if hasattr(states, "ddg_pathway"):
        norm, recycle, steps, seq_color = _states_from_cha_bridge_assembly(states)
    elif hasattr(states, "bridge") and hasattr(states.bridge, "ddg_pathway"):
        norm, recycle, steps, seq_color = _states_from_cha_bridge_assembly(states.bridge)
    else:
        norm, recycle, steps, seq_color = list(states), None, None, None

    out = []
    for st in norm:
        if isinstance(st, dict):
            energy = st["energy"]
            comps = st["components"]
            cap = st.get("caption")
            scale = st.get("scale", 1.0)
        else:  # (energy, components[, caption[, scale]])
            energy, comps = st[0], st[1]
            cap = st[2] if len(st) > 2 else None
            scale = st[3] if len(st) > 3 else 1.0
        if cap is None:
            cap = " +\n".join(_species_label(c) for c in comps)
        sw, sh = (scale, scale) if not isinstance(scale, (tuple, list)) else scale
        out.append({"energy": float(energy), "components": list(comps),
                    "caption": cap, "scale": (sw, sh)})
    return out, recycle, steps, seq_color


def draw_assembly_landscape(
    states,
    *,
    engine=None,
    fig=None,
    axes=None,
    recycle=None,
    step_labels=None,
    seq_color=None,
    title="Thermodynamic driving force",
    energy_label="Free energy (kcal mol$^{-1}$)",
    viz_title="assembled complex",
    show_net=True,
    energy_pad=3.5,
    width_ratios=(0.85, 1.05),
    figsize=(4.3, 4.6),
):
    """
    Draw a reaction pathway as an energy staircase beside its assembled complexes.

    Two aligned columns are produced: a classic energy-level diagram (each
    macrostate a horizontal level at its free energy, with downhill arrows and
    per-step ΔΔG), and a column of native-:mod:`strider.viz` cartoons showing the
    complex(es) assembled at each level.  Faint leaders tie each structure to its
    energy level; an optional curved arrow shows a recycled catalyst.

    Parameters
    ----------
    states
        A :class:`~strider.bridge.mantis_bridge.CHABridge` (or any object exposing
        a ``.bridge`` with a ``ddg_pathway``), in which case the CHA macrostate
        ladder is built automatically, **or** an explicit list of states.  (For a
        ``CHA`` circuit template, build the bridge first:
        ``CHABridge(cha.sequences, engine=engine)``.)  Each explicit state is
        a dict ``{energy, components, caption?, scale?}`` (or a tuple
        ``(energy, components, caption?, scale?)``).  ``components`` is a list of
        species — each a ``(label, sequences)`` pair, a ``Complex``, or a sequence
        string/list — drawn side by side.  ``scale`` is a scalar or
        ``(width, height)`` multiplier enlarging that state's cartoon.
    engine
        Thermo engine used to fold each component for its cartoon.
    axes
        Optional ``(ax_energy, ax_viz)`` to draw into (for embedding in a larger
        figure); otherwise a standalone two-column figure is created.
    recycle
        ``{"src": i, "dst": j, "label": str, "color": c}`` to draw a curved
        catalyst-recycling arrow from state ``i`` up to state ``j``.  Defaults to
        the CHA miR-21 loop when ``states`` is a bridge; pass ``False`` to omit.
    step_labels
        Optional names for the ΔΔG steps (e.g. ``["R1", "R2", "R3"]``).
    seq_color
        Optional ``{sequence: color}`` map fixing each strand's color across
        cartoons; auto-assigned (first-seen) when omitted.

    Returns
    -------
    ``(ax_energy, ax_viz)``
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch, FancyArrowPatch

    norm, default_recycle, default_steps, default_seq_color = _normalize_states(
        states, engine)
    if not norm:
        raise ValueError("no states to draw")
    if recycle is None:
        recycle = default_recycle
    if step_labels is None:
        step_labels = default_steps
    if seq_color is None:
        seq_color = default_seq_color or _assembly_seq_color(norm)

    if axes is not None:
        ax_energy, ax_viz = axes
        fig = ax_energy.figure
    else:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=0.32)
        ax_energy = fig.add_subplot(gs[0, 0])
        ax_viz = fig.add_subplot(gs[0, 1])

    levels = [st["energy"] for st in norm]
    n = len(norm)
    y_hi = max(levels) + energy_pad
    y_lo = min(levels) - energy_pad

    # ── left column: energy-level staircase ───────────────────────────────────
    xL, xR, arrow_x = 0.18, 0.78, 0.42
    for lv in levels:
        ax_energy.hlines(lv, xL, xR, color="#3A3A3A", lw=2.4, zorder=3)
    for k in range(n - 1):
        ax_energy.annotate(
            "", xy=(arrow_x, levels[k + 1]), xytext=(arrow_x, levels[k]),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6,
                            shrinkA=1.5, shrinkB=1.5), zorder=2)
        name = (f"{step_labels[k]}\n"
                if step_labels and k < len(step_labels) else "")
        ax_energy.annotate(
            f"{name}$\\Delta\\Delta G={levels[k + 1] - levels[k]:+.1f}$",
            (xR, (levels[k] + levels[k + 1]) / 2), xytext=(4, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=7.0, color=style.C_NUPACK, fontweight="bold")
    if show_net:
        net = levels[-1] - levels[0]
        ax_energy.text(
            0.5, 0.025, f"net $\\Delta G = {net:+.1f}$\nkcal mol$^{{-1}}$",
            transform=ax_energy.transAxes, ha="center", va="bottom", fontsize=8.0,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    ax_energy.set_ylabel(energy_label)
    if title:
        ax_energy.set_title(title, loc="left")
    ax_energy.set_xticks([])
    ax_energy.set_xlim(0, 1)
    ax_energy.set_ylim(y_lo, y_hi)
    for side in ("top", "right", "bottom"):
        ax_energy.spines[side].set_visible(False)

    # ── right column: assembled complexes, evenly spaced, with leaders ────────
    ax_viz.set_xlim(0, 1)
    ax_viz.set_ylim(-0.05, 1)
    ax_viz.axis("off")
    if viz_title:
        ax_viz.set_title(viz_title, fontsize=8.5, color="#555555")
    row_y = [1.0 - (k + 0.5) / n for k in range(n)]
    box_h = 0.74 / n
    cx0, cx1 = 0.06, 0.60
    for k, st in enumerate(norm):
        sw, sh = st["scale"]
        bw = (cx1 - cx0) * sw
        bh = box_h * sh
        inset = ax_viz.inset_axes([cx0, row_y[k] - bh / 2, bw, bh],
                                  transform=ax_viz.transData)
        inset.set_facecolor("none")
        _draw_assembly_ensemble(inset, st["components"], engine, seq_color)
        if st["caption"]:
            ax_viz.text(cx0 + bw + 0.04, row_y[k], st["caption"], ha="left",
                        va="center", fontsize=7.6, color="#333333")
        con = ConnectionPatch(
            xyA=(xR, levels[k]), coordsA=ax_energy.transData,
            xyB=(cx0, row_y[k]), coordsB=ax_viz.transData,
            color="#C0C0C0", lw=0.8, ls=(0, (2, 2)), zorder=0)
        fig.add_artist(con)

    if recycle:
        ya, yb = row_y[recycle["src"]], row_y[recycle["dst"]]
        col = recycle.get("color", style.C_WARN)
        arr = FancyArrowPatch(
            (0.96, ya), (0.96, yb), connectionstyle="arc3,rad=0.45",
            arrowstyle="-|>", mutation_scale=12, color=col, lw=1.7, ls="--",
            zorder=5, clip_on=False)
        ax_viz.add_patch(arr)
        if recycle.get("label"):
            ax_viz.text(1.02, (ya + yb) / 2, recycle["label"], color=col,
                        ha="center", va="center", fontsize=7.2, fontweight="bold",
                        rotation=90)

    return ax_energy, ax_viz


def draw_cascade(
    steps, *, engine=None, fig=None, show_ddg=True, show_rates=False, title=None,
):
    """
    Render a reaction cascade as stacked reactant → product panels.

    ``steps`` may be an explicit list of ``(reactants, products, meta)``, an
    ``EnumerationResult``, a ``CHABridge``, or a ``CHA`` template.  Returns the
    Figure.
    """
    import matplotlib.pyplot as plt

    norm = _normalize_steps(steps, engine)
    if not norm:
        raise ValueError("no reaction steps to draw")

    def row_cells(reactants, products):
        return len(reactants) + len(products) + 1 + max(len(reactants) - 1, 0) + max(len(products) - 1, 0)

    ncols = max(row_cells(r, p) for r, p, _ in norm)
    nrows = len(norm)

    if fig is None:
        fig = plt.figure(figsize=(2.5 * ncols, 2.8 * nrows))
    gs = fig.add_gridspec(nrows, ncols)
    seq_color = _build_seq_color(norm)

    for row, (reactants, products, meta) in enumerate(norm):
        meta = dict(meta)
        if not show_ddg:
            meta.pop("ddg", None)
        if not show_rates:
            meta.pop("rate", None)
        _draw_row(fig, gs, row, ncols, reactants, products, meta, engine, seq_color=seq_color)

    fig.suptitle(title or "Reaction cascade", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
