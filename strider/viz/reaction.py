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
