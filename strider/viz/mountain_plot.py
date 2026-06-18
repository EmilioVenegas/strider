"""Mountain plot and energy landscape visualization."""

from __future__ import annotations

from strider.viz import style


def mountain_plot(
    sequence: str,
    structures: list[str] | None = None,
    pair_probs=None,
    engine=None,
    ax=None,
    title: str = "Mountain Plot",
):
    """
    Draw mountain plot: nesting depth vs. sequence position.

    Multiple structures can be overlaid. If pair_probs is given,
    the expected mountain height is computed from probabilities.
    """
    import matplotlib.pyplot as plt
    from strider.structure.mountain import mountain_vector, mountain_from_probs

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(sequence) // 4), 3))

    if structures is None and engine is not None:
        result = engine.mfe(sequence)
        structures = [result.structure]

    colors = style.STRAND_CYCLE

    if pair_probs is not None:
        expected = mountain_from_probs(pair_probs)
        ax.fill_between(range(len(expected)), expected, alpha=0.2, color="gray", label="Expected")
        ax.plot(expected, color="gray", linewidth=1, linestyle="--")

    for i, struct in enumerate(structures or []):
        mv = mountain_vector(struct)
        c = colors[i % len(colors)]
        label = f"Structure {i + 1}" if len(structures) > 1 else "MFE"
        ax.plot(mv, color=c, linewidth=1.5, label=label)
        ax.fill_between(range(len(mv)), mv, alpha=0.15, color=c)

    ax.set_xlabel("Position")
    ax.set_ylabel("Nesting depth")
    ax.set_xlim(0, len(sequence) - 1)
    ax.set_title(title)
    if structures and len(structures) > 1:
        ax.legend()
    return ax


def energy_landscape(
    pathway: dict[str, float],
    barriers: dict[str, float] | None = None,
    ax=None,
    title: str = "Energy Landscape",
):
    """
    Reaction coordinate diagram from state names → ΔG values.

    pathway  : {state_name: ΔG_kcal_mol}
    barriers : optional {transition_name: ΔG_barrier} between states
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(pathway) * 2), 4))

    states = list(pathway.items())
    n = len(states)
    xs = list(range(n))
    ys = [g for _, g in states]
    labels = [name for name, _ in states]

    # Draw energy levels as horizontal lines
    for x, y, label in zip(xs, ys, labels):
        ax.hlines(y, x - 0.3, x + 0.3, linewidth=3, color=style.C_NATIVE)
        ax.text(x, y + 0.15, label.replace("_", "\n"), ha="center", fontsize=8)

    # Connect states with lines
    for i in range(n - 1):
        ax.plot([xs[i] + 0.3, xs[i + 1] - 0.3], [ys[i], ys[i + 1]],
                "k--", linewidth=0.8, alpha=0.5)
        mid_y = (ys[i] + ys[i + 1]) / 2
        ddg = ys[i + 1] - ys[i]
        ax.text((xs[i] + xs[i + 1]) / 2, mid_y, f"ΔΔG={ddg:.1f}",
                ha="center", fontsize=7, color="gray")

    ax.set_ylabel("Free energy (kcal/mol)")
    ax.set_xticks([])
    ax.set_title(title)
    return ax
