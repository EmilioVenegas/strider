"""
[EXPERIMENTAL API — outside the semver contract until stabilized; see README
"API stability and versioning".]

Surface-tethered biophysics: the electrode/bead transducer layer.

NUPACK is a bulk-solution calculator that stops at static equilibrium.  This
subpackage adds the surface physics clinical biosensors actually run on:

  * :mod:`strider.surface.transducer` — diffusion-limited capture (Shoup–Szabo),
    finite-capacity ULOQ, label read-out, and a limit of detection, from any
    concentration trace C(t) (e.g. a mantis ``SimulationResult`` species).
  * :mod:`strider.surface.labels` — reporter models (charge per captured event),
    incl. a Prussian-Blue nanoparticle with a counterion-addressability sub-model,
    plus the read-out electronics floor.
  * :mod:`strider.surface.thermo` — tether-entropy and double-layer/local-salt
    ΔG corrections that warp bulk hybridization at a surface; the salt part plugs
    into ``ThermoEngine(correction_model=…)``.
  * :mod:`strider.surface.stochastic` — low-copy capture shot noise: Poisson
    counting statistics, the Currie detection-limit framework, and a mantis
    Gillespie SSA driver — the regime where counting noise, not the electronics,
    sets the limit of detection.
"""

from strider.surface.labels import (
    LabelModel, PrussianBlueLabel, ReadoutChain,
    pbnp_redox_centres, pb_addressable_fraction,
)
from strider.surface.transducer import (
    SurfaceModel, SurfaceParams, TransduceResult,
    shoup_szabo_f, captured_count,
)
from strider.surface.thermo import (
    SurfaceCorrection, tether_dg, double_layer_local_salt, debye_length_m,
    SPACER_TETHER_DG,
)
from strider.surface.stochastic import (
    StochasticSurfaceModel, CurrieLevels, CaptureSamples,
    currie_levels, detection_probability, readout_sigma_counts, K_CURRIE,
)

__all__ = [
    # transducer
    "SurfaceModel", "SurfaceParams", "TransduceResult",
    "shoup_szabo_f", "captured_count",
    # low-copy stochastic capture (shot-noise-limited detection)
    "StochasticSurfaceModel", "CurrieLevels", "CaptureSamples",
    "currie_levels", "detection_probability", "readout_sigma_counts", "K_CURRIE",
    # labels
    "LabelModel", "PrussianBlueLabel", "ReadoutChain",
    "pbnp_redox_centres", "pb_addressable_fraction",
    # surface thermodynamics
    "SurfaceCorrection", "tether_dg", "double_layer_local_salt", "debye_length_m",
    "SPACER_TETHER_DG",
]
