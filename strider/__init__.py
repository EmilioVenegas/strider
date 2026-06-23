"""
strider — Nucleic Acid Thermodynamics, Kinetics, and Circuit Design

The biophysics layer below mantis-delta: sequences → thermodynamics →
TMSD kinetics → CRNetwork.

Quick start:
    from strider import ThermoEngine, CHABridge, LeakageEnumerator

    engine = ThermoEngine(material='dna', celsius=37, sodium=0.137, magnesium=0.01)

    bridge = CHABridge(
        sequences={'mirna': MIR21, 'H1': h1_seq, 'H2': h2_seq, 'CP': cp_seq},
        engine=engine,
    )
    report = bridge.verify()
    print(report)

    rn = bridge.to_crnetwork()   # → mantis CRNetwork
    rn.simulate(bridge._default_ic(), (0, 7200))
"""

from strider.thermo.engine import ThermoEngine, MFEResult, PFuncResult
from strider.thermo.nn_dna import (
    duplex_dg,
    melting_temperature,
    reverse_complement,
    is_self_complementary,
)
from strider.thermo.salt import owczarzy_tm_correction
from strider.thermo.hairpin import (
    hairpin_tm, hairpin_thermo, fraction_folded, HairpinThermo,
)
from strider.thermo.dimer_thermo import (
    dimer_tm, dimer_thermo, dimer_thermo_subopt, DimerThermo,
)
from strider.thermo.modified import ModificationSite, apply_modifications
from strider.thermo.parameters import (
    ParameterSet, load_parameters, list_parameter_sets, param_search_paths,
)

from strider.structure.mfe import fold_mfe
from strider.structure.sampling import sample_structures, subopt_structures
from strider.structure.dot_bracket import parse_pairs, to_dot_bracket, validate
from strider.structure.pseudoknot import fold_pseudoknot
from strider.structure.quadruplex import (
    find_g4_motifs,
    fold_quadruplex,
    g4_thermodynamics,
    quadruplex_ensemble,
    G4Motif,
    G4Fold,
    QuadruplexEnsemble,
)
from strider.structure.mountain import mountain_vector, compare_structures
from strider.structure.cotranscriptional import (
    fold_cotranscriptional,
    CotranscriptionalTrajectory,
    PrefixFold,
)

from strider.kinetics.tmsd import (
    toehold_kf,
    displacement_kf,
    leakage_kf,
    rates_from_ddg,
    TMSDRateSet,
    TMSDKineticModel,
)
from strider.kinetics.arrhenius import (
    arrhenius,
    detailed_balance_kr,
    k_eq_from_ddg,
    ddg_from_k_eq,
)
from strider.kinetics.leakage import LeakageEnumerator, LeakageReport, SpuriousReaction
from strider.kinetics.enumerator import (
    DomainReactionEnumerator,
    EnumerationResult,
    DomainReaction,
)

from strider.design.objective import DesignObjective
from strider.design.constraints import HardConstraint
from strider.design.optimizer import SequenceDesigner, DomainSpec, DesignResult
# DifferentiableDesigner / DiffObjective require torch (optional extra: strider-dna[diff]).
# They are loaded lazily via __getattr__ below so `import strider` works without torch.
from strider.design.mutation import MutationAnalyzer, MutationProfile
from strider.design.assay import Assay, AssayPanel, Assembly
from strider.design.policies import (
    MutationPolicy, RandomMutationPolicy, DefectWeightedPolicy,
    ConstraintAwarePolicy, per_residue_defect_from_ensemble,
)
from strider.design.decomposition import (
    build_strand_graph, connected_components, decompose_assays,
)
from strider.design.rerank import design_with_rerank, RerankResult

from strider.screen.offtarget import OffTargetScreener, ScreeningReport

from strider.surface import (
    SurfaceModel, SurfaceParams, TransduceResult,
    LabelModel, PrussianBlueLabel, ReadoutChain,
    SurfaceCorrection, tether_dg, double_layer_local_salt, debye_length_m,
    StochasticSurfaceModel, CurrieLevels, CaptureSamples,
    currie_levels, detection_probability,
)

from strider.bridge.mantis_bridge import (
    CHABridge, CHAVerificationReport, CircuitBridge, rates_to_crnetwork,
)

from strider.sweep.cache import DiskCache
from strider.sweep.batch import ParameterSweep, SweepResult

from strider.export.formats import to_vienna, to_ct, to_bpseq, to_fasta, to_oxdna, write

from strider.equilibrium import (
    solve_equilibrium, equilibrium_from_engine, EquilibriumResult,
    cyclic_symmetry, water_molarity,
)
from strider.tube import (
    Strand, Complex, SetSpec, ComplexSet, Tube, TubeResult, tube_analysis,
)

from strider.dsd import DSDCompiler

from strider.circuits import (
    CircuitTemplate, HCR, Translator, SeesawGate, CHA,
    CheckRegistry, CircuitReport, CheckResult,
    toehold_accessible, stability_in_range, reaction_driving_force,
    no_spurious_dimer, leakage_below_signal,
)

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    # Derived from git tags at build time via setuptools-scm.
    __version__ = _pkg_version("strider-dna")
except PackageNotFoundError:  # source tree that was never installed
    __version__ = "0.0.0+unknown"

__author__ = "Emilio Venegas"
__license__ = "MIT"

__all__ = [
    # Core engine
    "ThermoEngine", "MFEResult", "PFuncResult",
    # Thermodynamics
    "duplex_dg", "melting_temperature", "reverse_complement", "is_self_complementary",
    "owczarzy_tm_correction",
    "hairpin_tm", "hairpin_thermo", "fraction_folded", "HairpinThermo",
    "dimer_tm", "dimer_thermo", "dimer_thermo_subopt", "DimerThermo",
    "ModificationSite", "apply_modifications",
    "ParameterSet", "load_parameters", "list_parameter_sets", "param_search_paths",
    # Structure
    "fold_mfe", "fold_pseudoknot", "sample_structures", "subopt_structures",
    "parse_pairs", "to_dot_bracket", "validate",
    "mountain_vector", "compare_structures",
    "find_g4_motifs", "fold_quadruplex", "g4_thermodynamics", "quadruplex_ensemble",
    "G4Motif", "G4Fold", "QuadruplexEnsemble",
    # Kinetics
    "toehold_kf", "displacement_kf", "leakage_kf", "rates_from_ddg",
    "TMSDRateSet", "TMSDKineticModel",
    "arrhenius", "detailed_balance_kr", "k_eq_from_ddg", "ddg_from_k_eq",
    "LeakageEnumerator", "LeakageReport", "SpuriousReaction",
    "DomainReactionEnumerator", "EnumerationResult", "DomainReaction",
    # Design
    "DesignObjective", "HardConstraint",
    "SequenceDesigner", "DomainSpec", "DesignResult",
    "DifferentiableDesigner", "DiffObjective",
    "MutationAnalyzer", "MutationProfile",
    "Assay", "AssayPanel", "Assembly",
    "MutationPolicy", "RandomMutationPolicy", "DefectWeightedPolicy",
    "ConstraintAwarePolicy", "per_residue_defect_from_ensemble",
    "build_strand_graph", "connected_components", "decompose_assays",
    "design_with_rerank", "RerankResult",
    # Screening
    "OffTargetScreener", "ScreeningReport",
    # Surface-tethered biophysics (transducer / LOD / surface ΔG)
    "SurfaceModel", "SurfaceParams", "TransduceResult",
    "LabelModel", "PrussianBlueLabel", "ReadoutChain",
    "SurfaceCorrection", "tether_dg", "double_layer_local_salt", "debye_length_m",
    "StochasticSurfaceModel", "CurrieLevels", "CaptureSamples",
    "currie_levels", "detection_probability",
    # Bridge to mantis
    "CHABridge", "CHAVerificationReport", "CircuitBridge", "rates_to_crnetwork",
    # Sweep & cache
    "DiskCache", "ParameterSweep", "SweepResult",
    # Equilibrium
    "solve_equilibrium", "equilibrium_from_engine", "EquilibriumResult",
    "cyclic_symmetry", "water_molarity",
    # Tube / multi-strand analysis
    "Strand", "Complex", "SetSpec", "ComplexSet", "Tube", "TubeResult",
    "tube_analysis",
    # DSD
    "DSDCompiler",
    # Circuit templates
    "CircuitTemplate", "HCR", "Translator", "SeesawGate", "CHA",
    "CheckRegistry", "CircuitReport", "CheckResult",
    "toehold_accessible", "stability_in_range", "reaction_driving_force",
    "no_spurious_dimer", "leakage_below_signal",
    # Export
    "to_vienna", "to_ct", "to_bpseq", "to_fasta", "to_oxdna", "write",
    # Visualization (lazy; require matplotlib)
    "draw_structure", "draw_complex", "draw_cascade", "draw_reaction_step",
    "draw_assembly_landscape",
    "draw_accessibility_track", "arc_diagram", "mountain_plot", "energy_landscape",
    "cha_circuit", "layout_structure",
]

# Lazily import the torch-backed differentiable API (PEP 562) so the rest of the
# package is usable without torch. Accessing these names installs a clear hint if
# the optional dependency is missing.
_LAZY = {
    # name: (module_path, attr, optional_extra)
    "DifferentiableDesigner": ("strider.design.diff_designer", "DifferentiableDesigner", "diff"),
    "DiffObjective": ("strider.thermo.diff_design", "DiffObjective", "diff"),
    # Visualization (matplotlib is a core dep, so these normally import fine; kept
    # lazy so `import strider` does not pull matplotlib for non-plotting users).
    "draw_structure": ("strider.viz.structure2d", "draw_structure", "viz"),
    "draw_complex": ("strider.viz.structure2d", "draw_complex", "viz"),
    "draw_cascade": ("strider.viz.reaction", "draw_cascade", "viz"),
    "draw_reaction_step": ("strider.viz.reaction", "draw_reaction_step", "viz"),
    "draw_assembly_landscape": ("strider.viz.reaction", "draw_assembly_landscape", "viz"),
    "draw_accessibility_track": ("strider.viz.annotate", "draw_accessibility_track", "viz"),
    "arc_diagram": ("strider.viz.arc", "arc_diagram", "viz"),
    "mountain_plot": ("strider.viz.mountain_plot", "mountain_plot", "viz"),
    "energy_landscape": ("strider.viz.mountain_plot", "energy_landscape", "viz"),
    "cha_circuit": ("strider.viz.circuit_diagram", "cha_circuit", "viz"),
    "layout_structure": ("strider.viz.layout", "layout_structure", "viz"),
}


def __getattr__(name):  # noqa: D401 - module-level lazy loader
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr, extra = target
    try:
        import importlib
        module = importlib.import_module(module_path)
    except ImportError as exc:  # missing optional dependency
        raise ImportError(
            f"strider.{name} requires the optional '{extra}' dependency group. "
            f"Install it with:  pip install 'strider-dna[{extra}]'"
        ) from exc
    value = getattr(module, attr)
    globals()[name] = value  # cache so __getattr__ isn't hit again
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
