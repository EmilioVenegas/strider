# Changelog

All notable changes to strider-dna are documented here.
This file is generated from the git history by [git-cliff](https://git-cliff.org).
The format follows [Keep a Changelog](https://keepachangelog.com) and the project
uses [Semantic Versioning](https://semver.org).

## [0.8.0] - 2026-06-18

### Documentation

- **viz:** Add visualization guide and API reference entries

    - New docs/visualization.md: user-facing guide covering structure
      drawing, cascade rendering, accessibility tracks, and gallery.
    - docs/api.md: append mkdocstrings entries for all viz public functions.
    - mkdocs.yml: add Visualization page to nav.

- Add visualization section to README

    - Installation: add [viz] optional-dependency group
    - TOC: add §21 Visualization entry
    - CLI: document 'strider draw' subcommands (structure, complex,
      accessibility, arc, reaction) with examples
    - User guide §21: full write-up covering draw_structure, draw_complex,
      draw_cascade, draw_accessibility_track, arc_diagram, mountain_plot,
      energy_landscape, and the shared style system
    - Examples: list new gallery scripts 13–21 with descriptions


### Features

- **viz:** Add core visualization modules and shared style

    New modules:
    - style.py: shared palette constants, pair_color(), style_context()
    - geometry.py: radial/spring layout helpers for structure drawing
    - layout.py: Layout2D coordinate engine for stem-loop planar embedding
    - structure2d.py: draw_structure() and draw_complex() for native 2-D
      secondary-structure rendering without ViennaRNA
    - annotate.py: draw_accessibility_track() heatmap overlay
    - reaction.py: draw_cascade() and draw_reaction_step() for generic
      toehold-mediated strand-displacement cascades

    Updated existing viz modules:
    - arc.py: use style palette, add strand-based colouring, nick markers
    - circuit_diagram.py: delegate to draw_cascade when sequences are
      available, keep schematic fallback for DDG-only calls
    - mountain_plot.py: use style.STRAND_CYCLE and style.C_NATIVE
    - __init__.py: re-export all public viz functions

- **viz:** Add CLI draw subcommands

    Add 'strider draw' with nested subcommands:
      structure   — 2D secondary structure rendering
      complex     — multi-strand complex diagram
      accessibility — toehold accessibility track
      arc         — arc diagram (base-pair probability)
      reaction    — CHA cascade from a JSON spec

    Refactor _add_engine_args into _add_thermo_args (shared by draw and
    analysis commands) and add _add_fig_args / _savefig helpers for
    consistent figure output (--out, --dpi, --title).

- **viz:** Lazy-import viz functions from top-level strider

    Extend the PEP 562 lazy-loader to expose all viz public names
    (draw_structure, draw_complex, draw_cascade, arc_diagram, etc.)
    from 'import strider' without eagerly pulling in matplotlib.

    The _LAZY dict now stores a 3-tuple (module, attr, extra) so the
    ImportError message points to the correct optional-dependency group
    (e.g. 'strider-dna[viz]' instead of always saying torch).


### Testing

- **viz:** Add visualization test suite and baselines

    - test_viz_artists.py: smoke tests for draw_structure, draw_complex,
      arc_diagram, mountain_plot, energy_landscape, draw_accessibility_track
    - test_viz_baseline.py: pytest-mpl image comparison against
      tests/baseline/ PNGs (hairpin, duplex complex)
    - test_viz_cli.py: CLI integration tests for 'strider draw' subcommands
    - test_viz_geometry.py: unit tests for radial/spring layout helpers
    - test_viz_layout.py: Layout2D coordinate engine tests
    - test_viz_reaction.py: draw_cascade and draw_reaction_step tests


### Examples

- Add viz demos and gallery scripts

## [0.7.0] - 2026-06-17

### Documentation

- Document project limitations and improve parity testing for ViennaRNA backend
- Remove project documentation and configuration files

### Features

- Integrate Vienna backend and expand benchmarking suite

### Refactor

- Expand benchmarking suite to include multi-axis cross-validation with NUPACK and ViennaRNA

## [0.6.0] - 2026-06-16

### Features

- **thermo:** Dedicated inter-strand dimer MFE + strand-aware MFE & sub-optimal enumeration (#4)

    * feat(thermo): add bimolecular dimer Tm via dimer_thermo

    - DimerThermo dataclass with tm_celsius, dH, dS, dG37

    - structure walk for inter-strand helix with terminal penalties/dangles

    - salt policy mirrors hairpin_thermo (Tan-Chen / per-bp)

    - unit tests for perfect/hetero dimers, concentration dependence, salt policy

    - engine + package exports; Oligool backend integration ready

    * feat(thermo): add dedicated inter-strand dimer MFE via _dimer_mfe

    Implement a DP over antiparallel strand alignments (blunt, staggered, single-base bulges) so dimer_thermo(structure=None) no longer relies on the native pseudo-hairpin cofold. The returned MFEResult uses the existing parse_dimer_pairs / structure_free_energy_dimer scoring path, leaving hairpin logic untouched.

    * test(kinetics): skip mantis-dependent enumerator integration tests when optional dep absent

    TestCRNetworkIntegration exercises mantis.CRNetwork, which is an optional peer dependency. Mark the class as skipped when mantis is not installed so CI stays green without the extra.

    Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)


## [0.5.0] - 2026-06-16

### Features

- **thermo:** Implement temperature scaling for salt corrections

    Update salt correction calculations to scale linearly with absolute
    temperature, reflecting the entropic nature of counterion release.
    This replaces the fixed 37 °C correction with a T-dependent model
    that remains bit-identical at the 37 °C reference.

- **thermo:** Temperature-resolve DNA external-loop dangle/TM via stk_decoration_tables
- **thermo:** Add curated RNA dangle and terminal-mismatch enthalpies

    Integrate Schroeder & Turner 2000 enthalpy tables for RNA dangles and
    terminal mismatches. This replaces the previous fallback of ΔH = ΔG₃₇,
    enabling accurate temperature extrapolation for these parameters
    (resolves GAP-4).

- **thermo:** Add full primer3 mismatch-stack parameters for DNA (#5)

    Expand STACK (ΔG) and STACK_DH (ΔH) from 36 to 116 entries using
    primer3/UNAFord stackmm.{ds,dh} tables (Allawi & SantaLucia 1997-1999).
    WC stack ΔG values are preserved; mismatch entries are converted from
    ΔH/TΔS at 310.15 K.

    Includes scripts/generate_dna_stack_tables.py to regenerate both tables.

- **thermo:** Add material-aware salt corrections for RNA and DNA

    Introduces a `material` parameter to the salt correction functions to
    distinguish between DNA and RNA. RNA salt dependence is scaled using the
    Tan & Chen 2007 ratio to account for the tighter A-form helix, while
    maintaining the Owczarzy 2004 baseline for DNA.

- **thermo:** Make salt and dangle corrections temperature-resolved

    Update salt corrections to be temperature-resolved and material-aware,
    applying entropic scaling and DNA/RNA specific coefficients. Also
    implement temperature resolution for dangles and terminal mismatches
    to ensure thermodynamic consistency across the temperature range.


## [0.4.0] - 2026-06-16

### Bug Fixes

- **thermo:** Fix Mg correction in Owczarzy Tm calculation

    Pass the actual sequence length to the magnesium correction functions
    instead of using a hardcoded value. This corrects an over-weighted
    length term that caused significant Tm errors for normal-length oligos.


### Documentation

- Document PyTorch-based differentiable thermodynamics and the new diff installation extra
- Document salt and temperature propagation in folding engine

    Add a detailed section explaining how sodium, magnesium, and celsius
    parameters propagate through the folding engine (MFE, pfunc, duplex).
    Clarify the distinctions between the per-bp salt correction and the
    whole-helix models used for Tm calculations, including provenance.


### Features

- **thermo:** Add magnesium support to duplex ΔG calculations

    Implement duplex_salt_dg using the √[Mg2+] combining rule to account
    for divalent cations. This aligns the two-state duplex stability
    model with the per-base-pair salt corrections used by the main
    folding engine.

- **thermo:** Add Owczarzy salt model for hairpin thermodynamics

    Implement the "owczarzy" salt model by grafting the GC-aware Owczarzy
    Tm shift onto the hairpin stem. This adds a new closed-state ΔG
    offset calculation that reproduces the Owczarzy ΔTm relative to the
    1 M Na+ reference.

- **thermo:** Add salt to native MFE and temperature-blended ParameterSet

    Thread [Na+]/[Mg2+] through fold_mfe (per-bp dg_per_bp_salt on each closed
    pair) and add a temperature-blended ParameterSet (DNA+RNA loop-init ΔH) via
    _param_override. 37 °C / 1 M Na+ / 0 Mg2+ stays bit-identical.


## [0.3.4] - 2026-06-14

### Bug Fixes

- Implement lazy loading for differentiable API and add optional torch dependency support

## [0.3.3] - 2026-06-13

### Features

- Add gradient-based differentiable sequence design with hybrid simulated-annealing polish.
- Implement Tan-Chen (2007) whole-helix salt correction as the default for stems ≥6 bp

## [0.3.2] - 2026-06-11

### Features

- **thermo:** Add DNA loop ΔH tables for temperature-resolved energetics

    Populate the native DNA ParameterSet's ΔH dict with real loop enthalpies
    (SantaLucia 2004 / Mathews 1999, via the UNAFold/primer3 `.dh` tables) instead
    of the previous "copied from ΔG" approximation, which made any ΔH/ΔS-derived
    quantity (e.g. a hairpin melting temperature) incorrect:

    - stack ΔH: unchanged (already SantaLucia; now cross-checked against the source)
    - loop-size ΔH = 0: loop initiation is purely entropic (per UNAFold loops.dh)
    - hairpin_mismatch (96/96) and interior_mismatch (59/96) from tstack / stackmm
    - triloop / tetraloop ΔH bonuses

    The mismatch mapping is validated: reconstructing ΔG from the primer3 tstack
    table reproduces strider's existing HAIRPIN_MISMATCH ΔG to ~0.05 kcal/mol,
    confirming the source and key convention are consistent (primer3 index
    [c5][m5][c3][m3] for strider mismatch key m3+c3+c5+m5).

- **thermo:** Structure-resolved ΔG/ΔH for hairpin Tm; unify hairpin API

    Adopt the structure-resolved ΔG/ΔH decomposition from #2 as the engine behind
    the unimolecular hairpin Tm, replacing hairpin.py's stack-only ΔH sum. ΔG and ΔH
    are now walked over the same per-element decomposition (stacks, bulges/interior
    loops, hairpin loop) against the real ΔH tables from #1, so:

      * the loop-closing terminal-mismatch enthalpy is now counted (matters for
        GC-rich stems; the stack-only sum dropped it), and
      * bulge/internal-loop hairpins are scored instead of raising ValueError.

    The validated 1 M seqfold anchor is unchanged (the new walk reproduces the old
    ΔH there exactly). new structure_thermo.py exposes the low-level
    structure_free_energy / structure_enthalpy used here and validated to reproduce
    fold_mfe energy exactly (scripts/validate_loop_decomposition.py: 1669/1669, 0
    error).

    Kept from the existing hairpin.py, in preference to #2's variant:
      * the per-base-pair ΔG salt model (salt.dg_per_bp_salt) over an Owczarzy Tm
        shift — Owczarzy is duplex-calibrated and over-corrects Mg for a
        unimolecular fold (~+12 °C at 10 mM Mg), and
      * fraction_folded + the HairpinThermo public API.

    The duplicate hairpin_thermo (and its Owczarzy path) in #2 is dropped so there is
    a single hairpin entry point.


## [0.3.1] - 2026-06-11

### Features

- Add hairpin melting temperature calculation and thermodynamics module
- Add hairpin thermodynamic functions and update test coverage badge

## [0.3.0] - 2026-06-11

### Features

- Implement StochasticSurfaceModel to estimate shot-noise-limited LOD using Currie statistics and mantis SSA
- Implement differentiable soft_forward for sequence-to-energy gradient optimization

### Refactor

- Simplify Tm calculation and add project branding assets to README

## [0.2.0] - 2026-06-08

### Bug Fixes

- Correct K_CURRIE constant to match the canonical Currie limit of 2.71 counts

### Documentation

- Refresh README for Workstream A–D APIs and new design optimiser

    Reflect the work that landed across the four workstreams:

    - Update front-matter blurb: zero external thermodynamic dependencies,
      swappable `ParameterSet`, defect-weighted parallel-tempered designer.
    - §8 (Sequence design): document `DefectWeightedPolicy` +
      `per_residue_defect_from_ensemble`, parallel tempering knobs,
      `DomainSpec.gc_band` early rejection, `ConstraintAwarePolicy` +
      `HardConstraint.propose`, `Assay.to_objective(equilibrium=True)`,
      `decompose_assays`, and the `scripts/bench_design.py` runner.
    - API reference: add `mutation_policy` / `parallel_tempering` /
      `n_chains` / `swap_every` to `SequenceDesigner.design`, document
      `MutationPolicy` variants, leaf-decomposition helpers, and the
      `HardConstraint.propose` contract.  Note the `equilibrium=True`
      branch of `Assay.to_objective`.
    - Refresh test-suite badge and counts (278 → 293 passed, plus the
      `slow` convergence target).  Add a row for the new
      `test_design_convergence.py` file.
    - Surface the new symbols in the Quick-start import block.

- Surface benchmark receipts in README + refresh test counts

    Add a "Receipts" panel at the top of the README so anyone scanning the
    landing page sees the structural F-measure (0.99), Zhang & Winfree TMSD
    round-trip (0% error), and MFE timing numbers without having to dig
    into the benchmark script.

    Also bump the test-count badge 293 → 310 (+14 benchmark tests) and the
    "all green" copy beneath the test-coverage table, and add a row for the
    new `tests/test_benchmarks.py` file.

- **design:** Add closed-loop example + document new objectives

    • examples/09_dynamical_design.py — wraps a single A + B <-> AB
        bridge as a `(seqs) -> mantis.CRNetwork` closure, uses
        `DesignObjective.kinetic_trajectory` to score the normalized MSE
        against a target step-response, and lets `SequenceDesigner` find a
        toehold whose simulated trajectory matches.  Each SA step rebuilds
        the CRN with the latest sequence and reruns the mantis ODE — the
        feedback loop is honest.

      • examples/dynamical_design.png — output of the example showing the
        target, baseline, and optimized [AB](t) trajectories plus a
        per-trial SA convergence bar chart.

      • README.md — new "Dynamical (closed-loop) objectives" table in §8
        documenting the five new factories, plus an example-09 entry and
        a test-badge bump (315 → 330).


### Features

- **thermo:** Loadable Turner-style ParameterSet (Workstream A)

    Add a swappable nearest-neighbor parameter system so user-supplied
    Turner-style JSONs can be loaded alongside the built-in MIT-licensed
    defaults.

    - `strider/thermo/parameters.py` — `ParameterSet` dataclass plus the
      loader API (`load_parameters`, `list_parameter_sets`, `param_search_paths`).
      Schema covers stack / hairpin / bulge / internal-loop / multiloop /
      dangle / coaxial / Ninio / terminal-penalty / terminal-mismatch tables.
    - `strider/thermo/parameters_native.py` — builds the MIT-clean `native-dna`
      and `native-rna` sets from SantaLucia 2004 + Mathews 1999 / Turner 2004
      primary tables (matches the existing module-level constants exactly).
    - `ThermoEngine.parameter_set` constructor argument with a lazy
      `engine.params` property; the parameter-set name is baked into the
      SHA-256 cache key so swapping sets invalidates memoised pfunc/MFE
      values automatically.
    - 23 tests covering native-adapter sanity, schema round-trip via a
      synthetic JSON in `tmp_path`, and engine integration.

    User-supplied JSONs can be dropped into `strider/thermo/parameters/` or
    discovered via `$STRIDER_PARAMS_DIR`.

- **structure:** Full Zuker MFE sharing energy code with pfunc (Workstream B)

    Replace the toy Zuker MFE with a complete Zuker–Stiegler dynamic program
    that shares its loop-energy functions with the McCaskill partition-function
    DP in `strider/thermo/ensemble.py`, so MFE energies and ensemble ΔG cannot
    drift apart.

    - `strider/structure/mfe.py` — full V[i,j] / W[i,j] / WM[i,j] / WM1[i,j]
      tables.  Hairpin / stack / internal-loop / bulge / multi-branch loop
      branches all wired up.  Traceback recovers a deterministic dot-bracket.
    - All energy lookups routed through the same `_hairpin_loop_energy` /
      `_stack_energy` / `_interior_bulge_energy` functions consumed by the
      ensemble DP.  Backward-compatible private API
      (`_can_pair_fn`, `_stack_fn`, `_hairpin_energy`, `_normalize`)
      preserved for `structure/pseudoknot.py`.
    - `scripts/bench_mfe.py` — pure-Python timing across sequence lengths plus
      optional ViennaRNA head-to-head if `import RNA` succeeds.
    - `tests/test_mfe.py` — 4 new `TestFullZukerEnergetics` cases (tetraloop
      hairpin, MFE ≤ ensemble ΔG consistency, internal-loop coverage,
      multi-loop coverage).

    Measured performance (single thread, pure Python): 100-nt MFE ~590 ms;
    the original plan target was 300 ms — closing the gap requires Numba
    JIT, which would add a build-time dependency.  Deferred.

- **tube:** Strand / Complex / Tube / ComplexSet API (Workstream C)

    User-facing multi-strand tube-analysis layer on top of the existing
    equilibrium solver.

    - `strider/tube.py` — `Strand`, `Complex`, `SetSpec`, `ComplexSet`,
      `Tube`, `TubeResult`, and the `tube_analysis(tubes, engine)` batch
      driver.  `Complex` supports both *resolved* construction (carrying
      `Strand` objects with sequences) and *name-only* construction
      (`Complex.from_names(...)`) for design-spec sites where sequences are
      not yet known.  Equality and hashing are canonical-strand-name based,
      so cyclic rotations of a homomer collapse to one entry.
    - `strider/equilibrium.py:equilibrium_from_engine` now delegates to
      `Tube.analyze` — backward-compatible signature, single canonical
      code path.
    - Post-C unification: `Assembly` refactored to compositionally wrap a
      name-only `Complex`.  The original positional signature
      `Assembly(name, strands, structure, concentration)` is preserved
      alongside a new `Assembly.from_complex(...)` classmethod.  One shared
      primitive ("a multi-strand complex"), two layered types: `Complex`
      for analysis (with sequences), `Assembly` for design-spec metadata.
    - `strider/__init__.py` — export the new types.
    - `tests/test_tube.py` — 29 tests covering Strand/Complex/SetSpec/
      ComplexSet semantics, `Tube.analyze` numerical agreement with the
      underlying solver, lazy `pair_probabilities` / `defect` access,
      duplicate-tube-name rejection, and `equilibrium_from_engine`
      backward compat.
    - `tests/test_assay.py` — 6 unification tests covering both Assembly
      construction paths and `Assembly.complex` ↔ Tube `Complex` equality.
    - `examples/08_tube_analysis.py` — runnable demo across dilute + dense
      tubes with side-by-side per-species output.

- **design:** Defect-based design optimizer (Workstream D)

    The defect-weighted, parallel-tempered sequence designer.  Brings
    ensemble-defect optimisation up to the level where 90%+ of trials on
    canonical hairpin/duplex tasks converge to the engine's physical
    defect floor.

    New modules
    - `strider/design/policies.py` — `MutationPolicy` strategy hierarchy
      (`RandomMutationPolicy`, `DefectWeightedPolicy`, `ConstraintAwarePolicy`)
      and the `per_residue_defect_from_ensemble(engine, names, target)`
      helper that turns the Zadeh-Wolfe-Pierce per-residue defect into a
      `defect_fn` ready for `DefectWeightedPolicy` to sample from.
    - `strider/design/decomposition.py` — `build_strand_graph`,
      `connected_components`, and `decompose_assays(...)` to split any
      `Assay` / `AssayPanel` into the smallest independent leaves so each
      can be optimised in isolation.
    - `strider/design/benchmarks.py` + `scripts/bench_design.py` — runner
      for the canonical hairpin-12 / hairpin-20 / duplex-12 design tasks.

    Refactors
    - `optimizer.py` — `SequenceDesigner.design(..., mutation_policy=...,
      parallel_tempering=True, n_chains=4, swap_every=20)`.  Adjacent-chain
      Metropolis swaps on a geometric `T_end → T_start` ladder; single-chain
      Monte Carlo otherwise.  `DomainSpec.gc_band` enables an early-rejection
      pre-check that drops out-of-band mutations before the (expensive)
      objective evaluation.  Constraint-feasible initial sequences via
      bounded rejection sampling.
    - `constraints.py` — `HardConstraint.proposer` field + `HardConstraint.propose(
      name, seq, pos, rng, bases) -> base | None`.  Default behaviour is
      generic reject-resample; an explicit proposer can short-circuit it.
    - `objective.py` — `DesignObjective.ensemble_defect_tube(engine,
      tube_factory, on_targets, ...)` factory.  Each evaluation builds a
      fresh `Tube`, runs `Tube.analyze`, and weights every on-target's
      defect by its real equilibrium concentration
      (Wolfe & Pierce 2015 §2.2).
    - `assay.py` — `Assay.to_objective(engine, equilibrium=True)` weights
      on-target defects by the post-equilibrium concentration from a
      `Tube` solve over the union of all declared assemblies, instead of
      by the static `Assembly.concentration`.

    Tests + infra
    - `tests/test_design_convergence.py` — 15 fast tests covering every
      new module + 1 `@pytest.mark.slow` convergence target (10/10 trials
      hit defect ≤ 0.10 on the 12-nt hairpin task at 5 000 iterations).
    - `pyproject.toml` — register the `slow` marker and deselect by default.
    - `strider/__init__.py` — export the new policies + decomposition API.

    The original plan's "< 1e-3 defect" target is unreachable on the
    native McCaskill DP (hand-tuned 12-nt hairpins floor at defect ~0.06).
    The slow test uses the physically realistic threshold and is honest
    about why in its docstring.

- **thermo:** Thread ParameterSet into the energy DP (Workstream A follow-up)

    Make a user-supplied `parameter_set=` argument to `ThermoEngine`
    actually change `pfunc` / `mfe` / sampling output, closing the honesty
    gap that was flagged in the Workstream A summary.

    Design — keep the default path bit-identical
    - `strider/thermo/_param_context.py` — tiny `contextvars`-backed
      override channel.  `param_context(ps)` activates a `ParameterSet`
      for the duration of a `with` block; `lookup_table(name, fallback)`
      and `lookup_scalar(name, fallback)` are the helpers the energy
      functions call.  When no override is active, both helpers return
      the supplied module-constant fallback — zero allocation, zero
      branching cost on the hot path.
    - `ThermoEngine._uses_custom_params()` returns False for
      `parameter_set=None` and for the built-in `"native"` / `"native-dna"`
      / `"native-rna"` aliases, so default callers never enter the
      override context.  Only an explicit non-native `ParameterSet`
      instance (or a non-native name) flips the switch.

    Threaded sub-tables (the ones the native adapter exposes)
    - `stack`, `terminal_penalty`, `hairpin_size`, `bulge_size`,
      `interior_size`, `asymmetry_ninio`, `log_loop_penalty`,
      `multiloop_init`, `multiloop_pair`, `multiloop_base`.
    - Sites updated: `ensemble.py:_hairpin_loop_energy`, `_stack_energy`,
      `_interior_bulge_energy`, `_terminal_pair_penalty`,
      `_fill_dp_nicks` (multiloop coefficients);
      `mfe.py:_multiloop_params`;
      `sampling.py:_sample_Qb` / `_sample_QM` (multiloop coefficients).
    - Advanced DNA-specific tables not yet exposed by `ParameterSet`
      (COAXIAL_STACK, STK_BARE_FACTOR, STK_D5/D3/TM_DELTA, INTERIOR_1_1/
      1_2/2_2, HAIRPIN_TRILOOP/TETRALOOP, INTERIOR/TERMINAL_MISMATCH)
      continue to read module constants unconditionally.

    Tests
    - `TestCustomParamsAffectNumerics` (3 new tests, 296 total):
      doubled stack ΔG drives `pfunc` more negative; a 100 kcal/mol
      multi-loop init pin suppresses multi-branch MFEs; the `"native-dna"`
      alias produces bit-identical output to the no-paramset default.

- **benchmarks:** Head-to-head accuracy + timing receipts

    Add `strider/benchmarks/` and `scripts/bench_accuracy.py` so the
    "competes with NUPACK on the canonical thermodynamic working set"
    claim is backed by concrete numbers rather than assertion.

    What the suite produces
    - **Structure prediction accuracy** — mean sensitivity 1.00, mean PPV
      0.98, mean F-measure 0.99, 10/11 exact matches across 11 canonical
      hairpins from primary literature (Cheong 1990, Heus & Pardi 1991,
      Antao 1991, Mathews 1999, SantaLucia 2004, Lu 2006).  When
      ViennaRNA is installed, the runner also reports mean
      |ΔG_native − ΔG_vienna| side-by-side.
    - **TMSD kinetics** — `toehold_kf` lookup round-trips Zhang & Winfree
      2009 Fig. 4 at 0% relative error across all 13 toehold lengths;
      Arrhenius extrapolation to 37 °C is strictly monotonic in toehold
      length.
    - **Wall-clock timing** — MFE / pfunc median + p95 across a length
      sweep with optional ViennaRNA head-to-head.

    Modules
    - `strider/benchmarks/structure_refs.py` — `StructureRef` dataclass +
      11 canonical hairpins with citations.  No third-party dataset is
      bundled; every entry is a primary-literature example.
    - `strider/benchmarks/accuracy.py` — sensitivity / PPV / F-measure
      (the Mathews-2004 convention), dot-bracket comparison helper,
      relative-error helpers.
    - `strider/benchmarks/runners.py` — `run_structure_benchmark`,
      `run_tmsd_benchmark`, `run_timing_benchmark` plus structured
      report dataclasses.
    - `scripts/bench_accuracy.py` — CLI: `--section all|structure|tmsd|timing`,
      `--lengths`, `--reps`, `--no-vienna`, `--material`.

    Tests (+14 fast, +3 slow)
    - `tests/test_benchmarks.py` exercises every metric helper, validates
      the reference list, asserts the runner outputs meet the published
      thresholds (mean F ≥ 0.95, max TMSD rel-err = 0, timing rows
      positive-finite).  Slow-marked tests gate the full structure /
      timing sweeps.

- **thermo:** Expand ParameterSet schema to every advanced sub-table

    Followup to commit a9ecac7 — extend the threadable parameter surface
    from the basic 7 tables up to the full 22 sub-tables consumed by the
    energy DP, so a user-supplied custom JSON can override every numerical
    input strider's MFE / pfunc / sampling code reads.

    Schema additions (universal — DNA + RNA)
    - `dangle_3`, `dangle_5`
    - `terminal_mismatch`
    - `hairpin_triloop`, `hairpin_tetraloop`

    Schema additions (DNA-specific, where the bundled tables exist)
    - `hairpin_mismatch`
    - `interior_mismatch`
    - `interior_1_1`, `interior_1_2`, `interior_2_2`
    - `coaxial_stack`

    The precomputed Boltzmann-factor tables (`STK_BARE_FACTOR`,
    `STK_D5/D3/TM_DELTA`) deliberately stay out of the schema — they're
    temperature-dependent derived quantities computed from DANGLE_* +
    TERMINAL_MISMATCH at 37 °C and would need regeneration when those
    upstream tables change.  Documented in the §11 comparison doc.

    Threaded sites in `ensemble.py`
    - `_hairpin_loop_energy`: hairpin_triloop / tetraloop + DNA hairpin_mismatch
      / RNA terminal_mismatch first-mismatch override.
    - `_interior_bulge_energy`: interior_mismatch / 1_1 / 1_2 / 2_2 (DNA).
    - `_fill_dp_nicks`: dangle_5 / dangle_3 in the external-loop branch.
    - `_apply_coaxial_external`: coaxial_stack (DNA only).

    Other work
    - `strider/thermo/parameters_native.py` — populate every new sub-table
      from the existing module constants so the in-memory native paramset
      remains complete.
    - `strider/thermo/parameters.py` — accessor methods (`dangle_5`,
      `terminal_mismatch`, `interior_1_1`, ...) returning `None` when the
      set doesn't define the table.
    - `scripts/export_paramset.py` — CLI that exports any ParameterSet
      (native or user-supplied) to a Turner-style JSON file users can edit
      in place; demonstrates the round-trip and is the recommended starting
      point for a custom set.
    - `strider/thermo/parameters/dna-low-salt-50mM-Na.json` — first bundled
      curated alternative: a 771-byte partial-override JSON shifting every
      stack ΔG by +0.115 kcal/mol (Owczarzy 2004 salt correction from
      0.137 → 0.05 M Na⁺).  Demonstrates that partial overrides work — the
      loader honors the `stack` entry and every other sub-table falls back
      to the native default.

    Tests (+8 in `TestAdvancedTableOverrides`; 315 total)
    - Each test perturbs one of the newly-threaded sub-tables and asserts
      the change shows up in `pfunc` output.
    - One end-to-end smoke verifies the bundled `dna-low-salt` JSON loads
      and shifts hairpin ΔG by at least 0.2 kcal/mol.

    Docs
    - README §15 fully rewritten with the new 22-row schema table, the
      exporter workflow, and a note on the override gating (default /
      native paths stay bit-identical to every prior release).

- **design:** Add closed-loop dynamical objective factories

    Five new factory methods on `DesignObjective` that drive sequence
    optimization from a kinetic cost (ODE simulation or bifurcation scan)
    instead of a static equilibrium defect.  Each accepts a
    `network_factory: (seqs) -> mantis.CRNetwork` closure — typically a
    bound `CircuitBridge.to_crnetwork` — so every score evaluation rebuilds
    the CRN with the latest rate constants.

      • kinetic_trajectory  — normalized MSE vs a target [species](t) curve
      • maximize_kcat       — -d[species]/dt, scale-normalized
      • minimize_leak       — (log10(leak/threshold))² for no-trigger sims
      • bistable_threshold  — log-deviation at the bifurcation midpoint
      • from_simulation     — escape hatch: arbitrary cost_fn(SimulationResult)

    They compose with the existing static factories (gc_content, ddg_target,
    ensemble_defect_tube, …) under the same weighted-sum protocol.

    Implements item 1 of outperform_nupack.md (Closed-Loop Dynamical Sequence
    Design).

- Implement differentiable PyTorch thermodynamics engine for trainable McCaskill partition functions and add associated tests
- Add surface transducer model for signal prediction and implement associated structure sampling logic
- **kinetics:** Template-free domain-level reaction enumerator (Peppercorn paradigm)

    Add DomainReactionEnumerator (strider/kinetics/enumerator.py): reads a circuit's
    strand topology in domain space and *derives* the reaction network that DSDCompiler
    deliberately left to the user — the Visual DSD / Peppercorn job. It enumerates the
    reachable complexes plus the transitions between them:

      - bind    : a complementary toehold pair (one per complex) hybridises and merges
                  the two complexes (bimolecular; long/blunt-end binding gated behind
                  include_leak)
      - migrate : 3-way branch migration — an unbound domain adjacent to a junction
                  displaces an identical incumbent, splitting off any freed strand
      - open    : a toehold-length helix dissociates (long helices are irreversible —
                  the lever that keeps enumeration finite)

    Rates use the real thermodynamics: forward kf from Zhang-Winfree (toehold_kf /
    displacement_kf), reverse kr = kf*exp(dG/RT) from the active ThermoEngine's helix
    dG, so Keq = kf/kr = exp(-dG/RT) holds exactly. EnumerationResult.to_crnetwork()
    emits a simulable mantis.CRNetwork.

    Complexes are hashed by a permutation-canonical signature (same physical complex =
    one node); a planarity check forbids pseudoknotted bonds; max_complexes/max_strands
    caps guarantee termination on polymerising motifs (strict=True raises instead of
    truncating). Scope v1: non-pseudoknotted, 3-way only — 4-way migration and
    intramolecular hairpin re-closure are out.

    Validated end-to-end on the canonical 4-nt-toehold TMSD network (bind -> branch
    migration -> output release -> ODE sim releases ~65% in 1 h). New tests
    tests/test_enumerator.py and examples/10_domain_enumeration.py; exported from
    strider and documented in README §12. Closes Tier-3 §4 / Frontier §5 core in
    STRIDER_VS_NUPACK.md.

    Also bundles prior, separately-completed and tested work present in the tree:
    multi-strand pair-probability nick-junction exactness (thermo/ensemble.py) and the
    multi-tube / multistate ensemble-defect design objective (tube.py, design/objective.py)
    with their tests.

    Full suite: 388 passed, 1 skipped (torch absent), 4 deselected.

- **thermo:** Align differentiable McCaskill + batched/GPU backend (native speed)

    Close the differentiable engine's ~3-4 kcal/mol residual against the native
    engine and promote it into a fast batched/GPU backend (Tier-2 §3 / Frontier §4
    of STRIDER_VS_NUPACK.md).

    Root cause of the residual: stacks were keyed on the 2-base dinucleotide (the
    16-entry Watson-Crick RNA_NN table), so any helix containing a GU/UG wobble was
    scored with WC stack energies -> systematic over-stabilization (up to 4.5
    kcal/mol). Fixes:

      - full 36-entry stack table: a learnable 256-slot tensor populated from STACK,
        keyed by all four bases of the step (5'-(i)(i+1)-3' / 3'-(j)(j-1)-5'),
        exactly matching ensemble._stack_energy -- covers all six pair types incl.
        wobble, not just the 16 WC dinucleotides
      - single-base-bulge stack-across term (ensemble._interior_bulge_energy n==1),
        previously missing

    Residual on random sequences: ~0.72 mean / 4.54 max -> 0.28 mean / 1.04 max
    (RNA) and 0.25 / 0.77 (DNA); the worst prior case went -4.54 -> -0.35 kcal/mol.

    Batched/GPU backend: vectorized the host-side setup (base_indices, can_pair_mask
    via a 4x4 pair LUT, hairpin_bonus) to remove the per-cell Python/sync overhead
    that was bottlenecking the GPU. The batched DP now runs ~9-12x faster than the
    pure-Python native engine (L=60 B=64: 11.9x CPU / 11.4x CUDA; L=100 B=32: 12.3x
    CUDA) -- closing most of native's gap to a C kernel while staying learnable. New
    public helper differentiable.batched_free_energy(seqs, material, device=...),
    kept out of the eager package import so torch stays optional.

    stack_dG37 (16) is renamed stack_table (256); train.py and the differentiable
    tests updated. tests/test_differentiable.py extended with a native-agreement
    gate (RNA+DNA), GU-wobble regression, full-stack-table presence, helper/model
    equivalence, and CUDA<->CPU consistency (10 pass).

    Full suite: 398 passed, 1 skipped (unrelated screener case), 4 deselected.

- Implement G-quadruplex folding thermodynamics and motif identification

### Testing

- **design:** Cover dynamical objective factories

    Adds tests/test_design_dynamical.py with 10 tests built on a synthetic
    CRN whose rates depend on GC content of dummy strands — so each
    factory's response to sequence change can be verified without paying
    the full ThermoEngine cost.


## [0.1.0] - 2026-05-18

### Features

- Initialize Strider DNA/RNA design and thermodynamic simulation framework
- Add equilibrium concentration solver, differentiable thermo parameters, and accuracy benchmarking tools
- Implement circuit design framework with CHA and Seesaw templates and add associated verification tools and CLI.

### Refactor

- Restructure documentation by consolidating guides into index and generating a centralized API reference file
- Update thermo energy calculations to use multi-strand partition functions and improve off-target screening index and logic.


