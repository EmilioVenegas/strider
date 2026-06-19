# Known limitations

strider is an open, auditable, differentiable thermodynamics + kinetics + circuit stack. It is
**not** a drop-in NUPACK/ViennaRNA replacement, and we state the boundaries plainly so you can
judge fit. These fall into two groups.

## strider-specific (where the established tools are ahead)

- **~0.9 kcal/mol mean RNA ΔΔG vs NUPACK.** On RNA folding the native parameter lineage
  (Turner/Mathews-derived, re-built from primary literature) differs from NUPACK's `rna06` set by
  a mean ~0.9 kcal/mol in ensemble ΔG. Topology (which pairs form) agrees well; absolute free
  energies carry this offset. Use NUPACK/ViennaRNA when sub-kcal RNA accuracy is the priority.
- **Speed.** The native engine is pure-Python O(n³) DP with no JIT — roughly **~970× slower than
  NUPACK's C kernel** on single sequences (~4 ms at 20 nt, ~1 s at 100 nt for pfunc + pair probs).
  For long sequences (> ~200 nt) use the optional `vienna` backend, the GPU-batched
  `thermo.differentiable` path, or reserve native for screening/design at < 100 nt.
- **~13 °C hairpin-Tm offset.** Predicted molecular-beacon hairpin melting temperatures run ~13 °C
  below an experimental qPCR panel. The leading hypothesis is **end-stacking of the fluorophore/
  quencher labels** (an effect outside the bare-sequence NN model), not a parameter bug; this is
  flagged and tracked, not yet closed. Treat hairpin **Tm** as approximate; ΔG and *relative*
  comparisons are unaffected.

## Shared with NUPACK/ViennaRNA (modeling choices common to NN-based tools, not strider bugs)

- **ΔCp = 0 → linear ΔG(T).** Heat capacity change on folding is taken as zero, so ΔG(T) is linear
  (ΔG₃₇ blended with ΔH toward the enthalpy limit). Accurate near 37 °C; the linear extrapolation
  degrades at temperature extremes far from the reference.
- **Two-state hairpin/dimer thermodynamics.** Melt curves and Tm use a two-state (folded ⇄
  unfolded) approximation per hairpin/dimer; multi-state intermediates are not modeled in the Tm
  path.
- **Restricted / off-by-default pseudoknots.** The DP covers nested secondary structure and a
  restricted H-type pseudoknot class only; general pseudoknots are out of scope and disabled by
  default.
- **Multi-strand MFE: exact order-invariance only up to small complexes.** A linear DP can only
  represent structures non-crossing for one strand concatenation, so a complex's MFE is intrinsically
  strand-order-dependent (Dirks et al. 2007). `engine.mfe` removes this artifact by searching strand
  arrangements: it is **exactly order-invariant** for complexes small enough to fold every distinct
  cut (dimers, trimers, small 4-strand — within a length-scaled fold budget). Larger complexes use a
  sequence-affinity + crossing-minimisation **heuristic** (a few folds, never worse than the input
  order) that is order-invariant in practice but not guaranteed to find the global optimum for large
  fused networks (e.g. a 6-strand dendrimer). The partition function (`engine.pfunc`) is **not yet
  order-invariant** — it still folds the supplied order.
- **Multi-strand absolute-ΔG offset vs NUPACK (a convention difference, *not* ensemble breadth).**
  strider applies the per-association `(L−1)·ΔG_assoc` term (DNA 1.96 / RNA 4.09), the same one
  NUPACK includes in its complex free energy (verified: zeroing NUPACK's `join_penalty` shifts its
  complex ΔG by exactly `(L−1)·1.96`), plus a coaxial-stacking correction at flush strand nicks.
  The dominant residual vs NUPACK was previously characterised here as "ensemble breadth"; closer
  analysis (`STRIDER_VS_NUPACK.md` Part III·K and `scratch/probe_*`) shows it is instead a
  **sequence- and length-independent per-helix-terminus free-energy offset**: NUPACK assigns each
  helix terminus an extra **≈ −1.24 kcal/mol** (≈ −2.47 per blunt duplex, *exactly* constant across
  all four closing-pair identities and every tested length). The evidence that it is **structural,
  not partition-function broadening**: (i) it is present in NUPACK's **`nostacking`** ensemble and
  in its MFE *structure* energy, not only its pfunc; (ii) it does **not** cancel in **binding** ΔG
  (`G(complex) − ΣG(strand)` still differs ~2.5–4 kcal/mol per duplex); (iii) it is exactly
  proportional to a structural count (exterior-facing helix termini), with no sequence dependence
  beyond the standard +0.05 terminal-AT penalty. strider's native backend follows the
  **SantaLucia 1998/2004** nearest-neighbour initiation (≈ +1.96 per duplex), the experimentally-fit,
  IDT/qPCR-aligned convention; **ViennaRNA** sits at an *intermediate* offset (~ −1.0 per duplex,
  not −2.47), so the three tools disagree among themselves — this is a parameterisation /
  reference-state convention for terminal-initiation free energy, **not** a universal physical term
  strider lacks. Matching NUPACK's *absolute* complex (and binding) ΔG would require **adding** the
  ≈ −1.24/terminus offset, which would move strider's duplex ΔG ~2.4 kcal/mol *away* from the
  SantaLucia experimental baseline. A genuinely separate, much smaller component (~ −0.6 kcal/mol
  per duplex) is real **ensemble breadth**: NUPACK's `stacking` dangle/coaxial sub-ensemble broadens
  its pfunc below its own `nostacking` value (−0.77/dimer) more than strider's leading-order model
  does (−0.17/dimer). Single-strand folding (hairpins, internal loops, nested structures) matches
  NUPACK's `stacking` ensemble to ≈0.05–0.4 kcal/mol — strider's hairpin/loop energies already
  absorb the per-terminus term for the single exterior terminus of a folded strand; the deficit
  surfaces only for the *extra* exterior termini that multi-strandedness (nicks) introduces. The
  coaxial-junction part is separately recovered (a flush coaxial junction contributes ≈ −2 kcal/mol
  that strider's nick-aware DP omits). See `STRIDER_VS_NUPACK.md`.

For the divalent-cation regime (Na⁺×Mg²⁺×T) strider is, if anything, *ahead* of both tools —
neither NUPACK nor ViennaRNA models Mg²⁺.
