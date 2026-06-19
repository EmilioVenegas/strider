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
- **No bimolecular association penalty (ΔG_assoc).** strider does not apply the per-association
  (`(L−1)·ΔG_assoc`) term that NUPACK includes in complex free energies, so multi-strand ΔG is on a
  different additive reference than NUPACK's (it omits the association/nucleation cost). See
  `STRIDER_VS_NUPACK.md`.

For the divalent-cation regime (Na⁺×Mg²⁺×T) strider is, if anything, *ahead* of both tools —
neither NUPACK nor ViennaRNA models Mg²⁺.
