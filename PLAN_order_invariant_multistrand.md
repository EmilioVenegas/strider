# Plan: Order-Invariant Multi-Strand Folding (B) + Thermodynamic Parity (C)

Status: **proposed** · Owner: TBD · Created: 2026-06-18

## 1. Problem

The native folding backend runs a **linear** Zuker/McCaskill DP on the strands
*concatenated in the order the caller supplied* (`strider/structure/mfe.py:37-38`,
`strider/thermo/ensemble.py:6`). The set of structures a linear DP can represent is
exactly those that are **non-crossing for that one cyclic strand order**. Crossing
(interleaving) of helices is an intrinsic property of the cyclic order, so the
predicted MFE/ensemble of a complex **changes if you relabel the strand order** —
even though the real equilibrium of a complex depends only on *which strands are
present*, not how a human writes them down.

This is why the dendrimer example (`examples/21_gallery_dendrimer.py:90-97`) has to
apologise for needing "ring order": in any other concatenation its junction helices
interleave on the line and the DP leaves the loop half-open. **The order-dependence
is an algorithmic artifact, not physics**, and we want to remove it.

Reference for the correct treatment: **Dirks, Bois, Schaeffer, Winfree & Pierce,
"Thermodynamic Analysis of Interacting Nucleic Acid Strands," SIAM Review 49(1):65–88
(2007)** — the *ordered-complex* (circular) formulation, the `(L−1)·ΔG_assoc`
association penalty, and the rotational-symmetry correction σ. Everything below is to
be re-derived from this primary source, **not** copied from NUPACK (see
`strider-nupack-licensing` constraint — no NUPACK-derived data/strings).

## 2. Current state (what already exists — audit before building)

| Piece | Where | State |
|---|---|---|
| Linear nick-aware MFE DP | `structure/mfe.py` `_build_mfe_matrices` | ✅ correct for one cyclic order; **no** assoc penalty at the W-nick split (`mfe.py:201`) |
| Linear nick-aware McCaskill DP | `thermo/ensemble.py` `_multistrand_dg`, `multistrand_pairs` | ✅ correct for one cyclic order |
| `JOIN_PENALTY` constant | `parameters_dna.py:84` (1.96), `parameters_rna.py:52` (4.09) | ⚠️ referenced only in a **docstring** (`ensemble.py:937`); **grep shows it is not actually applied** in either DP — audit & confirm |
| σ rotational-symmetry correction | `thermo/engine.py:415-419, 499-501, 548` | ✅ applied to native+vienna MFE and native pfunc via `cyclic_symmetry` |
| `cyclic_symmetry`, σ, rotation-invariant `Complex.canonical_name` | `equilibrium.py:66`, `tube.py:160-211` | ✅ exists — reuse for necklace dedup |
| Concentration solver w/ `(n−1)·log c0` translational term | `equilibrium.py:164` | ✅ consumes per-complex ΔG |

**Key takeaway:** the symmetry + concentration bookkeeping is largely in place; the
two real gaps are (B) the engine only folds *one* order, and (C) the association
penalty is inconsistent/missing and unverified.

## 3. Goals & success criteria

1. **Order invariance.** For any permutation of the input strands, `engine.mfe(*seqs)`
   and `engine.pfunc(*seqs)` return the same ΔG (and the same structure up to strand
   relabeling), to numerical precision. *Property test, not anecdote.*
2. **Thermodynamic parity.** Multi-strand ΔG matches the Dirks 2007 definition:
   per-association penalty + σ correction, consistent between MFE and pfunc. Validate
   against the existing NUPACK receipts (`paper/receipts/*vs_nupack*`).
3. **No regression.** Single-strand and 2-strand (dimer) results, and all existing
   `tube`/concentration numbers, are unchanged at the 1-strand / canonical-order case.
4. **`draw_complex` stops depending on hand-supplied ring order** (it folds via the
   engine, so it inherits the fix for free).

## 4. Part B — Circular / ordered-complex DP

### 4.1 Theory (the honest version)

A pseudoknot-free complex is, by definition, non-crossing for **some** cyclic strand
order. But **no single order captures all structures simultaneously** — different
candidate structures need different orders — so the rigorous object is the
**ordered complex**: each distinct cyclic order (necklace, modulo rotation *and*
reflection) is folded on its own, and the results are combined. This is the Dirks 2007
"circular" formulation. Within a *fixed* cyclic order the existing linear DP is already
a valid cut of the circle (cutting a connected complex at any strand break yields the
same non-crossing set), so **the per-order kernel is the DP we already have** — the new
work is enumeration + combination + connectivity, not a new recurrence.

> Note: my earlier "make pairs span the seam" framing was imprecise — the seam is just
> part of the exterior loop and the linear DP already handles it. The genuine fix is
> ordered-complex enumeration, below.

### 4.2 Algorithm

```
fold_complex(strands):
    orders = distinct_cyclic_orders(strands)        # necklaces mod rotation+reflection
    best = +inf
    for order in orders:
        struct, dG = linear_DP(concat(order))       # existing nick-aware DP
        if not connected(struct, order): continue   # reject disconnected species
        dG += assoc_and_symmetry_corrections(order)  # Part C
        best = min(best, (dG, struct, order))
    return best                                      # MFE
# pfunc: Z_complex = Σ_orders Z(order) / σ(order), then dG = -RT ln Z_complex
```

### 4.3 Tasks

- **B1.** `distinct_cyclic_orders(strands)` — enumerate necklaces modulo rotation and
  reflection. Reuse `Complex.canonical_name` / `cyclic_symmetry` for dedup. Add a
  `max_strands_per_complex` cap (config, default e.g. 8) — `(L−1)!/2` blows up
  (L=8 → 2520 orders); refuse or warn beyond the cap.
- **B2.** **Connectivity constraint.** The linear DP will happily leave a strand
  unpaired (that is a *different, lower-order* species, e.g. complex + free monomer).
  A defined ordered complex must be **connected**: every strand reachable through the
  base-pair graph. Add a post-fold connectivity check (reject) — or, better, a DP-level
  constraint so the partition sum only counts connected structures. Start with the
  post-fold reject (simple, correct for MFE); evaluate the DP-level version for pfunc
  exactness.
- **B3.** New module `strider/structure/complex_fold.py` (or extend `mfe.py`) exposing
  `fold_complex(strands, ...) -> (structure, dG, order)` and a pfunc analogue, wrapping
  the per-order kernels. Keep single-strand and 2-strand on the existing fast path.
- **B4.** **Pruning / performance.** Rank candidate orders by `geometry.crossing_count`
  / `best_strand_order` (already in `viz/geometry.py`) and fold low-crossing orders
  first; cache per-order folds; allow early-exit once the remaining orders cannot beat
  the incumbent (loose energy bound). Parallelize independent folds.
- **B5.** Wire into `engine._mfe_native` / `_pfunc_native_inner` (`engine.py:430-504`)
  behind a flag (`order_search=True` default for ≥3 strands; ≤2 strands unchanged).
  `draw_complex` then inherits it and `reorder=` becomes a no-op fallback.

## 5. Part C — Thermodynamic parity (association + symmetry)

- **C1. Audit `JOIN_PENALTY`.** grep shows it is referenced only in a docstring
  (`ensemble.py:937`) and "intentionally omitted" elsewhere (`structure_thermo.py:160`,
  `dimer_thermo.py:81`); confirm whether the bimolecular association is applied *at all*
  in `_multistrand_dg`, and where. Document the ground truth before changing anything.
- **C2. Apply association consistently.** Charge `ΔG_assoc` **once per inter-strand
  association** (`(L−1)` times for a connected L-strand complex, Dirks 2007) in **both**:
  - native MFE: add at the W-nick split (`mfe.py:201`) and in traceback (`mfe.py:243-249`);
  - native pfunc: as a Boltzmann factor at the same nick split in `_fill_dp_nicks`.
  Single source the constant from `parameters_{dna,rna}.JOIN_PENALTY`. This makes MFE
  and pfunc mutually consistent (they currently share loop energies but not this term).
- **C3. σ correction × ordered-complex enumeration — avoid double counting.** This is
  the main correctness risk. The necklace enumeration (B1) already accounts for
  *distinct* arrangements; σ corrects for *rotational symmetry within* a homomeric
  ordered complex. Derive the combined bookkeeping from Dirks 2007 eqs. 10–11 and write
  it down explicitly. Today σ is added unconditionally in `engine.py`; re-home it so it
  composes with B and is applied exactly once.
- **C4. Concentration path.** Confirm `equilibrium.solve_equilibrium` (`equilibrium.py:164`)
  consumes the corrected per-complex ΔG and that the `(n−1)·log c0` translational term
  is not double-counting the association penalty (they are distinct: ΔG_assoc is a
  sequence-independent nucleation cost; the log-c0 term is the ideal-solution entropy).

## 6. Validation

- **Order-invariance property test:** for a panel of complexes (dimer, 3-way junction,
  the dendrimer core ring, a homomeric trimer), assert `mfe`/`pfunc` are invariant to
  `random.shuffle(strands)` within tolerance. This is the headline regression guard.
- **NUPACK parity:** extend the multi-strand rows of `paper/receipts/*vs_nupack*` and
  `scripts/bench_vs_nupack.py`; target agreement within the existing dimer tolerance.
- **No-regression:** single-strand ΔG, dimer `ddg`/`dimer_tm`, and `tube_analysis`
  concentrations unchanged on the canonical order. Run `tests/` + the viz suite.
- **Dendrimer:** `engine.mfe` of the dendrimer must now reach 9/9 arms / dG≈−144 in
  *any* strand order, not just ring order — then drop the apology comment in example 21.

## 7. Risks & scope

- **Combinatorial blow-up** — capped by `max_strands_per_complex`; document the cap as a
  limitation (NUPACK caps complex size too). Pruning (B4) mitigates typical cases.
- **σ / assoc double-counting** (C3) — the subtle correctness trap; gate on the
  order-invariance + NUPACK-parity tests.
- **Connectivity for pfunc** — post-fold reject is exact for MFE but only approximate
  for the partition sum; a DP-level connected-ensemble constraint may be needed for
  exact multi-strand pfunc. Stage it (MFE first).
- **Out of scope:** genuinely cyclic bond graphs that cross on *every* order are true
  pseudoknots — unrepresentable by any nested method (NUPACK included); they remain the
  separate `strider/structure/pseudoknot.py` track. Note in `docs/limitations.md`.

## 8. Suggested sequencing

1. C1 audit (know the ground truth) → C2 apply assoc in MFE+pfunc + consistency tests.
2. B1+B2+B3 ordered-complex MFE (reject-disconnected) + order-invariance test.
3. C3 σ composition + NUPACK parity.
4. B4 performance/pruning; B5 engine + `draw_complex` wiring; drop example-21 apology.
5. pfunc ordered-complex sum (B2 DP-level connectivity if parity needs it).
6. Cross-link this plan from `STRIDER_VS_NUPACK.md` (roadmap source of truth) and
   update `docs/limitations.md`.

## 9. Files likely touched

`structure/mfe.py` · `structure/complex_fold.py` (new) · `thermo/ensemble.py` ·
`thermo/engine.py` · `equilibrium.py` · `tube.py` · `parameters_{dna,rna}.py` (constants) ·
`examples/21_gallery_dendrimer.py` (drop apology) · `scripts/bench_vs_nupack.py` ·
`docs/limitations.md` · tests under `tests/`.
