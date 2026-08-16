# Third-Party Notices

Strider is released under the MIT License (see `LICENSE`). This document records
the provenance of every external scientific parameter, dataset, and tool that
strider consumes, and states plainly what is — and is not — redistributed in
this package.

## Summary

- **Everything strider ships is either original code or numeric thermodynamic
  constants drawn from primary peer-reviewed literature.** Individual measured
  physical constants (ΔG, ΔH, ΔS, salt coefficients) are facts, not
  copyrightable expression; they are re-keyed into strider's own MIT-licensed
  modules and cited to the original measurement.
- **No NUPACK code, data, or parameters are redistributed.** NUPACK is used
  only as an external comparison baseline in benchmarks. References to "NUPACK"
  in the source tree are comparative prose in docstrings and comments only.
- **ViennaRNA's `rna_turner2004.par` is *not* redistributed.** It is a
  build-time input, fetched on demand, used only to transcribe Turner-2004 RNA
  enthalpies into strider's own table module. It is `.gitignore`d (the whole
  `data/` directory is ignored) and absent from the source and wheel.

---

## 1. Thermodynamic nearest-neighbor parameters (shipped — primary literature)

These values live in strider's own MIT-licensed modules
(`strider/thermo/parameters_dna.py`, `parameters_rna.py`, `parameters_native.py`,
`_dna_enthalpy_generated.py`, `_rna_enthalpy_generated.py`). They are physical
measurements reproduced from the original publications.

### DNA (37 °C, 1 M NaCl) — SantaLucia & Hicks 2004 unified set

- SantaLucia J. & Hicks D. (2004). *The thermodynamics of DNA structural
  motifs.* Annu. Rev. Biophys. Biomol. Struct. **33**:415-440. (Unified set;
  Table 1 consolidates the works below.)
- SantaLucia J. (1998). PNAS **95**:1460-1465. (Watson–Crick stacks; terminal penalty.)
- Allawi H.T. & SantaLucia J. (1997). Biochemistry **36**:10581-10594. (Stacks.)
- Peyret N., Seneviratne P.A., Allawi H.T., SantaLucia J. (1999).
  Biochemistry **38**:3468-3477. (Mismatches.)
- Allawi H.T. & SantaLucia J. (1998). Biochemistry **37**:9435-9444. (1×1 interior loops.)
- Allawi H.T. & SantaLucia J. (1998). Biochemistry **37**:2170-2179. (1×2, 2×2 interior loops.)
- Bommarito S., Peyret N., SantaLucia J. (2000). Nucleic Acids Res.
  **28**:1929-1934. (Dangle and terminal-mismatch ΔG and ΔH.)

### RNA (37 °C, 1 M NaCl) — Turner 2004 / Mathews lineage

- Mathews D.H., Sabina J., Zuker M., Turner D.H. (1999). J. Mol. Biol.
  **288**:911-940. (Expanded sequence dependence; loop-size penalties.)
- Mathews D.H., Disney M.D., Childs J.L., Schroeder S.J., Zuker M.,
  Turner D.H. (2004). PNAS **101**:7287-7292.
- Turner D.H. & Mathews D.H. (2010). *NNDB: the nearest neighbor parameter
  database.* Nucleic Acids Res. **38**:D280-D282.
- Schroeder S.J. & Turner D.H. (2000). Biochemistry **39**:9257-9274.
  (RNA dangle and terminal-mismatch enthalpies.)

### Loop / coaxial / multiloop model terms

- Zuker M. (1994). PNAS **91**:9218-9222. (Coaxial stacking model.)
- Jaeger J.A., Turner D.H., Zuker M. (1989). PNAS **86**:7706-7710. (Ninio
  asymmetry / loop model.)

### Salt corrections (`strider/thermo/salt.py`)

- Owczarzy R. et al. (2004). Biochemistry **43**:3537-3554. (Na⁺ correction.)
- Owczarzy R. et al. (2008). Biochemistry **47**:5336-5353. (Mg²⁺ correction.)
- Tan Z.-J. & Chen S.-J. (2006). Biophys. J. **90**:1175-1190. (Unified
  monovalent/divalent model.)

---

## 2. Build-time inputs (NOT redistributed)

These are consumed by scripts under `scripts/` to *generate* the table modules
above. They are not shipped: the `data/` directory is `.gitignore`d in its
entirety and excluded from the wheel.

- **ViennaRNA `dna_mathews2004.par`** — used by
  `scripts/generate_mathews2004_params.py` to transcribe the DNA ΔG₃₇ and ΔH
  tables into `strider/thermo/parameters/mathews2004-dna.json`. **Identity of
  the transcribed file was re-confirmed at value level in this review round:**
  ViennaRNA ships both a 1999 and a 2004 DNA set whose stack matrices differ
  (e.g. `ATAT` = −0.8 in the 1999 file vs −0.9 in the 2004 file; `GT/CG` = +1.3
  vs +1.2).  The JSON here carries ATAT = −0.9 and GTGT = +1.2, matching
  `dna_mathews2004.par` exactly and excluding the 1999 set. The JSON is
  regenerated with ViennaRNA installed at development time (`pip install
  ViennaRNA`) and verified structure-by-structure against ViennaRNA's own
  `eval_structure` at `dangles=0`. Primary literature for the set is Mathews
  D.H., Sabina J., Zuker M., Turner D.H. (1999) *J. Mol. Biol.* **288**:911-940;
  the refined values distributed in the 2004 `.par` track that lineage plus
  the Turner group corrections ultimately consolidated in the NNDB (Turner &
  Mathews 2010, Nucleic Acids Res. **38**:D280-D282, cited above). These are
  physical measurements, not copyrightable expression, so the JSON retains only
  those numeric constants (as does the analogous adoption of the same parameter
  set in e.g. NUPACK, RNAstructure, and Biopython). Neither the `.par` file nor
  ViennaRNA (© Institute for Theoretical Chemistry, University of Vienna) is
  redistributed or required at runtime by strider;
  `strider/thermo/parameters/mathews2004-dna.json`
  is self-contained static data under the package's MIT license.

- **ViennaRNA `rna_turner2004.par`** — used by
  `scripts/generate_rna_enthalpy_tables.py` to transcribe Turner-2004 RNA loop
  enthalpies. Fetch on demand:

  ```
  curl -fsSLo data/rna_turner2004.par \
    https://raw.githubusercontent.com/ViennaRNA/ViennaRNA/master/misc/rna_turner2004.par
  ```

  The generated ΔG sections are validated against `parameters_rna` before the
  ΔH values are grafted in; the resulting module carries only the numeric
  Turner-2004 / Schroeder-Turner-2000 constants with their primary citations.
  ViennaRNA itself is © Institute for Theoretical Chemistry, University of
  Vienna, and is **not** redistributed here.

- **primer3 `.dh` tables** — the open primer3 distribution of SantaLucia & Hicks
  2004 / Mathews 1999 DNA enthalpies, used by
  `scripts/generate_dna_enthalpy_tables.py`. Same status: facts transcribed,
  source file not redistributed.

- **ArchiveII subsets** (`data/datasets/`) — RNA secondary-structure benchmark
  set used only for accuracy/training evaluation. Not shipped; cite the original
  ArchiveII / RNAstructure distribution if reused.

---

## 3. NUPACK

NUPACK (© California Institute of Technology) is used **only** as an external
comparison baseline in `scripts/bench_vs_nupack.py` and related benchmarks, run
against a separately licensed local NUPACK install. **No NUPACK source, headers,
parameters, or derived numeric tables are included in or derived into this
package.** The benchmark receipt `data/receipts_vs_nupack_rna.json` contains only
strider's own measured timings and ΔΔG residuals, and is `.gitignore`d in any
case. Occurrences of the string "NUPACK" in the source tree are comparative
discussion in docstrings and comments.
