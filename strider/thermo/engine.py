"""
ThermoEngine — central dispatch for all thermodynamic calculations.

Backend selection order (automatic):
    vienna  (if ViennaRNA installed, GPL, optional)
    native  (built-in NN implementation, always available, MIT)
"""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

from strider.equilibrium import cyclic_symmetry

if TYPE_CHECKING:
    from strider.sweep.cache import DiskCache
    from strider.thermo.modified import ModificationSite
    from strider.thermo.parameters import ParameterSet

BackendName = Literal["auto", "native", "vienna"]

R = 1.987e-3  # kcal / (mol · K)

COMPLEMENT_DNA = str.maketrans("ACGT", "TGCA")


@dataclass
class MFEResult:
    energy: float           # kcal/mol
    structure: str          # dot-bracket
    base_pairs: list[tuple[int, int]] = field(default_factory=list)
    sequence: str = ""
    # For a multi-strand complex the MFE is found by an order-invariant search
    # over strand arrangements; ``structure``/``base_pairs``/``sequence`` are in
    # the winning order.  ``strand_order`` maps new slot k → the index of the
    # input strand placed there (identity when unchanged / single strand).
    strand_order: tuple[int, ...] = ()


@dataclass
class PFuncResult:
    free_energy: float          # ensemble ΔG (kcal/mol)
    partition_function: float   # dimensionless Q
    pair_probs: np.ndarray      # shape (n, n)


class ThermoEngine:
    """
    Central thermodynamic engine.

    Parameters
    ----------
    material : 'dna' or 'rna'
    celsius  : temperature in Celsius
    sodium   : [Na+] in molar
    magnesium: [Mg2+] in molar
    backend  : 'auto' | 'native' | 'vienna'
    cache    : optional DiskCache for persistent memoization
    correction_model : optional callable(sequence) -> float for ML corrections
    dangles  : exterior-stem dangling-end handling (0 or 2, default 0).
        ``2`` = best single *negative* 5′/3′ dangle stack per exterior stem.
        NOTE: this is **not** exactly ViennaRNA ``dangles=2``, which sums both
        flanks unconditionally; on a 42-case grid the two-flank cases deviate
        0.2-0.9 kcal/mol (38/42 within 0.4).  Scope: ``mfe`` and ``subopt``
        only — the partition function (``pfunc``), ensemble defect matmul,
        differentiable path and equilibrium solving always use the no-dangle
        model regardless of this flag.
    """

    def __init__(
        self,
        material: Literal["dna", "rna"] = "dna",
        celsius: float = 37.0,
        sodium: float = 0.137,
        magnesium: float = 0.01,
        backend: BackendName = "auto",
        cache: "DiskCache | None" = None,
        correction_model: Callable[[str], float] | None = None,
        parameter_set: "str | ParameterSet | None" = None,
        dangles: int = 0,
    ) -> None:
        self.material = material
        self.celsius = celsius
        self.sodium = sodium
        self.magnesium = magnesium
        self.cache = cache
        self.correction_model = correction_model
        self._backend = self._resolve_backend(backend)
        self._parameter_set_arg = parameter_set
        self._params_cache: "ParameterSet | None" = None
        if dangles not in (0, 2):
            raise ValueError(
                "dangles must be 0 (no exterior dangling ends) or 2 (best single "
                "negative dangle per exterior stem; MFE/subopt only - not "
                "identical to ViennaRNA dangles=2, see ThermoEngine docs)"
            )
        self.dangles = dangles

    @property
    def params(self) -> "ParameterSet":
        """
        Lazily-loaded :class:`ParameterSet` for this engine.

        Selection order:
          1. explicit ``parameter_set`` argument (string name or instance)
          2. ``"native-rna"`` / ``"native-dna"`` matching ``self.material``
        """
        if self._params_cache is not None:
            return self._params_cache

        from strider.thermo.parameters import ParameterSet, load_parameters

        arg = self._parameter_set_arg
        if isinstance(arg, ParameterSet):
            self._params_cache = arg
        elif isinstance(arg, str):
            self._params_cache = load_parameters(arg)
        else:
            default = "native-rna" if self.material == "rna" else "native-dna"
            self._params_cache = load_parameters(default)
        return self._params_cache

    def _uses_custom_params(self) -> bool:
        """
        True iff the user supplied a non-default parameter set.

        The default (``parameter_set=None`` or one of ``"native"`` /
        ``"native-dna"`` / ``"native-rna"``) leaves the energy DP reading
        the module-level constants in :mod:`strider.thermo.parameters_dna`
        / :mod:`strider.thermo.parameters_rna` — numerically identical to
        every prior release.  Only an *explicit* non-native paramset
        opens the override channel; this keeps default behaviour
        bit-identical and bounds the blast radius of the override path.
        """
        arg = self._parameter_set_arg
        if arg is None:
            return False
        if isinstance(arg, str):
            return arg not in ("native", "native-dna", "native-rna")
        return True

    # ─── public API ──────────────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        """The active backend name: 'native' or 'vienna'."""
        return self._backend

    @classmethod
    def available_backends(cls) -> list[str]:
        """Return a list of backend names importable in the current environment."""
        backends = ["native"]
        try:
            import RNA  # noqa: F401
            backends.append("vienna")
        except ImportError:
            pass
        return backends

    def mfe(self, *sequences: str) -> MFEResult:
        """Minimum free energy structure for one or more strands."""
        if not sequences:
            raise ValueError("mfe requires at least one sequence")
        self._validate_alphabet(sequences)
        key = self._cache_key("mfe", sequences)
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        result = self._mfe_dispatch(sequences)
        if self.cache:
            self.cache.set(key, result)
        return result

    def pfunc(self, *sequences: str, pair_probs: bool = True) -> PFuncResult:
        """Ensemble free energy and pair probability matrix.

        ``pair_probs=False`` skips the outside recurrence and returns a zero
        matrix for ``pair_probs``; ``free_energy`` is unchanged (see
        :func:`strider.thermo.ensemble.ensemble_dg`).
        """
        op = "pfunc" if pair_probs else "pfunc_dg"
        key = self._cache_key(op, sequences)
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        result = self._pfunc_dispatch(sequences, pair_probs=pair_probs)
        if self.correction_model is not None:
            combined = "".join(sequences)
            result = PFuncResult(
                result.free_energy + self.correction_model(combined),
                result.partition_function,
                result.pair_probs,
            )
        if self.cache:
            self.cache.set(key, result)
        return result

    def sample(
        self,
        sequence: str,
        n_samples: int,
        seed: int | None = None,
    ) -> list[tuple[str, list[tuple[int, int]]]]:
        """Draw ``n_samples`` Boltzmann-distributed structures for a single strand."""
        from strider.structure.sampling import sample_structures
        return sample_structures(
            sequence, n_samples, celsius=self.celsius, material=self.material, seed=seed,
            sodium_M=self.sodium, magnesium_M=self.magnesium,
        )

    def subopt(
        self,
        *sequences: str,
        gap: float = 1.0,
        max_structures: int = 200,
    ) -> list[tuple[str, float, list[tuple[int, int]]]]:
        """Enumerate suboptimal structures within ``gap`` kcal/mol of the MFE.

        Pass one strand for intramolecular folding, or several strands (or a
        single ``'&'``/``'+'``-joined string) for a dimer / multi-strand complex
        — e.g. ``subopt("AAAA", "TTTT")`` or ``subopt("AAAA&TTTT")``.  Each
        returned dot-bracket carries strand separators; ``pair_list`` is indexed
        over the concatenated sequence, consistent with :meth:`mfe`.

        For a multi-strand complex the enumeration is *order-invariant* (like
        :meth:`mfe`): suboptimals are gathered across strand arrangements,
        deduplicated, and reported in the MFE-winning strand order (pseudoknot
        brackets ``[]``/``{}`` mark pairs that cross in that order).

        Energies are ``mfe``-consistent free energies: each structure's loop
        energy plus the **component-aware** association penalty ``(L−k)·ΔG_assoc``
        (``k`` = number of connected components of *that* structure, so a
        suboptimal that lets a strand float free pays one fewer association) plus
        its coaxial-nick stabilisation.  Hence for a heteromeric complex
        ``subopt(*strands)[0] == mfe(*strands).energy``; they differ only by the
        complex-level rotational-symmetry term σ (a ``−RT ln σ`` ensemble
        correction, not a per-structure energy), which is nonzero only for
        homomeric complexes.  Because the association discount can pull a
        partly-dissociated structure into the window, the gap is applied on these
        corrected energies (the enumeration widens internally to capture them;
        deeply dissociated states of a binding complex sit far above the gap and
        are better read from the dissociated species directly).
        """
        if not sequences:
            raise ValueError("subopt requires at least one sequence")
        strands = "&".join(sequences).replace("+", "&").split("&")
        from strider.structure.sampling import subopt_complex, subopt_structures
        from strider.structure.complex_fold import n_components
        from strider.thermo._param_context import param_context
        with param_context(self._param_override()):
            if len(strands) <= 1:
                return subopt_structures(
                    strands[0], gap=gap, celsius=self.celsius, material=self.material,
                    max_structures=max_structures,
                    sodium_M=self.sodium, magnesium_M=self.magnesium,
                    dangles=self.dangles,
                )
            n = len(strands)
            # Widen the structural enumeration so suboptimals that the
            # association discount brings into the gap are not missed, then
            # re-window on the corrected free energies.  One ΔG_assoc of slack
            # captures a singly-dissociated component (a strand floating free);
            # this is the regime where disconnected structures are competitive,
            # since for a strongly-bound complex they sit far above the gap (and
            # `mfe()` itself returns a disconnected structure when the strands do
            # not all bind).  Deliberately not scaled by strand count: a wider
            # window only re-enumerates connected structures that are filtered
            # back out, at large cost for big complexes.
            per_assoc = self._assoc_correction(2)        # ΔG_assoc for this material
            enum_gap = gap + per_assoc
            raw, order = subopt_complex(
                strands, gap=enum_gap, celsius=self.celsius, material=self.material,
                max_structures=max(2 * max_structures, 200),
                sodium_M=self.sodium, magnesium_M=self.magnesium,
            )
            ordered_lens = [len(strands[i]) for i in order]
            ordered_seq = "".join(strands[i] for i in order)
            corrected: list[tuple[str, float, list[tuple[int, int]]]] = []
            for db, e_struct, plist in raw:
                k = n_components(plist, ordered_lens)
                energy = (
                    e_struct
                    + self._assoc_correction(n, k)
                    + self._coaxial_correction(plist, ordered_lens, ordered_seq)
                )
                corrected.append((db, energy, plist))
            corrected.sort(key=lambda t: t[1])
            if corrected:
                lo = corrected[0][1]
                corrected = [t for t in corrected if t[1] <= lo + gap + 1e-7]
            return corrected[:max_structures]

    def pairs(self, *sequences: str) -> np.ndarray:
        """Pair-probability matrix P[i,j] for the given (multi-)strand complex."""
        return self.pfunc(*sequences).pair_probs

    def ensemble_defect(
        self,
        sequences: str | tuple[str, ...],
        target_structure: str,
        normalize: bool = True,
    ) -> float:
        """
        Ensemble defect of a target dot-bracket structure for the given complex.

        Defect = Σ_i (1 − P_correct(i)), where
          - if position i is unpaired in the target, P_correct(i) = 1 − Σ_j P(i,j)
          - if position i pairs with j in the target, P_correct(i) = P(i,j)

        If ``normalize`` is True (default), the defect is divided by sequence
        length so the value lies in [0, 1].
        """
        from strider.structure.dot_bracket import parse_pairs
        if isinstance(sequences, str):
            seqs = (sequences,)
        else:
            seqs = tuple(sequences)
        clean_target = target_structure.replace("&", "").replace("+", "")
        n = sum(len(s) for s in seqs)
        if len(clean_target) != n:
            raise ValueError(
                f"target structure length {len(clean_target)} != total sequence length {n}"
            )

        probs = self.pairs(*seqs)
        target_pairs = dict()
        for i, j in parse_pairs(target_structure):
            target_pairs[i] = j
            target_pairs[j] = i

        defect = 0.0
        for i in range(n):
            if i in target_pairs:
                j = target_pairs[i]
                p_correct = float(probs[i][j])
            else:
                p_correct = 1.0 - float(probs[i].sum())
            defect += max(0.0, 1.0 - p_correct)

        return defect / n if normalize else defect

    def duplex_dg(self, seq1: str, seq2: str | None = None) -> float:
        """
        ΔG (kcal/mol) of hybridization.

        seq2=None → hairpin (intramolecular folding of seq1).
        """
        if seq2 is None:
            return self.pfunc(seq1).free_energy
        return self._duplex_dg_native(seq1, seq2)

    def ddg(
        self,
        reactants: list[str | list[str]],
        products: list[str | list[str]],
    ) -> float:
        """
        ΔΔG = Σ G(products) - Σ G(reactants) (kcal/mol).

        Each element of reactants/products is either:
          - a single sequence string → compute pfunc of that strand alone
          - a list of sequences → compute pfunc of that multi-strand complex
        """
        def g(item):
            if isinstance(item, str):
                return self.pfunc(item).free_energy
            return self.pfunc(*item).free_energy

        g_react = sum(g(r) for r in reactants)
        g_prod = sum(g(p) for p in products)
        return g_prod - g_react

    def toehold_accessibility(
        self,
        sequence: str,
        toehold_positions: slice | list[int],
    ) -> float:
        """
        Fraction of ensemble where all toehold positions are unpaired.
        """
        result = self.pfunc(sequence)
        pair_probs = result.pair_probs
        n = len(sequence)

        if isinstance(toehold_positions, slice):
            positions = list(range(*toehold_positions.indices(n)))
        else:
            positions = list(toehold_positions)

        if not positions:
            return 1.0

        # Probability position i is unpaired = 1 - Σ_j P(i,j)
        probs_unpaired = [1.0 - pair_probs[i].sum() for i in positions]
        # Joint probability (lower bound, assuming independence)
        return float(np.prod(probs_unpaired))

    def mfe_batch(
        self,
        strand_groups: list[tuple[str, ...]],
        n_workers: int = 1,
    ) -> list[MFEResult]:
        """Parallelized batch MFE computation."""
        if n_workers <= 1 or len(strand_groups) < 4:
            return [self.mfe(*grp) for grp in strand_groups]

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(self.mfe, *grp) for grp in strand_groups]
            return [f.result() for f in futures]

    def melting_temperature(
        self,
        seq: str,
        strand_conc_M: float = 250e-9,
    ) -> float:
        """Melting temperature (°C) for the given sequence."""
        from strider.thermo.nn_dna import melting_temperature as _mt_dna
        from strider.thermo.nn_rna import duplex_dg_rna
        if self.material == "dna":
            return _mt_dna(seq, strand_conc_M, self.sodium, self.magnesium)
        # RNA: approximate
        dH = -80.0  # placeholder; full RNA Tm needs Turner tables
        return dH

    def dimer_thermo(
        self,
        seq1: str,
        seq2: str | None = None,
        *,
        structure: str | list[tuple[int, int]] | None = None,
        strand_conc_M: float = 250e-9,
        salt_model: str = "auto",
    ):
        """Two-state thermodynamics of a bimolecular duplex."""
        from strider.thermo.dimer_thermo import dimer_thermo as _dt
        return _dt(
            seq1,
            seq2,
            sodium_M=self.sodium,
            magnesium_M=self.magnesium,
            material=self.material,
            structure=structure,
            strand_conc_M=strand_conc_M,
            salt_model=salt_model,
        )

    def dimer_tm(
        self,
        seq1: str,
        seq2: str | None = None,
        *,
        strand_conc_M: float = 250e-9,
        salt_model: str = "auto",
    ) -> float:
        """Melting temperature (°C) of a bimolecular duplex."""
        return self.dimer_thermo(
            seq1,
            seq2,
            strand_conc_M=strand_conc_M,
            salt_model=salt_model,
        ).tm_celsius

    # ─── validation ─────────────────────────────────────────────────────────

    _DNA_SET = set("ACGT")
    _RNA_SET = set("ACGU")

    def _validate_alphabet(self, sequences: tuple[str, ...]) -> None:
        """Reject mixed DNA/RNA alphabets when the engine material is fixed."""
        all_bases = set("".join(sequences))
        if self.material == "dna" and all_bases & self._RNA_SET - self._DNA_SET:
            uracil_seqs = [s for s in sequences if "U" in s]
            raise ValueError(
                f"RNA base(s) found in DNA engine: {uracil_seqs}"
            )
        if self.material == "rna" and all_bases & self._DNA_SET - self._RNA_SET:
            thymine_seqs = [s for s in sequences if "T" in s]
            raise ValueError(
                f"DNA base(s) found in RNA engine: {thymine_seqs}"
            )

    # ─── dispatch ────────────────────────────────────────────────────────────

    def _mfe_dispatch(self, sequences: tuple[str, ...]) -> MFEResult:
        """Route MFE calculation to the active backend."""
        if self._backend == "vienna":
            return self._mfe_vienna(sequences)
        return self._mfe_native(sequences)

    def _mfe_sigma_correction(self, sequences: tuple[str, ...], celsius: float) -> float:
        """Rotational-symmetry correction for MFE of a multi-strand complex."""
        sigma = cyclic_symmetry(list(sequences))
        if sigma > 1:
            return R * (celsius + 273.15) * math.log(sigma)
        return 0.0

    def _assoc_correction(self, n_strands: int, n_components: int = 1) -> float:
        """Bimolecular association penalty ``(L−k)·ΔG_assoc`` for a structure.

        Each strand beyond the first in a connected piece costs one association
        (loss of translational / rotational entropy on binding); Dirks et al.
        (2007) charge this once per association.  A structure that splits ``L``
        strands into ``k = n_components`` independent pieces therefore pays
        ``(L−k)·ΔG_assoc`` — full ``(L−1)·ΔG_assoc`` only when the whole complex
        is connected (``k = 1``), and ``0`` when every strand is free
        (``k = L``), since a dissociated piece never paid the association cost.
        NUPACK includes the identical term in its complex free energy (verified:
        zeroing ``join_penalty`` in ``dna04`` shifts the connected complex ΔG by
        exactly ``(L−1)·1.96``), so applying it here is required for thermodynamic
        parity and for physically meaningful concentrations.

        Sourced from ``parameters_{dna,rna}.JOIN_PENALTY`` (DNA 1.96, RNA 4.09 at
        37 °C).  Returns 0 for a single strand.
        """
        n_assoc = n_strands - n_components
        if n_strands < 2 or n_assoc <= 0:
            return 0.0
        if self.material == "dna":
            from strider.thermo.parameters_dna import JOIN_PENALTY
        else:
            from strider.thermo.parameters_rna import JOIN_PENALTY
        return n_assoc * float(JOIN_PENALTY)

    def _coaxial_correction(
        self, pairs: list[tuple[int, int]], strand_lens: list[int], seq: str
    ) -> float:
        """Coaxial-stacking stabilisation at flush strand nicks (kcal/mol, ≤ 0).

        When two helices abut across a strand nick with no intervening unpaired
        bases (a *coaxial junction*), their terminal base pairs stack as if the
        backbone were continuous — a strong stabilisation (~−1 to −2 kcal/mol per
        junction) that NUPACK includes but strider's nick-aware DP omits, treating
        the cross-nick terminus as a bare helix end instead.  This is the dominant,
        junction-count-scaling part of the strider–NUPACK multi-strand gap (a
        contiguous duplex and a nicked duplex differ by ~2.5 kcal/mol; the gap
        grows with strand count).

        Computed from the folded structure (leading-order: the dominant structure
        carries the coaxial stacks), so it composes with the engine's σ and
        association corrections without touching the DP or the pair-probability
        recurrence.  Returns the summed coaxial stack ΔG over flush nicks where the
        bases on both sides are paired and the two closing pairs are mutually
        coaxial (their partners adjacent), keyed through ``COAXIAL_STACK`` (falling
        back to the nearest-neighbour ``STACK`` value).
        """
        if len(strand_lens) < 2 or not pairs:
            return 0.0
        from strider.thermo._param_context import lookup_table
        if self.material == "dna":
            from strider.thermo.parameters_dna import COAXIAL_STACK, STACK
        else:
            from strider.thermo.parameters_rna import STACK
            COAXIAL_STACK = {}
        COAXIAL_STACK = lookup_table("coaxial_stack", COAXIAL_STACK) if COAXIAL_STACK else {}
        STACK = lookup_table("stack", STACK)

        n = len(seq)
        partner = [-1] * n
        for i, j in pairs:
            partner[i] = j
            partner[j] = i
        # Nick positions: index of the first base of each strand after the first.
        nicks = []
        pos = 0
        for length in strand_lens[:-1]:
            pos += length
            nicks.append(pos)

        total = 0.0
        for nk in nicks:
            a, b = nk - 1, nk           # bases flanking the nick (left 3', right 5')
            pa, pb = partner[a], partner[b]
            if pa < 0 or pb < 0:
                continue                # a dangle/loop, not a coaxial junction
            if pa == b:
                continue                # a and b pair each other: one continuous
                                        # helix through the nick, not a junction
            # Flush coaxial stack: two *distinct* closing pairs (a,pa) and (b,pb)
            # align so their partners are adjacent across the helix axis
            # (|pa − pb| == 1), in either stacking orientation.
            if pa == pb + 1:
                key = seq[a] + seq[b] + seq[pb] + seq[pa]
            elif pb == pa + 1:
                key = seq[pa] + seq[pb] + seq[b] + seq[a]
            else:
                continue
            dg = COAXIAL_STACK.get(key)
            if dg is None:
                dg = STACK.get(key, 0.0)
            if dg < 0:
                total += dg
        return total

    def _pfunc_dispatch(self, sequences: tuple[str, ...], pair_probs: bool = True) -> PFuncResult:
        """Route partition function calculation to the active backend."""
        if self._backend == "vienna":
            return self._pfunc_vienna(sequences)
        return self._pfunc_native(sequences, pair_probs=pair_probs)

    # ─── native backend ───────────────────────────────────────────────────────

    def _mfe_native(self, sequences: tuple[str, ...]) -> MFEResult:
        """MFE via the built-in Zuker-style DP (strider.structure.mfe).

        For a multi-strand complex the result is folded *order-invariantly*: the
        linear kernel only represents structures non-crossing for one strand
        concatenation, so the predicted MFE would otherwise change under a mere
        relabelling of strand order (Dirks et al. 2007).  ``fold_complex`` folds
        the distinct arrangements and returns the global minimum over a structure
        that connects all strands; the reported ``sequence``/``structure`` are in
        the winning order (a structure nested in that order may be pseudoknotted
        in the caller's, so it cannot be rendered in the caller's order).
        """
        from strider.structure.mfe import fold_mfe
        from strider.structure.complex_fold import fold_complex, DEFAULT_MAX_STRANDS
        from strider.thermo._param_context import param_context
        with param_context(self._param_override()):
            if len(sequences) <= 1:
                seq = sequences[0] if sequences else ""
                structure, energy, pairs = fold_mfe(
                    seq, self.celsius, self.material, self.sodium, self.magnesium,
                    dangles=self.dangles,
                )
                order = tuple(range(len(sequences)))
            elif len(sequences) <= DEFAULT_MAX_STRANDS:
                cf = fold_complex(
                    list(sequences), self.celsius, self.material,
                    self.sodium, self.magnesium, dangles=self.dangles,
                )
                structure, energy, pairs, order = (
                    cf.structure, cf.energy, cf.pairs, cf.order,
                )
            else:
                # Too many strands for an order search; fold the given order.
                order = tuple(range(len(sequences)))
                structure, energy, pairs = fold_mfe(
                    "&".join(sequences), self.celsius, self.material,
                    self.sodium, self.magnesium, dangles=self.dangles,
                )
        seq = "&".join(sequences[i] for i in order)
        energy += self._mfe_sigma_correction(sequences, self.celsius)
        if len(sequences) > 1:
            from strider.structure.complex_fold import n_components
            ordered_lens = [len(sequences[i]) for i in order]
            k = n_components(pairs, ordered_lens)
            energy += self._assoc_correction(len(sequences), k)
            energy += self._coaxial_correction(pairs, ordered_lens, seq.replace("&", ""))
        return MFEResult(
            energy=energy, structure=structure, base_pairs=pairs,
            sequence=seq, strand_order=tuple(order),
        )

    def _pfunc_native(self, sequences: tuple[str, ...], pair_probs: bool = True) -> PFuncResult:
        """Partition function via the built-in McCaskill DP (single- or multi-strand)."""
        from strider.thermo._param_context import param_context
        with param_context(self._param_override()):
            return self._pfunc_native_inner(sequences, pair_probs=pair_probs)

    def _param_override(self) -> "ParameterSet | None":
        """Resolve the :class:`ParameterSet` override active for the native DP.

        Three regimes, in order:

        * **celsius == 37 °C, default params** → ``None``.  The DP falls through
          to the module-level constants in :mod:`strider.thermo.parameters_dna`
          / ``parameters_rna``, numerically identical to every prior release.
        * **celsius == 37 °C, custom params** → the custom set, unblended.
        * **celsius != 37 °C** → a temperature-blended set (``ΔG(T) = ΔG₃₇·T/Tref
          + ΔH·(1 − T/Tref)``).  For a custom set we blend the whole set (its
          ``dG``/``dH`` share keys); for the default we synthesize a *minimal*
          override carrying only the temperature-varying tables sourced straight
          from the module constants (see
          :func:`strider.thermo.temperature.native_temperature_paramset`), so
          untouched tables fall back to those same constants.  The STK_* dangle
          / terminal-mismatch Boltzmann factors are baked at 37 °C and are not
          exposed via the schema, so they stay at 37 °C (small terminal terms;
          documented limitation).
        """
        custom = self.params if self._uses_custom_params() else None
        if self.celsius == 37.0:
            return custom
        from strider.thermo.temperature import (
            blend_paramset, native_temperature_paramset,
        )
        if custom is not None:
            return blend_paramset(custom, self.celsius)
        return native_temperature_paramset(self.material, self.celsius)

    def _pfunc_native_inner(self, sequences: tuple[str, ...], pair_probs: bool = True) -> PFuncResult:
        """Body of :meth:`_pfunc_native`; called inside the override context."""
        if len(sequences) == 1:
            from strider.thermo.ensemble import ensemble_dg
            dG, probs = ensemble_dg(
                sequences[0], self.celsius, self.material,
                self.sodium, self.magnesium,
                pair_probs=pair_probs,
            )
        else:
            # Multi-strand: nick-aware McCaskill DP on concatenated sequence.
            # Returns ensemble ΔG of the complex (not the binding ΔG).
            # engine.ddg() subtracts individual strand energies to get ΔΔG.
            from strider.thermo.ensemble import multistrand_pairs
            from strider.equilibrium import cyclic_symmetry
            dG, probs = multistrand_pairs(
                list(sequences), self.celsius, self.material,
                self.sodium, self.magnesium,
                pair_probs=pair_probs,
            )
            # Rotational-symmetry correction: the nick-aware DP is for the
            # *ordered* concatenation, so a homomeric complex over-counts by σ.
            # The σ correction follows Dirks et al. (2007) SIAM Review 49:65-88.
            sigma = cyclic_symmetry(list(sequences))
            if sigma > 1:
                dG += R * (self.celsius + 273.15) * math.log(sigma)
            # Bimolecular association penalty (L−1)·ΔG_assoc — the same term
            # NUPACK includes in its complex free energy (see _assoc_correction).
            dG += self._assoc_correction(len(sequences))
            # Coaxial stacking at flush strand nicks (leading-order, from the
            # dominant structure) — recovers the junction-scaling part of the
            # NUPACK gap that the nick-aware DP omits (see _coaxial_correction).
            from strider.structure.complex_fold import fold_complex, DEFAULT_MAX_STRANDS
            if len(sequences) <= DEFAULT_MAX_STRANDS:
                cf = fold_complex(
                    list(sequences), self.celsius, self.material,
                    self.sodium, self.magnesium,
                )
                ordered_lens = [len(sequences[i]) for i in cf.order]
                ordered_seq = "".join(sequences[i] for i in cf.order)
                dG += self._coaxial_correction(cf.pairs, ordered_lens, ordered_seq)

        Z = math.exp(-dG / (R * (self.celsius + 273.15)))
        return PFuncResult(free_energy=dG, partition_function=Z, pair_probs=probs)

    def _duplex_dg_native(self, seq1: str, seq2: str) -> float:
        """Dispatch duplex ΔG to the correct NN table (DNA/DNA, RNA/RNA, or DNA:RNA hybrid)."""
        from strider.thermo.nn_dna import duplex_dg
        from strider.thermo.nn_rna import duplex_dg_rna
        from strider.thermo.nn_dna_rna import hybrid_duplex_dg

        s1 = seq1.upper()
        s2 = seq2.upper()
        has_u1 = "U" in s1
        has_u2 = "U" in s2

        if has_u1 or has_u2:
            if has_u1 and has_u2:
                return duplex_dg_rna(s1, self.celsius, self.sodium)
            # hybrid
            dna = s2 if has_u1 else s1
            return hybrid_duplex_dg(dna, self.celsius, self.sodium)
        return duplex_dg(s1, s2, self.celsius, self.sodium, self.magnesium)

    # ─── vienna backend ───────────────────────────────────────────────────────

    def _mfe_vienna(self, sequences: tuple[str, ...]) -> MFEResult:
        """MFE via ViennaRNA RNA.fold() (single strand) or RNA.cofold() (two strands)."""
        if len(sequences) > 2:
            raise ValueError("Vienna backend supports at most two strands for MFE")
        from strider.thermo import vienna_backend as vb
        from strider.structure.dot_bracket import parse_pairs

        if len(sequences) == 1:
            seq = sequences[0]
            structure, energy = vb.fold(seq, self.celsius, self.material)
        else:
            seq = "&".join(sequences)
            structure, energy = vb.co_fold(
                sequences[0], sequences[1], self.celsius, self.material
            )
            # ViennaRNA mfe_dimer() returns a flat structure without '&';
            # reinsert it so len(structure) == len(seq).
            nick = len(sequences[0])
            structure = structure[:nick] + "&" + structure[nick:]

        pairs = parse_pairs(structure.replace("&", ""))
        energy += self._mfe_sigma_correction(sequences, self.celsius)
        return MFEResult(energy=energy, structure=structure, base_pairs=pairs, sequence=seq)

    def _pfunc_vienna(self, sequences: tuple[str, ...]) -> PFuncResult:
        """Partition function via ViennaRNA pf_fold()."""
        from strider.thermo import vienna_backend as vb
        seq = sequences[0] if len(sequences) == 1 else sequences[0] + "&" + sequences[-1]
        dG, probs = vb.pf_fold(sequences[0], self.celsius, self.material)
        Z = math.exp(-dG / (R * (self.celsius + 273.15)))
        return PFuncResult(free_energy=dG, partition_function=Z, pair_probs=probs)

    # ─── helpers ─────────────────────────────────────────────────────────────

    def _resolve_backend(self, backend: BackendName) -> str:
        """Resolve 'auto' to strider's own native engine; pass explicit names through.

        ``native`` is the authoritative, always-available, dependency-free engine
        and the default.  ``vienna`` is an *optional* cross-check backend you must
        request explicitly (``backend='vienna'``) — it is never auto-selected, so
        strider's results never silently depend on an external library.
        """
        if backend != "auto":
            return backend
        return "native"

    def _cache_key(self, op: str, sequences: tuple[str, ...]) -> str:
        """Build a SHA-256 cache key from the operation, conditions, parameter set, and sequence content."""
        # Resolve parameter-set name without forcing a load if no override is set.
        ps_arg = self._parameter_set_arg
        if ps_arg is None:
            ps_name = "default"
        elif isinstance(ps_arg, str):
            ps_name = ps_arg
        else:  # ParameterSet instance
            ps_name = getattr(ps_arg, "name", "custom")
        # dangles is scoped to MFE-style results only (pfunc/ensemble explicitly
        # ignores it, see __init__ docs); folding it into a pfunc cache key would
        # imply an ensemble effect that does not exist.  "pfunc_dg" is the
        # free-energy-only variant of pfunc (pair_probs=False) and likewise
        # ignores dangles.
        dang = f"|d{self.dangles}" if op not in ("pfunc", "pfunc_dg") else ""
        raw = (
            f"{op}|{self.material}|{self.celsius}|{self.sodium}|{self.magnesium}|"
            f"{ps_name}{dang}|{'|'.join(sequences)}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def __repr__(self) -> str:
        ps_arg = self._parameter_set_arg
        ps_name = ps_arg if isinstance(ps_arg, str) else getattr(ps_arg, "name", None)
        ps_part = f", parameter_set={ps_name!r}" if ps_name else ""
        return (
            f"ThermoEngine(material={self.material!r}, celsius={self.celsius}, "
            f"sodium={self.sodium}, magnesium={self.magnesium}, "
            f"backend={self._backend!r}{ps_part})"
        )

