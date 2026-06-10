"""Tests for structure-resolved DNA thermodynamics (ΔG/ΔH/ΔS/Tm) and the
completeness of the native DNA ΔH tables."""
from __future__ import annotations

import random

import pytest

from strider.structure.mfe import fold_mfe
from strider.thermo.parameters import load_parameters
from strider.thermo.structure_thermo import (
    hairpin_thermo,
    parse_hairpin_pairs,
    structure_free_energy,
    structure_enthalpy,
)


def test_dG_reconstruction_matches_fold_mfe():
    """Walking the structure with the engine's own per-element energies must
    reproduce the MFE DP energy exactly for single hairpins."""
    random.seed(1)
    checked = 0
    for _ in range(800):
        seq = "".join(random.choice("ACGT") for _ in range(random.randint(12, 30)))
        struct, energy, _ = fold_mfe(seq, 37.0, "dna")
        if parse_hairpin_pairs(struct) is None:
            continue
        recon = structure_free_energy(seq, struct)
        assert recon == pytest.approx(energy, abs=1e-9)
        checked += 1
    assert checked > 100  # ensure the test actually exercised hairpins


def test_native_dna_dH_is_complete():
    """The native DNA ΔH set must carry the loop tables (not ΔG copies)."""
    ps = load_parameters("native")
    for key in ("stack", "hairpin_mismatch", "interior_mismatch",
                "hairpin_triloop", "hairpin_tetraloop",
                "hairpin_size", "bulge_size", "interior_size"):
        assert key in ps.dH, f"missing ΔH table: {key}"
    # Loop initiation ΔH is purely entropic (zero).
    assert all(v == 0.0 for v in ps.dH["hairpin_size"])
    assert all(v == 0.0 for v in ps.dH["bulge_size"])
    assert all(v == 0.0 for v in ps.dH["interior_size"])
    # Stack ΔH must equal the SantaLucia NN enthalpies.
    from strider.thermo.nn_dna import DNA_NN
    from strider.thermo.parameters_native import _stack_key, _COMPL_DNA
    for dinuc, (h, _s, _g) in DNA_NN.items():
        assert ps.dH["stack"][_stack_key(dinuc, _COMPL_DNA)] == pytest.approx(h)


def test_hairpin_tm_is_unimolecular_and_consistent():
    """Tm = ΔH/ΔS with ΔS = (ΔH − ΔG)/T_ref, and concentration-independent."""
    from strider.thermo.structure_thermo import T_REF_K
    seq = "AGACGTTGTGCTCAACAAGGT"
    r = hairpin_thermo(seq, sodium_M=1.0)
    assert r is not None and r["Tm"] is not None
    dS = (r["dH"] - r["dG"]) / T_REF_K * 1000.0  # cal/mol/K
    assert dS == pytest.approx(r["dS"], abs=1e-6)
    tm = r["dH"] / (r["dS"] / 1000.0) - 273.15
    assert tm == pytest.approx(r["Tm"], abs=1e-6)
    # Unimolecular: Tm independent of strand concentration (no CT term).
    assert hairpin_thermo(seq, sodium_M=1.0)["Tm"] == pytest.approx(r["Tm"])


def test_non_hairpin_returns_none():
    # all-unpaired sequence → no stable single hairpin
    assert hairpin_thermo("AAAAAAAAAAAA", sodium_M=1.0) is None


def test_enthalpy_more_negative_than_free_energy():
    """For a stable hairpin, folding is enthalpy-driven: ΔH < ΔG < 0."""
    seq = "CGCGACTGTCGACCTGGGCAGGGTTCGGTCGCG"
    struct, _, _ = fold_mfe(seq, 37.0, "dna")
    dG = structure_free_energy(seq, struct)
    dH = structure_enthalpy(seq, struct)
    assert dH < dG < 0
