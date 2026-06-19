"""Pure-thermodynamic equilibrium concentration solver tests."""
import math
import pytest

from strider import solve_equilibrium, equilibrium_from_engine


R = 1.987e-3   # kcal / (mol · K)


class TestSolveEquilibrium:
    def test_bimolecular_matches_analytical(self):
        """A + B <-> AB matches the closed-form Langmuir solution."""
        dG = -10.0
        c_total = 1e-7
        result = solve_equilibrium(
            complexes={
                "A":  (["A"], 0.0),
                "B":  (["B"], 0.0),
                "AB": (["A", "B"], dG),
            },
            totals={"A": c_total, "B": c_total},
            celsius=37.0,
        )
        assert result.converged
        assert result.residual < 1e-9

        # Closed form: y = (2 + 1/K_eff - sqrt(...)) / 2,  y = [AB]/c_total
        RT = R * 310.15
        K = math.exp(-dG / RT)
        K_eff = K * c_total
        y = ((2 + 1 / K_eff) - math.sqrt((2 + 1 / K_eff) ** 2 - 4)) / 2
        analytical_ab = y * c_total
        assert abs(result.concentrations["AB"] - analytical_ab) < 1e-12
        # Mass balance
        assert abs(result.concentrations["A"] + result.concentrations["AB"] - c_total) < 1e-12

    def test_homodimer(self):
        """2A <-> AA with ΔG = -10 kcal/mol."""
        result = solve_equilibrium(
            complexes={
                "A":  (["A"], 0.0),
                "AA": (["A", "A"], -10.0),
            },
            totals={"A": 1e-7},
            celsius=37.0,
        )
        assert result.converged
        # Mass balance: [A] + 2[AA] = 1e-7
        mass = result.concentrations["A"] + 2 * result.concentrations["AA"]
        assert abs(mass - 1e-7) < 1e-12

    def test_strong_binding_drives_complex(self):
        """Very negative ΔG should leave almost no free strand."""
        result = solve_equilibrium(
            complexes={
                "A":  (["A"], 0.0),
                "B":  (["B"], 0.0),
                "AB": (["A", "B"], -25.0),  # K_eq ~ 10^17
            },
            totals={"A": 1e-7, "B": 1e-7},
            celsius=37.0,
        )
        assert result.concentrations["AB"] > 0.99 * 1e-7
        assert result.concentrations["A"] < 1e-10
        assert result.concentrations["B"] < 1e-10

    def test_weak_binding_keeps_strands_free(self):
        """Positive ΔG leaves complexes negligible."""
        result = solve_equilibrium(
            complexes={
                "A":  (["A"], 0.0),
                "B":  (["B"], 0.0),
                "AB": (["A", "B"], +5.0),
            },
            totals={"A": 1e-6, "B": 1e-6},
            celsius=37.0,
        )
        assert result.concentrations["A"] > 0.99 * 1e-6
        assert result.concentrations["AB"] < 1e-9

    def test_three_strand_competition(self):
        """A binds both B and C; the more stable complex dominates."""
        result = solve_equilibrium(
            complexes={
                "A":  (["A"], 0.0),
                "B":  (["B"], 0.0),
                "C":  (["C"], 0.0),
                "AB": (["A", "B"], -10.0),  # weaker
                "AC": (["A", "C"], -15.0),  # stronger
            },
            totals={"A": 1e-7, "B": 1e-7, "C": 1e-7},
            celsius=37.0,
        )
        assert result.converged
        assert result.concentrations["AC"] > result.concentrations["AB"]

    def test_unknown_strand_raises(self):
        with pytest.raises(ValueError):
            solve_equilibrium(
                complexes={"AB": (["A", "B"], -10.0)},
                totals={"A": 1e-7},  # B missing
            )

    def test_result_includes_free_strands(self):
        result = solve_equilibrium(
            complexes={"A": (["A"], 0.0), "B": (["B"], 0.0), "AB": (["A", "B"], -10.0)},
            totals={"A": 1e-7, "B": 1e-7},
        )
        assert "A" in result.strand_free
        assert result.strand_free["A"] == result.concentrations["A"]


class TestCyclicSymmetry:
    def test_monomer(self):
        from strider import cyclic_symmetry
        assert cyclic_symmetry(["A"]) == 1

    def test_heterodimer(self):
        from strider import cyclic_symmetry
        assert cyclic_symmetry(["A", "B"]) == 1

    def test_homodimer(self):
        from strider import cyclic_symmetry
        assert cyclic_symmetry(["A", "A"]) == 2

    def test_homotrimer(self):
        from strider import cyclic_symmetry
        assert cyclic_symmetry(["A", "A", "A"]) == 3

    def test_aba_period_two(self):
        from strider import cyclic_symmetry
        # [A,B,A,B] has 2 cyclic rotations matching original
        assert cyclic_symmetry(["A", "B", "A", "B"]) == 2

    def test_aab_no_symmetry(self):
        from strider import cyclic_symmetry
        assert cyclic_symmetry(["A", "A", "B"]) == 1


class TestSymmetryInEngine:
    """σ correction is applied at the pfunc layer (in ThermoEngine), not by
    the solver, so every backend produces a comparable species-level ΔG.
    """

    def test_native_homodimer_pfunc_includes_sigma_and_assoc(self):
        from strider.thermo.engine import ThermoEngine
        from strider.thermo.ensemble import multistrand_pairs
        from strider.thermo.parameters_dna import JOIN_PENALTY
        eng = ThermoEngine(material="dna", celsius=37.0,
                           sodium=0.137, magnesium=0.01, backend="native")
        # engine.pfunc = raw DP + σ correction (RT·ln 2 for a homodimer)
        # + the (L−1)·ΔG_assoc bimolecular association penalty.  Pass the
        # engine's salt to the raw DP so both sides see the same per-bp salt.
        seq = "GCGCGCAAAA"
        raw, _ = multistrand_pairs([seq, seq], 37.0, "dna", 0.137, 0.01)
        corrected = eng.pfunc(seq, seq).free_energy
        import math
        expected_delta = 1.987e-3 * 310.15 * math.log(2) + JOIN_PENALTY
        assert abs((corrected - raw) - expected_delta) < 1e-6

    def test_heterodimer_includes_assoc_only(self):
        from strider.thermo.engine import ThermoEngine
        from strider.thermo.ensemble import multistrand_pairs
        from strider.thermo.parameters_dna import JOIN_PENALTY
        eng = ThermoEngine(material="dna", celsius=37.0,
                           sodium=0.137, magnesium=0.01, backend="native")
        a, b = "GCGCGCAAAA", "TTTTGCGCGC"
        raw, _ = multistrand_pairs([a, b], 37.0, "dna", 0.137, 0.01)
        corrected = eng.pfunc(a, b).free_energy
        # σ = 1 (no symmetry shift); only the association penalty applies.
        assert abs((corrected - raw) - JOIN_PENALTY) < 1e-6


class TestEquilibriumFromEngine:
    @pytest.fixture
    def engine(self):
        from strider.thermo.engine import ThermoEngine
        return ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)

    def test_auto_enumerates_complexes(self, engine):
        result = equilibrium_from_engine(
            engine,
            strands={"S1": "GCGCGCAA", "S2": "TTGCGCGC"},
            totals={"S1": 1e-6, "S2": 1e-6},
            max_size=2,
        )
        # Should include monomers + pairs (S1, S2, S1_S1, S1_S2, S2_S2)
        assert "S1" in result.concentrations
        assert "S2" in result.concentrations
        assert "S1_S2" in result.concentrations
        assert result.converged
