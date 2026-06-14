"""
Tests for the differentiable design layer: BPP-based objectives
(:mod:`strider.thermo.diff_design`) and the gradient designer with SA hand-off
(:mod:`strider.design.diff_designer`).
"""

import pytest

pytest.importorskip("torch", reason="torch not installed (optional 'diff' extra)")
import torch

from strider import ThermoEngine
from strider.thermo.differentiable import (
    ThermoParameters,
    BatchedPartitionFunction,
    seq_to_probs,
    concat_with_nicks,
)
from strider.thermo.diff_design import DiffObjective
from strider.design.optimizer import DomainSpec
from strider.design.objective import DesignObjective
from strider.design.diff_designer import DifferentiableDesigner


@pytest.fixture(scope="module")
def model():
    return BatchedPartitionFunction(ThermoParameters(material="rna"))


@pytest.fixture(scope="module")
def engine():
    return ThermoEngine(material="rna", celsius=37.0)


# ── objective correctness vs the native engine ──────────────────────────────

@pytest.mark.parametrize("seq,target", [
    ("GGGCUUCGGCCC", "((((....))))"),
    ("GCAGCUUCGGCUGC", "(((((....)))))"),
    ("GGGAAACUCGAGUUUCCC", "((((((......))))))"),
])
def test_soft_ensemble_defect_matches_native(model, engine, seq, target):
    """The soft ensemble defect tracks the native engine on well-defined hairpins."""
    soft = float(DiffObjective.ensemble_defect(target).bind(model)(seq_to_probs([seq])).item())
    native = engine.ensemble_defect(seq, target, normalize=True)
    assert abs(soft - native) < 0.1


def test_defect_gradient_flows_to_sequence(model):
    target = "((((((....))))))"
    probs = torch.softmax(torch.randn(1, len(target), 4, dtype=torch.float64), dim=-1)
    probs = probs.detach().requires_grad_(True)
    DiffObjective.ensemble_defect(target).bind(model)(probs).sum().backward()
    assert probs.grad is not None and probs.grad.abs().sum() > 0.0


def test_free_energy_target_term(model):
    """The ΔG-target term is zero at the target and grows away from it."""
    seq = "GGGCUUCGGCCC"
    fe = model.soft_forward(seq_to_probs([seq])).item()
    at_target = DiffObjective.free_energy_target(fe).bind(model)(seq_to_probs([seq])).item()
    off_target = DiffObjective.free_energy_target(fe + 3.0).bind(model)(seq_to_probs([seq])).item()
    assert at_target < 1e-6 < off_target


def test_gc_and_motif_terms_are_sequence_gradients(model):
    """GC and forbidden-motif terms produce finite gradients to the sequence."""
    probs = torch.softmax(torch.randn(1, 16, 4, dtype=torch.float64), dim=-1)
    probs = probs.detach().requires_grad_(True)
    obj = DiffObjective.gc_content(0.5) + DiffObjective.forbidden_motifs(["GGGG", "AAAA"])
    obj.bind(model)(probs).sum().backward()
    assert torch.isfinite(probs.grad).all() and probs.grad.abs().sum() > 0.0


def test_accessibility_term_rewards_unpaired_window(model):
    """A single-stranded sequence scores ~0 accessibility penalty for its toehold."""
    seq = "AAAAAAAAAAAA"          # no structure: fully accessible
    obj = DiffObjective.accessibility(list(range(0, 6)), min_prob=0.8).bind(model)
    assert obj(seq_to_probs([seq])).item() < 1e-3


# ── multi-strand objectives ─────────────────────────────────────────────────

def test_complex_defect_matches_native(model, engine):
    a, b = "GGGAAA", "UUUCCC"
    seq, nicks = concat_with_nicks([a, b])
    target = "((((((+))))))"
    soft = float(DiffObjective.ensemble_defect(target).bind(model)(
        seq_to_probs([seq]), nicks=nicks).item())
    native = engine.ensemble_defect((a, b), target, normalize=True)
    assert abs(soft - native) < 0.1


# ── gradient designer (hybrid hand-off) ─────────────────────────────────────

def test_gradient_designer_inverse_folds_hairpin(engine):
    """Gradient design + SA polish recovers a sequence that folds to the target."""
    target = "((((((....))))))"
    n = len(target)
    des = DifferentiableDesigner(material="rna", engine=engine, seed=0)
    obj = (DiffObjective.ensemble_defect(target)
           + 0.2 * DiffObjective.gc_band(0.4, 0.6))
    sa_obj = DesignObjective.ensemble_defect(engine, "hp", target)
    res = des.design(
        {"hp": DomainSpec(length=n, material="rna")},
        obj, n_restarts=6, n_steps=120, lr=0.25,
        sa_polish=True, sa_objective=sa_obj, sa_iterations=120,
    )
    seq = res.sequences["hp"]
    assert set(seq) <= set("ACGU")                 # RNA alphabet in the result
    assert engine.ensemble_defect(seq, target, normalize=True) < 0.2


def test_gradient_designer_fixed_domain_is_preserved(engine):
    """A fixed domain is clamped and never mutated by the gradient phase."""
    des = DifferentiableDesigner(material="rna", engine=engine, seed=1)
    obj = DiffObjective.gc_content(0.5)
    res = des.design(
        {"fixed": DomainSpec(sequence="ACGUACGU", material="rna"),
         "free": DomainSpec(length=8, material="rna")},
        obj, n_restarts=2, n_steps=20, sa_polish=False,
    )
    assert res.sequences["fixed"] == "ACGUACGU"
    assert len(res.sequences["free"]) == 8
