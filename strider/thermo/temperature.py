"""
Temperature-dependent ΔG via ΔG/ΔH blending.

strider's native McCaskill / Zuker DP reads the 37 °C ΔG tables and lets
temperature enter only as the ``R·T`` Boltzmann prefactor — so without help the
energies themselves are ΔG₃₇ regardless of the working temperature.  This module
extrapolates a :class:`~strider.thermo.parameters.ParameterSet` to an arbitrary
temperature using the two-state relation

    ΔS    = (ΔH − ΔG₃₇) / T_ref           (T_ref = 310.15 K)
    ΔG(T) = ΔH − T·ΔS
          = ΔG₃₇·(T/T_ref) + ΔH·(1 − T/T_ref)

which returns ΔG₃₇ exactly at ``T = T_ref``.  The blended set is fed to the DP
through :class:`~strider.thermo._param_context.param_context`, so the inner loops
need no changes (Dirks-style override channel).

Two entry points:

* :func:`blend_paramset` — blend an arbitrary set over **all** of its ``dG``
  keys.  Safe for the bundled/custom JSON sets, whose ``dG`` and ``dH`` subtrees
  share keys by construction.

* :func:`native_temperature_paramset` — build a *minimal* override for the
  default (native) engine path, carrying only the tables that actually vary with
  T, sourced **directly from the module constants** the DP reads
  (:mod:`strider.thermo.parameters_dna` / ``parameters_rna``).  This is
  deliberate: :func:`strider.thermo.parameters_native.build_native_paramset`
  diverges from those constants for several tables (the stack table omits the
  20 non-Watson–Crick keys; the loop-size arrays use a different index
  convention; multiloop/terminal-penalty values differ), so blending *it* and
  routing the result through the override channel would silently corrupt the
  baseline at any non-37 °C temperature.  Sourcing from the constants keeps the
  37 °C result bit-identical and confines the change to a genuine temperature
  shift.

Scope / limitations (first pass; see DIMER_SALT_TEMP_PLAN.md §3):

* The STK_* dangle / terminal-mismatch factors are precomputed 37 °C Boltzmann
  factors not exposed via the schema, so they stay at 37 °C.  These are small
  terminal corrections; the bulk (stacks, mismatches, loop initiation) is
  extrapolated.
* Tables without a curated ΔH (interior_1_1/1_2/2_2, dangles, terminal mismatch,
  terminal penalty, coaxial, multiloop, asymmetry) degrade to ΔH = ΔG₃₇, i.e.
  temperature-independent.
* RNA loop-initiation ΔH is now curated from Turner-2004
  (:mod:`strider.thermo._rna_enthalpy_generated`, ΔG-validated against the
  ``parameters_rna`` size arrays), so RNA hairpin/bulge/interior loops carry a
  real enthalpy — unlike DNA, where loop-init ΔH is genuinely zero.  RNA
  terminal-mismatch / dangle ΔH remain uncurated (degrade to ΔH = ΔG₃₇).  The
  separate two-state ``build_native_paramset`` RNA ``dH`` still uses the coarse
  ΔG-copy and a divergent ``_INTERIOR`` table; the engine temperature path
  (this function) is the curated one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from strider.thermo.parameters import ParameterSet

T_REF_K = 310.15  # 37 °C — reference temperature of the ΔG parameter tables


def _blend_scalar(g: float, h: "float | None", frac: float) -> float:
    """ΔG(T) for one scalar; ``h is None`` ⇒ ΔH = ΔG ⇒ temperature-independent."""
    if h is None:
        return g
    return g * frac + h * (1.0 - frac)


def _blend_value(g_val: Any, h_val: Any, frac: float) -> Any:
    """Blend a single ``dG`` entry (scalar, ndarray, or dict-of-scalars).

    Missing ΔH (``h_val is None`` or a per-key gap inside a dict) degrades that
    element to ΔH = ΔG, leaving it temperature-independent — never dropped, so
    every key the DP looks up stays present.
    """
    if isinstance(g_val, np.ndarray):
        if h_val is None:
            return g_val
        return g_val * frac + np.asarray(h_val, dtype=float) * (1.0 - frac)
    if isinstance(g_val, dict):
        h_dict = h_val if isinstance(h_val, dict) else {}
        return {
            k: _blend_scalar(float(gv), _maybe_float(h_dict.get(k)), frac)
            for k, gv in g_val.items()
        }
    # scalar (int/float); leave anything exotic untouched
    try:
        return _blend_scalar(float(g_val), _maybe_float(h_val), frac)
    except (TypeError, ValueError):
        return g_val


def _maybe_float(v: Any) -> "float | None":
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def blend_paramset(paramset: ParameterSet, celsius: float) -> ParameterSet:
    """Temperature-extrapolate every ``dG`` table of ``paramset`` to ``celsius``.

    Each entry is blended toward its matching ``dH`` entry; entries (or
    individual dict keys) without a ΔH counterpart stay at their 37 °C value.
    At ``celsius == 37`` the result equals the input ΔG exactly.
    """
    frac = (celsius + 273.15) / T_REF_K
    new_dG = {
        key: _blend_value(g_val, paramset.dH.get(key), frac)
        for key, g_val in paramset.dG.items()
    }
    return ParameterSet(
        name=f"{paramset.name}@{celsius:g}C",
        material=paramset.material,
        default_wobble_pairing=paramset.default_wobble_pairing,
        dG=new_dG,
        dH=paramset.dH,
        source_path=None,
        comment=f"temperature-blended from {paramset.name} at {celsius} °C",
    )


def native_temperature_paramset(material: str, celsius: float) -> ParameterSet:
    """Minimal temperature override for the default (native) engine path.

    Carries only the temperature-varying tables, with ΔG sourced directly from
    the module constants the DP reads and ΔH from the curated enthalpy tables
    (per-key fallback to ΔH = ΔG where a key is uncurated).  Tables omitted here
    fall back to the same module constants inside the DP, so they remain at
    37 °C.  At ``celsius == 37`` every blended value equals its source.
    """
    frac = (celsius + 273.15) / T_REF_K
    mat = material.lower()

    if mat == "dna":
        import strider.thermo.parameters_dna as p
        from strider.thermo._dna_enthalpy_generated import (
            STACK_DH, HAIRPIN_MISMATCH_DH, INTERIOR_MISMATCH_DH,
            HAIRPIN_TRILOOP_DH, HAIRPIN_TETRALOOP_DH,
        )
        dict_tables = {
            "stack": (p.STACK, STACK_DH),
            "hairpin_mismatch": (p.HAIRPIN_MISMATCH, HAIRPIN_MISMATCH_DH),
            "interior_mismatch": (p.INTERIOR_MISMATCH, INTERIOR_MISMATCH_DH),
            "hairpin_triloop": (p.HAIRPIN_TRILOOP, HAIRPIN_TRILOOP_DH),
            "hairpin_tetraloop": (p.HAIRPIN_TETRALOOP, HAIRPIN_TETRALOOP_DH),
        }
        # DNA loop initiation is purely entropic (ΔH = 0 ⇒ ΔG(T) = ΔG₃₇·T/T_ref).
        size_tables = {
            "hairpin_size": (p.HAIRPIN_SIZE, None),
            "bulge_size": (p.BULGE_SIZE, None),
            "interior_size": (p.INTERIOR_SIZE, None),
        }
        wobble = False
    elif mat == "rna":
        import strider.thermo.parameters_rna as p
        from strider.thermo.parameters_native import _stack_dh_rna
        from strider.thermo._rna_enthalpy_generated import (
            HAIRPIN_SIZE_DH, BULGE_SIZE_DH, INTERIOR_SIZE_DH,
            HAIRPIN_TRILOOP_DH, HAIRPIN_TETRALOOP_DH,
        )
        dict_tables = {
            "stack": (p.STACK, _stack_dh_rna()),
            "hairpin_triloop": (p.HAIRPIN_TRILOOP, HAIRPIN_TRILOOP_DH),
            "hairpin_tetraloop": (p.HAIRPIN_TETRALOOP, HAIRPIN_TETRALOOP_DH),
        }
        # Unlike DNA, Turner-2004 RNA loop-initiation ΔH is non-zero (curated in
        # _rna_enthalpy_generated.py, ΔG-validated against these size arrays).
        size_tables = {
            "hairpin_size": (p.HAIRPIN_SIZE, HAIRPIN_SIZE_DH),
            "bulge_size": (p.BULGE_SIZE, BULGE_SIZE_DH),
            "interior_size": (p.INTERIOR_SIZE, INTERIOR_SIZE_DH),
        }
        wobble = True
    else:
        raise ValueError(f"material must be 'dna' or 'rna', got {material!r}")

    new_dG: dict[str, Any] = {}
    for name, (g_dict, h_dict) in dict_tables.items():
        new_dG[name] = {
            k: _blend_scalar(float(gv), _maybe_float(h_dict.get(k)), frac)
            for k, gv in g_dict.items()
        }
    for name, (g_arr, h_arr) in size_tables.items():
        g = np.asarray(g_arr, dtype=float)
        if h_arr is None:
            new_dG[name] = g * frac                     # ΔH = 0 (entropic)
        else:
            new_dG[name] = g * frac + np.asarray(h_arr, dtype=float) * (1.0 - frac)

    return ParameterSet(
        name=f"native-{mat}@{celsius:g}C",
        material=material.upper(),
        default_wobble_pairing=wobble,
        dG=new_dG,
        dH={},
        source_path=None,
        comment=f"native module constants, temperature-blended to {celsius} °C",
    )
