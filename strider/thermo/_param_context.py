"""
Thread-local override channel for nearest-neighbor parameter tables.

The native McCaskill / Zuker DP code in :mod:`strider.thermo.ensemble` and
:mod:`strider.structure.mfe` reads its stacking, hairpin, bulge,
interior-loop, multi-loop, asymmetry, and terminal-penalty tables from the
module-level constants in :mod:`strider.thermo.parameters_dna` and
:mod:`strider.thermo.parameters_rna`.  When a caller wants to swap those
tables — typically by handing :class:`~strider.thermo.engine.ThermoEngine`
an explicit ``parameter_set`` argument — we expose the override via this
module's :class:`param_context` context manager.

The helpers are intentionally tiny and zero-allocating on the default code
path: :func:`get_override` returns ``None`` when no override is active and
the DP functions fall straight through to the constants, so callers that
never opt in pay no measurable runtime cost.

Reference: a similar context-variable approach is used in NumPy's
``threadsafe_random`` and in the PyTorch ``set_grad_enabled`` family for
the same reason — push optional per-call configuration through call sites
that you do not want to widen with extra keyword arguments.
"""

from __future__ import annotations

import contextvars
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strider.thermo.parameters import ParameterSet


_param_override: contextvars.ContextVar["ParameterSet | None"] = contextvars.ContextVar(
    "strider_param_override", default=None,
)


@lru_cache(maxsize=8)
def _resolve_name(name: str) -> "ParameterSet":
    """Resolve a parameter-set name (e.g. ``"mathews2004-dna"``) to its
    :class:`ParameterSet`.  Memoized so repeated overrides by name never
    re-parse from disk on the hot path."""
    from strider.thermo.parameters import load_parameters

    return load_parameters(name)


def get_override() -> "ParameterSet | None":
    """Return the currently active :class:`ParameterSet` override, or ``None``."""
    return _param_override.get()


class param_context:
    """
    Activate a :class:`ParameterSet` override for the duration of a ``with`` block.

    Example
    -------
    >>> from strider.thermo._param_context import param_context, get_override
    >>> with param_context(my_paramset):
    ...     # energy lookups inside this block see my_paramset
    ...     ...
    >>> # outside the block the override is cleared automatically
    >>> get_override() is None
    True

    The context manager is a no-op (and skips the ``ContextVar.set`` /
    ``reset`` calls) when ``params`` is ``None``, so the engine's default
    code path remains zero-overhead.
    """

    __slots__ = ("params", "_token")

    def __init__(self, params: "ParameterSet | str | None") -> None:
        # Accept either a ParameterSet instance or a resolvable name (memoized
        # load) so callers never have to special-case the two forms.
        if isinstance(params, str):
            params = _resolve_name(params)
        self.params = params
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "param_context":
        if self.params is not None:
            self._token = _param_override.set(self.params)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._token is not None:
            _param_override.reset(self._token)
            self._token = None


def lookup_table(
    name: str,
    fallback,
):
    """
    Return ``override.dG[name]`` if a per-call override is active and exposes
    the requested sub-table, otherwise the supplied module-constant
    ``fallback``.  The two-arg signature keeps the call site to a single
    expression so the DP inner loops stay compact.

    Hot path: this runs ~50k times per folding DP fill on short oligos, so it
    fast-paths to a single ContextVar read and, at most, one dict lookup.
    """
    override = _param_override.get()
    return fallback if override is None else override.dG.get(name, fallback)


def lookup_scalar(name: str, fallback: float) -> float:
    """Same as :func:`lookup_table` but coerces the value to ``float``."""
    override = _param_override.get()
    if override is None:
        return fallback
    val = override.dG.get(name)
    return fallback if val is None else float(val)
