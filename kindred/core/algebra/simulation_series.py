"""Helpers for evaluating algebra observables against simulation time series."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import numpy as np

from kindred.core.algebra.errors import AlgebraNameError
from kindred.core.algebra.evaluator import (
    AlgebraObservableError,
    EvaluationContext,
    evaluate_block_partial,
)
from kindred.core.algebra.parser import (
    AlgebraBlock,
    _iter_identifiers,
    _iter_species_refs,
)
from kindred.core.algebra.symbols import SymbolTable

__all__ = [
    "let_dependency_facts",
    "observable_parameter_roots",
    "reject_unknown_species_references",
]


def reject_unknown_species_references(
    block: AlgebraBlock,
    *,
    species_names: set[str],
) -> None:
    """Refuse a param or let line that reads a species the mechanism does not have."""
    for stmt in block.lines:
        unknown_species = sorted(
            set(_iter_species_refs(stmt.expr)) - species_names
        )
        if unknown_species:
            raise AlgebraNameError(
                "Unknown species " + ", ".join(repr(name) for name in unknown_species) + ".",
                line=stmt.line, col=stmt.col, line_text=stmt.line_text,
            )


def observable_closure(block: AlgebraBlock, names: Iterable[str]) -> frozenset[str]:
    """The retained lets the named outputs read, the named lets included."""
    statements = {stmt.name: stmt for stmt in block.lines}
    closure: set[str] = set()
    pending = [name for name in names if name in statements]
    while pending:
        current = pending.pop()
        if current not in closure:
            closure.add(current)
            pending.extend(name for name in _iter_identifiers(statements[current].expr) if name in statements)
    return frozenset(closure)


def observable_species_refs(block: AlgebraBlock, name: str, kind: str) -> frozenset[str]:
    """The species one retained observable closure reads, of ``kind``."""
    statements = {stmt.name: stmt for stmt in block.lines}
    return frozenset(
        species for let in observable_closure(block, (name,))
        for species in _iter_species_refs(statements[let].expr, kind)
    )


def observable_parameter_roots(block: AlgebraBlock, name: str) -> tuple[str, ...]:
    """External identifiers required by one retained observable closure."""
    statements = {stmt.name: stmt for stmt in block.lines}
    roots: set[str] = set()
    visited: set[str] = set()

    def visit(current: str) -> None:
        if current in visited:
            return
        visited.add(current)
        if current not in statements:
            roots.add(current)
            return
        for dependency in _iter_identifiers(statements[current].expr):
            visit(dependency)

    visit(name)
    return tuple(sorted(roots))


def let_dependency_facts(spec, name: str) -> dict[str, object]:
    """What a let that cannot be calculated depends on: the independent inputs
    and starting concentrations its closure reads, and whether it reads a
    current concentration."""
    roots = observable_parameter_roots(spec.observable_block, name)
    starting = observable_species_refs(spec.observable_block, name, "init")
    return {
        "parameter_names": tuple(sorted({*spec.input_dependencies(roots), *(f"[{species}]_0" for species in starting)})),
        "reads_species": bool(observable_species_refs(spec.observable_block, name, "now")),
    }


def _build_simulation_context(
    *,
    t: np.ndarray,
    species_series: Dict[str, np.ndarray],
    initials: Dict[str, float],
    symbol_resolver: Callable[[str, int], float | None] | None,
    runtime_symbol_names: frozenset[str],
    cancellation_check: Callable[[], bool] | None,
) -> EvaluationContext:
    """Build one row-aligned context for the retained Algebra program."""
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    return EvaluationContext(
        t=t_arr,
        species_series={k: np.asarray(v, dtype=float).reshape(-1) for k, v in species_series.items()},
        initials={str(k): float(v) for k, v in initials.items()},
        symtab=SymbolTable(),
        symbol_resolver=symbol_resolver,
        runtime_symbol_names=runtime_symbol_names,
        cancellation_check=cancellation_check,
    )


def evaluate_algebra_block_for_simulation_partial(
    block: AlgebraBlock,
    *,
    t: np.ndarray,
    species_series: Dict[str, np.ndarray],
    initials: Dict[str, float],
    symbol_resolver: Callable[[str, int], float | None],
    runtime_symbol_names: frozenset[str],
    cancellation_check: Callable[[], bool] | None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], list[AlgebraObservableError]]:
    """Evaluate retained observables while preserving per-observable failures."""
    ctx = _build_simulation_context(
        t=t,
        species_series=species_series,
        initials=initials,
        symbol_resolver=symbol_resolver,
        runtime_symbol_names=runtime_symbol_names,
        cancellation_check=cancellation_check,
    )
    series, scalars, errors = evaluate_block_partial(block, ctx)
    series_out = {str(name): np.asarray(values, dtype=float).reshape(-1) for name, values in series.items()}
    scalars_out: Dict[str, float] = {}
    for name, value in scalars.items():
        scalars_out[str(name)] = float(value)
    return series_out, scalars_out, list(errors)
