"""Helpers for evaluating algebra observables against simulation time series."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Tuple

import numpy as np

from kindred.core.algebra.errors import AlgebraNameError
from kindred.core.algebra.evaluator import (
    AlgebraObservableError,
    EvaluationContext,
    PreparedAlgebraStopEvaluator,
    evaluate_block_partial,
)
from kindred.core.algebra.parser import (
    AlgebraBlock,
    BinaryNode,
    CallNode,
    ExprNode,
    SpeciesRefNode,
    UnaryNode,
    _iter_identifiers,
)
from kindred.core.algebra.symbols import SymbolTable

__all__ = [
    "observable_parameter_roots",
    "prepare_algebra_stop_evaluator",
    "reject_unknown_species_references",
]


def _iter_species_refs(expr: ExprNode):
    if isinstance(expr, SpeciesRefNode):
        yield str(expr.name)
        return
    if isinstance(expr, UnaryNode):
        yield from _iter_species_refs(expr.rhs)
        return
    if isinstance(expr, BinaryNode):
        yield from _iter_species_refs(expr.lhs)
        yield from _iter_species_refs(expr.rhs)
        return
    if isinstance(expr, CallNode):
        for arg in expr.args:
            yield from _iter_species_refs(arg)
        return


def reject_unknown_species_references(
    block: AlgebraBlock,
    *,
    species_names: set[str],
) -> None:
    """Refuse a let line that reads a species the mechanism does not have."""
    for stmt in block.lines:
        unknown_species = sorted(
            set(_iter_species_refs(stmt.expr)) - species_names
        )
        if unknown_species:
            raise AlgebraNameError(
                "Unknown species " + ", ".join(repr(name) for name in unknown_species) + ".",
                line=stmt.line, col=stmt.col, line_text=stmt.line_text,
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


def prepare_algebra_stop_evaluator(
    block: AlgebraBlock,
    *,
    species_names: Tuple[str, ...],
    initials: Mapping[str, float],
    symbol_values: Mapping[str, float],
    runtime_symbol_names: set[str],
) -> PreparedAlgebraStopEvaluator:
    """Bind stopping to the accepted retained AST; the caller admits each target."""
    return PreparedAlgebraStopEvaluator(
        block,
        species_names=species_names,
        initials=initials,
        symbol_values=symbol_values,
        runtime_symbol_names=runtime_symbol_names,
    )


def _build_simulation_context(
    *,
    t: np.ndarray,
    species_series: Dict[str, np.ndarray],
    initials: Dict[str, float],
    symbol_series: Mapping[str, np.ndarray | float],
    symbol_values: Mapping[str, float],
    symbol_resolver: Callable[[str], float | np.ndarray | None] | None,
    runtime_symbol_names: frozenset[str],
    cancellation_check: Callable[[], bool] | None,
) -> EvaluationContext:
    """Build one row-aligned context for the retained Algebra program."""
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    runtime_series: Dict[str, np.ndarray | float] = dict(symbol_series)
    runtime_series.setdefault("t", t_arr)
    symtab = SymbolTable()
    return EvaluationContext(
        t=t_arr,
        species_series={k: np.asarray(v, dtype=float).reshape(-1) for k, v in species_series.items()},
        initials={str(k): float(v) for k, v in initials.items()},
        species_names=set(species_series.keys()),
        symtab=symtab,
        symbol_series=runtime_series,
        symbol_values={str(name): float(value) for name, value in symbol_values.items()},
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
    symbol_series: Mapping[str, np.ndarray | float],
    symbol_values: Mapping[str, float],
    symbol_resolver: Callable[[str], float | np.ndarray | None] | None = None,
    runtime_symbol_names: frozenset[str] = frozenset(),
    cancellation_check: Callable[[], bool] | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], list[AlgebraObservableError]]:
    """Evaluate retained observables while preserving per-observable failures."""
    ctx = _build_simulation_context(
        t=t,
        species_series=species_series,
        initials=initials,
        symbol_series=symbol_series,
        symbol_values=symbol_values,
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
