"""Helpers for evaluating algebra observables against simulation time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Tuple

import numpy as np

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
    IdentNode,
    SpeciesRefNode,
    UnaryNode,
)
from kindred.core.algebra.symbols import SymbolTable

__all__ = [
    "compiled_algebra_for_mechanism",
    "compile_algebra_observables",
    "prepare_algebra_stop_evaluator",
]


def _iter_identifiers(expr: ExprNode):
    if isinstance(expr, IdentNode):
        yield str(expr.name)
        return
    if isinstance(expr, SpeciesRefNode):
        return
    if isinstance(expr, UnaryNode):
        yield from _iter_identifiers(expr.rhs)
        return
    if isinstance(expr, BinaryNode):
        yield from _iter_identifiers(expr.lhs)
        yield from _iter_identifiers(expr.rhs)
        return
    if isinstance(expr, CallNode):
        for arg in expr.args:
            yield from _iter_identifiers(arg)
        return


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


def _reject_unknown_species_references(
    block: AlgebraBlock,
    *,
    species_names: set[str] | None = None,
) -> None:
    for stmt in block.lines:
        unknown_species = sorted(
            set(_iter_species_refs(stmt.expr)) - set(species_names or ())
        )
        if unknown_species:
            raise ValueError(
                "Unknown species in Algebra expression: "
                + ", ".join(repr(name) for name in unknown_species)
            )


@dataclass(frozen=True)
class CompiledAlgebraSeries:
    """Pre-parsed algebra observables for repeated evaluation."""

    block: AlgebraBlock
    observable_names: Tuple[str, ...]

    def parameter_roots(self, name: str) -> tuple[str, ...]:
        """External identifiers required by one retained observable closure."""
        statements = {stmt.name: stmt for stmt in self.block.lines}
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


def compiled_algebra_for_mechanism(
    mechanism: object,
) -> CompiledAlgebraSeries | None:
    """Return the Algebra program retained by the accepted mechanism parse."""
    return getattr(mechanism, "_compiled_algebra_series", None)


def compile_algebra_observables(
    *,
    retained_spec: object,
    species_names: set[str] | None = None,
) -> CompiledAlgebraSeries:
    """Compile the retained observable AST for repeated evaluation.

    The accepted DSL admission owns parsing and supplies the partitioned block;
    this boundary has no raw-source parsing route.
    """
    algebra_block = retained_spec.observable_block
    _reject_unknown_species_references(algebra_block, species_names=species_names)
    observable_names = tuple(str(stmt.name) for stmt in (algebra_block.lines or []))
    return CompiledAlgebraSeries(
        block=algebra_block,
        observable_names=observable_names,
    )


def prepare_algebra_stop_evaluator(
    compiled: CompiledAlgebraSeries,
    *,
    target_names: Tuple[str, ...],
    species_names: Tuple[str, ...],
    initials: Mapping[str, float],
    symbol_values: Mapping[str, float],
    runtime_symbol_names: set[str],
) -> PreparedAlgebraStopEvaluator:
    """Bind enabled stopping targets directly to the accepted retained AST."""
    return PreparedAlgebraStopEvaluator(
        compiled.block,
        target_names=target_names,
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
        initials={str(k): float(v) for k, v in (initials or {}).items()},
        species_names=set(species_series.keys()),
        symtab=symtab,
        symbol_series=runtime_series,
        symbol_values={str(name): float(value) for name, value in symbol_values.items()},
        symbol_resolver=symbol_resolver,
        runtime_symbol_names=runtime_symbol_names,
        cancellation_check=cancellation_check,
    )


def evaluate_compiled_algebra_series_for_simulation_partial(
    compiled: CompiledAlgebraSeries,
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
    series, scalars, errors = evaluate_block_partial(compiled.block, ctx)
    series_out = {str(name): np.asarray(values, dtype=float).reshape(-1) for name, values in (series or {}).items()}
    scalars_out: Dict[str, float] = {}
    for name, value in (scalars or {}).items():
        scalars_out[str(name)] = float(value)
    return series_out, scalars_out, list(errors or [])
