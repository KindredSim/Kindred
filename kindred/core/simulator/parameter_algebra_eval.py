"""Runtime evaluation of the admitted parameter Algebra program.

The parser owns admission, canonical AST retention, and the parameter
dependency mapping.  This module only consumes those retained facts; it does
not parse source text or rediscover dependencies from expression trees.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Dict

import numpy as np

from kindred.core.algebra.errors import AlgebraError
from kindred.core.algebra.evaluator import EvaluationContext, evaluate_block
from kindred.core.algebra.parser import AlgebraBlock
from kindred.core.algebra.symbols import SymbolTable
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra_spec import (
    ParameterAlgebraSpec,
)

__all__ = [
    "evaluate_parameter_algebra",
    "evaluate_parameter_algebra_in_context",
]


def _retained_block(
    spec: ParameterAlgebraSpec,
    names: Sequence[str],
) -> AlgebraBlock:
    retained = spec.parameter_block
    if retained is None:
        raise DSLError(
            "Parameter algebra has no retained admitted AST.",
            suggestion="Re-admit the mechanism source before evaluating parameter algebra.",
        )
    statements_by_name = {
        str(statement.name): statement
        for statement in retained.lines
    }
    ordered_names = tuple(str(name) for name in names)
    missing = [name for name in ordered_names if name not in statements_by_name]
    if missing:
        raise DSLError(
            "Parameter algebra is missing retained assignment(s): "
            + ", ".join(repr(name) for name in missing)
        )
    lines = tuple(statements_by_name[name] for name in ordered_names)
    static_values = retained.static_value_map()
    return AlgebraBlock(
        lines=lines,
        ast=tuple(statement.expr for statement in lines),
        static_values={
            name: value
            for name, value in static_values.items()
            if name in ordered_names
        },
        evaluation_order=ordered_names,
    )


def _symbol_table(
    spec: ParameterAlgebraSpec,
    base_values: Mapping[str, float],
) -> SymbolTable:
    symtab = SymbolTable()
    admitted_names = (
        set(spec.mechanism_param_names)
        | set(spec.scalar_input_names)
        | set(spec.param_assignment_names())
    )
    for raw_name, raw_value in base_values.items():
        name = str(raw_name)
        if name not in admitted_names or name in symtab.protected_names():
            continue
        try:
            symtab.define_user(name, float(raw_value), species_names=set())
        except (TypeError, ValueError, OverflowError) as exc:
            raise DSLError(f"Invalid parameter algebra input {name!r}: {exc}") from exc
    return symtab


def _evaluate_names(
    spec: ParameterAlgebraSpec,
    names: Sequence[str],
    *,
    base_values: Mapping[str, float],
) -> Dict[str, float]:
    if not names:
        return {}
    block = _retained_block(spec, names)
    context = EvaluationContext(
        t=np.array([0.0]),
        species_series={},
        initials={},
        species_names=set(),
        symtab=_symbol_table(spec, base_values),
    )
    try:
        series, scalars = evaluate_block(block, context)
    except AlgebraError as exc:
        assignment = next(
            (
                item
                for item in spec.param_statements
                if int(getattr(item, "line_number", 0) or 0)
                == int(getattr(exc, "line", 0) or 0)
            ),
            None,
        )
        if assignment is not None:
            raise DSLError(
                str(exc),
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            ) from exc
        raise DSLError(str(exc)) from exc

    values: Dict[str, float] = {}
    for name in names:
        if name in scalars:
            value = float(scalars[name])
        else:
            values_array = series.get(name)
            if values_array is None or values_array.size < 1:
                raise DSLError(f"Failed to evaluate parameter assignment {name!r}.")
            value = float(values_array[0])
        if not math.isfinite(value):
            raise DSLError(
                f"Parameter assignment {name!r} evaluated to a non-finite value: {value!r}"
            )
        values[name] = value
    return values


def evaluate_parameter_algebra_in_context(
    spec: ParameterAlgebraSpec,
    *,
    base_values: Dict[str, float],
) -> Dict[str, float]:
    """Evaluate all admitted assignments using the retained graph and AST."""

    if not isinstance(spec, ParameterAlgebraSpec):
        raise TypeError("spec must be a ParameterAlgebraSpec.")
    mapping = spec.parameter_dependencies
    if not mapping:
        return {}

    independent_names = tuple(
        name
        for name in mapping
        if spec.parameter_assignment_kind(name) == "independent scalar"
    )
    missing_defaults = tuple(
        name for name in independent_names if name not in base_values
    )
    if missing_defaults:
        defaults = _evaluate_names(
            spec,
            missing_defaults,
            base_values=base_values,
        )
        base_values.update(defaults)

    dependent_names = tuple(
        name
        for name in mapping
        if name not in independent_names
    )
    derived = _evaluate_names(
        spec,
        dependent_names,
        base_values=base_values,
    )
    return {
        name: float(base_values[name])
        for name in independent_names
    } | derived


def evaluate_parameter_algebra(
    spec: ParameterAlgebraSpec,
    *,
    base_values: Dict[str, float],
) -> Dict[str, float]:
    """Evaluate only mechanism-bound and derived scalar assignments."""

    values = evaluate_parameter_algebra_in_context(
        spec,
        base_values=base_values,
    )
    return {
        name: value
        for name, value in values.items()
        if spec.parameter_assignment_kind(name) != "independent scalar"
    }
