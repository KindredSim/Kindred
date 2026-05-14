"""Helpers for evaluating algebra observables against simulation time series."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np

from kindred.core.algebra.evaluator import (
    AlgebraObservableError,
    EvaluationContext,
    evaluate_block,
    evaluate_block_partial,
)
from kindred.core.algebra.parser import (
    AlgebraBlock,
    BinaryNode,
    CallNode,
    ExprNode,
    IdentNode,
    LetStatement,
    SpeciesRefNode,
    UnaryNode,
    parse_algebra,
)
from kindred.core.algebra.identifier_canonicalization import canonicalize_algebra_block
from kindred.core.algebra.printer import algebra_block_to_source
from kindred.core.algebra.symbol_table import build_algebra_symbol_table
from kindred.core.simulator.parameter_algebra_spec import (
    classify_parameter_algebra_declaration,
    invalid_parameter_algebra_identifier_reference_message,
    is_protected_indexed_parameter_identifier,
    is_protected_step_key_identifier,
)
from kindred.core.simulator.parameter_namespace import MechanismParameterNamespace

logger = logging.getLogger(__name__)

__all__ = [
    "build_algebra_symbol_table",
    "compile_algebra_observables",
    "evaluate_algebra_series_for_simulation_with_errors",
    "evaluate_compiled_algebra_series_for_simulation",
    "evaluate_algebra_series_for_simulation",
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


def _reject_invalid_identifier_references(
    block: AlgebraBlock,
    *,
    mechanism_namespace: MechanismParameterNamespace | None = None,
) -> None:
    for stmt in block.lines:
        for name in sorted(set(_iter_identifiers(stmt.expr))):
            message = invalid_parameter_algebra_identifier_reference_message(
                name,
                mechanism_namespace=mechanism_namespace,
                reject_unresolved_protected_indexed=mechanism_namespace is None,
            )
            if message is not None:
                raise ValueError(message)


@dataclass(frozen=True)
class CompiledAlgebraSeries:
    """Pre-parsed algebra observables for repeated evaluation."""

    processed_text: str
    block: AlgebraBlock
    observable_names: Tuple[str, ...]
    time_ref_statements: Tuple[LetStatement, ...]


def _preprocess_algebra_text(
    algebra_text: str,
    *,
    mechanism_namespace: MechanismParameterNamespace | None = None,
) -> str:
    algebra_lines_raw = str(algebra_text).strip().split("\n")
    algebra_lines_processed = ["# Algebra"]
    for line in algebra_lines_raw:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            code_part = stripped
            if "#" in stripped:
                bracket_depth = 0
                for i, char in enumerate(stripped):
                    if char == "[":
                        bracket_depth += 1
                    elif char == "]":
                        bracket_depth -= 1
                    elif char == "#" and bracket_depth == 0:
                        code_part = stripped[:i].rstrip()
                        break
            if not code_part:
                continue
            classification = classify_parameter_algebra_declaration(
                code_part,
                mechanism_namespace=mechanism_namespace,
                allow_non_algebra_bare_assignment=False,
            )
            if classification.kind == "param":
                continue
            if classification.kind == "let":
                if is_protected_indexed_parameter_identifier(classification.raw_name):
                    raise ValueError(
                        f"{classification.raw_name!r} is a protected indexed parameter identifier "
                        "and cannot be declared as an observable."
                    )
                if is_protected_step_key_identifier(classification.raw_name):
                    raise ValueError(
                        f"{classification.raw_name!r} is a step-local DSL key "
                        "and cannot be declared as an observable."
                    )
                algebra_lines_processed.append(code_part)
                continue
            if classification.kind == "invalid_step_key_identifier":
                raise ValueError(
                    f"{classification.raw_name!r} is a step-local DSL key "
                    "and cannot be declared as an observable."
                )
            if classification.kind == "unsupported_bare_assignment":
                raise ValueError(
                    f"Bare algebra assignment {classification.raw_name!r} is not supported. "
                    "Use 'let name = expr' or 'param name = expr'."
                )
            algebra_lines_processed.append(code_part)
        elif stripped:
            algebra_lines_processed.append(stripped)
    return "\n".join(algebra_lines_processed) + "\n"


def compile_algebra_observables(
    algebra_text: str,
    *,
    mechanism_namespace: MechanismParameterNamespace | None = None,
) -> CompiledAlgebraSeries:
    """
    Compile the Algebra observables block once for repeated evaluation.

    Notes
    -----
    - `param ...` statements are skipped here (they are handled by parameter algebra).
    - Baseline time references ([A](T0)) are *not* rejected here; fitting callers
      should enforce any additional constraints.
    """
    processed = _preprocess_algebra_text(str(algebra_text), mechanism_namespace=mechanism_namespace)
    algebra_block = parse_algebra(processed)
    if mechanism_namespace is not None:
        algebra_block = canonicalize_algebra_block(algebra_block, mechanism_namespace=mechanism_namespace)
        processed = algebra_block_to_source(algebra_block)
    _reject_invalid_identifier_references(algebra_block, mechanism_namespace=mechanism_namespace)
    observable_names = tuple(str(stmt.name) for stmt in (algebra_block.lines or []))
    time_ref_stmts = tuple(stmt for stmt in (algebra_block.lines or []) if stmt.expr.has_time_ref())
    return CompiledAlgebraSeries(
        processed_text=processed,
        block=algebra_block,
        observable_names=observable_names,
        time_ref_statements=time_ref_stmts,
    )


def evaluate_compiled_algebra_series_for_simulation(
    mechanism: object,
    compiled: CompiledAlgebraSeries,
    *,
    t: np.ndarray,
    species_series: Dict[str, np.ndarray],
    initials: Dict[str, float],
    temperature_K: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Evaluate compiled algebra declarations against simulation outputs."""
    species_names = set(species_series.keys())
    symtab = build_algebra_symbol_table(mechanism)
    if temperature_K is not None:
        try:
            symtab.update_temperature(float(temperature_K))
        except Exception as exc:
            logger.debug("Failed to update Algebra temperature: %s", exc, exc_info=True)
    ctx = EvaluationContext(
        t=np.asarray(t, dtype=float).reshape(-1),
        species_series={k: np.asarray(v, dtype=float).reshape(-1) for k, v in species_series.items()},
        initials={str(k): float(v) for k, v in (initials or {}).items()},
        species_names=species_names,
        symtab=symtab,
        baseline=None,
    )

    series, scalars = evaluate_block(compiled.block, ctx)
    series_out = {str(name): np.asarray(values, dtype=float).reshape(-1) for name, values in (series or {}).items()}
    scalars_out: Dict[str, float] = {}
    for name, value in (scalars or {}).items():
        try:
            scalars_out[str(name)] = float(value)
        except Exception:
            scalars_out[str(name)] = float("nan")
    for name, values in list(series_out.items()):
        if values.size != ctx.t.size:
            # Best-effort: broadcast scalar-like series to match time grid.
            if values.size == 1 and ctx.t.size > 1 and math.isfinite(float(values[0])):
                series_out[name] = np.full_like(ctx.t, float(values[0]), dtype=float)
            else:
                logger.debug(
                    "Dropping algebra series '%s' due to shape mismatch (%s vs %s).",
                    name,
                    values.size,
                    ctx.t.size,
                )
                series_out.pop(name, None)
    return series_out, scalars_out


def _error_from_exception(name: str, exc: Exception) -> AlgebraObservableError:
    code = getattr(exc, "code", None)
    line = getattr(exc, "line", None)
    col = getattr(exc, "col", None)
    line_text = getattr(exc, "line_text", None)
    return AlgebraObservableError(
        name=str(name),
        exc_type=exc.__class__.__name__,
        message=str(exc),
        code=str(code) if code is not None else None,
        line=int(line) if line is not None else None,
        col=int(col) if col is not None else None,
        line_text=str(line_text) if line_text is not None else None,
    )


def evaluate_compiled_algebra_series_for_simulation_partial(
    mechanism: object,
    compiled: CompiledAlgebraSeries,
    *,
    t: np.ndarray,
    species_series: Dict[str, np.ndarray],
    initials: Dict[str, float],
    temperature_K: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], list[AlgebraObservableError]]:
    """Best-effort variant of evaluate_compiled_algebra_series_for_simulation()."""
    species_names = set(species_series.keys())
    symtab = build_algebra_symbol_table(mechanism)
    if temperature_K is not None:
        try:
            symtab.update_temperature(float(temperature_K))
        except Exception as exc:
            logger.debug("Failed to update Algebra temperature: %s", exc, exc_info=True)
    ctx = EvaluationContext(
        t=np.asarray(t, dtype=float).reshape(-1),
        species_series={k: np.asarray(v, dtype=float).reshape(-1) for k, v in species_series.items()},
        initials={str(k): float(v) for k, v in (initials or {}).items()},
        species_names=species_names,
        symtab=symtab,
        baseline=None,
    )

    series, scalars, errors = evaluate_block_partial(compiled.block, ctx)
    series_out = {str(name): np.asarray(values, dtype=float).reshape(-1) for name, values in (series or {}).items()}
    scalars_out: Dict[str, float] = {}
    for name, value in (scalars or {}).items():
        try:
            scalars_out[str(name)] = float(value)
        except Exception:
            scalars_out[str(name)] = float("nan")
    for name, values in list(series_out.items()):
        if values.size != ctx.t.size:
            # Best-effort: broadcast scalar-like series to match time grid.
            if values.size == 1 and ctx.t.size > 1 and math.isfinite(float(values[0])):
                series_out[name] = np.full_like(ctx.t, float(values[0]), dtype=float)
            else:
                logger.debug(
                    "Dropping algebra series '%s' due to shape mismatch (%s vs %s).",
                    name,
                    values.size,
                    ctx.t.size,
                )
                series_out.pop(name, None)
                errors = list(errors or [])
                errors.append(
                    AlgebraObservableError(
                        name=str(name),
                        exc_type="AlgebraTypeError",
                        message=f"shape mismatch: {values.size} vs {ctx.t.size}",
                        code=None,
                        line=None,
                        col=None,
                        line_text=None,
                    )
                )
    return series_out, scalars_out, list(errors or [])


def evaluate_algebra_series_for_simulation_with_errors(
    mechanism: object,
    *,
    t: np.ndarray,
    species_series: Dict[str, np.ndarray],
    initials: Dict[str, float],
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], list[AlgebraObservableError]]:
    """
    Best-effort evaluation of compiled algebra declarations extracted from the
    mechanism DSL against time series arrays.

    Returns (series, scalars, errors). This does not raise for per-observable evaluation errors.
    """
    meta = getattr(mechanism, "metadata", {}) or {}
    algebra_text = meta.get("algebra_text")
    if not algebra_text:
        return {}, {}, []

    try:
        from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

        compiled = compile_algebra_observables(
            str(algebra_text),
            mechanism_namespace=build_namespace_from_mechanism(mechanism),
        )
    except Exception as exc:
        return {}, {}, [_error_from_exception("__parse__", exc)]

    try:
        series, scalars, errors = evaluate_compiled_algebra_series_for_simulation_partial(
            mechanism,
            compiled,
            t=t,
            species_series=species_series,
            initials=initials,
            temperature_K=None,
        )
    except Exception as exc:
        return {}, {}, [_error_from_exception("__eval__", exc)]

    return series, scalars, errors


def evaluate_algebra_series_for_simulation(
    mechanism: object,
    *,
    t: np.ndarray,
    species_series: Dict[str, np.ndarray],
    initials: Dict[str, float],
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Evaluate compiled algebra declarations extracted from the mechanism DSL
    against time series arrays.

    Returns
    -------
    (series, scalars)
        series: dict of time-series outputs {name: y(t)}
        scalars: dict of scalar outputs {name: value}
    """
    series, scalars, _errors = evaluate_algebra_series_for_simulation_with_errors(
        mechanism,
        t=t,
        species_series=species_series,
        initials=initials,
    )
    return series, scalars
