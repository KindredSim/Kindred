"""
Dynamic evaluator for the Algebra DSL with per-time memoization.

Current contract
----------------
- Dynamic evaluation per time t with memoization keyed by (t, symbol).
- [A] pulls series at time t; [A]_0 pulls initial concentration.
- Builtins/helpers and protected symbols: see symbols.py and constants.py.
- No shadowing of species or builtins for user-defined algebra symbols (E120).
- Error taxonomy uses the algebra error classes and codes defined in errors.py.
- Static vs dynamic: expressions independent of [X] terms fold to scalars.

Out of scope
------------
- ODE assembly, solver integration, plotting. This module only evaluates
  a compiled internal AlgebraBlock against provided series/initials and returns
  algebra scalars and algebra time series.

Constraints
-----------
- Pure in-memory; no filesystem, no cwd, no network, no registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

from .errors import (
    AlgebraBoolCastError,
    AlgebraDomainError,
    AlgebraError,
    AlgebraNameError,
    AlgebraShadowError,
    AlgebraTypeError,
    AlgebraZeroDivError,
)
from .parser import (
    AlgebraBlock,
    BinaryNode,
    CallNode,
    ExprNode,
    IdentNode,
    LetStatement,
    NumberNode,
    SpeciesRefNode,
    UnaryNode,
)
from .symbols import SymbolTable

logger = logging.getLogger(__name__)


__all__ = [
    "EvaluationContext",
    "AlgebraObservableError",
    "evaluate_block",
    "evaluate_block_partial",
]


# ----------------------------- context model ---------------------------------


@dataclass
class EvaluationContext:
    """
    Inputs required to evaluate an AlgebraBlock.

    Attributes
    ----------
    t : ndarray
        Row-aligned evaluation grid; finite and nondecreasing; shape (N,).
    species_series : mapping
        Species time series on native grid: name -> ndarray (len N).
    initials : mapping
        Initial concentrations: name -> float.
    species_names : set
        Declared species names (for shadow checks).
    symtab : SymbolTable
        Protected symbols and scalar user symbols. `T` is read-only and may be
        updated by the application, not by user algebra.
    symbol_series : mapping, optional
        Row-aligned runtime symbols.  These values are resolved before the
        static symbol table and are intended for step-local ``t``, ``T``,
        mechanism coefficients, and changing scalar inputs.  A scalar value
        is broadcast to the supplied grid; an array must have exactly the
        same length as ``t``.
    symbol_values : mapping, optional
        The admitted static scalar snapshot available to observable ``let``
        expressions. Fit candidates must prepare this complete snapshot before
        solving; row evaluation never overrides retained ``param`` assignments.
    """
    t: np.ndarray
    species_series: Mapping[str, np.ndarray]
    initials: Mapping[str, float]
    species_names: Set[str]
    symtab: SymbolTable
    symbol_series: Mapping[str, np.ndarray] = field(default_factory=dict)
    symbol_values: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AlgebraObservableError:
    """
    Structured error for a single failed algebra observable (let statement).

    This is intended for UI/diagnostics. It is *not* used to control evaluation
    flow in strict mode (evaluate_block).
    """

    name: str
    exc_type: str
    message: str
    code: Optional[str] = None
    line: Optional[int] = None
    col: Optional[int] = None
    line_text: Optional[str] = None

    @staticmethod
    def from_exception(name: str, exc: Exception, *, stmt: LetStatement) -> "AlgebraObservableError":
        code = getattr(exc, "code", None)
        line = getattr(exc, "line", None) or getattr(stmt, "line", None)
        col = getattr(exc, "col", None) or getattr(stmt, "col", None)
        line_text = getattr(exc, "line_text", None) or getattr(stmt, "line_text", None)
        return AlgebraObservableError(
            name=str(name),
            exc_type=exc.__class__.__name__,
            message=str(exc),
            code=str(code) if code is not None else None,
            line=int(line) if line is not None else None,
            col=int(col) if col is not None else None,
            line_text=str(line_text) if line_text is not None else None,
        )


# ----------------------------- utilities -------------------------------------


def _as_float(x: Any, *, stmt: LetStatement) -> float:
    """Coerce to float but reject booleans (E170)."""
    if isinstance(x, bool):
        raise AlgebraBoolCastError("boolean in numeric-only context", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
    try:
        xf = float(x)
    except Exception:
        logger.debug(f"Failed to convert value to float: {x}", exc_info=True)
        raise AlgebraTypeError("non scalar where scalar required", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
    if not np.isfinite(xf):
        raise AlgebraTypeError("non-finite result in numeric context", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
    return xf


def _is_time_varying(expr: ExprNode, sym_timevary: Set[str]) -> bool:
    """Decide if an expression depends on time t."""
    if isinstance(expr, SpeciesRefNode):
        return expr.kind == "now"
    if isinstance(expr, IdentNode):
        return expr.name in sym_timevary
    if isinstance(expr, (NumberNode,)):
        return False
    if isinstance(expr, UnaryNode):
        return _is_time_varying(expr.rhs, sym_timevary)
    if isinstance(expr, BinaryNode):
        return _is_time_varying(expr.lhs, sym_timevary) or _is_time_varying(expr.rhs, sym_timevary)
    if isinstance(expr, CallNode):
        return any(_is_time_varying(a, sym_timevary) for a in expr.args)
    return True


# ----------------------------- evaluator core --------------------------------


class _Evaluator:
    def __init__(self, block: AlgebraBlock, ctx: EvaluationContext) -> None:
        self.block = block
        self.static_values = block.static_value_map()
        self.ctx = ctx
        self.N = int(ctx.t.size)
        self.symbol_series: Dict[str, np.ndarray] = {}
        for name, values in (ctx.symbol_series or {}).items():
            arr = np.asarray(values, dtype=float)
            if arr.ndim == 0:
                arr = np.full(self.N, float(arr), dtype=float)
            self.symbol_series[str(name)] = arr.reshape(-1)
        self.symbol_values: Dict[str, float] = {
            str(name): float(value)
            for name, value in (ctx.symbol_values or {}).items()
        }
        # memo[(i, name)] = float
        self.memo: Dict[Tuple[int, str], float] = {}
        # results
        self.series: Dict[str, np.ndarray] = {}
        self.scalars: Dict[str, float] = {}
        # dependency classification for user-defined symbols
        # Runtime row-aligned symbols are dynamic roots even when their values
        # happen to be constant over one particular grid.
        self.timevary: Set[str] = {"t"} | set(self.symbol_series)

    # Public entry
    def run(self) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        # Pre-classify time variance based on AST shape and dependencies
        for stmt in self.block.statements_in_evaluation_order():
            name = stmt.name
            # shadowing guards
            self._guard_shadow(name, stmt)
            # dynamic if it contains a current-state reference or depends on
            # prior dynamic symbols
            is_dyn = _is_time_varying(stmt.expr, self.timevary)
            if is_dyn:
                self.timevary.add(name)

            if not is_dyn and name in self.static_values:
                # Fully static fold available
                val = _as_float(self.static_values[name], stmt=stmt)
                self.scalars[name] = val
                self.ctx.symtab.define_user(name, val, species_names=self.ctx.species_names)
            else:
                # Compute series across t with memo
                arr = np.empty(self.N, dtype=float)
                for i in range(self.N):
                    arr[i] = self._eval_expr(stmt.expr, i, stmt=stmt)
                self.series[name] = arr

        return self.series, self.scalars

    def run_partial(self) -> Tuple[Dict[str, np.ndarray], Dict[str, float], List[AlgebraObservableError]]:
        """
        Best-effort evaluation: continue after per-observable failures.

        - Each `let name = expr` is evaluated independently.
        - On failure, the observable is omitted from outputs and an error record is returned.
        - Strict invariants (shape checks, shadowing guards) still apply per statement.
        """
        errors: List[AlgebraObservableError] = []

        for stmt in self.block.statements_in_evaluation_order():
            name = stmt.name
            try:
                self._guard_shadow(name, stmt)
                is_dyn = _is_time_varying(stmt.expr, self.timevary)
                if is_dyn:
                    self.timevary.add(name)

                if (not is_dyn) and (name in self.static_values):
                    val = _as_float(self.static_values[name], stmt=stmt)
                    self.scalars[name] = val
                    self.ctx.symtab.define_user(name, val, species_names=self.ctx.species_names)
                    continue

                arr = np.empty(self.N, dtype=float)
                for i in range(self.N):
                    arr[i] = self._eval_expr(stmt.expr, i, stmt=stmt)
                self.series[name] = arr
            except Exception as exc:
                errors.append(AlgebraObservableError.from_exception(name, exc, stmt=stmt))
                self.series.pop(name, None)
                self.scalars.pop(name, None)
                continue

        return self.series, self.scalars, errors

    # Guards
    def _guard_shadow(self, name: str, stmt: LetStatement) -> None:
        # Protected or species names are forbidden
        if name in self.ctx.symtab.protected_names():
            raise AlgebraShadowError(f"attempted shadowing of protected symbol {name!r}", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
        if name in self.ctx.symtab.functions().keys():
            raise AlgebraShadowError(f"attempted shadowing of builtin function {name!r}", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
        if name in self.ctx.species_names:
            raise AlgebraShadowError(f"attempted shadowing of species {name!r}", line=stmt.line, col=stmt.col, line_text=stmt.line_text)

    # Eval
    def _eval_expr(self, node: ExprNode, i: int, *, stmt: LetStatement) -> float | bool:
        # Number
        if isinstance(node, NumberNode):
            return float(node.value)

        # Identifier: may be scalar from symtab or user-defined scalar/series
        if isinstance(node, IdentNode):
            name = node.name
            # user-defined scalar
            if name in self.scalars:
                return float(self.scalars[name])
            # user-defined series: memoize by (i, name)
            if name in self.series:
                key = (i, name)
                if key in self.memo:
                    return self.memo[key]
                val = float(self.series[name][i])
                self.memo[key] = val
                return val
            if name == "t":
                return float(self.ctx.t[i])
            # Accepted runtime context takes precedence over the static table,
            # allowing T, t, coefficients, and changing scalar inputs to vary
            # independently at each result row.
            if name in self.symbol_series:
                return float(self.symbol_series[name][i])
            if name in self.symbol_values:
                return float(self.symbol_values[name])
            # protected or previously defined user scalar in symtab
            if self.ctx.symtab.has(name):
                return float(self.ctx.symtab.get(name))
            # unknown
            raise AlgebraNameError(f"unknown symbol {name!r}", line=stmt.line, col=stmt.col, line_text=stmt.line_text)

        # Species references
        if isinstance(node, SpeciesRefNode):
            nm = node.name
            if node.kind == "now":
                try:
                    return float(self.ctx.species_series[nm][i])
                except KeyError:
                    raise AlgebraNameError(f"unknown species {nm!r}", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
            if node.kind == "init":
                if nm not in self.ctx.initials:
                    raise AlgebraNameError(f"unknown species {nm!r}", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
                return float(self.ctx.initials[nm])
        # Unary
        if isinstance(node, UnaryNode):
            v = self._eval_expr(node.rhs, i, stmt=stmt)
            if node.op == "!":
                return not bool(v)
            # numeric
            x = _as_float(v, stmt=stmt)
            return +x if node.op == "+" else -x

        # Binary
        if isinstance(node, BinaryNode):
            # Short-circuit logicals
            if node.op == "||":
                lhsb = bool(self._eval_expr(node.lhs, i, stmt=stmt))
                if lhsb:
                    return True
                return bool(self._eval_expr(node.rhs, i, stmt=stmt))
            if node.op == "&&":
                lhsb = bool(self._eval_expr(node.lhs, i, stmt=stmt))
                if not lhsb:
                    return False
                return bool(self._eval_expr(node.rhs, i, stmt=stmt))

            # Numeric ops
            lv = _as_float(self._eval_expr(node.lhs, i, stmt=stmt), stmt=stmt)
            rv = _as_float(self._eval_expr(node.rhs, i, stmt=stmt), stmt=stmt)
            op = node.op
            try:
                if op == "+":
                    return lv + rv
                if op == "-":
                    return lv - rv
                if op == "*":
                    return lv * rv
                if op == "/":
                    if rv == 0.0:
                        raise AlgebraZeroDivError("division by zero", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
                    return lv / rv
                if op in ("**", "^"):
                    return float(lv ** rv)
                if op == "==":
                    return lv == rv
                if op == "!=":
                    return lv != rv
                if op == "<":
                    return lv < rv
                if op == "<=":
                    return lv <= rv
                if op == ">":
                    return lv > rv
                if op == ">=":
                    return lv >= rv
            except AlgebraZeroDivError:
                raise
            except Exception as e:
                logger.debug(f"Binary operation {op} failed", exc_info=True)
                # Generic math domain/type issues
                raise AlgebraDomainError(f"invalid operation: {e}", line=stmt.line, col=stmt.col, line_text=stmt.line_text)

        # Call
        if isinstance(node, CallNode):
            fn = self.ctx.symtab.functions().get(node.name)
            if fn is None:
                raise AlgebraNameError(f"unknown function {node.name!r}", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
            args: List[Any] = []
            for index, a in enumerate(node.args):
                av = self._eval_expr(a, i, stmt=stmt)
                if node.name == "ifelse" and index == 0:
                    args.append(av)
                else:
                    args.append(_as_float(av, stmt=stmt))
            try:
                out = float(fn(*args))
            except ZeroDivisionError:
                raise AlgebraZeroDivError("division by zero", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
            except ValueError as e:
                # e.g., sqrt of negative
                logger.debug(f"Function {node.name} raised ValueError", exc_info=True)
                raise AlgebraDomainError(str(e), line=stmt.line, col=stmt.col, line_text=stmt.line_text)
            except Exception as e:
                logger.debug(f"Function {node.name} call failed", exc_info=True)
                raise AlgebraTypeError(f"invalid function call: {e}", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
            if not np.isfinite(out):
                raise AlgebraTypeError("non-finite function result", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
            return out

        # Should not reach here
        raise AlgebraError("internal evaluator error", code="E150", line=stmt.line, col=stmt.col, line_text=stmt.line_text)


# ----------------------------- public API ------------------------------------


def evaluate_block(
    block: AlgebraBlock,
    ctx: EvaluationContext,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Evaluate an AlgebraBlock over ctx.t.

    Returns
    -------
    (series, scalars)
        series: dict of time-varying algebra series, one ndarray per name
        scalars: dict of algebra scalars

    Notes
    -----
    - The caller should merge species series with returned algebra series
      when constructing SimulationResult.
    - Memoization is internal and keyed by (t_index, symbol) for series references.
    """
    # Validate basic shapes once, early
    t = np.asarray(ctx.t, dtype=float)
    if t.ndim != 1 or t.size == 0 or np.any(~np.isfinite(t)) or np.any(np.diff(t) < 0):
        raise AlgebraTypeError("time grid t must be 1D, finite, and nondecreasing", line=1, col=1, line_text="# Algebra")
    for nm, arr in ctx.species_series.items():
        a = np.asarray(arr, dtype=float)
        if a.ndim != 1 or a.size != t.size or np.any(~np.isfinite(a)):
            raise AlgebraTypeError(f"series[{nm!r}] must be 1D, finite, length equal to len(t)", line=1, col=1, line_text="# Algebra")
    for nm, raw in (ctx.symbol_series or {}).items():
        a = np.asarray(raw, dtype=float)
        if a.ndim == 0:
            a = np.full(t.size, float(a), dtype=float)
        if a.ndim != 1 or a.size != t.size or np.any(~np.isfinite(a)):
            raise AlgebraTypeError(
                f"symbol_series[{nm!r}] must be scalar or 1D, finite, length equal to len(t)",
                line=1,
                col=1,
                line_text="# Algebra",
            )
    for nm, raw in (ctx.symbol_values or {}).items():
        try:
            value = float(raw)
        except Exception as exc:
            raise AlgebraTypeError(
                f"symbol_values[{nm!r}] must be numeric",
                line=1,
                col=1,
                line_text="# Algebra",
            ) from exc
        if not np.isfinite(value):
            raise AlgebraTypeError(
                f"symbol_values[{nm!r}] must be finite",
                line=1,
                col=1,
                line_text="# Algebra",
            )
    ev = _Evaluator(block, ctx)
    return ev.run()


def evaluate_block_partial(
    block: AlgebraBlock,
    ctx: EvaluationContext,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], List[AlgebraObservableError]]:
    """
    Best-effort variant of evaluate_block().

    Unlike evaluate_block(), this does not fail the entire compiled AlgebraBlock
    when a single observable fails. It returns successful outputs plus a
    structured error list.
    """
    # Validate basic shapes once, early (same behavior as strict mode).
    t = np.asarray(ctx.t, dtype=float)
    if t.ndim != 1 or t.size == 0 or np.any(~np.isfinite(t)) or np.any(np.diff(t) < 0):
        raise AlgebraTypeError("time grid t must be 1D, finite, and nondecreasing", line=1, col=1, line_text="# Algebra")
    for nm, arr in ctx.species_series.items():
        a = np.asarray(arr, dtype=float)
        if a.ndim != 1 or a.size != t.size or np.any(~np.isfinite(a)):
            raise AlgebraTypeError(f"series[{nm!r}] must be 1D, finite, length equal to len(t)", line=1, col=1, line_text="# Algebra")
    for nm, raw in (ctx.symbol_series or {}).items():
        a = np.asarray(raw, dtype=float)
        if a.ndim == 0:
            a = np.full(t.size, float(a), dtype=float)
        if a.ndim != 1 or a.size != t.size or np.any(~np.isfinite(a)):
            raise AlgebraTypeError(
                f"symbol_series[{nm!r}] must be scalar or 1D, finite, length equal to len(t)",
                line=1,
                col=1,
                line_text="# Algebra",
            )
    for nm, raw in (ctx.symbol_values or {}).items():
        try:
            value = float(raw)
        except Exception as exc:
            raise AlgebraTypeError(
                f"symbol_values[{nm!r}] must be numeric",
                line=1,
                col=1,
                line_text="# Algebra",
            ) from exc
        if not np.isfinite(value):
            raise AlgebraTypeError(
                f"symbol_values[{nm!r}] must be finite",
                line=1,
                col=1,
                line_text="# Algebra",
            )
    ev = _Evaluator(block, ctx)
    return ev.run_partial()
