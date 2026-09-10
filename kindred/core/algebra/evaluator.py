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
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

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
    "PreparedAlgebraStopEvaluator",
    "bind_scalar_expression_evaluator",
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
        Already evaluated static symbols. Missing scientific symbols can be
        resolved on first use inside the consuming observable's error boundary.
    symbol_resolver : callable, optional
        Resolve a scientific symbol to a scalar or row-aligned array, or None
        when the name is not a scientific parameter.
    runtime_symbol_names : set
        Row-dependent symbols, classified before their values are evaluated.
    """
    t: np.ndarray
    species_series: Mapping[str, np.ndarray]
    initials: Mapping[str, float]
    species_names: Set[str]
    symtab: SymbolTable
    symbol_series: Mapping[str, np.ndarray] = field(default_factory=dict)
    symbol_values: Mapping[str, float] = field(default_factory=dict)
    symbol_resolver: Callable[[str], float | np.ndarray | None] | None = None
    runtime_symbol_names: frozenset[str] = frozenset()


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


# ----------------------------- evaluator core --------------------------------


def bind_scalar_expression_evaluator(block: AlgebraBlock, values: dict[str, float]):
    """Bind admitted scalar AST arithmetic to one current numerical map."""
    context = EvaluationContext(
        t=np.array([0.0]), species_series={}, initials={},
        species_names=set(), symtab=SymbolTable(),
    )
    evaluator = _Evaluator(block, context)
    evaluator.symbol_values = values

    def evaluate(statement: LetStatement) -> float:
        return _as_float(evaluator._eval_expr(statement.expr, 0, stmt=statement), stmt=statement)

    return evaluate


class _Evaluator:
    def __init__(self, block: AlgebraBlock, ctx: EvaluationContext) -> None:
        self.block = block
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
        self.timevary: Set[str] = {"t"} | set(self.symbol_series) | set(ctx.runtime_symbol_names)

    def _scalar_let_names(self) -> Set[str]:
        """Classify the complete retained let DAG by row dependence.

        Parser static folding is intentionally conservative and only covers
        literal RHS trees.  A retained identifier can nevertheless refer to a
        scalar let several edges away, so classifying one statement from the
        current output dictionaries makes an equivalent expression a series.
        Resolve the retained dependency graph recursively instead; a let is
        row-varying only when its transitive closure reaches a current species,
        time, or row-aligned runtime symbol.
        """
        statements = {str(stmt.name): stmt for stmt in self.block.lines}
        state: Dict[str, bool] = {}

        def expr_dynamic(expr: ExprNode) -> bool:
            if isinstance(expr, SpeciesRefNode):
                return expr.kind == "now"
            if isinstance(expr, IdentNode):
                if expr.name in statements:
                    return let_dynamic(expr.name)
                return expr.name in self.timevary
            if isinstance(expr, NumberNode):
                return False
            if isinstance(expr, UnaryNode):
                return expr_dynamic(expr.rhs)
            if isinstance(expr, BinaryNode):
                return expr_dynamic(expr.lhs) or expr_dynamic(expr.rhs)
            if isinstance(expr, CallNode):
                return any(expr_dynamic(arg) for arg in expr.args)
            return True

        def let_dynamic(name: str) -> bool:
            if name in state:
                return state[name]
            # The parser retains and validates a topological order, so this is
            # only defensive protection for manually constructed blocks.
            state[name] = True
            state[name] = expr_dynamic(statements[name].expr)
            return state[name]

        for name in self.block.evaluation_order:
            let_dynamic(str(name))
        return {name for name, dynamic in state.items() if not dynamic}

    # Public entry
    def run(self) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        scalar_names = self._scalar_let_names()
        for stmt in self.block.statements_in_evaluation_order():
            name = stmt.name
            # shadowing guards
            self._guard_shadow(name, stmt)
            is_dyn = name not in scalar_names
            if is_dyn:
                self.timevary.add(name)

            if not is_dyn:
                # Evaluate scalar lets through the typed evaluator, including
                # transitive constants and immediate binary/domain failures.
                val = _as_float(self._eval_expr(stmt.expr, 0, stmt=stmt), stmt=stmt)
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
        scalar_names = self._scalar_let_names()

        for stmt in self.block.statements_in_evaluation_order():
            name = stmt.name
            try:
                self._guard_shadow(name, stmt)
                is_dyn = stmt.name not in scalar_names
                if is_dyn:
                    self.timevary.add(name)

                if not is_dyn:
                    val = _as_float(self._eval_expr(stmt.expr, 0, stmt=stmt), stmt=stmt)
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
            if self.ctx.symbol_resolver is not None:
                resolved = self.ctx.symbol_resolver(name)
                if resolved is not None:
                    arr = np.asarray(resolved, dtype=float)
                    if arr.ndim == 0:
                        value = _as_float(float(arr), stmt=stmt)
                        self.symbol_values[name] = value
                        return value
                    if arr.ndim != 1 or arr.size != self.N or np.any(~np.isfinite(arr)):
                        raise AlgebraTypeError(
                            f"symbol_series[{name!r}] must be finite and row-aligned",
                            line=stmt.line, col=stmt.col, line_text=stmt.line_text,
                        )
                    self.symbol_series[name] = arr
                    return float(arr[i])
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
                    return _as_float(lv + rv, stmt=stmt)
                if op == "-":
                    return _as_float(lv - rv, stmt=stmt)
                if op == "*":
                    return _as_float(lv * rv, stmt=stmt)
                if op == "/":
                    if rv == 0.0:
                        raise AlgebraZeroDivError("division by zero", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
                    return _as_float(lv / rv, stmt=stmt)
                if op in ("**", "^"):
                    return _as_float(lv ** rv, stmt=stmt)
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
            except (AlgebraTypeError, AlgebraZeroDivError):
                raise
            except OverflowError as exc:
                raise AlgebraTypeError(
                    "non-finite result in numeric context",
                    line=stmt.line,
                    col=stmt.col,
                    line_text=stmt.line_text,
                ) from exc
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


class _PointEvaluator(_Evaluator):
    """Resolve retained dependencies while sharing the ordinary arithmetic."""

    def __init__(self, prepared: PreparedAlgebraStopEvaluator, ctx: EvaluationContext):
        super().__init__(prepared.block, ctx)
        self.prepared = prepared

    def quantity(self, name: str) -> float:
        if name not in self.scalars:
            stmt = self.prepared.statements[name]
            try:
                self.scalars[name] = _as_float(
                    self._eval_expr(stmt.expr, 0, stmt=stmt), stmt=stmt
                )
            except AlgebraError as exc:
                raise ValueError(f"Algebra quantity {name!r}: {exc}") from exc
        return self.scalars[name]

    def _eval_expr(self, node: ExprNode, i: int, *, stmt: LetStatement) -> float | bool:
        if id(node) in self.prepared.static_values:
            return self.prepared.static_values[id(node)]
        if isinstance(node, IdentNode):
            if node.name in self.prepared.statements:
                return self.quantity(node.name)
            if (
                node.name in self.prepared.runtime_names
                and node.name != "t"
                and node.name not in self.symbol_series
            ):
                raise AlgebraNameError(
                    f"missing runtime symbol {node.name!r}",
                    line=stmt.line, col=stmt.col, line_text=stmt.line_text,
                )
        return super()._eval_expr(node, i, stmt=stmt)


class PreparedAlgebraStopEvaluator:
    """One run's selected retained LET targets and input-only subtree values.

    Eligibility is deliberately separate from ordinary result evaluation. The
    caller constructs this only for enabled stopping and supplies raw solver
    state and the correct one-sided runtime symbols at each evaluation.
    """

    def __init__(
        self,
        block: AlgebraBlock,
        *,
        target_names: Tuple[str, ...],
        species_names: Tuple[str, ...],
        initials: Mapping[str, float],
        symbol_values: Mapping[str, float],
        runtime_symbol_names: Set[str],
    ) -> None:
        self.block = block
        self.statements = {stmt.name: stmt for stmt in block.lines}
        self.species_names = species_names
        self.runtime_names = set(runtime_symbol_names) | {"t", "T"}
        self.static_values: Dict[int, float | bool] = {}
        self._dynamic: Dict[int, bool] = {}
        self._admitted: Set[int] = set()
        self._context = EvaluationContext(
            t=np.array([0.0]),
            species_series={},
            initials=dict(initials),
            species_names=set(species_names),
            symtab=SymbolTable(),
            symbol_values=dict(symbol_values),
        )
        static_evaluator = _PointEvaluator(self, self._context)
        for name in target_names:
            if name not in self.statements:
                raise ValueError(f"stop target {name!r}: unknown authored Algebra quantity")
            stmt = self.statements[name]
            try:
                self._admit(stmt.expr, stmt, static_evaluator)
                if not self._is_dynamic(stmt.expr):
                    static_evaluator.quantity(name)
            except (AlgebraError, ValueError) as exc:
                raise ValueError(f"stop target {name!r}: {exc}") from exc

    def _is_dynamic(self, node: ExprNode) -> bool:
        key = id(node)
        if key not in self._dynamic:
            if isinstance(node, SpeciesRefNode):
                dynamic = node.kind == "now"
            elif isinstance(node, IdentNode):
                dynamic = (
                    self._is_dynamic(self.statements[node.name].expr)
                    if node.name in self.statements
                    else node.name in self.runtime_names
                )
            elif isinstance(node, UnaryNode):
                dynamic = self._is_dynamic(node.rhs)
            elif isinstance(node, BinaryNode):
                dynamic = self._is_dynamic(node.lhs) or self._is_dynamic(node.rhs)
            elif isinstance(node, CallNode):
                dynamic = any(self._is_dynamic(arg) for arg in node.args)
            else:
                dynamic = False
            self._dynamic[key] = dynamic
        return self._dynamic[key]

    def _static_value(self, node: ExprNode, stmt: LetStatement, evaluator: _PointEvaluator) -> float | bool:
        key = id(node)
        if key not in self.static_values:
            value = evaluator._eval_expr(node, 0, stmt=stmt)
            self.static_values[key] = value if isinstance(value, bool) else _as_float(value, stmt=stmt)
        return self.static_values[key]

    def _safe_denominator(self, node: ExprNode, stmt: LetStatement, evaluator: _PointEvaluator) -> bool:
        if not self._is_dynamic(node):
            return _as_float(self._static_value(node, stmt, evaluator), stmt=stmt) != 0.0
        if isinstance(node, IdentNode) and node.name in self.statements:
            dependency = self.statements[node.name]
            return self._safe_denominator(dependency.expr, dependency, evaluator)
        if isinstance(node, CallNode) and node.name in {"min", "max"} and len(node.args) == 2:
            bound = node.args[1]
            if not self._is_dynamic(bound):
                value = _as_float(self._static_value(bound, stmt, evaluator), stmt=stmt)
                return value > 0.0 if node.name == "max" else value < 0.0
        return False

    def _admit(self, node: ExprNode, stmt: LetStatement, evaluator: _PointEvaluator) -> None:
        if id(node) in self._admitted:
            return
        if not self._is_dynamic(node):
            self._static_value(node, stmt, evaluator)
            self._admitted.add(id(node))
            return

        def reject(reason: str) -> None:
            raise AlgebraTypeError(
                f"quantity {stmt.name!r} is unsupported as a stop target: {reason}",
                line=stmt.line, col=stmt.col, line_text=stmt.line_text,
            )

        def admit_power(exponent: ExprNode) -> None:
            if self._is_dynamic(exponent):
                reject("the exponent must be an input-only nonnegative integer")
            value = _as_float(self._static_value(exponent, stmt, evaluator), stmt=stmt)
            if value < 0.0 or not value.is_integer():
                reject("the exponent must be an input-only nonnegative integer")

        if isinstance(node, IdentNode):
            if node.name in self.statements:
                dependency = self.statements[node.name]
                self._admit(dependency.expr, dependency, evaluator)
        elif isinstance(node, SpeciesRefNode):
            pass
        elif isinstance(node, UnaryNode):
            if node.op not in {"+", "-"}:
                reject(f"runtime operator {node.op!r}")
            self._admit(node.rhs, stmt, evaluator)
        elif isinstance(node, BinaryNode):
            if node.op in {"^", "**"}:
                admit_power(node.rhs)
            elif node.op == "/":
                if not self._safe_denominator(node.rhs, stmt, evaluator):
                    reject("division needs a finite nonzero input-only denominator or a signed min/max guard")
            elif node.op not in {"+", "-", "*"}:
                reject(f"runtime operator {node.op!r}")
            self._admit(node.lhs, stmt, evaluator)
            self._admit(node.rhs, stmt, evaluator)
        elif isinstance(node, CallNode):
            if node.name == "pow" and len(node.args) == 2:
                admit_power(node.args[1])
            elif node.name not in {"abs", "min", "max", "clip", "exp", "expm1", "sin", "cos", "erf"}:
                reject(f"runtime function {node.name!r}")
            for arg in node.args:
                self._admit(arg, stmt, evaluator)
        else:
            reject("runtime expression")
        self._admitted.add(id(node))

    def evaluate(
        self, name: str, t: float, y: np.ndarray, runtime_symbols: Mapping[str, float]
    ) -> float:
        """Evaluate one selected quantity without changing caller-owned inputs."""
        ctx = EvaluationContext(
            t=np.array([t]),
            species_series={species: np.array([y[index]]) for index, species in enumerate(self.species_names)},
            initials=self._context.initials,
            species_names=self._context.species_names,
            symtab=self._context.symtab,
            symbol_values=self._context.symbol_values,
            symbol_series=runtime_symbols,
        )
        try:
            return _PointEvaluator(self, ctx).quantity(name)
        except (AlgebraError, ValueError) as exc:
            raise ValueError(f"stop target {name!r}: {exc}") from exc


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
        raise AlgebraTypeError("time grid t must be 1D, finite, and nondecreasing", line=1, col=1)
    for nm, arr in ctx.species_series.items():
        a = np.asarray(arr, dtype=float)
        if a.ndim != 1 or a.size != t.size or np.any(~np.isfinite(a)):
            raise AlgebraTypeError(f"series[{nm!r}] must be 1D, finite, length equal to len(t)", line=1, col=1)
    for nm, raw in (ctx.symbol_series or {}).items():
        a = np.asarray(raw, dtype=float)
        if a.ndim == 0:
            a = np.full(t.size, float(a), dtype=float)
        if a.ndim != 1 or a.size != t.size or np.any(~np.isfinite(a)):
            raise AlgebraTypeError(
                f"symbol_series[{nm!r}] must be scalar or 1D, finite, length equal to len(t)",
                line=1,
                col=1,
            )
    for nm, raw in (ctx.symbol_values or {}).items():
        try:
            value = float(raw)
        except Exception as exc:
            raise AlgebraTypeError(
                f"symbol_values[{nm!r}] must be numeric",
                line=1,
                col=1,
            ) from exc
        if not np.isfinite(value):
            raise AlgebraTypeError(
                f"symbol_values[{nm!r}] must be finite",
                line=1,
                col=1,
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
        raise AlgebraTypeError("time grid t must be 1D, finite, and nondecreasing", line=1, col=1)
    for nm, arr in ctx.species_series.items():
        a = np.asarray(arr, dtype=float)
        if a.ndim != 1 or a.size != t.size or np.any(~np.isfinite(a)):
            raise AlgebraTypeError(f"series[{nm!r}] must be 1D, finite, length equal to len(t)", line=1, col=1)
    for nm, raw in (ctx.symbol_series or {}).items():
        a = np.asarray(raw, dtype=float)
        if a.ndim == 0:
            a = np.full(t.size, float(a), dtype=float)
        if a.ndim != 1 or a.size != t.size or np.any(~np.isfinite(a)):
            raise AlgebraTypeError(
                f"symbol_series[{nm!r}] must be scalar or 1D, finite, length equal to len(t)",
                line=1,
                col=1,
            )
    for nm, raw in (ctx.symbol_values or {}).items():
        try:
            value = float(raw)
        except Exception as exc:
            raise AlgebraTypeError(
                f"symbol_values[{nm!r}] must be numeric",
                line=1,
                col=1,
            ) from exc
        if not np.isfinite(value):
            raise AlgebraTypeError(
                f"symbol_values[{nm!r}] must be finite",
                line=1,
                col=1,
            )
    ev = _Evaluator(block, ctx)
    return ev.run_partial()
