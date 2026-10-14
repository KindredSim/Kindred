"""
Dynamic evaluator for the Algebra DSL.

Current contract
----------------
- Dynamic evaluation per time t.
- [A] pulls series at time t; [A]_0 pulls initial concentration.
- Builtins/helpers and protected symbols: see symbols.py and constants.py.
- Error taxonomy uses the algebra error classes and codes defined in errors.py.
- Static vs dynamic: lets independent of [X] terms evaluate to scalars.

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

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

from kindred.core.exceptions import ErrorContext, KindredError, SimulationCancelled, ValueDomainError

from .errors import (
    AlgebraBoolCastError,
    AlgebraDomainError,
    AlgebraError,
    AlgebraNameError,
    AlgebraNonFiniteError,
    AlgebraTypeError,
    AlgebraUndefinedError,
    AlgebraZeroDivError,
    UNCOMPUTABLE_ERRORS,
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
from .symbols import POWER_DOMAIN, SymbolTable


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
    symtab : SymbolTable
        Protected symbols. `T` is read-only and may be updated by the
        application, not by user algebra.
    symbol_series : mapping, optional
        Row-aligned runtime symbols.  These values are resolved before the
        static symbol table and are intended for step-local ``T``, mechanism
        coefficients, and changing scalar inputs; ``t`` is read from ``t``.  A scalar value
        is broadcast to the supplied grid; an array must have exactly the
        same length as ``t``.
    symbol_values : mapping, optional
        Already evaluated static symbols. Missing scientific symbols can be
        resolved on first use inside the consuming observable's error boundary.
    symbol_resolver : callable, optional
        Resolve a scientific symbol at one row to its value, or None when the
        name is not a scientific parameter. A name outside
        ``runtime_symbol_names`` is asked once and kept.
    runtime_symbol_names : set
        Row-dependent symbols, classified before their values are evaluated.
    cancellation_check : callable, optional
        Nonblocking cancellation poll supplied by the execution owner.
    """
    t: np.ndarray
    species_series: Mapping[str, np.ndarray]
    initials: Mapping[str, float]
    symtab: SymbolTable
    symbol_series: Mapping[str, np.ndarray] = field(default_factory=dict)
    symbol_values: Mapping[str, float] = field(default_factory=dict)
    symbol_resolver: Callable[[str, int], float | None] | None = None
    runtime_symbol_names: frozenset[str] = frozenset()
    cancellation_check: Callable[[], bool] | None = None

    def check_cancelled(self) -> None:
        if self.cancellation_check is not None and self.cancellation_check():
            raise SimulationCancelled()


@dataclass(frozen=True)
class AlgebraObservableError:
    """
    Structured error for a single failed algebra observable (let statement).

    This is intended for UI/diagnostics. It is *not* used to control evaluation
    flow in strict mode (evaluate_block).
    """

    name: str
    message: str
    # A value that cannot be calculated for these inputs, rather than a
    # formula Kindred cannot evaluate at all.
    value_failure: bool = False
    code: Optional[str] = None
    line: Optional[int] = None
    col: Optional[int] = None
    line_text: Optional[str] = None
    # The error's own facts, such as the values a failing param reads.
    details: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_exception(name: str, exc: KindredError, *, stmt: LetStatement) -> "AlgebraObservableError":
        # An error keeps its message and its own authored line apart; one
        # without a line is located on the statement that read it.
        context = exc.context or ErrorContext(line=stmt.line, line_text=stmt.line_text)
        return AlgebraObservableError(
            name=str(name),
            message=exc.message,
            value_failure=isinstance(exc, UNCOMPUTABLE_ERRORS),
            code=exc.code,
            line=context.line,
            col=exc.col if isinstance(exc, AlgebraError) else stmt.col,
            line_text=context.line_text,
            details=dict(exc.details),
        )


# ----------------------------- utilities -------------------------------------

# Failures that leave a let undefined at one time point rather than failing it.
_UNDEFINED_POINT_ERRORS = (*UNCOMPUTABLE_ERRORS, AlgebraUndefinedError)


def _as_float(x: Any, *, stmt: LetStatement) -> float:
    """Coerce to float but reject booleans (E170)."""
    if isinstance(x, bool):
        raise AlgebraBoolCastError("A true/false result cannot be used as a number.", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
    xf = float(x)
    if not np.isfinite(xf):
        raise AlgebraNonFiniteError("The result is not a finite number.", line=stmt.line, col=stmt.col,
                                    line_text=stmt.line_text)
    return xf


# ----------------------------- evaluator core --------------------------------


def bind_scalar_expression_evaluator(
    block: AlgebraBlock, values: dict[str, float], *,
    literal_reader: Callable[[NumberNode], float] | None = None,
    resolver: Callable[[str, int], float | None] | None = None,
):
    """Bind admitted scalar AST arithmetic to one current numerical map; a
    name missing from it is asked of ``resolver`` when read and kept there."""
    context = EvaluationContext(
        t=np.array([0.0]), species_series={}, initials={}, symtab=SymbolTable(),
        symbol_resolver=resolver,
    )
    evaluator = _Evaluator(block, context, literal_reader=literal_reader)
    evaluator.symbol_values = values

    def evaluate(statement: LetStatement) -> float:
        return _as_float(evaluator._eval_expr(statement.expr, 0, stmt=statement), stmt=statement)

    return evaluate


class _Evaluator:
    def __init__(self, block: AlgebraBlock, ctx: EvaluationContext, *,
                 literal_reader: Callable[[NumberNode], float] | None = None) -> None:
        self.block = block
        self.ctx = ctx
        self.literal_reader = literal_reader
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
        # results
        self.series: Dict[str, np.ndarray] = {}
        self.scalars: Dict[str, float] = {}
        # lets whose partial evaluation failed
        self.failed: Set[str] = set()
        # dependency classification for user-defined symbols
        # Runtime row-aligned symbols are dynamic roots even when their values
        # happen to be constant over one particular grid.
        self.timevary: Set[str] = {"t"} | set(self.symbol_series) | set(ctx.runtime_symbol_names)

    def _scalar_let_names(self) -> Set[str]:
        """Classify the complete retained let DAG by row dependence.

        A retained identifier can refer to a scalar let several edges away,
        so classifying one statement from the current output dictionaries
        makes an equivalent expression a series.  Resolve the retained
        dependency graph recursively instead; a let is
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
            state[name] = expr_dynamic(statements[name].expr)
            return state[name]

        for name in self.block.evaluation_order:
            let_dynamic(str(name))
        return {name for name, dynamic in state.items() if not dynamic}

    # Public entry
    def run(self) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        scalar_names = self._scalar_let_names()
        for stmt in self.block.statements_in_evaluation_order():
            self.ctx.check_cancelled()
            name = stmt.name
            is_dyn = name not in scalar_names
            if is_dyn:
                self.timevary.add(name)

            if not is_dyn:
                # Evaluate scalar lets through the typed evaluator, including
                # transitive constants and immediate binary/domain failures.
                val = _as_float(self._eval_expr(stmt.expr, 0, stmt=stmt), stmt=stmt)
                self.scalars[name] = val
            else:
                # Compute series across t
                arr = np.empty(self.N, dtype=float)
                for i in range(self.N):
                    if i % 256 == 0:
                        self.ctx.check_cancelled()
                    arr[i] = _as_float(self._eval_expr(stmt.expr, i, stmt=stmt), stmt=stmt)
                self.series[name] = arr

        self.ctx.check_cancelled()
        return self.series, self.scalars

    def run_partial(self) -> Tuple[Dict[str, np.ndarray], Dict[str, float], List[AlgebraObservableError]]:
        """
        Best-effort evaluation: continue after per-observable failures.

        - Each `let name = expr` is evaluated independently.
        - A time point a let cannot compute, or at which it reads an undefined
          let, is undefined (NaN) there only; the let keeps its other points and
          one error record names the first undefined point and their count.
        - A let undefined at every point, or failing otherwise, is omitted from
          outputs with an error record.
        - Strict invariants (shape checks) still apply per statement.
        """
        errors: List[AlgebraObservableError] = []
        scalar_names = self._scalar_let_names()

        for stmt in self.block.statements_in_evaluation_order():
            self.ctx.check_cancelled()
            name = stmt.name
            try:
                is_dyn = stmt.name not in scalar_names
                if is_dyn:
                    self.timevary.add(name)

                if not is_dyn:
                    val = _as_float(self._eval_expr(stmt.expr, 0, stmt=stmt), stmt=stmt)
                    self.scalars[name] = val
                    continue

                arr = np.empty(self.N, dtype=float)
                # Keyed by whether the point is undefined only through another let.
                firsts: dict[bool, tuple[int, KindredError]] = {}
                counts = {True: 0, False: 0}
                dependencies: dict[str, None] = {}
                for i in range(self.N):
                    if i % 256 == 0:
                        self.ctx.check_cancelled()
                    try:
                        arr[i] = _as_float(self._eval_expr(stmt.expr, i, stmt=stmt), stmt=stmt)
                    except _UNDEFINED_POINT_ERRORS as exc:
                        arr[i] = np.nan
                        inherited = isinstance(exc, AlgebraUndefinedError)
                        counts[inherited] += 1
                        firsts.setdefault(inherited, (i, exc))
                        if inherited:
                            dependencies.setdefault(exc.dependency)
                if firsts:
                    # Each cause describes its own points: the let's own failure
                    # when it has one, else every let it reads where undefined,
                    # whose own causes describe their points.
                    inherited = False not in firsts
                    index, exc = firsts[inherited]
                    if inherited and len(dependencies) > 1:
                        names = [repr(dependency) for dependency in dependencies]
                        exc = AlgebraUndefinedError(
                            f"It depends on {', '.join(names[:-1])} and {names[-1]}, "
                            "which are undefined wherever it is.",
                            dependency=next(iter(dependencies)),
                            line=stmt.line, col=stmt.col, line_text=stmt.line_text,
                        )
                    if counts[True] + counts[False] == self.N:
                        raise exc
                    error = AlgebraObservableError.from_exception(name, exc, stmt=stmt)
                    errors.append(replace(error, details={
                        **error.details, "undefined_points": counts[inherited], "points": self.N,
                        "first_undefined_time": float(self.ctx.t[index]),
                    }))
                self.series[name] = arr
            except (AlgebraError, ValueDomainError) as exc:
                errors.append(AlgebraObservableError.from_exception(name, exc, stmt=stmt))
                self.failed.add(name)
                continue

        self.ctx.check_cancelled()
        return self.series, self.scalars, errors

    # Eval
    def _eval_expr(self, node: ExprNode, i: int, *, stmt: LetStatement) -> float | bool:
        value = self._eval_unscaled_expr(node, i, stmt=stmt)
        if node.unit_factor == 1.0:
            return value
        return _as_float(_as_float(value, stmt=stmt) * node.unit_factor, stmt=stmt)

    def _eval_unscaled_expr(self, node: ExprNode, i: int, *, stmt: LetStatement) -> float | bool:
        # Number
        if isinstance(node, NumberNode):
            return float(node.value) if self.literal_reader is None else self.literal_reader(node)

        # Identifier: may be scalar from symtab or user-defined scalar/series
        if isinstance(node, IdentNode):
            name = node.name
            # user-defined scalar
            if name in self.scalars:
                return float(self.scalars[name])
            # user-defined series
            if name in self.series:
                value = float(self.series[name][i])
                if math.isnan(value):
                    raise AlgebraUndefinedError(
                        f"It depends on {name!r}, which is undefined wherever it is.",
                        dependency=name, line=stmt.line, col=stmt.col, line_text=stmt.line_text,
                    )
                return value
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
                # Asked at the point that reads it, so a value undefined at
                # some points fails only there.
                resolved = self.ctx.symbol_resolver(name, i)
                if resolved is not None:
                    value = float(resolved)
                    if name not in self.ctx.runtime_symbol_names:
                        self.symbol_values[name] = value
                    return value
            # protected symbol in symtab
            if self.ctx.symtab.has(name):
                return float(self.ctx.symtab.get(name))
            if name in self.failed:
                # Undefined at every point, so a let reading it is undefined
                # only where it reads it.
                raise AlgebraUndefinedError(
                    f"It depends on {name!r}, which could not be calculated.",
                    dependency=name, line=stmt.line, col=stmt.col, line_text=stmt.line_text,
                )
            # unknown
            raise AlgebraNameError(f"Unknown name {name!r}.", line=stmt.line, col=stmt.col, line_text=stmt.line_text)

        # Species references
        if isinstance(node, SpeciesRefNode):
            nm = node.name
            if node.kind == "now":
                try:
                    return float(self.ctx.species_series[nm][i])
                except KeyError:
                    raise AlgebraNameError(f"Unknown species {nm!r}.", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
            if node.kind == "init":
                if nm not in self.ctx.initials:
                    raise AlgebraNameError(f"Unknown species {nm!r}.", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
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
            if op == "+":
                return _as_float(lv + rv, stmt=stmt)
            if op == "-":
                return _as_float(lv - rv, stmt=stmt)
            if op == "*":
                return _as_float(lv * rv, stmt=stmt)
            if op == "/":
                if rv == 0.0:
                    raise AlgebraZeroDivError("Division by zero.", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
                return _as_float(lv / rv, stmt=stmt)
            if op in ("**", "^"):
                try:
                    return _as_float(math.pow(lv, rv), stmt=stmt)
                except ValueError:
                    raise AlgebraDomainError("^ " + POWER_DOMAIN, line=stmt.line, col=stmt.col,
                                             line_text=stmt.line_text) from None
                except OverflowError:
                    raise AlgebraNonFiniteError("The result is not a finite number.", line=stmt.line,
                                                col=stmt.col, line_text=stmt.line_text) from None
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

        # Call
        if isinstance(node, CallNode):
            fn = self.ctx.symtab.functions().get(node.name)
            if fn is None:
                raise AlgebraNameError(f"Unknown function {node.name!r}.", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
            if node.name == "ifelse":
                # Only the selected value is calculated, as && and || calculate
                # only what decides them.
                selected = node.args[1] if bool(self._eval_expr(node.args[0], i, stmt=stmt)) else node.args[2]
                return _as_float(self._eval_expr(selected, i, stmt=stmt), stmt=stmt)
            args: List[Any] = [_as_float(self._eval_expr(a, i, stmt=stmt), stmt=stmt) for a in node.args]
            try:
                out = float(fn(*args))
            except ZeroDivisionError:
                raise AlgebraZeroDivError("Division by zero.", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
            except OverflowError:
                raise AlgebraNonFiniteError("The function result is not a finite number.", line=stmt.line, col=stmt.col,
                                            line_text=stmt.line_text) from None
            except ValueError as e:
                # e.g., sqrt of negative
                raise AlgebraDomainError(str(e), line=stmt.line, col=stmt.col, line_text=stmt.line_text) from None
            if not np.isfinite(out):
                raise AlgebraNonFiniteError("The function result is not a finite number.", line=stmt.line, col=stmt.col,
                                            line_text=stmt.line_text)
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
            # A refusal is the failing statement's sentence, on its line; the
            # stopping declaration that watches the quantity names it.
            self.scalars[name] = _as_float(
                self._eval_expr(stmt.expr, 0, stmt=stmt), stmt=stmt
            )
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
        species_names: Tuple[str, ...],
        initials: Mapping[str, float],
        symbol_resolver: Callable[[str, int], float | None],
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
            symtab=SymbolTable(),
            symbol_resolver=symbol_resolver,
        )
        self._static_evaluator = _PointEvaluator(self, self._context)

    def admit(self, name: str) -> None:
        """Admit one stop target; its stopping declaration names it in a refusal."""
        stmt = self.statements[name]
        self._admit(stmt.expr, stmt, self._static_evaluator)
        if not self._is_dynamic(stmt.expr):
            self._static_evaluator.quantity(name)

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
            # A fixed denominator of zero is a value that cannot be calculated,
            # not a form a stop cannot watch.
            if _as_float(self._static_value(node, stmt, evaluator), stmt=stmt) == 0.0:
                raise AlgebraZeroDivError("Division by zero.", line=stmt.line, col=stmt.col, line_text=stmt.line_text)
            return True
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
                f"{stmt.name} uses {reason}.",
                line=stmt.line, col=stmt.col, line_text=stmt.line_text,
            )

        def admit_power(exponent: ExprNode) -> None:
            if self._is_dynamic(exponent):
                reject("an exponent that is not an input-only nonnegative integer")
            value = _as_float(self._static_value(exponent, stmt, evaluator), stmt=stmt)
            if value < 0.0 or not value.is_integer():
                reject("an exponent that is not an input-only nonnegative integer")

        if isinstance(node, IdentNode):
            if node.name in self.statements:
                dependency = self.statements[node.name]
                self._admit(dependency.expr, dependency, evaluator)
        elif isinstance(node, SpeciesRefNode):
            pass
        elif isinstance(node, UnaryNode):
            if node.op not in {"+", "-"}:
                reject(f"the runtime operator {node.op!r}")
            self._admit(node.rhs, stmt, evaluator)
        elif isinstance(node, BinaryNode):
            if node.op in {"^", "**"}:
                admit_power(node.rhs)
            elif node.op == "/":
                if not self._safe_denominator(node.rhs, stmt, evaluator):
                    reject("a division whose denominator is neither finite, nonzero, and input-only "
                           "nor a signed min/max guard")
            elif node.op not in {"+", "-", "*"}:
                reject(f"the runtime operator {node.op!r}")
            self._admit(node.lhs, stmt, evaluator)
            self._admit(node.rhs, stmt, evaluator)
        elif isinstance(node, CallNode):
            if node.name == "pow" and len(node.args) == 2:
                admit_power(node.args[1])
            elif node.name not in {"abs", "min", "max", "clip", "exp", "expm1", "sin", "cos", "erf"}:
                reject(f"the runtime function {node.name!r}")
            for arg in node.args:
                self._admit(arg, stmt, evaluator)
        else:
            reject("an unsupported runtime expression")
        self._admitted.add(id(node))

    def evaluate(
        self, name: str, t: float, y: np.ndarray, runtime_symbols: Mapping[str, float]
    ) -> float:
        """Evaluate one selected quantity without changing caller-owned inputs."""
        ctx = EvaluationContext(
            t=np.array([t]),
            species_series={species: np.array([y[index]]) for index, species in enumerate(self.species_names)},
            initials=self._context.initials,
            symtab=self._context.symtab,
            symbol_series=runtime_symbols,
        )
        # The stopping declaration that watches the quantity names it.
        return _PointEvaluator(self, ctx).quantity(name)


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
    """
    ctx.check_cancelled()
    # Execution output and admitted scalars arrive finite and row-aligned.
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
    ctx.check_cancelled()
    # Execution output and admitted scalars arrive finite and row-aligned.
    ev = _Evaluator(block, ctx)
    return ev.run_partial()
