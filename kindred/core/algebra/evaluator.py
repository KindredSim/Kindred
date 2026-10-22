"""
Dynamic evaluator for the Algebra DSL.

Current contract
----------------
- Each statement compiles once into a Python function of the time point,
  with the arithmetic, checks and refusals of the operations it contains.
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
    if not math.isfinite(xf):
        raise AlgebraNonFiniteError("The result is not a finite number.", line=stmt.line, col=stmt.col,
                                    line_text=stmt.line_text)
    return xf


# ----------------------------- compiled statements ----------------------------

# Compiled code by source; the source names its values as constants, so
# statements of the same form share one code object.
_CODE_CACHE: Dict[str, Any] = {}
_INF = math.inf


class _Compiler:
    """Generate one statement as straight Python: operations in source order,
    each with its checks and refusals, and only the selected side of a
    condition calculated."""

    def __init__(self, ev: _Evaluator, stmt: LetStatement, root: ExprNode | None = None,
                 point: PreparedAlgebraStopEvaluator | None = None) -> None:
        self.ev = ev
        self.stmt = stmt
        self.root = stmt.expr if root is None else root
        self.point = point
        self.lines: List[str] = []
        self.g: Dict[str, Any] = {"math": math, "_INF": _INF}
        self.n = 0

    def const(self, obj: Any) -> str:
        name = f"_c{len(self.g)}"
        self.g[name] = obj
        return name

    def tmp(self) -> str:
        self.n += 1
        return f"v{self.n}"

    def emit(self, line: str, ind: int) -> None:
        self.lines.append("    " * ind + line)

    def err(self, cls: type, message: str, **extra: Any) -> str:
        stmt = self.stmt
        def raise_() -> None:
            raise cls(message, line=stmt.line, col=stmt.col, line_text=stmt.line_text, **extra)
        return self.const(raise_)

    def finite(self, v: str, ind: int, message: str = "The result is not a finite number.") -> None:
        self.emit(f"if not (-_INF < {v} < _INF): {self.err(AlgebraNonFiniteError, message)}()", ind)

    def as_float(self, v: str, kind: str, ind: int) -> str:
        if kind == "b":
            self.emit(f"{self.err(AlgebraBoolCastError, 'A true/false result cannot be used as a number.')}()", ind)
            return v
        if kind == "f":
            self.finite(v, ind)
            return v
        r = self.tmp()
        self.emit(f"{r} = {self.const(_as_float)}({v}, stmt={self.const(self.stmt)})", ind)
        return r

    def expr(self, node: ExprNode, ind: int) -> Tuple[str, str]:
        point = self.point
        if point is not None:
            # A stop reads input-only subtrees as admitted and retained lets
            # through its own quantities.
            if id(node) in point.static_values:
                value = point.static_values[id(node)]
                r = self.tmp()
                self.emit(f"{r} = {self.const(point.static_values)}[{self.const(id(node))}]", ind)
                return r, ("b" if isinstance(value, bool) else "f")
            if isinstance(node, IdentNode):
                if node.name in point.statements:
                    r = self.tmp()
                    self.emit(f"{r} = q({node.name!r})", ind)
                    return r, "f"
                if node.name in point.runtime_names and node.name != "t":
                    missing = self.err(AlgebraNameError, f"missing runtime symbol {node.name!r}")
                    self.emit(f"if {node.name!r} not in rs: {missing}()", ind)
        v, kind = self.unscaled(node, ind)
        if node.unit_factor == 1.0:
            return v, kind
        x = self.as_float(v, kind, ind)
        r = self.tmp()
        self.emit(f"{r} = {x} * {node.unit_factor!r}", ind)
        self.finite(r, ind)
        return r, "f"

    def unscaled(self, node: ExprNode, ind: int) -> Tuple[str, str]:
        ev, stmt = self.ev, self.stmt
        r = self.tmp()
        if isinstance(node, NumberNode):
            if ev.literal_reader is None:
                return repr(float(node.value)), "f"
            self.emit(f"{r} = {self.const(ev.literal_reader)}({self.const(node)})", ind)
            return r, "a"
        if isinstance(node, IdentNode) and self.point is not None:
            name = node.name
            if name == "t":
                self.emit(f"{r} = float(tv)", ind)
                return r, "f"
            self.emit(f"if {name!r} in rs: {r} = float(rs[{name!r}])", ind)
            self.emit(f"else: {r} = lk({name!r}, i, {self.const(stmt)})", ind)
            return r, "f"
        if isinstance(node, IdentNode):
            name = node.name
            if name in ev.scalars:
                self.emit(f"{r} = float({self.const(ev.scalars)}[{name!r}])", ind)
                return r, "f"
            if name in ev.series:
                self.emit(f"{r} = float({self.const(ev.series[name])}[i])", ind)
                undefined = self.err(
                    AlgebraUndefinedError, f"It depends on {name!r}, which is undefined wherever it is.",
                    dependency=name)
                self.emit(f"if {r} != {r}: {undefined}()", ind)
                return r, "f"
            if name == "t":
                self.emit(f"{r} = float({self.const(ev.ctx.t)}[i])", ind)
                return r, "f"
            if name in ev.symbol_series:
                self.emit(f"{r} = float({self.const(ev.symbol_series[name])}[i])", ind)
                return r, "f"
            self.emit(f"{r} = {self.const(ev._lookup)}({name!r}, i, {self.const(stmt)})", ind)
            return r, "f"
        if isinstance(node, SpeciesRefNode):
            nm = node.name
            if node.kind == "now" and self.point is not None:
                unknown = self.err(AlgebraNameError, f'Unknown species {nm!r}.')
                if nm not in self.point.species_names:
                    self.emit(f"{unknown}()", ind)
                    return "0.0", "f"
                self.emit(f"if y is None: {unknown}()", ind)
                self.emit(f"{r} = float(y[{self.point.species_names.index(nm)}])", ind)
                return r, "f"
            if node.kind == "now":
                if nm in ev.ctx.species_series:
                    self.emit(f"{r} = float({self.const(ev.ctx.species_series[nm])}[i])", ind)
                    return r, "f"
                self.emit(f"{self.err(AlgebraNameError, f'Unknown species {nm!r}.')}()", ind)
                return "0.0", "f"
            if node.kind == "init":
                if nm not in ev.ctx.initials:
                    self.emit(f"{self.err(AlgebraNameError, f'Unknown species {nm!r}.')}()", ind)
                    return "0.0", "f"
                self.emit(f"{r} = float({self.const(ev.ctx.initials)}[{nm!r}])", ind)
                return r, "f"
        if isinstance(node, UnaryNode):
            v, kind = self.expr(node.rhs, ind)
            if node.op == "!":
                self.emit(f"{r} = not bool({v})", ind)
                return r, "b"
            x = self.as_float(v, kind, ind)
            self.emit(f"{r} = {'+' if node.op == '+' else '-'}{x}", ind)
            return r, "f"
        if isinstance(node, BinaryNode):
            if node.op in ("||", "&&"):
                lv, _ = self.expr(node.lhs, ind)
                if node.op == "||":
                    self.emit(f"if bool({lv}): {r} = True", ind)
                else:
                    self.emit(f"if not bool({lv}): {r} = False", ind)
                self.emit("else:", ind)
                rv, _ = self.expr(node.rhs, ind + 1)
                self.emit(f"{r} = bool({rv})", ind + 1)
                return r, "b"
            lv = self.as_float(*self.expr(node.lhs, ind), ind)
            rv = self.as_float(*self.expr(node.rhs, ind), ind)
            op = node.op
            if op in ("+", "-", "*"):
                self.emit(f"{r} = {lv} {op} {rv}", ind)
                self.finite(r, ind)
                return r, "f"
            if op == "/":
                self.emit(f"if {rv} == 0.0: {self.err(AlgebraZeroDivError, 'Division by zero.')}()", ind)
                self.emit(f"{r} = {lv} / {rv}", ind)
                self.finite(r, ind)
                return r, "f"
            if op in ("**", "^"):
                self.emit("try:", ind)
                self.emit(f"{r} = math.pow({lv}, {rv})", ind + 1)
                self.emit("except ValueError:", ind)
                self.emit(f"{self.err(AlgebraDomainError, '^ ' + POWER_DOMAIN)}()", ind + 1)
                self.emit("except OverflowError:", ind)
                self.emit(f"{self.err(AlgebraNonFiniteError, 'The result is not a finite number.')}()", ind + 1)
                self.finite(r, ind)
                return r, "f"
            if op in ("==", "!=", "<", "<=", ">", ">="):
                self.emit(f"{r} = {lv} {op} {rv}", ind)
                return r, "b"
        if isinstance(node, CallNode):
            fn = ev.ctx.symtab.functions().get(node.name)
            if fn is None:
                self.emit(f"{self.err(AlgebraNameError, f'Unknown function {node.name!r}.')}()", ind)
                return "0.0", "f"
            if node.name == "ifelse":
                c, _ = self.expr(node.args[0], ind)
                self.emit(f"if bool({c}):", ind)
                a = self.as_float(*self.expr(node.args[1], ind + 1), ind + 1)
                self.emit(f"{r} = {a}", ind + 1)
                self.emit("else:", ind)
                b = self.as_float(*self.expr(node.args[2], ind + 1), ind + 1)
                self.emit(f"{r} = {b}", ind + 1)
                return r, "f"
            args = [self.as_float(*self.expr(a, ind), ind) for a in node.args]
            message = "The function result is not a finite number."
            self.emit("try:", ind)
            self.emit(f"{r} = float({self.const(fn)}({', '.join(args)}))", ind + 1)
            self.emit("except ZeroDivisionError:", ind)
            self.emit(f"{self.err(AlgebraZeroDivError, 'Division by zero.')}()", ind + 1)
            self.emit("except OverflowError:", ind)
            self.emit(f"{self.err(AlgebraNonFiniteError, message)}()", ind + 1)
            self.emit("except ValueError as e:", ind)
            self.emit(f"{self.const(self._domain)}(e)", ind + 1)
            self.finite(r, ind, message)
            return r, "f"
        self.emit(f"{self.err(AlgebraError, 'internal evaluator error', code='E150')}()", ind)
        return "0.0", "f"

    def _domain(self, e: ValueError) -> None:
        stmt = self.stmt
        raise AlgebraDomainError(str(e), line=stmt.line, col=stmt.col, line_text=stmt.line_text) from None

    def build(self) -> Callable[[int], float | bool]:
        v, _ = self.expr(self.root, 1)
        # A stop's point reads the state, runtime symbols, retained lets, the
        # time and the remaining names through its arguments.
        signature = "def row(i, y, rs, q, tv, lk):" if self.point is not None else "def row(i):"
        source = "\n".join((signature, *self.lines, f"    return {v}\n"))
        code = _CODE_CACHE.get(source)
        if code is None:
            code = _CODE_CACHE[source] = compile(source, "<algebra>", "exec")
        exec(code, self.g)
        return self.g["row"]


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
    rows: Dict[int, Callable[[int], float | bool]] = {}

    def evaluate(statement: LetStatement) -> float:
        row = rows.get(id(statement))
        if row is None:
            row = rows[id(statement)] = _Compiler(evaluator, statement).build()
        return _as_float(row(0), stmt=statement)

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

    def _lookup(self, name: str, i: int, stmt: LetStatement) -> float:
        if name in self.symbol_values:
            return float(self.symbol_values[name])
        if self.ctx.symbol_resolver is not None:
            resolved = self.ctx.symbol_resolver(name, i)
            if resolved is not None:
                value = float(resolved)
                if name not in self.ctx.runtime_symbol_names:
                    self.symbol_values[name] = value
                return value
        if self.ctx.symtab.has(name):
            return float(self.ctx.symtab.get(name))
        if name in self.failed:
            raise AlgebraUndefinedError(
                f"It depends on {name!r}, which could not be calculated.",
                dependency=name, line=stmt.line, col=stmt.col, line_text=stmt.line_text,
            )
        raise AlgebraNameError(f"Unknown name {name!r}.", line=stmt.line, col=stmt.col, line_text=stmt.line_text)

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
                val = _as_float(_Compiler(self, stmt).build()(0), stmt=stmt)
                self.scalars[name] = val
            else:
                # Compute series across t
                row = _Compiler(self, stmt).build()
                arr = np.empty(self.N, dtype=float)
                for i in range(self.N):
                    if i % 256 == 0:
                        self.ctx.check_cancelled()
                    arr[i] = _as_float(row(i), stmt=stmt)
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
                    val = _as_float(_Compiler(self, stmt).build()(0), stmt=stmt)
                    self.scalars[name] = val
                    continue

                row = _Compiler(self, stmt).build()
                arr = np.empty(self.N, dtype=float)
                # Keyed by whether the point is undefined only through another let.
                firsts: dict[bool, tuple[int, KindredError]] = {}
                counts = {True: 0, False: 0}
                dependencies: dict[str, None] = {}
                for i in range(self.N):
                    if i % 256 == 0:
                        self.ctx.check_cancelled()
                    try:
                        arr[i] = _as_float(row(i), stmt=stmt)
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
        self._static_evaluator = _Evaluator(block, self._context)
        self._lookup = _Evaluator(block, EvaluationContext(
            t=np.array([0.0]), species_series={}, initials=self._context.initials, symtab=self._context.symtab,
        ))._lookup
        self._rows: Dict[str, Callable[..., float | bool]] = {}

    def _static_quantity(self, name: str) -> float:
        scalars = self._static_evaluator.scalars
        if name not in scalars:
            stmt = self.statements[name]
            row = _Compiler(self._static_evaluator, stmt, point=self).build()
            scalars[name] = _as_float(
                row(0, None, {}, self._static_quantity, 0.0, self._static_evaluator._lookup), stmt=stmt
            )
        return scalars[name]

    def admit(self, name: str) -> None:
        """Admit one stop target; its stopping declaration names it in a refusal."""
        stmt = self.statements[name]
        self._rows.clear()
        self._admit(stmt.expr, stmt, self._static_evaluator)
        if not self._is_dynamic(stmt.expr):
            self._static_quantity(name)

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

    def _static_value(self, node: ExprNode, stmt: LetStatement, evaluator: _Evaluator) -> float | bool:
        key = id(node)
        if key not in self.static_values:
            row = _Compiler(evaluator, stmt, node, point=self).build()
            value = row(0, None, {}, self._static_quantity, 0.0, evaluator._lookup)
            self.static_values[key] = value if isinstance(value, bool) else _as_float(value, stmt=stmt)
        return self.static_values[key]

    def _safe_denominator(self, node: ExprNode, stmt: LetStatement, evaluator: _Evaluator) -> bool:
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

    def _admit(self, node: ExprNode, stmt: LetStatement, evaluator: _Evaluator) -> None:
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
        memo: Dict[str, float] = {}

        def q(target: str) -> float:
            if target not in memo:
                row = self._rows.get(target)
                stmt = self.statements[target]
                if row is None:
                    row = self._rows[target] = _Compiler(self._static_evaluator, stmt, point=self).build()
                memo[target] = _as_float(row(0, y, runtime_symbols, q, t, self._lookup), stmt=stmt)
            return memo[target]

        # The stopping declaration that watches the quantity names it.
        return q(name)


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
