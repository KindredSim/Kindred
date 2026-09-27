"""Restricted, accepted-source symbolic Calculator; no execution parameter binding."""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from html import escape
import math
from fractions import Fraction
from kindred.core.quantity_units import (
    Unit, ONE_SCALE, DIMENSIONLESS, MOLAR, SECOND, conversion_scale,
    format_unit_for_display, scientific_unit,
)
from kindred.core.algebra.unit_preparation import UnitOperation, operation_units
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from kindred.core import constants
from kindred.core.algebra.errors import AlgebraError
from kindred.core.algebra.parser import BinaryNode, CallNode, IdentNode, NumberNode, SpeciesRefNode, UnaryNode
from kindred.core.algebra.printer import expression_to_source
from kindred.core.algebra.simulation_series import compiled_algebra_for_mechanism
from kindred.core.algebra.symbols import BUILTIN_FUNCTIONS, PROTECTED_NAMES
from kindred.core.mechanism_metadata import MechanismMetadataView
from kindred.core.simulator.parameter_algebra import parameter_algebra_spec_from_mechanism
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

from .backend import require_sympy
from .calculator_chemistry import build_calculator_chemistry_relations
from .errors import UnsupportedSymbolicExpressionError
from .jacobian import build_symbolic_mechanism_public_projection, classify_symbolic_jacobian_support

_FIXED_CONSTANTS = tuple(name for name in constants.__all__)
_TRANSFORMS = frozenset({"simplify", "factor", "expand", "cancel"})
_FUNCTIONS = frozenset(BUILTIN_FUNCTIONS) - {"ifelse"} | {"log"}
_OPERATIONS = _TRANSFORMS | _FUNCTIONS | {
    "collect", "subs", "approx", "partial", "dt",
    "odes", "jacobian", "matrix", "eq", "at", "transpose", "det", "initial",
    "solve", "steady_state", "stationary_state", "solution", "integrate", "limit", "series", "remove_order",
    "pi", "e", "infinity",
}


class SymbolicCalculatorError(UnsupportedSymbolicExpressionError):
    """A factual invalid, unsupported, or undefined Calculator expression."""


class SymbolicCalculatorUnavailable(SymbolicCalculatorError):
    """The accepted mechanism is outside the supported symbolic system."""


Quantity = Unit | tuple["Quantity", ...] | None


@dataclass(frozen=True, slots=True)
class _Value:
    """Finished current mathematics; construction and metadata changes are passive."""
    expression: Any
    conditions: tuple[Any, ...] = ()
    notices: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()
    quantity: Quantity = None


@dataclass(frozen=True, slots=True)
class _Relationships:
    replacements: Mapping[Any, Any]


@dataclass(frozen=True, slots=True)
class _UnresolvedProblem:
    problem: Any
    explanation: str


@dataclass(frozen=True, slots=True)
class _UnavailableDiagnostic:
    explanation: str


@dataclass(frozen=True, slots=True)
class _Alternatives:
    values: tuple[_Value, ...]


def _with_quantity(result, quantity):
    if isinstance(result, _Alternatives):
        return _Alternatives(tuple(replace(value, quantity=quantity) for value in result.values))
    return replace(result, quantity=quantity)


def _quantity_map(expression, quantity, operation):
    """Transform scalar quantities without treating equations as scalar values."""
    sp = require_sympy()
    if isinstance(expression, sp.Equality):
        return DIMENSIONLESS if _known_quantity(quantity) else None
    if isinstance(expression, sp.Tuple):
        return tuple(_quantity_map(part, quantity[i] if isinstance(quantity, tuple) else None, operation)
                     for i, part in enumerate(expression))
    if isinstance(expression, sp.MatrixBase):
        return tuple(tuple(_quantity_map(expression[i,j], quantity[i][j] if isinstance(quantity, tuple) else None, operation)
                           for j in range(expression.cols)) for i in range(expression.rows))
    return operation(quantity)


def _quantity_text(quantity):
    if isinstance(quantity, tuple):
        return "[" + ", ".join(_quantity_text(item) for item in quantity) + "]"
    return quantity.symbol if isinstance(quantity, Unit) else "unspecified"


def _quantity_html(quantity):
    if isinstance(quantity, tuple):
        return "[" + ", ".join(_quantity_html(item) for item in quantity) + "]"
    return format_unit_for_display(quantity) if isinstance(quantity, Unit) else "unspecified"


def _entries(result):
    return result.values if isinstance(result, _Alternatives) else (result,)


class _ConditionInapplicable(SymbolicCalculatorError):
    """A current condition is false, distinct from an operation failure."""


def _fixed_constant_values():
    """Canonical identities and the owner's declared values as exact decimals."""
    sp = require_sympy()
    return {sp.Symbol(name, positive=True): sp.Rational(str(getattr(constants, name))) for name in _FIXED_CONSTANTS}


def _boolean_result(parts, transform, *, conjunction=True):
    """A decisive Boolean operand precedes errors in irrelevant companions."""
    sp = require_sympy()
    decisive = sp.false if conjunction else sp.true
    pending, errors = [], []
    for part in parts:
        try:
            part = (_boolean_result(part.args, transform, conjunction=isinstance(part, sp.And))
                    if isinstance(part, (sp.And, sp.Or)) else transform(part))
        except (ValueError, TypeError, ArithmeticError, NotImplementedError) as exc:
            errors.append(exc)
            continue
        if part == decisive:
            return decisive
        pending.append(part)
    if errors:
        raise errors[0]
    return (sp.And if conjunction else sp.Or)(*pending)


def _conditions(current=(), *, replacements=None, before_substitution=(), additional=()):
    """Evaluate changed/fresh predicates; additional conditions are already in the new coordinates."""
    sp = require_sympy()
    additional = tuple(dict.fromkeys(c for c in additional if c != sp.true))
    affected = tuple(c for c in current if replacements and c.free_symbols.intersection(replacements))
    substituted = tuple(dict.fromkeys((*affected, *before_substitution)))
    additional = tuple(c for c in additional if c not in current or c in affected)
    if not substituted and not additional:
        return current
    fixed = _fixed_constant_values()
    def evaluate(part):
        condition, mapping = part
        if mapping:
            condition = _native_substitute(condition, mapping, equations=False)
        return _native_substitute(condition, fixed, equations=False)
    changed = _boolean_result((*((c, replacements) for c in substituted), *((c, None) for c in additional)), evaluate)
    return _condition_tuple(sp.And(*(c for c in current if c not in affected), changed))


def _condition_tuple(result):
    """Package an evaluated predicate without evaluating it again."""
    sp = require_sympy()
    if result == sp.false:
        raise _ConditionInapplicable("The answer's conditions are not satisfied.")
    return result.args if isinstance(result, sp.And) else () if result == sp.true else (result,)


def _metadata(value, *inputs):
    """Display inheritance never evaluates mathematics or applicability."""
    notices = tuple(dict.fromkeys((*value.notices, *(n for v in inputs for n in v.notices))))
    provenance = tuple(dict.fromkeys((*value.provenance, *(p for v in inputs for p in v.provenance))))
    return value if (notices, provenance) == (value.notices, value.provenance) else replace(value, notices=notices, provenance=provenance)


def _branches(values):
    """Enumerate one complete family; no status, mathematics or metadata work."""
    families = [i for i, value in enumerate(values) if isinstance(value, _Alternatives)]
    if len(families) > 1:
        raise SymbolicCalculatorError("Select independent answer operands explicitly with solution(result,n); branches are not paired automatically.")
    if not families:
        yield tuple(values)
    else:
        index = families[0]
        for value in values[index].values:
            yield (*values[:index], value, *values[index+1:])


def _unusable(values):
    """Unavailable operands take precedence; unresolved operands stay current."""
    return next((v for v in values if isinstance(v.expression, _UnavailableDiagnostic)), None) or next(
        (v for v in values if isinstance(v.expression, _UnresolvedProblem)), None)


def _scalar(value: _Value, operation: str, *, finite=False) -> Any:
    sp = require_sympy()
    expression = value.expression
    if not isinstance(expression, sp.Expr):
        raise SymbolicCalculatorError(f"Unsupported: {operation} requires a scalar expression.")
    if finite and expression in (sp.oo, -sp.oo):
        raise SymbolicCalculatorError(f"Invalid input: {operation} requires a finite value.")
    return expression


def _map(expression: Any, operation) -> Any:
    """Scalar operations have no meaning on set predicates or variable binders."""
    sp = require_sympy()
    if isinstance(expression, sp.Tuple):
        return sp.Tuple(*(_map(item, operation) for item in expression))
    if isinstance(expression, sp.MatrixBase):
        return sp.ImmutableMatrix(expression.applyfunc(operation))
    if isinstance(expression, sp.Equality):
        return sp.Eq(_map(expression.lhs, operation), _map(expression.rhs, operation), evaluate=False)
    if not isinstance(expression, sp.Expr):
        raise SymbolicCalculatorError("Unsupported: this operation requires expressions, equations, tuples or matrices; apply solutions with subs first.")
    return operation(expression)


def _current_symbols(result) -> set[Any]:
    """Discover current mathematics and conditions, never labels or provenance."""
    payload = result.expression
    if isinstance(payload, _UnavailableDiagnostic):
        return set().union(*(c.free_symbols for c in result.conditions))
    if isinstance(payload, _Relationships):
        payload = require_sympy().Tuple(*payload.replacements.values())
    elif isinstance(payload, _UnresolvedProblem):
        payload = payload.problem
    return set(payload.free_symbols) | set().union(*(c.free_symbols for c in result.conditions))


def _native_substitute(expression: Any, mapping: Mapping[Any, Any], *, equations=True) -> Any:
    """Substitute free coordinates without capturing them in native binders."""
    sp = require_sympy()
    from sympy.logic.boolalg import Boolean
    from sympy.core.relational import Relational

    def sub(item, replacements=mapping, *, equations=equations):
        return _native_substitute(item, replacements, equations=equations)

    def scope(body, coordinates):
        targets = body.free_symbols | {x for derivative in body.atoms(sp.Derivative) for x in derivative.variables}
        free = {x: y for x, y in mapping.items() if x != y and x not in coordinates and x in targets}
        introduced = set().union(*(y.free_symbols for y in free.values()))
        fresh = {x: sp.Dummy(x.name, **x.assumptions0) for x in coordinates if x in introduced}
        return body.xreplace(fresh), tuple(fresh.get(x, x) for x in coordinates), free

    if isinstance(expression, sp.Tuple):
        return sp.Tuple(*(sub(item) for item in expression))
    if isinstance(expression, sp.MatrixBase):
        return sp.ImmutableMatrix(expression.applyfunc(sub))
    if isinstance(expression, sp.ConditionSet):
        bound = tuple(expression.sym) if isinstance(expression.sym, sp.Tuple) else (expression.sym,)
        condition, bound, free = scope(expression.condition, bound)
        renamed = sp.Tuple(*bound) if isinstance(expression.sym, sp.Tuple) else bound[0]
        return sp.ConditionSet(renamed, sub(condition, free, equations=False), sub(expression.base_set))
    if isinstance(expression, sp.Lambda):
        body, bound, free = scope(expression.expr, expression.variables)
        return sp.Lambda(bound, sub(body, free))
    if isinstance(expression, sp.Subs):
        body, bound, free = scope(expression.expr, expression.variables)
        return sp.Subs(sub(body, free), bound, tuple(sub(x) for x in expression.point))
    if isinstance(expression, sp.Limit):
        body, coordinate, point, direction = expression.args
        body, bound, free = scope(body, (coordinate,))
        return sp.Limit(sub(body, free), bound[0], sub(point), dir=str(direction))
    if isinstance(expression, sp.Integral):
        required = {x: y for x, y in mapping.items() if x != y and x in expression.free_symbols}
        if not required:
            return expression
        introduced = set().union(*(y.free_symbols for y in required.values()))
        bound = {limit[0] for limit in expression.limits if len(limit) > 1}
        # Native renaming distinguishes bound variables from exterior endpoints
        # and indefinite outputs when a replacement crosses those scopes.
        renamed = expression.as_dummy() if bound.intersection(required.keys() | introduced) else expression
        output_coordinates = tuple(dict.fromkeys(limit[0] for limit in renamed.limits if len(limit) == 1))
        targets = tuple(x for x in output_coordinates if x in required or x in introduced)
        if targets:
            fresh = {x: sp.Dummy(x.name, **x.assumptions0) for x in targets}
            body = renamed.xreplace(fresh)
            others = {x: y for x, y in required.items() if x not in targets}
            return sp.Subs(sub(body, others), tuple(fresh.values()), tuple(required.get(x, x) for x in targets))
        return sp.Integral(sub(renamed.function, required),
                           *(tuple([limit[0], *(sub(x, required) for x in limit[1:])]) for limit in renamed.limits))
    if isinstance(expression, sp.Equality) and equations:
        return sp.Eq(sub(expression.lhs), sub(expression.rhs), evaluate=False)
    if isinstance(expression, (sp.And, sp.Or)):
        return _boolean_result(expression.args, lambda part: sub(part, equations=False),
                               conjunction=isinstance(expression, sp.And))
    if isinstance(expression, sp.Set) or isinstance(expression, (Boolean, Relational)) and not isinstance(expression, sp.Expr):
        return expression.func(*(sub(x, equations=False) for x in expression.args)) if expression.args else expression
    for order in expression.atoms(sp.Order):
        if any(x in mapping and mapping[x] != x for x in order.variables):
            raise SymbolicCalculatorError("Unsupported: remove_order before substituting a series coordinate.")
    if isinstance(expression, sp.Derivative):
        coordinates = set(expression.variables)
        required = {x: y for x, y in mapping.items() if x != y and x in expression.free_symbols | coordinates}
        if not required:
            return expression
        introduced = set().union(*(y.free_symbols for y in required.values()))
        if coordinates.intersection(required) or coordinates.intersection(introduced):
            # Fresh targets keep native Subs simultaneous even for coordinate swaps.
            fresh = {x: sp.Dummy(x.name, **x.assumptions0) for x in required}
            return sp.Subs(expression.xreplace(fresh), tuple(fresh.values()), tuple(required.values()))
        return expression.func(*(sub(x) for x in expression.args), evaluate=False)
    if expression.has(sp.Integral, sp.Limit, sp.Subs, sp.Lambda, sp.ConditionSet, sp.Derivative):
        return expression.func(*(sub(x) for x in expression.args))
    return expression.subs(mapping, simultaneous=True)


def _substitute_value(value, replacements: Mapping[Any, Any], *, replacement_conditions=()):
    """Transform one finished current value, preserving native binding and status."""
    sp = require_sympy()
    conditions = tuple(dict.fromkeys((*value.conditions, *replacement_conditions)))
    try:
        roots = _current_symbols(value)
        mapping = {key: rhs for key, rhs in replacements.items() if key != rhs and key in roots}
        payload = value.expression
        current = value.conditions
        domains = _relationship_domains({key: rhs for key, rhs in payload.replacements.items()
                                        if rhs.free_symbols.intersection(mapping)}) if isinstance(payload, _Relationships) else ()
        positive = tuple(sp.Gt(rhs, 0, evaluate=False) for key, rhs in mapping.items() if key.is_positive is True)
        conditions = (_conditions(current, replacements=mapping, before_substitution=domains,
                                  additional=(*positive, *replacement_conditions))
                      if mapping or replacement_conditions else current)
        if isinstance(payload, _Relationships):
            expression = _Relationships({key: _native_substitute(rhs, mapping) for key, rhs in payload.replacements.items()}) if mapping else payload
        elif isinstance(payload, _UnresolvedProblem):
            expression = replace(payload, problem=_native_substitute(payload.problem, mapping)) if mapping else payload
        elif isinstance(payload, _UnavailableDiagnostic):
            expression = payload
        else:
            expression = _native_substitute(payload, mapping) if mapping else payload
        if expression == value.expression:
            return value if conditions == value.conditions else replace(value, conditions=conditions)
        return _with_quantity(_admit_native(expression, conditions), value.quantity)
    except _ConditionInapplicable:
        raise
    except (SymbolicCalculatorError, ValueError, TypeError, ArithmeticError, NotImplementedError) as exc:
        return _Value(_UnavailableDiagnostic(f"Operation is unavailable: {exc}"), conditions)


def _calculus_domain(variable: Any) -> Any:
    """Only the coordinate's owned real domain; conditions stay in the problem."""
    sp = require_sympy()
    return sp.Interval.open(0, sp.oo) if variable.is_positive else sp.S.Reals


def _relationship_domains(replacements):
    """Current output obligations, deferred until joint applicability evaluation."""
    sp = require_sympy()
    return tuple(sp.Contains(rhs, _calculus_domain(key), evaluate=False) for key, rhs in replacements.items())


def _native_arithmetic(left: _Value, operator: str, right: _Value) -> Any:
    sp = require_sympy()
    lhs, rhs = left.expression, right.expression
    matrix = isinstance(lhs, sp.MatrixBase) or isinstance(rhs, sp.MatrixBase)
    if not matrix:
        lhs, rhs = _scalar(left, "arithmetic"), _scalar(right, "arithmetic")
    elif operator not in {"+", "-", "*"}:
        raise SymbolicCalculatorError("Unsupported: matrix arithmetic supports +, - and *.")
    try:
        if operator == "+":
            result = lhs + rhs
        elif operator == "-":
            result = lhs - rhs
        elif operator == "*":
            result = lhs * rhs
        elif operator == "/":
            result = lhs / rhs
        else:
            result = lhs ** rhs
    except (TypeError, ValueError, sp.ShapeError) as exc:
        raise SymbolicCalculatorError(f"Invalid input: incompatible arithmetic operands ({exc}).") from exc
    return result


@dataclass(frozen=True, slots=True)
class SymbolicCalculatorResult:
    query: str
    result_html: str
    copy_text: str
    assumptions: tuple[str, ...]
    symbol_legend: Mapping[str, Sequence[str]]
    mechanism_source: str
    reuse_queries: tuple[tuple[str, str], ...] = ()

    def full_context_copy_text(self, authored_definitions: tuple[str, ...]) -> str:
        legend = [f"{kind}: {', '.join(names) or 'none'}" for kind, names in self.symbol_legend.items()]
        result = self.copy_text
        if self.query == "odes()":
            result += "\nRows: " + ", ".join(f"d{name}/dt" for name in self.symbol_legend["Species"])
        return "\n".join((f"Query: {self.query}", f"Result: {result}", "Assumptions:",
                          *self.assumptions, "Scientific-coordinate legend (calculation units):", *legend,
                          "Authored definitions (source units):", *(authored_definitions or ("none",)),
                          "Mechanism source:", self.mechanism_source))


@dataclass(frozen=True, slots=True)
class CalculatorContext:
    """Exact symbolic mathematics for one accepted mechanism."""
    mechanism: Any
    mechanism_source: str
    parameter_namespace: Any
    species_names: tuple[str, ...]
    symbols: Mapping[str, Any]
    values: Mapping[str, _Value | _Alternatives]
    coordinates: frozenset[Any]
    rhs: tuple[_Value | _Alternatives, ...]
    jacobian: _Value | _Alternatives
    parameter_definitions: Mapping[str, str]
    let_definitions: Mapping[str, str]
    coordinate_units: Mapping[str, Unit | None]
    authored_definitions: tuple[str, ...]


def symbolic_calculator_available(mechanism: Any) -> tuple[bool, str]:
    """Cheap structural availability; expression construction is first-query work."""
    metadata = MechanismMetadataView.from_metadata(mechanism.metadata)
    if metadata.temperature_schedule is not None:
        return False, "Symbolic calculator is unavailable for scheduled temperature."
    support = classify_symbolic_jacobian_support(mechanism)
    if not support.supported:
        return False, str(support.reason)
    namespace = build_namespace_from_mechanism(mechanism)
    spec = parameter_algebra_spec_from_mechanism(mechanism)
    if spec is None:
        return False, "Calculator requires the retained admitted parameter specification."
    reserved = set(namespace.canonical_names) | set(spec.param_assignment_names()) | set(spec.observable_names)
    reserved |= set(PROTECTED_NAMES) | set(BUILTIN_FUNCTIONS)
    collisions = [name for name in mechanism.species_names() if name in reserved
                  or namespace.resolve(name).canonical_name is not None
                  or namespace.invalid_protected_indexed_identifier(name) is not None]
    if collisions:
        return False, f"Species collide with parameter or Algebra names: {', '.join(collisions)}."
    return True, ""


def build_calculator_context(mechanism: Any, *, mechanism_source: str) -> CalculatorContext:
    available, reason = symbolic_calculator_available(mechanism)
    if not available:
        raise SymbolicCalculatorUnavailable(reason)
    sp = require_sympy()
    namespace = build_namespace_from_mechanism(mechanism)
    spec = parameter_algebra_spec_from_mechanism(mechanism)
    graph = spec.evaluation_dependencies
    statements = {stmt.name: stmt for stmt in spec.parameter_block.lines}
    names = tuple(mechanism.species_names())
    symbols = {name: sp.Symbol(name, real=True) for name in
               (*names, *namespace.canonical_names, *spec.param_assignment_names(), "t")}
    symbols.update({symbol.name: symbol for symbol in _fixed_constant_values()})
    symbols["T"] = sp.Symbol("T", positive=True)
    symbols.update({f"initial({name})": sp.Symbol(f"initial({name})", real=True) for name in names})
    kinds = {name: spec.parameter_assignment_kind(name) for name in statements}
    independent = {name for name, deps in graph.items() if not deps and kinds.get(name) != "mechanism-bound"}
    coordinates = {symbol for name, symbol in symbols.items() if name not in graph and name not in _FIXED_CONSTANTS}
    coordinates.update(symbols[name] for name in independent)
    compiled = compiled_algebra_for_mechanism(mechanism)
    quantities = dict(spec.quantity_units)
    quantities.update({name: MOLAR for name in (*names, *(f"initial({n})" for n in names))})
    values = {name: _Value(symbol, quantity=quantities.get(name)) for name, symbol in symbols.items() if name not in graph}
    authored_definitions = tuple(
        f"{kind} {statement.name} = {expression_to_source(statement.expr)}"
        + (f"; unit={statement.declared_unit.symbol}" if statement.declared_unit is not None else "")
        for kind, block in (
            ("param", spec.parameter_block),
            ("let", spec.observable_block),
        )
        if block is not None
        for statement in block.lines
    )
    context = CalculatorContext(
        mechanism, mechanism_source, namespace, names, MappingProxyType(symbols), MappingProxyType(values),
        frozenset(coordinates), (), _Value(sp.ImmutableMatrix([])),
        MappingProxyType({name: expression_to_source(stmt.expr) for name, stmt in statements.items()}),
        MappingProxyType({stmt.name: expression_to_source(stmt.expr) for stmt in compiled.block.lines} if compiled else {}),
        MappingProxyType(quantities),
        authored_definitions,
    )
    chemistry = build_calculator_chemistry_relations(mechanism, parameter_spec=spec, symbols=symbols)
    closures = {}

    def project(expression, dependencies):
        """Install completed dependencies before admitting a native projection."""
        selected = tuple(name for name in graph if name in dependencies)
        operands = tuple(values[name] for name in selected)
        answers = []
        family = any(isinstance(v, _Alternatives) for v in operands)
        for args in _branches(operands):
            try:
                conditions = _conditions(additional=(c for item in args for c in item.conditions))
                unavailable = _unusable(args)
                mapping = {symbols[name]: item.expression for name, item in zip(selected, args, strict=True)}
                result = (replace(unavailable, conditions=conditions) if unavailable is not None
                          else _admit_native(_native_substitute(expression, mapping), conditions))
            except _ConditionInapplicable:
                continue
            except (SymbolicCalculatorError, ValueError, TypeError, ArithmeticError, NotImplementedError) as exc:
                result = _Value(_UnavailableDiagnostic(str(exc)))
            family |= isinstance(result, _Alternatives)
            answers.extend(_metadata(item, *args) for item in _entries(result))
        return _Alternatives(tuple(answers)) if family or not answers else answers[0]

    for name, dependencies in graph.items():
        closures[name] = set(dependencies).union(*(closures[dep] for dep in dependencies))
        def definition():
            if kinds.get(name) in {"mechanism-bound", "derived scalar"}:
                value = _translate_node(context, statements[name].expr)
                return _with_quantity(value, quantities.get(name)) if name in spec.mechanism_param_names else value
            if name in chemistry:
                return _with_quantity(project(chemistry[name], closures[name]), quantities.get(name))
            return _Value(symbols[name], quantity=quantities.get(name))
        try:
            values[name] = definition()
        except (SymbolicCalculatorError, AlgebraError, ValueError, TypeError, ArithmeticError, NotImplementedError) as exc:
            values[name] = _Value(_UnavailableDiagnostic(str(exc)))
    if compiled:
        for statement in compiled.block.statements_in_evaluation_order():
            try:
                values[statement.name] = _translate_node(context, statement.expr)
            except (SymbolicCalculatorError, AlgebraError, ValueError, TypeError, ArithmeticError, NotImplementedError) as exc:
                values[statement.name] = _Value(_UnavailableDiagnostic(str(exc)))
    projection = build_symbolic_mechanism_public_projection(mechanism, exact_stoichiometric_coefficients=True)
    canonical = {symbol: symbols[str(symbol)] for expression in projection.rhs_expressions for symbol in expression.free_symbols}
    def field(expression):
        expression = expression.xreplace(canonical)
        roots = {name for name in graph if symbols[name] in expression.free_symbols}
        return project(expression, roots.union(*(closures[name] for name in roots)))
    return replace(context, rhs=tuple(_with_quantity(field(x), MOLAR.divide(SECOND)) for x in projection.rhs_expressions),
                   jacobian=_with_quantity(field(sp.ImmutableMatrix(projection.jacobian_expressions)),
                                           tuple(tuple(DIMENSIONLESS.divide(SECOND) for _ in names) for _ in names)))


def _canonical_symbol(context: CalculatorContext, name: str) -> Any:
    if name in context.symbols:
        return context.symbols[name]
    resolved = context.parameter_namespace.resolve(name)
    if resolved.canonical_name in context.symbols:
        return context.symbols[resolved.canonical_name]
    message = context.parameter_namespace.invalid_protected_indexed_identifier_message(name)
    raise SymbolicCalculatorError(message or f"Invalid input: unknown symbolic identifier {name!r}.")


def _exact_number(node: NumberNode) -> Any:
    exact = node.exact_value
    if exact is None:
        raise SymbolicCalculatorError("Undefined: a retained coefficient is not finite and exact.")
    return require_sympy().Rational(exact.numerator, exact.denominator)


def _translate_node(context: CalculatorContext, node: Any) -> _Value:
    sp = require_sympy()
    def lower(child):
        return _translate_node(context, child)
    def operation(name, args):
        # Source admission already decided every conversion. CAS simplification
        # must not establish a second interpretation of retained source.
        return _evaluate_call(context, name, args,
                              conversion=UnitOperation(node.quantity_unit, (ONE_SCALE,) * len(args), node.unit_scale))
    def leaf(value):
        if node.unit_scale == ONE_SCALE:
            return _with_quantity(value, node.quantity_unit)
        # Parent argument conversions may be retained on a leaf as well as on
        # an operator. Keep dependency conditions and normal native admission.
        return operation("*", (value, _Value(sp.Integer(1), quantity=DIMENSIONLESS)))
    if isinstance(node, NumberNode):
        return leaf(_Value(_exact_number(node), quantity=DIMENSIONLESS))
    if isinstance(node, IdentNode):
        name = node.name if node.name in context.values else str(_canonical_symbol(context, node.name))
        return leaf(context.values[name])
    if isinstance(node, SpeciesRefNode):
        return leaf(_Value(context.symbols[node.name if node.kind == "now" else f"initial({node.name})"], quantity=MOLAR))
    if isinstance(node, UnaryNode) and node.op in {"+", "-"}:
        return operation("*", (_Value(sp.Integer(1 if node.op == "+" else -1), quantity=DIMENSIONLESS), lower(node.rhs)))
    if isinstance(node, BinaryNode) and node.op in {"+", "-", "*", "/", "^", "**"}:
        return operation(node.op, (lower(node.lhs), lower(node.rhs)))
    if isinstance(node, CallNode):
        if node.name in _FUNCTIONS:
            return operation(node.name, tuple(lower(child) for child in node.args))
    raise SymbolicCalculatorError(
        f"Unsupported: admitted {type(node).__name__}; dynamic conditions and logical expressions are not Calculator mathematics."
    )


def _math_call(name: str, arguments: Sequence[_Value]) -> Any:
    sp = require_sympy()
    args = [_scalar(arg, name) for arg in arguments]
    unary = {"sqrt": sp.sqrt, "ln": sp.log, "log": sp.log,
             "log10": lambda x: sp.log(x, 10), "log1p": lambda x: sp.log(1+x),
             "exp": sp.exp, "expm1": lambda x: sp.exp(x)-1,
             "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "abs": sp.Abs, "erf": sp.erf,
             "heaviside": lambda x: sp.Heaviside(x, 0)}
    if name in unary and len(args) == 1:
        x = args[0]
        expression = unary[name](x)
    elif name in {"min", "max"} and args:
        expression = (sp.Min if name == "min" else sp.Max)(*args)
    elif name == "pow" and len(args) == 2:
        return args[0] ** args[1]
    elif name == "clip" and len(args) == 3:
        x, lo, hi = args
        expression = sp.Min(sp.Max(x, sp.Min(lo, hi)), sp.Max(lo, hi))
    else:
        raise SymbolicCalculatorError(f"Invalid input: wrong argument count for {name}.")
    return expression


def _coordinate(context: CalculatorContext, symbol: Any, operation: str) -> Any:
    sp = require_sympy()
    if not isinstance(symbol, sp.Symbol) or symbol not in context.symbols.values() or str(symbol) in _FIXED_CONSTANTS:
        raise SymbolicCalculatorError(f"Invalid input: {operation} requires a model coordinate, not a constant or expression.")
    if symbol not in context.coordinates:
        definition = context.parameter_definitions.get(str(symbol))
        detail = f" Defined as {definition}." if definition is not None else " It is a derived quantity."
        raise SymbolicCalculatorError(
            f"Invalid input: {operation} requires an independent coordinate; {symbol} is not one.{detail} "
            "Use independent coordinates from the definition."
        )
    return symbol


def _integer(value: _Value, operation: str, *, minimum=0) -> int:
    expression = _scalar(value, operation)
    if isinstance(value.quantity, Unit) and value.quantity.compatible(DIMENSIONLESS):
        expression *= conversion_scale(value.quantity, DIMENSIONLESS).symbolic()
    if expression.is_Integer is not True or expression < minimum:
        raise SymbolicCalculatorError(f"Invalid input: {operation} requires an integer >= {minimum}.")
    return int(expression)


def _differentiate(context: CalculatorContext, value: _Value, variable: Any | None, order: int):
    """Admit each derivative step, without reentering command orchestration."""
    sp = require_sympy()
    for step in range(order):
        answers = []
        family = isinstance(value, _Alternatives)
        for item in _entries(value):
            if step and isinstance(item.expression, (_UnavailableDiagnostic, _UnresolvedProblem)):
                answers.append(item)
                continue
            try:
                if variable is not None:
                    expression = _map(item.expression, lambda x: sp.diff(x, variable))
                    result = item if expression == item.expression else _admit_native(expression, item.conditions)
                else:
                    result = _kinetic_derivative(context, item)
                family |= isinstance(result, _Alternatives)
                answers.extend(_entries(result))
            except _ConditionInapplicable:
                if not isinstance(value, _Alternatives):
                    raise
            except (SymbolicCalculatorError, ValueError, TypeError, ArithmeticError, NotImplementedError) as exc:
                answers.append(_Value(_UnavailableDiagnostic(str(exc)), item.conditions))
        value = _Alternatives(tuple(answers)) if family else answers[0]
    return value


def _kinetic_derivative(context, value):
    sp = require_sympy()
    partials = {}
    def differentiate_leaf(expression):
        partials[expression] = sp.Tuple(sp.diff(expression, context.symbols["t"]),
                                       *(sp.diff(expression, context.symbols[name]) for name in context.species_names))
        return expression
    _map(value.expression, differentiate_leaf)
    differentiated = _admit_native(sp.Tuple(*partials.values()), value.conditions)
    answers = []
    family = isinstance(differentiated, _Alternatives)
    for derivatives in _entries(differentiated):
        if isinstance(derivatives.expression, (_UnavailableDiagnostic, _UnresolvedProblem)):
            answers.append(derivatives)
            continue
        parts = dict(zip(partials, derivatives.expression, strict=True))
        required = [i for i in range(len(context.species_names)) if any(row[i+1] != 0 for row in parts.values())]
        fields = tuple(context.rhs[i] for i in required)
        family |= any(isinstance(field, _Alternatives) for field in fields)
        for rates in _branches(fields):
            conditions = tuple(dict.fromkeys((*derivatives.conditions, *(c for rate in rates for c in rate.conditions))))
            try:
                conditions = _conditions(derivatives.conditions, additional=(c for rate in rates for c in rate.conditions))
                unavailable = _unusable(rates)
                if unavailable is not None:
                    result = replace(unavailable, conditions=conditions)
                else:
                    expression = _map(value.expression, lambda x: parts[x][0] + sum(
                        parts[x][i+1] * rate.expression for i, rate in zip(required, rates, strict=True)))
                    result = (value if conditions == value.conditions else replace(value, conditions=conditions)) if expression == value.expression else _admit_native(expression, conditions)
                family |= isinstance(result, _Alternatives)
                answers.extend(_entries(result))
            except _ConditionInapplicable:
                if not family:
                    raise
            except (SymbolicCalculatorError, ValueError, TypeError, ArithmeticError, NotImplementedError) as exc:
                answers.append(_Value(_UnavailableDiagnostic(str(exc)), conditions))
    return _Alternatives(tuple(answers)) if family else answers[0]


def _scalar_operation(name, args, *, conversion=None):
    """Consume source preparation, or prepare a new query's scalar operation."""
    if conversion is None:
        exponent = None
        if name in {"^", "**", "pow"} and len(args) == 2 and args[1].expression.is_Rational:
            exponent = Fraction(int(args[1].expression.p), int(args[1].expression.q))
        conversion = operation_units("ln" if name == "log" else name,
            tuple(arg.quantity if isinstance(arg.quantity, Unit) else None for arg in args),
            zeros=tuple(arg.expression == 0 and isinstance(arg.quantity, Unit) for arg in args), exponent=exponent)
    working = tuple(replace(arg, expression=arg.expression * factor.symbolic())
                    for arg, factor in zip(args, conversion.arguments))
    expression = (_native_arithmetic(working[0], name, working[1])
                  if name in {"+", "-", "*", "/", "^", "**"} else _math_call(name, working))
    return expression * conversion.result.symbolic(), conversion.unit


def _matrix_cell(value, i, j):
    unit = value.quantity[i][j] if isinstance(value.quantity, tuple) else None
    return replace(value, expression=value.expression[i, j], quantity=unit)


def _matrix_arithmetic(name, left, right):
    """Compose matrix entries from scalar operations, before native admission."""
    sp = require_sympy()
    lm, rm = isinstance(left.expression, sp.MatrixBase), isinstance(right.expression, sp.MatrixBase)
    if name not in {"+", "-", "*"}:
        raise SymbolicCalculatorError("Unsupported: matrix arithmetic supports +, - and *.")
    if name in {"+", "-"}:
        if not lm or not rm or left.expression.shape != right.expression.shape:
            raise SymbolicCalculatorError("Invalid input: incompatible arithmetic operands.")
        rows, cols = left.expression.shape
    elif lm and rm:
        if left.expression.cols != right.expression.rows:
            raise SymbolicCalculatorError("Invalid input: incompatible arithmetic operands.")
        rows, cols = left.expression.rows, right.expression.cols
    else:
        rows, cols = (left if lm else right).expression.shape
    cells = []
    for i in range(rows):
        row = []
        for j in range(cols):
            if name == "*" and lm and rm:
                terms = []
                for k in range(left.expression.cols):
                    expression, unit = _scalar_operation("*", (_matrix_cell(left, i, k), _matrix_cell(right, k, j)))
                    terms.append(_Value(expression, quantity=unit))
                value = terms[0] if terms else _Value(sp.Integer(0), quantity=DIMENSIONLESS)
                for term in terms[1:]:
                    expression, unit = _scalar_operation("+", (value, term))
                    value = _Value(expression, quantity=unit)
            else:
                expression, unit = _scalar_operation(name, (_matrix_cell(left, i, j) if lm else left,
                                                           _matrix_cell(right, i, j) if rm else right))
                value = _Value(expression, quantity=unit)
            row.append(value)
        cells.append(row)
    return sp.ImmutableMatrix(rows, cols, [v.expression for row in cells for v in row]), tuple(tuple(v.quantity for v in row) for row in cells)


def _scientific(expression, unit):
    """Express a finished known quantity, or each matrix entry, in its scientific coordinates."""
    sp = require_sympy()
    if isinstance(expression, sp.MatrixBase):
        return sp.ImmutableMatrix(expression.rows, expression.cols, [
            _scientific(expression[i, j], unit[i][j] if isinstance(unit, tuple) else None)
            for i in range(expression.rows) for j in range(expression.cols)])
    # Equations have dimensionless meaning; _equations already made their sides scientific.
    if not isinstance(unit, Unit) or not isinstance(expression, sp.Expr):
        return expression
    return expression * conversion_scale(unit, scientific_unit(unit.dimensions)).symbolic()


def _determinant(value):
    """Factor row/column dimensions; use native determinant on aligned entries."""
    matrix = value.expression
    n = matrix.rows
    if not n:
        return matrix.det(), DIMENSIONLESS

    def unestablished():
        # Without a dimension contract, entries enter in their scientific coordinates.
        return _scientific(matrix, value.quantity).det(), None

    rows, columns = [None]*n, [None]*n
    cells = {(i, j): _matrix_cell(value, i, j) for i in range(n) for j in range(n)
             if matrix[i, j] != 0 or _matrix_cell(value, i, j).quantity is None}
    if any(not isinstance(cell.quantity, Unit) for cell in cells.values()):
        return unestablished()
    for start in range(n):
        if rows[start] is not None:
            continue
        rows[start] = DIMENSIONLESS
        pending = True
        while pending:
            pending = False
            for (i, j), cell in cells.items():
                if rows[i] is not None and columns[j] is None:
                    columns[j] = cell.quantity.divide(rows[i])
                    pending = True
                elif columns[j] is not None and rows[i] is None:
                    rows[i] = cell.quantity.divide(columns[j])
                    pending = True
                elif rows[i] is not None and columns[j] is not None and not cell.quantity.compatible(rows[i].multiply(columns[j])):
                    return unestablished()
    if any(unit is None for unit in columns):
        return unestablished()
    aligned = matrix.as_mutable()
    for (i, j), cell in cells.items():
        aligned[i, j] *= conversion_scale(cell.quantity, rows[i].multiply(columns[j])).symbolic()
    raw = DIMENSIONLESS
    for unit in (*rows, *columns):
        raw = raw.multiply(unit)
    normalized = scientific_unit(raw.dimensions)
    return aligned.det() * conversion_scale(raw, normalized).symbolic(), normalized


def _equations(left: _Value, right: _Value):
    sp = require_sympy()
    if isinstance(left.expression, sp.MatrixBase):
        matrix = left.expression
        if right.expression == 0:
            right = replace(right, expression=sp.zeros(*matrix.shape),
                            quantity=tuple(tuple(right.quantity for _ in range(matrix.cols)) for _ in range(matrix.rows)))
        if not isinstance(right.expression, sp.MatrixBase) or right.expression.shape != matrix.shape:
            raise SymbolicCalculatorError("Invalid input: equation matrix shapes must agree.")
        entries = [_equations(_matrix_cell(left, i, j), _matrix_cell(right, i, j))
                   for i in range(matrix.rows) for j in range(matrix.cols)]
        return sp.Tuple(*(expr for expr, _unit in entries)), tuple(unit for _expr, unit in entries)
    if not isinstance(left.expression, sp.Expr) or not isinstance(right.expression, sp.Expr):
        raise SymbolicCalculatorError("Invalid input: eq requires scalar expressions or matching matrices.")
    conversion = operation_units("==", (left.quantity, right.quantity),
        zeros=tuple(v.expression == 0 and isinstance(v.quantity, Unit) for v in (left, right)))
    lhs, rhs = (v.expression * factor.symbolic() for v, factor in zip((left, right), conversion.arguments))
    return sp.Eq(lhs, rhs, evaluate=False), conversion.unit


def _known_quantity(quantity):
    return all(_known_quantity(part) for part in quantity) if isinstance(quantity, tuple) else isinstance(quantity, Unit)


def _coordinate_argument(context, coordinate, value):
    """Interpret a number or convert a finished quantity at a coordinate boundary."""
    target = context.coordinate_units.get(str(coordinate))
    source = value.quantity
    if target is not None and isinstance(source, Unit):
        if source.compatible(DIMENSIONLESS) and value.expression.is_number:
            return value.expression * conversion_scale(source, DIMENSIONLESS).symbolic(), True
        if source.compatible(target):
            return value.expression * conversion_scale(source, target).symbolic(), True
    return _scientific(value.expression, source), False


def _lose_units(expression, quantity, unknown):
    """Only affected entries lose meaning and continue in their scientific
    coordinates; finished CAS is never re-inferred."""
    sp = require_sympy()
    if isinstance(expression, _Relationships):
        parts = [_lose_units(rhs, quantity[i] if isinstance(quantity, tuple) else None, unknown)
                 for i, rhs in enumerate(expression.replacements.values())]
        return (_Relationships(dict(zip(expression.replacements, (e for e, _ in parts)))),
                tuple(q for _, q in parts))
    if isinstance(expression, sp.MatrixBase):
        rows = [[_lose_units(expression[i, j], quantity[i][j] if isinstance(quantity, tuple) else None, unknown)
                 for j in range(expression.cols)] for i in range(expression.rows)]
        return (sp.ImmutableMatrix(expression.rows, expression.cols, [e for row in rows for e, _ in row]),
                tuple(tuple(q for _, q in row) for row in rows))
    if isinstance(expression, sp.Tuple):
        parts = [_lose_units(part, quantity[i] if isinstance(quantity, tuple) else None, unknown)
                 for i, part in enumerate(expression)]
        return sp.Tuple(*(e for e, _ in parts)), tuple(q for _, q in parts)
    if isinstance(expression, sp.Basic) and expression.free_symbols.intersection(unknown):
        return _scientific(expression, quantity), None
    return expression, quantity


def _substitute_coordinates(context, source, targets, replacements, units, *, conditions):
    converted, unknown = {}, set()
    for target, expression, unit in zip(targets, replacements, units, strict=True):
        expression, known = _coordinate_argument(context, target, _Value(expression, quantity=unit))
        converted[target] = expression
        if not known:
            unknown.add(target)
    expression, quantity = _lose_units(source.expression, source.quantity, unknown)
    source = replace(source, expression=expression, quantity=quantity)
    return _substitute_value(source, converted, replacement_conditions=conditions)


def _solution_set(equations, conditions, variables):
    sp = require_sympy()
    return sp.ConditionSet(sp.Tuple(*variables), sp.And(*equations, *conditions),
                           sp.ProductSet(*(_calculus_domain(x) for x in variables)))


def _solve(context: CalculatorContext, problem: _Value, coordinates: _Value, conditions):
    """Own this invocation's native solving, candidate admission and current remainder."""
    applicable = conditions
    coherent = _known_quantity(problem.quantity)
    try:
        sp = require_sympy()
        from sympy.solvers.solveset import NonlinearError
        labels = coordinates.expression if isinstance(coordinates.expression, sp.Tuple) else (coordinates.expression,)
        if not labels or len(set(labels)) != len(labels):
            raise SymbolicCalculatorError("Invalid input: solve requires distinct coordinates.")
        for symbol in labels:
            _coordinate(context, symbol, "solve")
        equations = problem.expression if isinstance(problem.expression, sp.Tuple) else (problem.expression,)
        if not equations or any(not isinstance(item, sp.Equality) for item in equations):
            raise SymbolicCalculatorError("Invalid input: solve requires equations made with eq.")
        variables = tuple(sp.Dummy(x.name, **x.assumptions0) for x in labels)
        mapping = {**_fixed_constant_values(), **dict(zip(labels, variables, strict=True))}
        conditions = _conditions(conditions, replacements=mapping)
        bound = _admit_native(_native_substitute(sp.Tuple(*equations), mapping), conditions)
        def solve(equations, residual):
            if isinstance(residual.expression, (_UnavailableDiagnostic, _UnresolvedProblem)):
                return _Alternatives((residual,))
            expressions, conditions = residual.expression, residual.conditions
            parameters = equations.free_symbols - set(variables)
            candidates = []
            complete = False
            explanation = "The native solver has not resolved these remaining possibilities."
            try:
                try:
                    matrix, vector = sp.linear_eq_to_matrix(expressions, variables)
                except NonlinearError:
                    if len(variables) == len(expressions) == 1:
                        native = sp.solveset(expressions[0], variables[0], domain=_calculus_domain(variables[0]))
                        def finite_roots(result):
                            if isinstance(result, sp.FiniteSet):
                                return list(result)
                            if isinstance(result, sp.Union):
                                return [x for part in result.args for x in finite_roots(part)]
                            if isinstance(result, sp.Intersection):
                                finite = next((part for part in result.args if isinstance(part, sp.FiniteSet)), None)
                                return list(finite) if finite is not None else []
                            return []
                        candidates = [sp.Tuple(x) for x in finite_roots(native)]
                    else:
                        native = sp.nonlinsolve(expressions, variables)
                        candidates = list(native) if isinstance(native, sp.FiniteSet) else []
                else:
                    native = sp.linsolve((matrix, vector), variables)
                    candidates = list(native) if isinstance(native, sp.FiniteSet) else []
                    constant_matrix = not matrix.free_symbols and all(x.is_finite is True for x in matrix)
                    unique = (matrix.rows == matrix.cols and all(x.is_finite is True for x in matrix)
                              and matrix.det().is_zero is False)
                    complete = isinstance(native, sp.FiniteSet) and (constant_matrix or unique)
                complete |= not parameters and (isinstance(native, sp.FiniteSet) or native == sp.S.EmptySet)
            except (NotImplementedError, ValueError, TypeError, ArithmeticError, KeyError) as exc:
                explanation = f"Native solver could not resolve this system: {exc}."

            answers = []
            for candidate in candidates:
                if not isinstance(candidate, sp.Tuple) or len(candidate) != len(variables) or any(not isinstance(x, sp.Expr) for x in candidate):
                    complete = False
                    continue
                replacements = dict(zip(variables, candidate, strict=True))
                try:
                    restrictions = _conditions(conditions, replacements=replacements, additional=(
                        *(sp.Ne(sp.denom(x), 0, evaluate=False) for x in candidate), *_relationship_domains(replacements)))
                    residuals = tuple(sp.simplify(eq.subs(replacements, simultaneous=True)) for eq in equations)
                    restrictions = _conditions(restrictions, additional=residuals)
                    admitted = _admit_native(candidate, restrictions)
                except _ConditionInapplicable:
                    continue
                except (SymbolicCalculatorError, ValueError, TypeError, ArithmeticError):
                    complete = False
                    continue
                for item in _entries(admitted):
                    if isinstance(item.expression, _UnavailableDiagnostic):
                        complete = False
                    else:
                        answers.append(item)
            answers = tuple(sorted(dict.fromkeys(answers), key=lambda v: sp.default_sort_key(v.expression)))
            if not complete:
                original = _solution_set(equations, conditions, variables)
                represented = sp.Union(*(_solution_set(
                    tuple(sp.Eq(x, y) for x, y in zip(variables, v.expression, strict=True)), v.conditions, variables) for v in answers))
                remainder = sp.Complement(original, represented)
                if remainder != sp.S.EmptySet:
                    answers += _entries(_admit_native(_UnresolvedProblem(remainder, explanation)))
            return _Alternatives(tuple(answers))
        answers = []
        for branch in _entries(bound):
            if isinstance(branch.expression, (_UnavailableDiagnostic, _UnresolvedProblem)):
                answers.append(branch)
                continue
            residuals = _admit_native(sp.Tuple(*(eq.lhs - eq.rhs for eq in branch.expression)), branch.conditions)
            answers.extend(answer for residual in _entries(residuals) for answer in solve(branch.expression, residual).values)
        aliases = dict(zip(variables, labels, strict=True))
        def canonical(expression, *, equations=True):
            mapping = {symbol: aliases[symbol] for symbol in expression.free_symbols.intersection(aliases)}
            return _native_substitute(expression, mapping, equations=equations) if mapping else expression
        published = []
        for item in answers:
            payload = item.expression
            if isinstance(payload, _UnresolvedProblem):
                problem = canonical(payload.problem)
                payload = payload if problem == payload.problem else replace(payload, problem=problem)
            elif not isinstance(payload, _UnavailableDiagnostic):
                payload = _Relationships({key: canonical(rhs)
                                          for key, rhs in zip(labels, payload, strict=True)})
            quantity = tuple(context.coordinate_units.get(str(label)) if coherent else None for label in labels) if isinstance(payload, _Relationships) else None
            published.append(replace(item, expression=payload, quantity=quantity,
                                     conditions=tuple(canonical(c, equations=False) for c in item.conditions)))
        return _Alternatives(tuple(published))
    except _ConditionInapplicable:
        raise
    except (SymbolicCalculatorError, ValueError, TypeError, ArithmeticError, NotImplementedError) as exc:
        return _Value(_UnavailableDiagnostic(f"Operation is unavailable: {exc}"), applicable)


def _evaluate_call(context: CalculatorContext, name: str, args: Sequence[_Value | _Alternatives], *, direction=None, conversion=None):
    """Own complete operand admission, family lifting and command metadata."""
    args = tuple(args)
    selection = name == "solution" and len(args) == 2
    if name == "subs" and len(args) == 2:
        entries = _entries(args[1])
        usable = tuple(v for v in entries if not isinstance(v.expression, (_UnresolvedProblem, _UnavailableDiagnostic)))
        if any(not isinstance(v.expression, _Relationships) for v in usable):
            raise SymbolicCalculatorError("Invalid input: subs(F, solutions) requires resolved solution relationships, not reduced expression results.")
        if all(isinstance(v, _Alternatives) for v in args):
            if len(entries) != 1 or len(usable) != 1:
                raise SymbolicCalculatorError("Select the replacement explicitly with solution(result,n) when alternatives or unresolved possibilities remain.")
            args = (args[0], usable[0])
    if selection and all(isinstance(v, _Alternatives) for v in args):
        raise SymbolicCalculatorError("Select independent answer operands explicitly with solution(result,n); branches are not paired automatically.")
    operands = args[1:] if selection else args
    family = any(isinstance(v, _Alternatives) for v in operands)
    answers = []
    for values in _branches(operands):
        try:
            if selection:
                result = _select(args[0], values[0])
            else:
                result = _dispatch_call(context, name, values, direction=direction, conversion=conversion)
            family |= isinstance(result, _Alternatives)
        except _ConditionInapplicable as exc:
            if any(isinstance(v, _Alternatives) for v in operands):
                continue
            result = _Value(_UnavailableDiagnostic(str(exc)))
        answers.extend(_metadata(item, *values) for item in _entries(result))
    return _Alternatives(tuple(answers)) if family else answers[0]


def _select(result, index):
    """Inspect the complete primary collection; selection performs no mathematics."""
    if isinstance(index.expression, (_UnavailableDiagnostic, _UnresolvedProblem)):
        values = []
        for item in _entries(result):
            conditions = tuple(dict.fromkeys((*item.conditions, *index.conditions)))
            try:
                conditions = _conditions(item.conditions, additional=index.conditions)
                value = replace(index, conditions=conditions)
            except _ConditionInapplicable as exc:
                if isinstance(result, _Alternatives):
                    continue
                value = _Value(_UnavailableDiagnostic(str(exc)))
            except (SymbolicCalculatorError, ValueError, TypeError, ArithmeticError, NotImplementedError) as exc:
                value = _Value(_UnavailableDiagnostic(str(exc)), conditions)
            values.append(_metadata(value, item))
        return _Alternatives(tuple(values)) if isinstance(result, _Alternatives) else values[0]
    selected, conditions = index, index.conditions
    try:
        if isinstance(result, _Value) and isinstance(result.expression, _UnavailableDiagnostic):
            selected = result
        else:
            usable = tuple(v for v in _entries(result) if not isinstance(v.expression, (_UnresolvedProblem, _UnavailableDiagnostic)))
            offset = _integer(index, "solution", minimum=1) - 1
            if offset >= len(usable):
                raise SymbolicCalculatorError("Invalid input: no resolved solution with this number; unresolved and unavailable alternatives cannot be selected.")
            selected = usable[offset]
        conditions = tuple(dict.fromkeys((*selected.conditions, *index.conditions)))
        conditions = _conditions(selected.conditions, additional=index.conditions)
        return selected if conditions == selected.conditions else replace(selected, conditions=conditions)
    except _ConditionInapplicable as exc:
        return replace(selected, expression=_UnavailableDiagnostic(str(exc)), conditions=())
    except (SymbolicCalculatorError, ValueError, TypeError, ArithmeticError, NotImplementedError) as exc:
        return replace(selected, expression=_UnavailableDiagnostic(f"Operation is unavailable: {exc}"), conditions=conditions)


def _shape(value):
    return value.expression.problem if isinstance(value.expression, _UnresolvedProblem) else value.expression


def _dispatch_call(context: CalculatorContext, name: str, args: Sequence[_Value], *, direction=None, conversion=None):
    """Execute one branch; each operation owns its applicability and failures."""
    conditions = tuple(dict.fromkeys(c for value in args for c in value.conditions))
    try:
        sp = require_sympy()
        def count(*counts):
            if len(args) not in counts:
                raise SymbolicCalculatorError(f"Invalid input: {name} requires {' or '.join(map(str, counts))} arguments.")
        structural = name in {"tuple", "matrix", "eq"}
        if name == "subs":
            count(2, 3)
            if _unusable(args[1:]) is None:
                if len(args) == 2:
                    source, relationships = args
                    provenance = ("Original relationships used: " + _render(relationships.expression),)
                    if relationships.conditions:
                        provenance += ("Original applicability (provenance, not current conditions): "
                                       + "; ".join(_render(c) for c in relationships.conditions),)
                    mapping = relationships.expression.replacements
                    units = relationships.quantity if isinstance(relationships.quantity, tuple) else (None,) * len(mapping)
                    result = _substitute_coordinates(context, source, tuple(mapping), tuple(mapping.values()), units, conditions=relationships.conditions)
                    entries = tuple(replace(v, provenance=(*v.provenance, *provenance)) for v in _entries(result))
                    return _Alternatives(entries) if isinstance(result, _Alternatives) else entries[0]
                targets = args[1].expression if isinstance(args[1].expression, sp.Tuple) else (args[1].expression,)
                values = args[2].expression if isinstance(args[2].expression, sp.Tuple) else (args[2].expression,)
                if len(targets) != len(values) or len(set(targets)) != len(targets):
                    raise SymbolicCalculatorError("Invalid input: substitution rosters must match and targets must be unique.")
                targets = tuple(_coordinate(context, target, name) for target in targets)
                units = args[2].quantity if isinstance(args[2].expression, sp.Tuple) and isinstance(args[2].quantity, tuple) else (args[2].quantity,) * len(values)
                return _substitute_coordinates(context, args[0], targets, values, units, conditions=(*args[1].conditions, *args[2].conditions))
        conditions = _conditions(args[0].conditions, additional=(c for v in args[1:] for c in v.conditions)) if args else ()
        unresolved = _unusable(args)
        if unresolved is not None and (isinstance(unresolved.expression, _UnavailableDiagnostic)
                                       or not (structural or name in {"steady_state", "stationary_state"})):
            return replace(unresolved, conditions=conditions)
        if name == "solve":
            count(2)
            return _solve(context, *args, conditions)
        if name in {"integrate", "limit", "series"}:
            count(*({"integrate": (2, 4), "limit": (3,), "series": (4,)}[name]))
            return _calculus_call(context, name, args, conditions, direction=direction)
        if name in {"dt", "partial"}:
            count(*((1, 2) if name == "dt" else (2, 3)))
            variable = None if name == "dt" else _coordinate(context, args[1].expression, name)
            order = _integer(args[-1], name) if len(args) == (2 if name == "dt" else 3) else 1
            coordinate = SECOND if variable is None else context.coordinate_units.get(str(variable))
            quantity = _quantity_map(args[0].expression, args[0].quantity, lambda unit: unit.divide(coordinate.power(Fraction(order)))
                                     if isinstance(unit, Unit) and coordinate is not None else None)
            return _with_quantity(_differentiate(context, args[0] if conditions == args[0].conditions else replace(args[0], conditions=conditions), variable, order), quantity)
        if name == "approx":
            count(1, 2)
            digits = _integer(args[1], name, minimum=1) if len(args) == 2 else 15
            expression = _map(args[0].expression, lambda x: _native_substitute(x, _fixed_constant_values()).evalf(digits))
            return _with_quantity(_admit_native(expression, conditions, notices=(f"Approximation: {digits} significant decimal digits.",)), args[0].quantity)
        if name in {"steady_state", "stationary_state"}:
            if name == "steady_state":
                if not args:
                    raise SymbolicCalculatorError("Supply species: steady_state(Int) or steady_state([X,Y]). Use stationary_state() for the whole mechanism.")
                count(1, 2)
                selected = _shape(args[0]) if isinstance(_shape(args[0]), sp.Tuple) else (_shape(args[0]),)
                if not selected or len(set(selected)) != len(selected) or any(x not in [context.symbols[n] for n in context.species_names] for x in selected):
                    raise SymbolicCalculatorError("Invalid input: steady_state requires distinct accepted species, not equations or parameter names.")
                extra_args = args[1:]
            else:
                count(0, 1)
                selected = tuple(context.symbols[n] for n in context.species_names)
                extra_args = args
            def solve_rates(rates):
                applicable = _conditions(conditions, additional=(c for rate in rates for c in rate.conditions))
                failure = _unusable(rates) or unresolved
                if failure is not None and isinstance(failure.expression, _UnavailableDiagnostic):
                    return replace(failure, conditions=applicable)
                equations = tuple(sp.Eq(_shape(rate), 0, evaluate=False) for rate in rates)
                if extra_args:
                    extra = _shape(extra_args[0])
                    equations += tuple(extra) if isinstance(extra, sp.Tuple) else (extra,)
                if failure is not None:
                    return _Value(_UnresolvedProblem(sp.Tuple(*equations), failure.expression.explanation), applicable)
                meanings = tuple(DIMENSIONLESS if _known_quantity(rate.quantity) else None for rate in rates)
                if extra_args:
                    extra_units = extra_args[0].quantity
                    meanings += extra_units if isinstance(extra_units, tuple) else (extra_units,)
                return _solve(context, _Value(sp.Tuple(*equations), applicable, quantity=meanings), _Value(sp.Tuple(*selected)), applicable)
            rates = tuple(context.rhs[context.species_names.index(str(x))] for x in selected)
            answers = []
            for branch in _branches(rates):
                try:
                    answers.extend(_entries(solve_rates(branch)))
                except _ConditionInapplicable:
                    continue
            return _Alternatives(tuple(answers))
        if name in {"odes", "jacobian"}:
            count(0)
            if name == "jacobian":
                return context.jacobian
            answers = []
            for rates in _branches(context.rhs):
                applicable = _conditions(additional=(c for rate in rates for c in rate.conditions))
                unavailable = _unusable(rates)
                expression = sp.ImmutableMatrix([_shape(rate) for rate in rates]) if unavailable is None or isinstance(unavailable.expression, _UnresolvedProblem) else None
                answers.append(replace(unavailable, conditions=applicable) if expression is None else
                               _Value(_UnresolvedProblem(expression, unavailable.expression.explanation) if unavailable else expression, applicable,
                                      quantity=tuple((rate.quantity,) for rate in rates)))
            return _Alternatives(tuple(answers)) if any(isinstance(rate, _Alternatives) for rate in context.rhs) else answers[0]
        quantity = None
        def native():
            nonlocal quantity
            if name in {"+", "-", "*", "/", "^", "**"}:
                if any(isinstance(arg.expression, sp.MatrixBase) for arg in args):
                    expression, quantity = _matrix_arithmetic(name, *args)
                else:
                    expression, quantity = _scalar_operation(name, args, conversion=conversion)
                return expression
            if name in {"pi", "e", "infinity"}:
                count(0)
                quantity = DIMENSIONLESS
                return {"pi": sp.pi, "e": sp.E, "infinity": sp.oo}[name]
            if name in _FUNCTIONS:
                expression, quantity = _scalar_operation(name, args, conversion=conversion)
                return expression
            if name in _TRANSFORMS:
                count(1)
                quantity = args[0].quantity
                return _map(args[0].expression, getattr(sp, name))
            if name == "collect":
                count(2)
                quantity = args[0].quantity
                variable = _coordinate(context, args[1].expression, name)
                return _map(args[0].expression, lambda x: sp.collect(x, variable))
            if name == "initial":
                count(1)
                species = str(args[0].expression)
                if species not in context.species_names:
                    raise SymbolicCalculatorError("Invalid input: initial requires a species name.")
                quantity = MOLAR
                return context.symbols[f"initial({species})"]
            if name == "tuple":
                quantity = tuple(arg.quantity for arg in args)
                return sp.Tuple(*(_shape(item) for item in args))
            if name == "matrix":
                count(1)
                rows = _shape(args[0])
                if isinstance(rows, sp.Tuple):
                    rows = sp.Tuple(*(row if isinstance(row, sp.Tuple) else sp.Tuple(row) for row in rows))
                if not isinstance(rows, sp.Tuple) or not rows or any(not row for row in rows):
                    raise SymbolicCalculatorError("Invalid input: matrix requires nonempty rows, e.g. matrix([[1,2],[3,4]]).")
                if len({len(row) for row in rows}) != 1 or any(not isinstance(x, sp.Expr) for row in rows for x in row):
                    raise SymbolicCalculatorError("Invalid input: matrix rows require equally many scalar entries.")
                quantity = tuple(row if isinstance(row, tuple) else (row,) for row in args[0].quantity) if isinstance(args[0].quantity, tuple) else None
                return sp.ImmutableMatrix(rows)
            if name == "eq":
                count(2)
                expression, quantity = _equations(*args)
                return expression
            if name == "at":
                count(2, 3)
                expression = args[0].expression
                indices = tuple(_integer(arg, name) for arg in args[1:])
                try:
                    if len(indices) == 2 and isinstance(expression, sp.MatrixBase):
                        quantity = _matrix_cell(args[0], *indices).quantity
                        return expression[indices]
                    if len(indices) == 1 and isinstance(expression, (sp.Tuple, sp.MatrixBase)):
                        units = args[0].quantity
                        if isinstance(units, tuple):
                            units = tuple(cell for row in units for cell in row) if isinstance(expression, sp.MatrixBase) else units
                            quantity = units[indices[0]]
                        return expression[indices[0]]
                    raise TypeError("unsupported indexed object")
                except (IndexError, TypeError) as exc:
                    raise SymbolicCalculatorError(f"Invalid input: {name}: {exc}.") from exc
            if name in {"transpose", "det"}:
                count(1)
                matrix = args[0].expression
                if not isinstance(matrix, sp.MatrixBase):
                    raise SymbolicCalculatorError(f"Invalid input: {name} requires a matrix.")
                if name == "det" and matrix.rows != matrix.cols:
                    raise SymbolicCalculatorError("Invalid input: det requires a square matrix.")
                if name == "transpose":
                    quantity = tuple(zip(*args[0].quantity)) if isinstance(args[0].quantity, tuple) else None
                    return matrix.T
                expression, quantity = _determinant(args[0])
                return expression
            if name == "remove_order":
                count(1)
                quantity = args[0].quantity
                def remove(x):
                    result = x.removeO()
                    if result.has(sp.Order):
                        raise SymbolicCalculatorError("Unsupported: native remove_order does not remove this nested remainder; apply it to the series before nesting.")
                    return result
                return _map(args[0].expression, remove)
            raise SymbolicCalculatorError(f"Unsupported calculator operation {name!r}.")
        try:
            expression = native()
        except (SymbolicCalculatorError, ValueError, TypeError, ArithmeticError, NotImplementedError):
            if unresolved is not None:
                return replace(unresolved, conditions=conditions)
            raise
        notices = ("Series order term explicitly removed; the result is truncated.",) if name == "remove_order" and args[0].expression.has(sp.Order) else ()
        if unresolved is not None:
            return _Value(_UnresolvedProblem(expression, unresolved.expression.explanation), conditions)
        if args and expression == args[0].expression:
            return replace(args[0], quantity=quantity, conditions=conditions, notices=(*args[0].notices, *notices))
        if structural or name in {"at", "transpose", "initial"}:
            return _Value(expression, conditions, notices, quantity=quantity)
        return _with_quantity(_admit_native(expression, conditions, notices=notices), quantity)
    except _ConditionInapplicable:
        raise
    except (SymbolicCalculatorError, AlgebraError, ValueError, TypeError, ArithmeticError, NotImplementedError) as exc:
        if name == "subs":
            conditions = _conditions(args[0].conditions, additional=(c for value in args[1:] for c in value.conditions)) if args else ()
        return _Value(_UnavailableDiagnostic(f"Operation is unavailable: {exc}"), conditions)


def _conditions_cover_interval(conditions, variable, lower, upper):
    """Establish active applicability on the open interval using native truth."""
    sp = require_sympy()
    try:
        copied = sp.Tuple(*conditions, lower, upper)
        copied = copied.xreplace({x: sp.Rational(x) for x in copied.atoms(sp.Float)})
        lower, upper = (_native_substitute(x, _fixed_constant_values()) for x in copied[-2:])
        if sp.simplify(lower > upper, doit=False) is sp.true:
            lower, upper = upper, lower
        if lower == upper and lower.is_finite is True:
            return True
        interior = sp.Dummy("interior", positive=True)
        if lower == -sp.oo and upper == sp.oo:
            point = sp.Dummy("interior", real=True)
        elif upper == sp.oo and lower.is_real is True and lower.is_finite is True:
            point = lower + interior
        elif lower == -sp.oo and upper.is_real is True and upper.is_finite is True:
            point = upper - interior
        elif (all(x.is_real is True and x.is_finite is True for x in (lower, upper))
              and sp.simplify(lower < upper, doit=False) is sp.true):
            point = sp.cancel((lower + upper*interior)/(1 + interior))
        else:
            return False
        remaining = _conditions(before_substitution=copied[:-2], replacements={variable: point})
        return sp.simplify(sp.And(*remaining), doit=False) is sp.true
    except (SymbolicCalculatorError, ValueError, TypeError, ArithmeticError, NotImplementedError):
        # Unestablished interval coverage is not false applicability of the input.
        return False


def _calculus_call(context: CalculatorContext, name: str, args: Sequence[_Value], conditions, *, direction=None) -> _Value | _Alternatives:
    """Use native calculus, with bounded adaptation of actual answer conditions."""
    applicable, notices = conditions, ()
    try:
        sp = require_sympy()
        expression = _scalar(args[0], name)
        variable = _coordinate(context, args[1].expression, name)
        coordinate_unit = context.coordinate_units.get(str(variable))
        quantity = args[0].quantity
        if name == "integrate":
            quantity = _quantity_map(expression, quantity, lambda unit: unit.multiply(coordinate_unit)
                                     if isinstance(unit, Unit) and coordinate_unit is not None else None)
        args = list(args)
        positions = range(2, 4) if name == "integrate" and len(args) == 4 else (2,) if name in {"limit", "series"} else ()
        for index in positions:
            point, known = _coordinate_argument(context, variable, args[index])
            args[index] = replace(args[index], expression=point)
            if not known:
                # An unknown point or bound leaves the result in the scientific
                # coordinates of its known finished quantity.
                expression, quantity = _scientific(expression, quantity), None
        companion = tuple(c for arg in args[1:] for c in arg.conditions)
        dependent = tuple(c for c in conditions if c in args[0].conditions and variable in c.free_symbols)
        independent = tuple(c for c in conditions if c not in dependent or c in companion)
        restricted = False
        operators = (sp.Integral, sp.Limit, sp.Subs, sp.Derivative, sp.ConditionSet)
        inherited = {item for item in expression.atoms(*operators) if variable not in item.free_symbols}
        if name == "integrate" and len(args) == 2:
            result = sp.integrate(expression, variable)
            independent = conditions
        else:
            if name == "limit" and dependent:
                raise SymbolicCalculatorError("Unsupported: Calculator cannot adapt a limit with answer conditions on its active coordinate. Specialize or select an applicable answer first.")
            if name == "integrate":
                lower, upper = (_scalar(arg, name) for arg in args[2:])
                fixed = _fixed_constant_values().keys()
                if dependent and ((lower.free_symbols | upper.free_symbols) - fixed
                                  or any(c.free_symbols - {variable} - fixed for c in dependent)):
                    raise SymbolicCalculatorError("Unsupported: definite integration with active answer conditions requires coordinate-free bounds and conditions involving only the integration coordinate. Substitute other coordinates first.")
                if dependent and _conditions_cover_interval(dependent, variable, lower, upper):
                    dependent = ()
                restricted = bool(dependent)
            # This wrapper exists only for this backend call. It represents a legitimate
            # answer condition, never a domain inferred from ordinary operands.
            bound = sp.Dummy(variable.name, real=True) if dependent else variable
            prepared = _native_substitute(expression, {variable: bound}) if restricted else expression
            operand = (sp.Piecewise(
                (prepared if restricted else expression.xreplace({variable: bound}),
                 _native_substitute(sp.And(*dependent), {variable: bound}, equations=False)),
                (sp.nan, True), evaluate=False,
            ) if dependent else expression)
            if name == "integrate":
                applicable = independent
                result = sp.integrate(operand, (bound, lower, upper), conds="piecewise")
            elif name == "limit":
                point = _scalar(args[2], name)
                direction = ("-" if direction == "left" else "+") if direction else "+-"
                applicable = independent
                result = sp.Limit(operand, bound, point, dir=direction).doit()
            else:
                point = _scalar(args[2], name, finite=True)
                order = _integer(args[3], name, minimum=1)
                direction = direction or "right"
                result = sp.series(operand, bound, point, order, dir="-" if direction == "left" else "+")
                result = result.xreplace({bound: variable})
                independent = conditions
                notices = (f"Local series about {_render(variable)}={_render(point)}, direction {direction}.",)
            if name == "series" and dependent:
                if any(item.has(sp.nan) for item in result.atoms(*operators)):
                    raise SymbolicCalculatorError("Unsupported: stock SymPy retained the condition-restricted operator; Calculator cannot publish that temporary restriction as an ordinary expression.")

        def classify(body, restrictions, conditional):
            if restricted:
                current = sp.Tuple(body, *restrictions)
                if current.atoms(*operators) - inherited or bound in current.free_symbols:
                    raise SymbolicCalculatorError("Unsupported: stock SymPy retained the condition-restricted operator; Calculator cannot publish that temporary restriction as an ordinary expression.")
            if conditional and body.atoms(*operators) - inherited:
                return _UnresolvedProblem(body, "The native conditional fallback remains unresolved.")
            return body

        return _with_quantity(_admit_native(result, independent, notices=notices, classify=classify), quantity)
    except _ConditionInapplicable:
        raise
    except (SymbolicCalculatorError, ValueError, TypeError, ArithmeticError, NotImplementedError) as exc:
        return _Value(_UnavailableDiagnostic(f"Operation is unavailable: {exc}"), applicable, notices)


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    text: str


def _validate_numeric_literal(literal: str) -> None:
    literal = literal.replace("_", "")
    try:
        exact = Decimal(literal)
        value = float(literal)
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise SymbolicCalculatorError("Invalid input: numeric literal.") from exc
    if not math.isfinite(value):
        raise SymbolicCalculatorError("Invalid input: numeric literal exceeds finite query ingress.")
    if exact != 0 and value == 0:
        raise SymbolicCalculatorError("Invalid input: numeric literal is below query ingress range.")


class _CalculatorExpressionParser:
    """No eval, Python AST execution, attribute access, imports, or arbitrary calls."""
    def __init__(self, context: CalculatorContext, source: str):
        self.context = context
        self.tokens = self._tokenize(source)
        self.index = 0

    @staticmethod
    def _tokenize(source: str) -> list[_Token]:
        tokens = []
        i = 0
        while i < len(source):
            char = source[i]
            if char.isspace():
                i += 1
                continue
            start = i
            if char.isalpha() or char == "_":
                i += 1
                while i < len(source) and (source[i].isalnum() or source[i] == "_"):
                    i += 1
                tokens.append(_Token("IDENT", source[start:i]))
                continue
            if char.isdigit() or char == ".":
                i += 1
                while i < len(source) and (source[i].isdigit() or source[i] in "_."):
                    i += 1
                if i < len(source) and source[i] in "eE":
                    i += 1
                    if i < len(source) and source[i] in "+-":
                        i += 1
                    while i < len(source) and (source[i].isdigit() or source[i] == "_"):
                        i += 1
                literal = source[start:i]
                _validate_numeric_literal(literal)
                tokens.append(_Token("NUMBER", literal))
                continue
            if source[i:i+2] == "**":
                tokens.append(_Token("^", "**"))
                i += 2
            elif char in "+-*/^(),[]":
                tokens.append(_Token(char, char))
                i += 1
            else:
                raise SymbolicCalculatorError(f"Invalid input: unexpected token {char!r}.")
        return [*tokens, _Token("EOF", "")]

    def at(self, kind):
        return self.tokens[self.index].kind == kind

    def take(self, kind=None):
        token = self.tokens[self.index]
        if kind is not None and token.kind != kind:
            raise SymbolicCalculatorError(f"Invalid input: expected {kind}, got {token.text or 'end of query'!r}.")
        self.index += 1
        return token.text

    def parse(self) -> _Value | _Alternatives:
        value = self.additive()
        self.take("EOF")
        return value

    def additive(self):
        value = self.multiplicative()
        while self.at("+") or self.at("-"):
            operator = self.take()
            value = _evaluate_call(self.context, operator, (value, self.multiplicative()))
        return value

    def multiplicative(self):
        value = self.unary()
        while self.at("*") or self.at("/"):
            operator = self.take()
            value = _evaluate_call(self.context, operator, (value, self.unary()))
        return value

    def unary(self):
        if self.at("+") or self.at("-"):
            operator = self.take()
            value = self.unary()
            return value if operator == "+" else _evaluate_call(self.context, "*", (_Value(require_sympy().Integer(-1), quantity=DIMENSIONLESS), value))
        value = self.primary()
        if self.at("^"):
            self.take()
            value = _evaluate_call(self.context, "^", (value, self.unary()))
        return value

    def direct_targets(self, tokens):
        """Find authored names in an already parsed direct target or roster."""
        while tokens and tokens[0].kind == "+":
            tokens = tokens[1:]
        if len(tokens) == 1 and tokens[0].kind == "IDENT":
            yield tokens[0].text
            return
        if not tokens or tokens[0].kind not in {"(", "["}:
            return
        depth = 0
        for index, token in enumerate(tokens):
            depth += token.kind in {"(", "["}
            depth -= token.kind in {")", "]"}
            if depth == 0:
                if index != len(tokens)-1:
                    return
                break
        inner = tokens[1:-1]
        if tokens[0].kind == "(":
            yield from self.direct_targets(inner)
        else:
            start = depth = 0
            for index, token in enumerate(inner):
                depth += token.kind in {"(", "["}
                depth -= token.kind in {")", "]"}
                if token.kind == "," and depth == 0:
                    yield from self.direct_targets(inner[start:index])
                    start = index+1
            yield from self.direct_targets(inner[start:])

    def check_targets(self, name, args, spans):
        target = (
            1 if name in {"solve", "partial", "integrate", "limit", "series", "collect"}
            or name == "subs" and len(args) == 3
            else 0 if name in {"initial", "steady_state"}
            else None
        )
        if target is None or len(spans) <= target:
            return
        for authored in self.direct_targets(spans[target]):
            definition = self.context.let_definitions.get(authored)
            if definition is None:
                symbol = _canonical_symbol(self.context, authored)
                if symbol in self.context.coordinates or str(symbol) in _FIXED_CONSTANTS:
                    continue
                definition = self.context.parameter_definitions.get(str(symbol), "derived model quantity")
            raise SymbolicCalculatorError(
                f"{name}: {authored} is defined as {definition}; use independent coordinates from its definition."
            )

    def primary(self):
        sp = require_sympy()
        if self.at("NUMBER"):
            return _Value(sp.Rational(self.take().replace("_", "")), quantity=DIMENSIONLESS)
        if self.at("("):
            self.take()
            value = self.additive()
            self.take(")")
            return value
        if self.at("["):
            self.take()
            # Existing Algebra bracket spelling is unambiguous for one species.
            if self.at("IDENT") and self.tokens[self.index+1].kind == "]" and self.tokens[self.index].text in self.context.species_names:
                name = self.take()
                self.take("]")
                if self.at("IDENT") and self.tokens[self.index].text == "_0":
                    self.take()
                    name = f"initial({name})"
                return _Value(self.context.symbols[name], quantity=MOLAR)
            values = []
            if not self.at("]"):
                values.append(self.additive())
                while self.at(","):
                    self.take()
                    values.append(self.additive())
            self.take("]")
            return _evaluate_call(self.context, "tuple", values)
        name = self.take("IDENT")
        if name == "d" and self.at("("):
            self.take()
            inner_start = self.index
            inner = self.additive()
            inner_tokens = self.tokens[inner_start:self.index]
            self.take(")")
            self.take("/")
            denominator = self.take("IDENT")
            if denominator == "dt":
                return _evaluate_call(self.context, "dt", (inner,))
            legacy_rate = (len(inner_tokens) == 3 and inner_tokens[0].text.startswith("d")
                           and inner_tokens[0].text[1:] in self.context.species_names
                           and inner_tokens[1].kind == "/" and inner_tokens[2].text == "dt")
            if legacy_rate and denominator.startswith("d") and denominator[1:] in self.context.species_names and denominator not in self.context.symbols and denominator not in self.context.let_definitions:
                return _evaluate_call(self.context, "partial", (inner, _Value(self.context.symbols[denominator[1:]])))
            raise SymbolicCalculatorError("Invalid input: use dt(F) or partial(F, coordinate).")
        if name not in self.context.symbols and name not in self.context.let_definitions and name.startswith("d") and name[1:] in self.context.species_names and self.at("/"):
            denominator = self.tokens[self.index+1]
            if denominator.kind == "IDENT" and (denominator.text == "dt" or denominator.text.startswith("d") and denominator.text[1:] in self.context.species_names and denominator.text not in self.context.symbols and denominator.text not in self.context.let_definitions):
                self.take()
                denominator = self.take()
                rhs = self.context.rhs[self.context.species_names.index(name[1:])]
                if denominator == "dt":
                    return rhs
                other = self.context.rhs[self.context.species_names.index(denominator[1:])]
                result = _evaluate_call(self.context, "/", (rhs, other))
                return _evaluate_call(self.context, "simplify", (result,))
        if self.at("("):
            if name not in _OPERATIONS:
                raise SymbolicCalculatorError(f"Unsupported calculator operation {name!r}.")
            self.take()
            args, spans = [], []
            direction = None
            if not self.at(")"):
                while True:
                    if name == "limit" and len(args) == 3 or name == "series" and len(args) == 4:
                        direction = self.take("IDENT")
                        if direction not in {"left", "right"}:
                            raise SymbolicCalculatorError("Invalid input: direction is left or right.")
                        break
                    start = self.index
                    args.append(self.additive())
                    spans.append(self.tokens[start:self.index])
                    if not self.at(","):
                        break
                    self.take()
            self.take(")")
            self.check_targets(name, args, spans)
            return _evaluate_call(self.context, name, args, direction=direction)
        if name in self.context.let_definitions:
            return self.context.values[name]
        symbol = _canonical_symbol(self.context, name)
        return self.context.values[str(symbol)]


def _render(expression: Any) -> str:
    sp = require_sympy()
    from sympy.printing.str import StrPrinter
    if isinstance(expression, _Relationships):
        return ", ".join(f"{_render(key)} = {_render(rhs)}" for key, rhs in expression.replacements.items())

    class Printer(StrPrinter):
        def _print_ImaginaryUnit(self, expr):
            return "sqrt(-1)"

        def _print_Infinity(self, expr):
            return "infinity()"

        def _print_NegativeInfinity(self, expr):
            return "-infinity()"

        def _print_Pi(self, expr):
            return "pi()"

        def _print_Exp1(self, expr):
            return "e()"

        def _print_Abs(self, expr):
            return "abs(" + self._print(expr.args[0]) + ")"

        def _print_Min(self, expr):
            return "min(" + ", ".join(self._print(arg) for arg in expr.args) + ")"

        def _print_Max(self, expr):
            return "max(" + ", ".join(self._print(arg) for arg in expr.args) + ")"

        def _print_Heaviside(self, expr):
            if expr.args[1] == 0:
                return "heaviside(" + self._print(expr.args[0]) + ")"
            return "Heaviside(" + ", ".join(self._print(arg) for arg in expr.args) + ")"

        def _print_Float(self, expr):
            from sympy.core.evalf import prec_to_dps
            return f"approx({super()._print_Float(expr)}, {prec_to_dps(expr._prec)})"

    printer = Printer()
    if isinstance(expression, sp.Tuple):
        items = [_render(item) for item in expression]
        return "[" + (f"({items[0]})" if len(items) == 1 else ", ".join(items)) + "]"
    if isinstance(expression, sp.MatrixBase):
        return "matrix([" + ", ".join("[" + ", ".join(_render(x) for x in expression.row(i)) + "]" for i in range(expression.rows)) + "])"
    if isinstance(expression, sp.Equality):
        return f"eq({_render(expression.lhs)}, {_render(expression.rhs)})"
    return printer.doprint(expression)


def _pretty_render(expression: Any) -> str:
    """Lay out mathematics without changing literal coordinate names."""
    sp = require_sympy()
    from sympy.printing.pretty.pretty import PrettyPrinter
    from sympy.printing.pretty.stringpict import prettyForm

    class Printer(PrettyPrinter):
        def _print_Symbol(self, expr, bold_name=False):
            return prettyForm(expr.name)

    printer = Printer({"use_unicode": True, "wrap_line": False})
    if isinstance(expression, _Relationships):
        return "\n\n".join(printer.doprint(sp.Eq(key, rhs, evaluate=False))
                           for key, rhs in expression.replacements.items())
    return printer.doprint(expression)


def _native_components(expression):
    """Expose evaluated children and retain the native shape and binding scope."""
    sp = require_sympy()
    if isinstance(expression, _Relationships):
        return tuple(expression.replacements.values()), lambda values: _Relationships(dict(zip(expression.replacements, values, strict=True)))
    if isinstance(expression, sp.MatrixBase):
        return tuple(expression), lambda values: sp.ImmutableMatrix(expression.rows, expression.cols, values)
    if isinstance(expression, (sp.Lambda, sp.Derivative)):
        return (), lambda _: expression
    if isinstance(expression, sp.ConditionSet):
        return (expression.base_set,), lambda values: sp.ConditionSet(expression.sym, expression.condition, values[0])
    if isinstance(expression, sp.Integral):
        exposed, positions, outer = [], [], set()
        for i in reversed(range(len(expression.limits))):
            limit = expression.limits[i]
            for j, bound in enumerate(limit[1:], 1):
                if not bound.free_symbols & outer:
                    positions.append((i, j))
                    exposed.append(bound)
            outer.add(limit[0])
        def integral(values):
            limits = [list(limit) for limit in expression.limits]
            for (i, j), value in zip(positions, values, strict=True):
                limits[i][j] = value
            return sp.Integral(expression.function, *limits)
        return tuple(exposed), integral
    if isinstance(expression, sp.Subs):
        return tuple(expression.point), lambda values: sp.Subs(expression.expr, expression.variables, values)
    if isinstance(expression, sp.Limit):
        body, coordinate, point, direction = expression.args
        return (point,), lambda values: sp.Limit(body, coordinate, values[0], dir=str(direction))
    return expression.args, lambda values: (expression if values == expression.args else
        expression.func(*values, evaluate=False) if isinstance(expression, sp.Equality) else expression.func(*values))


def _admit_native(expression, conditions=(), *, notices=(), classify=None):
    """Admit new native branches directly, under established current applicability."""
    sp = require_sympy()
    fixed = _fixed_constant_values()
    unresolved = expression if isinstance(expression, _UnresolvedProblem) else None
    template = unresolved.problem if unresolved else expression
    if unresolved and isinstance(template, sp.Set):
        problem = _native_substitute(template, fixed, equations=False)
        return _Alternatives(()) if problem == sp.S.EmptySet else _Value(replace(unresolved, problem=problem), conditions, notices)

    def conditional(node, choices):
        if isinstance(node, sp.Piecewise):
            return conditional(choices[node], choices) if node in choices else node
        children, _ = _native_components(node)
        return next((found for child in children if (found := conditional(child, choices)) is not None), None)

    def materialize(node, choices, *, check=False):
        if isinstance(node, sp.Piecewise):
            return materialize(choices[node], choices, check=check)
        if check and isinstance(node, sp.Symbol):
            node = fixed.get(node, node)
        children, rebuild = _native_components(node)
        values = tuple(materialize(child, choices, check=check) for child in children)
        result = rebuild(values)
        if check and (result is sp.nan or result is sp.zoo):
            raise SymbolicCalculatorError("Undefined: native mathematics produced an undefined value.")
        return result

    unchanged = tuple(c for c in conditions if conditional(c, {}) is None)
    pending = [((), {})]
    answers, branched = [], False
    while pending:
        guards, choices = pending.pop()
        node = next((found for root in (*guards, *conditions, template)
                     if (found := conditional(root, choices)) is not None), None)
        if node is not None:
            branched = True
            prior, branches = sp.false, []
            for body, predicate in node.args:
                branches.append(((*guards, sp.And(predicate, sp.Not(prior))), {**choices, node: body}))
                prior = sp.Or(prior, predicate)
            if prior != sp.true:
                branches.append(((*guards, sp.Not(prior)), {**choices, node: sp.nan}))
            pending.extend(reversed(branches))
            continue
        restrictions = conditions
        try:
            if branched:
                # Predicates are transformed lazily: a false guard wins over an
                # invalid comparison belonging only to that inapplicable branch.
                changed = _boolean_result((*guards, *(c for c in conditions if c not in unchanged)),
                    lambda c: _native_substitute(materialize(c, choices), fixed, equations=False))
                restrictions = _condition_tuple(sp.And(*unchanged, changed))
            materialize(template, choices, check=True)
            body = materialize(template, choices) if choices else template
            body = replace(unresolved, problem=body) if unresolved else body
            if classify is not None:
                body = classify(body, restrictions, branched)
        except _ConditionInapplicable:
            continue
        except (SymbolicCalculatorError, ValueError, TypeError, ArithmeticError, NotImplementedError) as exc:
            body = _UnavailableDiagnostic(str(exc))
        answers.append(_Value(body, restrictions, notices))
    return _Alternatives(tuple(answers)) if branched else answers[0]


def _printed_input_supported(context: CalculatorContext, expression, text: str) -> bool:
    """Check emitted vocabulary/number ingress without executing another query."""
    sp = require_sympy()
    if expression.atoms(sp.NumberSymbol) - {sp.pi, sp.E}:
        return False
    try:
        tokens = _CalculatorExpressionParser._tokenize(text)
    except SymbolicCalculatorError:
        return False
    for index, token in enumerate(tokens):
        if token.kind != "IDENT":
            continue
        if tokens[index + 1].kind == "(":
            if token.text not in _OPERATIONS:
                return False
        elif token.text not in context.symbols and token.text not in context.let_definitions:
            return False
    return True


def evaluate_symbolic_query(context: CalculatorContext, query: str) -> SymbolicCalculatorResult:
    query = query.strip()
    if not query:
        raise SymbolicCalculatorError("Invalid input: empty Calculator query.")
    try:
        value = _CalculatorExpressionParser(context, query).parse()
        return _package_result(context, query, value)
    except SymbolicCalculatorError:
        raise
    except (ValueError, TypeError, ArithmeticError, NotImplementedError, AttributeError, RecursionError) as exc:
        raise SymbolicCalculatorError(f"Unsupported or undefined mathematics: {exc}") from exc


def _package_result(context: CalculatorContext, query: str, result) -> SymbolicCalculatorResult:
    expression = query
    family = isinstance(result, _Alternatives)
    entries = _entries(result)
    lines, display_html_lines, choices = [], [], []

    def append(text, display=None, display_html=None):
        lines.append(text)
        visible = text if display is None else display
        display_html_lines.append(escape(visible) if display_html is None else display_html)

    for value in entries:
        payload = value.expression
        if isinstance(payload, _UnavailableDiagnostic):
            append("Unavailable expression (not selectable):")
            append(payload.explanation)
        elif isinstance(payload, _UnresolvedProblem):
            append("Unresolved alternative (not a selectable solved answer):")
            append(_render(payload.problem), _pretty_render(payload.problem))
            append(payload.explanation)
        else:
            label = _render(payload)
            index = len(choices) + 1
            if family:
                append(f"Solution {index}")
            heading = "Relationships:" if isinstance(payload, _Relationships) else "Expression:" if family else ""
            append((heading + " " if heading else "") + label,
                   (heading + "\n" if heading else "") + _pretty_render(payload))
            reusable = f"solution({expression}, {index})" if family else query
            choices.append((f"Solution {index}: {label}" if family else "Result", reusable))
            if value.quantity is not None:
                unit_text = _quantity_text(value.quantity)
                append(
                    "Units: " + unit_text,
                    display_html='Units: <span style="white-space: nowrap">'
                    + _quantity_html(value.quantity) + "</span>",
                )
            if value.quantity is not None or family or isinstance(payload, _Relationships) or value.conditions or value.notices or value.provenance or not _printed_input_supported(context, payload, label):
                append("Reuse query: " + reusable)
        if value.conditions:
            append("Conditions: " + "; ".join(_render(c) for c in value.conditions),
                   "Conditions:\n" + "\n\n".join(_pretty_render(c) for c in value.conditions))
        for note in (*value.notices, *value.provenance):
            append(note)
        if family:
            append("")
    if not entries:
        append("No applicable real solutions (EmptySet).")
    if family:
        append("Complete query (all answers and unresolved possibilities): " + query)
    copy_text = "\n".join(lines).rstrip()
    if entries and all(isinstance(v.expression, _UnavailableDiagnostic) for v in entries):
        raise SymbolicCalculatorError(copy_text)
    if expression == "odes()":
        rows_text = "Rows: " + ", ".join(f"d{name}/dt" for name in context.species_names)
        display_html_lines.append(escape(rows_text))
    assumptions = (
        "Accepted-source symbolic system; no schedules, execution overrides, Fit candidates or Slider drafts.",
        "Coordinates are real; T is positive and fixed in time unless explicit expression dependence is supplied.",
        "Definitions are prepared before calculation. Independent parameters and initial concentrations are fixed in time; partial differentiates prepared mathematics.",
        "Independent inputs remain symbolic; supply concrete values explicitly with subs. Accepted defaults are not Calculator inputs.",
        "Physical constants retain their names and owned values; approximation does not increase source precision.",
        "Special functions, conditional/unresolved sets and some root/order forms may not round-trip as direct input; nested operation composition remains supported.",
        "Stock SymPy 1.14.0 supplies mathematical algorithms. Its known parameter-dependent rational integration bug (sympy/sympy#28596) is not corrected or independently verified by Kindred.",
        "Ordinary algebra and formal calculus use native CAS semantics, without historical operand domains. A simplified result does not certify the original expression's domain or a parameterized integral's regularity.",
        "Actual answer conditions, unavailable cases and unresolved alternatives are retained. Active-condition definite integrals require coordinate-free bounds and no other condition parameters; active-condition limits remain unsupported. Series retain direction and order, not inferred neighborhood conditions.",
    )
    return SymbolicCalculatorResult(
        query=query,
        result_html="<pre>" + "\n".join(display_html_lines).rstrip() + "</pre>",
        copy_text=copy_text,
        assumptions=assumptions,
        symbol_legend={"Species": context.species_names, "Parameters": tuple(sorted(context.parameter_namespace.canonical_names | context.parameter_definitions.keys())),
         "Fixed constants": _FIXED_CONSTANTS, "Time and temperature": ("t [s]", "T [K]"),
         "Energy coordinates": ("Ea, dG_act, dG_eq in J/mol",),
         "Coordinate units": tuple(f"{name} [{unit.symbol}]" for name, unit in dict(parameter_algebra_spec_from_mechanism(context.mechanism).quantity_units).items() if unit is not None and name in context.symbols),
         "Initial coordinates": tuple(f"initial({name})" for name in context.species_names)},
        mechanism_source=context.mechanism_source,
        reuse_queries=tuple(choices),
    )
