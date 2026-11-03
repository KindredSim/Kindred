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
from kindred.core.exceptions import located
from kindred.core.algebra.parser import BinaryNode, CallNode, IdentNode, NumberNode, SpeciesRefNode, UnaryNode
from kindred.core.algebra.printer import expression_to_source
from kindred.core.algebra.symbols import BUILTIN_FUNCTIONS, PROTECTED_NAMES
from kindred.core.mechanism_metadata import MechanismMetadataView
from kindred.core.mechanism_metadata import MechanismMetadataKeys
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

from .backend import require_sympy
from .calculator_chemistry import build_calculator_chemistry_relations
from .errors import UnsupportedSymbolicExpressionError
from .jacobian import build_symbolic_mechanism_public_projection, fractional_order_step

_FIXED_CONSTANTS = tuple(name for name in constants.__all__)
_TRANSFORMS = frozenset({"simplify", "factor", "expand", "cancel"})
_FUNCTIONS = frozenset(BUILTIN_FUNCTIONS) - {"ifelse"} | {"log"}
_OPERATIONS = _TRANSFORMS | _FUNCTIONS | {
    "collect", "subs", "approx", "partial", "dt",
    "odes", "jacobian", "matrix", "eq", "at", "transpose", "det", "initial",
    "solve", "steady_state", "stationary_state", "solution", "integrate", "limit", "series", "remove_order",
    "pi", "e", "infinity", "definitions",
}
# Answers are generic: a zero rate constant is an exceptional value, not part of the coordinate domain.
_POSITIVE_ROLES = frozenset({"k", "kf", "kr", "Keq", "A", "kappa", "degeneracy"})
_INCOMPLETE = "Found by SymPy's general solver, which can miss solutions."


class SymbolicCalculatorError(UnsupportedSymbolicExpressionError):
    """A factual invalid, unsupported, or undefined Calculator expression."""


# A tuple of quantities for tuples and relationship rosters; one quantity for a matrix.
Quantity = Unit | tuple["Quantity", ...] | None


@dataclass(frozen=True, slots=True)
class _Value:
    """One finished result: native mathematics, solve relationships or a diagnostic."""
    expression: Any
    notices: tuple[str, ...] = ()
    quantity: Quantity = None


@dataclass(frozen=True, slots=True)
class _Relationships:
    replacements: Mapping[Any, Any]


@dataclass(frozen=True, slots=True)
class _UnavailableDiagnostic:
    explanation: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class _Alternatives:
    """Two or more answers, or none, from one solve."""
    values: tuple[_Value, ...]


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


def _backend_failures():
    """Errors that end one calculation; SymPy reports mathematics it cannot complete with these."""
    from sympy.polys.polyerrors import BasePolynomialError
    return (SymbolicCalculatorError, ValueError, TypeError, ArithmeticError, NotImplementedError, BasePolynomialError)


def _unavailable(exc: Exception) -> _UnavailableDiagnostic:
    """Kindred states the failure; library text stays a copied detail."""
    if isinstance(exc, SymbolicCalculatorError):
        return _UnavailableDiagnostic(str(exc))
    return _UnavailableDiagnostic("Calculator could not complete this calculation.", str(exc))


def _fixed_constant_values():
    """Canonical identities and the owner's declared values as exact decimals."""
    sp = require_sympy()
    return {sp.Symbol(name, positive=True): sp.Rational(str(getattr(constants, name))) for name in _FIXED_CONSTANTS}


def _metadata(value, *inputs):
    """Notices of the operands carry over to the result."""
    notices = tuple(dict.fromkeys((*value.notices, *(n for v in inputs for n in v.notices))))
    return value if notices == value.notices else replace(value, notices=notices)


def _branches(values):
    """Enumerate one family of answers; no mathematics or metadata work."""
    families = [i for i, value in enumerate(values) if isinstance(value, _Alternatives)]
    if len(families) > 1:
        raise SymbolicCalculatorError("Only one operand can hold alternative answers; select an answer from the others with solution(result,n).")
    if not families:
        yield tuple(values)
    else:
        index = families[0]
        for value in values[index].values:
            yield (*values[:index], value, *values[index+1:])


def _unusable(values):
    """An unavailable operand makes the operation unavailable."""
    return next((v for v in values if isinstance(v.expression, _UnavailableDiagnostic)), None)


def _scalar(value: _Value, operation: str, *, finite=False) -> Any:
    sp = require_sympy()
    expression = value.expression
    if not isinstance(expression, sp.Expr):
        raise SymbolicCalculatorError(f"Unsupported: {operation} requires a scalar expression.")
    if finite and expression in (sp.oo, -sp.oo):
        raise SymbolicCalculatorError(f"Invalid input: {operation} requires a finite value.")
    return expression


def _map(expression: Any, operation) -> Any:
    """Apply a scalar operation to each expression; solve relationships keep their labels."""
    sp = require_sympy()
    if isinstance(expression, _Relationships):
        return _Relationships({key: _map(rhs, operation) for key, rhs in expression.replacements.items()})
    if isinstance(expression, sp.Tuple):
        return sp.Tuple(*(_map(item, operation) for item in expression))
    if isinstance(expression, sp.MatrixBase):
        return sp.ImmutableMatrix(expression.applyfunc(operation))
    if isinstance(expression, sp.Equality):
        return sp.Eq(_map(expression.lhs, operation), _map(expression.rhs, operation), evaluate=False)
    if not isinstance(expression, sp.Expr):
        raise SymbolicCalculatorError("Unsupported: this operation requires expressions, equations, tuples, matrices, or solve relationships.")
    return operation(expression)


def _current_symbols(result) -> set[Any]:
    """Discover current mathematics, never relationship labels."""
    payload = result.expression
    if isinstance(payload, _Relationships):
        payload = require_sympy().Tuple(*payload.replacements.values())
    return set(payload.free_symbols)


def _expand_factors(expression: Any, lets: Mapping[str, Any]) -> Any:
    """Each rate factor, a function of the concentrations its let reads, as
    that let's value; its derivatives are then taken."""
    if not lets:
        return expression
    sp = require_sympy()
    from sympy.core.function import AppliedUndef
    expanded = expression.replace(
        lambda item: isinstance(item, AppliedUndef) and item.func.__name__ in lets,
        lambda item: lets[item.func.__name__],
    )
    return expanded.replace(lambda item: isinstance(item, sp.Derivative), lambda item: item.doit())


def _native_substitute(expression: Any, mapping: Mapping[Any, Any]) -> Any:
    """Simultaneous native substitution. Equations stay unevaluated equations, and an
    integral keeps its integration variable apart from a replacement that introduces that name."""
    sp = require_sympy()
    if isinstance(expression, sp.Tuple):
        return sp.Tuple(*(_native_substitute(item, mapping) for item in expression))
    if isinstance(expression, sp.MatrixBase):
        return sp.ImmutableMatrix(expression.applyfunc(lambda item: _native_substitute(item, mapping)))
    if isinstance(expression, sp.Equality):
        return sp.Eq(_native_substitute(expression.lhs, mapping), _native_substitute(expression.rhs, mapping), evaluate=False)
    if any(x in mapping for order in expression.atoms(sp.Order) for x in order.variables):
        raise SymbolicCalculatorError("Unsupported: remove_order before substituting a series coordinate.")
    introduced = set().union(*(value.free_symbols for value in mapping.values()))
    # An antiderivative is written as its integral up to its own variable, so that variable can be renamed too.
    captured = {integral: sp.Integral(integral.function, *((x[0], x[0]) if len(x) == 1 else x for x in integral.limits)).as_dummy()
                for integral in expression.atoms(sp.Integral) if introduced.intersection(integral.variables)}
    return expression.xreplace(captured).subs(mapping, simultaneous=True)


def _substitute_value(value, replacements: Mapping[Any, Any]):
    """Substitute into one finished value; relationship labels stay labels."""
    roots = _current_symbols(value)
    mapping = {key: rhs for key, rhs in replacements.items() if key != rhs and key in roots}
    if not mapping:
        return value
    payload = value.expression
    if isinstance(payload, _Relationships):
        expression = _Relationships({key: _native_substitute(rhs, mapping) for key, rhs in payload.replacements.items()})
    else:
        expression = _native_substitute(payload, mapping)
    return replace(_admit_native(expression, value.notices), quantity=value.quantity)


def _calculus_domain(variable: Any) -> Any:
    """The coordinate's own domain."""
    sp = require_sympy()
    if variable.is_positive:
        return sp.Interval.open(0, sp.oo)
    return sp.Interval(0, sp.oo) if variable.is_nonnegative else sp.S.Reals


def _native_arithmetic(lhs: Any, operator: str, rhs: Any) -> Any:
    """Native scalar arithmetic; operands are admitted scalar expressions."""
    if operator == "+":
        return lhs + rhs
    if operator == "-":
        return lhs - rhs
    if operator == "*":
        return lhs * rhs
    if operator == "/":
        return lhs / rhs
    return lhs ** rhs


@dataclass(frozen=True, slots=True)
class SymbolicCalculatorResult:
    query: str
    result_html: str
    copy_text: str
    assumptions: tuple[str, ...]
    symbol_legend: Mapping[str, Sequence[str]]
    mechanism_source: str
    reuse_queries: tuple[tuple[str, str], ...] = ()
    failed: bool = False

    def full_context_copy_text(self, authored_definitions: tuple[str, ...]) -> str:
        legend = [f"{kind}: {', '.join(names) or 'none'}" for kind, names in self.symbol_legend.items()]
        return "\n".join((f"Query: {self.query}", f"Result: {self.copy_text}", "Assumptions:",
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
    values: Mapping[str, _Value]
    coordinates: frozenset[Any]
    rhs: tuple[_Value, ...]
    jacobian: _Value
    parameter_definitions: Mapping[str, str]
    let_definitions: Mapping[str, str]
    coordinate_units: Mapping[str, Unit | None]
    authored_definitions: tuple[str, ...]
    # Each rate constant or Keq that is a coordinate: its definition, and the energy it is written through.
    definitions: Mapping[Any, Any] = MappingProxyType({})
    rate_energies: Mapping[Any, Any] = MappingProxyType({})


def _written_through(value: _Value, rates: Mapping[Any, Any]) -> _Value:
    """exp(log(k/prefactor) + y) as k/prefactor*exp(y): an energy written through its rate constant
    leaves that rate constant outside the exponential."""
    sp = require_sympy()
    if not rates or not isinstance(value.expression, sp.Basic):
        return value
    return replace(value, expression=value.expression.replace(
        lambda e: isinstance(e, sp.exp) and any(log.free_symbols & rates.keys() for log in e.args[0].atoms(sp.log)),
        lambda e: sp.expand_power_exp(sp.exp(sp.expand(e.args[0], log=False)))))


def symbolic_calculator_available(mechanism: Any) -> tuple[bool, str]:
    """Cheap structural availability; expression construction is first-query work."""
    metadata = MechanismMetadataView.from_metadata(mechanism.metadata)
    if metadata.temperature_changes:
        return False, "Calculator does not support temperature schedules."
    step = fractional_order_step(mechanism)
    if step is not None:
        return False, located("Calculator does not support fractional reaction orders.", step["line"], step["line_text"])
    namespace = build_namespace_from_mechanism(mechanism)
    spec = mechanism.metadata[MechanismMetadataKeys.PARAMETER_ALGEBRA_SPEC]
    reserved = set(namespace.canonical_names) | set(spec.param_assignment_names()) | set(spec.observable_names)
    reserved |= set(PROTECTED_NAMES) | set(BUILTIN_FUNCTIONS)
    collisions = [name for name in mechanism.species_names() if name in reserved
                  or namespace.resolve(name).canonical_name is not None
                  or namespace.invalid_protected_indexed_identifier_message(name) is not None]
    if collisions:
        return False, f"Species collide with parameter or Algebra names: {', '.join(collisions)}."
    return True, ""


def build_calculator_context(mechanism: Any, *, mechanism_source: str) -> CalculatorContext:
    # The owner checked symbolic_calculator_available before starting this child.
    sp = require_sympy()
    namespace = build_namespace_from_mechanism(mechanism)
    spec = mechanism.metadata[MechanismMetadataKeys.PARAMETER_ALGEBRA_SPEC]
    graph = spec.evaluation_dependencies
    statements = {stmt.name: stmt for stmt in spec.parameter_block.lines}
    names = tuple(mechanism.species_names())
    # Coordinates carry the domains of their scientific roles.
    roles = {item.canonical_name: item.info.role for item in namespace.ordered_items}
    roles.update({item.canonical_name: item.role for item in namespace.source_coordinates})

    def coordinate(name):
        return sp.Symbol(name, positive=True) if roles.get(name) in _POSITIVE_ROLES else sp.Symbol(name, real=True)
    symbols = {name: sp.Symbol(name, nonnegative=True) for name in names}
    symbols.update({name: coordinate(name) for name in (*namespace.canonical_names, *spec.param_assignment_names(), "t")})
    symbols.update({symbol.name: symbol for symbol in _fixed_constant_values()})
    symbols["T"] = sp.Symbol("T", positive=True)
    symbols.update({f"initial({name})": sp.Symbol(f"initial({name})", nonnegative=True) for name in names})
    kinds = {name: spec.parameter_assignment_kind(name) for name in statements}
    independent = {name for name, deps in graph.items() if not deps and kinds.get(name) not in {"mechanism-bound", "derived scalar"}}
    coordinates = {symbol for name, symbol in symbols.items() if name not in graph and name not in _FIXED_CONSTANTS}
    coordinates.update(symbols[name] for name in independent)
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
        for statement in block.lines
    )
    context = CalculatorContext(
        mechanism, mechanism_source, namespace, names, MappingProxyType(symbols), MappingProxyType(values),
        frozenset(coordinates), (), _Value(sp.ImmutableMatrix([])),
        MappingProxyType({name: expression_to_source(stmt.expr) for name, stmt in statements.items()}),
        MappingProxyType({stmt.name: expression_to_source(stmt.expr) for stmt in spec.observable_block.lines}),
        MappingProxyType(quantities),
        authored_definitions,
    )
    chemistry, energies = build_calculator_chemistry_relations(mechanism, parameter_spec=spec, symbols=symbols)
    closures = {}
    for name, dependencies in graph.items():
        closures[name] = set(dependencies).union(*(closures[dep] for dep in dependencies))

    def project(expression, dependencies, factors=()):
        """Install completed dependencies, and the lets of the rate factors,
        before admitting a native projection."""
        selected = tuple(name for name in graph if name in dependencies)
        operands = tuple(values[name] for name in (*selected, *factors))
        failure = _unusable(operands)
        if failure is not None:
            return failure
        try:
            mapping = {symbols[name]: item.expression for name, item in zip(selected, operands)}
            lets = dict(zip(factors, (item.expression for item in operands[len(selected):])))
            return _admit_native(_native_substitute(_expand_factors(expression, lets), mapping))
        except _backend_failures() as exc:
            return _Value(_unavailable(exc))

    def definition(name):
        if kinds.get(name) in {"mechanism-bound", "derived scalar"}:
            value = _translate_node(context, statements[name].expr)
            return replace(value, quantity=quantities.get(name)) if name in spec.mechanism_param_names else value
        if name in chemistry:
            return replace(project(chemistry[name], closures[name]), quantity=quantities.get(name))
        return _Value(symbols[name], quantity=quantities.get(name))

    def prepare(inputs, rates):
        for name in graph:
            try:
                value = inputs[name] if name in inputs else definition(name)
            except _backend_failures() as exc:
                value = _Value(_unavailable(exc))
            values[name] = _written_through(value, rates)
    prepare({}, {})
    # Mechanism algebra works in rate constants. An Arrhenius or Eyring rate constant, or a Keq from
    # dG_eq, whose energy is an independent coordinate is a coordinate itself when no other input of its
    # definition depends on such an energy, so its prefactor is the same in either form. Its energy is
    # then the value -R*T*log(k/prefactor), and every other value is prepared again from these inputs.
    R, T = symbols["R"], symbols["T"]
    candidates = {name: energy for name, energy in energies.items()
                  if energy in coordinates and isinstance(values[name].expression, sp.Expr)}
    energy_names = {str(energy) for energy in candidates.values()}
    rates = {symbols[name]: energy for name, energy in candidates.items()
             if not any(energy_names & closures[dependency] for dependency in graph[name])}
    definitions = {rate: values[str(rate)].expression for rate in rates}
    if rates:
        inputs = {str(rate): replace(values[str(rate)], expression=rate) for rate in rates}
        inputs.update({str(energy): replace(values[str(energy)], expression=-R * T * sp.log(rate / definitions[rate].xreplace({energy: 0})))
                       for rate, energy in rates.items()})
        prepare(inputs, rates)
        coordinates.update(rates)
        context = replace(context, coordinates=frozenset(coordinates), definitions=MappingProxyType(definitions),
                          rate_energies=MappingProxyType(rates))
    for statement in spec.observable_block.statements_in_evaluation_order():
        try:
            values[statement.name] = _written_through(_translate_node(context, statement.expr), rates)
        except _backend_failures() as exc:
            values[statement.name] = _Value(_unavailable(exc))
    projection = build_symbolic_mechanism_public_projection(mechanism, exact_stoichiometric_coefficients=True)
    canonical = {symbol: symbols[str(symbol)] for expression in projection.rhs_expressions for symbol in expression.free_symbols}
    def field(expression):
        expression = expression.xreplace(canonical)
        roots = {name for name in graph if symbols[name] in expression.free_symbols}
        factors = tuple(sorted({item.func.__name__ for item in expression.atoms(sp.core.function.AppliedUndef)}))
        return project(expression, roots.union(*(closures[name] for name in roots)), factors)
    return replace(context, rhs=tuple(replace(field(x), quantity=MOLAR.divide(SECOND)) for x in projection.rhs_expressions),
                   jacobian=replace(field(sp.ImmutableMatrix(projection.jacobian_expressions)), quantity=DIMENSIONLESS.divide(SECOND)))


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
            return replace(value, quantity=node.quantity_unit)
        # Parent argument conversions may be retained on a leaf as well as on
        # an operator. Keep normal native admission.
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
        "Unsupported: the Calculator does not handle comparisons, logical operators, or ifelse."
    )


def _math_call(name: str, args: Sequence[Any]) -> Any:
    sp = require_sympy()
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
        detail = f" It is defined as {definition}." if definition is not None else " It is a derived quantity."
        raise SymbolicCalculatorError(
            f"Invalid input: {operation} requires an independent coordinate, and {symbol} is not one.{detail} "
            "Use independent coordinates from the definition."
        )
    return symbol


def _independent_of_rates(context: CalculatorContext, operation: str, operand: _Value, targets: _Value, *values: _Value) -> None:
    """A rate-constant coordinate and the variables of its definition are not independent: an
    operation varies one while the operand holds the other only after the definitions are written out."""
    sp = require_sympy()
    expression = targets.expression
    # Replacing a rate constant by its own definition writes it out, which is always consistent.
    replacements = expression.replacements if isinstance(expression, _Relationships) else {}
    targets = set(replacements) if replacements else set(expression) if isinstance(expression, sp.Tuple) else {expression}
    present = _current_symbols(operand)
    if operation == "subs":
        # Judge what the substitution leaves: its targets go and the symbols of its values come in.
        present = present - targets | set().union(*(_current_symbols(v) for v in values), *(v.free_symbols for v in replacements.values()))
    pairs = sorted({f"{rate} and {x}" for rate in present & context.definitions.keys()
                    for x in context.definitions[rate].free_symbols & targets}
                   | {f"{rate} and {context.rate_energies[rate]}" for rate in targets & context.rate_energies.keys()
                      if context.rate_energies[rate] in present and replacements.get(rate) != context.definitions[rate]})
    if pairs:
        raise SymbolicCalculatorError(
            f"Invalid input: {operation} cannot treat {', '.join(pairs)} as independent, because each rate or "
            "equilibrium constant is defined through that variable. Use the rate constants alone, or write them out "
            "with subs(F, definitions()) and use the variables of their definitions.")


def _integer(value: _Value, operation: str, *, minimum=0) -> int:
    expression = _scalar(value, operation)
    if isinstance(value.quantity, Unit) and value.quantity.compatible(DIMENSIONLESS):
        expression *= conversion_scale(value.quantity, DIMENSIONLESS).symbolic()
    if expression.is_Integer is not True or expression < minimum:
        raise SymbolicCalculatorError(f"Invalid input: {operation} requires an integer >= {minimum}.")
    return int(expression)


def _general(expression, variable, fixed=frozenset()):
    """A case that holds only where sums, products, powers, exponentials and logarithms of the variable
    are zero holds at isolated values of it, where it has no derivative and no effect on a limit or
    series: keep the formula for its general values. Conditions that involve a fixed symbol stay."""
    sp = require_sympy()
    def isolated(c):
        return isinstance(c, (sp.Eq, sp.Ne)) and variable in c.free_symbols and not c.free_symbols & fixed and all(
            isinstance(node, (sp.Symbol, sp.Add, sp.Mul, sp.Pow, sp.exp, sp.log))
            for node in sp.preorder_traversal(c.lhs - c.rhs) if node.has(variable))
    return expression.replace(lambda p: isinstance(p, sp.Piecewise), lambda p: sp.Piecewise(*(
        (body, condition.replace(isolated, lambda c: sp.false if isinstance(c, sp.Eq) else sp.true)) for body, condition in p.args)))


def _differentiate(context: CalculatorContext, value: _Value, variable: Any | None, order: int) -> _Value:
    """Admit each derivative step, without reentering command orchestration."""
    sp = require_sympy()
    if isinstance(value.expression, _Relationships):
        return _Value(_UnavailableDiagnostic("Unsupported: derivatives need an expression; apply solve relationships with subs first."), value.notices)
    for _ in range(order):
        try:
            expression = (_map(value.expression, lambda x: sp.diff(_general(x, variable), variable)) if variable is not None
                          else _kinetic_expression(context, value.expression))
        except _backend_failures() as exc:
            return _Value(_unavailable(exc), value.notices)
        if expression != value.expression:
            value = _admit_native(expression, value.notices)
        if isinstance(value.expression, _UnavailableDiagnostic):
            break
    return value


def _kinetic_expression(context, expression):
    """partial(F,t) plus the species partials times the accepted right-hand side."""
    sp = require_sympy()
    time = context.symbols["t"]
    species = tuple(context.symbols[name] for name in context.species_names)
    partials = {}
    def differentiate_leaf(leaf):
        partials[leaf] = tuple(sp.diff(_general(leaf, x), x) for x in (time, *species))
        return leaf
    _map(expression, differentiate_leaf)
    required = [i for i in range(len(species)) if any(row[i+1] != 0 for row in partials.values())]
    failure = _unusable([context.rhs[i] for i in required])
    if failure is not None:
        raise SymbolicCalculatorError(failure.expression.explanation)
    return _map(expression, lambda leaf: partials[leaf][0] + sum(
        partials[leaf][i+1] * context.rhs[i].expression for i in required))


def _compose_units(expression, quantity, operation, other):
    """Each entry's quantity divided by ("/") or multiplied by ("*") another unit through the
    shared unit rule, which normalizes the unit and scales the entry to match."""
    sp = require_sympy()
    if isinstance(expression, sp.Tuple):
        composed = [_compose_units(part, quantity[i] if isinstance(quantity, tuple) else None, operation, other)
                    for i, part in enumerate(expression)]
        return sp.Tuple(*(entry for entry, _ in composed)), tuple(unit for _, unit in composed)
    if isinstance(expression, sp.Equality):
        return expression, quantity
    conversion = operation_units(operation, (quantity if isinstance(quantity, Unit) else None, other))
    return expression * conversion.arguments[0].symbolic() * conversion.result.symbolic(), conversion.unit


def _scalar_operation(name, args, *, conversion=None):
    """Consume source preparation, or prepare a new query's scalar operation."""
    arithmetic = name in {"+", "-", "*", "/", "^", "**"}
    for arg in args:
        _scalar(arg, "arithmetic" if arithmetic else name)
    if conversion is None:
        exponent = None
        if name in {"^", "**", "pow"} and len(args) == 2 and args[1].expression.is_Rational:
            exponent = Fraction(int(args[1].expression.p), int(args[1].expression.q))
        conversion = operation_units("ln" if name == "log" else name,
            tuple(arg.quantity if isinstance(arg.quantity, Unit) else None for arg in args),
            zeros=tuple(arg.expression == 0 and isinstance(arg.quantity, Unit) for arg in args), exponent=exponent)
    working = tuple(arg.expression * factor.symbolic() for arg, factor in zip(args, conversion.arguments))
    expression = _native_arithmetic(working[0], name, working[1]) if arithmetic else _math_call(name, working)
    return expression * conversion.result.symbolic(), conversion.unit


def _matrix_cell(value, i, j):
    return replace(value, expression=value.expression[i, j])


def _matrix_arithmetic(name, left, right):
    """Native matrix arithmetic; each matrix carries one unit, or none."""
    sp = require_sympy()
    lm, rm = isinstance(left.expression, sp.MatrixBase), isinstance(right.expression, sp.MatrixBase)
    if name not in {"+", "-", "*"}:
        raise SymbolicCalculatorError("Unsupported: matrix arithmetic supports +, - and *.")
    for operand, is_matrix in ((left, lm), (right, rm)):
        if not is_matrix:
            _scalar(operand, "arithmetic")
    if (name in {"+", "-"} and (not lm or not rm or left.expression.shape != right.expression.shape)
            or name == "*" and lm and rm and left.expression.cols != right.expression.rows):
        raise SymbolicCalculatorError("Invalid input: incompatible arithmetic operands.")
    conversion = operation_units(name, tuple(v.quantity if isinstance(v.quantity, Unit) else None for v in (left, right)))
    lhs, rhs = (v.expression * factor.symbolic() for v, factor in zip((left, right), conversion.arguments))
    return sp.ImmutableMatrix(_native_arithmetic(lhs, name, rhs) * conversion.result.symbolic()), conversion.unit


def _scientific(expression, unit):
    """Express a finished known scalar or matrix quantity in its scientific coordinates."""
    sp = require_sympy()
    # Equations have dimensionless meaning; _equations already made their sides scientific.
    if not isinstance(unit, Unit) or not isinstance(expression, (sp.Expr, sp.MatrixBase)):
        return expression
    return expression * conversion_scale(unit, scientific_unit(unit.dimensions)).symbolic()


def _determinant(value):
    """A determinant carries its matrix's unit to the power of the matrix size."""
    matrix = value.expression
    if not matrix.rows:
        return matrix.det(), DIMENSIONLESS
    conversion = operation_units("^", (value.quantity if isinstance(value.quantity, Unit) else None, DIMENSIONLESS),
                                 exponent=Fraction(matrix.rows))
    return matrix.det(method="berkowitz") * conversion.result.symbolic(), conversion.unit


def _read_number(value, other):
    """A number without units is read in the unit of the other side, as substitution reads it."""
    if (value.expression.is_number and isinstance(value.quantity, Unit) and value.quantity.compatible(DIMENSIONLESS)
            and isinstance(other.quantity, Unit) and not other.expression.is_number):
        return replace(value, expression=value.expression * conversion_scale(value.quantity, DIMENSIONLESS).symbolic(),
                       quantity=other.quantity)
    return value


def _equations(left: _Value, right: _Value):
    sp = require_sympy()
    if isinstance(left.expression, sp.MatrixBase):
        matrix = left.expression
        if right.expression == 0:
            right = replace(right, expression=sp.zeros(*matrix.shape))
        if not isinstance(right.expression, sp.MatrixBase) or right.expression.shape != matrix.shape:
            raise SymbolicCalculatorError("Invalid input: equation matrix shapes must agree.")
        entries = [_equations(_matrix_cell(left, i, j), _matrix_cell(right, i, j))
                   for i in range(matrix.rows) for j in range(matrix.cols)]
        return sp.Tuple(*(expr for expr, _unit in entries)), tuple(unit for _expr, unit in entries)
    if not isinstance(left.expression, sp.Expr) or not isinstance(right.expression, sp.Expr):
        raise SymbolicCalculatorError("Invalid input: eq requires scalar expressions or matching matrices.")
    left, right = _read_number(left, right), _read_number(right, left)
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
    if isinstance(expression, (_Relationships, sp.Tuple)):
        parts = tuple(expression.replacements.values()) if isinstance(expression, _Relationships) else tuple(expression)
        lost = [_lose_units(part, quantity[i] if isinstance(quantity, tuple) else None, unknown)
                for i, part in enumerate(parts)]
        entries = tuple(entry for entry, _ in lost)
        rebuilt = (_Relationships(dict(zip(expression.replacements, entries))) if isinstance(expression, _Relationships)
                   else sp.Tuple(*entries))
        return rebuilt, tuple(unit for _, unit in lost)
    if isinstance(expression, sp.Basic) and expression.free_symbols.intersection(unknown):
        return _scientific(expression, quantity), None
    return expression, quantity


def _substitute_coordinates(context, source, targets, replacements, units):
    converted, unknown = {}, set()
    for target, expression, unit in zip(targets, replacements, units, strict=True):
        expression, known = _coordinate_argument(context, target, _Value(expression, quantity=unit))
        converted[target] = expression
        if not known:
            unknown.add(target)
    expression, quantity = _lose_units(source.expression, source.quantity, unknown)
    return _substitute_value(replace(source, expression=expression, quantity=quantity), converted)


def _linear_solutions(matrix, vector, variables):
    """Domain elimination; each answer entry is cancelled once, in linsolve's form."""
    sp = require_sympy()
    from sympy.polys.matrices import DomainMatrix
    augmented = DomainMatrix.from_Matrix(matrix.row_join(vector), field=True, extension=True)
    if augmented.domain.is_EX:
        # Clearing denominators over general expressions grows with the numbers in exponents.
        reduced, pivots = augmented.to_dense().rref()
        denominator = augmented.domain.one
    else:
        reduced, denominator, pivots = augmented.rref_den(method="CD", keep_domain=False)
    count = len(variables)
    if pivots and pivots[-1] == count:
        return sp.S.EmptySet
    ring, field = reduced.domain, augmented.domain
    denominator = field.convert_from(denominator, ring)
    rows = reduced.to_sdm()
    solution = list(variables)
    for row, pivot in enumerate(pivots):
        entries = rows.get(row, {})

        def value(j, entries=entries):
            return field.convert_from(entries[j], ring) / denominator

        solution[pivot] = sp.Add(*(field.to_sympy(value(j)) if j == count else variables[j] * field.to_sympy(-value(j))
                                   for j in sorted(entries) if j not in pivots))
    return sp.FiniteSet(sp.Tuple(*solution))


def _linear_answers(matrix, vector, unknowns):
    """Exact elimination. Each exponential stands in as a power of its own symbol, so
    exponentials that share T or R keep the coefficients in one rational-function domain."""
    sp = require_sympy()
    entries = [sp.expand_power_exp(e.replace(sp.exp, lambda argument: sp.exp(sp.expand(argument)))) for e in (*matrix, *vector)]
    exponentials = set().union(*(entry.atoms(sp.exp) for entry in entries))
    # Only outermost exponentials stand in; one nested in another's exponent goes with it.
    outer = {e for e in exponentials if not any(e != f and f.args[0].has(e) for f in exponentials)}
    parts = {}
    for e in outer:
        coefficient, rest = e.args[0].as_coeff_Mul()
        parts[e] = (sp.Rational(str(coefficient)) if coefficient.is_Float else coefficient, rest)
    # exp(c*x) is a whole power of exp(g*x), g the greatest common divisor of the multiples c,
    # an approx decimal taken at its exact value.
    unit = {}
    for coefficient, rest in parts.values():
        unit[rest] = sp.gcd(unit.get(rest, 0), coefficient)
    symbols = {rest: sp.Dummy(positive=True) for rest in unit}
    stand_ins = {e: symbols[rest] ** (coefficient / unit[rest]) for e, (coefficient, rest) in parts.items()}

    def prepare(entry):
        entry = entry.xreplace(stand_ins)
        # Decisions on approx decimals use each decimal's exact value.
        return entry.xreplace({f: sp.Rational(str(f)) for f in entry.atoms(sp.Float)})

    rows, cols = matrix.shape
    prepared = [prepare(entry) for entry in entries]
    native = _linear_solutions(sp.Matrix(rows, cols, prepared[:rows * cols]), sp.Matrix(prepared[rows * cols:]), unknowns)
    if native == sp.S.EmptySet:
        return []
    restore = {symbols[rest]: sp.exp(unit[rest] * rest) for rest in unit}
    (solution,) = native
    return [{x: sp.powsimp(value.xreplace(restore), combine="exp") if restore else value
             for x, value in zip(unknowns, solution)}]


def _finite(result, domain):
    """The finite answers of a native solution set; exclusions at exceptional parameter
    values are not answers, and an intersection keeps its answers only when it is with
    the coordinate's whole domain. None when the set is not finite."""
    sp = require_sympy()
    if result == sp.S.EmptySet:
        return []
    if isinstance(result, sp.FiniteSet):
        return list(result)
    if isinstance(result, sp.Union):
        parts = [_finite(part, domain) for part in result.args]
        return None if any(part is None for part in parts) else [x for part in parts for x in part]
    if isinstance(result, sp.Intersection):
        finite = next((part for part in result.args if isinstance(part, sp.FiniteSet)), None)
        whole = finite is not None and all(domain.is_subset(part) for part in result.args if part is not finite)
        return list(finite) if whole else None
    if isinstance(result, sp.Complement):
        return _finite(result.args[0], domain)
    return None


def _exact_decimals(expression):
    """Decimals become their exact values, except inside function arguments, where the
    exact fraction of a long decimal makes SymPy's polynomial work explode."""
    sp = require_sympy()
    shield = {function: sp.Dummy() for function in expression.atoms(sp.Function)}
    exact = expression.xreplace(shield)
    exact = exact.xreplace({f: sp.Rational(str(f)) for f in exact.atoms(sp.Float)})
    return exact.xreplace({dummy: function for function, dummy in shield.items()})


def _solutions(residuals, unknowns):
    """Answers from the SymPy method suited to the kind of problem, and whether that method finds them all.

    Linear systems use exact elimination; one equation in one unknown uses solveset over the
    unknown's domain; polynomial systems with rational coefficients and finitely many solutions
    use exact roots; anything else uses SymPy's general solver, which can miss solutions."""
    sp = require_sympy()
    from sympy.solvers.solveset import NonlinearError
    if not unknowns:
        return ([] if residuals else [{}]), True
    residuals = [_exact_decimals(r) for r in residuals]
    try:
        matrix, vector = sp.linear_eq_to_matrix(residuals, unknowns)
    except NonlinearError:
        pass
    else:
        return _linear_answers(matrix, vector, unknowns), True
    # A factor without unknowns, nonzero for general values, leaves an equation's roots unchanged.
    polynomials = ([sp.Poly(r, *unknowns).primitive()[1].as_expr() for r in residuals]
                   if all(r.is_polynomial(*unknowns) for r in residuals) else [])
    if (polynomials and not set().union(*(r.free_symbols for r in polynomials)) - set(unknowns)
            and all(c.is_Rational for r in polynomials for c in sp.Poly(r, *unknowns).coeffs())):
        basis = sp.groebner(polynomials, *unknowns, order="grevlex")
        if list(basis) == [1]:
            return [], True
        if basis.is_zero_dimensional:
            found = _shape_roots(polynomials, unknowns)
            if found is not None:
                return found, True
    if len(residuals) == len(unknowns) == 1:
        x = unknowns[0]
        found = _exponential_answers(residuals[0], x)
        if found is not None:
            return found, True
        # solveset writes max, min, heaviside, abs and other cases as case formulas and takes every case
        # condition to be on x (Piecewise.as_expr_set_pairs), so such a function of other quantities enters as
        # a coefficient symbol with its sign facts, and a case formula of x with conditions on other quantities
        # is unsupported.
        facts = ("positive", "nonnegative", "real", "nonzero")
        coefficients = {f: sp.Dummy(**{fact: getattr(f, f"is_{fact}") for fact in facts if getattr(f, f"is_{fact}") is not None})
                        for f in residuals[0].atoms(sp.Max, sp.Min, sp.Heaviside, sp.Piecewise, sp.Abs) if not f.has(x)}
        residual = sp.powsimp(residuals[0].xreplace(coefficients), combine="exp")
        if any(c.free_symbols and x not in c.free_symbols for p in residual.atoms(sp.Piecewise) for _, c in p.args):
            raise SymbolicCalculatorError("Unsupported: an equation whose cases depend on other quantities. Substitute values for them first.")
        native = sp.solveset(residual, x, domain=_calculus_domain(x)).xreplace({d: f for f, d in coefficients.items()})
        roots = _finite(native, _calculus_domain(x))
        return ([{x: root} for root in roots] if roots is not None else native), True

    def satisfied(answer):
        # A general solver can drop an equation it cannot solve or leave an unknown unsolved,
        # so each of its answers must make every equation hold for general values.
        return all(isinstance(value, sp.Expr) for value in answer.values()) and all(
            sp.cancel(residual) == 0 or sp.simplify(residual) == 0
            for residual in (r.subs(answer, simultaneous=True) for r in residuals))

    def nonlinsolve():
        native = sp.nonlinsolve(residuals, list(unknowns))
        return [dict(zip(unknowns, values)) for values in native] if isinstance(native, sp.FiniteSet) else []

    def solve():
        return sp.solve(residuals, list(unknowns), dict=True, rational=False)
    # nonlinsolve suits polynomial systems; solve also handles exponentials and logarithms of the unknowns.
    answers = []
    for method in ((nonlinsolve, solve) if all(r.is_polynomial(*unknowns) for r in residuals) else (solve, nonlinsolve)):
        try:
            answers = answers or [answer for answer in method() if satisfied(answer)]
        except (*_backend_failures(), AttributeError):
            pass
    return [{x: sp.expand_log(value) for x, value in answer.items()} for answer in answers], False


def _exponential_answers(residual, x):
    """An equation polynomial in exponentials exp(c*x*s) of x, with rational c and one s free of x,
    is a polynomial in one positive exponential, whose real roots SymPy finds exactly. SymPy
    inverts a two-term equation with logarithms. None for any other equation."""
    sp = require_sympy()
    parts = {}
    for e in residual.atoms(sp.exp):
        if e.has(x):
            independent, dependent = sp.expand(e.args[0]).as_independent(x, as_Add=True)
            coefficient, rest = sp.factor_terms(dependent).as_coeff_Mul()
            parts[e] = (sp.exp(independent), *((-coefficient, -rest) if rest.could_extract_minus_sign() else (coefficient, rest)))
    rests = {rest for _, _, rest in parts.values()}
    if len(rests) != 1 or any(not c.is_Rational for _, c, _ in parts.values()):
        return None
    slope = sp.cancel(rests.pop() / x)
    # exp(c*x*s) is a whole power of u = exp(g*x*s/L), L the common denominator and g the common numerator of the c.
    scale = sp.ilcm(1, *(c.q for _, c, _ in parts.values()))
    step = sp.igcd(0, *(c.p * scale // c.q for _, c, _ in parts.values()))
    u = sp.Dummy(positive=True)
    # Clearing a denominator that is a power of the positive u keeps the roots unchanged.
    polynomial, denominator = sp.fraction(sp.together(residual.xreplace({e: f * u ** (c * scale / step) for e, (f, c, _) in parts.items()})))
    if (slope.has(x) or sp.Tuple(polynomial, denominator).has(x) or not (polynomial.is_polynomial(u) and denominator.is_polynomial(u))
            or not sp.Poly(denominator, u).is_monomial or len(sp.Poly(polynomial, u).terms()) < 3):
        return None
    # SymPy factors the polynomial for its exact roots, which beyond this degree takes too long.
    found = None if sp.Poly(polynomial, u).degree() > 300 else _solutions([polynomial], (u,))[0]
    if found is None or isinstance(found, sp.Set):
        return sp.ConditionSet(x, sp.Eq(residual, 0), _calculus_domain(x))
    return [{x: scale * sp.expand_log(sp.log(answer[u])) / (step * slope)} for answer in found if not _outside(u, answer[u])]


def _shape_roots(residuals, unknowns):
    """Exact real roots when some coordinate's lex Groebner basis is in shape position:
    every other coordinate a polynomial in that one, which a univariate polynomial fixes."""
    sp = require_sympy()
    for last in reversed(unknowns):
        order = (*(x for x in unknowns if x != last), last)
        basis = list(sp.groebner(residuals, *order, order="lex"))
        if len(basis) != len(order) or basis[-1].free_symbols != {last} or any(
                g.free_symbols - {x, last} or sp.degree(g, x) != 1 or sp.Poly(g, x).LC().free_symbols
                for g, x in zip(basis[:-1], order[:-1])):
            continue
        # Each element is linear in its coordinate; the domain is judged with the other answers.
        values = {x: -g.subs(x, 0) / g.diff(x) for g, x in zip(basis[:-1], order[:-1])}
        return [{**{x: v.subs(last, root) for x, v in values.items()}, last: root}
                for root in sp.Poly(basis[-1], last).real_roots()]
    return None


def _outside(coordinate, value):
    """An answer decidably outside its coordinate's domain is not a solution."""
    if coordinate.is_positive:
        return value.is_positive is False
    if coordinate.is_nonnegative:
        return value.is_nonnegative is False
    return value.is_real is False


def _decimals(value, digits):
    """Answers from decimal equations continue in the stated digits, without rounding residue."""
    sp = require_sympy()
    return sp.nfloat(value, digits).replace(lambda n: n.is_Add and n.is_number, lambda n: n.evalf(digits))


def _solve(context: CalculatorContext, problem: _Value, coordinates: _Value):
    """Solve the supplied equations once for the distinct requested coordinates."""
    sp = require_sympy()
    from sympy.core.evalf import prec_to_dps
    labels = coordinates.expression if isinstance(coordinates.expression, sp.Tuple) else (coordinates.expression,)
    if not labels or len(set(labels)) != len(labels):
        raise SymbolicCalculatorError("Invalid input: solve requires distinct coordinates.")
    for symbol in labels:
        _coordinate(context, symbol, "solve")
    equations = problem.expression if isinstance(problem.expression, sp.Tuple) else (problem.expression,)
    if not equations or any(not isinstance(item, sp.Equality) for item in equations):
        raise SymbolicCalculatorError("Invalid input: solve requires equations made with eq.")
    # Kindred's R is exactly k_B*N_A; N_A, absent from the rate laws, carries the identity.
    identity = {context.symbols["N_A"]: context.symbols["R"] / context.symbols["k_B"]}
    residuals = [r for r in ((eq.lhs - eq.rhs).xreplace(identity) for eq in equations) if r != 0]
    floats = set().union(*(r.atoms(sp.Float) for r in residuals))
    digits = min((prec_to_dps(f._prec) for f in floats), default=None)
    # A coordinate that no equation mentions is free.
    unknowns = tuple(x for x in labels if any(x in r.free_symbols for r in residuals))
    # Answers are generic: a case that holds only at isolated values of another quantity is set aside,
    # while a condition involving a requested coordinate stays, since it can hold at an answer.
    if any(r.has(sp.Piecewise) for r in residuals):
        for s in set().union(*(r.free_symbols for r in residuals)) - set(unknowns):
            residuals = [_general(r, s, frozenset(unknowns)) for r in residuals]
    try:
        found, complete = _solutions(residuals, unknowns)
    except SymbolicCalculatorError:
        raise
    except _backend_failures() as exc:
        return _Value(_UnavailableDiagnostic("Calculator could not solve these equations.", str(exc)))
    if isinstance(found, sp.Set):
        if found.has(sp.ConditionSet):
            return _Value(_UnavailableDiagnostic("Calculator could not solve this equation."))
        # A requested coordinate that the equation does not mention ranges over its domain.
        return _Value(found if len(labels) == 1 else sp.ProductSet(*(found if x in unknowns else _calculus_domain(x) for x in labels)))
    answers = []
    for answer in found:
        # A coordinate the answer leaves undetermined is free.
        values = tuple(answer.get(x, x) for x in labels)
        if floats:
            values = tuple(_decimals(v, digits) for v in values)
        # Answers are generic: concentrations are taken as nonzero when judging the domain.
        general = {s: sp.Symbol(s.name, positive=True) for v in values for s in v.free_symbols
                   if s.is_nonnegative and not s.is_positive}
        if not any(_outside(x, sp.together(v.xreplace(general))) for x, v in zip(labels, values)):
            answers.append(values)
    quantity = tuple(context.coordinate_units.get(str(x)) for x in labels) if _known_quantity(problem.quantity) else None
    notices = () if complete else (_INCOMPLETE,)
    results = [replace(_admit_native(_Relationships(dict(zip(labels, values))), notices), quantity=quantity)
               for values in sorted(dict.fromkeys(answers), key=lambda values: sp.default_sort_key(sp.Tuple(*values)))]
    if not results and not complete:
        return _Value(_UnavailableDiagnostic(
            "Calculator could not find solutions in the coordinate domains. SymPy's general solver can miss solutions, so this does not prove that none exist."))
    return results[0] if len(results) == 1 else _Alternatives(tuple(results))


def _evaluate_call(context: CalculatorContext, name: str, args: Sequence[_Value | _Alternatives], *, direction=None, conversion=None):
    """Lift one family of answers through a command; two families need an explicit selection."""
    args = tuple(args)
    if name == "solution":
        if len(args) != 2:
            raise SymbolicCalculatorError("Invalid input: solution requires 2 arguments.")
        if isinstance(args[1], _Alternatives):
            raise SymbolicCalculatorError("Only one operand can hold alternative answers; select an answer from the others with solution(result,n).")
        return _select(args[0], args[1])
    family = any(isinstance(v, _Alternatives) for v in args)
    answers = []
    for values in _branches(args):
        result = _dispatch_call(context, name, values, direction=direction, conversion=conversion)
        family |= isinstance(result, _Alternatives)
        answers.extend(_metadata(item, *values) for item in _entries(result))
    # One answer is an ordinary value, also when a family's other answers dropped out.
    return _Alternatives(tuple(answers)) if family and len(answers) != 1 else answers[0]


def _select(result, index):
    """Usable answer n, counting from 1; selection performs no mathematics."""
    if isinstance(index.expression, _UnavailableDiagnostic):
        return index
    if isinstance(result, _Value) and isinstance(result.expression, _UnavailableDiagnostic):
        return result
    usable = tuple(v for v in _entries(result) if not isinstance(v.expression, _UnavailableDiagnostic))
    offset = _integer(index, "solution", minimum=1) - 1
    if offset >= len(usable):
        raise SymbolicCalculatorError("Invalid input: no solution with this number. Unavailable entries cannot be selected.")
    return usable[offset]


def _dispatch_call(context: CalculatorContext, name: str, args: Sequence[_Value], *, direction=None, conversion=None):
    """Execute one branch; each operation owns its failures."""
    try:
        sp = require_sympy()
        def count(*counts):
            if len(args) not in counts:
                raise SymbolicCalculatorError(f"Invalid input: {name} requires {' or '.join(map(str, counts))} arguments.")
        failure = _unusable(args)
        if failure is not None:
            return failure
        if name in {"solve", "subs", "partial", "collect", "integrate", "limit", "series"} and len(args) > 1:
            _independent_of_rates(context, name, *args)
        structural = name in {"tuple", "matrix", "eq"}
        if name == "subs":
            count(2, 3)
            if len(args) == 2:
                source, relationships = args
                if not isinstance(relationships.expression, _Relationships):
                    raise SymbolicCalculatorError("Invalid input: subs(F, solutions) requires the relationships of a solve, steady_state or stationary_state result.")
                mapping = relationships.expression.replacements
                units = relationships.quantity if isinstance(relationships.quantity, tuple) else (None,) * len(mapping)
                return _substitute_coordinates(context, source, tuple(mapping), tuple(mapping.values()), units)
            targets = args[1].expression if isinstance(args[1].expression, sp.Tuple) else (args[1].expression,)
            values = args[2].expression if isinstance(args[2].expression, sp.Tuple) else (args[2].expression,)
            if any(not isinstance(value, sp.Expr) for value in values):
                raise SymbolicCalculatorError("Invalid input: subs(F, x, value) requires expressions as values; apply solve relationships with subs(F, solutions).")
            if len(targets) != len(values) or len(set(targets)) != len(targets):
                raise SymbolicCalculatorError("Invalid input: substitution rosters must match and targets must be unique.")
            targets = tuple(_coordinate(context, target, name) for target in targets)
            units = args[2].quantity if isinstance(args[2].expression, sp.Tuple) and isinstance(args[2].quantity, tuple) else (args[2].quantity,) * len(values)
            return _substitute_coordinates(context, args[0], targets, values, units)
        if name == "solve":
            count(2)
            return _solve(context, *args)
        if name in {"integrate", "limit", "series"}:
            count(*({"integrate": (2, 4), "limit": (3,), "series": (4,)}[name]))
            return _calculus_call(context, name, args, direction=direction)
        if name in {"dt", "partial"}:
            count(*((1, 2) if name == "dt" else (2, 3)))
            variable = None if name == "dt" else _coordinate(context, args[1].expression, name)
            order = _integer(args[-1], name) if len(args) == (2 if name == "dt" else 3) else 1
            coordinate = SECOND if variable is None else context.coordinate_units.get(str(variable))
            result = _differentiate(context, args[0], variable, order)
            if isinstance(result.expression, _UnavailableDiagnostic):
                return result
            expression, quantity = _compose_units(result.expression, args[0].quantity, "/",
                                                  None if coordinate is None else coordinate.power(Fraction(order)))
            return replace(result, expression=expression, quantity=quantity)
        if name == "approx":
            count(1, 2)
            digits = _integer(args[1], name, minimum=1) if len(args) == 2 else 15
            values = _fixed_constant_values()

            def approximate(x):
                # Series keep their order terms; an integral or root sum that is a number is evaluated,
                # and one that SymPy cannot evaluate to the digits (a divergent integral) fails.
                orders = {order: sp.Dummy() for order in x.atoms(sp.Order)}
                x = _native_substitute(x.xreplace(orders), values).replace(
                    lambda e: isinstance(e, (sp.Integral, sp.RootSum)) and not e.free_symbols,
                    lambda e: (sp.Integral(*e.args) if isinstance(e, sp.Integral) else e).evalf(digits, strict=True))
                if x.is_number and x.has(sp.I):
                    # Products of sums, such as logarithms taken on either side of a root, are distributed
                    # so that imaginary parts that cancel do so exactly before evaluation.
                    x = sp.expand_mul(x)
                # A delta keeps its whole-number derivative order.
                deltas = {d: sp.Dummy() for d in x.atoms(sp.DiracDelta)}
                x = sp.nfloat(x.xreplace(deltas), digits).xreplace(
                    {dummy: sp.DiracDelta(sp.nfloat(d.args[0], digits), *d.args[1:]) for d, dummy in deltas.items()})
                # A part that SymPy could not resolve to any digit has precision 1 and is not significant.
                x = x.xreplace({f: sp.S.Zero for f in x.atoms(sp.Float) if f._prec == 1})
                return x.xreplace({dummy: order for order, dummy in orders.items()})
            expression = _map(args[0].expression, approximate)
            return replace(_admit_native(expression, (f"Approximation: {digits} significant decimal digits.",)),
                           quantity=args[0].quantity)
        if name in {"steady_state", "stationary_state"}:
            species = tuple(context.symbols[n] for n in context.species_names)
            if name == "steady_state":
                if not args:
                    raise SymbolicCalculatorError("Supply species: steady_state(Int) or steady_state([X,Y]). Use stationary_state() for the whole mechanism.")
                count(1, 2)
                selected = args[0].expression if isinstance(args[0].expression, sp.Tuple) else (args[0].expression,)
                if not selected or len(set(selected)) != len(selected) or any(x not in species for x in selected):
                    raise SymbolicCalculatorError("Invalid input: steady_state requires distinct accepted species, not equations or parameter names.")
                extra_args = args[1:]
            else:
                count(0, 1)
                selected = species
                extra_args = args
            rates = tuple(context.rhs[context.species_names.index(str(x))] for x in selected)
            failure = _unusable(rates)
            if failure is not None:
                return failure
            equations = tuple(sp.Eq(rate.expression, 0, evaluate=False) for rate in rates)
            meanings = tuple(DIMENSIONLESS if _known_quantity(rate.quantity) else None for rate in rates)
            if extra_args:
                extra, extra_units = extra_args[0].expression, extra_args[0].quantity
                equations += tuple(extra) if isinstance(extra, sp.Tuple) else (extra,)
                meanings += extra_units if isinstance(extra_units, tuple) else (extra_units,)
            return _solve(context, _Value(sp.Tuple(*equations), quantity=meanings), _Value(sp.Tuple(*selected)))
        if name == "definitions":
            count(0)
            if not context.definitions:
                raise SymbolicCalculatorError(
                    "Invalid input: definitions() requires a rate constant or Keq whose energy is an independent parameter "
                    "and whose other inputs do not depend on any such energy.")
            return _Value(_Relationships(dict(context.definitions)),
                          quantity=tuple(context.coordinate_units.get(str(rate)) for rate in context.definitions))
        if name in {"odes", "jacobian"}:
            count(0)
            if name == "jacobian":
                return context.jacobian
            failure = _unusable(context.rhs)
            return failure or _Value(sp.ImmutableMatrix([rate.expression for rate in context.rhs]), quantity=MOLAR.divide(SECOND))
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
                if any(not isinstance(arg.expression, sp.Basic) for arg in args):
                    raise SymbolicCalculatorError("Unsupported: tuples hold expressions, equations, tuples or matrices.")
                quantity = tuple(arg.quantity for arg in args)
                return sp.Tuple(*(arg.expression for arg in args))
            if name == "matrix":
                count(1)
                rows = args[0].expression
                if isinstance(rows, sp.Tuple):
                    # A one-entry row such as [A] reads as the species itself.
                    rows = sp.Tuple(*(row if isinstance(row, sp.Tuple) else sp.Tuple(row) for row in rows))
                if not isinstance(rows, sp.Tuple) or not rows or any(not row for row in rows):
                    raise SymbolicCalculatorError("Invalid input: matrix requires nonempty rows, for example matrix([[1,2],[3,4]]).")
                if len({len(row) for row in rows}) != 1 or any(not isinstance(x, sp.Expr) for row in rows for x in row):
                    raise SymbolicCalculatorError("Invalid input: matrix rows require equally many scalar entries.")
                row_units = args[0].quantity if isinstance(args[0].quantity, tuple) else (None,) * len(rows)
                cells = tuple(x for row in rows for x in row)
                units = tuple(unit if isinstance(unit, Unit) else None for row in row_units
                              for unit in (row if isinstance(row, tuple) else (row,)))
                # Entries share their compatible unit; otherwise the matrix is in scientific coordinates.
                conversion_ = operation_units("+", units, zeros=tuple(x == 0 and isinstance(u, Unit) for x, u in zip(cells, units)))
                quantity = conversion_.unit
                return sp.ImmutableMatrix(len(rows), len(rows[0]),
                                          [x * factor.symbolic() for x, factor in zip(cells, conversion_.arguments)])
            if name == "eq":
                count(2)
                expression, quantity = _equations(*args)
                return expression
            if name == "at":
                count(2, 3)
                expression = args[0].expression
                indices = tuple(_integer(arg, name) for arg in args[1:])
                if len(indices) == 2 and isinstance(expression, sp.MatrixBase):
                    if indices[0] >= expression.rows or indices[1] >= expression.cols:
                        raise SymbolicCalculatorError(f"Invalid input: at index ({indices[0]}, {indices[1]}) is out of range. Indices start at 0, and this matrix has {expression.rows} rows and {expression.cols} columns.")
                    quantity = args[0].quantity
                    return expression[indices]
                if len(indices) == 1 and isinstance(expression, (sp.Tuple, sp.MatrixBase)):
                    if indices[0] >= len(expression):
                        raise SymbolicCalculatorError(f"Invalid input: at index {indices[0]} is out of range. Indices start at 0, and this {'matrix' if isinstance(expression, sp.MatrixBase) else 'tuple'} has {len(expression)} entries.")
                    units = args[0].quantity
                    quantity = units[indices[0]] if isinstance(units, tuple) else units
                    return expression[indices[0]]
                raise SymbolicCalculatorError("Invalid input: at takes a tuple or matrix with one index, or a matrix with two indices.")
            if name in {"transpose", "det"}:
                count(1)
                matrix = args[0].expression
                if not isinstance(matrix, sp.MatrixBase):
                    raise SymbolicCalculatorError(f"Invalid input: {name} requires a matrix.")
                if name == "det" and matrix.rows != matrix.cols:
                    raise SymbolicCalculatorError("Invalid input: det requires a square matrix.")
                if name == "transpose":
                    quantity = args[0].quantity
                    return matrix.T
                expression, quantity = _determinant(args[0])
                return expression
            if name == "remove_order":
                count(1)
                quantity = args[0].quantity
                def remove(x):
                    result = x.removeO()
                    if result.has(sp.Order):
                        raise SymbolicCalculatorError("Unsupported: remove_order cannot remove an order term inside another expression. Apply remove_order to each series first.")
                    return result
                return _map(args[0].expression, remove)
            raise SymbolicCalculatorError(f"Unsupported calculator operation {name!r}.")
        expression = native()
        notices = ("Series order term explicitly removed. The result is truncated.",) if name == "remove_order" and expression != args[0].expression else ()
        if args and expression == args[0].expression:
            return replace(args[0], quantity=quantity, notices=(*args[0].notices, *notices))
        if structural or name in {"at", "transpose", "initial"}:
            return _Value(expression, notices, quantity=quantity)
        return replace(_admit_native(expression, notices), quantity=quantity)
    except _backend_failures() as exc:
        return _Value(_unavailable(exc))


def _log_conversion_unreliable(term, variable):
    """SymPy 1.14.0 writes the integral of each rational term in real form and can drop logarithms when
    the term has parameters and its denominator has an irreducible factor of degree 2 or more (sympy/sympy#28596)."""
    sp = require_sympy()
    term = _exact_decimals(term.as_independent(variable)[1])
    if not term.is_rational_function(variable) or term.is_polynomial(variable) or not term.free_symbols - {variable}:
        return False
    # Functions in the coefficients are free of the variable; as plain symbols they keep the factoring small.
    term = term.xreplace({f: sp.Dummy() for f in term.atoms(sp.Function)})
    _, factors = sp.factor_list(sp.denom(sp.cancel(term)), variable)
    return any(sp.degree(factor, variable) >= 2 for factor, _ in factors)


def _calculus_call(context: CalculatorContext, name: str, args: Sequence[_Value], *, direction=None) -> _Value:
    """Native integration, limits and series on finished values."""
    sp = require_sympy()
    from sympy.integrals.rationaltools import ratint
    expression = _scalar(args[0], name)
    variable = _coordinate(context, args[1].expression, name)
    quantity = args[0].quantity
    args = list(args)
    positions = range(2, 4) if name == "integrate" and len(args) == 4 else (2,) if name in {"limit", "series"} else ()
    for index in positions:
        point, known = _coordinate_argument(context, variable, args[index])
        args[index] = replace(args[index], expression=point)
        if not known:
            # An unknown point or bound leaves the result in the scientific
            # coordinates of its known finished quantity.
            expression, quantity = _scientific(expression, quantity), None
    notices = ()
    if name == "integrate":
        terms = sp.Add.make_args(expression)
        unreliable = [_log_conversion_unreliable(term, variable) for term in terms]
        if len(args) == 2:
            # Complex logarithms keep every term where the real form would lose some.
            result = sp.Add(*(ratint(term, variable, real=False) if flagged else sp.integrate(term, variable)
                              for term, flagged in zip(terms, unreliable))) if any(unreliable) else sp.integrate(expression, variable)
        else:
            if any(unreliable):
                raise SymbolicCalculatorError(
                    "Unsupported: a definite integral of a rational term with parameters whose denominator has a quadratic "
                    "or higher factor (sympy/sympy#28596). Substitute values for the parameters first, or apply approx "
                    "when physical constants such as R remain.")
            lower, upper = (_scalar(arg, name) for arg in args[2:])
            result = sp.integrate(expression, (variable, lower, upper), conds="piecewise")
        result, quantity = _compose_units(result, quantity, "*", context.coordinate_units.get(str(variable)))
    elif name == "limit":
        point = _scalar(args[2], name)
        expression = _general(expression, variable)
        # SymPy takes the case that holds at the point itself, not beside it.
        if any(variable in condition.free_symbols for case in expression.atoms(sp.Piecewise) for _, condition in case.args):
            raise SymbolicCalculatorError(
                f"Unsupported: a limit of a formula whose cases depend on {variable}. Substitute values or take the limit of one case.")
        result = sp.Limit(expression, variable, point, dir=("-" if direction == "left" else "+") if direction else "+-").doit()
    else:
        point = _scalar(args[2], name, finite=True)
        order = _integer(args[3], name, minimum=1)
        direction = direction or "right"
        result = sp.series(_general(expression, variable), variable, point, order, dir="-" if direction == "left" else "+")
        notices = (f"Local series about {_render(variable)}={_render(point)}, direction {direction}.",)
    return replace(_admit_native(result, notices), quantity=quantity)


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    text: str


def _validate_numeric_literal(literal: str) -> None:
    try:
        exact = Decimal(literal.replace("_", ""))
        value = float(literal.replace("_", ""))
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise SymbolicCalculatorError(f"Invalid input: {literal} is not a number.") from exc
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
                raise SymbolicCalculatorError(f"Invalid input: {char} cannot be used in a query.")
        return [*tokens, _Token("EOF", "")]

    def at(self, kind):
        return self.tokens[self.index].kind == kind

    def take(self, kind=None):
        token = self.tokens[self.index]
        if kind is not None and token.kind != kind:
            expected = "a value" if kind == "IDENT" else "the end of the query" if kind == "EOF" else f"'{kind}'"
            got = "the end of the query" if token.kind == "EOF" else f"'{token.text}'"
            raise SymbolicCalculatorError(f"Invalid input: expected {expected}, got {got}.")
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
                f"{name}: {authored} is defined as {definition}. Use independent coordinates from its definition."
            )
        # A named energy reads as its value through its rate constant; as a target it is the energy itself.
        energies = {self.context.values[str(energy)].expression: energy for energy in self.context.rate_energies.values()}
        if energies and isinstance(args[target], _Value) and isinstance(args[target].expression, require_sympy().Basic):
            args[target] = replace(args[target], expression=args[target].expression.xreplace(energies))

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
            denominator = self.take() if self.at("IDENT") else ""
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
                raise SymbolicCalculatorError(f"Unsupported: {name} is not a Calculator operation.")
            self.take()
            args, spans = [], []
            direction = None
            if not self.at(")"):
                while True:
                    if name == "limit" and len(args) == 3 or name == "series" and len(args) == 4:
                        direction = self.take() if self.at("IDENT") else ""
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
            if expr.args[1].is_zero:
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


def _admit_native(expression, notices=()):
    """A finished result that is nan or complex infinity has no defined value."""
    sp = require_sympy()
    parts = tuple(expression.replacements.values()) if isinstance(expression, _Relationships) else (expression,)
    if any(isinstance(part, sp.Basic) and part.has(sp.nan, sp.zoo) for part in parts):
        return _Value(_UnavailableDiagnostic("Undefined: this expression has no defined value."), notices)
    return _Value(expression, notices)


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
    # The owner strips queries and drops empty ones.
    try:
        value = _CalculatorExpressionParser(context, query).parse()
        return _package_result(context, query, value)
    except SymbolicCalculatorError:
        raise
    except (*_backend_failures(), AttributeError, RecursionError) as exc:
        return _package_result(context, query, _Value(_unavailable(exc)))


def _package_result(context: CalculatorContext, query: str, result) -> SymbolicCalculatorResult:
    sp = require_sympy()
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
            if payload.detail:
                lines.append("Detail: " + " ".join(payload.detail.split()))
        else:
            label = _render(payload)
            index = len(choices) + 1
            if family:
                append(f"Solution {index}")
            heading = ("Relationships:" if isinstance(payload, _Relationships) else "Solution set:" if isinstance(payload, sp.Set)
                       else "Expression:" if family else "")
            append((heading + " " if heading else "") + label,
                   (heading + "\n" if heading else "") + _pretty_render(payload))
            reusable = f"solution({query}, {index})" if family else query
            choices.append((f"Solution {index}: {label}" if family else "Result", reusable))
            if value.quantity is not None:
                unit_text = _quantity_text(value.quantity)
                append(
                    "Units: " + unit_text,
                    display_html='Units: <span style="white-space: nowrap">'
                    + _quantity_html(value.quantity) + "</span>",
                )
            if (value.quantity is not None or family or isinstance(payload, (_Relationships, sp.Set)) or value.notices
                    or not _printed_input_supported(context, payload, label)):
                append("Reuse query: " + reusable)
        for note in value.notices:
            append(note)
        if family:
            append("")
    if not entries:
        append("No solutions in the coordinate domains.")
    elif family:
        append("Complete query (all answers): " + query)
    if query == "odes()":
        append("Rows: " + ", ".join(f"d{name}/dt" for name in context.species_names))
    copy_text = "\n".join(lines).rstrip()
    settings = MechanismMetadataView.from_metadata(context.mechanism.metadata)
    assumptions = (
        "Accepted-source symbolic system, with no schedules, execution overrides, fit candidates, or slider drafts.",
        f"Plain-number energies in the source are in {settings.energy_unit}, and the standard concentration C0 is {settings.standard_conc_M:g} M.",
        "Concentrations are nonnegative. Rate constants, equilibrium constants, prefactors, kappa, degeneracy, T and the "
        "physical constants are positive. Energies, time and other parameters are real. T is fixed in time unless explicit "
        "expression dependence is supplied.",
        "Definitions are prepared before calculation, except that an Arrhenius or Eyring rate constant, or a Keq from dG_eq, "
        "whose energy is an independent parameter, and whose other inputs do not depend on any such energy, is a symbol and "
        "its energy is written through it; subs(F, definitions()) "
        "writes such rate constants out. Independent parameters and initial concentrations are fixed in time, and partial "
        "differentiates prepared mathematics.",
        "Independent inputs remain symbolic. Supply concrete values explicitly with subs. Accepted defaults are not Calculator inputs.",
        "Physical constants keep their names (solve writes N_A as R/k_B) and the values Kindred uses. Approximation does not add precision beyond the source.",
        "Answers hold for general values in these domains. At special values, such as a zero rate constant, substitute the values before solving or integrating, and use a limit for a derivative there.",
        "Special functions, solution sets, and some root or order forms may not be accepted back as typed input. Nested operations still work.",
        "Calculations use SymPy 1.14.0. Integrals of rational terms with parameters whose denominator has a quadratic or higher factor use complex logarithms when indefinite and are unsupported when definite (sympy/sympy#28596).",
        "Algebra and calculus follow the expression as written. A simplified result does not show where the original expression is defined, or that an integral with parameters is well behaved.",
    )
    return SymbolicCalculatorResult(
        query=query,
        result_html="<pre>" + "\n".join(display_html_lines).rstrip() + "</pre>",
        copy_text=copy_text,
        assumptions=assumptions,
        symbol_legend={"Species": context.species_names, "Parameters": tuple(sorted(context.parameter_namespace.canonical_names | context.parameter_definitions.keys())),
         "Fixed constants": _FIXED_CONSTANTS, "Time and temperature": ("t [s]", "T [K]"),
         "Energy coordinates": ("Ea, dG_act, dG_eq in J/mol",),
         "Coordinate units": tuple(f"{name} [{unit.symbol}]" for name, unit in dict(context.mechanism.metadata[MechanismMetadataKeys.PARAMETER_ALGEBRA_SPEC].quantity_units).items() if unit is not None and name in context.symbols),
         "Initial coordinates": tuple(f"initial({name})" for name in context.species_names)},
        mechanism_source=context.mechanism_source,
        reuse_queries=tuple(choices),
        failed=bool(entries) and all(isinstance(v.expression, _UnavailableDiagnostic) for v in entries),
    )
