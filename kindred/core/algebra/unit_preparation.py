"""Unit arithmetic shared by admitted Algebra and Calculator operations."""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from collections.abc import Callable

from kindred.core.quantity_units import (
    Unit, UnitScale, ONE_SCALE, DIMENSIONLESS, MOLAR, scientific_unit,
    conversion_scale,
)
from .parser import ExprNode, NumberNode, IdentNode, SpeciesRefNode, UnaryNode, BinaryNode, CallNode
from .symbols import BUILTIN_FUNCTIONS
from kindred.core._bounded_rational import literal_fraction, rational_binary


@dataclass(frozen=True)
class UnitOperation:
    unit: Unit | None
    arguments: tuple[UnitScale, ...]
    result: UnitScale = ONE_SCALE


def operation_units(
    operation: str,
    units: tuple[Unit | None, ...],
    *,
    zeros: tuple[bool, ...] = (),
    exponent: Fraction | None = None,
) -> UnitOperation:
    """Decide conversions without reading numerical values or defaults."""
    try:
        result = _operation_units(operation, units, zeros=zeros, exponent=exponent)
        for factor in (*result.arguments, result.result):
            factor.check_exact_work()
            factor.number()
    except ValueError:
        # Unit inference cannot make an otherwise retained numerical operation
        # a source-admission error. An inferred factor that cannot be represented
        # by numerical execution leaves the operation's units unspecified.
        return UnitOperation(None, (ONE_SCALE,) * len(units))
    return result


def _operation_units(operation, units, *, zeros, exponent):
    factors = (ONE_SCALE,) * len(units)
    unknown = UnitOperation(None, factors)
    zeros = zeros or (False,) * len(units)
    rule = BUILTIN_FUNCTIONS[operation].unit_rule if operation in BUILTIN_FUNCTIONS else operation
    if not units:
        return unknown
    if rule in {"+", "-", "compatible", "branches", "==", "!=", "<", "<=", ">", ">="}:
        indices = tuple(range(1 if rule == "branches" else 0, len(units)))
        nonzero = [units[i] for i in indices if not zeros[i]]
        if any(unit is None for unit in nonzero):
            return unknown
        base = nonzero[0] if nonzero else DIMENSIONLESS
        if any(not base.compatible(unit) for unit in nonzero):
            return unknown
        result = scientific_unit(base.dimensions)
        factors = tuple(conversion_scale(unit, result) if i in indices and not zeros[i] else ONE_SCALE
                        for i, unit in enumerate(units))
        return UnitOperation(DIMENSIONLESS if rule in {"==", "!=", "<", "<=", ">", ">="} else result, factors)
    if rule in {"^", "**", "power", "sqrt"}:
        if rule == "sqrt":
            exponent = Fraction(1, 2)
        else:
            if len(units) != 2 or units[1] is None or not units[1].compatible(DIMENSIONLESS):
                return unknown
            factors = (ONE_SCALE, units[1].scale)
            exponent_scale = units[1].scale.rational()
            exponent = exponent * exponent_scale if exponent is not None and exponent_scale is not None else None
        # The exponent's numerical meaning is independent of whether the base
        # has established units. Preserve its conversion on the fallback path.
        if units[0] is None:
            return UnitOperation(None, factors)
        if exponent is None:
            return UnitOperation(DIMENSIONLESS, (units[0].scale, *factors[1:])) if units[0].compatible(DIMENSIONLESS) else UnitOperation(None, factors)
        raw = units[0].power(exponent)
    elif any(unit is None for unit in units):
        return unknown
    elif rule in {"*", "/"}:
        raw = units[0].multiply(units[1]) if rule == "*" else units[0].divide(units[1])
    elif rule in {"preserve", "unary"}:
        raw = units[0]
    elif rule in {"dimensionless", "!", "&&", "||"}:
        return UnitOperation(DIMENSIONLESS, tuple(unit.scale for unit in units)) if all(unit.compatible(DIMENSIONLESS) for unit in units) else unknown
    elif rule == "predicate":
        return UnitOperation(DIMENSIONLESS, factors)
    else:
        return unknown
    result = scientific_unit(raw.dimensions)
    return UnitOperation(result, factors, conversion_scale(raw, result))


_RATIONAL_OPERATIONS = {"+", "-", "*", "/", "^", "**"}
_ZERO_RULES = {"+", "-", "compatible", "branches", "==", "!=", "<", "<=", ">", ">="}


def _children(node):
    if isinstance(node, UnaryNode):
        return (node.rhs,)
    if isinstance(node, BinaryNode):
        return node.lhs, node.rhs
    if isinstance(node, CallNode):
        return node.args
    raise TypeError(f"Unsupported admitted expression node: {type(node).__name__}")


def _with_children(node, children):
    if isinstance(node, UnaryNode):
        return replace(node, rhs=children[0])
    if isinstance(node, BinaryNode):
        return replace(node, lhs=children[0], rhs=children[1])
    return replace(node, args=tuple(children))


def _constant(node, constants):
    if any(value is None for value in constants):
        return None
    try:
        if isinstance(node, UnaryNode) and node.op in {"+", "-"}:
            return constants[0] if node.op == "+" else -constants[0]
        if isinstance(node, BinaryNode) and node.op in _RATIONAL_OPERATIONS:
            return rational_binary(node.op, *constants)
    except (ArithmeticError, ValueError):
        return None
    return None


def _finish(node, children, constants):
    operation = node.name if isinstance(node, CallNode) else (
        "unary" if isinstance(node, UnaryNode) and node.op != "!" else node.op)
    rule = BUILTIN_FUNCTIONS[operation].unit_rule if operation in BUILTIN_FUNCTIONS else operation
    result = operation_units(
        operation, tuple(child.quantity_unit for child in children),
        zeros=tuple(value == 0 for value in constants) if rule in _ZERO_RULES else (),
        exponent=constants[1] if operation in {"^", "**", "pow"} and len(children) == 2 else None,
    )
    try:
        scales = tuple(child.unit_scale * factor for child, factor in zip(children, result.arguments))
        for scale in scales:
            scale.check_exact_work()
        scaled = tuple(replace(child, unit_scale=scale, unit_factor=scale.number())
                       for child, scale in zip(children, scales))
    except ValueError:
        result = UnitOperation(None, (ONE_SCALE,) * len(children))
        scaled = children
    return replace(_with_children(node, scaled), quantity_unit=result.unit,
                   unit_scale=result.result, unit_factor=result.result.number())


def _prepare(node, lookup, *, constant_needed=False, authored=False):
    """Prepare each subtree once, carrying only locally needed constant facts."""
    if isinstance(node, NumberNode):
        constant = None
        if constant_needed:
            try:
                constant = literal_fraction(node.literal if node.literal is not None else str(node.value))
            except ValueError:
                pass
        return replace(node, quantity_unit=DIMENSIONLESS), constant
    if isinstance(node, IdentNode):
        return replace(node, quantity_unit=lookup(node.name)), None
    if isinstance(node, SpeciesRefNode):
        return replace(node, quantity_unit=MOLAR), None
    children = _children(node)
    operation = node.name if isinstance(node, CallNode) else node.op
    rule = BUILTIN_FUNCTIONS[operation].unit_rule if operation in BUILTIN_FUNCTIONS else operation
    propagate = constant_needed and isinstance(node, (UnaryNode, BinaryNode)) and operation in _RATIONAL_OPERATIONS
    pairs = tuple(_prepare(child, lookup,
                           constant_needed=(propagate or rule in _ZERO_RULES and (rule != "branches" or i > 0)
                                            or operation in {"^", "**", "pow"} and i == 1),
                           authored=authored)
                  for i, child in enumerate(children))
    prepared_children, constants = tuple(zip(*pairs)) if pairs else ((), ())
    prepared = _finish(node, prepared_children, constants)
    constant = _constant(node, constants) if constant_needed else None
    if authored:
        # Generated arithmetic uses canonical coordinate numbers. Compensate
        # each operation locally; parents consume these already-prepared facts.
        for prime, exponent in prepared.unit_scale.powers:
            exponent = -exponent
            power = NumberNode(float(exponent.numerator), str(exponent.numerator))
            if exponent.denominator != 1:
                power = BinaryNode("/", power, NumberNode(float(exponent.denominator), str(exponent.denominator)))
            factor, factor_constant = _prepare(
                BinaryNode("^", NumberNode(float(prime), str(prime)), power), lookup,
                constant_needed=constant_needed,
            )
            combined = BinaryNode("*", prepared, factor)
            prepared = _finish(combined, (prepared, factor), (constant, factor_constant))
            constant = _constant(combined, (constant, factor_constant)) if constant_needed else None
    return prepared, constant


def prepare_expression(node: ExprNode, lookup: Callable[[str], Unit | None]) -> ExprNode:
    """Annotate admitted syntax; numerical execution only applies fixed factors."""
    return _prepare(node, lookup)[0]


def authored_projection(node: ExprNode, lookup: Callable[[str], Unit | None]) -> ExprNode:
    """Express generated canonical arithmetic as source that prepares identically."""
    return _prepare(node, lookup, authored=True)[0]
