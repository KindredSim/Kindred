"""Compose unit facts from admitted parameter relationships and chemistry roles."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from kindred.core.algebra.parser import CallNode, IdentNode, NumberNode, SpeciesRefNode, UnaryNode, _iter_identifiers
from kindred.core.algebra.printer import expression_to_source
from kindred.core.algebra.symbols import BUILTIN_FUNCTIONS
from kindred.core.algebra.unit_preparation import _ZERO_RULES, _children, prepare_expression
from kindred.core.mechanism_metadata import MechanismMetadataView
from kindred.core.quantity_units import (
    CONSTANT_UNITS, DIMENSIONLESS, ENERGY_PER_MOLE, MOLAR, SECOND,
    conversion_scale, parse_unit, scientific_unit,
)
from kindred.core.stoichiometry import directional_order_fraction
from kindred.core.units import input_unit_bounds, role_domain
from .parameter_coordinates import build_parameter_target_map
from .errors import DSLError


def _refuse_energy_offsets(node, stmt, energy_input):
    """An energy never meets a plain number in a sum, comparison or selection."""
    if isinstance(node, (NumberNode, IdentNode, SpeciesRefNode)):
        return
    children = _children(node)
    for child in children:
        _refuse_energy_offsets(child, stmt, energy_input)
    operation = node.name if isinstance(node, CallNode) else node.op
    rule = BUILTIN_FUNCTIONS[operation].unit_rule if operation in BUILTIN_FUNCTIONS else operation
    if isinstance(node, UnaryNode) or rule not in _ZERO_RULES:
        return
    units = [child.quantity_unit for child in children[rule == "branches":]
             if child.quantity_unit is not None and not (isinstance(child, NumberNode) and child.value == 0)]
    if any(unit.compatible(ENERGY_PER_MOLE) for unit in units) and any(unit.compatible(DIMENSIONLESS) for unit in units):
        raise DSLError(
            f"'{expression_to_source(node)}' mixes an energy with a plain number. Give the number an energy unit "
            f"with a parameter, for example 'param d = 10; unit = {energy_input.symbol}'.",
            line_number=stmt.line, line_content=stmt.line_text,
        )


def _has_unit_leaf(node) -> bool:
    """Whether a quantity carrying a unit appears anywhere in a prepared formula."""
    if isinstance(node, NumberNode):
        return False
    if isinstance(node, (IdentNode, SpeciesRefNode)):
        return node.quantity_unit is None or not node.quantity_unit.compatible(DIMENSIONLESS)
    return any(_has_unit_leaf(child) for child in _children(node))


def _unknown_unit_part(node):
    """The innermost part of a prepared formula whose unit cannot be determined."""
    unknown = next((child for child in _children(node) if child.quantity_unit is None), None) \
        if not isinstance(node, (NumberNode, IdentNode, SpeciesRefNode)) else None
    return node if unknown is None else _unknown_unit_part(unknown)


def _coordinate_unit_message(name, declared, unit, energy_input, *, energy):
    if energy:
        return (f"{name} is an energy, so its unit must be an energy per mole such as "
                f"{energy_input.symbol}, not {declared}.")
    if unit.compatible(DIMENSIONLESS):
        return f"{name} is a plain number here. Remove unit= or write unit = 1."
    return f"{declared} is not a unit of {name}. Use a unit such as {unit.symbol}."


def _formula_result(unit) -> str:
    return "has no unit" if unit.compatible(DIMENSIONLESS) else f"comes out in {unit.symbol}"


def prepare_spec_units(spec, mechanism):
    """Prepare before identity proofs; chemistry coordinates are fixed-unit leaves."""
    units = dict(CONSTANT_UNITS)
    inputs = {}
    metadata = MechanismMetadataView.from_metadata(mechanism.metadata)
    energy_input = parse_unit(metadata.energy_unit)
    energies = set()
    coordinates = set()
    for name, (_route, kind, index, role, _entry) in build_parameter_target_map(mechanism).items():
        step = mechanism.reactions[index] if kind == "reaction" else mechanism.equilibria[index]
        coordinates.add(name)
        if role in {"Ea", "dG_act", "dG_eq"}:
            unit = ENERGY_PER_MOLE
            energies.add(name)
        elif role in {"kappa", "degeneracy"}:
            unit = DIMENSIONLESS
        elif role == "Keq":
            unit = MOLAR.power(directional_order_fraction(step.stoich_back) - directional_order_fraction(step.stoich_forward))
        else:
            orders = step.rate_orders if kind == "reaction" else step.stoich_back if role == "kr" else step.stoich_forward
            unit = MOLAR.power(Fraction(1) - directional_order_fraction(orders)).divide(SECOND)
        unit = scientific_unit(unit.dimensions)
        units[name] = unit
        inputs[name] = energy_input if role in {"Ea", "dG_act", "dG_eq"} else unit
    statements = {stmt.name: stmt for block in (spec.parameter_block, spec.observable_block) for stmt in block.lines}
    # A param line of plain numbers is an input: its formula stays in its input
    # unit and crosses the input conversion once, when seeded or overridden.
    authored = {assignment.name for assignment in spec.param_statements
                if spec.parameter_assignment_kind(assignment.name) in {"mechanism input", "independent scalar"}}
    for assignment in spec.param_statements:
        if assignment.name in authored and assignment.name not in spec.mechanism_param_names:
            input_unit = assignment.declared_unit or DIMENSIONLESS
            try:
                input_unit_bounds(input_unit.symbol, *role_domain("scalar")[:2])
            except ValueError as exc:
                raise DSLError(
                    f"Unsupported input unit {input_unit.symbol!r} for {assignment.name}: "
                    f"the complete editable parameter range and its conversions must be supported. {exc}",
                    line_number=assignment.line_number, line_content=assignment.line_content,
                ) from exc
            inputs[assignment.name] = input_unit
            units[assignment.name] = scientific_unit(input_unit.dimensions)
    prepared = {}
    visiting = set()
    quantities = set(coordinates)

    def lookup(name):
        name = spec.mechanism_namespace.resolve(name).canonical_name or name
        if name in units:
            return units[name]
        if name not in statements:
            return None
        if name in visiting:
            raise DSLError(f"Circular unit dependency for parameter {name!r}.")
        prepare(name)
        return units[name]

    def prepare(name):
        if name in prepared:
            return
        visiting.add(name)
        stmt = statements[name]
        try:
            expr = prepare_expression(stmt.expr, lookup)
        except ValueError as exc:
            raise DSLError(str(exc), line_number=stmt.line, line_content=stmt.line_text) from exc
        _refuse_energy_offsets(expr, stmt, energy_input)
        if (stmt.declared_unit is not None and expr.quantity_unit is None
                and not stmt.declared_unit.compatible(DIMENSIONLESS)):
            raise DSLError(
                f"Kindred cannot determine the unit of '{expression_to_source(_unknown_unit_part(expr))}' "
                f"in the formula for {name}, so unit = {stmt.declared_unit.symbol} cannot apply.",
                line_number=stmt.line, line_content=stmt.line_text,
            )
        declared = None
        if name in coordinates:
            declared = stmt.declared_unit or inputs[name]
            if not declared.compatible(units[name]):
                raise DSLError(_coordinate_unit_message(name, declared.symbol, units[name], energy_input,
                                                        energy=name in energies),
                               line_number=stmt.line, line_content=stmt.line_text)
        elif (stmt.declared_unit is not None and not stmt.declared_unit.compatible(DIMENSIONLESS)
              and name not in authored and not _has_unit_leaf(expr)):
            declared = stmt.declared_unit
        # A formula reading a reaction parameter (even one whose unit is 1), or a
        # parameter whose own formula reads one, takes its unit from its content.
        reads_quantity = any(item in quantities for item in _iter_identifiers(expr))
        if reads_quantity:
            quantities.add(name)
        if (declared is not None and not reads_quantity
                and expr.quantity_unit is not None and expr.quantity_unit.compatible(DIMENSIONLESS)):
            # A unitless formula that reads no reaction parameter is read in its
            # declared unit; a mechanism coordinate without one uses its own
            # input unit (energy= for energies).
            if name in coordinates:
                inputs[name] = declared
            if name not in authored:
                target = units[name] if name in coordinates else scientific_unit(declared.dimensions)
                scale = expr.unit_scale * conversion_scale(declared, target)
                expr = replace(expr, quantity_unit=target, unit_scale=scale, unit_factor=scale.number())
        elif name in coordinates:
            if expr.quantity_unit is None:
                # Numbers of a formula without a determinable unit are taken in SI.
                inputs[name] = units[name]
            elif not expr.quantity_unit.compatible(units[name]):
                expected = ("no unit" if units[name].compatible(DIMENSIONLESS)
                            else energy_input.symbol if name in energies else units[name].symbol)
                raise DSLError(
                    f"The formula for {name} {_formula_result(expr.quantity_unit)}, which does not fit "
                    f"{name} ({expected}). Check the formula and the units of its parameters.",
                    line_number=stmt.line, line_content=stmt.line_text,
                )
        elif (stmt.declared_unit is not None and name not in inputs and expr.quantity_unit is not None
              and not stmt.declared_unit.compatible(expr.quantity_unit)):
            raise DSLError(
                f"The formula for {name} {_formula_result(expr.quantity_unit)}, so unit = {stmt.declared_unit.symbol} does not fit it.",
                line_number=stmt.line, line_content=stmt.line_text,
            )
        prepared[name] = replace(stmt, expr=expr)
        units.setdefault(name, expr.quantity_unit)
        visiting.remove(name)

    for name in statements:
        prepare(name)

    def block(source):
        lines = tuple(prepared[stmt.name] for stmt in source.lines)
        return replace(source, lines=lines, ast=tuple(stmt.expr for stmt in lines))

    return replace(spec, parameter_block=block(spec.parameter_block), observable_block=block(spec.observable_block),
                   quantity_units=tuple(units.items()), input_units=tuple(inputs.items()))
