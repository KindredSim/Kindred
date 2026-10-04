"""Compose unit facts from admitted parameter relationships and chemistry roles."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from kindred.core.algebra.parser import CallNode, IdentNode, NumberNode, SpeciesRefNode, UnaryNode
from kindred.core.algebra.printer import expression_to_source
from kindred.core.algebra.symbols import BUILTIN_FUNCTIONS
from kindred.core.algebra.unit_preparation import _ZERO_RULES, _children, prepare_expression
from kindred.core.mechanism_metadata import MechanismMetadataView
from kindred.core.quantity_units import (
    CONSTANT_UNITS, DIMENSIONLESS, ENERGY_PER_MOLE, MOLAR, SECOND,
    conversion_scale, parse_unit, scientific_unit,
)
from kindred.core.stoichiometry import directional_order_fraction
from kindred.core.units import ordinary_parameter_input_bounds
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
            f"with a parameter, for example: param d = 10; unit = {energy_input.symbol}",
            line_number=stmt.line, line_content=stmt.line_text,
        )


def prepare_spec_units(spec, mechanism, graph):
    """Prepare before identity proofs; chemistry coordinates are fixed-unit leaves."""
    units = dict(CONSTANT_UNITS)
    inputs = {}
    metadata = MechanismMetadataView.from_metadata(mechanism.metadata)
    energy_input = parse_unit(metadata.energy_unit)
    energies = set()
    for name, (_route, kind, index, role, _entry) in build_parameter_target_map(mechanism).items():
        step = mechanism.reactions[index] if kind == "reaction" else mechanism.equilibria[index]
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
    for assignment in spec.param_statements:
        if assignment.name not in spec.mechanism_param_names and not graph[assignment.name]:
            input_unit = assignment.declared_unit or DIMENSIONLESS
            try:
                ordinary_parameter_input_bounds(input_unit.symbol)
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
        if name in energies:
            declared = stmt.declared_unit or energy_input
            if not declared.compatible(ENERGY_PER_MOLE):
                raise DSLError(f"{name} is an energy, so its unit must be an energy per mole such as "
                               f"{energy_input.symbol}, not {declared.symbol}.",
                               line_number=stmt.line, line_content=stmt.line_text)
            if expr.quantity_unit is not None and expr.quantity_unit.compatible(DIMENSIONLESS):
                # A plain-number energy is read in its declared unit, else the energy= unit.
                scale = expr.unit_scale * conversion_scale(declared, units[name])
                expr = replace(expr, quantity_unit=units[name], unit_scale=scale, unit_factor=scale.number())
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
