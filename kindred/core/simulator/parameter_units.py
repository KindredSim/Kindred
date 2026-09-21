"""Compose unit facts from admitted parameter relationships and chemistry roles."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from kindred.core.algebra.unit_preparation import prepare_expression
from kindred.core.mechanism_metadata import MechanismMetadataView
from kindred.core.quantity_units import (
    CONSTANT_UNITS, DIMENSIONLESS, ENERGY_PER_MOLE, MOLAR, SECOND,
    parse_unit, scientific_unit,
)
from kindred.core.stoichiometry import directional_order_fraction
from kindred.core.units import ordinary_parameter_input_bounds
from .parameter_coordinates import build_parameter_target_map
from .errors import DSLError


def prepare_spec_units(spec, mechanism, graph):
    """Prepare before identity proofs; chemistry coordinates are fixed-unit leaves."""
    units = dict(CONSTANT_UNITS)
    inputs = {}
    metadata = MechanismMetadataView.from_metadata(mechanism.metadata)
    energy_input = parse_unit(metadata.energy_unit)
    for name, (_route, kind, index, role, _entry) in build_parameter_target_map(mechanism).items():
        step = mechanism.reactions[index] if kind == "reaction" else mechanism.equilibria[index]
        if role in {"Ea", "dG_act", "dG_eq"}:
            unit = ENERGY_PER_MOLE
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
        prepared[name] = replace(stmt, expr=expr)
        units.setdefault(name, expr.quantity_unit)
        visiting.remove(name)

    for name in statements:
        prepare(name)

    def block(source):
        lines = tuple(prepared[stmt.name] for stmt in source.lines)
        return replace(source, lines=lines, ast=tuple(stmt.expr for stmt in lines), static_values=())

    return replace(spec, parameter_block=block(spec.parameter_block), observable_block=block(spec.observable_block),
                   quantity_units=tuple(units.items()), input_units=tuple(inputs.items()))
