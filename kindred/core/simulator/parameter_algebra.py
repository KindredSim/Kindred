"""
Parameter algebra for rate/equilibrium constants.

This adds an explicit, unambiguous syntax in mechanism DSL text:

    param k1 = 4*k2

`let` declares observables. Bare `name = expr` declarations are not supported.
"""

from __future__ import annotations

from collections.abc import Mapping

from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec
from kindred.core.algebra.evaluator import bind_scalar_expression_evaluator
from kindred.core.algebra.errors import AlgebraError
from kindred.core.simulator.parameter_coordinates import (
    ParameterTarget,
    apply_mechanism_coordinate_values,
)
from kindred.core.quantity_units import convert_value
from kindred.core.units import admit_role_value
from kindred.core.validation import parse_finite_numeric_text

_PARAMETER_ALGEBRA_SPEC_META_KEY = "parameter_algebra_spec"


__all__ = [
    "apply_parameter_algebra_spec_to_mechanism",
    "parameter_algebra_spec_from_mechanism",
]


def parameter_algebra_spec_from_mechanism(mechanism: object) -> ParameterAlgebraSpec:
    return mechanism.metadata[_PARAMETER_ALGEBRA_SPEC_META_KEY]


def apply_parameter_algebra_spec_to_mechanism(
    spec: ParameterAlgebraSpec,
    *,
    mechanism: object,
    parameter_target_map: Mapping[str, ParameterTarget],
    mechanism_coordinate_overrides: Mapping[str, float] | None = None,
    scalar_parameter_overrides: Mapping[str, float] | None = None,
) -> None:
    """Seed independent inputs and publish admitted overrides, never derived values."""
    meta = mechanism.metadata
    scalars = dict(meta.get("scalar_params") or {})
    statements = {statement.name: statement for statement in spec.parameter_block.lines}
    literal = bind_scalar_expression_evaluator(
        spec.parameter_block, {},
        literal_reader=lambda node: parse_finite_numeric_text(node.literal, field="Parameter literal")
        if node.literal is not None else node.value,
    )
    overrides = scalar_parameter_overrides or {}
    input_units = dict(spec.input_units)
    scientific_units = dict(spec.quantity_units)

    def admit_scalar(value: float, name: str) -> float:
        return admit_role_value(value, role="scalar", unit=input_units[name].symbol, allow_exact_zero=True, field=name)

    for assignment in spec.param_statements:
        name = assignment.name
        if name in spec.mechanism_param_names:
            if spec.parameter_assignment_kind(name) == "mechanism input":
                try:
                    # Its plain numbers are in its input unit, converted once
                    # like an override; admission is refused at this param line.
                    apply_mechanism_coordinate_values(
                        mechanism, {name: convert_value(literal(statements[name]), input_units[name],
                                                        scientific_units[name])},
                        parameter_target_map=parameter_target_map, units=input_units,
                    )
                except (AlgebraError, ValueError) as exc:
                    raise DSLError(str(exc), line_number=assignment.line_number,
                                   line_content=assignment.line_content) from exc
            continue
        derived = spec.parameter_assignment_kind(name) != "independent scalar"
        if derived:
            scalars.pop(name, None)
            continue
        if name in overrides:
            scalars[name] = admit_scalar(convert_value(float(overrides[name]), input_units[name], scientific_units[name]), name)
        elif name not in scalars:
            try:
                scalars[name] = admit_scalar(convert_value(literal(statements[name]), input_units[name], scientific_units[name]), name)
            except (AlgebraError, ValueError) as exc:
                raise DSLError(str(exc), line_number=assignment.line_number,
                               line_content=assignment.line_content) from exc

    apply_mechanism_coordinate_values(
        mechanism, {name: convert_value(value, input_units[name], scientific_units[name])
                    for name, value in (mechanism_coordinate_overrides or {}).items()},
        parameter_target_map=parameter_target_map, units=input_units,
    )
    meta[_PARAMETER_ALGEBRA_SPEC_META_KEY] = spec
    meta["scalar_params"] = scalars
