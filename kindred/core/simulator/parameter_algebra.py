"""
Parameter algebra for rate/equilibrium constants.

This adds an explicit, unambiguous syntax in mechanism DSL text:

    param k1 = 4*k2

`let` declares observables. Bare `name = expr` declarations are not supported.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Dict

from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec
from kindred.core.algebra.evaluator import bind_scalar_expression_evaluator
from kindred.core.algebra.errors import AlgebraError
from kindred.core.simulator.parameter_coordinates import (
    ParameterTarget,
    apply_mechanism_coordinate_values,
)
from kindred.core.stoichiometry import directional_order_fraction
from kindred.core.units import (
    rate_constant_unit,
)

logger = logging.getLogger(__name__)

_PARAMETER_ALGEBRA_SPEC_META_KEY = "parameter_algebra_spec"


def _admitted_step_entries(
    parameter_target_map: Mapping[str, ParameterTarget],
) -> tuple[Mapping[str, object], ...]:
    """Return each admitted step entry once, without rescanning the roster."""

    entries: list[Mapping[str, object]] = []
    seen: set[int] = set()
    for target in parameter_target_map.values():
        _route, _kind, _index, _role, entry = target
        identity = id(entry)
        if identity in seen:
            continue
        seen.add(identity)
        entries.append(entry)
    return tuple(entries)

__all__ = [
    "apply_parameter_algebra_spec_to_mechanism",
    "parameter_algebra_spec_from_mechanism",
    "solver_parameter_units_from_mechanism",
]


def parameter_algebra_spec_from_mechanism(mechanism: object) -> ParameterAlgebraSpec | None:
    return (getattr(mechanism, "metadata", {}) or {}).get(_PARAMETER_ALGEBRA_SPEC_META_KEY)


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
    scalar_info = {}
    statements = {statement.name: statement for statement in spec.parameter_block.lines}
    literal = bind_scalar_expression_evaluator(spec.parameter_block, {})
    overrides = scalar_parameter_overrides or {}
    for assignment in spec.param_statements:
        name = assignment.name
        if name in spec.mechanism_param_names:
            continue
        derived = spec.parameter_assignment_kind(name) != "independent scalar"
        scalar_info[name] = {
            "line": assignment.line_number, "expr": assignment.expr_src,
            "derived": derived, "editable": not derived, "unit": "1",
        }
        if derived:
            scalars.pop(name, None)
            continue
        if name in overrides:
            scalars[name] = float(overrides[name])
        elif name not in scalars:
            try:
                scalars[name] = literal(statements[name])
            except AlgebraError as exc:
                raise DSLError(str(exc), line_number=assignment.line_number,
                               line_content=assignment.line_content) from exc

    apply_mechanism_coordinate_values(
        mechanism, mechanism_coordinate_overrides or {},
        parameter_target_map=parameter_target_map,
    )
    meta[_PARAMETER_ALGEBRA_SPEC_META_KEY] = spec
    meta["scalar_params"] = scalars
    meta["scalar_param_info"] = scalar_info


def solver_parameter_units_from_mechanism(
    mechanism: object,
    *,
    parameter_target_map: Mapping[str, ParameterTarget],
) -> Dict[str, str]:
    """
    Unit map for solver parameters backed by authoritative step metadata.

    - k{i}: based on Reaction.order (mass-action), units M^(1-order)/s
    - kf{i}, kr{i}: based on equilibrium forward/back molecularity
    - Keq{i}: concentration-form unit M^(nr-nf)
    - scalar params: dimensionless ("1") (if present in metadata)
    """
    units: Dict[str, str] = {}
    rxns = getattr(mechanism, "reactions", []) or []
    eqs = getattr(mechanism, "equilibria", []) or []
    for name, target in parameter_target_map.items():
        route, kind, idx, role, entry = target
        if route != "indexed":
            continue
        if kind == "reaction" and role == "k":
            if not (0 <= idx < len(rxns)):
                raise ValueError(
                    f"Authoritative step metadata for {name!r} has reaction_index {idx} out of range."
                )
            order = float(
                directional_order_fraction(
                    getattr(rxns[idx], "rate_orders"),
                )
            )
            units[name] = rate_constant_unit(order)
        elif kind == "equilibrium":
            if not (0 <= idx < len(eqs)):
                raise ValueError(
                    f"Authoritative step metadata for {name!r} has equilibrium_index {idx} out of range."
                )
            eq = eqs[idx]
            if role == "Keq":
                nf = directional_order_fraction(
                    getattr(eq, "stoich_forward", {}) or {},
                )
                nr = directional_order_fraction(
                    getattr(eq, "stoich_back", {}) or {},
                )
                from kindred.core.units import equilibrium_constant_unit
                units[name] = equilibrium_constant_unit(float(nr - nf))
                continue
            fwd_order = float(
                directional_order_fraction(
                    getattr(eq, "stoich_forward"),
                )
            )
            back_order = float(
                directional_order_fraction(
                    getattr(eq, "stoich_back"),
                )
            )
            if role == "kf":
                units[name] = rate_constant_unit(fwd_order)
            elif role == "kr":
                units[name] = rate_constant_unit(back_order)

    meta = getattr(mechanism, "metadata", {}) or {}
    scalar_info = meta.get("scalar_param_info") or {}
    if isinstance(scalar_info, dict):
        for nm, info in scalar_info.items():
            if isinstance(info, dict):
                units[str(nm)] = str(info.get("unit") or "1")
            else:
                units[str(nm)] = "1"
    return units
