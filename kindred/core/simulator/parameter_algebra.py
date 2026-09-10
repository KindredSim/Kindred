"""
Parameter algebra for rate/equilibrium constants.

This adds an explicit, unambiguous syntax in mechanism DSL text:

    param k1 = 4*k2

`let` declares observables. Bare `name = expr` declarations are not supported.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Dict

from kindred.core.rate_binding import RateBinding
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec
from kindred.core.simulator.parameter_algebra_eval import bind_parameter_evaluation
from kindred.core.mechanism_metadata import MechanismMetadataView
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


def _ensure_scalar_param_storage(mechanism: object, *, require_mutable: bool) -> None:
    meta = getattr(mechanism, "metadata", None)
    if not isinstance(meta, dict):
        return
    if require_mutable:
        meta.setdefault("scalar_param_bindings", {})
        meta.setdefault("scalar_param_info", {})
    else:
        meta.setdefault("scalar_params", {})
        meta.setdefault("scalar_param_info", {})


def parameter_algebra_spec_from_mechanism(mechanism: object) -> ParameterAlgebraSpec | None:
    return (getattr(mechanism, "metadata", {}) or {}).get(_PARAMETER_ALGEBRA_SPEC_META_KEY)


def apply_parameter_algebra_spec_to_mechanism(
    spec: ParameterAlgebraSpec,
    *,
    mechanism: object,
    require_mutable: bool,
    parameter_target_map: Mapping[str, ParameterTarget],
    mechanism_coordinate_overrides: Mapping[str, float] | None = None,
    scalar_parameter_overrides: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    meta = getattr(mechanism, "metadata", None)
    direct_coordinates = mechanism_coordinate_overrides or {}
    scalar_overrides = scalar_parameter_overrides or {}
    base_values = dict(direct_coordinates)
    base_values.update(scalar_overrides)
    evaluate = bind_parameter_evaluation(spec, mechanism, base_values)
    roots = tuple(name for name in spec.parameter_dependencies
                  if spec.parameter_assignment_kind(name) in {"independent scalar", "mechanism-bound"})
    base_values = evaluate(float(MechanismMetadataView.from_metadata(mechanism.metadata).temperature_K), roots)
    derived = {name: base_values[name] for name in spec.parameter_dependencies
               if spec.parameter_assignment_kind(name) == "mechanism-bound"}

    coordinate_updates: dict[str, float] = dict(direct_coordinates)
    for name, value in tuple(derived.items()):
        if name not in spec.mechanism_param_names or name in spec.identity_checks:
            continue
        target = parameter_target_map.get(str(name))
        if target is None:
            raise DSLError(
                f"Unsupported mechanism parameter {name!r} for parameter algebra"
            )
        coordinate_updates[name] = float(value)

    if coordinate_updates:
        apply_mechanism_coordinate_values(
            mechanism,
            coordinate_updates,
            parameter_target_map=parameter_target_map,
            require_mutable=require_mutable,
        )

    if isinstance(meta, dict):
        meta[_PARAMETER_ALGEBRA_SPEC_META_KEY] = spec
    _ensure_scalar_param_storage(mechanism, require_mutable=require_mutable)

    meta = getattr(mechanism, "metadata", {}) or {}
    if require_mutable:
        scalar_bindings = meta.get("scalar_param_bindings")
        if not isinstance(scalar_bindings, dict):
            scalar_bindings = {}
            meta["scalar_param_bindings"] = scalar_bindings
        scalar_info = meta.get("scalar_param_info")
        if not isinstance(scalar_info, dict):
            scalar_info = {}
            meta["scalar_param_info"] = scalar_info
    else:
        scalar_vals = meta.get("scalar_params")
        if not isinstance(scalar_vals, dict):
            scalar_vals = {}
            meta["scalar_params"] = scalar_vals
        scalar_info = meta.get("scalar_param_info")
        if not isinstance(scalar_info, dict):
            scalar_info = {}
            meta["scalar_param_info"] = scalar_info

    for stmt in spec.param_statements:
        # Mechanism-bound assignments are written through the grouped
        # coordinate owner above; only scalar assignments have metadata
        # storage here.
        if stmt.name in spec.mechanism_param_names:
            continue
        is_derived = spec.parameter_assignment_kind(stmt.name) == "derived scalar"
        scalar_info[stmt.name] = {
            "line": stmt.line_number,
            "expr": stmt.expr_src,
            "derived": is_derived,
            "editable": not is_derived,
            "unit": "1",
        }
        if is_derived:
            # Derived scalars belong to the bound expression graph, not a
            # nominal numeric snapshot that can become stale or fail unused.
            for key in ("scalar_params", "scalar_param_bindings"):
                meta.get(key, {}).pop(stmt.name, None)
            continue
        cur_val = base_values[stmt.name]
        if require_mutable:
            b = scalar_bindings.get(stmt.name)
            if b is None:
                b = RateBinding(name=str(stmt.name), value=float(cur_val))
                scalar_bindings[stmt.name] = b
            elif stmt.name in scalar_overrides:
                # Publish a fresh binding only after mechanism coordinates have
                # passed admission; the old binding remains untouched on error.
                b = RateBinding(
                    name=str(stmt.name),
                    value=float(scalar_overrides[stmt.name]),
                )
                scalar_bindings[stmt.name] = b
        else:
            if stmt.name not in scalar_vals and math.isfinite(cur_val):
                scalar_vals[stmt.name] = float(cur_val)
            if stmt.name in scalar_overrides:
                scalar_vals[stmt.name] = float(scalar_overrides[stmt.name])

    return derived


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
