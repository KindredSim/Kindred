"""
Parameter algebra for rate/equilibrium constants.

This adds an explicit, unambiguous syntax in mechanism DSL text:

    param k1 = 4*k2

`let` declares observables. Bare `name = expr` declarations are not supported.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from typing import Dict, Set

from kindred.core.equilibrium_rate_authority import (
    authority_fields_from_step_entry,
    effective_reverse_rate_from_keq,
    step_entry_authored_role_is_editable,
)
from kindred.core.rate_binding import RateBinding
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec
from kindred.core.simulator.parameter_algebra_eval import evaluate_parameter_algebra
from kindred.core.simulator.parameter_coordinates import (
    ParameterTarget,
    apply_mechanism_coordinate_values,
    read_mechanism_parameter_values,
)
from kindred.core.stoichiometry import directional_order_fraction
from kindred.core.units import (
    admit_keq,
    admit_rate,
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


def _read_scalar_param_values(mechanism: object, *, require_mutable: bool) -> Dict[str, float]:
    meta = getattr(mechanism, "metadata", {}) or {}
    if require_mutable:
        bindings = meta.get("scalar_param_bindings") or {}
        out: Dict[str, float] = {}
        if isinstance(bindings, dict):
            for nm, b in bindings.items():
                try:
                    out[str(nm)] = float(b()) if callable(b) else float(b)
                except Exception as exc:
                    logger.debug("Failed to read scalar param binding %r: %s", nm, exc, exc_info=True)
                    continue
        return out
    vals = meta.get("scalar_params") or {}
    out2: Dict[str, float] = {}
    if isinstance(vals, dict):
        for nm, v in vals.items():
            try:
                out2[str(nm)] = float(v)
            except (TypeError, ValueError, OverflowError) as exc:
                logger.debug("Skipping non-numeric scalar param %r=%r: %s", nm, v, exc, exc_info=True)
                continue
    return out2


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


def _active_equilibrium_keq_names(
    mechanism: object,
    spec: ParameterAlgebraSpec,
    *,
    parameter_target_map: Mapping[str, ParameterTarget],
) -> Set[str]:
    active: Set[str] = set()
    entry_by_keq_name: dict[str, dict] = {}
    for entry in _admitted_step_entries(parameter_target_map):
        if str(entry.get("kind") or "") != "equilibrium":
            continue
        try:
            name = f"Keq{int(entry.get('step_index'))}"
        except (TypeError, ValueError):
            continue
        entry_by_keq_name[name] = entry
        if not bool(step_entry_authored_role_is_editable(entry, "Keq")):
            continue
        active.add(name)
    for stmt in spec.param_statements or []:
        name = str(stmt.name)
        if not re.match(r"^Keq\d+$", name):
            continue
        entry = entry_by_keq_name.get(name)
        authority = authority_fields_from_step_entry(entry or {})
        if authority and bool(authority.get("has_thermo_param")) and not bool(step_entry_authored_role_is_editable(entry or {}, "Keq")):
            raise ValueError(f"{name} cannot override dG_eq equilibrium authority.")
        active.add(name)
    return active


def _apply_equilibrium_Keq_constraints_to_values(
    mechanism: object,
    base_values: Dict[str, float],
    *,
    active_keq_names: Set[str],
    parameter_target_map: Mapping[str, ParameterTarget],
) -> None:
    """
    Populate derived equilibrium rate values implied by explicit Keq parameters.

    This ensures parameter-algebra expressions that reference the derived rate
    see a consistent value.
    """
    for entry in _admitted_step_entries(parameter_target_map):
        if str(entry.get("kind") or "") != "equilibrium":
            continue
        try:
            n = int(entry.get("step_index"))  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid equilibrium Keq-constraint step_index={entry.get('step_index')!r}."
            ) from exc
        authority = authority_fields_from_step_entry(entry)
        if not authority:
            raise ValueError(
                f"Equilibrium Keq-constraint step_index={entry.get('step_index')!r} is missing equilibrium_authority."
            )
        derive_rate = str(authority.get("derived_role") or "")
        if derive_rate not in {"kf", "kr"}:
            derive_rate = "kr"
        kf_key = f"kf{n}"
        kr_key = f"kr{n}"
        keq_key = f"Keq{n}"
        if keq_key not in active_keq_names:
            continue
        if keq_key not in base_values:
            raise ValueError(f"Active equilibrium constraint {keq_key!r} has no source value.")
        keq = float(base_values[keq_key])
        try:
            keq = admit_keq(keq, field=keq_key)
        except ValueError as exc:
            raise ValueError(
                f"Active equilibrium constraint {keq_key!r} has invalid value {keq!r}."
            ) from exc
        if derive_rate == "kf":
            if kr_key in base_values:
                try:
                    base_values[kf_key] = admit_rate(
                        float(base_values[kr_key]) * keq,
                        field=kf_key,
                        allow_zero=True,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Active equilibrium constraint {keq_key!r} derives unsupported {kf_key!r}."
                    ) from exc
            else:
                raise ValueError(f"Active equilibrium constraint {keq_key!r} cannot derive missing {kf_key!r}.")
        else:
            if kf_key in base_values:
                base_values[kr_key] = float(
                    effective_reverse_rate_from_keq(
                        base_values[kf_key],
                        keq,
                    )
                )
            else:
                raise ValueError(f"Active equilibrium constraint {keq_key!r} cannot derive missing {kr_key!r}.")


def parameter_algebra_spec_from_mechanism(mechanism: object) -> ParameterAlgebraSpec | None:
    meta = getattr(mechanism, "metadata", {}) or {}
    if not isinstance(meta, dict):
        return None
    spec = meta.get(_PARAMETER_ALGEBRA_SPEC_META_KEY)
    return spec if isinstance(spec, ParameterAlgebraSpec) else None


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
    active_keq_names = _active_equilibrium_keq_names(
        mechanism,
        spec,
        parameter_target_map=parameter_target_map,
    )
    base_values: Dict[str, float] = {}
    base_values.update(
        read_mechanism_parameter_values(
            mechanism,
            names=spec.mechanism_param_names,
            parameter_target_map=parameter_target_map,
            coordinate_overrides=direct_coordinates,
        )
    )
    # Direct request coordinates are part of the same candidate value map
    # consumed by Algebra.  They must be visible while deriving coupled
    # mechanism coordinates (for example dG_act1 + Keq1) and are published
    # only by the grouped compositor below.
    base_values.update(direct_coordinates)
    base_values.update(_read_scalar_param_values(mechanism, require_mutable=require_mutable))
    # Candidate scalar values are admitted into the evaluation map, but are
    # not published to mechanism metadata until every dependent physical and
    # indexed replacement has been constructed successfully.
    base_values.update(scalar_overrides)
    _apply_equilibrium_Keq_constraints_to_values(
        mechanism,
        base_values,
        active_keq_names=active_keq_names,
        parameter_target_map=parameter_target_map,
    )

    derived: Dict[str, float] = {}
    if spec.param_statements:
        derived = evaluate_parameter_algebra(spec, base_values=base_values)

    coordinate_updates: dict[str, float] = dict(direct_coordinates)
    for name, value in tuple(derived.items()):
        if name not in spec.mechanism_param_names:
            continue
        target = parameter_target_map.get(str(name))
        if target is None:
            raise DSLError(
                f"Unsupported mechanism parameter {name!r} for parameter algebra"
            )
        coordinate_updates[name] = float(value)

    proposed_values = dict(base_values)
    proposed_values.update(derived)
    # Validate every thermodynamic consequence against the complete proposed
    # map before publishing any physical or indexed replacement.
    _apply_equilibrium_Keq_constraints_to_values(
        mechanism,
        proposed_values,
        active_keq_names=active_keq_names,
        parameter_target_map=parameter_target_map,
    )

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
        if spec.parameter_assignment_kind(str(stmt.name)) == "mechanism-bound":
            continue
        cur_val = base_values.get(stmt.name)
        if cur_val is None:
            cur_val = float("nan")
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
            if stmt.name in derived:
                b.set(float(derived[stmt.name]))
            scalar_info[stmt.name] = {
                "line": stmt.line_number,
                "expr": stmt.expr_src,
                "derived": bool(stmt.name in derived),
                "editable": not bool(stmt.name in derived),
                "unit": "1",
            }
        else:
            if stmt.name not in scalar_vals and math.isfinite(cur_val):
                scalar_vals[stmt.name] = float(cur_val)
            if stmt.name in scalar_overrides:
                scalar_vals[stmt.name] = float(scalar_overrides[stmt.name])
            if stmt.name in derived:
                scalar_vals[stmt.name] = float(derived[stmt.name])
            scalar_info[stmt.name] = {
                "line": stmt.line_number,
                "expr": stmt.expr_src,
                "derived": bool(stmt.name in derived),
                "editable": not bool(stmt.name in derived),
                "unit": "1",
            }

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
