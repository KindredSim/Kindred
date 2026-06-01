"""
Parameter algebra for rate/equilibrium constants.

This adds an explicit, unambiguous syntax in mechanism DSL text:

    param k1 = 4*k2

`let` declares observables. Bare `name = expr` declarations are not supported.
"""

from __future__ import annotations

from dataclasses import replace
import logging
import math
import re
from typing import Dict, Optional, Set

from kindred.core.equilibrium_rate_authority import (
    EquilibriumRateAuthorityKind,
    authority_fields_from_step_entry,
    effective_equilibrium_keq,
    effective_equilibrium_reverse_rate,
    normalize_existing_equilibrium_rate_authority,
    require_step_entry_authored_role_is_editable,
    step_entry_authored_role_is_editable,
)
from kindred.core.mechanism_metadata import MechanismMetadataKeys
from kindred.core.rate_binding import RateBinding
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_namespace import (
    MechanismParameterNamespace,
    build_namespace_from_mechanism,
)
from kindred.core.simulator.parameter_algebra_eval import (
    evaluate_parameter_algebra,
    evaluate_parameter_algebra_in_context as _evaluate_param_block_in_context,
)
from kindred.core.simulator.parameter_algebra_spec import (
    ParameterAlgebraSpec,
    ParameterAssignment,
    ParameterOverrideWarning,
    extract_parameter_assignments_from_algebra_lines,
    extract_parameter_assignments_from_dsl_text,
    extract_observable_names_from_algebra_lines,
    mechanism_parameter_name_pattern,
    parse_parameter_algebra_spec_from_dsl_text,
)
from kindred.core.simulator.step_indexing import (
    get_step_index_map,
    iter_canonical_parameters,
    lookup_step_param_target,
)
logger = logging.getLogger(__name__)

_MECH_PARAM_RE = mechanism_parameter_name_pattern()
_WEGSCHEIDER_META_KEY = "wegscheider_cyclicity_enabled"
_PARAMETER_ALGEBRA_SPEC_META_KEY = "parameter_algebra_spec"

_PUBLIC_REEXPORTS = (
    ParameterAssignment,
    ParameterAlgebraSpec,
    ParameterOverrideWarning,
    extract_parameter_assignments_from_algebra_lines,
    extract_parameter_assignments_from_dsl_text,
    extract_observable_names_from_algebra_lines,
    parse_parameter_algebra_spec_from_dsl_text,
    evaluate_parameter_algebra,
)


def mechanism_parameter_namespace(mechanism: object) -> MechanismParameterNamespace:
    return build_namespace_from_mechanism(mechanism)


def mechanism_parameter_names(mechanism: object) -> Set[str]:
    return mechanism_parameter_namespace(mechanism).flat_names()


def read_mechanism_parameter_values(mechanism: object, *, names: Optional[Set[str]] = None) -> Dict[str, float]:
    wanted = names or mechanism_parameter_names(mechanism)
    out: Dict[str, float] = {}

    def _as_float(x: object) -> Optional[float]:
        if x is None:
            return None
        try:
            return float(x()) if callable(x) else float(x)
        except Exception:
            return None

    rxns = getattr(mechanism, "reactions", []) or []
    eqs = getattr(mechanism, "equilibria", []) or []
    mechanism_meta = getattr(mechanism, "metadata", {}) or {}
    temperature_K = _as_float(mechanism_meta.get(MechanismMetadataKeys.TEMPERATURE_K)) if isinstance(mechanism_meta, dict) else None
    if temperature_K is None:
        temperature_K = 298.15
    for name, entry, role in iter_canonical_parameters(mechanism):
        if name not in wanted:
            continue
        kind = str(entry.get("kind") or "")
        if kind == "reaction" and role == "k":
            idx = int(entry.get("reaction_index", -1))  # type: ignore[arg-type]
            if 0 <= idx < len(rxns):
                v = _as_float(getattr(rxns[idx], "rate", None))
                if v is not None:
                    out[name] = v
        elif kind == "equilibrium":
            idx = int(entry.get("equilibrium_index", -1))  # type: ignore[arg-type]
            if not (0 <= idx < len(eqs)):
                continue
            eq = eqs[idx]
            if role in {"kf", "kr"}:
                v = _as_float(getattr(eq, role, None))
                if role == "kr":
                    authority = normalize_existing_equilibrium_rate_authority(eq)
                    if authority.kind == EquilibriumRateAuthorityKind.KEQ:
                        v = effective_equilibrium_reverse_rate(eq, temperature_K=float(temperature_K))
                if v is not None:
                    out[name] = v
            elif role == "Keq":
                v = effective_equilibrium_keq(eq, temperature_K=float(temperature_K))
                if v is not None:
                    out[name] = v
    return out


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


def _set_mechanism_param(
    mechanism: object,
    name: str,
    value: float,
    *,
    require_mutable: bool,
) -> None:
    target = lookup_step_param_target(mechanism, name)
    if target is None:
        raise DSLError(f"Unsupported parameter name {name!r} for parameter algebra")
    kind, idx, role, entry = target
    if kind == "reaction" and role == "k":
        rxns = getattr(mechanism, "reactions", []) or []
        if not (0 <= idx < len(rxns)):
            raise DSLError(f"Unknown parameter {name!r} (reaction index out of range)")
        rxn = rxns[idx]
        current = getattr(rxn, "rate", None)
        if require_mutable:
            if hasattr(current, "set"):
                current.set(float(value))  # type: ignore[call-arg]
                return
            raise DSLError(f"Parameter {name!r} is not mutable in this run (missing binding)")
        rxns[idx] = replace(rxn, rate=float(value))
        return

    if kind == "equilibrium":
        eqs = getattr(mechanism, "equilibria", []) or []
        if not (0 <= idx < len(eqs)):
            raise DSLError(f"Unknown parameter {name!r} (equilibrium index out of range)")
        require_step_entry_authored_role_is_editable(entry, role, parameter_name=name)
        eq = eqs[idx]
        if role in {"kf", "kr"}:
            current = getattr(eq, role, None)
            if require_mutable:
                if hasattr(current, "set"):
                    current.set(float(value))  # type: ignore[call-arg]
                    return
                raise DSLError(f"Parameter {name!r} is not mutable in this run (missing binding)")
            eqs[idx] = replace(eq, **{role: float(value)})
            return
        if role == "Keq":
            meta = dict(getattr(eq, "metadata", {}) or {})
            current = meta.get("Keq_input")
            if require_mutable:
                if hasattr(current, "set"):
                    current.set(float(value))  # type: ignore[call-arg]
                    return
                # Allow creating a binding for Keq_input if not already mutable.
                b = RateBinding(name=str(name), value=float(value))
                meta["Keq_input"] = b
                eqs[idx] = replace(eq, metadata=meta)
                return
            meta["Keq_input"] = float(value)
            eqs[idx] = replace(eq, metadata=meta)
            return

    raise DSLError(f"Unsupported parameter target for {name!r}")


def _active_equilibrium_keq_names(mechanism: object, spec: ParameterAlgebraSpec) -> Set[str]:
    active: Set[str] = set()
    entry_by_keq_name: dict[str, dict] = {}
    for entry in get_step_index_map(mechanism):
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


def _validate_parameter_algebra_editable_targets(mechanism: object, spec: ParameterAlgebraSpec) -> None:
    for stmt in spec.param_statements or ():
        target = lookup_step_param_target(mechanism, str(stmt.name))
        if target is None:
            continue
        kind, _idx, role, entry = target
        if kind == "equilibrium":
            require_step_entry_authored_role_is_editable(entry, role, parameter_name=str(stmt.name))


def _apply_equilibrium_Keq_constraints_to_values(
    mechanism: object,
    base_values: Dict[str, float],
    *,
    active_keq_names: Set[str],
) -> None:
    """
    Populate derived equilibrium rate values implied by explicit Keq parameters.

    This ensures parameter-algebra expressions that reference the derived rate
    see a consistent value.
    """
    for entry in get_step_index_map(mechanism):
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
        if not math.isfinite(keq) or abs(keq) < 1e-30:
            raise ValueError(f"Active equilibrium constraint {keq_key!r} has invalid value {keq!r}.")
        if derive_rate == "kf":
            if kr_key in base_values:
                base_values[kf_key] = float(base_values[kr_key]) * keq
            else:
                raise ValueError(f"Active equilibrium constraint {keq_key!r} cannot derive missing {kf_key!r}.")
        else:
            if kf_key in base_values:
                base_values[kr_key] = float(base_values[kf_key]) / keq
            else:
                raise ValueError(f"Active equilibrium constraint {keq_key!r} cannot derive missing {kr_key!r}.")


def _apply_equilibrium_Keq_constraints_to_mechanism(
    mechanism: object,
    *,
    require_mutable: bool,
    active_keq_names: Set[str],
) -> Dict[str, float]:
    """
    Validate equilibrium constraints implied by explicit Keq parameters.

    Derived reverse/forward rates are effective values owned by the normalized
    equilibrium authority boundary. They are populated into evaluation values by
    `_apply_equilibrium_Keq_constraints_to_values`, but are not written back to
    raw rate slots or reported as applied parameter-algebra mutations here.
    """
    values = read_mechanism_parameter_values(mechanism)
    _apply_equilibrium_Keq_constraints_to_values(mechanism, values, active_keq_names=active_keq_names)
    _ = require_mutable
    return {}


def parameter_algebra_spec_from_mechanism(mechanism: object) -> ParameterAlgebraSpec | None:
    meta = getattr(mechanism, "metadata", {}) or {}
    if not isinstance(meta, dict):
        return None
    spec = meta.get(_PARAMETER_ALGEBRA_SPEC_META_KEY)
    return spec if isinstance(spec, ParameterAlgebraSpec) else None


def _parameter_override_warnings_for_spec(
    spec: ParameterAlgebraSpec,
    *,
    mechanism: object,
) -> tuple[ParameterOverrideWarning, ...]:
    warnings: list[ParameterOverrideWarning] = []
    rxns = getattr(mechanism, "reactions", []) or []
    eqs = getattr(mechanism, "equilibria", []) or []

    for stmt in spec.param_statements:
        target = lookup_step_param_target(mechanism, stmt.name)
        if target is None:
            continue
        kind, index, role, entry = target
        step_index = int(entry.get("step_index", 0) or 0)
        inline_name: str | None = None

        if kind == "reaction" and role == "k":
            if not (0 <= index < len(rxns)):
                continue
            overrides = getattr(rxns[index], "overrides", {}) or {}
            has_energy_model_override = overrides.get("model") in {"Arrhenius", "Eyring"} and any(
                overrides.get(key) is not None
                for key in ("A", "Ea", "Ea_J_per_mol", "dG_act_J_per_mol")
            )
            if not has_energy_model_override:
                inline_name = "k"
        elif kind == "equilibrium":
            if not (0 <= index < len(eqs)):
                continue
            if role == "kf" and bool(entry.get("user_provided_kf")):
                inline_name = "kf"
            elif role == "kr" and bool(entry.get("user_provided_kr")):
                inline_name = "kr"
            elif role == "Keq" and bool(step_entry_authored_role_is_editable(entry, "Keq")):
                inline_name = "Keq"

        if inline_name is None:
            continue
        warnings.append(
            ParameterOverrideWarning(
                param_name=str(stmt.name),
                inline_name=str(inline_name),
                step_index=int(step_index),
                message=f"param {stmt.name} overrides inline {inline_name} on step {step_index}",
            )
        )

    return tuple(warnings)


def apply_parameter_algebra_spec_to_mechanism(
    spec: ParameterAlgebraSpec,
    *,
    mechanism: object,
    require_mutable: bool,
) -> Dict[str, float]:
    spec = replace(
        spec,
        override_warnings=_parameter_override_warnings_for_spec(spec, mechanism=mechanism),
    )
    meta = getattr(mechanism, "metadata", None)
    if isinstance(meta, dict):
        meta[_PARAMETER_ALGEBRA_SPEC_META_KEY] = spec
    _validate_parameter_algebra_editable_targets(mechanism, spec)
    _ensure_scalar_param_storage(mechanism, require_mutable=require_mutable)
    active_keq_names = _active_equilibrium_keq_names(mechanism, spec)
    base_values: Dict[str, float] = {}
    base_values.update(read_mechanism_parameter_values(mechanism, names=spec.mechanism_param_names))
    base_values.update(_read_scalar_param_values(mechanism, require_mutable=require_mutable))
    _apply_equilibrium_Keq_constraints_to_values(
        mechanism,
        base_values,
        active_keq_names=active_keq_names,
    )

    derived: Dict[str, float] = {}
    if spec.param_statements:
        # Evaluate derived definitions, seeding scalar defaults in base_values if needed.
        derived = evaluate_parameter_algebra(spec, base_values=base_values)

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

    constrained: Dict[str, Dict[str, object]] = {}
    for stmt in spec.param_statements:
        is_mech = _MECH_PARAM_RE.match(stmt.name) is not None
        # Determine derived-ness for UI metadata: mechanism targets are always constrained.
        try:
            tmp_spec = ParameterAlgebraSpec(
                param_statements=[stmt],
                observable_names=set(spec.observable_names),
                mechanism_namespace=spec.mechanism_namespace,
                scalar_input_names=set(spec.scalar_input_names),
            )
            tmp_vals = _evaluate_param_block_in_context(tmp_spec, base_values=dict(base_values), enforce_defaults=False)
            default_val = float(tmp_vals.get(stmt.name, float("nan")))
        except Exception:
            default_val = float("nan")
        if not is_mech:
            # Base declaration iff expression has no parameter dependencies
            # (handled in evaluate_parameter_algebra), which seeds missing
            # base_values. Reflect base_values into storage.
            cur_val = base_values.get(stmt.name)
            if cur_val is None and math.isfinite(default_val):
                cur_val = default_val
            if cur_val is None:
                cur_val = float("nan")
            if require_mutable:
                b = scalar_bindings.get(stmt.name)
                if b is None:
                    b = RateBinding(name=str(stmt.name), value=float(cur_val))
                    scalar_bindings[stmt.name] = b
                # Do not overwrite base scalar params here; only derived scalar
                # params are present in `derived`.
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
                if stmt.name in derived:
                    scalar_vals[stmt.name] = float(derived[stmt.name])
                scalar_info[stmt.name] = {
                    "line": stmt.line_number,
                    "expr": stmt.expr_src,
                    "derived": bool(stmt.name in derived),
                    "editable": not bool(stmt.name in derived),
                    "unit": "1",
                }
        else:
            constrained[stmt.name] = {
                "line": stmt.line_number,
                "expr": stmt.expr_src,
                "constraint_reason": "algebra",
            }

    meta["constrained_params"] = constrained

    # Apply derived values to mechanism-bound parameters and derived scalar params.
    for nm, val in derived.items():
        if _MECH_PARAM_RE.match(nm):
            _set_mechanism_param(mechanism, nm, val, require_mutable=require_mutable)

    # Apply Keq-implied equilibrium constraints after any parameter algebra updates.
    eq_updates = _apply_equilibrium_Keq_constraints_to_mechanism(
        mechanism,
        require_mutable=require_mutable,
        active_keq_names=active_keq_names,
    )
    if eq_updates:
        derived = dict(derived)
        derived.update(eq_updates)

    meta = getattr(mechanism, "metadata", {}) or {}
    enabled = bool(meta.get(_WEGSCHEIDER_META_KEY, False)) if isinstance(meta, dict) else False
    if enabled:
        from kindred.core.simulator.wegscheider_symbolic import validate_wegscheider_cyclicity_resolved

        validate_wegscheider_cyclicity_resolved(
            mechanism,
            parameter_algebra_spec=spec,
        )
    return derived


def apply_parameter_algebra_to_mechanism(
    dsl_text: str,
    *,
    mechanism: object,
    require_mutable: bool,
) -> Dict[str, float]:
    """
    Apply `param` algebra from the DSL text to the given mechanism in-place.

    - In non-prepared mode (`require_mutable=False`), the mechanism is updated by
      replacing Reaction/Equilibrium objects with new numeric values.
    - In prepared/bound mode (`require_mutable=True`), updates must target mutable
      bindings (RateBinding-like objects with `.set()`).
    """
    mechanism_namespace = mechanism_parameter_namespace(mechanism)
    spec = parse_parameter_algebra_spec_from_dsl_text(
        dsl_text,
        mechanism_namespace=mechanism_namespace,
    )
    return apply_parameter_algebra_spec_to_mechanism(
        spec,
        mechanism=mechanism,
        require_mutable=require_mutable,
    )


def solver_parameter_units_from_mechanism(mechanism: object) -> Dict[str, str]:
    """
    Unit map for solver parameters backed by authoritative step metadata.

    - k{i}: based on Reaction.order (mass-action), units M^(1-order)/s
    - kf{i}, kr{i}: based on equilibrium forward/back molecularity
    - Keq{i}: dimensionless ("1")
    - scalar params: dimensionless ("1") (if present in metadata)
    """
    from kindred.core.simulator.parameter_units import rate_constant_unit

    units: Dict[str, str] = {}
    rxns = getattr(mechanism, "reactions", []) or []
    eqs = getattr(mechanism, "equilibria", []) or []
    for name, entry, role in iter_canonical_parameters(mechanism):
        kind = str(entry.get("kind") or "")
        if kind == "reaction" and role == "k":
            try:
                idx = int(entry.get("reaction_index", -1))  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Authoritative step metadata for {name!r} has an invalid reaction_index."
                ) from exc
            if not (0 <= idx < len(rxns)):
                raise ValueError(
                    f"Authoritative step metadata for {name!r} has reaction_index {idx} out of range."
                )
            try:
                order = int(getattr(rxns[idx], "order", 1))
            except Exception:
                order = 1
            units[name] = rate_constant_unit(order)
        elif kind == "equilibrium":
            try:
                idx = int(entry.get("equilibrium_index", -1))  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Authoritative step metadata for {name!r} has an invalid equilibrium_index."
                ) from exc
            if not (0 <= idx < len(eqs)):
                raise ValueError(
                    f"Authoritative step metadata for {name!r} has equilibrium_index {idx} out of range."
                )
            eq = eqs[idx]
            if role == "Keq":
                units[name] = "1"
                continue
            try:
                fwd_order = int(round(sum(getattr(eq, "stoich_forward", {}).values())))
            except Exception:
                fwd_order = 1
            try:
                back_order = int(round(sum(getattr(eq, "stoich_back", {}).values())))
            except Exception:
                back_order = 1
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
