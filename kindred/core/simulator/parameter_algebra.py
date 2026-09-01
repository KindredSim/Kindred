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
from collections.abc import Mapping
from typing import Dict, Optional, Set

from kindred.core.equilibrium_rate_authority import (
    EquilibriumRateAuthorityKind,
    authority_fields_from_step_entry,
    effective_equilibrium_keq,
    effective_equilibrium_reverse_rate,
    effective_reverse_rate_from_keq,
    normalize_existing_equilibrium_rate_authority,
    require_step_entry_authored_role_is_editable,
    step_entry_authored_role_is_editable,
)
from kindred.core.kinetics import (
    ConstantRateModel,
    ForwardRateModel,
    ArrheniusRateModel,
    EyringRateModel,
    forward_rate_model_for_equilibrium,
    forward_rate_model_for_reaction,
)
from kindred.core.mechanism_metadata import (
    EquilibriumMetadataKeys,
    MechanismMetadataKeys,
    MechanismMetadataView,
)
from kindred.core.rate_binding import ForwardRateBinding, RateBinding
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra_spec import (
    ParameterAlgebraSpec,
    mechanism_parameter_name_pattern,
)
from kindred.core.simulator.parameter_algebra_eval import evaluate_parameter_algebra
from kindred.core.units import (
    UnitsModel,
    admit_arrhenius_prefactor,
    admit_energy_j_per_mol,
    admit_keq,
    admit_rate,
    kcalmol_to_jmol,
    kjmol_to_jmol,
    rate_constant_unit,
)

from .dsl_build_input import source_coordinate_capabilities
logger = logging.getLogger(__name__)

_MECH_PARAM_RE = mechanism_parameter_name_pattern()
_PARAMETER_ALGEBRA_SPEC_META_KEY = "parameter_algebra_spec"

_PreparedParameterTarget = tuple[
    str,
    tuple[str, int, str, Mapping[str, object]],
]

def _target_parts(
    target: _PreparedParameterTarget,
) -> tuple[str, str, int, str, Mapping[str, object]]:
    route, facts = target
    kind, index, role, entry = facts
    return str(route), str(kind), int(index), str(role), entry


def _admitted_step_entries(
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
) -> tuple[Mapping[str, object], ...]:
    """Return each admitted step entry once, without rescanning the roster."""

    entries: list[Mapping[str, object]] = []
    seen: set[int] = set()
    for target in parameter_target_map.values():
        _route, _kind, _index, _role, entry = _target_parts(target)
        identity = id(entry)
        if identity in seen:
            continue
        seen.add(identity)
        entries.append(entry)
    return tuple(entries)

__all__ = [
    "apply_source_coordinate_values_to_mechanism",
    "apply_parameter_algebra_spec_to_mechanism",
    "parameter_algebra_spec_from_mechanism",
    "read_mechanism_parameter_values",
    "solver_parameter_units_from_mechanism",
]


def read_mechanism_parameter_values(
    mechanism: object,
    *,
    names: Optional[Set[str]] = None,
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
) -> Dict[str, float]:
    wanted = (
        set(parameter_target_map)
        if names is None
        else {str(name) for name in names}
    )
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
    for name, target in parameter_target_map.items():
        if name not in wanted:
            continue
        route, kind, idx, role, entry = _target_parts(target)
        if route != "indexed":
            continue
        if kind == "reaction" and role == "k":
            if 0 <= idx < len(rxns):
                v = _as_float(getattr(rxns[idx], "rate", None))
                if v is not None:
                    out[name] = v
        elif kind == "equilibrium":
            if not (0 <= idx < len(eqs)):
                continue
            eq = eqs[idx]
            if role in {"kf", "kr"}:
                raw_value = getattr(eq, "kf", None) if role == "kf" else getattr(eq, "kr", None)
                v = _as_float(raw_value)
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

    # Physical coordinates are admitted source identities, not independent
    # indexed readouts.  Read their current typed model values in the same
    # units used by the authored DSL so Algebra expressions can constrain the
    # real Constant/Arrhenius/Eyring authority.
    metadata_view = MechanismMetadataView.from_metadata(
        getattr(mechanism, "metadata", {}) or {}
    )
    energy_units = UnitsModel(energy_unit=metadata_view.energy_unit)
    for name, target in parameter_target_map.items():
        if name not in wanted:
            continue
        route, _kind, _index, role, _entry = _target_parts(target)
        if route != "physical" or not source_coordinate_capabilities(role).algebra_target:
            continue
        kind, index, target_role = _target_parts(target)[1:4]
        target_role = str(target_role)
        if kind == "reaction":
            if not 0 <= int(index) < len(rxns):
                continue
            model = forward_rate_model_for_reaction(
                rxns[int(index)],
                default_standard_conc_M=float(metadata_view.standard_conc_M),
            )
        elif kind == "equilibrium":
            if not 0 <= int(index) < len(eqs):
                continue
            model = forward_rate_model_for_equilibrium(
                eqs[int(index)],
                temperature_K=None,
                default_standard_conc_M=float(metadata_view.standard_conc_M),
            )
        else:
            continue
        value: float | None = None
        if target_role == "A" and isinstance(model, ArrheniusRateModel):
            value = float(model.A)
        elif target_role == "Ea" and isinstance(model, ArrheniusRateModel):
            value = float(energy_units.from_jmol(model.Ea_J_per_mol))
        elif target_role == "dG_act" and isinstance(model, EyringRateModel):
            value = float(energy_units.from_jmol(model.dG_act_J_per_mol))
        elif target_role == "kappa" and isinstance(model, EyringRateModel):
            value = float(model.kappa)
        elif target_role == "dG_eq" and kind == "equilibrium":
            raw_dg = dict(getattr(eqs[int(index)], "metadata", {}) or {}).get(
                EquilibriumMetadataKeys.DG_EQ_J_PER_MOL
            )
            try:
                value = energy_units.from_jmol(float(raw_dg))
            except (TypeError, ValueError, OverflowError):
                value = None
        if value is not None and math.isfinite(value):
            out[name] = float(value)
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


def _written_forward_rate(
    *,
    name: str,
    model: ForwardRateModel,
    value: float,
    reference_temperature_K: float,
    require_mutable: bool,
) -> float | RateBinding:
    """Construct one typed forward-rate coordinate without mutating ``current``."""

    if isinstance(model, ConstantRateModel):
        if not require_mutable:
            return float(value)
        return RateBinding(name=str(name), value=float(value))

    binding = ForwardRateBinding(
        name=str(name),
        model=model,
        reference_temperature_K=float(reference_temperature_K),
    )
    binding.set(value)
    return binding


def _admit_mechanism_parameter_value(
    mechanism: object,
    name: str,
    value: float,
    *,
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
) -> float:
    """Admit one indexed mechanism coordinate in its typed context."""

    target = parameter_target_map.get(str(name))
    if target is None:
        raise DSLError(f"Unsupported parameter name {name!r} for parameter algebra")
    route, kind, idx, role, _entry = _target_parts(target)
    if route != "indexed":
        raise DSLError(f"Unsupported parameter name {name!r} for parameter algebra")
    try:
        if role == "Keq":
            return admit_keq(value, field=name)

        allow_zero = False
        if kind == "reaction" and role == "k":
            rxns = getattr(mechanism, "reactions", []) or []
            if not 0 <= idx < len(rxns):
                raise DSLError(f"Unknown parameter {name!r} (reaction index out of range)")
            metadata = MechanismMetadataView.from_metadata(
                getattr(mechanism, "metadata", {}) or {}
            )
            model = forward_rate_model_for_reaction(
                rxns[idx],
                default_standard_conc_M=float(metadata.standard_conc_M),
            )
            allow_zero = isinstance(model, ConstantRateModel)
        elif kind == "equilibrium" and role == "kf":
            eqs = getattr(mechanism, "equilibria", []) or []
            if not 0 <= idx < len(eqs):
                raise DSLError(f"Unknown parameter {name!r} (equilibrium index out of range)")
            eq = eqs[idx]
            metadata = MechanismMetadataView.from_metadata(
                getattr(mechanism, "metadata", {}) or {}
            )
            model = forward_rate_model_for_equilibrium(
                eq,
                temperature_K=float(metadata.temperature_K),
                default_standard_conc_M=float(metadata.standard_conc_M),
            )
            authority = normalize_existing_equilibrium_rate_authority(eq)
            allow_zero = (
                isinstance(model, ConstantRateModel)
                and authority.kind == EquilibriumRateAuthorityKind.KEQ
            )
        elif kind != "equilibrium" or role != "kr":
            raise DSLError(f"Unsupported parameter target for {name!r}")

        return admit_rate(value, field=name, allow_zero=allow_zero)
    except ValueError as exc:
        raise DSLError(str(exc)) from exc


def _prepare_mechanism_param_replacement(
    mechanism: object,
    name: str,
    value: float,
    *,
    require_mutable: bool,
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
) -> object:
    """Construct one complete indexed replacement without publishing it."""

    numeric_value = _admit_mechanism_parameter_value(
        mechanism,
        name,
        float(value),
        parameter_target_map=parameter_target_map,
    )
    target = parameter_target_map.get(str(name))
    if target is None:
        raise DSLError(f"Unsupported parameter name {name!r} for parameter algebra")
    route, kind, idx, role, entry = _target_parts(target)
    if route != "indexed":
        raise DSLError(f"Unsupported parameter name {name!r} for parameter algebra")
    metadata = MechanismMetadataView.from_metadata(
        getattr(mechanism, "metadata", {}) or {}
    )
    if kind == "reaction" and role == "k":
        rxns = getattr(mechanism, "reactions", []) or []
        if not 0 <= idx < len(rxns):
            raise DSLError(f"Unknown parameter {name!r} (reaction index out of range)")
        reaction = rxns[idx]
        model = forward_rate_model_for_reaction(
            reaction,
            default_standard_conc_M=float(metadata.standard_conc_M),
        )
        return replace(
            reaction,
            rate=_written_forward_rate(
                name=name,
                model=model,
                value=numeric_value,
                reference_temperature_K=float(metadata.temperature_K),
                require_mutable=require_mutable,
            ),
        )

    if kind != "equilibrium":
        raise DSLError(f"Unsupported parameter target for {name!r}")
    eqs = getattr(mechanism, "equilibria", []) or []
    if not 0 <= idx < len(eqs):
        raise DSLError(f"Unknown parameter {name!r} (equilibrium index out of range)")
    require_step_entry_authored_role_is_editable(
        entry,
        role,
        parameter_name=name,
    )
    equilibrium = eqs[idx]
    if role in {"kf", "kr"}:
        if role == "kf":
            model = forward_rate_model_for_equilibrium(
                equilibrium,
                temperature_K=float(metadata.temperature_K),
                default_standard_conc_M=float(metadata.standard_conc_M),
            )
            return replace(
                equilibrium,
                kf=_written_forward_rate(
                    name=name,
                    model=model,
                    value=numeric_value,
                    reference_temperature_K=float(metadata.temperature_K),
                    require_mutable=require_mutable,
                ),
            )
        updated = (
            RateBinding(name=str(name), value=numeric_value)
            if require_mutable
            else numeric_value
        )
        return replace(equilibrium, kr=updated)
    if role == "Keq":
        equilibrium_metadata = dict(getattr(equilibrium, "metadata", {}) or {})
        equilibrium_metadata[EquilibriumMetadataKeys.KEQ_INPUT] = (
            RateBinding(name=str(name), value=numeric_value)
            if require_mutable
            else numeric_value
        )
        return replace(equilibrium, metadata=equilibrium_metadata)
    raise DSLError(f"Unsupported parameter target for {name!r}")


def _admit_source_coordinate_value(
    mechanism: object,
    name: str,
    value: float,
    *,
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
) -> float:
    target = parameter_target_map.get(str(name))
    if target is None:
        raise DSLError(f"Unsupported source coordinate {name!r} for parameter algebra")
    route, _kind, _index, role, _entry = _target_parts(target)
    if route != "physical":
        raise DSLError(f"Unsupported source coordinate {name!r} for parameter algebra")
    capabilities = source_coordinate_capabilities(str(role))
    if not capabilities.algebra_target:
        raise DSLError(f"Source coordinate {name!r} is not an Algebra target")
    try:
        if role == "A":
            return admit_arrhenius_prefactor(value, field=name)
        if role in {"Ea", "dG_act", "dG_eq"}:
            energy_unit = MechanismMetadataView.from_metadata(
                getattr(mechanism, "metadata", {}) or {}
            ).energy_unit
            conversion = {
                "J/mol": lambda raw: float(raw),
                "kJ/mol": kjmol_to_jmol,
                "kcal/mol": kcalmol_to_jmol,
            }.get(energy_unit)
            if conversion is None:
                raise ValueError(f"Unsupported energy_unit: {energy_unit!r}")
            return admit_energy_j_per_mol(
                conversion(value),
                field=name,
            )
        if role == "kappa":
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            return numeric
    except (TypeError, ValueError, OverflowError) as exc:
        raise DSLError(str(exc)) from exc
    raise DSLError(f"Unsupported source coordinate role {role!r}")


def apply_source_coordinate_values_to_mechanism(
    mechanism: object,
    values: Mapping[str, float],
    *,
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
) -> None:
    """Apply admitted physical source coordinates in grouped model updates.

    All coordinates are resolved and admitted before the first mechanism list
    assignment.  Coordinates belonging to one step are then folded into one
    typed model replacement, so a coupled ``A``/``Ea`` or ``dG_act``/``kappa``
    update cannot leave a stale model field behind.
    """

    grouped: dict[tuple[str, int], dict[str, float]] = {}
    grouped_entries: dict[tuple[str, int], Mapping[str, object]] = {}

    # Admission is deliberately a complete first pass.  The mapping is
    # trusted output from the retained Algebra spec, so no DTO or rollback is
    # needed between this owner and its caller.
    for raw_name, raw_value in values.items():
        name = str(raw_name)
        target = parameter_target_map.get(name)
        if target is None:
            raise DSLError(
                f"Unsupported source coordinate {name!r} for parameter algebra"
            )
        route, kind, index, role, entry = _target_parts(target)
        if route != "physical":
            raise DSLError(
                f"Unsupported source coordinate {name!r} for parameter algebra"
            )
        capabilities = source_coordinate_capabilities(str(role))
        if not capabilities.grouped_model_target:
            raise DSLError(
                f"Source coordinate {name!r} is not a physical model coordinate"
            )
        if bool(entry.get("locked", False)):
            raise DSLError(f"Source coordinate {name!r} is locked and not writable")
        if kind == "equilibrium" and role == "dG_eq":
            if not step_entry_authored_role_is_editable(entry, role):
                raise DSLError(f"Source coordinate {name!r} is not writable")
        admitted = _admit_source_coordinate_value(
            mechanism,
            name,
            raw_value,
            parameter_target_map=parameter_target_map,
        )
        key = (str(kind), int(index))
        role_values = grouped.setdefault(key, {})
        previous_entry = grouped_entries.setdefault(key, entry)
        if previous_entry is not entry:
            raise DSLError(
                f"Conflicting admitted metadata for source coordinate {name!r}."
            )
        if role in role_values:
            raise DSLError(f"Duplicate source coordinate {name!r}.")
        role_values[str(role)] = float(admitted)

    if not grouped:
        return

    metadata = MechanismMetadataView.from_metadata(
        getattr(mechanism, "metadata", {}) or {}
    )
    reactions = getattr(mechanism, "reactions", []) or []
    equilibria = getattr(mechanism, "equilibria", []) or []
    admitted_models: dict[tuple[str, int], tuple[object, ForwardRateModel]] = {}
    admitted_entries: dict[tuple[str, int], Mapping[str, object]] = {}
    # Complete family/index admission before the first list replacement.  The
    # owner remains non-transactional, but an invalid later coordinate cannot
    # leave an earlier coordinate from the same admitted mapping applied.
    reaction_replacements: dict[int, object] = {}
    equilibrium_replacements: dict[int, object] = {}
    for (kind, index), updates in grouped.items():
        if kind == "reaction":
            if not 0 <= index < len(reactions):
                raise DSLError(
                    f"Unknown source coordinate step {index + 1} (reaction index out of range)"
                )
            reaction = reactions[index]
            entry = grouped_entries[(kind, index)]
            model = forward_rate_model_for_reaction(
                reaction,
                default_standard_conc_M=float(metadata.standard_conc_M),
            )
            for role in updates:
                if role in {"A", "Ea"} and not isinstance(model, ArrheniusRateModel):
                    raise DSLError(
                        f"Source coordinate {role}{index + 1!s} is not backed by Arrhenius authority"
                    )
                if role in {"dG_act", "kappa"} and not isinstance(model, EyringRateModel):
                    raise DSLError(
                        f"Source coordinate {role}{index + 1!s} is not backed by Eyring authority"
                    )
                if role not in {"A", "Ea", "dG_act", "kappa"}:
                    raise DSLError(
                        f"Source coordinate role {role!r} is not a reaction model field"
                    )
            admitted_models[(kind, index)] = (reaction, model)
            admitted_entries[(kind, index)] = entry
            continue
        if kind != "equilibrium" or not 0 <= index < len(equilibria):
            raise DSLError(
                f"Unknown source coordinate step {index + 1} (equilibrium index out of range)"
            )
        equilibrium = equilibria[index]
        entry = grouped_entries[(kind, index)]
        model_updates = {
            role: value
            for role, value in updates.items()
            if role in {"dG_act", "kappa"}
        }
        if model_updates:
            model = forward_rate_model_for_equilibrium(
                equilibrium,
                temperature_K=None,
                default_standard_conc_M=float(metadata.standard_conc_M),
            )
            if not isinstance(model, EyringRateModel):
                raise DSLError(
                    f"Source coordinate step {index + 1} is not backed by Eyring authority"
                )
            admitted_models[(kind, index)] = (equilibrium, model)
            admitted_entries[(kind, index)] = entry
        for role in updates:
            if role not in {"dG_act", "kappa", "dG_eq"}:
                raise DSLError(
                    f"Source coordinate role {role!r} is not an equilibrium model field"
                )

    for (kind, index), updates in grouped.items():
        if kind == "reaction":
            reaction, model = admitted_models[(kind, index)]
            entry = admitted_entries[(kind, index)]
            model_fields: dict[str, float] = {}
            for role, value in updates.items():
                if role == "A":
                    model_fields["A"] = value
                elif role == "Ea":
                    model_fields["Ea_J_per_mol"] = value
                elif role == "dG_act":
                    model_fields["dG_act_J_per_mol"] = value
                elif role == "kappa":
                    model_fields["kappa"] = value
            updated_model = replace(model, **model_fields) if model_fields else model
            if updated_model == model:
                continue
            reaction_replacements[index] = replace(
                reaction,
                rate=ForwardRateBinding(
                    name=f"k{int(entry.get('step_index', index + 1))}",
                    model=updated_model,
                    reference_temperature_K=float(metadata.temperature_K),
                ),
            )
            continue

        equilibrium, model = admitted_models.get((kind, index), (equilibria[index], None))
        entry = admitted_entries.get((kind, index), {})
        equilibrium_metadata = dict(getattr(equilibrium, "metadata", {}) or {})
        model_updates = {
            role: value
            for role, value in updates.items()
            if role in {"dG_act", "kappa"}
        }
        updated_model = None
        if model_updates:
            assert isinstance(model, EyringRateModel)
            model_fields = {
                field_name: model_updates[role]
                for role, field_name in (
                    ("dG_act", "dG_act_J_per_mol"),
                    ("kappa", "kappa"),
                )
                if role in model_updates
            }
            updated_model = replace(model, **model_fields)
        if "dG_eq" in updates:
            equilibrium_metadata[EquilibriumMetadataKeys.DG_EQ_J_PER_MOL] = float(
                updates["dG_eq"]
            )
        equilibrium_updates: dict[str, object] = {
            "metadata": equilibrium_metadata,
        }
        if updated_model is not None:
            equilibrium_updates["kf"] = ForwardRateBinding(
                name=f"kf{int(entry.get('step_index', index + 1))}",
                model=updated_model,
                reference_temperature_K=float(metadata.temperature_K),
            )
        equilibrium_replacements[index] = replace(equilibrium, **equilibrium_updates)

    # Every candidate replacement has been constructed and admitted above;
    # only now publish the complete replacement set to the mechanism lists.
    for index, replacement in reaction_replacements.items():
        reactions[index] = replacement
    for index, replacement in equilibrium_replacements.items():
        equilibria[index] = replacement


def _active_equilibrium_keq_names(
    mechanism: object,
    spec: ParameterAlgebraSpec,
    *,
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
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


def _validate_parameter_algebra_editable_targets(
    mechanism: object,
    spec: ParameterAlgebraSpec,
    *,
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
) -> None:
    for stmt in spec.param_statements or ():
        name = str(stmt.name)
        target = parameter_target_map.get(name)
        if target is None:
            continue
        route, kind, _idx, role, entry = _target_parts(target)
        if route == "physical":
            if bool(entry.get("locked", False)):
                raise DSLError(f"Source coordinate {name!r} is locked and not writable")
            if kind == "equilibrium" and role == "dG_eq":
                require_step_entry_authored_role_is_editable(
                    entry,
                    role,
                    parameter_name=name,
                )
            continue
        if route != "indexed":
            continue
        if kind == "reaction" and role == "k":
            source_coordinates = entry.get("source_coordinates") or {}
            if isinstance(source_coordinates, Mapping) and any(
                str(source_role) in {"A", "Ea", "dG_act"}
                for source_role in source_coordinates
            ):
                raise DSLError(
                    f"mechanism parameter {name!r} is a derived rate readout and cannot be used as write authority."
                )
        if kind == "equilibrium":
            require_step_entry_authored_role_is_editable(
                entry,
                role,
                parameter_name=name,
            )


def _apply_equilibrium_Keq_constraints_to_values(
    mechanism: object,
    base_values: Dict[str, float],
    *,
    active_keq_names: Set[str],
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
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
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
    scalar_parameter_overrides: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    meta = getattr(mechanism, "metadata", None)
    scalar_overrides: dict[str, float] = {}
    if scalar_parameter_overrides:
        independent_scalars = {
            str(name)
            for name in spec.param_assignment_names()
            if spec.parameter_assignment_kind(str(name)) == "independent scalar"
        }
        for raw_name, raw_value in scalar_parameter_overrides.items():
            name = str(raw_name)
            if name not in independent_scalars:
                raise DSLError(
                    f"Unsupported scalar parameter override {name!r}."
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise DSLError(
                    f"Scalar parameter override {name!r} must be finite."
                ) from exc
            if not math.isfinite(value):
                raise DSLError(
                    f"Scalar parameter override {name!r} must be finite."
                )
            scalar_overrides[name] = value
    _validate_parameter_algebra_editable_targets(
        mechanism,
        spec,
        parameter_target_map=parameter_target_map,
    )
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
        )
    )
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

    physical_updates: dict[str, float] = {}
    indexed_replacements: dict[tuple[str, int], tuple[object, str, float]] = {}
    for name, value in tuple(derived.items()):
        if name not in spec.mechanism_param_names:
            continue
        target = parameter_target_map.get(str(name))
        if target is not None and _target_parts(target)[0] == "indexed":
            _route, kind, index, role, _entry = _target_parts(target)
            admitted = _admit_mechanism_parameter_value(
                mechanism,
                name,
                float(value),
                parameter_target_map=parameter_target_map,
            )
            prepared_replacement = _prepare_mechanism_param_replacement(
                mechanism,
                name,
                admitted,
                require_mutable=require_mutable,
                parameter_target_map=parameter_target_map,
            )
            replacement_key = (kind, index)
            previous = indexed_replacements.get(replacement_key)
            if previous is not None and kind == "equilibrium":
                prior_replacement = previous[0]
                if role == "kf":
                    prepared_replacement = replace(
                        prior_replacement,
                        kf=prepared_replacement.kf,
                    )
                elif role == "kr":
                    prepared_replacement = replace(
                        prior_replacement,
                        kr=prepared_replacement.kr,
                    )
                elif role == "Keq":
                    prepared_replacement = replace(
                        prior_replacement,
                        metadata=prepared_replacement.metadata,
                    )
            indexed_replacements[replacement_key] = (
                prepared_replacement,
                name,
                admitted,
            )
            continue
        if target is None or _target_parts(target)[0] != "physical":
            raise DSLError(
                f"Unsupported mechanism parameter {name!r} for parameter algebra"
            )
        _route, _kind, _index, role, _entry = _target_parts(target)
        if not source_coordinate_capabilities(str(role)).grouped_model_target:
            raise DSLError(
                f"Source coordinate {name!r} is not a physical model coordinate"
            )
        physical_updates[name] = float(value)

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

    # Physical coordinates are admitted and committed through their grouped
    # owner once per step/model; indexed rate bindings retain their direct
    # canonical path below.
    if physical_updates:
        apply_source_coordinate_values_to_mechanism(
            mechanism,
            physical_updates,
            parameter_target_map=parameter_target_map,
        )
    # The indexed replacements were fully admitted and constructed before the
    # physical publisher ran; publish them directly after that publisher.
    for (kind, index), (replacement, _name, _admitted) in indexed_replacements.items():
        if kind == "reaction":
            mechanism.reactions[index] = replacement
        elif kind == "equilibrium":
            mechanism.equilibria[index] = replacement

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

    constrained: Dict[str, Dict[str, object]] = {}
    for stmt in spec.param_statements:
        is_mech = _MECH_PARAM_RE.match(stmt.name) is not None
        if not is_mech:
            # Base declaration iff expression has no parameter dependencies.
            # Reflect base_values into storage.
            cur_val = base_values.get(stmt.name)
            if cur_val is None:
                cur_val = float("nan")
            if require_mutable:
                b = scalar_bindings.get(stmt.name)
                if b is None:
                    b = RateBinding(name=str(stmt.name), value=float(cur_val))
                    scalar_bindings[stmt.name] = b
                elif stmt.name in scalar_overrides:
                    # Publish a fresh binding after all mechanism replacements
                    # have passed admission; do not mutate the old binding
                    # while evaluating a potentially failing candidate.
                    b = RateBinding(
                        name=str(stmt.name),
                        value=float(scalar_overrides[stmt.name]),
                    )
                    scalar_bindings[stmt.name] = b
                # Do not overwrite base scalar params here; only derived scalar
                # params are present in ``derived``.
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
        else:
            constrained[stmt.name] = {
                "line": stmt.line_number,
                "expr": stmt.expr_src,
                "constraint_reason": "algebra",
            }

    meta["constrained_params"] = constrained

    return derived


def solver_parameter_units_from_mechanism(
    mechanism: object,
    *,
    parameter_target_map: Mapping[str, _PreparedParameterTarget],
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
        route, kind, idx, role, entry = _target_parts(target)
        if route != "indexed":
            continue
        if kind == "reaction" and role == "k":
            if not (0 <= idx < len(rxns)):
                raise ValueError(
                    f"Authoritative step metadata for {name!r} has reaction_index {idx} out of range."
                )
            order = float(getattr(rxns[idx], "order"))
            units[name] = rate_constant_unit(order)
        elif kind == "equilibrium":
            if not (0 <= idx < len(eqs)):
                raise ValueError(
                    f"Authoritative step metadata for {name!r} has equilibrium_index {idx} out of range."
                )
            eq = eqs[idx]
            if role == "Keq":
                nf = math.fsum(
                    float(v)
                    for v in dict(getattr(eq, "stoich_forward", {}) or {}).values()
                )
                nr = math.fsum(
                    float(v)
                    for v in dict(getattr(eq, "stoich_back", {}) or {}).values()
                )
                from kindred.core.units import equilibrium_constant_unit
                units[name] = equilibrium_constant_unit(nr - nf)
                continue
            fwd_order = math.fsum(
                float(value)
                for value in getattr(eq, "stoich_forward").values()
            )
            back_order = math.fsum(
                float(value)
                for value in getattr(eq, "stoich_back").values()
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
