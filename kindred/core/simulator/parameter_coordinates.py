"""Authoritative mechanism parameter coordinates and grouped composition.

The parser-admitted namespace contains two kinds of mechanism coordinates:
indexed readouts (``k1``, ``kf1``, ``kr1``, ``Keq1``) and physical source
coordinates (``A1``, ``Ea1``, ``dG_act1``, ``dG_eq1``, ``kappa1``).  This
module owns their target map and the one grouped writer used by Algebra and
execution preparation.  A step is composed from its current values and is
published once, so coupled coordinates cannot overwrite one another through
stale intermediate replacements.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
import math
from typing import Any, TypeAlias

from kindred.core.equilibrium_rate_authority import (
    EquilibriumRateAuthorityKind,
    effective_equilibrium_keq,
    effective_equilibrium_reverse_rate,
    normalize_existing_equilibrium_rate_authority,
    require_step_entry_authored_role_is_editable,
)
from kindred.core.kinetics import (
    ArrheniusRateModel,
    ConstantRateModel,
    EyringRateModel,
    ForwardRateModel,
    forward_rate_model_for_equilibrium,
    forward_rate_model_for_reaction,
)
from kindred.core.mechanism_metadata import (
    EquilibriumMetadataKeys,
    MechanismMetadataView,
)
from kindred.core.rate_binding import ForwardRateBinding, RateBinding
from kindred.core.simulator.dsl_build_input import source_coordinate_capabilities
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism
from kindred.core.simulator.step_indexing import get_step_index_map
from kindred.core.units import (
    admit_arrhenius_prefactor,
    admit_energy_j_per_mol,
    admit_keq,
    admit_rate,
    kcalmol_to_jmol,
    kjmol_to_jmol,
    UnitsModel,
)

ParameterTarget: TypeAlias = tuple[
    str,
    str,
    int,
    str,
    Mapping[str, object],
]

__all__ = [
    "ParameterTarget",
    "build_parameter_target_map",
    "apply_mechanism_coordinate_values",
    "read_mechanism_parameter_values",
    "admit_physical_coordinate_value",
]


def build_parameter_target_map(
    mechanism: Any,
) -> dict[str, ParameterTarget]:
    """Build the parser-admitted indexed/physical coordinate target map."""

    namespace = build_namespace_from_mechanism(mechanism)
    step_roster = get_step_index_map(mechanism)
    targets: dict[str, ParameterTarget] = {}

    for item in namespace.ordered_items:
        source_index = item.source_index
        if source_index is None:
            continue
        entry = step_roster[source_index]
        kind = str(entry["kind"])
        index_key = "reaction_index" if kind == "reaction" else "equilibrium_index"
        index = int(entry[index_key])
        role = str(item.info.role or "")
        if not role:
            continue
        targets[str(item.canonical_name)] = (
            "indexed",
            kind,
            index,
            role,
            entry,
        )

    for coordinate in namespace.source_coordinates:
        name = str(coordinate.canonical_name)
        if name in targets:
            continue
        source_index = coordinate.source_index
        if source_index is None:
            continue
        entry = step_roster[source_index]
        kind = str(entry["kind"])
        index_key = "reaction_index" if kind == "reaction" else "equilibrium_index"
        index = int(entry[index_key])
        targets[name] = (
            "physical",
            kind,
            index,
            str(coordinate.role),
            entry,
        )
    return targets


def admit_physical_coordinate_value(
    value: object,
    *,
    role: str,
    energy_unit: str,
    field: str,
) -> float:
    """Admit a physical coordinate, lowering authored energies to J/mol."""
    if role == "A":
        return admit_arrhenius_prefactor(value, field=field)
    if role == "kappa":
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be positive and finite") from exc
        if type(value) is bool or not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{field} must be positive and finite")
        return numeric
    if role not in {"Ea", "dG_act", "dG_eq"}:
        raise ValueError(f"Unsupported source coordinate role {role!r}")
    conversion = {
        "J/mol": lambda raw: float(raw),
        "kJ/mol": kjmol_to_jmol,
        "kcal/mol": kcalmol_to_jmol,
    }.get(energy_unit)
    if conversion is None:
        raise ValueError(f"Unsupported energy_unit: {energy_unit!r}")
    return admit_energy_j_per_mol(conversion(value), field=field)


def _admit_coordinate_value(
    mechanism: object,
    name: str,
    value: object,
    *,
    route: str,
    kind: str,
    index: int,
    role: str,
) -> float:
    """Admit one value in its target's concrete scientific context."""

    if route == "physical":
        return admit_physical_coordinate_value(
            value, role=role, field=name,
            energy_unit=MechanismMetadataView.from_metadata(
                getattr(mechanism, "metadata", {}) or {}
            ).energy_unit,
        )

    if role == "Keq":
        return admit_keq(value, field=name)

    if kind == "reaction" and role == "k":
        reactions = getattr(mechanism, "reactions", ()) or ()
        metadata = MechanismMetadataView.from_metadata(
            getattr(mechanism, "metadata", {}) or {}
        )
        model = forward_rate_model_for_reaction(
            reactions[index],
            default_standard_conc_M=float(metadata.standard_conc_M),
        )
        return admit_rate(
            value,
            field=name,
            allow_zero=isinstance(model, ConstantRateModel),
        )

    if kind == "equilibrium" and role in {"kf", "kr"}:
        equilibria = getattr(mechanism, "equilibria", ()) or ()
        equilibrium = equilibria[index]
        metadata = MechanismMetadataView.from_metadata(
            getattr(mechanism, "metadata", {}) or {}
        )
        model = forward_rate_model_for_equilibrium(
            equilibrium,
            temperature_K=float(metadata.temperature_K),
            default_standard_conc_M=float(metadata.standard_conc_M),
        )
        authority = normalize_existing_equilibrium_rate_authority(equilibrium)
        allow_zero = (
            role == "kf"
            and isinstance(model, ConstantRateModel)
            and authority.kind == EquilibriumRateAuthorityKind.KEQ
        )
        return admit_rate(value, field=name, allow_zero=allow_zero)

    raise ValueError(f"Unsupported parameter target for {name!r}")


def _written_forward_rate(
    *,
    name: str,
    model: ForwardRateModel,
    value: float | None,
    reference_temperature_K: float,
    require_mutable: bool,
) -> object:
    """Write a complete forward model while retaining mutable binding policy."""

    if value is not None and isinstance(model, ConstantRateModel):
        if not require_mutable:
            return float(value)
        return RateBinding(name=str(name), value=float(value))
    return ForwardRateBinding(
        name=str(name),
        model=model,
        reference_temperature_K=float(reference_temperature_K),
    )


def _admit_coordinate_groups(
    mechanism: object,
    values: Mapping[str, object],
    *,
    parameter_target_map: Mapping[str, ParameterTarget],
) -> dict[tuple[str, int], tuple[Mapping[str, object], dict[str, float]]]:
    grouped: dict[
        tuple[str, int],
        tuple[Mapping[str, object], dict[str, float]],
    ] = {}

    for raw_name, raw_value in values.items():
        name = str(raw_name)
        target = parameter_target_map.get(name)
        if target is None:
            raise DSLError(f"Unsupported mechanism coordinate {name!r}")
        route, kind, index, role, entry = target
        if route == "physical" and not source_coordinate_capabilities(
            role
        ).grouped_model_target:
            raise DSLError(f"Source coordinate {name!r} is not a model coordinate")
        if bool(entry.get("locked", False)):
            raise DSLError(f"Mechanism coordinate {name!r} is locked and not writable")
        if kind == "equilibrium":
            try:
                require_step_entry_authored_role_is_editable(
                    entry,
                    role,
                    parameter_name=name,
                )
            except ValueError as exc:
                # Indexed equilibrium roles and dG_eq all carry one normalized
                # editability decision.  The same rule is not needed for
                # reaction coordinates, whose source map is the authority.
                if route == "indexed" or role == "dG_eq":
                    raise DSLError(str(exc)) from exc
        try:
            admitted = _admit_coordinate_value(
                mechanism,
                name,
                raw_value,
                route=route,
                kind=kind,
                index=index,
                role=role,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise DSLError(str(exc)) from exc
        key = (kind, index)
        if key not in grouped:
            grouped[key] = (entry, {})
        grouped[key][1][role] = float(admitted)
    return grouped


def _compose_reaction(
    reaction: object,
    *,
    updates: Mapping[str, float],
    step_name: int,
    metadata: MechanismMetadataView,
    require_mutable: bool,
) -> object:
    model = forward_rate_model_for_reaction(
        reaction,
        default_standard_conc_M=float(metadata.standard_conc_M),
    )
    if "k" in updates:
        if not isinstance(model, ConstantRateModel):
            raise DSLError(f"Indexed k{step_name} is not a Constant authority")
        model = ConstantRateModel(k=updates["k"])
    model_fields: dict[str, float] = {}
    if "A" in updates:
        if not isinstance(model, ArrheniusRateModel):
            raise DSLError(f"A{step_name} is not backed by Arrhenius authority")
        model_fields["A"] = updates["A"]
    if "Ea" in updates:
        if not isinstance(model, ArrheniusRateModel):
            raise DSLError(f"Ea{step_name} is not backed by Arrhenius authority")
        model_fields["Ea_J_per_mol"] = updates["Ea"]
    if "dG_act" in updates:
        if not isinstance(model, EyringRateModel):
            raise DSLError(f"dG_act{step_name} is not backed by Eyring authority")
        model_fields["dG_act_J_per_mol"] = updates["dG_act"]
    if "kappa" in updates:
        if not isinstance(model, EyringRateModel):
            raise DSLError(f"kappa{step_name} is not backed by Eyring authority")
        model_fields["kappa"] = updates["kappa"]
    if model_fields:
        model = replace(model, **model_fields)
    return replace(
        reaction,
        rate=_written_forward_rate(
            name=f"k{step_name}",
            model=model,
            value=updates.get("k"),
            reference_temperature_K=float(metadata.temperature_K),
            require_mutable=require_mutable,
        ),
    )


def _compose_equilibrium(
    equilibrium: object,
    *,
    updates: Mapping[str, float],
    step_name: int,
    metadata: MechanismMetadataView,
    require_mutable: bool,
) -> object:
    model: ForwardRateModel | None = None
    if any(role in updates for role in ("A", "Ea", "dG_act", "kappa")):
        model = forward_rate_model_for_equilibrium(
            equilibrium,
            temperature_K=None,
            default_standard_conc_M=float(metadata.standard_conc_M),
        )
        if any(role in updates for role in ("A", "Ea")) and not isinstance(
            model, ArrheniusRateModel
        ):
            raise DSLError(
                f"Equilibrium step {step_name} is not backed by Arrhenius authority"
            )
        if any(role in updates for role in ("dG_act", "kappa")) and not isinstance(
            model, EyringRateModel
        ):
            raise DSLError(
                f"Equilibrium step {step_name} is not backed by Eyring authority"
            )
        model_fields = {
            field: updates[role]
            for role, field in (
                ("A", "A"),
                ("Ea", "Ea_J_per_mol"),
                ("dG_act", "dG_act_J_per_mol"),
                ("kappa", "kappa"),
            )
            if role in updates
        }
        model = replace(model, **model_fields)
    elif "kf" in updates:
        model = forward_rate_model_for_equilibrium(
            equilibrium,
            temperature_K=None,
            default_standard_conc_M=float(metadata.standard_conc_M),
        )

    equilibrium_metadata = dict(getattr(equilibrium, "metadata", {}) or {})
    if "dG_eq" in updates:
        equilibrium_metadata[EquilibriumMetadataKeys.DG_EQ_J_PER_MOL] = updates[
            "dG_eq"
        ]
    if "Keq" in updates:
        equilibrium_metadata[EquilibriumMetadataKeys.KEQ_INPUT] = (
            RateBinding(name=f"Keq{step_name}", value=updates["Keq"])
            if require_mutable
            else updates["Keq"]
        )

    replacement_fields: dict[str, object] = {"metadata": equilibrium_metadata}
    if model is not None:
        replacement_fields["kf"] = ForwardRateBinding(
            name=f"kf{step_name}",
            model=model,
            reference_temperature_K=float(metadata.temperature_K),
        )
    if "kf" in updates:
        if model is None:
            raise DSLError(f"Indexed kf{step_name} has no forward authority")
        if not isinstance(model, ConstantRateModel):
            raise DSLError(
                f"Indexed kf{step_name} cannot replace a non-Constant model"
            )
        replacement_fields["kf"] = _written_forward_rate(
            name=f"kf{step_name}",
            model=model,
            value=updates["kf"],
            reference_temperature_K=float(metadata.temperature_K),
            require_mutable=require_mutable,
        )
    if "kr" in updates:
        replacement_fields["kr"] = (
            RateBinding(name=f"kr{step_name}", value=updates["kr"])
            if require_mutable
            else updates["kr"]
        )
    else:
        thermo_authority = (
            equilibrium_metadata.get(EquilibriumMetadataKeys.KEQ_INPUT) is not None
            or equilibrium_metadata.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL)
            is not None
        )
        replacement_fields["kr"] = (
            None if thermo_authority else getattr(equilibrium, "kr", None)
        )
    replacement_fields["Keq"] = None
    return replace(equilibrium, **replacement_fields)


def _compose_coordinate_groups(
    mechanism: object,
    grouped: Mapping[
        tuple[str, int],
        tuple[Mapping[str, object], Mapping[str, float]],
    ],
    *,
    require_mutable: bool,
) -> Iterator[tuple[tuple[str, int], object]]:
    metadata = MechanismMetadataView.from_metadata(
        getattr(mechanism, "metadata", {}) or {}
    )
    reactions = getattr(mechanism, "reactions", ()) or ()
    equilibria = getattr(mechanism, "equilibria", ()) or ()
    for (kind, index), (entry, updates) in grouped.items():
        step_name = int(entry["step_index"])
        if kind == "reaction":
            yield (
                (kind, index),
                _compose_reaction(
                    reactions[index],
                    updates=updates,
                    step_name=step_name,
                    metadata=metadata,
                    require_mutable=require_mutable,
                ),
            )
            continue
        yield (
            (kind, index),
            _compose_equilibrium(
                equilibria[index],
                updates=updates,
                step_name=step_name,
                metadata=metadata,
                require_mutable=require_mutable,
            ),
        )


def apply_mechanism_coordinate_values(
    mechanism: object,
    values: Mapping[str, object],
    *,
    parameter_target_map: Mapping[str, ParameterTarget],
    require_mutable: bool,
) -> None:
    """Compose each complete changed step and publish it once."""

    grouped = _admit_coordinate_groups(
        mechanism,
        values,
        parameter_target_map=parameter_target_map,
    )
    if not grouped:
        return
    reactions = getattr(mechanism, "reactions", ()) or ()
    equilibria = getattr(mechanism, "equilibria", ()) or ()
    replacements = tuple(
        _compose_coordinate_groups(
            mechanism,
            grouped,
            require_mutable=require_mutable,
        )
    )
    for (kind, index), replacement in replacements:
        if kind == "reaction":
            reactions[index] = replacement
        else:
            equilibria[index] = replacement


def read_mechanism_parameter_values(
    mechanism: object,
    *,
    names: set[str] | None = None,
    parameter_target_map: Mapping[str, ParameterTarget],
    coordinate_overrides: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Read current or candidate mechanism-coordinate values without publication."""

    candidates: dict[tuple[str, int], object] = {}
    if coordinate_overrides:
        candidates = dict(
            _compose_coordinate_groups(
                mechanism,
                _admit_coordinate_groups(
                    mechanism,
                    coordinate_overrides,
                    parameter_target_map=parameter_target_map,
                ),
                require_mutable=False,
            )
        )

    wanted = set(parameter_target_map) if names is None else names
    out: dict[str, float] = {}
    reactions = getattr(mechanism, "reactions", ()) or ()
    equilibria = getattr(mechanism, "equilibria", ()) or ()
    metadata = MechanismMetadataView.from_metadata(
        getattr(mechanism, "metadata", {}) or {}
    )
    energy_units = UnitsModel(energy_unit=metadata.energy_unit)

    def _value(raw: object) -> float | None:
        if raw is None:
            return None
        try:
            return float(raw()) if callable(raw) else float(raw)
        except (TypeError, ValueError, OverflowError):
            return None

    for name, target in parameter_target_map.items():
        if name not in wanted:
            continue
        route, kind, index, role, _entry = target
        if kind == "reaction":
            step = candidates.get((kind, index), reactions[index])
        else:
            step = candidates.get((kind, index), equilibria[index])

        value: float | None = None
        if route == "indexed":
            if kind == "reaction" and role == "k":
                value = _value(getattr(step, "rate", None))
            elif kind == "equilibrium" and role == "kf":
                value = _value(getattr(step, "kf", None))
            elif kind == "equilibrium" and role == "kr":
                authority = normalize_existing_equilibrium_rate_authority(step)
                if authority.kind == EquilibriumRateAuthorityKind.KEQ:
                    value = effective_equilibrium_reverse_rate(
                        step,
                        temperature_K=float(metadata.temperature_K),
                    )
                else:
                    value = _value(getattr(step, "kr", None))
            elif kind == "equilibrium" and role == "Keq":
                value = effective_equilibrium_keq(
                    step,
                    temperature_K=float(metadata.temperature_K),
                )
        elif source_coordinate_capabilities(role).algebra_target:
            if kind == "reaction":
                model = forward_rate_model_for_reaction(
                    step,
                    default_standard_conc_M=float(metadata.standard_conc_M),
                )
            else:
                model = forward_rate_model_for_equilibrium(
                    step,
                    temperature_K=None,
                    default_standard_conc_M=float(metadata.standard_conc_M),
                )
            if role == "A" and isinstance(model, ArrheniusRateModel):
                value = float(model.A)
            elif role == "Ea" and isinstance(model, ArrheniusRateModel):
                value = float(energy_units.from_jmol(model.Ea_J_per_mol))
            elif role == "dG_act" and isinstance(model, EyringRateModel):
                value = float(energy_units.from_jmol(model.dG_act_J_per_mol))
            elif role == "kappa" and isinstance(model, EyringRateModel):
                value = float(model.kappa)
            elif role == "dG_eq" and kind == "equilibrium":
                value = _value(
                    getattr(step, "metadata", {}).get(
                        EquilibriumMetadataKeys.DG_EQ_J_PER_MOL
                    )
                )
                if value is not None:
                    value = float(energy_units.from_jmol(value))
        if value is not None and math.isfinite(value):
            out[name] = float(value)
    return out
