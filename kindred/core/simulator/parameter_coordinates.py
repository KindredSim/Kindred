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
    normalize_existing_equilibrium_rate_authority,
    require_step_entry_authored_role_is_editable,
    derived_equilibrium_ratio_from_directional_rates,
    effective_reverse_rate_from_keq,
)
from kindred.core.kinetics import ArrheniusRateModel, ConstantRateModel, EyringRateModel, ForwardRateModel, forward_rate_at_temperature
from kindred.core.mechanism_metadata import (
    EquilibriumMetadataKeys,
    MechanismMetadataView,
)
from kindred.core.stoichiometry import directional_order_fraction
from kindred.core.thermodynamics import concentration_Keq_from_deltaG_eq
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


def chemistry_parameter_dependencies(
    mechanism: object, targets: Mapping[str, ParameterTarget]
) -> dict[str, tuple[str, ...]]:
    """Describe derived readouts using the admitted typed chemistry authority."""
    dependencies: dict[str, tuple[str, ...]] = {}
    for name, (_route, kind, index, role, entry) in targets.items():
        step = (
            mechanism.reactions[index]
            if kind == "reaction"
            else mechanism.equilibria[index]
        )
        source = dict(entry.get("source_coordinates") or {})
        number = entry["step_index"]
        if role in {"k", "kf"}:
            model = step.forward_model
            roles = (
                ("A", "Ea")
                if isinstance(model, ArrheniusRateModel)
                else ("dG_act", "kappa", "degeneracy")
                if isinstance(model, EyringRateModel)
                else ()
            )
            if roles:
                dependencies[name] = tuple(source[r] for r in roles if r in source) + (
                    "T",
                )
        elif kind == "equilibrium":
            authority = normalize_existing_equilibrium_rate_authority(step)
            if role == "kr" and authority.kind == EquilibriumRateAuthorityKind.KEQ:
                dependencies[name] = (f"kf{number}", f"Keq{number}")
            elif role == "Keq":
                if authority.kind == EquilibriumRateAuthorityKind.KR:
                    dependencies[name] = (f"kf{number}", f"kr{number}")
                elif (
                    authority.dG_eq_J_per_mol is not None
                    and not authority.has_explicit_keq_param
                ):
                    dependencies[name] = (source["dG_eq"], "T")
    return dependencies


def bind_chemistry_readout(
    mechanism: object, target: ParameterTarget, values: dict[str, float]
):
    """Bind an owned calculation without copying formulas or mutating a step."""
    _route, kind, index, role, entry = target
    step = (
        mechanism.reactions[index]
        if kind == "reaction"
        else mechanism.equilibria[index]
    )
    metadata = MechanismMetadataView.from_metadata(mechanism.metadata)
    source = dict(entry.get("source_coordinates") or {})
    number = entry["step_index"]
    if role in {"k", "kf"}:
        model = step.forward_model
        fields = {
            "A": "A",
            "Ea": "Ea_J_per_mol",
            "dG_act": "dG_act_J_per_mol",
            "kappa": "kappa",
            "degeneracy": "degeneracy",
        }

        def forward():
            updates = {}
            for coordinate, field in fields.items():
                if coordinate in source:
                    value = values[source[coordinate]]
                    updates[field] = (
                        int(value)
                        if coordinate == "degeneracy"
                        else admit_physical_coordinate_value(
                            value,
                            role=coordinate,
                            energy_unit=metadata.energy_unit,
                            field=source[coordinate],
                        )
                    )
            return forward_rate_at_temperature(replace(model, **updates), values["T"])

        return forward
    if role == "kr":
        return lambda: float(
            effective_reverse_rate_from_keq(
                values[f"kf{number}"], values[f"Keq{number}"]
            )
        )
    authority = normalize_existing_equilibrium_rate_authority(step)
    if authority.kind == EquilibriumRateAuthorityKind.KR:
        return lambda: derived_equilibrium_ratio_from_directional_rates(
            values[f"kf{number}"], values[f"kr{number}"]
        )
    delta_n = float(
        directional_order_fraction(step.stoich_back)
        - directional_order_fraction(step.stoich_forward)
    )
    standard_conc = float(
        step.metadata.get(
            EquilibriumMetadataKeys.STANDARD_CONC_M, metadata.standard_conc_M
        )
    )
    return lambda: concentration_Keq_from_deltaG_eq(
        admit_physical_coordinate_value(
            values[source["dG_eq"]],
            role="dG_eq",
            energy_unit=metadata.energy_unit,
            field=source["dG_eq"],
        ),
        values["T"],
        standard_conc,
        delta_n,
    )


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
        model = reactions[index].forward_model
        return admit_rate(
            value,
            field=name,
            allow_zero=isinstance(model, ConstantRateModel),
        )

    if kind == "equilibrium" and role in {"kf", "kr"}:
        equilibria = getattr(mechanism, "equilibria", ()) or ()
        equilibrium = equilibria[index]
        model = equilibrium.forward_model
        authority = normalize_existing_equilibrium_rate_authority(equilibrium)
        allow_zero = (
            role == "kf"
            and isinstance(model, ConstantRateModel)
            and authority.kind == EquilibriumRateAuthorityKind.KEQ
        )
        return admit_rate(value, field=name, allow_zero=allow_zero)

    raise ValueError(f"Unsupported parameter target for {name!r}")


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
) -> object:
    model = reaction.forward_model
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
    return replace(reaction, forward_model=model)


def _compose_equilibrium(
    equilibrium: object,
    *,
    updates: Mapping[str, float],
    step_name: int,
) -> object:
    model: ForwardRateModel | None = None
    if any(role in updates for role in ("A", "Ea", "dG_act", "kappa")):
        model = equilibrium.forward_model
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
        model = equilibrium.forward_model

    equilibrium_metadata = dict(getattr(equilibrium, "metadata", {}) or {})
    if "dG_eq" in updates:
        equilibrium_metadata[EquilibriumMetadataKeys.DG_EQ_J_PER_MOL] = updates[
            "dG_eq"
        ]
    if "Keq" in updates:
        equilibrium_metadata[EquilibriumMetadataKeys.KEQ_INPUT] = updates["Keq"]
    if "kf" in updates:
        if not isinstance(model, ConstantRateModel):
            raise DSLError(f"Indexed kf{step_name} cannot replace a non-Constant model")
        model = ConstantRateModel(k=updates["kf"])
    return replace(
        equilibrium, forward_model=equilibrium.forward_model if model is None else model,
        kr=updates.get("kr", equilibrium.kr), metadata=equilibrium_metadata,
    )


def _compose_coordinate_groups(
    mechanism: object,
    grouped: Mapping[
        tuple[str, int],
        tuple[Mapping[str, object], Mapping[str, float]],
    ],
) -> Iterator[tuple[tuple[str, int], object]]:
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
                ),
            )
            continue
        yield (
            (kind, index),
            _compose_equilibrium(
                equilibria[index],
                updates=updates,
                step_name=step_name,
            ),
        )


def apply_mechanism_coordinate_values(
    mechanism: object,
    values: Mapping[str, object],
    *,
    parameter_target_map: Mapping[str, ParameterTarget],
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
) -> dict[str, float]:
    """Read independent authored coordinates without deriving coefficients."""

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
            step = reactions[index]
        else:
            step = equilibria[index]

        value: float | None = None
        if route == "indexed":
            if role in {"k", "kf"} and isinstance(step.forward_model, ConstantRateModel):
                value = step.forward_model.k
            elif kind == "equilibrium":
                authority = normalize_existing_equilibrium_rate_authority(step)
                if role == "kr" and authority.kind == EquilibriumRateAuthorityKind.KR:
                    value = authority.kr
                elif role == "Keq" and authority.has_explicit_keq_param:
                    value = authority.Keq
        elif source_coordinate_capabilities(role).algebra_target or role == "degeneracy":
            model = step.forward_model
            if role == "A" and isinstance(model, ArrheniusRateModel):
                value = float(model.A)
            elif role == "Ea" and isinstance(model, ArrheniusRateModel):
                value = float(energy_units.from_jmol(model.Ea_J_per_mol))
            elif role == "dG_act" and isinstance(model, EyringRateModel):
                value = float(energy_units.from_jmol(model.dG_act_J_per_mol))
            elif role == "kappa" and isinstance(model, EyringRateModel):
                value = float(model.kappa)
            elif role == "degeneracy" and isinstance(model, EyringRateModel):
                value = float(model.degeneracy)
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
