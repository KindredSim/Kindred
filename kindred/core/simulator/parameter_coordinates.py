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
from typing import Any, TypeAlias

from kindred.core.equilibrium_rate_authority import (
    EquilibriumRateAuthorityKind,
    derived_equilibrium_ratio_from_directional_rates,
    effective_reverse_rate_from_keq,
)
from kindred.core.kinetics import ArrheniusRateModel, ConstantRateModel, EyringRateModel, ForwardRateModel, forward_rate_at_temperature
from kindred.core.mechanism_metadata import EquilibriumMetadataKeys
from kindred.core.quantity_units import Unit
from kindred.core.stoichiometry import directional_order_fraction
from kindred.core.thermodynamics import concentration_Keq_from_deltaG_eq
from kindred.core.simulator.dsl_build_input import source_coordinate_capabilities
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism
from kindred.core.simulator.step_indexing import get_step_index_map
from kindred.core.units import admit_role_value

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
    "coordinate_allows_zero",
]


def build_parameter_target_map(
    mechanism: Any,
) -> dict[str, ParameterTarget]:
    """Build the parser-admitted indexed/physical coordinate target map."""

    namespace = build_namespace_from_mechanism(mechanism)
    step_roster = get_step_index_map(mechanism)
    targets: dict[str, ParameterTarget] = {}

    for item in namespace.ordered_items:
        entry = step_roster[item.source_index]
        kind = str(entry["kind"])
        index_key = "reaction_index" if kind == "reaction" else "equilibrium_index"
        index = int(entry[index_key])
        role = str(item.info.role)
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
        entry = step_roster[coordinate.source_index]
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
        source = dict(entry["source_coordinates"])
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
            authority = step.authority
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
    source = dict(entry["source_coordinates"])
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
                    updates[field] = int(value) if coordinate == "degeneracy" else value
            return forward_rate_at_temperature(replace(model, **updates), values["T"])

        return forward
    if role == "kr":
        return lambda: float(
            effective_reverse_rate_from_keq(
                values[f"kf{number}"], values[f"Keq{number}"]
            )
        )
    if step.authority.kind == EquilibriumRateAuthorityKind.KR:
        return lambda: derived_equilibrium_ratio_from_directional_rates(
            values[f"kf{number}"], values[f"kr{number}"]
        )
    delta_n = float(
        directional_order_fraction(step.stoich_back)
        - directional_order_fraction(step.stoich_forward)
    )
    return lambda: concentration_Keq_from_deltaG_eq(
        values[source["dG_eq"]],
        values["T"],
        step.metadata[EquilibriumMetadataKeys.STANDARD_CONC_M],
        delta_n,
    )


def coordinate_allows_zero(mechanism: object, target: ParameterTarget) -> bool:
    """Whether one writable coordinate admits exact zero in its concrete context.

    Writable targets are a reaction's ``k``, an equilibrium's ``kf``, ``kr`` or
    ``Keq``, or a continuous physical coordinate (A, Ea, dG_act, dG_eq, kappa).
    """

    route, kind, index, role, _entry = target
    if route == "physical":
        return role not in {"A", "kappa"}
    if role == "Keq":
        return False
    if kind == "reaction":
        return isinstance(mechanism.reactions[index].forward_model, ConstantRateModel)
    equilibrium = mechanism.equilibria[index]
    return (
        role == "kf"
        and isinstance(equilibrium.forward_model, ConstantRateModel)
        and equilibrium.authority.kind == EquilibriumRateAuthorityKind.KEQ
    )


def _admit_coordinate_value(
    mechanism: object, name: str, value: float, target: ParameterTarget, *, unit: str,
) -> float:
    """Admit one scientific value in its target's concrete context, reporting in ``unit``."""

    return admit_role_value(
        float(value), role=target[3], unit=unit,
        allow_exact_zero=coordinate_allows_zero(mechanism, target), field=name,
    )


def _admit_coordinate_groups(
    mechanism: object,
    values: Mapping[str, object],
    *,
    parameter_target_map: Mapping[str, ParameterTarget],
    units: Mapping[str, Unit],
) -> dict[tuple[str, int], tuple[Mapping[str, object], dict[str, float]]]:
    grouped: dict[
        tuple[str, int],
        tuple[Mapping[str, object], dict[str, float]],
    ] = {}

    for raw_name, raw_value in values.items():
        # Param lines and overrides name only writable model coordinates.
        name = str(raw_name)
        target = parameter_target_map[name]
        _route, kind, index, role, entry = target
        try:
            admitted = _admit_coordinate_value(
                mechanism, name, raw_value, target, unit=units[name].symbol,
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
) -> object:
    # The parser ties each authored role to its rate family.
    model = reaction.forward_model
    if "k" in updates:
        model = ConstantRateModel(k=updates["k"])
    model_fields: dict[str, float] = {}
    if "A" in updates:
        model_fields["A"] = updates["A"]
    if "Ea" in updates:
        model_fields["Ea_J_per_mol"] = updates["Ea"]
    if "dG_act" in updates:
        model_fields["dG_act_J_per_mol"] = updates["dG_act"]
    if "kappa" in updates:
        model_fields["kappa"] = updates["kappa"]
    if model_fields:
        model = replace(model, **model_fields)
    return replace(reaction, forward_model=model)


def _compose_equilibrium(
    equilibrium: object,
    *,
    updates: Mapping[str, float],
) -> object:
    model: ForwardRateModel | None = None
    if any(role in updates for role in ("A", "Ea", "dG_act", "kappa")):
        model = equilibrium.forward_model
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

    equilibrium_metadata = dict(equilibrium.metadata)
    if "dG_eq" in updates:
        equilibrium_metadata[EquilibriumMetadataKeys.DG_EQ_J_PER_MOL] = updates[
            "dG_eq"
        ]
    if "Keq" in updates:
        equilibrium_metadata[EquilibriumMetadataKeys.KEQ_INPUT] = updates["Keq"]
    if "kf" in updates:
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
    reactions = mechanism.reactions
    equilibria = mechanism.equilibria
    for (kind, index), (_entry, updates) in grouped.items():
        if kind == "reaction":
            yield (
                (kind, index),
                _compose_reaction(
                    reactions[index],
                    updates=updates,
                ),
            )
            continue
        yield (
            (kind, index),
            _compose_equilibrium(
                equilibria[index],
                updates=updates,
            ),
        )


def apply_mechanism_coordinate_values(
    mechanism: object,
    values: Mapping[str, object],
    *,
    parameter_target_map: Mapping[str, ParameterTarget],
    units: Mapping[str, Unit],
) -> None:
    """Compose each complete changed step and publish it once; refusals report ``units``."""

    grouped = _admit_coordinate_groups(
        mechanism,
        values,
        parameter_target_map=parameter_target_map,
        units=units,
    )
    if not grouped:
        return
    reactions = mechanism.reactions
    equilibria = mechanism.equilibria
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
    reactions = mechanism.reactions
    equilibria = mechanism.equilibria

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
                authority = step.authority
                if role == "kr" and authority.kind == EquilibriumRateAuthorityKind.KR:
                    value = authority.kr
                elif role == "Keq" and authority.has_explicit_keq_param:
                    value = authority.Keq
        elif source_coordinate_capabilities(role).algebra_target or role == "degeneracy":
            model = step.forward_model
            if role == "A" and isinstance(model, ArrheniusRateModel):
                value = float(model.A)
            elif role == "Ea" and isinstance(model, ArrheniusRateModel):
                value = float(model.Ea_J_per_mol)
            elif role == "dG_act" and isinstance(model, EyringRateModel):
                value = float(model.dG_act_J_per_mol)
            elif role == "kappa" and isinstance(model, EyringRateModel):
                value = float(model.kappa)
            elif role == "degeneracy" and isinstance(model, EyringRateModel):
                value = float(model.degeneracy)
            elif role == "dG_eq":
                value = step.authority.dG_eq_J_per_mol
        if value is not None:
            out[name] = float(value)
    return out
