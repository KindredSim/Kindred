"""Scientific parameter facts shared by execution and GUI coordinate presentation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping

from kindred.core.equilibrium_rate_authority import EquilibriumRateAuthorityKind
from kindred.core.kinetics import ArrheniusRateModel, EyringRateModel, ForwardRateModel, forward_rate_family
from kindred.core.mechanism_metadata import (
    MechanismMetadataView,
)
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.parameter_algebra import (
    apply_parameter_algebra_spec_to_mechanism,
)
from kindred.core.simulator.parameter_coordinates import build_parameter_target_map
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism
from kindred.core.simulator.dsl_build_input import source_coordinate_capabilities
from kindred.core.simulator.step_indexing import get_step_index_map
from kindred.core.quantity_units import convert_value
from kindred.core.units import (
    UnitsModel,
)

SimulationParameterScientificCategory = Literal["rate", "equilibrium_constant", "finite_scalar"]


@dataclass(frozen=True, slots=True)
class SimulationParameter:
    name: str
    value: float
    family: str
    role: str
    scientific_category: SimulationParameterScientificCategory
    allow_exact_zero: bool
    unit: str = "1"
    context: str = ""


def enumerate_simulation_parameters(
    prepared_dsl: PreparedDslRequestSet,
) -> tuple[SimulationParameter, ...]:
    """Enumerate writable coordinates from one accepted typed DSL authority."""

    mechanism = prepared_dsl.mutable_mechanism_shell()
    metadata = mechanism.metadata
    metadata_view = MechanismMetadataView.from_metadata(metadata)
    parameter_spec = prepared_dsl.parameter_algebra_spec_for_execution()
    parameter_target_map = build_parameter_target_map(mechanism)
    apply_parameter_algebra_spec_to_mechanism(
        parameter_spec,
        mechanism=mechanism,
        parameter_target_map=parameter_target_map,
    )
    namespace = build_namespace_from_mechanism(mechanism)
    source_coordinate_names = namespace.source_coordinate_names
    step_by_index: dict[int, Mapping[str, object]] = {}
    for entry in get_step_index_map(mechanism):
        step_by_index[int(entry["step_index"])] = entry
    units = {name: unit.symbol for name, unit in parameter_spec.input_units}
    parameter_assignment_names = parameter_spec.param_assignment_names()

    def assignment_kind(name: str) -> str | None:
        if name not in parameter_assignment_names:
            return None
        return parameter_spec.parameter_assignment_kind(name)

    reactions = tuple(mechanism.reactions)
    equilibria = tuple(mechanism.equilibria)
    coordinates: list[SimulationParameter] = []

    for item in namespace.ordered_items:
        info = item.info
        step_index = info.step_index
        entry = step_by_index.get(step_index, {})
        kind = info.step_kind
        role = info.role
        name = str(item.canonical_name)
        source_coordinates = {
            str(source_role): str(source_name)
            for source_role, source_name in dict(
                entry.get("source_coordinates") or {}
            ).items()
        }
        source_info = namespace.source_coordinate_info(name)
        if source_info is not None and not source_coordinate_capabilities(
            source_info.role
        ).continuous:
            continue
        if (
            name in parameter_assignment_names
            and assignment_kind(name) == "mechanism-bound"
        ):
            continue
        context = str(entry.get("context") or "").strip()
        family = "Constant"
        scientific_category: SimulationParameterScientificCategory
        allow_exact_zero = False
        model: ForwardRateModel | None = None
        value: float | None = None

        if kind == "reaction":
            # A canonical kN is an independent GUI/Fit coordinate only for
            # an authored Constant source.  Arrhenius/Eyring kN values are
            # derived readouts and their physical source coordinates are
            # projected below.
            if role != "k" or source_coordinates.get("k") != name:
                continue
            raw_index = int(entry["reaction_index"])
            model = reactions[int(raw_index)].forward_model
            family = forward_rate_family(model)
            value = model.k
            scientific_category = "rate"
            allow_exact_zero = family == "Constant"
        elif kind == "equilibrium":
            # The indexed forward/reverse/Keq names are independent only when
            # their exact source role was authored.  This keeps derived
            # kf/kr/Keq values out of Slider and Fit while retaining their
            # readout semantics in the rest of the namespace.
            if source_coordinates.get(role) != name:
                continue
            raw_index = int(entry["equilibrium_index"])
            equilibrium = equilibria[int(raw_index)]
            authority = equilibrium.authority
            if role == "kf":
                model = equilibrium.forward_model
                family = forward_rate_family(model)
                value = model.k
                scientific_category = "rate"
                allow_exact_zero = (
                    authority.kind == EquilibriumRateAuthorityKind.KEQ
                    and family == "Constant"
                )
            elif role == "kr":
                family = "reverse"
                value = authority.kr
                scientific_category = "rate"
                allow_exact_zero = False
            elif role == "Keq":
                family = "reverse"
                value = authority.Keq
                scientific_category = "equilibrium_constant"
                allow_exact_zero = False
        else:
            continue
        if value is None or not math.isfinite(float(value)):
            continue
        coordinates.append(
            SimulationParameter(
                name=name,
                context=context,
                value=float(value),
                family=family,
                role=role,
                scientific_category=scientific_category,
                allow_exact_zero=allow_exact_zero,
                unit=units[name],
            )
        )

    # Physical coordinates are projected only from the source-coordinate map
    # admitted into the step map.  Their names are reaction-local (A1/Ea1,
    # dG_act1, dG_eq1, ...); no global uniqueness test or unindexed alias is
    # allowed to collapse repeated reactions.
    physical_candidates: list[tuple[str, str, str, float, str]] = []
    for entry in step_by_index.values():
        kind = str(entry.get("kind") or "")
        source_coordinates = {
            str(role): str(name)
            for role, name in dict(entry.get("source_coordinates") or {}).items()
            if str(name) in source_coordinate_names
            and source_coordinate_capabilities(str(role)).continuous
        }
        if not source_coordinates:
            continue
        model: ForwardRateModel | None = None
        authority = None
        if kind == "reaction":
            raw_index = int(entry["reaction_index"])
            reaction = reactions[int(raw_index)]
            model = reaction.forward_model
        elif kind == "equilibrium":
            raw_index = int(entry["equilibrium_index"])
            equilibrium = equilibria[int(raw_index)]
            authority = equilibrium.authority
            model = equilibrium.forward_model
        if model is None:
            continue
        energy_units = UnitsModel(energy_unit=metadata_view.energy_unit)
        for role, name in source_coordinates.items():
            if role == "A" and isinstance(model, ArrheniusRateModel):
                physical_candidates.append(
                    (
                        name,
                        "Arrhenius",
                        role,
                        float(model.A),
                        units[name],
                    )
                )
            elif role == "Ea" and isinstance(model, ArrheniusRateModel):
                physical_candidates.append(
                    (
                        name,
                        "Arrhenius",
                        role,
                        float(energy_units.from_jmol(model.Ea_J_per_mol)),
                        metadata_view.energy_unit,
                    )
                )
            elif role == "dG_act" and isinstance(model, EyringRateModel):
                physical_candidates.append(
                    (
                        name,
                        "Eyring",
                        role,
                        float(energy_units.from_jmol(model.dG_act_J_per_mol)),
                        metadata_view.energy_unit,
                    )
                )
            elif role == "dG_eq":
                physical_candidates.append(
                    (
                        name,
                        "equilibrium",
                        role,
                        float(energy_units.from_jmol(float(authority.dG_eq_J_per_mol))),
                        metadata_view.energy_unit,
                    )
                )
            elif role == "kappa" and isinstance(model, EyringRateModel):
                physical_candidates.append(
                    (
                        name,
                        "Eyring",
                        role,
                        float(model.kappa),
                        "1",
                    )
                )

    for name, family, role, value, unit in physical_candidates:
        if (
            name in parameter_assignment_names
            and assignment_kind(name) == "mechanism-bound"
        ):
            continue
        if not math.isfinite(value):
            continue
        allow_zero = role not in {"A", "kappa"}
        coordinates.append(
            SimulationParameter(
                name=name,
                value=value,
                family=family,
                role=role,
                scientific_category="finite_scalar",
                allow_exact_zero=allow_zero,
                unit=unit,
            )
        )

    physical_names = source_coordinate_names | {
        item[0] for item in physical_candidates
    }
    scalar_params = dict(metadata.get("scalar_params") or {})
    input_units = dict(parameter_spec.input_units)
    scientific_units = dict(parameter_spec.quantity_units)
    for raw_name, value in scalar_params.items():
        name = str(raw_name)
        assignment = assignment_kind(name)
        if (
            name in physical_names
            or assignment in {"derived scalar", "mechanism-bound"}
        ):
            continue
        coordinates.append(
            SimulationParameter(
                name=name,
                value=convert_value(value, scientific_units[name], input_units[name]),
                family="scalar",
                role="scalar",
                scientific_category="finite_scalar",
                allow_exact_zero=True,
                unit=units[name],
            )
        )
    return tuple(coordinates)
