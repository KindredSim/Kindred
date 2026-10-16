"""Scientific parameter facts shared by execution and GUI coordinate presentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from kindred.core.kinetics import ForwardRateModel, forward_rate_family
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.parameter_algebra import (
    apply_parameter_algebra_spec_to_mechanism,
)
from kindred.core.simulator.parameter_coordinates import build_parameter_target_map, coordinate_allows_zero
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism
from kindred.core.simulator.dsl_build_input import source_coordinate_capabilities
from kindred.core.simulator.step_indexing import get_step_index_map
from kindred.core.quantity_units import convert_value
from kindred.core.units import input_unit_bounds, role_domain

@dataclass(frozen=True, slots=True)
class SimulationParameter:
    name: str
    value: float
    family: str
    role: str
    allow_exact_zero: bool
    unit: str = "1"
    context: str = ""

    @property
    def domain(self) -> tuple[float, float]:
        """Admissible magnitudes in ``unit``: the scientific domain projected inward.

        A zero floor is open; exact zero is ``allow_exact_zero``.
        """
        low, high, _signed = role_domain(self.role)
        return input_unit_bounds(self.unit, low, high)

    @property
    def signed(self) -> bool:
        """Whether the mirrored negative magnitudes are admissible too."""
        return role_domain(self.role)[2]



def enumerate_simulation_parameters(
    prepared_dsl: PreparedDslRequestSet,
) -> tuple[SimulationParameter, ...]:
    """Enumerate writable coordinates from one accepted typed DSL authority."""

    mechanism = prepared_dsl.mutable_mechanism_shell()
    metadata = mechanism.metadata
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
    # Every coordinate is reported in its input unit: the param line's unit
    # for a mechanism input, energy= for energies, otherwise scientific units.
    input_units = dict(parameter_spec.input_units)
    scientific_units = dict(parameter_spec.quantity_units)
    units = {name: unit.symbol for name, unit in input_units.items()}

    def input_value(name: str, value: float) -> float:
        """The shortest input-unit number that converts back to ``value`` exactly."""
        source, target = scientific_units[name], input_units[name]
        converted = convert_value(float(value), source, target)
        for digits in range(1, 17):
            shortest = float(f"{converted:.{digits}g}")
            if convert_value(shortest, target, source) == value:
                return shortest
        return converted

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
            for source_role, source_name in entry["source_coordinates"].items()
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
        context = str(entry["context"]).strip()
        family = "Constant"
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
            elif role == "kr":
                family = "reverse"
                value = authority.kr
            elif role == "Keq":
                family = "reverse"
                value = authority.Keq
        else:
            continue
        coordinates.append(
            SimulationParameter(
                name=name,
                context=context,
                value=input_value(name, value),
                family=family,
                role=role,
                allow_exact_zero=coordinate_allows_zero(mechanism, parameter_target_map[name]),
                unit=units[name],
            )
        )

    # Physical coordinates are projected only from the source-coordinate map
    # admitted into the step map.  Their names are reaction-local (A1/Ea1,
    # dG_act1, dG_eq1, ...); no global uniqueness test or unindexed alias is
    # allowed to collapse repeated reactions.
    physical_candidates: list[tuple[str, str, str, float]] = []
    for entry in step_by_index.values():
        kind = str(entry["kind"])
        source_coordinates = {
            str(role): str(name)
            for role, name in entry["source_coordinates"].items()
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
        for role, name in source_coordinates.items():
            # The parser ties each authored role to its rate family.
            if role == "A":
                physical_candidates.append((name, "Arrhenius", role, float(model.A)))
            elif role == "Ea":
                physical_candidates.append((name, "Arrhenius", role, float(model.Ea_J_per_mol)))
            elif role == "dG_act":
                physical_candidates.append((name, "Eyring", role, float(model.dG_act_J_per_mol)))
            elif role == "dG_eq":
                physical_candidates.append((name, "equilibrium", role, float(authority.dG_eq_J_per_mol)))
            elif role == "kappa":
                physical_candidates.append((name, "Eyring", role, float(model.kappa)))

    for name, family, role, value in physical_candidates:
        if (
            name in parameter_assignment_names
            and assignment_kind(name) == "mechanism-bound"
        ):
            continue
        coordinates.append(
            SimulationParameter(
                name=name,
                value=input_value(name, value),
                family=family,
                role=role,
                allow_exact_zero=coordinate_allows_zero(mechanism, parameter_target_map[name]),
                unit=units[name],
            )
        )

    physical_names = source_coordinate_names | {
        item[0] for item in physical_candidates
    }
    scalar_params = dict(metadata["scalar_params"])
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
                value=input_value(name, value),
                family="scalar",
                role="scalar",
                allow_exact_zero=True,
                unit=units[name],
            )
        )
    return tuple(coordinates)
