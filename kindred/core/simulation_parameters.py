"""Scientific parameter facts shared by execution and GUI coordinate presentation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping

from kindred.core.equilibrium_rate_authority import (
    EquilibriumRateAuthorityKind,
    effective_equilibrium_keq,
    effective_equilibrium_reverse_rate,
    normalize_existing_equilibrium_rate_authority,
    step_entry_authored_role_is_derived,
    step_entry_authored_role_is_editable,
)
from kindred.core.kinetics import (
    ArrheniusRateModel,
    EyringRateModel,
    ForwardRateModel,
    forward_rate_at_temperature,
    forward_rate_family,
    forward_rate_model_for_equilibrium,
    forward_rate_model_for_reaction,
)
from kindred.core.mechanism_metadata import (
    MechanismMetadataView,
)
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.parameter_algebra import (
    apply_parameter_algebra_spec_to_mechanism,
    solver_parameter_units_from_mechanism,
)
from kindred.core.simulator.parameter_coordinates import build_parameter_target_map
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism
from kindred.core.simulator.dsl_build_input import source_coordinate_capabilities
from kindred.core.simulator.step_indexing import get_step_index_map
from kindred.core.validation import try_parse_callable_finite_float, try_parse_int
from kindred.core.species import validate_temperature
from kindred.core.stoichiometry import directional_order_fraction
from kindred.core.units import (
    UnitsModel,
    rate_constant_unit,
)

SimulationParameterScientificCategory = Literal["rate", "equilibrium_constant", "finite_scalar"]


@dataclass(frozen=True, slots=True)
class SimulationParameter:
    name: str
    value: float
    family: str
    role: str
    writable: bool
    derived: bool
    scientific_category: SimulationParameterScientificCategory
    allow_exact_zero: bool
    unit: str = "1"
    context: str = ""
    temperature_K: float | None = None


def _finite(value: object) -> float | None:
    parsed, ok = try_parse_callable_finite_float(value)
    return float(parsed) if ok else None


def enumerate_simulation_parameters(
    prepared_dsl: PreparedDslRequestSet,
) -> tuple[SimulationParameter, ...]:
    """Enumerate writable coordinates from one accepted typed DSL authority."""

    temperature = float(prepared_dsl.analysis.temperature_K)
    temperature = validate_temperature(
        temperature,
        field="Prepared DSL nominal temperature",
    )
    mechanism = prepared_dsl.mutable_mechanism_shell()
    metadata = getattr(mechanism, "metadata", {}) or {}
    metadata_view = MechanismMetadataView.from_metadata(metadata)
    parameter_spec = prepared_dsl.parameter_algebra_spec_for_execution()
    parameter_target_map = build_parameter_target_map(mechanism)
    apply_parameter_algebra_spec_to_mechanism(
        parameter_spec,
        mechanism=mechanism,
        require_mutable=False,
        parameter_target_map=parameter_target_map,
    )
    namespace = build_namespace_from_mechanism(mechanism)
    source_coordinate_names = namespace.source_coordinate_names
    step_by_index: dict[int, Mapping[str, object]] = {}
    for entry in get_step_index_map(mechanism):
        parsed, ok = try_parse_int(entry.get("step_index"))
        if ok:
            step_by_index[int(parsed)] = entry
    units = solver_parameter_units_from_mechanism(
        mechanism,
        parameter_target_map=parameter_target_map,
    )
    parameter_assignment_names = parameter_spec.param_assignment_names()

    def assignment_kind(name: str) -> str | None:
        if name not in parameter_assignment_names:
            return None
        return parameter_spec.parameter_assignment_kind(name)

    reactions = tuple(getattr(mechanism, "reactions", ()) or ())
    equilibria = tuple(getattr(mechanism, "equilibria", ()) or ())
    coordinates: list[SimulationParameter] = []

    for item in namespace.ordered_items:
        info = item.info
        if info.step_index is None:
            continue
        step_index = int(info.step_index)
        entry = step_by_index.get(step_index, {})
        kind = str(info.step_kind or "")
        role = str(info.role or "")
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
        derived = False
        writable = True
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
            raw_index, ok = try_parse_int(entry.get("reaction_index", -1))
            if not ok or not (0 <= raw_index < len(reactions)):
                continue
            model = forward_rate_model_for_reaction(
                reactions[int(raw_index)],
                default_standard_conc_M=metadata_view.standard_conc_M,
            )
            family = forward_rate_family(model)
            value = forward_rate_at_temperature(model, temperature)
            scientific_category = "rate"
            allow_exact_zero = family == "Constant"
        elif kind == "equilibrium":
            # The indexed forward/reverse/Keq names are independent only when
            # their exact source role was authored.  This keeps derived
            # kf/kr/Keq values out of Slider and Fit while retaining their
            # readout semantics in the rest of the namespace.
            if source_coordinates.get(role) != name:
                continue
            raw_index, ok = try_parse_int(entry.get("equilibrium_index", -1))
            if not ok or not (0 <= raw_index < len(equilibria)):
                continue
            equilibrium = equilibria[int(raw_index)]
            authority = normalize_existing_equilibrium_rate_authority(equilibrium)
            editable = step_entry_authored_role_is_editable(entry, role)
            derived_truth = step_entry_authored_role_is_derived(entry, role)
            writable = bool(
                authority.authored_role_is_editable(role)
                if editable is None
                else editable
            )
            writable = writable and not bool(entry.get("locked", False))
            derived = bool(
                authority.authored_role_is_derived(role)
                if derived_truth is None
                else derived_truth
            )
            if role == "kf":
                model = forward_rate_model_for_equilibrium(
                    equilibrium,
                    temperature_K=temperature,
                    default_standard_conc_M=metadata_view.standard_conc_M,
                )
                family = forward_rate_family(model)
                value = forward_rate_at_temperature(model, temperature)
                scientific_category = "rate"
                allow_exact_zero = (
                    authority.kind == EquilibriumRateAuthorityKind.KEQ
                    and family == "Constant"
                )
            elif role == "kr":
                family = "reverse"
                value = effective_equilibrium_reverse_rate(
                    equilibrium,
                    temperature_K=temperature,
                )
                scientific_category = "rate"
                allow_exact_zero = False
            elif role == "Keq":
                family = "reverse"
                value = effective_equilibrium_keq(
                    equilibrium,
                    temperature_K=temperature,
                )
                scientific_category = "equilibrium_constant"
                allow_exact_zero = False
        else:
            continue
        if bool(entry.get("locked", False)):
            writable = False
        if value is None or not math.isfinite(float(value)):
            continue
        # Fit and Preview consume this API directly: derived readouts never
        # enter their request parameter maps.
        if not writable or derived:
            continue
        coordinates.append(
            SimulationParameter(
                name=name,
                context=context,
                temperature_K=temperature if model is not None and family in {"Arrhenius", "Eyring"} else None,
                value=float(value),
                family=family,
                role=role,
                writable=True,
                derived=False,
                scientific_category=scientific_category,
                allow_exact_zero=allow_exact_zero,
                unit=str(units.get(name, "1")),
            )
        )

    # Physical coordinates are projected only from the source-coordinate map
    # admitted into the step map.  Their names are reaction-local (A1/Ea1,
    # dG_act1, dG_eq1, ...); no global uniqueness test or unindexed alias is
    # allowed to collapse repeated reactions.
    physical_candidates: list[tuple[str, str, str, float, bool, str]] = []
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
        order = 1.0
        authority = None
        if kind == "reaction":
            raw_index, ok = try_parse_int(entry.get("reaction_index", -1))
            if not ok or not (0 <= raw_index < len(reactions)):
                continue
            reaction = reactions[int(raw_index)]
            model = forward_rate_model_for_reaction(
                reaction,
                default_standard_conc_M=metadata_view.standard_conc_M,
            )
            order = float(
                directional_order_fraction(
                    getattr(reaction, "rate_orders", {}) or {},
                )
            )
        elif kind == "equilibrium":
            raw_index, ok = try_parse_int(entry.get("equilibrium_index", -1))
            if not ok or not (0 <= raw_index < len(equilibria)):
                continue
            equilibrium = equilibria[int(raw_index)]
            authority = normalize_existing_equilibrium_rate_authority(equilibrium)
            model = forward_rate_model_for_equilibrium(
                equilibrium,
                temperature_K=temperature,
                default_standard_conc_M=metadata_view.standard_conc_M,
            )
            order = float(
                directional_order_fraction(
                    getattr(equilibrium, "stoich_forward", {}) or {},
                )
            )
        if model is None:
            continue
        editable = not bool(entry.get("locked", False))
        energy_units = UnitsModel(energy_unit=metadata_view.energy_unit)
        for role, name in source_coordinates.items():
            if role == "A" and isinstance(model, ArrheniusRateModel):
                physical_candidates.append(
                    (
                        name,
                        "Arrhenius",
                        role,
                        float(model.A),
                        editable,
                        rate_constant_unit(order),
                    )
                )
            elif role == "Ea" and isinstance(model, ArrheniusRateModel):
                physical_candidates.append(
                    (
                        name,
                        "Arrhenius",
                        role,
                        float(energy_units.from_jmol(model.Ea_J_per_mol)),
                        editable,
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
                        editable,
                        metadata_view.energy_unit,
                    )
                )
            elif (
                role == "dG_eq"
                and authority is not None
                and authority.dG_eq_J_per_mol is not None
                and authority.authored_role_is_editable("dG_eq")
                and not authority.authored_role_is_derived("dG_eq")
            ):
                physical_candidates.append(
                    (
                        name,
                        "equilibrium",
                        role,
                        float(energy_units.from_jmol(float(authority.dG_eq_J_per_mol))),
                        editable,
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
                        editable,
                        "1",
                    )
                )

    for name, family, role, value, editable, unit in physical_candidates:
        if (
            name in parameter_assignment_names
            and assignment_kind(name) == "mechanism-bound"
        ):
            continue
        if not editable or not math.isfinite(value):
            continue
        allow_zero = role not in {"A", "kappa"}
        coordinates.append(
            SimulationParameter(
                name=name,
                value=value,
                family=family,
                role=role,
                writable=True,
                derived=False,
                scientific_category="finite_scalar",
                allow_exact_zero=allow_zero,
                unit=unit,
            )
        )

    physical_names = source_coordinate_names | {
        item[0] for item in physical_candidates
    }
    scalar_params = dict(metadata.get("scalar_params") or {})
    for raw_name, raw_value in scalar_params.items():
        name = str(raw_name)
        value = _finite(raw_value)
        assignment = assignment_kind(name)
        if (
            value is None
            or name in physical_names
            or assignment in {"derived scalar", "mechanism-bound"}
        ):
            continue
        coordinates.append(
            SimulationParameter(
                name=name,
                value=value,
                family="scalar",
                role="scalar",
                writable=True,
                derived=False,
                scientific_category="finite_scalar",
                allow_exact_zero=True,
                unit=str(units.get(name, "1")),
            )
        )
    return tuple(coordinates)
