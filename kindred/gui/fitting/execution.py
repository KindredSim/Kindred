"""Qt-free fitting science and child-local cooperative controls."""

from __future__ import annotations

import math
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from kindred.core.analysis.fit_dataset_payload import (
    FitDatasetSpec,
    FitObservationSeries,
)
from kindred.core.analysis.global_fitting import fit_global
from kindred.core.analysis.global_fit_execution import (
    FitCoordinate,
    DatasetFitInfo,
    FitProgress,
    GlobalFitResult,
    ObjectiveDatasetInput,
    evaluate_dataset_simulation,
    normalize_weights,
)
from kindred.core.analysis.fit_evaluation_outcome import (
    ScientificUnavailable,
)
from kindred.core.exceptions import (
    FitSimulationError,
    FittingCancelled,
)
from kindred.core.fitting_evaluation import (
    FITTING_PARAM_ORIGIN_CONFIGURED_DATASET,
    FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
    FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
    FittingEvaluator,
    prepare_fitting_execution_context,
)
from kindred.core.fitting_completion import (
    FitExtensionProgress,
)
from kindred.core.result_repository import SimulationResultValue
from kindred.core.quantity_units import Unit, SECOND
from kindred.core.simulator.solver_types import SimulationCancellationControl
from kindred.core.simulation_failure import (
    FactualCause,
    factual_cause_from_exception,
)
from kindred.gui.curve_copy_headers import observable_copy_definitions
from kindred.gui.fitting.constants import INITIAL_PREFIX
from kindred.gui.fitting.launch import (
    FitDatasetLocalSettings,
    _FitDispatchCompilation,
    FitExecutionRequest,
    FitOriginSnapshot,
    FitSelectedObservation,
    FitPreparedDatasetConfig,
    FitSetupSnapshot,
    _replace_fit_parameter,
)
from kindred.gui.parameter_enumeration import admit_gui_parameter_value
from kindred.core.species import validate_concentration

__all__ = [
    "FitCandidate",
    "FitCurveDataset",
    "FitLiveUpdate",
    "fit_bound_state",
    "project_fit_candidate_curves",
]

_BEST_INTERVAL_SECONDS = 0.25
_CURVE_INTERVAL_SECONDS = 2.0


def _finite_items(
    values: Mapping[str, object],
    *,
    field: str,
) -> tuple[tuple[str, float], ...]:
    normalized: list[tuple[str, float]] = []
    for raw_name, raw_value in values.items():
        name = str(raw_name or "").strip()
        value = float(raw_value)
        if not name or not math.isfinite(value):
            raise ValueError(f"{field} contains an invalid value.")
        normalized.append((name, value))
    normalized.sort()
    return tuple(normalized)


FitBoundState = Literal["lower", "upper", "interior"]


def fit_bound_state(
    value: float,
    minimum: float,
    maximum: float,
    log_scale: bool = False,
) -> FitBoundState:
    """Classify an estimate against bounds in its optimizer coordinate.

    Fit results retain decoded physical values, while a log10 fit optimizes
    their base-10 logarithms.  Bound classification therefore has to happen
    after applying the same encoding used by the optimizer.  A side's
    tolerance is local to that encoded value and bound; the opposite bound
    must not make a small-scale estimate appear to be on a boundary.
    """

    value = float(value)
    minimum = float(minimum)
    maximum = float(maximum)

    def encode(raw_value: float) -> float | None:
        if not math.isfinite(raw_value):
            return None
        if log_scale:
            if raw_value <= 0.0:
                return None
            return math.log10(raw_value)
        return raw_value

    encoded_value = encode(value)
    encoded_minimum = encode(minimum)
    encoded_maximum = encode(maximum)
    if encoded_value is None:
        return "interior"

    def is_at_bound(encoded_bound: float | None) -> bool:
        if encoded_bound is None:
            return False
        # Exact equality is checked separately so a value equal to both
        # malformed/degenerate sides has deterministic lower precedence.
        if encoded_value == encoded_bound:
            return True
        absolute_tolerance = 8.0 * max(
            math.ulp(encoded_value), math.ulp(encoded_bound)
        )
        return math.isclose(
            encoded_value,
            encoded_bound,
            rel_tol=1.0e-9,
            abs_tol=absolute_tolerance,
        )

    if encoded_minimum is not None and encoded_value == encoded_minimum:
        return "lower"
    if encoded_maximum is not None and encoded_value == encoded_maximum:
        return "upper"
    if is_at_bound(encoded_minimum):
        return "lower"
    if is_at_bound(encoded_maximum):
        return "upper"
    return "interior"

def _settle_terminal_bounds(
    setup: FitSetupSnapshot,
    result: GlobalFitResult,
) -> GlobalFitResult:
    """Publish fitted values the report classifies at a bound as that bound."""

    parameters = {item.name: item.selected_config for item in setup.parameters}
    mapped_set_ids = {item.dataset_id: item.set_id for item in setup.origin.mapped_datasets}
    initials = {(item.set_id, item.species): item for item in setup.initials}
    groups = {
        (dataset_id, group.species): group
        for group in setup.shared_initial_groups
        for dataset_id in group.member_dataset_ids
    }

    def settle(value: float, minimum: float, maximum: float, log10: bool) -> float:
        state = fit_bound_state(value, minimum, maximum, log10)
        return minimum if state == "lower" else maximum if state == "upper" else value

    def parameter(name: str, value: float) -> float:
        config = parameters[name]
        return settle(value, config.minimum, config.maximum, config.log10)

    def dataset_value(dataset_id: str, name: str, value: float) -> float:
        if not name.startswith(INITIAL_PREFIX):
            return parameter(name, value)
        species = name[len(INITIAL_PREFIX):]
        initial = groups.get((dataset_id, species)) or initials[(mapped_set_ids[dataset_id], species)]
        return settle(value, initial.minimum, initial.maximum, initial.log10)

    return replace(
        result,
        shared_params={name: parameter(name, value) for name, value in result.shared_params.items()},
        dataset_params={
            dataset_id: {name: dataset_value(dataset_id, name, value) for name, value in values.items()}
            for dataset_id, values in result.dataset_params.items()
        },
        centred_fixed_effects_settlements=tuple(
            replace(
                item,
                overall_decoded_value=parameter(item.parameter_name, item.overall_decoded_value),
                dataset_values={
                    dataset_id: parameter(item.parameter_name, value)
                    for dataset_id, value in item.dataset_values.items()
                },
            )
            for item in result.centred_fixed_effects_settlements
        ),
    )


def _fit_coordinate_display_label(value: FitCoordinate) -> str:
    """Return a human label that cannot alias typed coordinate scopes."""

    if value.scope == "centred_overall":
        return f"Overall · {value.name}"
    return value.display_name


@dataclass(frozen=True, slots=True)
class FitCurveDataset:
    dataset_id: str
    display_name: str
    time_column_name: str | None
    simulation: SimulationResultValue
    observed_x: tuple[tuple[str, tuple[float, ...]], ...]
    observed_y: tuple[tuple[str, tuple[float, ...]], ...]
    observed_time: tuple[tuple[str, tuple[float, ...]], ...]
    observed_raw_columns: tuple[tuple[str, str], ...]
    model_definitions: tuple[tuple[str, str], ...]
    observed_x_name: str | None
    observed_x_unit: Unit | None
    observed_y_units: tuple[tuple[str, Unit | None], ...]
    x_name: str = "t"

    def __post_init__(self) -> None:
        x, y = dict(self.observed_x), dict(self.observed_y)
        if x.keys() != y.keys() or any(len(x[name]) != len(values) for name, values in y.items()):
            raise ValueError("Fit observation axes must align.")
        if tuple(name for name, _ in self.observed_y_units) != tuple(name for name, _ in self.observed_y):
            raise ValueError("Fit observation units must align with their Y targets.")
        times = dict(self.observed_time)
        if times.keys() != y.keys() or any(len(times[name]) not in (0, len(values)) for name, values in y.items()):
            raise ValueError("Fit observation times must align with their samples.")


@dataclass(frozen=True, slots=True)
class FitLiveUpdate:
    """Non-authoritative transient presentation of optimizer progress."""
    iteration: int
    cost: float
    parameters: tuple[tuple[str, float], ...]
    scalar_values_by_dataset: tuple[
        tuple[str, tuple[tuple[str, float], ...]], ...
    ]
    initial_values_by_set: tuple[
        tuple[str, tuple[tuple[str, float], ...]], ...
    ]


@dataclass(frozen=True, slots=True)
class FitCandidate:
    """Scientifically available terminal result, still local to the Fit window."""

    request: FitExecutionRequest
    result: GlobalFitResult
    setup_snapshot: FitSetupSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.request, FitExecutionRequest):
            raise TypeError("Fit candidate request must be retained.")
        if not isinstance(self.result, GlobalFitResult):
            raise TypeError("Fit candidate result must be canonical.")
        nominal_setup = self.setup_snapshot
        dispatch = self.request.dispatch
        expected_dataset_ids = {
            item.dataset_id
            for item in nominal_setup.datasets
            if item.included
        }
        if set(dispatch.included_dataset_ids) != expected_dataset_ids:
            raise ValueError(
                "Fit candidate dispatch roster must match its terminal setup."
            )
        scalar_dataset_ids = {
            dataset_id
            for dataset_id in self.result.dataset_params
        }
        result_dataset_ids = set(self.result.dataset_params)
        simulation_dataset_ids = {
            info.dataset_id for info in self.result.dataset_info
        }
        if (
            scalar_dataset_ids != expected_dataset_ids
            or result_dataset_ids != expected_dataset_ids
            or simulation_dataset_ids != expected_dataset_ids
        ):
            raise ValueError(
                "Fit candidate terminal rosters must exactly match the "
                "included datasets."
            )
    @property
    def origin(self) -> FitOriginSnapshot:
        return self.request.setup.origin

    @property
    def initial_values_by_set(
        self,
    ) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
        """Project fitted initial concentrations solely from the result."""

        return _initial_values_from_result(
            self.setup_snapshot,
            self.origin,
            self.result,
        )

def _observed_curve_parts(
    observations: tuple[FitSelectedObservation, ...],
    *,
    x_observations: FitObservationSeries | None = None,
) -> tuple[
    tuple[tuple[str, tuple[float, ...]], ...],
    tuple[tuple[str, tuple[float, ...]], ...],
]:
    x_by_row = (
        {}
        if x_observations is None
        else {
            int(row): float(value)
            for row, value in zip(
                x_observations.source_rows,
                x_observations.y,
                strict=True,
            )
        }
    )
    axes = {
        item.model_target: np.asarray(
            [x_by_row[int(row)] for row in item.source_rows],
            dtype=float,
        )
        if x_observations is not None
        else np.asarray(item.t, dtype=float)
        for item in observations
    }
    return (
        tuple(
            (
                item.model_target,
                tuple(
                    float(value)
                    for value in axes.get(
                        item.model_target,
                        np.asarray(item.t, dtype=float),
                    )
                ),
            )
            for item in observations
        ),
        tuple((item.model_target, item.y) for item in observations),
    )


def _series_from_selected(
    observation: FitSelectedObservation,
) -> FitObservationSeries:
    """Lower one retained immutable selection into the evaluator payload."""

    return FitObservationSeries(
        raw_observation_id=observation.raw_observation_id,
        raw_column=observation.raw_column,
        t=np.asarray(observation.t, dtype=float),
        y=np.asarray(observation.y, dtype=float),
        source_rows=observation.source_rows,
    )


def _compile_dataset_spec(
    prepared: FitPreparedDatasetConfig,
    local: FitDatasetLocalSettings,
) -> FitDatasetSpec:
    parametric = local.parametric
    selected = prepared.selected_observations
    observation_map = {
        item.model_target: _series_from_selected(item)
        for item in selected
    }
    x_observations = (
        None
        if prepared.parametric_x_observation is None
        else _series_from_selected(prepared.parametric_x_observation)
    )
    if x_observations is not None:
        x_rows = set(x_observations.source_rows)
        if any(
            not set(item.source_rows).issubset(x_rows)
            for item in observation_map.values()
        ):
            raise ValueError(
                f"Dataset {prepared.dataset_id!r}: Parametric-X capture is missing a selected Y source row."
            )
    return FitDatasetSpec(
        dataset_id=prepared.dataset_id,
        species_list=tuple(item.model_target for item in selected),
        observations=observation_map,
        target_weights=dict(local.target_weights),
        x_name=parametric.x_name if parametric is not None else "t",
        x_observations=x_observations,
    )


def _initial_values_from_result(
    setup: FitSetupSnapshot,
    origin: FitOriginSnapshot,
    result: GlobalFitResult,
) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    included_ids = {
        item.dataset_id for item in setup.datasets if item.included
    }
    mapped_by_id = {mapped.dataset_id: mapped for mapped in origin.mapped_datasets}
    values_by_set: dict[str, dict[str, float]] = {}
    grouped = {
        (dataset_id, group.species)
        for group in setup.shared_initial_groups
        for dataset_id in group.member_dataset_ids
    }

    def coupled_value(dataset_ids: Sequence[str], species: str) -> float:
        values: set[float] = set()
        parameter_name = f"{INITIAL_PREFIX}{species}"
        for dataset_id in dataset_ids:
            dataset_result = result.dataset_params.get(dataset_id)
            if dataset_result is None or parameter_name not in dataset_result:
                raise ValueError(
                    f"Fit result omitted coupled initial {species!r} for "
                    f"included dataset {dataset_id!r}."
                )
            value = validate_concentration(
                float(dataset_result[parameter_name]),
                field=f"fitted initial {species!r}",
            )
            if not math.isfinite(value):
                raise ValueError(f"Fit result contains a non-finite initial {species!r}.")
            values.add(value)
        if len(values) != 1:
            raise ValueError(f"Fit result disagrees across coupled initial rows for {species!r}.")
        return values.pop()

    for initial in sorted(setup.initials, key=lambda item: (item.set_id, item.species)):
        if not initial.fit:
            continue
        dataset_ids = sorted(
            mapped.dataset_id
            for mapped in origin.mapped_datasets
            if mapped.set_id == initial.set_id
            and mapped.dataset_id in included_ids
            and (mapped.dataset_id, initial.species) not in grouped
        )
        if not dataset_ids:
            continue
        value = coupled_value(dataset_ids, initial.species)
        values_by_set.setdefault(initial.set_id, {})[initial.species] = value

    for group in setup.shared_initial_groups:
        if not group.fit:
            continue
        applicable_ids = sorted(
            dataset_id
            for dataset_id in group.member_dataset_ids
            if dataset_id in included_ids
        )
        if not applicable_ids:
            continue
        value = coupled_value(applicable_ids, group.species)
        for dataset_id in applicable_ids:
            mapped = mapped_by_id[dataset_id]
            values_by_set.setdefault(mapped.set_id, {})[group.species] = value
    return tuple(
        (
            set_id,
            _finite_items(values, field=f"Fit initials {set_id}"),
        )
        for set_id, values in sorted(values_by_set.items())
    )


def _terminal_setup_snapshot(
    setup_snapshot: FitSetupSnapshot,
    *,
    origin: FitOriginSnapshot,
    result: GlobalFitResult,
    dispatch: _FitDispatchCompilation,
    effective_datasets: Mapping[str, FitDatasetLocalSettings],
    initial_values_by_set: tuple[
        tuple[str, tuple[tuple[str, float], ...]], ...
    ],
) -> FitSetupSnapshot:
    """Overlay publishable terminal values using admitted dispatch routes."""

    included_ids = {
        item.dataset_id for item in setup_snapshot.datasets if item.included
    }
    included_set_ids = {
        mapped.set_id
        for mapped in origin.mapped_datasets
        if mapped.dataset_id in included_ids
    }
    shared_results = {
        str(name): float(value)
        for name, value in result.shared_params.items()
    }
    dataset_results = {
        dataset_id: {
            str(name): float(value)
            for name, value in values.items()
        }
        for dataset_id, values in result.dataset_params.items()
        if dataset_id in included_ids
    }
    fixed_results = {
        dataset_id: {
            str(name): float(value) for name, value in values
        }
        for dataset_id, values in dispatch.fixed_by_dataset
        if dataset_id in included_ids
    }
    if set(fixed_results) != included_ids:
        raise ValueError("Fit request fixed-value roster is incomplete.")
    request_parameters = {
        parameter.name: parameter
        for parameter in setup_snapshot.parameters
    }
    parameters = []
    for parameter in setup_snapshot.parameters:
        requested = request_parameters.get(parameter.name)
        if requested is None:
            raise ValueError(
                f"Terminal Fit setup contains unknown parameter "
                f"{parameter.name!r}."
            )
        dataset_values = tuple(parameter.selected_config.dataset_values)
        requested_config = requested.selected_config
        if not requested_config.fit:
            retained_values: dict[str, float] = {}
            for dataset_id in included_ids:
                values = fixed_results.get(dataset_id)
                if values is None or parameter.name not in values:
                    raise ValueError(
                        f"Fit request omitted fixed parameter {parameter.name!r} "
                        f"for included dataset {dataset_id!r}."
                    )
                retained_values[dataset_id] = float(values[parameter.name])
            if requested.scope == "shared":
                unique = set(retained_values.values())
                if len(unique) != 1:
                    raise ValueError(
                        f"Fit request disagrees on fixed Shared parameter {parameter.name!r}."
                    )
                parameters.append(
                    _replace_fit_parameter(
                        parameter,
                        initial=unique.pop(),
                        dataset_values=dataset_values,
                    )
                )
            else:
                parameters.append(
                    _replace_fit_parameter(
                        parameter,
                        dataset_values=tuple(
                            (dataset_id, retained_values.get(dataset_id, value))
                            for dataset_id, value in dataset_values
                        ),
                    )
                )
            continue
        if requested.scope == "shared":
            if parameter.name not in shared_results:
                raise ValueError(
                    f"Fit result omitted shared parameter {parameter.name!r}."
                )
            value = shared_results[parameter.name]
            if not math.isfinite(value):
                raise ValueError(
                    f"Fit result contains non-finite parameter "
                    f"{parameter.name!r}."
                )
            parameters.append(
                _replace_fit_parameter(
                    parameter,
                    initial=value,
                    dataset_values=dataset_values,
                )
            )
            continue
        expected_ids = {
            dataset_id
            for dataset_id, _value in requested_config.dataset_values
        }
        if not included_ids.issubset(expected_ids):
            raise ValueError(
                f"Dataset-scoped parameter {parameter.name!r} no longer "
                "matches the included dataset identities."
            )
        fitted_by_id = dict(requested_config.dataset_values)
        for dataset_id, _launch_value in requested_config.dataset_values:
            if dataset_id not in included_ids:
                continue
            values = dataset_results.get(dataset_id)
            if values is None or parameter.name not in values:
                raise ValueError(
                    f"Fit result omitted dataset-scoped parameter "
                    f"{parameter.name!r} for {dataset_id!r}."
                )
            value = values[parameter.name]
            if not math.isfinite(value):
                raise ValueError(
                    f"Fit result contains non-finite parameter "
                    f"{parameter.name!r} for {dataset_id!r}."
                )
            fitted_by_id[dataset_id] = value
        fitted_values = [
            (dataset_id, fitted_by_id[dataset_id])
            for dataset_id, _value in requested_config.dataset_values
        ]
        parameters.append(
            _replace_fit_parameter(
                parameter,
                dataset_values=tuple(fitted_values),
            )
        )

    initial_results = {
        (set_id, species): value
        for set_id, values in initial_values_by_set
        for species, value in values
    }
    grouped = {
        (dataset_id, group.species)
        for group in setup_snapshot.shared_initial_groups
        for dataset_id in group.member_dataset_ids
    }
    initials = tuple(
        replace(
            initial,
            initial=(
                initial_results[(initial.set_id, initial.species)]
            ),
        )
        if (
            initial.fit
            and initial.set_id in included_set_ids
            and not any(
                (mapped.dataset_id, initial.species) in grouped
                for mapped in origin.mapped_datasets
                if mapped.set_id == initial.set_id
            )
            and (initial.set_id, initial.species) in initial_results
        )
        else initial
        for initial in setup_snapshot.initials
    )
    groups = []
    for group in setup_snapshot.shared_initial_groups:
        if not group.fit:
            groups.append(group)
            continue
        applicable_set_ids = {
            mapped.set_id
            for mapped in origin.mapped_datasets
            if mapped.dataset_id in included_ids
            and mapped.dataset_id in group.member_dataset_ids
        }
        values = {
            value
            for (set_id, species), value in initial_results.items()
            if species == group.species and set_id in applicable_set_ids
        }
        if values:
            if len(values) != 1:
                raise ValueError(
                    f"Fit result did not resolve one shared initial value for {group.species!r}."
                )
            groups.append(replace(group, initial=values.pop()))
        else:
            groups.append(group)
    return replace(
        setup_snapshot,
        datasets=tuple(
            effective_datasets.get(item.dataset_id, item)
            for item in setup_snapshot.datasets
        ),
        parameters=tuple(parameters),
        initials=initials,
        shared_initial_groups=tuple(groups),
    )


def _curves_from_simulations(
    *,
    request: FitExecutionRequest,
    effective_local: Mapping[str, FitDatasetLocalSettings],
    evaluations: Mapping[str, SimulationResultValue],
) -> tuple[FitCurveDataset, ...]:
    """Draw model trajectories in simulation order, independently of fit matching."""

    mapped_by_id = {
        item.dataset_id: item for item in request.setup.origin.mapped_datasets
    }
    prepared_by_id = {
        item.dataset_id: item for item in request.dispatch.prepared_datasets
    }
    if not any(evaluations.get(dataset_id) is not None for dataset_id in request.dispatch.included_dataset_ids):
        return ()
    model_definitions = observable_copy_definitions(request.setup.origin.prepared_dsl)
    curves: list[FitCurveDataset] = []
    for dataset_id in request.dispatch.included_dataset_ids:
        simulation = evaluations.get(dataset_id)
        if simulation is None:
            continue
        local = effective_local[dataset_id]
        mapped = mapped_by_id[dataset_id]
        prepared = prepared_by_id[dataset_id]
        if not isinstance(simulation, SimulationResultValue):
            raise TypeError("Fit curve source requires canonical simulations.")
        selected = prepared.selected_observations
        x_observations = (
            None
            if prepared.parametric_x_observation is None
            else _series_from_selected(prepared.parametric_x_observation)
        )
        observed_x, observed_y = _observed_curve_parts(
            selected,
            x_observations=x_observations,
        )
        x_name = local.parametric.x_name if local.parametric is not None else "t"
        curves.append(
            FitCurveDataset(
                dataset_id=dataset_id,
                display_name=mapped.display_name,
                time_column_name=mapped.time_column_name,
                simulation=simulation,
                observed_x_unit=SECOND if prepared.parametric_x_observation is None else prepared.parametric_x_observation.unit,
                observed_y_units=tuple((item.model_target, item.unit) for item in selected),
                model_definitions=model_definitions,
                observed_x_name=(
                    None if prepared.parametric_x_observation is None
                    else prepared.parametric_x_observation.raw_column
                ),
                observed_x=observed_x,
                observed_y=observed_y,
                observed_time=tuple((item.model_target, item.t) for item in selected),
                observed_raw_columns=tuple(
                    (item.model_target, item.raw_column) for item in selected
                ),
                x_name=x_name,
            )
        )
    return tuple(curves)


def project_fit_candidate_curves(
    candidate: FitCandidate,
) -> tuple[FitCurveDataset, ...]:
    """Project terminal curves from the candidate's canonical result."""

    if not isinstance(candidate, FitCandidate):
        raise TypeError("candidate must be FitCandidate.")
    setup = candidate.setup_snapshot
    info_by_id = {info.dataset_id: info for info in candidate.result.dataset_info}
    return _curves_from_simulations(
        request=candidate.request,
        effective_local={item.dataset_id: item for item in setup.datasets},
        evaluations={
            dataset_id: info_by_id[dataset_id].simulation_result
            for dataset_id in candidate.request.dispatch.included_dataset_ids
        },
    )


def _assemble_fit_terminal(
    *,
    request: FitExecutionRequest,
    result: GlobalFitResult,
    included_ids: tuple[str, ...],
    effective_local: Mapping[str, FitDatasetLocalSettings],
    setup_snapshot: FitSetupSnapshot,
    evaluators: Mapping[str, FittingEvaluator],
) -> FitCandidate:
    """Validate terminal provenance and build the sole retained candidate."""

    admitted_layout = request.dispatch.ordinary_optimizer_layout
    if not isinstance(result.optimizer_vector, np.ndarray):
        raise TypeError("Fit terminal optimizer vector must be a NumPy array.")
    admitted_layout.validate_optimizer_vector(result.optimizer_vector)
    if result.optimizer_vector.flags.writeable:
        raise ValueError("Fit terminal optimizer vector must be read-only.")

    effective_dataset_weights: dict[str, float] = {}
    for info in result.dataset_info:
        dataset_id = str(info.dataset_id or "").strip()
        weight = float(info.weight)
        if (
            not dataset_id
            or dataset_id in effective_dataset_weights
            or not math.isfinite(weight)
            or weight <= 0.0
        ):
            raise ValueError(
                "Fit terminal contains invalid effective dataset weights."
            )
        effective_dataset_weights[dataset_id] = weight
    if set(effective_dataset_weights) != set(included_ids):
        raise ValueError(
            "Fit terminal effective weights do not exactly match the "
            "included datasets."
        )
    included_id_set = set(included_ids)
    if set(result.dataset_params) != included_id_set:
        raise ValueError(
            "Fit terminal parameter results do not exactly match the included datasets."
        )
    fixed_by_dataset = {
        dataset_id: dict(values)
        for dataset_id, values in request.dispatch.fixed_by_dataset
    }
    setup = request.setup
    for parameter in setup.parameters:
        if parameter.scope == "shared":
            values = (
                (result.shared_params[parameter.name],)
                if parameter.selected_config.fit
                and parameter.name in result.shared_params
                else tuple(
                    fixed_by_dataset[dataset_id][parameter.name]
                    for dataset_id in included_ids
                )
            )
            for value in values:
                admitted = admit_gui_parameter_value(
                    value,
                    scientific_category=parameter.scientific_category,
                    role=parameter.role,
                    unit=parameter.unit,
                    allow_exact_zero=parameter.allow_exact_zero,
                    field=f"terminal fitted {parameter.name!r}",
                )
                if not parameter.selected_config.minimum <= admitted <= parameter.selected_config.maximum:
                    raise ValueError(
                        f"Terminal fitted {parameter.name!r} violates retained bounds."
                    )
        else:
            for dataset_id in included_ids:
                source_values = (
                    result.dataset_params
                    if parameter.selected_config.fit
                    else fixed_by_dataset
                )
                admitted = admit_gui_parameter_value(
                    source_values[dataset_id][parameter.name],
                    scientific_category=parameter.scientific_category,
                    role=parameter.role,
                    unit=parameter.unit,
                    allow_exact_zero=parameter.allow_exact_zero,
                    field=f"terminal fitted {dataset_id!r} {parameter.name!r}",
                )
                if not parameter.selected_config.minimum <= admitted <= parameter.selected_config.maximum:
                    raise ValueError(
                        f"Terminal fitted {dataset_id!r} {parameter.name!r} "
                        "violates retained bounds."
                    )
    if set(evaluators) != included_id_set:
        raise ValueError(
            "Fit terminal prepared identities do not exactly match the included datasets."
        )
    solver_provenance_by_dataset = {
        str(info.dataset_id or "").strip(): info.simulation_result.solver_provenance
        for info in result.dataset_info
    }
    if (
        any(not dataset_id for dataset_id in solver_provenance_by_dataset)
        or set(solver_provenance_by_dataset) != set(included_ids)
    ):
        raise ValueError(
            "Fit terminal solver provenance does not exactly match the "
            "included datasets."
        )
    result = _settle_terminal_bounds(setup, result)
    initial_values_by_set = _initial_values_from_result(
        setup_snapshot,
        setup.origin,
        result,
    )
    terminal_setup = _terminal_setup_snapshot(
        setup_snapshot,
        origin=setup.origin,
        result=result,
        dispatch=request.dispatch,
        effective_datasets=effective_local,
        initial_values_by_set=initial_values_by_set,
    )
    candidate = FitCandidate(
        request=request,
        result=result,
        setup_snapshot=terminal_setup,
    )
    return candidate


class _FitControl:
    """Child-owned cancellation, stop and pause at scientific checkpoints."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._stopped = threading.Event()
        self._pause_condition = threading.Condition()
        self._paused = False

    def pause(self) -> None:
        with self._pause_condition:
            self._paused = True

    def resume(self) -> None:
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()

    def request_cancel(self) -> None:
        """Request cancellation without waiting for worker settlement."""

        self._cancelled.set()
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()

    def request_stop(self) -> None:
        """Stop active work while retaining this run's complete best candidate."""
        self._stopped.set()
        self.resume()

    def _checkpoint(self, *, honor_stop: bool = True) -> bool:
        with self._pause_condition:
            while self._paused and not (self._cancelled.is_set() or self._stopped.is_set()):
                self._pause_condition.wait()
        return self._cancelled.is_set() or (honor_stop and self._stopped.is_set())

class _FitExecution(_FitControl):
    """Numerical execution, including the existing dataset thread pool."""

    def __init__(self, request: FitExecutionRequest, emit: Callable) -> None:
        self._request = request
        self._emit = emit
        super().__init__()
        self._cancellation_control = SimulationCancellationControl(
            lambda: self._cancelled.is_set() or self._stopped.is_set()
        )
        self._last_best_emit = 0.0
        self._last_curve_emit = 0.0
        self._terminal_unavailable: ScientificUnavailable | None = None
        self._executor: ThreadPoolExecutor | None = None

    def run(self) -> None:
        candidate: FitCandidate | None = None
        failure: FactualCause | None = None
        cancelled = False
        unavailable: ScientificUnavailable | None = None
        try:
            candidate = self._execute()
        except FittingCancelled:
            cancelled = True
        except BaseException as exc:  # noqa: BLE001
            failure = factual_cause_from_exception(
                exc,
                domain="fit",
                origin="fit_worker",
            )
        else:
            if candidate is None:
                if self._checkpoint():
                    cancelled = True
                else:
                    unavailable = self._terminal_unavailable
            # A GlobalFitResult has already settled the primary fit and every
            # requested optional analysis.  Once assembled, the candidate is
            # terminal science; a late stop cannot veto it.
        finally:
            executor = self._executor
            self._executor = None
            if executor is not None:
                executor.shutdown(wait=True)

        # Publish one direct typed terminal only after every worker-owned
        # evaluator and executor has settled.  The owner receives the exact
        # candidate/cause/unavailability value; cancellation has no payload.
        if self._cancelled.is_set():
            self._emit("cancelled", )
        elif failure is not None:
            self._emit("factual_failure", failure)
        elif cancelled:
            self._emit("cancelled", )
        elif unavailable is not None:
            self._emit("scientific_unavailable", unavailable)
        elif candidate is not None:
            self._emit("candidate_ready", candidate)
        else:
            self._emit("factual_failure",
                factual_cause_from_exception(
                    RuntimeError("Fit worker exited without a terminal value."),
                    domain="fit",
                    origin="fit_worker",
                )
            )

    def _execute(
        self,
    ) -> FitCandidate | None:
        request = self._request
        setup = request.setup
        if self._checkpoint():
            raise FittingCancelled()
        included = tuple(item for item in setup.datasets if item.included)
        mapped_by_id = {item.dataset_id: item for item in setup.origin.mapped_datasets}
        dispatch = request.dispatch
        included_ids = dispatch.included_dataset_ids
        if tuple(item.dataset_id for item in included) != included_ids:
            raise RuntimeError("Fit worker included roster diverged from admitted dispatch.")
        fixed_by_dataset = {
            dataset_id: dict(values)
            for dataset_id, values in dispatch.fixed_by_dataset
        }
        candidate_names = list(dispatch.candidate_names)
        prepared_datasets = {
            item.dataset_id: item for item in dispatch.prepared_datasets
        }
        if set(prepared_datasets) != set(included_ids):
            raise RuntimeError("Fit worker lost an admitted dataset solver policy.")
        evaluators: dict[str, FittingEvaluator] = {}
        for local in included:
            if self._checkpoint():
                raise FittingCancelled()
            dataset_context = prepared_datasets[local.dataset_id]
            try:
                context = prepare_fitting_execution_context(
                    prepared_dsl=setup.origin.prepared_dsl,
                    param_names=candidate_names,
                    t_start=dataset_context.execution_t_start,
                    t_end=dataset_context.execution_t_end,
                    num_points=int(dataset_context.solver_config.grid["N"]),
                    objective_sample_times=dataset_context.retained_times,
                    prepared_solver_config=dataset_context.solver_config,
                    fit_sampling_policy=dataset_context.fit_sampling_policy,
                    fit_sampling_identity=dict(dataset_context.fit_sampling_identity),
                    cancellation_control=self._cancellation_control,
                    initial_prefix=INITIAL_PREFIX,
                )
            except BaseException as exc:
                if isinstance(exc, FitSimulationError):
                    exc.details.setdefault("fit_solver", setup.solver)
                raise
            evaluator = FittingEvaluator(
                context,
            )
            evaluator.set_cancellation_control(self._cancellation_control)
            evaluators[local.dataset_id] = evaluator

        self._executor = ThreadPoolExecutor(
            max_workers=min(len(included), os.cpu_count() or 1),
            thread_name_prefix="kindred-fit-dataset",
        )

        effective_local: dict[str, FitDatasetLocalSettings] = {}
        starting_evaluations = {}
        starting_facts: list[DatasetFitInfo] = []
        execution_shared_variables = dispatch.ordinary_shared_variables
        execution_dataset_overrides = dispatch.ordinary_dataset_overrides
        execution_coupled_variables = dispatch.ordinary_coupled_variables
        fitted_shared_names = {
            item.name for item in execution_shared_variables
        }
        fitted_dataset_names = {
            item.name
            for values in execution_dataset_overrides
            for item in values.variable_params
        }
        fitted_dataset_names.update(
            item.parameter_name for item in execution_coupled_variables
        )
        fitted_dataset_names.update(
            item.parameter_name
            for item in dispatch.ordinary_centred_variables
        )
        starting_weights = normalize_weights(
            [
                _compile_dataset_spec(
                    prepared_datasets[item.dataset_id],
                    item,
                )
                for item in included
            ],
            {item.dataset_id: item.weight for item in included}
            if setup.dataset_weight_mode == "user" else None,
        )
        starting_futures = {}
        for local in included:
            mapped = mapped_by_id[local.dataset_id]
            starting_values = {
                parameter.name: (
                    dict(parameter.selected_config.dataset_values)[local.dataset_id]
                    if parameter.scope in {"per_dataset", "centred_fixed_effects"}
                    else parameter.selected_config.value
                )
                for parameter in setup.parameters
            }
            starting_values.update({item.name: item.initial for item in execution_shared_variables})
            starting_values.update(fixed_by_dataset[local.dataset_id])
            grouped_species = {
                group.species
                for group in setup.shared_initial_groups
                if local.dataset_id in group.member_dataset_ids
            }
            for initial in setup.initials:
                if initial.set_id == mapped.set_id and initial.species not in grouped_species:
                    starting_values[f"{INITIAL_PREFIX}{initial.species}"] = initial.initial
            for group in setup.shared_initial_groups:
                if local.dataset_id in group.member_dataset_ids:
                    starting_values[f"{INITIAL_PREFIX}{group.species}"] = group.initial
            starting_origins = {
                name: (
                    FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED
                    if name in fitted_shared_names
                    else FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET
                    if name in fitted_dataset_names
                    else FITTING_PARAM_ORIGIN_CONFIGURED_DATASET
                )
                for name in starting_values
            }
            starting_payload = _compile_dataset_spec(
                prepared_datasets[local.dataset_id],
                local,
            )
            starting_input = ObjectiveDatasetInput(
                payload=starting_payload,
                full_params=starting_values,
                parameter_origins=starting_origins,
                failed_param_snapshot={},
            )
            starting_futures[local.dataset_id] = self._executor.submit(
                evaluate_dataset_simulation,
                evaluators,
                starting_input,
                cancellation_control=self._cancellation_control,
                weights=starting_weights,
            )
        for local in included:
            chosen = local
            dataset_id = local.dataset_id
            mapped = mapped_by_id[dataset_id]
            evaluation = starting_futures[dataset_id].result()
            if isinstance(evaluation, ScientificUnavailable):
                fit_info = None
            elif isinstance(evaluation, DatasetFitInfo):
                fit_info = evaluation
            else:
                raise TypeError("Fit evaluator did not return DatasetFitInfo.")
            if fit_info is not None:
                starting_evaluations[dataset_id] = fit_info.simulation_result
                starting_facts.append(fit_info)
            effective_local[local.dataset_id] = chosen
        datasets = [
            _compile_dataset_spec(
                prepared_datasets[item.dataset_id],
                effective_local[item.dataset_id],
            )
            for item in included
        ]
        if self._checkpoint(honor_stop=False):
            raise FittingCancelled()
        initial_curves = _curves_from_simulations(
            request=request,
            evaluations=starting_evaluations,
            effective_local=effective_local,
        )
        if self._checkpoint(honor_stop=False):
            raise FittingCancelled()
        self._emit("curves_updated",
            initial_curves
        )
        if self._checkpoint(honor_stop=False):
            raise FittingCancelled()
        self._last_curve_emit = time.monotonic()
        def progress(update: FitProgress) -> None:
            if self._checkpoint():
                raise FittingCancelled()
            objective_calls = update.objective_calls
            cost = update.cost
            self._emit("progress",
                min(
                    95,
                    max(
                        1,
                        int(
                            100
                            * objective_calls
                            / max(1, setup.defaults.max_nfev)
                        ),
                    ),
                ),
                f"Objective calls {objective_calls}: weighted objective SSQ={cost:.4g}",
            )
            if self._checkpoint():
                raise FittingCancelled()
            if update.improved:
                now = time.monotonic()
                if now - self._last_best_emit >= _BEST_INTERVAL_SECONDS:
                    if self._checkpoint():
                        raise FittingCancelled()
                    self._emit("best_updated", self._live_update(update))
                    if self._checkpoint():
                        raise FittingCancelled()
                    self._last_best_emit = now
                if now - self._last_curve_emit >= _CURVE_INTERVAL_SECONDS:
                    try:
                        live_curves = (
                            _curves_from_simulations(
                                request=request,
                                evaluations=update.transient_simulation_results_by_dataset,
                                effective_local=effective_local,
                            )
                            if update.transient_simulation_results_by_dataset
                            else ()
                        )
                    except (FitSimulationError, KeyError, TypeError, ValueError):
                        live_curves = ()
                    if live_curves and not self._cancelled.is_set():
                        if self._checkpoint():
                            raise FittingCancelled()
                        self._emit("curves_updated", live_curves)
                        if self._checkpoint():
                            raise FittingCancelled()
                    self._last_curve_emit = now

        weights = (
            {item.dataset_id: item.weight for item in included}
            if setup.dataset_weight_mode == "user"
            else None
        )
        extension_progress_value = 95

        def execution_cancellation_check() -> bool:
            return self._checkpoint(honor_stop=False)

        def extension_progress(update: FitExtensionProgress) -> None:
            nonlocal extension_progress_value
            if self._checkpoint():
                raise FittingCancelled()
            has_profile = setup.profile_likelihood is not None
            if update.stage == "primary_complete":
                extension_progress_value = max(extension_progress_value, 95)
                self._emit("progress", extension_progress_value, "Primary global fit complete")
                return
            if update.phase == "profile_likelihood":
                fraction = (
                    float(update.completed_work) / float(update.planned_work)
                    if update.planned_work
                    else 0.0
                )
                percent = 95 + min(2, int(math.floor(3.0 * fraction)))
                if update.stage == "discovery":
                    detail = (
                        f"Profile likelihood · {update.coordinate} · {update.side} "
                        f"discovery {update.discovery_step}/"
                        f"{setup.profile_likelihood.max_bracketing_steps_per_side}"
                    )
                elif update.stage == "refinement":
                    detail = (
                        f"Profile likelihood · {update.coordinate} · {update.side} "
                        f"refinement {update.refinement_step}/64"
                    )
                else:
                    detail = f"Profile likelihood · {update.status}"
            else:
                fraction = (
                    float(update.completed_replicates or 0)
                    / float(update.planned_replicates)
                    if update.planned_replicates
                    else 0.0
                )
                base = 97 if has_profile else 95
                percent = base + min(4, int(math.floor(4.0 * fraction)))
                detail = (
                    f"Bootstrap · replicate {update.completed_replicates or 0}/"
                    f"{update.planned_replicates or 0}"
                    if update.stage != "settlement"
                    else f"Bootstrap · {update.status}"
                )
            extension_progress_value = max(
                extension_progress_value,
                max(95, min(99, percent)),
            )
            self._emit("progress", extension_progress_value, detail)
            if self._checkpoint():
                raise FittingCancelled()

        execution_cancellation_control = SimulationCancellationControl(
            self._cancelled.is_set
        )
        for evaluator in evaluators.values():
            evaluator.set_cancellation_control(self._cancellation_control)
        defaults = request.setup.defaults
        result_or_unavailable = fit_global(
            fit_evaluators=evaluators,
            datasets=datasets,
            weights=weights,
            method=defaults.method,
            max_nfev=int(defaults.max_nfev),
            ftol=float(defaults.ftol),
            xtol=float(defaults.xtol),
            seed=int(defaults.seed) if defaults.use_seed else None,
            progress_callback=progress,
            extension_progress_callback=extension_progress,
            cancellation_check=execution_cancellation_check,
            cancellation_control=execution_cancellation_control,
            stop_check=self._stopped.is_set,
            starting_facts=tuple(sorted(starting_facts, key=lambda info: info.dataset_id)),
            profile_likelihood=request.setup.profile_likelihood,
            residual_bootstrap=request.setup.residual_bootstrap,
            dataset_overrides=request.dispatch.ordinary_dataset_overrides,
            optimizer_layout=request.dispatch.ordinary_optimizer_layout,
            executor=self._executor,
        )
        if isinstance(result_or_unavailable, ScientificUnavailable):
            self._terminal_unavailable = result_or_unavailable
            return None
        if not isinstance(result_or_unavailable, GlobalFitResult):
            raise TypeError("Fit core returned an unsupported terminal value.")
        result = result_or_unavailable
        candidate = _assemble_fit_terminal(
            request=request,
            result=result,
            included_ids=included_ids,
            effective_local=effective_local,
            setup_snapshot=setup,
            evaluators=evaluators,
        )
        self._emit("progress", 100, result.message)
        return candidate

    def _live_update(self, best_progress: FitProgress) -> FitLiveUpdate:
        """Build one transient presentation directly from callback-local progress."""
        dispatch = self._request.dispatch
        layout = dispatch.ordinary_optimizer_layout
        setup = self._request.setup
        vector = np.asarray(best_progress.optimizer_vector, dtype=float).reshape(-1)
        layout.validate_optimizer_vector(vector)
        coordinate_values = {
            coordinate: layout.decode_coordinate(coordinate, float(raw_value))
            for coordinate, raw_value in zip(
                layout.opt_param_keys(),
                vector,
                strict=True,
            )
        }
        included_dataset_ids = {
            item.dataset_id for item in setup.datasets if item.included
        }
        included_set_ids = {
            mapped.set_id
            for mapped in setup.origin.mapped_datasets
            if mapped.dataset_id in included_dataset_ids
        }
        initials_by_set: dict[str, dict[str, float]] = {}
        grouped = {
            (dataset_id, group.species)
            for group in setup.shared_initial_groups
            for dataset_id in group.member_dataset_ids
        }
        for initial in setup.initials:
            if (
                not initial.fit
                or initial.set_id not in included_set_ids
                or any(
                    (mapped.dataset_id, initial.species) in grouped
                    for mapped in setup.origin.mapped_datasets
                    if mapped.set_id == initial.set_id
                )
            ):
                continue
            variable_id = f"initial:{initial.set_id}:{initial.species}"
            value = coordinate_values.get(FitCoordinate("coupled", variable_id))
            if value is not None and math.isfinite(float(value)):
                initials_by_set.setdefault(initial.set_id, {})[
                    initial.species
                ] = float(value)
        for group in setup.shared_initial_groups:
            if not group.fit:
                continue
            value = coordinate_values.get(
                FitCoordinate("coupled", f"initial:{group.species}:{group.origin_dataset_id}")
            )
            if value is None or not math.isfinite(float(value)):
                continue
            applicable_set_ids = {
                mapped.set_id for mapped in setup.origin.mapped_datasets
                if mapped.dataset_id in included_dataset_ids
                and mapped.dataset_id in group.member_dataset_ids
            }
            for set_id in applicable_set_ids:
                initials_by_set.setdefault(set_id, {})[group.species] = float(value)
        centred_values_by_dataset = {
            dataset_id: layout.centred_params_for_dataset(dataset_id, vector)
            for dataset_id in included_dataset_ids
        }
        return FitLiveUpdate(
            iteration=best_progress.objective_calls,
            cost=best_progress.cost,
            parameters=_finite_items(
                {
                    _fit_coordinate_display_label(coordinate): value
                    for coordinate, value in coordinate_values.items()
                    if coordinate.scope not in {"dataset", "centred_deviation"}
                    and not (
                        coordinate.scope == "coupled"
                        and coordinate.name.startswith("initial:")
                    )
                },
                field="live Fit parameters",
            ),
            scalar_values_by_dataset=tuple(
                (
                    dataset_id,
                    _finite_items(
                        {
                            parameter.name: coordinate_values.get(
                                FitCoordinate("dataset", parameter.name, dataset_id),
                                centred_values_by_dataset[dataset_id].get(
                                    parameter.name,
                                    dict(parameter.selected_config.dataset_values)[dataset_id],
                                ),
                            )
                            for parameter in setup.parameters
                            if (
                                parameter.selected_config.fit
                                and parameter.scope in {"per_dataset", "centred_fixed_effects"}
                                and dataset_id
                                in dict(parameter.selected_config.dataset_values)
                            )
                        },
                        field=f"live Fit scalars {dataset_id}",
                    ),
                )
                for dataset_id in sorted(included_dataset_ids)
            ),
            initial_values_by_set=tuple(
                (
                    set_id,
                    tuple(sorted(values.items())),
                )
                for set_id, values in sorted(initials_by_set.items())
            ),
        )
