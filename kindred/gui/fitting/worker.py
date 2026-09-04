"""One pausable/cancellable Fit QThread over immutable typed captures."""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from threading import Condition, Event

import numpy as np
from PySide6 import QtCore

from kindred.core.analysis.fit_dataset_payload import (
    FitDatasetSpec,
    FitObservationSeries,
)
from kindred.core.analysis.global_fitting import fit_global
from kindred.core.analysis.global_fit_execution import (
    FitCoordinate,
    DatasetFitInfo,
    FitParameterLayout,
    FitProgress,
    GlobalFitResult,
    ObjectiveDatasetInput,
    evaluate_dataset_simulation,
    fit_result_target_values,
    _select_fit_rows,
)
from kindred.core.analysis.parametric_alignment import (
    align_parametric_x,
    project_parametric_y,
)
from kindred.core.analysis.fit_evaluation_outcome import (
    ScientificUnavailable,
)
from kindred.core.exceptions import (
    FitCandidateDomainError,
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
from kindred.core.simulator.solver_types import SimulationCancellationControl
from kindred.core.simulation_failure import (
    FactualCause,
    factual_cause_from_exception,
)
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
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
    "FitCurveYSemantic",
    "FitLiveUpdate",
    "GlobalFitWorker",
    "project_fit_candidate_curves",
]

_BEST_INTERVAL_SECONDS = 0.25
_CURVE_INTERVAL_SECONDS = 2.0


class FitCurveYSemantic(str, Enum):
    """Typed physical meaning of one projected Fit Y series."""

    CONCENTRATION = "concentration"
    VALUE = "value"


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


def _fit_coordinate_display_label(value: FitCoordinate) -> str:
    """Return a human label that cannot alias typed coordinate scopes."""

    if value.scope == "centred_overall":
        return f"Overall · {value.name}"
    if value.scope == "centred_deviation":
        return f"Deviation · {value.dataset_id}::{value.name}"
    if value.scope in {"shared", "coupled"}:
        return value.display_name
    return value.display_name


def _curve_items(
    items: tuple[tuple[str, tuple[float, ...]], ...],
    *,
    field: str,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    normalized: list[tuple[str, tuple[float, ...]]] = []
    for raw_name, raw_values in items:
        name = str(raw_name or "").strip()
        values = tuple(float(value) for value in raw_values)
        if not name or not values or any(
            not math.isfinite(value) for value in values
        ):
            raise ValueError(f"{field} contains an invalid curve.")
        normalized.append((name, values))
    names = [name for name, _values in normalized]
    if len(names) != len(set(names)):
        raise ValueError(f"{field} repeats a curve series.")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class FitCurveDataset:
    dataset_id: str
    display_name: str
    y_semantics: tuple[tuple[str, FitCurveYSemantic], ...]
    model_x: tuple[tuple[str, tuple[float, ...]], ...]
    model_y: tuple[tuple[str, tuple[float, ...]], ...]
    observed_x: tuple[tuple[str, tuple[float, ...]], ...]
    observed_y: tuple[tuple[str, tuple[float, ...]], ...]
    model_segment_ids: tuple[tuple[str, tuple[int, ...]], ...]
    model_segment_keys: tuple[tuple[str, tuple[str, ...]], ...]
    model_discontinuity_ids: tuple[tuple[str, tuple[int, ...]], ...]
    objective_model_x: tuple[tuple[str, tuple[float, ...]], ...]
    objective_model_y: tuple[tuple[str, tuple[float, ...]], ...]
    objective_model_segment_ids: tuple[tuple[str, tuple[int, ...]], ...]
    objective_model_segment_keys: tuple[tuple[str, tuple[str, ...]], ...]
    objective_model_discontinuity_ids: tuple[tuple[str, tuple[int, ...]], ...]
    objective_selection_provenance: tuple[tuple[str, tuple[str, ...]], ...]
    objective_observation_ids: tuple[tuple[str, tuple[str, ...]], ...]
    objective_raw_columns: tuple[tuple[str, tuple[str, ...]], ...]
    objective_source_rows: tuple[tuple[str, tuple[int, ...]], ...]
    x_name: str = "t"

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id or "").strip()
        display_name = str(self.display_name or "").strip()
        x_name = str(self.x_name or "").strip()
        if not dataset_id or not display_name or not x_name:
            raise ValueError("Fit curve dataset identity must be complete.")
        model_x = _curve_items(self.model_x, field="model X")
        model_y = _curve_items(self.model_y, field="model Y")
        observed_x = _curve_items(self.observed_x, field="observed X")
        observed_y = _curve_items(self.observed_y, field="observed Y")
        model_segment_ids = tuple(
            (str(name), tuple(int(value) for value in values))
            for name, values in self.model_segment_ids
        )
        model_segment_keys = tuple(
            (
                str(name),
                tuple(str(value or "").strip() for value in values),
            )
            for name, values in self.model_segment_keys
        )
        model_discontinuity_ids = tuple(
            (str(name), tuple(int(value) for value in values))
            for name, values in self.model_discontinuity_ids
        )
        objective_model_x = _curve_items(
            self.objective_model_x,
            field="objective model X",
        )
        objective_model_y = _curve_items(
            self.objective_model_y,
            field="objective model Y",
        )
        objective_model_segment_ids = tuple(
            (str(name), tuple(int(value) for value in values))
            for name, values in self.objective_model_segment_ids
        )
        objective_model_segment_keys = tuple(
            (str(name), tuple(str(value).strip() for value in values))
            for name, values in self.objective_model_segment_keys
        )
        objective_model_discontinuity_ids = tuple(
            (str(name), tuple(int(value) for value in values))
            for name, values in self.objective_model_discontinuity_ids
        )
        objective_selection_provenance = tuple(
            (str(name), tuple(str(value) for value in values))
            for name, values in self.objective_selection_provenance
        )
        objective_observation_ids = tuple(
            (str(name), tuple(str(value) for value in values))
            for name, values in self.objective_observation_ids
        )
        objective_raw_columns = tuple(
            (str(name), tuple(str(value) for value in values))
            for name, values in self.objective_raw_columns
        )
        objective_source_rows = tuple(
            (str(name), tuple(int(value) for value in values))
            for name, values in self.objective_source_rows
        )
        names = tuple(name for name, _values in model_y)
        if type(self.y_semantics) is not tuple or any(
            type(item) is not tuple or len(item) != 2
            for item in self.y_semantics
        ):
            raise ValueError(
                "Fit curves require immutable typed Y semantic entries."
            )
        y_semantics = self.y_semantics
        if (
            any(
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(semantic, FitCurveYSemantic)
                for name, semantic in y_semantics
            )
            or len(y_semantics) != len(names)
            or tuple(name for name, _semantic in y_semantics) != names
        ):
            raise ValueError(
                "Fit curves require one ordered typed Y semantic for every "
                "projected series."
            )
        segment_names = tuple(name for name, _values in model_segment_ids)
        segment_key_names = tuple(
            name for name, _values in model_segment_keys
        )
        discontinuity_names = tuple(name for name, _values in model_discontinuity_ids)
        if (
            not names
            or set(names) != {name for name, _values in model_x}
            or set(names) != {name for name, _values in observed_x}
            or set(names) != {name for name, _values in observed_y}
            or set(names) != set(segment_names)
            or set(names) != set(segment_key_names)
            or set(names) != set(discontinuity_names)
            or len(segment_names) != len(set(segment_names))
            or len(segment_key_names) != len(set(segment_key_names))
        ):
            raise ValueError(
                "Fit curves require complete model and observation axes "
                "for every selected series."
            )
        objective_maps = (
            dict(objective_model_x),
            dict(objective_model_y),
            dict(objective_model_segment_ids),
            dict(objective_model_segment_keys),
            dict(objective_model_discontinuity_ids),
            dict(objective_selection_provenance),
            dict(objective_observation_ids),
            dict(objective_raw_columns),
            dict(objective_source_rows),
        )
        observed_y_by_name = dict(observed_y)
        if any(set(mapping) != set(names) for mapping in objective_maps):
            raise ValueError("Fit objective rows require complete paired identity.")
        for name in names:
            length = len(objective_maps[1][name])
            if (
                length == 0
                or length != len(observed_y_by_name[name])
                or any(len(mapping[name]) != length for mapping in objective_maps)
                or any(value < 0 for value in objective_maps[2][name])
                or any(not value for value in objective_maps[3][name])
                or any(value < 0 for value in objective_maps[4][name])
                or any(value not in {"direct", "interpolated"} for value in objective_maps[5][name])
                or any(not value for value in objective_maps[6][name])
                or any(not value for value in objective_maps[7][name])
                or any(value <= 0 for value in objective_maps[8][name])
            ):
                raise ValueError("Fit objective rows contain incomplete paired identity.")
        for axis, values, field in (
            (dict(model_x), dict(model_y), "model"),
            (dict(observed_x), dict(observed_y), "observed"),
        ):
            if any(
                len(axis[name]) != len(values[name])
                for name in names
            ):
                raise ValueError(f"Fit {field} curve X/Y lengths do not align.")
        model_y_by_name = dict(model_y)
        segments_by_name = dict(model_segment_ids)
        segment_keys_by_name = dict(model_segment_keys)
        discontinuities_by_name = dict(model_discontinuity_ids)
        if any(
            len(segments_by_name[name]) != len(model_y_by_name[name])
            or any(value < 0 for value in segments_by_name[name])
            or len(segment_keys_by_name[name]) != len(model_y_by_name[name])
            or any(not value for value in segment_keys_by_name[name])
            or len(discontinuities_by_name[name]) != len(model_y_by_name[name])
            or any(value < 0 for value in discontinuities_by_name[name])
            for name in names
        ):
            raise ValueError(
                "Fit curves require exact nonnegative segment identity and "
                "positive source-row/raw-observation provenance."
            )
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "y_semantics", y_semantics)
        object.__setattr__(self, "x_name", x_name)
        object.__setattr__(self, "model_x", model_x)
        object.__setattr__(self, "model_y", model_y)
        object.__setattr__(self, "observed_x", observed_x)
        object.__setattr__(self, "observed_y", observed_y)
        object.__setattr__(self, "model_segment_ids", model_segment_ids)
        object.__setattr__(
            self,
            "model_segment_keys",
            model_segment_keys,
        )
        object.__setattr__(self, "model_discontinuity_ids", model_discontinuity_ids)
        object.__setattr__(self, "objective_model_x", objective_model_x)
        object.__setattr__(self, "objective_model_y", objective_model_y)
        object.__setattr__(self, "objective_model_segment_ids", objective_model_segment_ids)
        object.__setattr__(self, "objective_model_segment_keys", objective_model_segment_keys)
        object.__setattr__(self, "objective_model_discontinuity_ids", objective_model_discontinuity_ids)
        object.__setattr__(self, "objective_selection_provenance", objective_selection_provenance)
        object.__setattr__(self, "objective_observation_ids", objective_observation_ids)
        object.__setattr__(self, "objective_raw_columns", objective_raw_columns)
        object.__setattr__(self, "objective_source_rows", objective_source_rows)


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
    resolved_auto_modes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, FitExecutionRequest):
            raise TypeError("Fit candidate request must be retained.")
        if not isinstance(self.result, GlobalFitResult):
            raise TypeError("Fit candidate result must be canonical.")
        nominal_setup = self.setup_snapshot
        dispatch = self.request.dispatch
        resolved_auto_modes = tuple(
            (str(dataset_id or "").strip(), str(mode or "").strip())
            for dataset_id, mode in self.resolved_auto_modes
        )
        resolved_dataset_ids = [
            dataset_id for dataset_id, _mode in resolved_auto_modes
        ]
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
        expected_auto_dataset_ids = {
            item.dataset_id
            for item in nominal_setup.datasets
            if (
                item.included
                and item.parametric is not None
                and item.parametric.requested_mode == "auto"
            )
        }
        if (
            any(not dataset_id for dataset_id in resolved_dataset_ids)
            or len(resolved_dataset_ids) != len(set(resolved_dataset_ids))
            or set(resolved_dataset_ids) != expected_auto_dataset_ids
            or any(
                mode not in {"monotone", "time_guided"}
                for _dataset_id, mode in resolved_auto_modes
            )
        ):
            raise ValueError(
                "Fit candidate resolved parametric modes must exactly match "
                "the included parametric datasets."
            )
        object.__setattr__(
            self,
            "resolved_auto_modes",
            resolved_auto_modes,
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
    if parametric is not None and parametric.resolved_mode is None:
        raise ValueError(
            f"Dataset {prepared.dataset_id!r}: Auto parametric mode must be "
            "resolved once before execution lowering."
        )
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
        x_mode=(
            parametric.resolved_mode
            if parametric is not None
            else "monotone"
        ),
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


def _fit_curve_y_semantics(
    prepared_dsl: PreparedDslRequestSet,
    names: Sequence[str],
) -> tuple[tuple[str, FitCurveYSemantic], ...]:
    """Project Y meaning only from the prepared DSL typed species roster."""

    concentration_names = {
        str(name) for name in prepared_dsl.analysis.species_names
    }
    return tuple(
        (
            str(name),
            FitCurveYSemantic.CONCENTRATION
            if str(name) in concentration_names
            else FitCurveYSemantic.VALUE,
        )
        for name in names
    )


def _construct_fit_curve_dataset(
    *,
    dataset_id: str,
    display_name: str,
    y_semantics: tuple[tuple[str, FitCurveYSemantic], ...],
    model_x: tuple[tuple[str, tuple[float, ...]], ...],
    model_y: tuple[tuple[str, tuple[float, ...]], ...],
    observed_x: tuple[tuple[str, tuple[float, ...]], ...],
    observed_y: tuple[tuple[str, tuple[float, ...]], ...],
    model_segment_ids: tuple[tuple[str, tuple[int, ...]], ...],
    model_segment_keys: tuple[tuple[str, tuple[str, ...]], ...],
    model_discontinuity_ids: tuple[tuple[str, tuple[int, ...]], ...],
    objective_model_x: tuple[tuple[str, tuple[float, ...]], ...],
    objective_model_y: tuple[tuple[str, tuple[float, ...]], ...],
    objective_model_segment_ids: tuple[tuple[str, tuple[int, ...]], ...],
    objective_model_segment_keys: tuple[tuple[str, tuple[str, ...]], ...],
    objective_model_discontinuity_ids: tuple[tuple[str, tuple[int, ...]], ...],
    objective_selection_provenance: tuple[tuple[str, tuple[str, ...]], ...],
    objective_observation_ids: tuple[tuple[str, tuple[str, ...]], ...],
    objective_raw_columns: tuple[tuple[str, tuple[str, ...]], ...],
    objective_source_rows: tuple[tuple[str, tuple[int, ...]], ...],
    x_name: str,
) -> FitCurveDataset:
    """Construct the sole immutable curve value from explicit field facts."""

    return FitCurveDataset(
        dataset_id=dataset_id,
        display_name=display_name,
        y_semantics=y_semantics,
        model_x=model_x,
        model_y=model_y,
        observed_x=observed_x,
        observed_y=observed_y,
        model_segment_ids=model_segment_ids,
        model_segment_keys=model_segment_keys,
        model_discontinuity_ids=model_discontinuity_ids,
        objective_model_x=objective_model_x,
        objective_model_y=objective_model_y,
        objective_model_segment_ids=objective_model_segment_ids,
        objective_model_segment_keys=objective_model_segment_keys,
        objective_model_discontinuity_ids=objective_model_discontinuity_ids,
        objective_selection_provenance=objective_selection_provenance,
        objective_observation_ids=objective_observation_ids,
        objective_raw_columns=objective_raw_columns,
        objective_source_rows=objective_source_rows,
        x_name=x_name,
    )


def _curves_from_live(
    *,
    request: FitExecutionRequest,
    effective_local: Mapping[str, FitDatasetLocalSettings],
    evaluations: Mapping[str, SimulationResultValue],
) -> tuple[FitCurveDataset, ...]:
    """Adapt callback-local simulations into plotting curves."""

    mapped_by_id = {
        item.dataset_id: item for item in request.setup.origin.mapped_datasets
    }
    included_ids = request.dispatch.included_dataset_ids
    prepared_by_id = {
        item.dataset_id: item for item in request.dispatch.prepared_datasets
    }
    curves: list[FitCurveDataset] = []
    for dataset_id in included_ids:
        simulation = evaluations.get(dataset_id)
        if simulation is None:
            continue
        if not isinstance(simulation, SimulationResultValue):
            raise TypeError("Live curve source requires canonical simulations.")
        local = effective_local[dataset_id]
        mapped = mapped_by_id[dataset_id]
        prepared = prepared_by_id[dataset_id]
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
        model_names = tuple(item.model_target for item in selected)
        sim_time = np.asarray(simulation.t, dtype=float)
        segment_ids = np.asarray(simulation.segment_ids, dtype=np.int64)
        discontinuity_ids = np.asarray(
            simulation.discontinuity_ids,
            dtype=np.int64,
        )
        projection = None
        if x_name != "t":
            if x_observations is None or local.parametric is None:
                raise FitSimulationError(
                    f"Dataset {dataset_id!r}: Parametric-X capture is missing."
                )
            x_by_row = {
                int(row): (float(time_value), float(value))
                for row, time_value, value in zip(
                    x_observations.source_rows,
                    x_observations.t,
                    x_observations.y,
                    strict=True,
                )
            }
            x_rows = tuple(int(row) for row in x_observations.source_rows)
            projection = align_parametric_x(
                observation_time=np.asarray(
                    [x_by_row[row][0] for row in x_rows],
                    dtype=float,
                ),
                observed_x=np.asarray(
                    [x_by_row[row][1] for row in x_rows],
                    dtype=float,
                ),
                source_rows=x_rows,
                model_time=sim_time,
                model_x=fit_result_target_values(simulation, x_name),
                segment_ids=segment_ids,
                segment_keys=simulation.segment_keys,
                discontinuity_ids=discontinuity_ids,
                mode=local.parametric.resolved_mode,
                dataset_label=dataset_id,
                x_name=x_name,
            )

        projected_x: dict[str, tuple[float, ...]] = {}
        projected_y: dict[str, tuple[float, ...]] = {}
        projected_segment_ids: dict[str, tuple[int, ...]] = {}
        projected_segment_keys: dict[str, tuple[str, ...]] = {}
        projected_discontinuity_ids: dict[str, tuple[int, ...]] = {}
        projected_provenance: dict[str, tuple[str, ...]] = {}
        projected_observation_ids: dict[str, tuple[str, ...]] = {}
        projected_raw_columns: dict[str, tuple[str, ...]] = {}
        projected_source_rows: dict[str, tuple[int, ...]] = {}
        for item in selected:
            model_values = np.asarray(
                fit_result_target_values(simulation, item.model_target),
                dtype=float,
            )
            if x_name == "t":
                rows = _select_fit_rows(
                    model_y=model_values,
                    sim_time=sim_time,
                    observation_time=np.asarray(item.t, dtype=float),
                    observation_id=item.raw_observation_id,
                    raw_column=item.raw_column,
                    source_rows=item.source_rows,
                    segment_ids=segment_ids,
                    segment_keys=simulation.segment_keys,
                    discontinuity_ids=discontinuity_ids,
                )
                selected_model_x, selected_model_y, row_segments, row_keys, row_discontinuities, sample_kinds, observation_ids, raw_columns, source_rows = rows
            else:
                if projection is None:
                    raise RuntimeError("Parametric-X projection was not prepared.")
                y_projection = projection.subset(item.source_rows)
                selected_model_x = y_projection.selected_model_x
                selected_model_y = project_parametric_y(
                    projection=y_projection,
                    model_time=sim_time,
                    model_y=model_values,
                    segment_ids=segment_ids,
                    segment_keys=simulation.segment_keys,
                    discontinuity_ids=discontinuity_ids,
                )
                row_segments = y_projection.segment_ids
                row_keys = y_projection.segment_keys
                row_discontinuities = y_projection.discontinuity_ids
                sample_kinds = y_projection.sample_kinds
                observation_ids = tuple(
                    item.raw_observation_id for _ in item.source_rows
                )
                raw_columns = tuple(item.raw_column for _ in item.source_rows)
                source_rows = tuple(int(row) for row in item.source_rows)
            name = item.model_target
            projected_x[name] = tuple(float(value) for value in selected_model_x)
            projected_y[name] = tuple(float(value) for value in selected_model_y)
            projected_segment_ids[name] = tuple(int(value) for value in row_segments)
            projected_segment_keys[name] = tuple(str(value) for value in row_keys)
            projected_discontinuity_ids[name] = tuple(
                int(value) for value in row_discontinuities
            )
            projected_provenance[name] = tuple(str(value) for value in sample_kinds)
            projected_observation_ids[name] = tuple(str(value) for value in observation_ids)
            projected_raw_columns[name] = tuple(str(value) for value in raw_columns)
            projected_source_rows[name] = tuple(int(value) for value in source_rows)

        model_x = tuple((name, projected_x[name]) for name in model_names)
        model_y = tuple((name, projected_y[name]) for name in model_names)
        model_segment_ids = tuple(
            (name, projected_segment_ids[name]) for name in model_names
        )
        model_segment_keys = tuple(
            (name, projected_segment_keys[name]) for name in model_names
        )
        model_discontinuity_ids = tuple(
            (name, projected_discontinuity_ids[name]) for name in model_names
        )
        objective_model_x = model_x
        objective_model_y = model_y
        objective_model_segment_ids = model_segment_ids
        objective_model_segment_keys = model_segment_keys
        objective_model_discontinuity_ids = model_discontinuity_ids
        objective_selection_provenance = tuple(
            (name, projected_provenance[name]) for name in model_names
        )
        objective_observation_ids = tuple(
            (name, projected_observation_ids[name]) for name in model_names
        )
        objective_raw_columns = tuple(
            (name, projected_raw_columns[name]) for name in model_names
        )
        objective_source_rows = tuple(
            (name, projected_source_rows[name]) for name in model_names
        )
        curves.append(
            _construct_fit_curve_dataset(
                dataset_id=dataset_id,
                display_name=mapped.display_name,
                y_semantics=_fit_curve_y_semantics(request.setup.origin.prepared_dsl, model_names),
                model_x=model_x,
                model_y=model_y,
                observed_x=observed_x,
                observed_y=observed_y,
                model_segment_ids=model_segment_ids,
                model_segment_keys=model_segment_keys,
                model_discontinuity_ids=model_discontinuity_ids,
                objective_model_x=objective_model_x,
                objective_model_y=objective_model_y,
                objective_model_segment_ids=objective_model_segment_ids,
                objective_model_segment_keys=objective_model_segment_keys,
                objective_model_discontinuity_ids=objective_model_discontinuity_ids,
                objective_selection_provenance=objective_selection_provenance,
                objective_observation_ids=objective_observation_ids,
                objective_raw_columns=objective_raw_columns,
                objective_source_rows=objective_source_rows,
                x_name=x_name,
            )
        )
    return tuple(curves)


def _curves_from_terminal(
    *,
    request: FitExecutionRequest,
    effective_local: Mapping[str, FitDatasetLocalSettings],
    result: GlobalFitResult,
) -> tuple[FitCurveDataset, ...]:
    """Adapt canonical terminal result fields into plotting curves."""

    mapped_by_id = {
        item.dataset_id: item for item in request.setup.origin.mapped_datasets
    }
    info_by_id = {info.dataset_id: info for info in result.dataset_info}
    prepared_by_id = {
        item.dataset_id: item for item in request.dispatch.prepared_datasets
    }
    curves: list[FitCurveDataset] = []
    for dataset_id in request.dispatch.included_dataset_ids:
        local = effective_local[dataset_id]
        mapped = mapped_by_id[dataset_id]
        prepared = prepared_by_id[dataset_id]
        info = info_by_id[dataset_id]
        simulation = info.simulation_result
        if not isinstance(simulation, SimulationResultValue):
            raise TypeError("Terminal curve source requires canonical simulations.")
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
        model_names = tuple(item.model_target for item in selected)
        model_x_values = simulation.t if x_name == "t" else fit_result_target_values(simulation, x_name)
        model_x = tuple(
            (name, tuple(float(value) for value in model_x_values))
            for name in model_names
        )
        model_y = tuple(
            (
                name,
                tuple(float(value) for value in fit_result_target_values(simulation, name)),
            )
            for name in model_names
        )
        model_segment_ids = tuple(
            (name, tuple(int(value) for value in simulation.segment_ids))
            for name in model_names
        )
        model_segment_keys = tuple(
            (name, tuple(str(value) for value in simulation.segment_keys))
            for name in model_names
        )
        model_discontinuity_ids = tuple(
            (name, tuple(int(value) for value in simulation.discontinuity_ids))
            for name in model_names
        )
        objective_model_series = info.model_series
        objective_model_x = tuple(
            (
                name,
                tuple(float(value) for value in info.model_x_by_species[name]),
            )
            for name in objective_model_series
        )
        objective_model_y = tuple(
            (name, tuple(float(value) for value in objective_model_series[name]))
            for name in objective_model_series
        )
        objective_model_segment_ids = tuple(
            (
                name,
                tuple(int(value) for value in info.model_segment_ids[name]),
            )
            for name in objective_model_series
        )
        objective_model_segment_keys = tuple(
            (
                name,
                tuple(str(value) for value in info.model_segment_keys[name]),
            )
            for name in objective_model_series
        )
        objective_model_discontinuity_ids = tuple(
            (
                name,
                tuple(int(value) for value in info.model_discontinuity_ids[name]),
            )
            for name in objective_model_series
        )
        objective_selection_provenance = tuple(
            (name, tuple(info.model_selection_provenance[name]))
            for name in objective_model_series
        )
        objective_observation_ids = tuple(
            (name, tuple(info.model_observation_ids[name]))
            for name in objective_model_series
        )
        objective_raw_columns = tuple(
            (name, tuple(info.model_raw_columns[name]))
            for name in objective_model_series
        )
        objective_source_rows = tuple(
            (name, tuple(info.model_source_rows[name]))
            for name in objective_model_series
        )
        curves.append(
            _construct_fit_curve_dataset(
                dataset_id=dataset_id,
                display_name=mapped.display_name,
                y_semantics=_fit_curve_y_semantics(request.setup.origin.prepared_dsl, model_names),
                model_x=model_x,
                model_y=model_y,
                observed_x=observed_x,
                observed_y=observed_y,
                model_segment_ids=model_segment_ids,
                model_segment_keys=model_segment_keys,
                model_discontinuity_ids=model_discontinuity_ids,
                objective_model_x=objective_model_x,
                objective_model_y=objective_model_y,
                objective_model_segment_ids=objective_model_segment_ids,
                objective_model_segment_keys=objective_model_segment_keys,
                objective_model_discontinuity_ids=objective_model_discontinuity_ids,
                objective_selection_provenance=objective_selection_provenance,
                objective_observation_ids=objective_observation_ids,
                objective_raw_columns=objective_raw_columns,
                objective_source_rows=objective_source_rows,
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
    return _curves_from_terminal(
        request=candidate.request,
        effective_local={item.dataset_id: item for item in setup.datasets},
        result=candidate.result,
    )


def _assemble_fit_terminal(
    *,
    request: FitExecutionRequest,
    result: GlobalFitResult,
    included_ids: tuple[str, ...],
    effective_local: Mapping[str, FitDatasetLocalSettings],
    setup_snapshot: FitSetupSnapshot,
    resolved_modes: tuple[tuple[str, str], ...],
    evaluators: Mapping[str, FittingEvaluator],
) -> FitCandidate:
    """Validate terminal provenance and build the sole retained candidate."""

    admitted_layout = request.dispatch.ordinary_optimizer_layout
    if not isinstance(admitted_layout, FitParameterLayout):
        raise ValueError("Fit terminal omitted its admitted optimizer layout.")
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
    policies = {
        (item.dataset_id, item.parameter_name): item
        for item in request.dispatch.physical_coordinate_policies
    }
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
            policy = policies.get((None, parameter.name))
            if policy is None:
                raise ValueError(
                    f"Fit terminal lost physical policy for {parameter.name!r}."
                )
            for value in values:
                admitted = admit_gui_parameter_value(
                    value,
                    scientific_category=policy.scientific_category,
                    allow_exact_zero=policy.allow_exact_zero,
                    field=f"terminal fitted {parameter.name!r}",
                )
                if not policy.minimum <= admitted <= policy.maximum:
                    raise ValueError(
                        f"Terminal fitted {parameter.name!r} violates retained bounds."
                    )
        else:
            for dataset_id in included_ids:
                policy = policies.get((dataset_id, parameter.name))
                if policy is None:
                    raise ValueError(
                        f"Fit terminal lost physical policy for {dataset_id!r} "
                        f"{parameter.name!r}."
                    )
                source_values = (
                    result.dataset_params
                    if parameter.selected_config.fit
                    else fixed_by_dataset
                )
                admitted = admit_gui_parameter_value(
                    source_values[dataset_id][parameter.name],
                    scientific_category=policy.scientific_category,
                    allow_exact_zero=policy.allow_exact_zero,
                    field=f"terminal fitted {dataset_id!r} {parameter.name!r}",
                )
                if not policy.minimum <= admitted <= policy.maximum:
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
        resolved_auto_modes=resolved_modes,
    )
    return candidate


def _execute_global_fit_model(
    *,
    request: FitExecutionRequest,
    evaluators: Mapping[str, object],
    datasets: Sequence[FitDatasetSpec],
    weights: Mapping[str, float] | None,
    progress_callback: Callable[[FitProgress], None],
    extension_progress_callback: Callable[[FitExtensionProgress], None],
    cancellation_check: Callable[[], bool],
    cancellation_control: SimulationCancellationControl,
    executor: ThreadPoolExecutor,
) -> GlobalFitResult | ScientificUnavailable:
    """Dispatch one exact ordinary or advanced Fit execution."""

    common = {
        "fit_evaluators": evaluators,
        "datasets": datasets,
        "weights": weights,
        "method": request.setup.defaults.method,
        "max_nfev": int(request.setup.defaults.max_nfev),
        "ftol": float(request.setup.defaults.ftol),
        "xtol": float(request.setup.defaults.xtol),
        "seed": (
            int(request.setup.defaults.seed)
            if request.setup.defaults.use_seed
            else None
        ),
        "progress_callback": progress_callback,
        "extension_progress_callback": extension_progress_callback,
        "cancellation_check": cancellation_check,
        "cancellation_control": cancellation_control,
        "profile_likelihood": request.setup.profile_likelihood,
        "residual_bootstrap": request.setup.residual_bootstrap,
    }
    dispatch = request.dispatch
    layout = dispatch.ordinary_optimizer_layout
    if not isinstance(layout, FitParameterLayout):
        raise RuntimeError("Ordinary Fit omitted its admitted optimizer layout.")
    return fit_global(
        dataset_overrides=dispatch.ordinary_dataset_overrides,
        optimizer_layout=layout,
        executor=executor,
        **common,
    )


class GlobalFitWorker(QtCore.QThread):
    """Run one immutable request and retain every evaluator until terminal."""

    progress = QtCore.Signal(int, str)
    auto_modes_resolved = QtCore.Signal(object)
    best_updated = QtCore.Signal(object)
    curves_updated = QtCore.Signal(object)
    # Terminal values cross the trusted Qt boundary in their ordinary owned
    # type.  There is deliberately no terminal/outcome envelope: exactly one
    # of these signals is emitted after all borrowed evaluator work has
    # drained.
    candidate_ready = QtCore.Signal(object)
    factual_failure = QtCore.Signal(object)
    scientific_unavailable = QtCore.Signal(object)
    cancelled = QtCore.Signal()

    def __init__(
        self,
        request: FitExecutionRequest,
        *,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(request, FitExecutionRequest):
            raise TypeError("request must be FitExecutionRequest.")
        self._request = request
        self._cancelled = Event()
        self._cancellation_control = SimulationCancellationControl(
            self._cancelled.is_set
        )
        self._pause_condition = Condition()
        self._paused = False
        self._last_best_emit = 0.0
        self._last_curve_emit = 0.0
        self._terminal_unavailable: ScientificUnavailable | None = None
        self._executor: ThreadPoolExecutor | None = None

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
        self.requestInterruption()

    def _checkpoint(self) -> bool:
        with self._pause_condition:
            while self._paused and not self._cancelled.is_set():
                self._pause_condition.wait()
        return self._cancelled.is_set()

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
            if self._checkpoint():
                cancelled = True
            else:
                unavailable = self._terminal_unavailable
        finally:
            executor = self._executor
            self._executor = None
            if executor is not None:
                executor.shutdown(wait=True)

        # Publish one direct typed terminal only after every worker-owned
        # evaluator and executor has settled.  The owner receives the exact
        # candidate/cause/unavailability value; cancellation has no payload.
        if failure is not None:
            self.factual_failure.emit(failure)
        elif cancelled:
            self.cancelled.emit()
        elif unavailable is not None:
            self.scientific_unavailable.emit(unavailable)
        elif candidate is not None:
            self.candidate_ready.emit(candidate)
        else:
            self.factual_failure.emit(
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
        resolved_mode_items: list[tuple[str, str]] = []
        starting_evaluations = {}
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
        starting_weights = (
            {item.dataset_id: item.weight for item in included}
            if setup.dataset_weight_mode == "user"
            else {item.dataset_id: 1.0 for item in included}
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
            starting_local = local
            if (
                local.parametric is not None
                and local.parametric.requested_mode == "auto"
            ):
                starting_local = replace(
                    local,
                    parametric=replace(local.parametric, resolved_mode="monotone"),
                )
            starting_payload = _compile_dataset_spec(
                prepared_datasets[local.dataset_id],
                starting_local,
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
            if (
                local.parametric is not None
                and local.parametric.requested_mode == "auto"
            ):
                if fit_info is None:
                    # A candidate-domain failure removes the starting
                    # trajectory, so Auto must choose the deterministic
                    # Time-guided admission for subsequent candidates.
                    resolved_mode = "time_guided"
                    chosen = replace(
                        local,
                        parametric=replace(
                            local.parametric,
                            resolved_mode=resolved_mode,
                        ),
                    )
                else:
                    resolved_mode = "monotone"
                    chosen = replace(
                        local,
                        parametric=replace(
                            local.parametric,
                            resolved_mode=resolved_mode,
                        ),
                    )
            if (
                chosen.parametric is not None
                and chosen.parametric.requested_mode == "auto"
            ):
                resolved_mode = chosen.parametric.resolved_mode
                if resolved_mode is None:
                    raise ValueError(
                        f"Dataset {mapped.display_name!r}: parametric mode "
                        "remained unresolved after starting evaluation."
                    )
                resolved_mode_items.append(
                    (
                        chosen.dataset_id,
                        resolved_mode,
                    )
                )
            effective_local[local.dataset_id] = chosen
        datasets = [
            _compile_dataset_spec(
                prepared_datasets[item.dataset_id],
                effective_local[item.dataset_id],
            )
            for item in included
        ]
        resolved_modes = tuple(
            sorted(resolved_mode_items)
        )
        if self._checkpoint():
            raise FittingCancelled()
        self.auto_modes_resolved.emit(resolved_modes)
        if self._checkpoint():
            raise FittingCancelled()
        try:
            initial_curves = _curves_from_live(
                request=request,
                evaluations=starting_evaluations,
                effective_local=effective_local,
            )
        except FitCandidateDomainError:
            # The starting evaluation is scientific input to Auto-mode
            # resolution, but its aligned curve is only presentation.  A
            # candidate-domain failure therefore suppresses this optional
            # projection without preventing the optimizer from exploring the
            # admitted domain.  Fatal Fit failures remain job-fatal.
            initial_curves = ()
        if self._checkpoint():
            raise FittingCancelled()
        self.curves_updated.emit(
            initial_curves
        )
        if self._checkpoint():
            raise FittingCancelled()
        self._last_curve_emit = time.monotonic()
        def progress(update: FitProgress) -> None:
            if self._checkpoint():
                raise FittingCancelled()
            objective_calls = update.objective_calls
            cost = update.cost
            self.progress.emit(
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
                    self.best_updated.emit(self._live_update(update))
                    if self._checkpoint():
                        raise FittingCancelled()
                    self._last_best_emit = now
                if now - self._last_curve_emit >= _CURVE_INTERVAL_SECONDS:
                    try:
                        live_curves = (
                            _curves_from_live(
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
                        self.curves_updated.emit(live_curves)
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
            return self._checkpoint()

        def execution_cancellation_poll() -> bool:
            return self._cancelled.is_set()

        def extension_progress(update: FitExtensionProgress) -> None:
            nonlocal extension_progress_value
            if self._checkpoint():
                raise FittingCancelled()
            has_profile = setup.profile_likelihood is not None
            if update.stage == "primary_complete":
                extension_progress_value = max(extension_progress_value, 95)
                self.progress.emit(extension_progress_value, "Primary global fit complete")
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
            self.progress.emit(extension_progress_value, detail)
            if self._checkpoint():
                raise FittingCancelled()

        execution_cancellation_control = SimulationCancellationControl(
            execution_cancellation_poll
        )
        for evaluator in evaluators.values():
            evaluator.set_cancellation_control(execution_cancellation_control)
        result_or_unavailable = _execute_global_fit_model(
            request=request,
            evaluators=evaluators,
            datasets=datasets,
            weights=weights,
            progress_callback=progress,
            extension_progress_callback=extension_progress,
            cancellation_check=execution_cancellation_check,
            cancellation_control=execution_cancellation_control,
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
            resolved_modes=resolved_modes,
            evaluators=evaluators,
        )
        if self._checkpoint():
            raise FittingCancelled()
        self.progress.emit(100, "Fit complete")
        if self._checkpoint():
            raise FittingCancelled()
        return candidate

    def _live_update(self, best_progress: FitProgress) -> FitLiveUpdate:
        """Build one transient presentation directly from callback-local progress."""
        dispatch = self._request.dispatch
        layout = dispatch.ordinary_optimizer_layout
        if not isinstance(layout, FitParameterLayout):
            raise RuntimeError("Live Fit update omitted its admitted optimizer layout.")
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
