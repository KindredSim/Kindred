"""One pausable/cancellable Fit QThread over immutable typed captures."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from threading import Condition, Event

import numpy as np
from PySide6 import QtCore

from kindred.core.algebra.observable_introspection import FitLocalObservable
from kindred.core.analysis.dataset_parameter_overrides import (
    FitCoupledVariableParamSpec,
    FitDatasetParameterOverrides,
    FitDatasetVariableParamSpec,
    FitSharedVariableParamSpec,
)
from kindred.core.analysis.fit_dataset_payload import (
    FitDatasetSpec,
    FitObservationSeries,
)
from kindred.core.analysis.global_fitting import GlobalFitResult, fit_global
from kindred.core.analysis.global_fit_execution import select_fit_rows
from kindred.core.analysis.parametric_alignment import (
    DeterministicParametricResolutionError,
    align_y_on_x_obs,
)
from kindred.core.exceptions import FitSimulationError, FittingCancelled
from kindred.core.fitting_evaluation import (
    FITTING_PARAM_ORIGIN_CONFIGURED_DATASET,
    SerialFittingEvaluator,
    prepare_fitting_execution_context,
)
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import SimulationExecutionSettings
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.wegscheider_symbolic import (
    UnresolvedWegscheiderCyclicityError,
    WegscheiderCyclicityReport,
)
from kindred.gui.fitting.constants import INITIAL_PREFIX
from kindred.gui.fitting.contracts import (
    FitRunIdentity,
)
from kindred.gui.fitting.launch import (
    FitDatasetLocalSettings,
    FitExecutionRequest,
    FitMappedDatasetSnapshot,
    FitSelectedObservation,
    FitSetupSnapshot,
    selected_observations,
)
from kindred.gui.widgets.mechanism_editor import (
    DetailedBalanceCorrectionRequest,
)

__all__ = [
    "FitCandidate",
    "FitCurveDataset",
    "FitLiveUpdate",
    "FitWorkerTerminal",
    "GlobalFitWorker",
]

_BEST_INTERVAL_SECONDS = 0.25
_CURVE_INTERVAL_SECONDS = 2.0


class _DetailedBalanceCorrectionRequired(RuntimeError):
    def __init__(self, request: DetailedBalanceCorrectionRequest) -> None:
        super().__init__(request.issue)
        self.request = request


def _unresolved_detailed_balance_error(
    error: BaseException,
) -> UnresolvedWegscheiderCyclicityError | None:
    """Find only the typed core correction requirement in one cause chain."""

    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(current, UnresolvedWegscheiderCyclicityError):
            return current
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException):
                pending.append(linked)
    return None


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


def _stamp_bound(value: float) -> float | None:
    number = float(value)
    if math.isfinite(number):
        return number
    return None


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
    model_x: tuple[tuple[str, tuple[float, ...]], ...]
    model_y: tuple[tuple[str, tuple[float, ...]], ...]
    observed_x: tuple[tuple[str, tuple[float, ...]], ...]
    observed_y: tuple[tuple[str, tuple[float, ...]], ...]
    model_segment_ids: tuple[tuple[str, tuple[int, ...]], ...]
    model_segment_keys: tuple[tuple[str, tuple[str, ...]], ...]
    model_discontinuity_ids: tuple[tuple[str, tuple[int, ...]], ...]
    observed_source_rows: tuple[tuple[str, tuple[int, ...]], ...]
    observed_raw_ids: tuple[tuple[str, str], ...]
    observed_raw_columns: tuple[tuple[str, str], ...]
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
        observed_source_rows = tuple(
            (str(name), tuple(int(value) for value in values))
            for name, values in self.observed_source_rows
        )
        observed_raw_ids = tuple(
            (str(name), str(raw_id))
            for name, raw_id in self.observed_raw_ids
        )
        observed_raw_columns = tuple(
            (str(name), str(raw_column))
            for name, raw_column in self.observed_raw_columns
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
        segment_names = tuple(name for name, _values in model_segment_ids)
        segment_key_names = tuple(
            name for name, _values in model_segment_keys
        )
        source_row_names = tuple(name for name, _values in observed_source_rows)
        raw_id_names = tuple(name for name, _raw_id in observed_raw_ids)
        raw_column_names = tuple(
            name for name, _raw_column in observed_raw_columns
        )
        discontinuity_names = tuple(name for name, _values in model_discontinuity_ids)
        if (
            not names
            or set(names) != {name for name, _values in model_x}
            or set(names) != {name for name, _values in observed_x}
            or set(names) != {name for name, _values in observed_y}
            or set(names) != set(segment_names)
            or set(names) != set(segment_key_names)
            or set(names) != set(source_row_names)
            or set(names) != set(raw_id_names)
            or set(names) != set(raw_column_names)
            or set(names) != set(discontinuity_names)
            or len(segment_names) != len(set(segment_names))
            or len(segment_key_names) != len(set(segment_key_names))
            or len(source_row_names) != len(set(source_row_names))
            or len(raw_id_names) != len(set(raw_id_names))
            or len(raw_column_names) != len(set(raw_column_names))
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
        rows_by_name = dict(observed_source_rows)
        raw_ids_by_name = dict(observed_raw_ids)
        raw_columns_by_name = dict(observed_raw_columns)
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
                or any(
                    value != raw_ids_by_name[name]
                    for value in objective_maps[6][name]
                )
                or any(
                    value != raw_columns_by_name[name]
                    for value in objective_maps[7][name]
                )
                or tuple(objective_maps[8][name]) != rows_by_name[name]
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
            or len(rows_by_name[name]) != len(observed_y_by_name[name])
            or any(value <= 0 for value in rows_by_name[name])
            or not raw_ids_by_name[name].strip()
            or not raw_columns_by_name[name].strip()
            for name in names
        ):
            raise ValueError(
                "Fit curves require exact nonnegative segment identity and "
                "positive source-row/raw-observation provenance."
            )
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "display_name", display_name)
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
        object.__setattr__(
            self,
            "observed_source_rows",
            observed_source_rows,
        )
        object.__setattr__(self, "observed_raw_ids", observed_raw_ids)
        object.__setattr__(self, "observed_raw_columns", observed_raw_columns)


@dataclass(frozen=True, slots=True)
class FitLiveUpdate:
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
    """Scientifically valid terminal result, still local to the Fit window."""

    origin_tab_id: str
    origin_capture_id: str
    mechanism_source: MechanismAuthoringSource
    prepared_dsl: PreparedDslRequestSet
    parameters: tuple[tuple[str, float], ...]
    scalar_values_by_dataset: tuple[
        tuple[str, tuple[tuple[str, float], ...]], ...
    ]
    initial_values_by_set: tuple[
        tuple[str, tuple[tuple[str, float], ...]], ...
    ]
    execution_by_set: tuple[tuple[str, SimulationExecutionSettings], ...]
    local_observables: tuple[FitLocalObservable, ...]
    result: GlobalFitResult
    curves: tuple[FitCurveDataset, ...]
    run_identity: FitRunIdentity
    setup_snapshot: FitSetupSnapshot
    resolved_auto_modes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prepared_dsl, PreparedDslRequestSet):
            raise TypeError("Fit candidate prepared_dsl must be retained.")
        if (
            self.prepared_dsl.analysis.normalized_text
            != self.mechanism_source.reactions_text.strip()
        ):
            raise ValueError(
                "Fit candidate prepared DSL must exactly match its source."
            )
        resolved_auto_modes = tuple(
            (str(dataset_id or "").strip(), str(mode or "").strip())
            for dataset_id, mode in self.resolved_auto_modes
        )
        resolved_dataset_ids = [
            dataset_id for dataset_id, _mode in resolved_auto_modes
        ]
        expected_dataset_ids = {
            item.dataset_id
            for item in self.setup_snapshot.datasets
            if (
                item.included
                and item.parametric is not None
                and item.parametric.requested_mode == "auto"
            )
        }
        if (
            any(not dataset_id for dataset_id in resolved_dataset_ids)
            or len(resolved_dataset_ids) != len(set(resolved_dataset_ids))
            or set(resolved_dataset_ids) != expected_dataset_ids
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


@dataclass(frozen=True, slots=True)
class FitWorkerTerminal:
    """The one result retained by a worker until native ``finished``."""

    candidate: FitCandidate | None
    message: str
    cancelled: bool = False

    def __post_init__(self) -> None:
        message = str(self.message or "").strip()
        if self.candidate is not None:
            if not isinstance(self.candidate, FitCandidate):
                raise TypeError("Fit worker candidate must be FitCandidate.")
            if message or self.cancelled:
                raise ValueError(
                    "A successful Fit terminal cannot also be failed or cancelled."
                )
        elif not message:
            raise ValueError("A non-success Fit terminal requires a message.")
        object.__setattr__(self, "message", message)


def _observed_curve_parts(
    observations: tuple[FitSelectedObservation, ...],
    *,
    x_by_target: Mapping[str, np.ndarray] | None = None,
) -> tuple[
    tuple[tuple[str, tuple[float, ...]], ...],
    tuple[tuple[str, tuple[float, ...]], ...],
]:
    axes = dict(x_by_target or {})
    return (
        tuple(
            (
                item.model_species,
                tuple(
                    float(value)
                    for value in axes.get(
                        item.model_species,
                        np.asarray(item.t, dtype=float),
                    )
                ),
            )
            for item in observations
        ),
        tuple((item.model_species, item.y) for item in observations),
    )


def _observed_x_by_target(
    mapped: FitMappedDatasetSnapshot,
    observations: tuple[FitSelectedObservation, ...],
    local: FitDatasetLocalSettings,
) -> dict[str, np.ndarray]:
    parametric = local.parametric
    if parametric is None:
        return {}
    if any(item.model_species == parametric.x_name for item in observations):
        raise ValueError(
            f"Dataset {mapped.display_name!r}: selected X={parametric.x_name!r} "
            "cannot also be a fitted Y target."
        )
    sources = [
        item for item in mapped.observations
        if item.model_species == parametric.x_name
    ]
    if len(sources) != 1:
        raise ValueError(
            f"Dataset {mapped.display_name!r} requires exactly one observed "
            f"source for X={parametric.x_name!r}."
        )
    source = sources[0]
    x_by_row = {
        int(row): float(value)
        for row, value in zip(source.source_rows, source.y, strict=True)
    }
    aligned: dict[str, np.ndarray] = {}
    for observation in observations:
        missing = [
            row for row in observation.source_rows if int(row) not in x_by_row
        ]
        if missing:
            raise ValueError(
                f"Dataset {mapped.display_name!r}: X={parametric.x_name!r} "
                f"is missing {len(missing)} physical source rows for "
                f"Y={observation.model_species!r}."
            )
        aligned[observation.model_species] = np.asarray(
            [x_by_row[int(row)] for row in observation.source_rows],
            dtype=float,
        )
    return aligned


def _dataset_spec(
    mapped: FitMappedDatasetSnapshot,
    local: FitDatasetLocalSettings,
) -> FitDatasetSpec:
    parametric = local.parametric
    if parametric is not None and parametric.resolved_mode is None:
        raise ValueError(
            f"Dataset {mapped.display_name!r}: Auto parametric mode must be "
            "resolved once before execution lowering."
        )
    selected = selected_observations(mapped, local)
    observation_map = {
        item.model_species: FitObservationSeries(
            raw_observation_id=item.raw_observation_id,
            raw_column=item.raw_column,
            t=np.asarray(item.t, dtype=float),
            y=np.asarray(item.y, dtype=float),
            source_rows=item.source_rows,
        )
        for item in selected
    }
    x_by_target = _observed_x_by_target(mapped, selected, local)
    return FitDatasetSpec(
        dataset_id=mapped.dataset_id,
        species_list=tuple(item.model_species for item in selected),
        observations=observation_map,
        target_weights=dict(local.target_weights),
        x_name=parametric.x_name if parametric is not None else "t",
        x_mode=(
            parametric.resolved_mode
            if parametric is not None
            else "monotone"
        ),
        x_obs_by_species=x_by_target,
    )


def _initial_values_from_result(
    request: FitExecutionRequest,
    result: GlobalFitResult,
) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    fitted_by_key: dict[tuple[str, str], object] = {}
    for initial in request.initials:
        if initial.fit:
            key = (initial.set_id, initial.species)
            if key in fitted_by_key:
                raise ValueError(
                    "Fit result projection repeats a fitted initial row."
                )
            fitted_by_key[key] = initial
    included_ids = {
        item.dataset_id for item in request.datasets if item.included
    }
    datasets_by_set: dict[str, list[str]] = {}
    for mapped in request.origin.mapped_datasets:
        if mapped.dataset_id in included_ids:
            datasets_by_set.setdefault(mapped.set_id, []).append(
                mapped.dataset_id
            )
    values_by_set: dict[str, dict[str, float]] = {}
    global_values: dict[str, set[float]] = {}
    for (set_id, species), initial in sorted(fitted_by_key.items()):
        dataset_ids = sorted(datasets_by_set.get(set_id, ()))
        if not dataset_ids:
            continue
        parameter_name = f"{INITIAL_PREFIX}{species}"
        coupled_values: set[float] = set()
        for dataset_id in dataset_ids:
            dataset_result = result.dataset_params.get(dataset_id)
            if dataset_result is None or parameter_name not in dataset_result:
                raise ValueError(
                    f"Fit result omitted coupled initial {species!r} for "
                    f"included dataset {dataset_id!r}."
                )
            value = float(dataset_result[parameter_name])
            if not math.isfinite(value):
                raise ValueError(
                    f"Fit result contains a non-finite initial {species!r}."
                )
            coupled_values.add(value)
        if len(coupled_values) != 1:
            raise ValueError(
                f"Fit result disagrees across coupled datasets for initial "
                f"{species!r} in set {set_id!r}."
            )
        value = coupled_values.pop()
        values_by_set.setdefault(set_id, {})[species] = value
        if getattr(initial, "scope", "") == "global":
            global_values.setdefault(species, set()).add(value)
    inconsistent_global = sorted(
        species
        for species, values in global_values.items()
        if len(values) != 1
    )
    if inconsistent_global:
        raise ValueError(
            "Fit result disagrees across globally coupled initial rows: "
            + ", ".join(inconsistent_global)
        )
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
    request: FitExecutionRequest,
    result: GlobalFitResult,
    effective_datasets: Mapping[str, FitDatasetLocalSettings],
    initial_values_by_set: tuple[
        tuple[str, tuple[tuple[str, float], ...]], ...
    ],
) -> FitSetupSnapshot:
    """Overlay only successful terminal values onto immutable launch rows."""

    included_ids = {
        item.dataset_id for item in request.datasets if item.included
    }
    included_set_ids = {
        mapped.set_id
        for mapped in request.origin.mapped_datasets
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
    request_parameters = {
        parameter.name: parameter
        for parameter in request.parameters
    }
    parameters = []
    for parameter in setup_snapshot.parameters:
        requested = request_parameters.get(parameter.name)
        if requested is None:
            raise ValueError(
                f"Terminal Fit setup contains unknown parameter "
                f"{parameter.name!r}."
            )
        dataset_values = tuple(parameter.dataset_values)
        if not requested.fit:
            parameters.append(
                replace(parameter, dataset_values=dataset_values)
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
                replace(
                    parameter,
                    initial=value,
                    dataset_values=dataset_values,
                )
            )
            continue
        expected_ids = {
            dataset_id
            for dataset_id, _value in requested.dataset_values
        }
        if not included_ids.issubset(expected_ids):
            raise ValueError(
                f"Per-dataset parameter {parameter.name!r} no longer "
                "matches the included dataset identities."
            )
        fitted_by_id = dict(requested.dataset_values)
        for dataset_id, _launch_value in requested.dataset_values:
            if dataset_id not in included_ids:
                continue
            values = dataset_results.get(dataset_id)
            if values is None or parameter.name not in values:
                raise ValueError(
                    f"Fit result omitted per-dataset parameter "
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
            for dataset_id, _value in requested.dataset_values
        ]
        fitted_unique_values = {
            value for _dataset_id, value in fitted_values
        }
        parameters.append(
            replace(
                parameter,
                initial=(
                    fitted_unique_values.pop()
                    if len(fitted_unique_values) == 1
                    else parameter.initial
                ),
                dataset_values=tuple(fitted_values),
            )
        )

    initial_results = {
        (set_id, species): value
        for set_id, values in initial_values_by_set
        for species, value in values
    }
    global_initial_results: dict[str, float] = {}
    for initial in request.initials:
        if not initial.fit or initial.scope != "global":
            continue
        values = {
            value
            for (set_id, species), value in initial_results.items()
            if species == initial.species
        }
        if len(values) != 1:
            raise ValueError(
                f"Fit result did not resolve one global value for initial "
                f"{initial.species!r}."
            )
        global_initial_results[initial.species] = values.pop()
    initials = tuple(
        replace(
            initial,
            initial=(
                global_initial_results[initial.species]
                if (
                    initial.scope == "global"
                    and initial.set_id in included_set_ids
                )
                else initial_results[(initial.set_id, initial.species)]
            ),
        )
        if (
            initial.fit
            and initial.set_id in included_set_ids
            and (
                initial.species in global_initial_results
                or (initial.set_id, initial.species) in initial_results
            )
        )
        else initial
        for initial in setup_snapshot.initials
    )
    parameter_by_name = {
        parameter.name: parameter
        for parameter in parameters
    }
    terminal_scalar_values: dict[str, float] = {}
    for name, parameter in parameter_by_name.items():
        if parameter.fit and parameter.scope == "per_dataset":
            values = {
                float(value)
                for _dataset_id, value in parameter.dataset_values
            }
            if len(values) == 1:
                terminal_scalar_values[name] = values.pop()
                continue
        terminal_scalar_values[name] = parameter.initial
    local_observables = tuple(
        replace(
            observable,
            scalar_parameters=tuple(
                replace(
                    scalar,
                    value=terminal_scalar_values[scalar.name],
                    minimum=parameter_by_name[scalar.name].minimum,
                    maximum=parameter_by_name[scalar.name].maximum,
                    scope=parameter_by_name[scalar.name].scope,
                )
                for scalar in observable.scalar_parameters
            ),
        )
        for observable in setup_snapshot.local_observables
    )
    return replace(
        setup_snapshot,
        datasets=tuple(
            effective_datasets.get(item.dataset_id, item)
            for item in setup_snapshot.datasets
        ),
        parameters=tuple(parameters),
        initials=initials,
        local_observables=local_observables,
    )


def _curves_from_result(
    request: FitExecutionRequest,
    result: GlobalFitResult,
) -> tuple[FitCurveDataset, ...]:
    mapped_by_id = {item.dataset_id: item for item in request.origin.mapped_datasets}
    local_by_id = {item.dataset_id: item for item in request.datasets}
    curves: list[FitCurveDataset] = []
    for dataset_id, model_series in result.plot_model_series.items():
        mapped = mapped_by_id[dataset_id]
        local = local_by_id[dataset_id]
        selected = selected_observations(mapped, local)
        observed_x, observed_y = _observed_curve_parts(
            selected,
            x_by_target=result.plot_observed_x.get(dataset_id),
        )
        axes = result.plot_model_x_by_species.get(dataset_id, {})
        curves.append(
            FitCurveDataset(
                dataset_id=dataset_id,
                display_name=mapped.display_name,
                model_x=tuple(
                    (
                        species,
                        tuple(float(value) for value in np.asarray(axes[species])),
                    )
                    for species in model_series
                    if species in axes
                ),
                model_y=tuple(
                    (
                        species,
                        tuple(float(value) for value in np.asarray(values)),
                    )
                    for species, values in model_series.items()
                    if species in axes
                ),
                observed_x=observed_x,
                observed_y=observed_y,
                model_segment_ids=tuple(
                    (
                        species,
                        tuple(
                            int(value)
                            for value in np.asarray(
                                result.plot_model_segment_ids[
                                    dataset_id
                                ][species],
                                dtype=np.int64,
                            )
                        ),
                    )
                    for species in model_series
                    if species in axes
                ),
                model_segment_keys=tuple(
                    (
                        species,
                        tuple(
                            result.plot_model_segment_keys[
                                dataset_id
                            ][species]
                        ),
                    )
                    for species in model_series
                    if species in axes
                ),
                model_discontinuity_ids=tuple(
                    (
                        species,
                        tuple(
                            int(value)
                            for value in np.asarray(
                                result.plot_model_discontinuity_ids[
                                    dataset_id
                                ][species],
                                dtype=np.int64,
                            )
                        ),
                    )
                    for species in model_series
                    if species in axes
                ),
                observed_source_rows=tuple(
                    (
                        species,
                        tuple(
                            result.plot_observed_source_rows[
                                dataset_id
                            ][species]
                        ),
                    )
                    for species in model_series
                    if species in axes
                ),
                observed_raw_ids=tuple(
                    (
                        species,
                        result.plot_observed_raw_ids[
                            dataset_id
                        ][species],
                    )
                    for species in model_series
                    if species in axes
                ),
                observed_raw_columns=tuple(
                    (item.model_species, item.raw_column)
                    for item in selected
                    if item.model_species in model_series
                    and item.model_species in axes
                ),
                objective_model_x=tuple(
                    (
                        species,
                        tuple(
                            float(value)
                            for value in result.model_x_by_species[dataset_id][species]
                        ),
                    )
                    for species in model_series
                    if species in axes
                ),
                objective_model_y=tuple(
                    (
                        species,
                        tuple(float(value) for value in result.model_series[dataset_id][species]),
                    )
                    for species in model_series
                    if species in axes
                ),
                objective_model_segment_ids=tuple(
                    (
                        species,
                        tuple(int(value) for value in result.model_segment_ids[dataset_id][species]),
                    )
                    for species in model_series
                    if species in axes
                ),
                objective_model_segment_keys=tuple(
                    (species, tuple(result.model_segment_keys[dataset_id][species]))
                    for species in model_series
                    if species in axes
                ),
                objective_model_discontinuity_ids=tuple(
                    (
                        species,
                        tuple(int(value) for value in result.model_discontinuity_ids[dataset_id][species]),
                    )
                    for species in model_series
                    if species in axes
                ),
                objective_selection_provenance=tuple(
                    (species, tuple(result.model_selection_provenance[dataset_id][species]))
                    for species in model_series
                    if species in axes
                ),
                objective_observation_ids=tuple(
                    (species, tuple(result.model_observation_ids[dataset_id][species]))
                    for species in model_series
                    if species in axes
                ),
                objective_raw_columns=tuple(
                    (species, tuple(result.model_raw_columns[dataset_id][species]))
                    for species in model_series
                    if species in axes
                ),
                objective_source_rows=tuple(
                    (species, tuple(result.model_source_rows[dataset_id][species]))
                    for species in model_series
                    if species in axes
                ),
                x_name=(
                    local.parametric.x_name
                    if local.parametric is not None
                    else "t"
                ),
            )
        )
    return tuple(curves)


def _curves_from_evaluations(
    request: FitExecutionRequest,
    evaluations: Mapping[str, object],
    effective_local: Mapping[str, FitDatasetLocalSettings],
) -> tuple[FitCurveDataset, ...]:
    """Build Fit-local live curves from exact typed evaluator payloads."""

    mapped_by_id = {
        item.dataset_id: item for item in request.origin.mapped_datasets
    }
    curves: list[FitCurveDataset] = []
    for dataset_id, raw_evaluation in evaluations.items():
        if not hasattr(raw_evaluation, "series") or not hasattr(
            raw_evaluation,
            "segment_ids",
        ):
            raise TypeError("Live Fit evaluation lost its typed series payload.")
        evaluation = raw_evaluation
        local = effective_local[dataset_id]
        mapped = mapped_by_id[dataset_id]
        selected = selected_observations(mapped, local)
        observed_axes = _observed_x_by_target(mapped, selected, local)
        observed_x, observed_y = _observed_curve_parts(
            selected,
            x_by_target=observed_axes,
        )
        series = evaluation.series
        segment_ids = tuple(
            int(value)
            for value in np.asarray(
                evaluation.segment_ids,
                dtype=np.int64,
            ).reshape(-1)
        )
        segment_keys = tuple(
            str(value)
            for value in evaluation.segment_keys
        )
        discontinuity_ids = tuple(
            int(value)
            for value in np.asarray(
                evaluation.discontinuity_ids,
                dtype=np.int64,
            ).reshape(-1)
        )
        x_name = (
            local.parametric.x_name
            if local.parametric is not None
            else "t"
        )
        model_axis = (
            np.asarray(series.t, dtype=float)
            if x_name == "t"
            else np.asarray(series.species[x_name], dtype=float)
        )
        objective_rows = {
            item.model_species: select_fit_rows(
                model_y=np.asarray(series.species[item.model_species], dtype=float),
                model_x=(
                    None
                    if x_name == "t"
                    else np.asarray(series.species[x_name], dtype=float)
                ),
                x_name=x_name,
                x_mode=(
                    local.parametric.resolved_mode
                    if local.parametric is not None
                    else "monotone"
                ),
                sim_time=np.asarray(series.t, dtype=float),
                observation_time=np.asarray(item.t, dtype=float),
                observation_x=np.asarray(
                    item.t
                    if x_name == "t"
                    else observed_axes[item.model_species],
                    dtype=float,
                ),
                observation_id=item.raw_observation_id,
                raw_column=item.raw_column,
                source_rows=item.source_rows,
                segment_ids=np.asarray(evaluation.segment_ids, dtype=np.int64),
                segment_keys=evaluation.segment_keys,
                discontinuity_ids=np.asarray(
                    evaluation.discontinuity_ids,
                    dtype=np.int64,
                ),
                dataset_label=dataset_id,
                y_name=item.model_species,
            )
            for item in selected
        }
        curves.append(
            FitCurveDataset(
                dataset_id=dataset_id,
                display_name=mapped.display_name,
                model_x=tuple(
                    (
                        item.model_species,
                        tuple(float(value) for value in model_axis),
                    )
                    for item in selected
                ),
                model_y=tuple(
                    (
                        item.model_species,
                        tuple(
                            float(value)
                            for value in np.asarray(
                                series.species[item.model_species],
                                dtype=float,
                            )
                        ),
                    )
                    for item in selected
                ),
                observed_x=observed_x,
                observed_y=observed_y,
                model_segment_ids=tuple(
                    (item.model_species, segment_ids)
                    for item in selected
                ),
                model_segment_keys=tuple(
                    (item.model_species, segment_keys)
                    for item in selected
                ),
                model_discontinuity_ids=tuple(
                    (item.model_species, discontinuity_ids)
                    for item in selected
                ),
                observed_source_rows=tuple(
                    (item.model_species, item.source_rows)
                    for item in selected
                ),
                observed_raw_ids=tuple(
                    (
                        item.model_species,
                        item.raw_observation_id,
                    )
                    for item in selected
                ),
                observed_raw_columns=tuple(
                    (item.model_species, item.raw_column)
                    for item in selected
                ),
                objective_model_x=tuple(
                    (item.model_species, tuple(float(value) for value in objective_rows[item.model_species][0]))
                    for item in selected
                ),
                objective_model_y=tuple(
                    (item.model_species, tuple(float(value) for value in objective_rows[item.model_species][1]))
                    for item in selected
                ),
                objective_model_segment_ids=tuple(
                    (item.model_species, tuple(int(value) for value in objective_rows[item.model_species][2]))
                    for item in selected
                ),
                objective_model_segment_keys=tuple(
                    (item.model_species, tuple(objective_rows[item.model_species][3]))
                    for item in selected
                ),
                objective_model_discontinuity_ids=tuple(
                    (item.model_species, tuple(int(value) for value in objective_rows[item.model_species][4]))
                    for item in selected
                ),
                objective_selection_provenance=tuple(
                    (item.model_species, tuple(objective_rows[item.model_species][5]))
                    for item in selected
                ),
                objective_observation_ids=tuple(
                    (item.model_species, tuple(objective_rows[item.model_species][6]))
                    for item in selected
                ),
                objective_raw_columns=tuple(
                    (item.model_species, tuple(objective_rows[item.model_species][7]))
                    for item in selected
                ),
                objective_source_rows=tuple(
                    (item.model_species, tuple(objective_rows[item.model_species][8]))
                    for item in selected
                ),
                x_name=x_name,
            )
        )
    return tuple(curves)


class GlobalFitWorker(QtCore.QThread):
    """Run one immutable request and retain every evaluator until terminal."""

    progress = QtCore.Signal(int, str)
    auto_modes_resolved = QtCore.Signal(object)
    best_updated = QtCore.Signal(object)
    curves_updated = QtCore.Signal(object)
    detailed_balance_correction_required = QtCore.Signal(object)

    def __init__(
        self,
        request: FitExecutionRequest,
        *,
        setup_snapshot: FitSetupSnapshot | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(request, FitExecutionRequest):
            raise TypeError("request must be FitExecutionRequest.")
        self._request = request
        self._setup_snapshot = (
            setup_snapshot or FitSetupSnapshot.from_request(request)
        )
        self._cancelled = Event()
        self._pause_condition = Condition()
        self._paused = False
        self._last_best_emit = 0.0
        self._last_curve_emit = 0.0
        self._best_cost = math.inf
        self._best_iteration = 0
        self._best_params: dict[str, float] = {}
        self._resolved_auto_modes: tuple[tuple[str, str], ...] = ()
        self._detailed_balance_condition = Condition()
        self._detailed_balance_response: tuple[
            FitExecutionRequest,
            FitSetupSnapshot,
        ] | None = None
        self._detailed_balance_answered = False
        self._pending_detailed_balance_request_id: str | None = None
        self._detailed_balance_rejection_message: str | None = None
        self._terminal_result: FitWorkerTerminal | None = None

    def pause(self) -> None:
        with self._pause_condition:
            self._paused = True

    @property
    def resolved_auto_modes(self) -> tuple[tuple[str, str], ...]:
        return self._resolved_auto_modes

    @property
    def publication_fenced(self) -> bool:
        return self._cancelled.is_set() or self.isInterruptionRequested()

    @property
    def terminal_result(self) -> FitWorkerTerminal | None:
        """Return the retained terminal after the worker's ``run`` exits."""

        return self._terminal_result

    def resume(self) -> None:
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()
        with self._detailed_balance_condition:
            self._detailed_balance_answered = True
            self._detailed_balance_response = None
            self._detailed_balance_condition.notify_all()
        self.requestInterruption()

    def submit_detailed_balance_correction(
        self,
        request_id: str,
        corrected_request: FitExecutionRequest | None,
        corrected_snapshot: FitSetupSnapshot | None = None,
        *,
        rejection_message: str | None = None,
    ) -> None:
        with self._detailed_balance_condition:
            if self._detailed_balance_answered:
                return
            if str(request_id) != self._pending_detailed_balance_request_id:
                raise ValueError(
                    "Detailed-balance correction request identity does not match."
                )
            if corrected_request is None:
                self._detailed_balance_response = None
                self._detailed_balance_rejection_message = (
                    str(rejection_message or "").strip()
                    or "Detailed-balance correction was not accepted; Fit aborted."
                )
            else:
                if corrected_snapshot is None:
                    raise TypeError("A corrected Fit setup snapshot is required.")
                if rejection_message is not None:
                    raise ValueError(
                        "An accepted detailed-balance correction cannot carry "
                        "a rejection message."
                    )
                self._detailed_balance_response = (
                    corrected_request,
                    corrected_snapshot,
                )
                self._detailed_balance_rejection_message = None
            self._detailed_balance_answered = True
            self._detailed_balance_condition.notify_all()

    def _checkpoint(self) -> bool:
        with self._pause_condition:
            while self._paused and not self._cancelled.is_set():
                self._pause_condition.wait()
        return self._cancelled.is_set()

    def run(self) -> None:
        terminal: FitWorkerTerminal
        try:
            try:
                candidate = self._execute(
                    allow_detailed_balance_correction=True
                )
            except _DetailedBalanceCorrectionRequired as correction:
                with self._detailed_balance_condition:
                    self._detailed_balance_answered = False
                    self._detailed_balance_response = None
                    self._detailed_balance_rejection_message = None
                    self._pending_detailed_balance_request_id = (
                        correction.request.request_id
                    )
                if self._checkpoint():
                    raise FittingCancelled()
                self.detailed_balance_correction_required.emit(
                    correction.request
                )
                if self._checkpoint():
                    raise FittingCancelled()
                with self._detailed_balance_condition:
                    while (
                        not self._detailed_balance_answered
                        and not self._cancelled.is_set()
                    ):
                        self._detailed_balance_condition.wait()
                    response = self._detailed_balance_response
                    rejection_message = (
                        self._detailed_balance_rejection_message
                    )
                    self._pending_detailed_balance_request_id = None
                if self._cancelled.is_set():
                    raise FittingCancelled()
                if response is None:
                    raise RuntimeError(
                        rejection_message
                        or "Detailed-balance correction was not accepted; "
                        "Fit aborted."
                    )
                self._request, self._setup_snapshot = response
                self._best_cost = math.inf
                self._best_iteration = 0
                self._best_params = {}
                self._resolved_auto_modes = ()
                candidate = self._execute(
                    allow_detailed_balance_correction=False
                )
        except FittingCancelled:
            terminal = FitWorkerTerminal(
                candidate=None,
                message="Fit cancelled.",
                cancelled=True,
            )
        except BaseException as exc:  # noqa: BLE001
            terminal = FitWorkerTerminal(
                candidate=None,
                message=str(exc) or type(exc).__name__,
            )
        else:
            if self._checkpoint():
                terminal = FitWorkerTerminal(
                    candidate=None,
                    message="Fit cancelled.",
                    cancelled=True,
                )
            else:
                terminal = FitWorkerTerminal(
                    candidate=candidate,
                    message="",
                )
        self._terminal_result = terminal

    def _execute(
        self,
        *,
        allow_detailed_balance_correction: bool,
    ) -> FitCandidate:
        def fitting_checkpoint() -> bool:
            return self._checkpoint()

        fitting_checkpoint._kindred_nonblocking_cancelled = (  # type: ignore[attr-defined]
            lambda: self._cancelled.is_set() or self.isInterruptionRequested()
        )
        request = self._request
        included = [item for item in request.datasets if item.included]
        mapped_by_id = {item.dataset_id: item for item in request.origin.mapped_datasets}
        if not included:
            raise ValueError("Select at least one mapped dataset for fitting.")
        included_ids = tuple(item.dataset_id for item in included)

        shared_variables: list[FitSharedVariableParamSpec] = []
        coupled_variables: list[FitCoupledVariableParamSpec] = []
        variable_by_dataset: dict[str, list[FitDatasetVariableParamSpec]] = {
            dataset_id: [] for dataset_id in included_ids
        }
        fixed_by_dataset: dict[str, dict[str, float]] = {
            dataset_id: {} for dataset_id in included_ids
        }

        for parameter in request.parameters:
            dataset_values = dict(parameter.dataset_values)
            if parameter.fit and parameter.scope == "shared":
                shared_variables.append(
                    FitSharedVariableParamSpec(
                        name=parameter.name,
                        initial=parameter.initial,
                        minimum=parameter.minimum,
                        maximum=parameter.maximum,
                        log10=parameter.log10,
                    )
                )
            elif parameter.fit:
                for dataset_id in included_ids:
                    variable_by_dataset[dataset_id].append(
                        FitDatasetVariableParamSpec(
                            name=parameter.name,
                            initial=dataset_values[dataset_id],
                            minimum=parameter.minimum,
                            maximum=parameter.maximum,
                            log10=parameter.log10,
                        )
                    )
            elif parameter.scope == "per_dataset":
                for dataset_id in included_ids:
                    fixed_by_dataset[dataset_id][parameter.name] = float(
                        dataset_values[dataset_id]
                    )
            else:
                for dataset_id in included_ids:
                    fixed_by_dataset[dataset_id][parameter.name] = float(
                        parameter.initial
                    )

        datasets_for_set: dict[str, tuple[str, ...]] = {}
        for mapped in request.origin.mapped_datasets:
            if mapped.dataset_id in included_ids:
                datasets_for_set.setdefault(mapped.set_id, ())
                datasets_for_set[mapped.set_id] = (
                    *datasets_for_set[mapped.set_id],
                    mapped.dataset_id,
                )
        initials_by_species: dict[str, list[object]] = {}
        for initial in request.initials:
            initials_by_species.setdefault(initial.species, []).append(initial)
        for species, rows in initials_by_species.items():
            active_rows = [
                item
                for item in rows
                if datasets_for_set.get(item.set_id)
            ]
            global_rows = [
                item for item in active_rows if item.scope == "global"
            ]
            if global_rows:
                if len(global_rows) != len(active_rows):
                    raise ValueError(
                        f"Initial {species!r} mixes Global and Local scope."
                    )
                signatures = {
                    (
                        item.initial,
                        item.minimum,
                        item.maximum,
                        item.fit,
                        item.log10,
                    )
                    for item in global_rows
                }
                if len(signatures) != 1:
                    raise ValueError(
                        f"Global initial {species!r} has inconsistent "
                        "coupled setup rows."
                    )
                (
                    initial_value,
                    minimum,
                    maximum,
                    fitted,
                    log10,
                ) = signatures.pop()
                if fitted:
                    coupled_variables.append(
                        FitCoupledVariableParamSpec(
                            variable_id=f"initial:global:{species}",
                            parameter_name=f"{INITIAL_PREFIX}{species}",
                            dataset_ids=included_ids,
                            initial=initial_value,
                            minimum=minimum,
                            maximum=maximum,
                            log10=log10,
                        )
                    )
                else:
                    for dataset_id in included_ids:
                        fixed_by_dataset[dataset_id][f"{INITIAL_PREFIX}{species}"] = (
                            initial_value
                        )
                continue
            for initial in active_rows:
                target_datasets = datasets_for_set.get(initial.set_id, ())
                if not target_datasets:
                    continue
                value = initial.initial
                if initial.fit:
                    coupled_variables.append(
                        FitCoupledVariableParamSpec(
                            variable_id=f"initial:{initial.set_id}:{species}",
                            parameter_name=f"{INITIAL_PREFIX}{species}",
                            dataset_ids=target_datasets,
                            initial=value,
                            minimum=initial.minimum,
                            maximum=initial.maximum,
                            log10=initial.log10,
                        )
                    )
                else:
                    for dataset_id in target_datasets:
                        fixed_by_dataset[dataset_id][f"{INITIAL_PREFIX}{species}"] = value

        overrides = tuple(
            FitDatasetParameterOverrides(
                dataset_id=dataset_id,
                fixed_params=tuple(sorted(fixed_by_dataset[dataset_id].items())),
                variable_params=tuple(variable_by_dataset[dataset_id]),
            )
            for dataset_id in included_ids
        )
        candidate_names = sorted(
            parameter.name
            for parameter in request.parameters
            if not parameter.name.startswith(INITIAL_PREFIX)
        )
        evaluators: dict[str, SerialFittingEvaluator] = {}
        prepared_ids: dict[str, dict[str, object]] = {}
        for local in included:
            if self._checkpoint():
                raise FittingCancelled()
            science = request.origin.scientific_settings
            execution = local.execution
            try:
                context = prepare_fitting_execution_context(
                    prepared_dsl=request.prepared_dsl,
                    param_names=candidate_names,
                    t_start=float(execution.t_start),
                    t_end=float(execution.t_end),
                    num_points=int(execution.integration_points),
                    objective_sample_times=tuple(
                        sorted(
                            {
                                float(time_value)
                                for observation in selected_observations(
                                    mapped_by_id[local.dataset_id],
                                    local,
                                )
                                for time_value in observation.t
                            }
                        )
                    ),
                    solver=execution.solver,
                    rtol=float(science.rtol),
                    atol=float(science.atol),
                    use_sparse_jacobian=bool(science.sparse_jacobian_policy),
                    wegscheider_cyclicity_enabled=bool(
                        science.wegscheider_policy
                    ),
                    initial_prefix=INITIAL_PREFIX,
                )
            except BaseException as exc:
                unresolved = _unresolved_detailed_balance_error(exc)
                if (
                    allow_detailed_balance_correction
                    and unresolved is not None
                ):
                    issue = str(unresolved)
                    identity = FitRunIdentity.build(
                        {
                            "kind": "detailed_balance_correction",
                            "schema_version": 1,
                            "origin_capture_id": request.origin.capture_id,
                            "tab_id": request.origin.tab_id,
                            "source": (
                                request.origin.mechanism_source.reactions_text
                            ),
                            "temperature_K": float(science.temperature),
                            "wegscheider_cyclicity_enabled": bool(
                                science.wegscheider_policy
                            ),
                            "issue": issue,
                        }
                    )
                    raise _DetailedBalanceCorrectionRequired(
                        DetailedBalanceCorrectionRequest(
                            request_id=identity.short_hash,
                            origin="Fit",
                            tab_id=request.origin.tab_id,
                            source=(
                                request.origin.mechanism_source.reactions_text
                            ),
                            issue=issue,
                            temperature_K=float(science.temperature),
                            report=WegscheiderCyclicityReport(tuple(unresolved.cycles)),
                            source_energy_unit=str(
                                getattr(request.prepared_dsl.analysis, "structure", {})
                                .get(
                                    "source_energy_unit",
                                    getattr(
                                        request.prepared_dsl.analysis,
                                        "structure",
                                        {},
                                    ).get("energy_unit", "kJ/mol"),
                                )
                            ),
                        )
                    ) from exc
                raise
            evaluator = SerialFittingEvaluator(
                context,
                fixed_param_origins={
                    name: FITTING_PARAM_ORIGIN_CONFIGURED_DATASET
                    for name in fixed_by_dataset[local.dataset_id]
                },
            )
            evaluator.set_cancellation_check(fitting_checkpoint)
            evaluators[local.dataset_id] = evaluator
            metadata = evaluator.prepared_metadata
            prepared_ids[local.dataset_id] = {
                "prepared_simulation": metadata.to_serializable_dict(),
                "declarative_analysis_identity": (
                    request.prepared_dsl.analysis.identity
                ),
                "execution": {
                    "solver": execution.solver,
                    "t_start": float(execution.t_start),
                    "t_end": float(execution.t_end),
                    "integration_points": int(execution.integration_points),
                },
            }

        effective_local: dict[str, FitDatasetLocalSettings] = {}
        resolved_mode_items: list[tuple[str, str]] = []
        starting_evaluations = {}
        for local in included:
            chosen = local
            dataset_id = local.dataset_id
            mapped = mapped_by_id[dataset_id]
            starting_values = {
                parameter.name: (
                    dict(parameter.dataset_values)[dataset_id]
                    if parameter.scope == "per_dataset"
                    else parameter.initial
                )
                for parameter in request.parameters
            }
            starting_values.update(fixed_by_dataset[dataset_id])
            for initial in request.initials:
                if initial.set_id != mapped.set_id:
                    continue
                starting_values[
                    f"{INITIAL_PREFIX}{initial.species}"
                ] = initial.initial
            evaluation = evaluators[
                dataset_id
            ].evaluate_series_with_parameter_origins(
                starting_values,
                {
                    name: FITTING_PARAM_ORIGIN_CONFIGURED_DATASET
                    for name in starting_values
                },
            )
            starting_evaluations[dataset_id] = evaluation
            prepared_ids[dataset_id]["validated_executable_schedule"] = (
                evaluation.intervention_execution.provenance_projection()
            )
            if (
                local.parametric is not None
                and local.parametric.requested_mode == "auto"
            ):
                provisional = _dataset_spec(
                    mapped,
                    replace(
                        local,
                        parametric=replace(
                            local.parametric,
                            resolved_mode="monotone",
                        ),
                    ),
                )
                series = evaluation.series
                try:
                    for target_name, observation in provisional.observations.items():
                        align_y_on_x_obs(
                            t_obs=np.asarray(
                                observation.t,
                                dtype=float,
                            ),
                            x_obs=np.asarray(
                                provisional.x_obs_by_species[target_name],
                                dtype=float,
                            ),
                            t_sim=series.t,
                            x_model=series.species[provisional.x_name],
                            y_model=series.species[target_name],
                            segment_ids=evaluation.segment_ids,
                            segment_keys=evaluation.segment_keys,
                            discontinuity_ids=evaluation.discontinuity_ids,
                            dataset_label=dataset_id,
                            x_name=provisional.x_name,
                            y_name=target_name,
                        )
                except DeterministicParametricResolutionError:
                    resolved_mode = "time_guided"
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
            _dataset_spec(mapped_by_id[item.dataset_id], effective_local[item.dataset_id])
            for item in included
        ]
        resolved_modes = tuple(
            sorted(resolved_mode_items)
        )
        self._resolved_auto_modes = resolved_modes
        if self._checkpoint():
            raise FittingCancelled()
        self.auto_modes_resolved.emit(resolved_modes)
        if self._checkpoint():
            raise FittingCancelled()
        initial_curves = _curves_from_evaluations(
            request,
            starting_evaluations,
            effective_local,
        )
        if self._checkpoint():
            raise FittingCancelled()
        self.curves_updated.emit(
            initial_curves
        )
        if self._checkpoint():
            raise FittingCancelled()
        self._last_curve_emit = time.monotonic()
        run_identity_payload = {
            "kind": "global_fit_run",
            "schema_version": 1,
            "origin_tab_id": request.origin.tab_id,
            "origin_capture_id": request.origin.capture_id,
            "mechanism_text": request.mechanism_source.reactions_text,
            "dataset_weight_mode": request.dataset_weight_mode,
            "local_observables": [
                    {
                        "name": item.name,
                        "expression": item.expression,
                        "scalars": [
                            {
                                "name": scalar.name,
                                "value": float(scalar.value),
                                "minimum": _stamp_bound(scalar.minimum),
                                "maximum": _stamp_bound(scalar.maximum),
                                "scope": scalar.scope,
                            }
                            for scalar in item.scalar_parameters
                        ],
                    }
                    for item in request.local_observables
                ],
            "datasets": [
                    {
                        "dataset_id": item.dataset_id,
                        "set_id": mapped_by_id[item.dataset_id].set_id,
                        "observations": {
                            name: {
                                "raw_observation_id": (
                                    series.raw_observation_id
                                ),
                                "t": [
                                    float(value)
                                    for value in np.asarray(
                                        series.t,
                                        dtype=float,
                                    ).reshape(-1)
                                ],
                                "y": [
                                    float(value)
                                    for value in np.asarray(
                                        series.y,
                                        dtype=float,
                                    ).reshape(-1)
                                ],
                                "source_rows": list(
                                    series.source_rows
                                ),
                                "observed_x": {
                                    "identity": (
                                        (
                                            next(
                                                observation.raw_observation_id
                                                for observation in mapped_by_id[
                                                    item.dataset_id
                                                ].observations
                                                if observation.model_species
                                                == item.x_name
                                            )
                                        )
                                        if item.x_name != "t"
                                        else series.raw_observation_id
                                    ),
                                    "kind": (
                                        "raw_observation"
                                        if item.x_name != "t"
                                        else "observation_time"
                                    ),
                                    "name": item.x_name,
                                    "values": [
                                        float(value)
                                        for value in np.asarray(
                                            (
                                                item.x_obs_by_species[name]
                                                if item.x_name != "t"
                                                else series.t
                                            ),
                                            dtype=float,
                                        ).reshape(-1)
                                    ],
                                    "paired_y_raw_observation_id": (
                                        series.raw_observation_id
                                    ),
                                    "paired_source_rows": list(
                                        series.source_rows
                                    ),
                                },
                            }
                            for name, series in item.observations.items()
                        },
                        "mappings": [
                            {
                                "raw_observation_id": raw_id,
                                "raw_column": raw_column,
                                "model_target": model_target,
                            }
                            for raw_id, model_target in next(
                                local.observation_targets
                                for local in included
                                if local.dataset_id == item.dataset_id
                            )
                            for raw_column in (
                                next(
                                    observation.raw_column
                                    for observation in mapped_by_id[
                                        item.dataset_id
                                    ].observations
                                    if observation.raw_observation_id
                                    == raw_id
                                ),
                            )
                        ],
                        "targets": item.species_list,
                        "target_weights": dict(item.target_weights),
                        "sampling": {
                            "t_min": float(
                                effective_local[item.dataset_id].sample_t_min
                            ),
                            "t_max": float(
                                effective_local[item.dataset_id].sample_t_max
                            ),
                            "points": (
                                effective_local[
                                    item.dataset_id
                                ].sample_points
                            ),
                        },
                        **(
                            {
                                "dataset_weight": float(
                                    effective_local[item.dataset_id].weight
                                )
                            }
                            if request.dataset_weight_mode == "user"
                            else {}
                        ),
                        "execution": {
                            "solver": effective_local[
                                item.dataset_id
                            ].execution.solver,
                            "t_start": float(
                                effective_local[
                                    item.dataset_id
                                ].execution.t_start
                            ),
                            "t_end": float(
                                effective_local[
                                    item.dataset_id
                                ].execution.t_end
                            ),
                            "integration_points": int(
                                effective_local[
                                    item.dataset_id
                                ].execution.integration_points
                            ),
                        },
                        "x_name": item.x_name,
                        "x_mapping_requested": (
                            effective_local[
                                item.dataset_id
                            ].parametric.requested_mode
                            if effective_local[
                                item.dataset_id
                            ].parametric is not None
                            else "time"
                        ),
                        "x_mapping_resolved": (
                            effective_local[
                                item.dataset_id
                            ].parametric.resolved_mode
                            if effective_local[
                                item.dataset_id
                            ].parametric is not None
                            else "time"
                        ),
                    }
                    for item in datasets
                ],
            "parameters": [
                    {
                        "name": item.name,
                        "display_name": item.display_name,
                        "fit": item.fit,
                        "initial": item.initial,
                        "minimum": _stamp_bound(item.minimum),
                        "maximum": _stamp_bound(item.maximum),
                        "log10": item.log10,
                        "scope": item.scope,
                        "dataset_values": dict(item.dataset_values),
                        "family": item.family,
                        "role": item.role,
                        "unit": item.unit,
                    }
                    for item in request.parameters
                ],
            "initials": [
                    {
                        "set_id": item.set_id,
                        "species": item.species,
                        "fit": item.fit,
                        "initial": item.initial,
                        "minimum": _stamp_bound(item.minimum),
                        "maximum": _stamp_bound(item.maximum),
                        "log10": item.log10,
                        "scope": item.scope,
                    }
                    for item in request.initials
                ],
            "settings": {
                    "method": request.defaults.method,
                    "max_nfev": request.defaults.max_nfev,
                    "ftol": request.defaults.ftol,
                    "xtol": request.defaults.xtol,
                    "use_seed": request.defaults.use_seed,
                    "seed": request.defaults.seed if request.defaults.use_seed else None,
                },
            "scientific_settings": {
                    "temperature": float(
                        request.origin.scientific_settings.temperature
                    ),
                    "rtol": float(request.origin.scientific_settings.rtol),
                    "atol": float(request.origin.scientific_settings.atol),
                    "sparse_jacobian_policy": bool(
                        request.origin.scientific_settings.sparse_jacobian_policy
                    ),
                    "wegscheider_policy": bool(
                        request.origin.scientific_settings.wegscheider_policy
                    ),
                },
            "prepared_identities": prepared_ids,
            "resolved_auto_modes": dict(resolved_modes),
        }

        def live_evaluations(
            params: Mapping[str, float],
        ) -> dict[str, object]:
            evaluations: dict[str, object] = {}
            for local in included:
                dataset_id = local.dataset_id
                values = dict(fixed_by_dataset[dataset_id])
                for variable in shared_variables:
                    values[variable.name] = float(
                        params.get(variable.name, variable.initial)
                    )
                for variable in variable_by_dataset[dataset_id]:
                    key = f"{dataset_id}::{variable.name}"
                    values[variable.name] = float(
                        params.get(key, variable.initial)
                    )
                for variable in coupled_variables:
                    if dataset_id not in variable.dataset_ids:
                        continue
                    values[variable.parameter_name] = float(
                        params.get(variable.variable_id, variable.initial)
                    )
                evaluations[dataset_id] = evaluators[
                    dataset_id
                ].evaluate_series_with_parameter_origins(
                    values,
                    {
                        name: FITTING_PARAM_ORIGIN_CONFIGURED_DATASET
                        for name in values
                    },
                )
            return evaluations

        def progress(iteration: int, cost: float, params: dict[str, float]) -> None:
            if self._checkpoint():
                raise FittingCancelled()
            self.progress.emit(
                min(95, max(1, int(100 * iteration / max(1, request.defaults.max_nfev)))),
                f"Iteration {iteration}: objective SSQ={cost:.4g}",
            )
            if self._checkpoint():
                raise FittingCancelled()
            if math.isfinite(cost) and cost < self._best_cost:
                self._best_cost = float(cost)
                self._best_iteration = int(iteration)
                self._best_params = dict(params)
                now = time.monotonic()
                if now - self._last_best_emit >= _BEST_INTERVAL_SECONDS:
                    if self._checkpoint():
                        raise FittingCancelled()
                    self.best_updated.emit(self._live_update())
                    if self._checkpoint():
                        raise FittingCancelled()
                    self._last_best_emit = now
                if now - self._last_curve_emit >= _CURVE_INTERVAL_SECONDS:
                    try:
                        live_curves = _curves_from_evaluations(
                            request,
                            live_evaluations(self._best_params),
                            effective_local,
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

        method = str(request.defaults.method).strip().lower()
        result = fit_global(
            fit_evaluators=evaluators,
            datasets=datasets,
            shared_variables=shared_variables,
            dataset_overrides=overrides,
            coupled_variables=coupled_variables,
            weights=(
                {
                    item.dataset_id: item.weight
                    for item in included
                }
                if request.dataset_weight_mode == "user"
                else None
            ),
            method="de" if method == "differential_evolution" else method,
            max_nfev=int(request.defaults.max_nfev),
            ftol=float(request.defaults.ftol),
            xtol=float(request.defaults.xtol),
            seed=int(request.defaults.seed) if request.defaults.use_seed else None,
            progress_callback=progress,
            cancellation_check=fitting_checkpoint,
        )
        if self._checkpoint():
            raise FittingCancelled()
        if result.completion.status == "fail":
            message = result.message or "Fit did not produce a candidate."
            raise RuntimeError(message)

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
        solver_provenance_by_dataset = {
            str(dataset_id or "").strip(): provenance
            for dataset_id, provenance in (
                result.solver_provenance_by_dataset.items()
            )
        }
        if (
            any(not dataset_id for dataset_id in solver_provenance_by_dataset)
            or set(solver_provenance_by_dataset) != set(included_ids)
        ):
            raise ValueError(
                "Fit terminal solver provenance does not exactly match the "
                "included datasets."
            )

        scalar_values = tuple(
            (
                dataset_id,
                _finite_items(
                    {
                        name: value
                        for name, value in values.items()
                        if not name.startswith(INITIAL_PREFIX)
                    },
                    field=f"Fit scalar values {dataset_id}",
                ),
            )
            for dataset_id, values in sorted(result.dataset_params.items())
        )
        execution_by_set_map: dict[
            str,
            SimulationExecutionSettings,
        ] = {}
        execution_conflict = False
        for item in included:
            set_id = mapped_by_id[item.dataset_id].set_id
            previous = execution_by_set_map.get(set_id)
            if previous is not None and previous != item.execution:
                execution_conflict = True
            execution_by_set_map[set_id] = item.execution
        execution_by_set = (
            ()
            if execution_conflict
            else tuple(execution_by_set_map.items())
        )
        curves = _curves_from_result(request, result)
        initial_values_by_set = _initial_values_from_result(
            request,
            result,
        )
        terminal_snapshot = _terminal_setup_snapshot(
            self._setup_snapshot,
            request=request,
            result=result,
            effective_datasets=effective_local,
            initial_values_by_set=initial_values_by_set,
        )
        if self._checkpoint():
            raise FittingCancelled()
        for dataset_id, evaluator in evaluators.items():
            prepared_ids[dataset_id]["prepared_simulation"] = (
                evaluator.prepared_metadata.to_serializable_dict()
            )
        run_identity_payload["terminal_replay"] = {
            "status": result.completion.status,
            "optimizer_converged": bool(
                result.completion.optimizer_converged
            ),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "weighted_residual_sse": float(result.weighted_residual_sse),
            "weighted_residual_mse": float(result.weighted_residual_mse),
            "effective_dataset_weights": {
                dataset_id: effective_dataset_weights[dataset_id]
                for dataset_id in sorted(effective_dataset_weights)
            },
            "solver_provenance_by_dataset": {
                dataset_id: solver_provenance_by_dataset[dataset_id]
                for dataset_id in sorted(solver_provenance_by_dataset)
            },
            "shared_parameters": {
                str(name): float(value)
                for name, value in sorted(result.shared_params.items())
            },
            "dataset_parameters": {
                str(dataset_id): {
                    str(name): float(value)
                    for name, value in sorted(values.items())
                }
                for dataset_id, values in sorted(
                    result.dataset_params.items()
                )
            },
            "objective_rows": [
                {
                    "dataset_id": dataset_id,
                    "series": [
                        {
                            "model_target": series_name,
                            "model_x": [
                                float(value)
                                for value in np.asarray(
                                    result.model_x_by_species[dataset_id][
                                        series_name
                                    ],
                                    dtype=float,
                                ).reshape(-1)
                            ],
                            "model_y": [
                                float(value)
                                for value in np.asarray(
                                    result.model_series[dataset_id][series_name],
                                    dtype=float,
                                ).reshape(-1)
                            ],
                            "segment_ids": [
                                int(value)
                                for value in np.asarray(
                                    result.model_segment_ids[dataset_id][
                                        series_name
                                    ],
                                    dtype=np.int64,
                                ).reshape(-1)
                            ],
                            "segment_keys": list(
                                result.model_segment_keys[dataset_id][
                                    series_name
                                ]
                            ),
                            "discontinuity_ids": [
                                int(value)
                                for value in np.asarray(
                                    result.model_discontinuity_ids[dataset_id][
                                        series_name
                                    ],
                                    dtype=np.int64,
                                ).reshape(-1)
                            ],
                            "selection_provenance": list(
                                result.model_selection_provenance[dataset_id][
                                    series_name
                                ]
                            ),
                            "raw_observation_ids": list(
                                result.model_observation_ids[dataset_id][
                                    series_name
                                ]
                            ),
                            "raw_columns": list(
                                result.model_raw_columns[dataset_id][series_name]
                            ),
                            "source_rows": [
                                int(value)
                                for value in result.model_source_rows[dataset_id][
                                    series_name
                                ]
                            ],
                        }
                        for series_name in sorted(
                            result.model_series[dataset_id]
                        )
                    ],
                }
                for dataset_id in sorted(result.model_series)
            ],
        }
        run_identity = FitRunIdentity.build(run_identity_payload)
        if self._checkpoint():
            raise FittingCancelled()
        self.progress.emit(100, f"Fit complete · {run_identity.short_hash}")
        if self._checkpoint():
            raise FittingCancelled()
        return FitCandidate(
            origin_tab_id=request.origin.tab_id,
            origin_capture_id=request.origin.capture_id,
            mechanism_source=request.mechanism_source,
            prepared_dsl=request.prepared_dsl,
            parameters=_finite_items(result.shared_params, field="Fit parameters"),
            scalar_values_by_dataset=scalar_values,
            initial_values_by_set=initial_values_by_set,
            execution_by_set=execution_by_set,
            local_observables=terminal_snapshot.local_observables,
            result=result,
            curves=curves,
            run_identity=run_identity,
            setup_snapshot=terminal_snapshot,
            resolved_auto_modes=resolved_modes,
        )

    def _live_update(self) -> FitLiveUpdate:
        included_dataset_ids = {
            item.dataset_id for item in self._request.datasets if item.included
        }
        included_set_ids = {
            mapped.set_id
            for mapped in self._request.origin.mapped_datasets
            if mapped.dataset_id in included_dataset_ids
        }
        initials_by_set: dict[str, dict[str, float]] = {}
        for initial in self._request.initials:
            if not initial.fit or initial.set_id not in included_set_ids:
                continue
            variable_id = (
                f"initial:global:{initial.species}"
                if initial.scope == "global"
                else f"initial:{initial.set_id}:{initial.species}"
            )
            value = self._best_params.get(variable_id)
            if value is not None and math.isfinite(float(value)):
                initials_by_set.setdefault(initial.set_id, {})[
                    initial.species
                ] = float(value)
        return FitLiveUpdate(
            iteration=self._best_iteration,
            cost=self._best_cost,
            parameters=_finite_items(
                {
                    name: value
                    for name, value in self._best_params.items()
                    if "::" not in name and not name.startswith("initial:")
                },
                field="live Fit parameters",
            ),
            scalar_values_by_dataset=tuple(
                (
                    dataset_id,
                    _finite_items(
                        {
                            parameter.name: self._best_params.get(
                                f"{dataset_id}::{parameter.name}",
                                dict(parameter.dataset_values)[dataset_id],
                            )
                            for parameter in self._request.parameters
                            if (
                                parameter.fit
                                and parameter.scope == "per_dataset"
                                and dataset_id
                                in dict(parameter.dataset_values)
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
