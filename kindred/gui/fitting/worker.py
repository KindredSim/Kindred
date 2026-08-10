"""One pausable/cancellable Fit QThread over immutable typed captures."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
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
from kindred.core.analysis.global_fitting import fit_global
from kindred.core.analysis.global_fit_execution import (
    FitProgress,
    GlobalFitResult,
    select_fit_rows,
)
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
from kindred.core.result_repository import SimulationResultValue
from kindred.core.simulator.solver_types import SimulationCancellationControl
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
    FitOriginSnapshot,
    FitSelectedObservation,
    FitSetupSnapshot,
    _replace_fit_parameter,
    fit_local_observable_definitions,
    selected_observations,
)
from kindred.gui.workflow_contracts import (
    DetailedBalanceCorrectionRequest,
)

__all__ = [
    "FitCandidate",
    "FitCurveDataset",
    "FitCurveYSemantic",
    "FitLiveUpdate",
    "FitWorkerTerminal",
    "GlobalFitWorker",
    "project_fit_candidate_curves",
]

_BEST_INTERVAL_SECONDS = 0.25
_CURVE_INTERVAL_SECONDS = 2.0


class _DetailedBalanceCorrectionRequired(RuntimeError):
    def __init__(self, request: DetailedBalanceCorrectionRequest) -> None:
        super().__init__(request.issue)
        self.request = request


class FitCurveYSemantic(str, Enum):
    """Typed physical meaning of one projected Fit Y series."""

    CONCENTRATION = "concentration"
    VALUE = "value"


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
    y_semantics: tuple[tuple[str, FitCurveYSemantic], ...]
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

    origin: FitOriginSnapshot
    parameters: tuple[tuple[str, float], ...]
    scalar_values_by_dataset: tuple[
        tuple[str, tuple[tuple[str, float], ...]], ...
    ]
    initial_values_by_set: tuple[
        tuple[str, tuple[tuple[str, float], ...]], ...
    ]
    local_observables: tuple[FitLocalObservable, ...]
    result: GlobalFitResult
    run_identity: FitRunIdentity
    setup_snapshot: FitSetupSnapshot
    resolved_auto_modes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.origin, FitOriginSnapshot):
            raise TypeError("Fit candidate origin must be retained.")
        if not isinstance(self.result, GlobalFitResult):
            raise TypeError("Fit candidate result must be canonical.")
        if (
            self.setup_snapshot.origin_tab_id != self.origin.tab_id
            or self.setup_snapshot.origin_capture_id != self.origin.capture_id
        ):
            raise ValueError(
                "Fit candidate origin and terminal setup identities must match."
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
    raw_id_by_target = {
        model_target: raw_id
        for raw_id, model_target in local.observation_targets
    }
    target_by_raw_id = {
        raw_id: model_target
        for raw_id, model_target in local.observation_targets
    }
    x_raw_id = raw_id_by_target.get(parametric.x_name)
    if x_raw_id is None:
        raise ValueError(
            f"Dataset {mapped.display_name!r} requires exactly one observed "
            f"source for X={parametric.x_name!r}."
        )
    active_y_raw_ids = {
        item.raw_observation_id for item in observations
    }
    if x_raw_id in active_y_raw_ids:
        raise ValueError(
            f"Dataset {mapped.display_name!r}: selected X={parametric.x_name!r} "
            "cannot reuse a selected Y observation column."
        )
    source = next(
        (
            item
            for item in mapped.observations
            if item.raw_observation_id == x_raw_id
        ),
        None,
    )
    if source is None:
        raise ValueError(
            f"Dataset {mapped.display_name!r} requires exactly one observed "
            f"source for X={parametric.x_name!r}."
        )
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
            target = target_by_raw_id.get(observation.raw_observation_id, "")
            raise ValueError(
                f"Dataset {mapped.display_name!r}: X={parametric.x_name!r} "
                f"is missing {len(missing)} physical source rows for "
                f"Y={target!r}."
            )
        target = target_by_raw_id.get(observation.raw_observation_id)
        if target is None:
            raise ValueError(
                f"Dataset {mapped.display_name!r}: active Y observation "
                f"{observation.raw_observation_id!r} has no current mapping."
            )
        aligned[target] = np.asarray(
            [x_by_row[int(row)] for row in observation.source_rows],
            dtype=float,
        )
    return aligned


def _compile_dataset_spec(
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


@dataclass(frozen=True, slots=True)
class _FitOptimizerCompilation:
    """Immutable optimizer coordinates lowered from one Fit capture.

    This carrier deliberately contains only evaluator inputs.  It has no Qt
    state and no worker callbacks, so the coordinate authority can be checked
    independently of thread custody.
    """

    included_ids: tuple[str, ...]
    shared_variables: tuple[FitSharedVariableParamSpec, ...]
    coupled_variables: tuple[FitCoupledVariableParamSpec, ...]
    fixed_by_dataset: tuple[
        tuple[str, tuple[tuple[str, float], ...]], ...
    ]
    overrides: tuple[FitDatasetParameterOverrides, ...]
    candidate_names: tuple[str, ...]


def _compile_optimizer_coordinates(
    request: FitExecutionRequest,
) -> _FitOptimizerCompilation:
    """Compile Shared, Per-dataset, Local, and Global coordinates once."""

    included = tuple(item for item in request.datasets if item.included)
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
        config = parameter.selected_config
        dataset_values = dict(config.dataset_values)
        if config.fit and parameter.scope == "shared":
            shared_variables.append(
                FitSharedVariableParamSpec(
                    name=parameter.name,
                    initial=config.value,
                    minimum=config.minimum,
                    maximum=config.maximum,
                    log10=config.log10,
                )
            )
        elif config.fit:
            for dataset_id in included_ids:
                variable_by_dataset[dataset_id].append(
                    FitDatasetVariableParamSpec(
                        name=parameter.name,
                        initial=dataset_values[dataset_id],
                        minimum=config.minimum,
                        maximum=config.maximum,
                        log10=config.log10,
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
                    config.value
                )

    datasets_for_set: dict[str, tuple[str, ...]] = {}
    applicable_dataset_ids: dict[str, set[str]] = {}
    for mapped in request.origin.mapped_datasets:
        if mapped.dataset_id not in included_ids:
            continue
        datasets_for_set.setdefault(mapped.set_id, ())
        datasets_for_set[mapped.set_id] = (
            *datasets_for_set[mapped.set_id],
            mapped.dataset_id,
        )
        for species, _value in mapped.initial_values:
            applicable_dataset_ids.setdefault(species, set()).add(
                mapped.dataset_id
            )

    scopes = dict(request.initial_scopes)
    global_by_species = {item.species: item for item in request.global_initials}
    for species, config in global_by_species.items():
        if scopes.get(species) != "global":
            continue
        applicable_ids = tuple(sorted(applicable_dataset_ids.get(species, ())))
        if not applicable_ids:
            continue
        if config.fit:
            coupled_variables.append(
                FitCoupledVariableParamSpec(
                    variable_id=f"initial:global:{species}",
                    parameter_name=f"{INITIAL_PREFIX}{species}",
                    dataset_ids=applicable_ids,
                    initial=config.initial,
                    minimum=config.minimum,
                    maximum=config.maximum,
                    log10=config.log10,
                )
            )
        else:
            for dataset_id in applicable_ids:
                fixed_by_dataset[dataset_id][
                    f"{INITIAL_PREFIX}{species}"
                ] = config.initial
    for initial in request.initials:
        if scopes.get(initial.species) != "local":
            continue
        target_datasets = datasets_for_set.get(initial.set_id, ())
        if not target_datasets:
            continue
        species = initial.species
        if initial.fit:
            coupled_variables.append(
                FitCoupledVariableParamSpec(
                    variable_id=f"initial:{initial.set_id}:{species}",
                    parameter_name=f"{INITIAL_PREFIX}{species}",
                    dataset_ids=target_datasets,
                    initial=initial.initial,
                    minimum=initial.minimum,
                    maximum=initial.maximum,
                    log10=initial.log10,
                )
            )
        else:
            for dataset_id in target_datasets:
                fixed_by_dataset[dataset_id][
                    f"{INITIAL_PREFIX}{species}"
                ] = initial.initial

    fixed_items = tuple(
        (dataset_id, tuple(values.items()))
        for dataset_id, values in fixed_by_dataset.items()
    )
    overrides = tuple(
        FitDatasetParameterOverrides(
            dataset_id=dataset_id,
            fixed_params=tuple(sorted(fixed_by_dataset[dataset_id].items())),
            variable_params=tuple(variable_by_dataset[dataset_id]),
        )
        for dataset_id in included_ids
    )
    candidate_names = tuple(
        sorted(
            parameter.name
            for parameter in request.parameters
            if not parameter.name.startswith(INITIAL_PREFIX)
        )
    )
    return _FitOptimizerCompilation(
        included_ids=included_ids,
        shared_variables=tuple(shared_variables),
        coupled_variables=tuple(coupled_variables),
        fixed_by_dataset=fixed_items,
        overrides=overrides,
        candidate_names=candidate_names,
    )


@dataclass(frozen=True, slots=True)
class _FitDatasetContextCompilation:
    """Stable dataset sampling inputs used to build one evaluator context."""

    selected: tuple[FitSelectedObservation, ...]
    retained_times: tuple[float, ...]
    grid_points: int
    fit_sampling_policy: str
    fit_sampling_identity: Mapping[str, object]


def _compile_dataset_context(
    mapped: FitMappedDatasetSnapshot,
    local: FitDatasetLocalSettings,
    *,
    solver: str,
) -> _FitDatasetContextCompilation:
    selected = selected_observations(mapped, local)
    retained_times = tuple(
        sorted({float(time_value) for item in selected for time_value in item.t})
    )
    is_parametric = local.parametric is not None
    grid_points = max(100, len(retained_times)) if is_parametric else 100
    policy = (
        "parametric_regular_grid_union_retained_endpoints"
        if is_parametric
        else "fit_internal_regular_grid_union_retained_endpoints"
    )
    return _FitDatasetContextCompilation(
        selected=selected,
        retained_times=retained_times,
        grid_points=grid_points,
        fit_sampling_policy=policy,
        fit_sampling_identity={
            "solver": solver,
            "t_start": float(local.t_start),
            "t_end": float(local.data_range_max),
            "regular_grid_count": int(grid_points),
            "retained_observation_times": retained_times,
            "union_endpoints": (
                float(local.t_start),
                float(local.data_range_max),
            ),
        },
    )


def _initial_values_from_result(
    request: FitExecutionRequest,
    result: GlobalFitResult,
) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    scopes = dict(request.initial_scopes)
    fitted_by_key: dict[tuple[str, str], object] = {
        (initial.set_id, initial.species): initial
        for initial in request.initials
        if initial.fit and scopes.get(initial.species) == "local"
    }
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
    for global_initial in request.global_initials:
        if not global_initial.fit or scopes.get(global_initial.species) != "global":
            continue
        parameter_name = f"{INITIAL_PREFIX}{global_initial.species}"
        values: set[float] = set()
        applicable_ids = sorted(
            mapped.dataset_id
            for mapped in request.origin.mapped_datasets
            if mapped.dataset_id in included_ids
            and any(
                species == global_initial.species
                for species, _value in mapped.initial_values
            )
        )
        for dataset_id in applicable_ids:
            dataset_result = result.dataset_params.get(dataset_id)
            if dataset_result is None or parameter_name not in dataset_result:
                raise ValueError(
                    f"Fit result omitted Global initial {global_initial.species!r} "
                    f"for included dataset {dataset_id!r}."
                )
            value = float(dataset_result[parameter_name])
            if not math.isfinite(value):
                raise ValueError(
                    f"Fit result contains a non-finite Global initial "
                    f"{global_initial.species!r}."
                )
            values.add(value)
            set_ids = {
                mapped.set_id
                for mapped in request.origin.mapped_datasets
                if mapped.dataset_id == dataset_id
            }
            for set_id in set_ids:
                values_by_set.setdefault(set_id, {})[global_initial.species] = value
        if values:
            global_values[global_initial.species] = values
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
        dataset_values = tuple(parameter.selected_config.dataset_values)
        requested_config = requested.selected_config
        if not requested_config.fit:
            parameters.append(
                _replace_fit_parameter(parameter, dataset_values=dataset_values)
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
                f"Per-dataset parameter {parameter.name!r} no longer "
                "matches the included dataset identities."
            )
        fitted_by_id = dict(requested_config.dataset_values)
        for dataset_id, _launch_value in requested_config.dataset_values:
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
            for dataset_id, _value in requested_config.dataset_values
        ]
        fitted_unique_values = {
            value for _dataset_id, value in fitted_values
        }
        parameters.append(
            _replace_fit_parameter(
                parameter,
                initial=(
                    fitted_unique_values.pop()
                    if len(fitted_unique_values) == 1
                    else parameter.selected_config.value
                ),
                dataset_values=tuple(fitted_values),
            )
        )

    initial_results = {
        (set_id, species): value
        for set_id, values in initial_values_by_set
        for species, value in values
    }
    scopes = dict(request.initial_scopes)
    global_initial_results: dict[str, float] = {}
    for global_initial in request.global_initials:
        if not global_initial.fit or scopes.get(global_initial.species) != "global":
            continue
        values = {
            value
            for (set_id, species), value in initial_results.items()
            if species == global_initial.species and set_id in included_set_ids
        }
        if len(values) != 1:
            raise ValueError(
                f"Fit result did not resolve one global value for initial "
                f"{global_initial.species!r}."
            )
        global_initial_results[global_initial.species] = values.pop()
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
            and scopes.get(initial.species) == "local"
            and (initial.set_id, initial.species) in initial_results
        )
        else initial
        for initial in setup_snapshot.initials
    )
    global_initials = tuple(
        replace(
            initial,
            initial=global_initial_results[initial.species],
        )
        if (
            initial.species in global_initial_results
            and scopes.get(initial.species) == "global"
        )
        else initial
        for initial in setup_snapshot.global_initials
    )
    return replace(
        setup_snapshot,
        datasets=tuple(
            effective_datasets.get(item.dataset_id, item)
            for item in setup_snapshot.datasets
        ),
        parameters=tuple(parameters),
        initials=initials,
        global_initials=global_initials,
        initial_scopes=setup_snapshot.initial_scopes,
        local_observables=setup_snapshot.local_observables,
    )


def _canonical_curve_projection(
    result: SimulationResultValue,
    selected: Sequence[FitSelectedObservation],
    *,
    x_name: str,
) -> tuple[
    tuple[str, ...],
    np.ndarray,
    dict[str, np.ndarray],
    tuple[int, ...],
    tuple[str, ...],
    tuple[int, ...],
]:
    """Project one canonical result for display without copying its authority."""

    names = tuple(dict.fromkeys(item.model_species for item in selected))
    time = np.asarray(result.t, dtype=float).reshape(-1)
    segment_ids = np.asarray(result.segment_ids, dtype=np.int64).reshape(-1)
    segment_keys = tuple(result.segment_keys)
    discontinuity_ids = np.asarray(
        result.discontinuity_ids,
        dtype=np.int64,
    ).reshape(-1)
    axis = (
        time
        if x_name == "t"
        else np.asarray(result.series[x_name], dtype=float).reshape(-1)
    )
    values = {
        name: np.asarray(result.series[name], dtype=float).reshape(-1)
        for name in names
    }
    return (
        names,
        axis,
        values,
        tuple(int(value) for value in segment_ids),
        segment_keys,
        tuple(int(value) for value in discontinuity_ids),
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


def _curves_from_result(
    origin: FitOriginSnapshot,
    setup_snapshot: FitSetupSnapshot,
    result: GlobalFitResult,
) -> tuple[FitCurveDataset, ...]:
    mapped_by_id = {item.dataset_id: item for item in origin.mapped_datasets}
    local_by_id = {item.dataset_id: item for item in setup_snapshot.datasets}
    curves: list[FitCurveDataset] = []
    for dataset_id, simulation in result.simulation_results_by_dataset.items():
        if not isinstance(simulation, SimulationResultValue):
            raise TypeError("Terminal Fit result lost its canonical simulation result.")
        mapped = mapped_by_id[dataset_id]
        local = local_by_id[dataset_id]
        selected = selected_observations(mapped, local)
        observed_axes = _observed_x_by_target(mapped, selected, local)
        observed_x, observed_y = _observed_curve_parts(
            selected,
            x_by_target=observed_axes,
        )
        x_name = local.parametric.x_name if local.parametric is not None else "t"
        (
            model_names,
            model_axis,
            model_values,
            model_segment_ids,
            model_segment_keys,
            model_discontinuity_ids,
        ) = _canonical_curve_projection(
            simulation,
            selected,
            x_name=x_name,
        )
        objective_model_series = result.model_series[dataset_id]
        curves.append(
            FitCurveDataset(
                dataset_id=dataset_id,
                display_name=mapped.display_name,
                y_semantics=_fit_curve_y_semantics(
                    setup_snapshot.prepared_dsl,
                    model_names,
                ),
                model_x=tuple(
                    (name, tuple(float(value) for value in model_axis))
                    for name in model_names
                ),
                model_y=tuple(
                    (name, tuple(float(value) for value in model_values[name]))
                    for name in model_names
                ),
                observed_x=observed_x,
                observed_y=observed_y,
                model_segment_ids=tuple(
                    (name, model_segment_ids) for name in model_names
                ),
                model_segment_keys=tuple(
                    (name, model_segment_keys) for name in model_names
                ),
                model_discontinuity_ids=tuple(
                    (name, model_discontinuity_ids) for name in model_names
                ),
                observed_source_rows=tuple(
                    (item.model_species, item.source_rows) for item in selected
                ),
                observed_raw_ids=tuple(
                    (item.model_species, item.raw_observation_id)
                    for item in selected
                ),
                observed_raw_columns=tuple(
                    (item.model_species, item.raw_column) for item in selected
                ),
                objective_model_x=tuple(
                    (
                        name,
                        tuple(
                            float(value)
                            for value in result.model_x_by_species[dataset_id][name]
                        ),
                    )
                    for name in objective_model_series
                ),
                objective_model_y=tuple(
                    (
                        name,
                        tuple(
                            float(value)
                            for value in objective_model_series[name]
                        ),
                    )
                    for name in objective_model_series
                ),
                objective_model_segment_ids=tuple(
                    (
                        name,
                        tuple(
                            int(value)
                            for value in result.model_segment_ids[dataset_id][name]
                        ),
                    )
                    for name in objective_model_series
                ),
                objective_model_segment_keys=tuple(
                    (
                        name,
                        tuple(result.model_segment_keys[dataset_id][name]),
                    )
                    for name in objective_model_series
                ),
                objective_model_discontinuity_ids=tuple(
                    (
                        name,
                        tuple(
                            int(value)
                            for value in result.model_discontinuity_ids[dataset_id][name]
                        ),
                    )
                    for name in objective_model_series
                ),
                objective_selection_provenance=tuple(
                    (
                        name,
                        tuple(result.model_selection_provenance[dataset_id][name]),
                    )
                    for name in objective_model_series
                ),
                objective_observation_ids=tuple(
                    (
                        name,
                        tuple(result.model_observation_ids[dataset_id][name]),
                    )
                    for name in objective_model_series
                ),
                objective_raw_columns=tuple(
                    (
                        name,
                        tuple(result.model_raw_columns[dataset_id][name]),
                    )
                    for name in objective_model_series
                ),
                objective_source_rows=tuple(
                    (
                        name,
                        tuple(result.model_source_rows[dataset_id][name]),
                    )
                    for name in objective_model_series
                ),
                x_name=x_name,
            )
        )
    return tuple(curves)


def project_fit_candidate_curves(
    candidate: FitCandidate,
) -> tuple[FitCurveDataset, ...]:
    """Lazily project terminal curves from the candidate's canonical truth."""

    if not isinstance(candidate, FitCandidate):
        raise TypeError("candidate must be FitCandidate.")
    return _curves_from_result(
        candidate.origin,
        candidate.setup_snapshot,
        candidate.result,
    )


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
        if not isinstance(raw_evaluation, SimulationResultValue):
            raise TypeError("Live Fit evaluation lost its canonical result.")
        evaluation = raw_evaluation
        local = effective_local[dataset_id]
        mapped = mapped_by_id[dataset_id]
        selected = selected_observations(mapped, local)
        observed_axes = _observed_x_by_target(mapped, selected, local)
        observed_x, observed_y = _observed_curve_parts(
            selected,
            x_by_target=observed_axes,
        )
        result_series = evaluation.series
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
            np.asarray(evaluation.t, dtype=float)
            if x_name == "t"
            else np.asarray(result_series[x_name], dtype=float)
        )
        objective_rows = {
            item.model_species: select_fit_rows(
                model_y=np.asarray(
                    result_series[item.model_species],
                    dtype=float,
                ),
                model_x=(
                    None
                    if x_name == "t"
                    else np.asarray(result_series[x_name], dtype=float)
                ),
                x_name=x_name,
                x_mode=(
                    local.parametric.resolved_mode
                    if local.parametric is not None
                    else "monotone"
                ),
                sim_time=np.asarray(evaluation.t, dtype=float),
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
                y_semantics=_fit_curve_y_semantics(
                    request.prepared_dsl,
                    tuple(item.model_species for item in selected),
                ),
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
                                result_series[item.model_species],
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


def _array_identity(value: object) -> dict[str, object]:
    """Describe an array without retaining its scientific contents."""

    array = np.asarray(value)
    if array.dtype.kind in {"O", "U", "S"}:
        content = json.dumps(
            array.reshape(-1).tolist(),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    else:
        content = np.ascontiguousarray(array).tobytes(order="C")
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode("ascii", "replace"))
    hasher.update(json.dumps(tuple(int(item) for item in array.shape)).encode("ascii"))
    hasher.update(content)
    return {
        "dtype": str(array.dtype),
        "shape": tuple(int(item) for item in array.shape),
        "sha256": hasher.hexdigest(),
    }


def _compact_metadata(value: object) -> object:
    """Normalize preparation metadata while replacing arrays with digests."""

    if isinstance(value, np.ndarray):
        return _array_identity(value)
    if isinstance(value, np.generic):
        return _compact_metadata(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _compact_metadata(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_compact_metadata(item) for item in value]
    return value


def _metadata_digest(value: object) -> str:
    normalized = _compact_metadata(value)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _simulation_result_array_identity(
    result: GlobalFitResult,
) -> list[dict[str, object]]:
    """Hash canonical terminal simulations without building curve mirrors."""

    return [
        {
            "dataset_id": dataset_id,
            "time": _array_identity(simulation.t),
            "series": {
                name: _array_identity(values)
                for name, values in sorted(simulation.series.items())
            },
            "segment_ids": _array_identity(simulation.segment_ids),
            "segment_keys": _array_identity(simulation.segment_keys),
            "discontinuity_ids": _array_identity(
                simulation.discontinuity_ids
            ),
        }
        for dataset_id, simulation in sorted(
            result.simulation_results_by_dataset.items()
        )
    ]


def _result_array_identity(result: GlobalFitResult) -> list[dict[str, object]]:
    """Retain dtype-sensitive digests before presentation casts occur."""

    rows: list[dict[str, object]] = []
    for dataset_id in sorted(result.model_series):
        rows.append(
            {
                "dataset_id": dataset_id,
                "series": {
                    series_name: {
                        "model_x": _array_identity(
                            result.model_x_by_species[dataset_id][series_name]
                        ),
                        "model_y": _array_identity(
                            result.model_series[dataset_id][series_name]
                        ),
                        "segment_ids": _array_identity(
                            result.model_segment_ids[dataset_id][series_name]
                        ),
                        "segment_keys": _array_identity(
                            result.model_segment_keys[dataset_id][series_name]
                        ),
                        "discontinuity_ids": _array_identity(
                            result.model_discontinuity_ids[dataset_id][series_name]
                        ),
                        "selection_provenance": _array_identity(
                            result.model_selection_provenance[dataset_id][series_name]
                        ),
                        "observation_ids": _array_identity(
                            result.model_observation_ids[dataset_id][series_name]
                        ),
                        "raw_columns": _array_identity(
                            result.model_raw_columns[dataset_id][series_name]
                        ),
                        "source_rows": _array_identity(
                            result.model_source_rows[dataset_id][series_name]
                        ),
                    }
                    for series_name in sorted(result.model_series[dataset_id])
                },
            }
        )
    return rows


def _compile_fit_run_identity_payload(
    *,
    request: FitExecutionRequest,
    datasets: Sequence[FitDatasetSpec],
    included: tuple[FitDatasetLocalSettings, ...],
    mapped_by_id: Mapping[str, FitMappedDatasetSnapshot],
    effective_local: Mapping[str, FitDatasetLocalSettings],
    prepared_ids: Mapping[str, Mapping[str, object]],
    resolved_modes: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    """Build a compact identity; scientific arrays are represented by digests."""

    dataset_payloads: list[dict[str, object]] = []
    for item in datasets:
        mapped = mapped_by_id[item.dataset_id]
        local = effective_local[item.dataset_id]
        launch_local = next(
            value for value in included if value.dataset_id == item.dataset_id
        )
        raw_columns = {
            observation.raw_observation_id: observation.raw_column
            for observation in mapped.observations
        }
        observations = {
            name: {
                "raw_observation_id": series.raw_observation_id,
                "t": _array_identity(series.t),
                "y": _array_identity(series.y),
                "source_rows": _array_identity(series.source_rows),
                "raw_column": raw_columns[series.raw_observation_id],
                "observed_x": _array_identity(
                    item.x_obs_by_species[name] if item.x_name != "t" else series.t
                ),
            }
            for name, series in item.observations.items()
        }
        retained_times = tuple(
            sorted(
                {
                    float(time_value)
                    for observation in selected_observations(mapped, local)
                    for time_value in observation.t
                }
            )
        )
        dataset_payloads.append(
            {
                "dataset_id": item.dataset_id,
                "set_id": mapped.set_id,
                "observations": observations,
                "mappings": [
                    {
                        "raw_observation_id": raw_id,
                        "raw_column": raw_columns[raw_id],
                        "model_target": model_target,
                    }
                    for raw_id, model_target in launch_local.observation_targets
                ],
                "targets": item.species_list,
                "target_weights": dict(item.target_weights),
                "sampling": {
                    "data_range_min": float(local.data_range_min),
                    "data_range_max": float(local.data_range_max),
                    "maximum_observations_per_target": local.maximum_observations_per_target,
                },
                **(
                    {"dataset_weight": float(local.weight)}
                    if request.dataset_weight_mode == "user"
                    else {}
                ),
                "fit_solver": request.solver,
                "fit_interval": {
                    "t_start": float(local.t_start),
                    "data_range_max": float(local.data_range_max),
                },
                "fit_sampling_policy": (
                    "parametric_regular_grid_union_retained_endpoints"
                    if local.parametric is not None
                    else "fit_internal_regular_grid_union_retained_endpoints"
                ),
                "fit_sampling_identity": {
                    "retained_observation_times": _array_identity(retained_times),
                    "retained_observation_count": len(retained_times),
                    "regular_grid_count": max(100, len(retained_times))
                    if local.parametric is not None
                    else 100,
                },
                "x_name": item.x_name,
                "x_mapping_requested": (
                    local.parametric.requested_mode if local.parametric is not None else "time"
                ),
                "x_mapping_resolved": (
                    local.parametric.resolved_mode if local.parametric is not None else "time"
                ),
            }
        )
    return {
        "kind": "global_fit_run",
        "schema_version": 2,
        "origin_tab_id": request.origin.tab_id,
        "origin_capture_id": request.origin.capture_id,
        "mechanism_text": request.mechanism_source.reactions_text,
        "fit_solver": request.solver,
        "dataset_weight_mode": request.dataset_weight_mode,
        "local_observables": [
            {
                "name": definition.name,
                "expression": definition.expression,
                "scalars": [
                    {
                        "name": scalar.name,
                        "value": float(scalar.value),
                        "minimum": _stamp_bound(scalar.minimum),
                        "maximum": _stamp_bound(scalar.maximum),
                        "scope": scalar.scope,
                    }
                    for scalar in definition.scalar_parameters
                ],
            }
            for definition in fit_local_observable_definitions(
                request.local_observables,
                request.parameters,
            )
        ],
        "datasets": dataset_payloads,
        "parameters": [
            {
                "name": item.name,
                "display_name": item.display_name,
                "fit": item.selected_config.fit,
                "initial": item.selected_config.value,
                "minimum": _stamp_bound(item.selected_config.minimum),
                "maximum": _stamp_bound(item.selected_config.maximum),
                "log10": item.selected_config.log10,
                "scope": item.scope,
                "dataset_values": dict(item.selected_config.dataset_values),
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
                "scope": "local",
            }
            for item in request.initials
        ],
        "global_initials": [
            {
                "species": item.species,
                "fit": item.fit,
                "initial": item.initial,
                "minimum": _stamp_bound(item.minimum),
                "maximum": _stamp_bound(item.maximum),
                "log10": item.log10,
                "scope": dict(request.initial_scopes).get(item.species),
            }
            for item in request.global_initials
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
            "temperature": float(request.origin.scientific_settings.temperature),
            "rtol": float(request.origin.scientific_settings.rtol),
            "atol": float(request.origin.scientific_settings.atol),
            "analytic_jacobian_policy": bool(request.origin.scientific_settings.analytic_jacobian_policy),
            "wegscheider_policy": bool(request.origin.scientific_settings.wegscheider_policy),
        },
        "prepared_identities": {
            dataset_id: {"sha256": _metadata_digest(metadata)}
            for dataset_id, metadata in sorted(prepared_ids.items())
        },
        "resolved_auto_modes": dict(resolved_modes),
    }


@dataclass(frozen=True, slots=True)
class _FitTerminalAssembly:
    """Deterministic successful terminal projection owned by no Qt object."""

    candidate: FitCandidate
    run_identity: FitRunIdentity


def _assemble_fit_terminal(
    *,
    request: FitExecutionRequest,
    result: GlobalFitResult,
    included: tuple[FitDatasetLocalSettings, ...],
    included_ids: tuple[str, ...],
    mapped_by_id: Mapping[str, FitMappedDatasetSnapshot],
    effective_local: Mapping[str, FitDatasetLocalSettings],
    setup_snapshot: FitSetupSnapshot,
    resolved_modes: tuple[tuple[str, str], ...],
    evaluators: Mapping[str, SerialFittingEvaluator],
    prepared_ids: dict[str, dict[str, object]],
    run_identity_payload: dict[str, object],
) -> _FitTerminalAssembly:
    """Validate provenance and build candidate/curves without worker custody."""

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
        str(dataset_id or "").strip(): simulation.provenance.solver
        for dataset_id, simulation in result.simulation_results_by_dataset.items()
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
    initial_values_by_set = _initial_values_from_result(request, result)
    terminal_snapshot = _terminal_setup_snapshot(
        setup_snapshot,
        request=request,
        result=result,
        effective_datasets=effective_local,
        initial_values_by_set=initial_values_by_set,
    )
    for dataset_id, evaluator in evaluators.items():
        prepared_ids[dataset_id]["prepared_simulation"] = (
            evaluator.prepared_metadata.to_serializable_dict()
        )
    run_identity_payload["prepared_identities"] = {
        dataset_id: {"sha256": _metadata_digest(metadata)}
        for dataset_id, metadata in sorted(prepared_ids.items())
    }
    run_identity_payload["terminal"] = {
        "fit_solver": request.solver,
        "status": result.completion.status,
        "optimizer_converged": bool(result.completion.optimizer_converged),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "weighted_residual_sse": float(result.weighted_residual_sse),
        "weighted_residual_mse": float(result.weighted_residual_mse),
        "effective_dataset_weights": {
            dataset_id: effective_dataset_weights[dataset_id]
            for dataset_id in sorted(effective_dataset_weights)
        },
        "solver_provenance_identity_by_dataset": {
            dataset_id: {
                "sha256": _metadata_digest(
                    solver_provenance_by_dataset[dataset_id]
                )
            }
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
            for dataset_id, values in sorted(result.dataset_params.items())
        },
        "result_array_identity": _result_array_identity(result),
        "simulation_result_array_identity": (
            _simulation_result_array_identity(result)
        ),
    }
    run_identity = FitRunIdentity.build(run_identity_payload)
    candidate = FitCandidate(
        origin=request.origin,
        parameters=_finite_items(result.shared_params, field="Fit parameters"),
        scalar_values_by_dataset=scalar_values,
        initial_values_by_set=initial_values_by_set,
        local_observables=terminal_snapshot.local_observables,
        result=result,
        run_identity=run_identity,
        setup_snapshot=terminal_snapshot,
        resolved_auto_modes=resolved_modes,
    )
    return _FitTerminalAssembly(candidate=candidate, run_identity=run_identity)


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
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(request, FitExecutionRequest):
            raise TypeError("request must be FitExecutionRequest.")
        self._request = request
        self._setup_snapshot = FitSetupSnapshot.from_request(request)
        self._cancelled = Event()
        self._cancellation_control = SimulationCancellationControl(
            self._cancelled.is_set
        )
        self._pause_condition = Condition()
        self._paused = False
        self._last_best_emit = 0.0
        self._last_curve_emit = 0.0
        self._best_progress: FitProgress | None = None
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
                if rejection_message is not None:
                    raise ValueError(
                        "An accepted detailed-balance correction cannot carry "
                        "a rejection message."
                    )
                self._detailed_balance_response = (
                    corrected_request,
                    FitSetupSnapshot.from_request(corrected_request),
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
                self._best_progress = None
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
        request = self._request
        included = tuple(item for item in request.datasets if item.included)
        mapped_by_id = {item.dataset_id: item for item in request.origin.mapped_datasets}
        coordinates = _compile_optimizer_coordinates(request)
        included_ids = coordinates.included_ids
        shared_variables = list(coordinates.shared_variables)
        coupled_variables = list(coordinates.coupled_variables)
        fixed_by_dataset = {
            dataset_id: dict(values)
            for dataset_id, values in coordinates.fixed_by_dataset
        }
        overrides = coordinates.overrides
        candidate_names = list(coordinates.candidate_names)
        evaluators: dict[str, SerialFittingEvaluator] = {}
        prepared_ids: dict[str, dict[str, object]] = {}
        for local in included:
            if self._checkpoint():
                raise FittingCancelled()
            science = request.origin.scientific_settings
            dataset_context = _compile_dataset_context(
                mapped_by_id[local.dataset_id],
                local,
                solver=request.solver,
            )
            try:
                context = prepare_fitting_execution_context(
                    prepared_dsl=request.prepared_dsl,
                    param_names=candidate_names,
                    t_start=float(local.t_start),
                    t_end=float(local.data_range_max),
                    num_points=dataset_context.grid_points,
                    objective_sample_times=dataset_context.retained_times,
                    solver=request.solver,
                    fit_sampling_policy=dataset_context.fit_sampling_policy,
                    fit_sampling_identity=dataset_context.fit_sampling_identity,
                    cancellation_control=self._cancellation_control,
                    rtol=science.rtol,
                    atol=science.atol,
                    use_analytic_jacobian=science.analytic_jacobian_policy,
                    wegscheider_cyclicity_enabled=science.wegscheider_policy,
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
                            "fit_solver": request.solver,
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
                                request.prepared_dsl.analysis.energy_unit
                            ),
                    )
                    ) from exc
                if isinstance(exc, FitSimulationError):
                    exc.details.setdefault("fit_solver", request.solver)
                raise
            evaluator = SerialFittingEvaluator(
                context,
                fixed_param_origins={
                    name: FITTING_PARAM_ORIGIN_CONFIGURED_DATASET
                    for name in fixed_by_dataset[local.dataset_id]
                },
            )
            evaluator.set_cancellation_control(self._cancellation_control)
            evaluators[local.dataset_id] = evaluator
            metadata = evaluator.prepared_metadata
            prepared_ids[local.dataset_id] = {
                "prepared_simulation": metadata.to_serializable_dict(),
                "declarative_analysis_identity": (
                    request.prepared_dsl.analysis.identity
                ),
                "fit_solver": request.solver,
                "fit_interval": {
                    "t_start": float(local.t_start),
                    "data_range_max": float(local.data_range_max),
                },
                "fit_sampling_policy": context.fit_sampling_policy,
                "fit_sampling_identity": context.fit_sampling_identity,
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
                    dict(parameter.selected_config.dataset_values)[dataset_id]
                    if parameter.scope == "per_dataset"
                    else parameter.selected_config.value
                )
                for parameter in request.parameters
            }
            starting_values.update(fixed_by_dataset[dataset_id])
            for initial in request.initials:
                if (
                    initial.set_id != mapped.set_id
                    or dict(request.initial_scopes).get(initial.species) != "local"
                ):
                    continue
                starting_values[
                    f"{INITIAL_PREFIX}{initial.species}"
                ] = initial.initial
            for initial in request.global_initials:
                if (
                    dict(request.initial_scopes).get(initial.species) == "global"
                    and mapped.set_id in {
                        item.set_id
                        for item in request.initials
                        if item.species == initial.species
                    }
                ):
                    starting_values[f"{INITIAL_PREFIX}{initial.species}"] = initial.initial
            evaluation = evaluators[
                dataset_id
            ].evaluate_series_with_parameter_origins(
                starting_values,
                {
                    name: FITTING_PARAM_ORIGIN_CONFIGURED_DATASET
                    for name in starting_values
                },
            )
            if not isinstance(evaluation, SimulationResultValue):
                raise TypeError("Fit evaluator did not return its canonical result.")
            starting_evaluations[dataset_id] = evaluation
            prepared_ids[dataset_id]["validated_executable_schedule"] = (
                evaluation.intervention_execution.provenance_projection()
            )
            if (
                local.parametric is not None
                and local.parametric.requested_mode == "auto"
            ):
                provisional = _compile_dataset_spec(
                    mapped,
                    replace(
                        local,
                        parametric=replace(
                            local.parametric,
                            resolved_mode="monotone",
                        ),
                    ),
                )
                series = evaluation
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
                            x_model=series.series[provisional.x_name],
                            y_model=series.series[target_name],
                            segment_ids=series.segment_ids,
                            segment_keys=series.segment_keys,
                            discontinuity_ids=series.discontinuity_ids,
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
            _compile_dataset_spec(
                mapped_by_id[item.dataset_id], effective_local[item.dataset_id]
            )
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
        try:
            initial_curves = _curves_from_evaluations(
                request,
                starting_evaluations,
                effective_local,
            )
        except DeterministicParametricResolutionError:
            # The starting evaluation is scientific input to Auto-mode
            # resolution, but its aligned curve is only presentation.  A
            # candidate-domain no-crossing therefore suppresses this optional
            # projection without preventing the optimizer from exploring the
            # admitted domain.
            initial_curves = ()
        if self._checkpoint():
            raise FittingCancelled()
        self.curves_updated.emit(
            initial_curves
        )
        if self._checkpoint():
            raise FittingCancelled()
        self._last_curve_emit = time.monotonic()
        run_identity_payload = _compile_fit_run_identity_payload(
            request=request,
            datasets=datasets,
            included=included,
            mapped_by_id=mapped_by_id,
            effective_local=effective_local,
            prepared_ids=prepared_ids,
            resolved_modes=resolved_modes,
        )

        def progress(update: FitProgress) -> None:
            if self._checkpoint():
                raise FittingCancelled()
            if not isinstance(update, FitProgress):
                raise TypeError("Fit progress callback requires FitProgress.")
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
                            / max(1, request.defaults.max_nfev)
                        ),
                    ),
                ),
                f"Objective calls {objective_calls}: objective SSQ={cost:.4g}",
            )
            if self._checkpoint():
                raise FittingCancelled()
            if update.improved:
                self._best_progress = update
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
                        live_curves = (
                            _curves_from_evaluations(
                                request,
                                update.best_simulation_results_by_dataset,
                                effective_local,
                            )
                            if update.best_simulation_results_by_dataset
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

        method = request.defaults.method
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
            method=method,
            max_nfev=int(request.defaults.max_nfev),
            ftol=float(request.defaults.ftol),
            xtol=float(request.defaults.xtol),
            seed=int(request.defaults.seed) if request.defaults.use_seed else None,
            progress_callback=progress,
            cancellation_check=fitting_checkpoint,
            cancellation_control=self._cancellation_control,
        )
        if self._checkpoint():
            raise FittingCancelled()
        if result.completion.status == "fail":
            message = result.message or "Fit did not produce a candidate."
            raise RuntimeError(message)

        if self._checkpoint():
            raise FittingCancelled()
        terminal_assembly = _assemble_fit_terminal(
            request=request,
            result=result,
            included=included,
            included_ids=included_ids,
            mapped_by_id=mapped_by_id,
            effective_local=effective_local,
            setup_snapshot=self._setup_snapshot,
            resolved_modes=resolved_modes,
            evaluators=evaluators,
            prepared_ids=prepared_ids,
            run_identity_payload=run_identity_payload,
        )
        run_identity = terminal_assembly.run_identity
        if self._checkpoint():
            raise FittingCancelled()
        self.progress.emit(100, f"Fit complete · {run_identity.short_hash}")
        if self._checkpoint():
            raise FittingCancelled()
        return terminal_assembly.candidate

    def _live_update(self) -> FitLiveUpdate:
        best_progress = self._best_progress
        if best_progress is None:
            raise RuntimeError("Live Fit update requested before a best candidate.")
        best_params = best_progress.parameters
        included_dataset_ids = {
            item.dataset_id for item in self._request.datasets if item.included
        }
        included_set_ids = {
            mapped.set_id
            for mapped in self._request.origin.mapped_datasets
            if mapped.dataset_id in included_dataset_ids
        }
        initials_by_set: dict[str, dict[str, float]] = {}
        scopes = dict(self._request.initial_scopes)
        for initial in self._request.initials:
            if (
                not initial.fit
                or initial.set_id not in included_set_ids
                or scopes.get(initial.species) != "local"
            ):
                continue
            variable_id = f"initial:{initial.set_id}:{initial.species}"
            value = best_params.get(variable_id)
            if value is not None and math.isfinite(float(value)):
                initials_by_set.setdefault(initial.set_id, {})[
                    initial.species
                ] = float(value)
        for initial in self._request.global_initials:
            if not initial.fit or scopes.get(initial.species) != "global":
                continue
            value = best_params.get(f"initial:global:{initial.species}")
            if value is None or not math.isfinite(float(value)):
                continue
            applicable_set_ids = {
                mapped.set_id
                for mapped in self._request.origin.mapped_datasets
                if mapped.dataset_id in included_dataset_ids
                and any(
                    species == initial.species
                    for species, _value in mapped.initial_values
                )
            }
            for set_id in applicable_set_ids:
                initials_by_set.setdefault(set_id, {})[initial.species] = float(value)
        return FitLiveUpdate(
            iteration=best_progress.objective_calls,
            cost=best_progress.cost,
            parameters=_finite_items(
                {
                    name: value
                    for name, value in best_params.items()
                    if "::" not in name and not name.startswith("initial:")
                },
                field="live Fit parameters",
            ),
            scalar_values_by_dataset=tuple(
                (
                    dataset_id,
                    _finite_items(
                        {
                            parameter.name: best_params.get(
                                f"{dataset_id}::{parameter.name}",
                                dict(parameter.selected_config.dataset_values)[dataset_id],
                            )
                            for parameter in self._request.parameters
                            if (
                                parameter.selected_config.fit
                                and parameter.scope == "per_dataset"
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
