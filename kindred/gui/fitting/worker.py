"""One pausable/cancellable Fit QThread over immutable typed captures."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from threading import Condition, Event

import numpy as np
from PySide6 import QtCore

from kindred.core.algebra.observable_introspection import (
    AdmittedAlgebraProgram,
    FitLocalObservable,
)
from kindred.core.analysis.fit_dataset_payload import (
    FitDatasetSpec,
    FitObservationSeries,
)
from kindred.core.analysis.global_fitting import (
    _shared_refit_provenance,
    _validate_shared_refit_replay,
    fit_global,
    fit_global_shared_refit,
)
from kindred.core.analysis.global_fit_execution import (
    _assemble_fit_replay_state,
    _construct_global_fit_result,
    CentredFixedEffectsSettlement,
    FitCoordinate,
    FitParameterLayout,
    FitProgress,
    GlobalFitResult,
    fit_result_target_values,
    normalize_weights,
    _select_fit_rows,
)
from kindred.core.analysis.parametric_alignment import (
    DeterministicParametricResolutionError,
    align_parametric_x,
    project_parametric_y,
)
from kindred.core.analysis.fit_evaluation_outcome import (
    FitEvaluationOutcome,
    FittingSeriesOutcome,
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
    SerialFittingEvaluator,
    prepare_fitting_execution_context,
)
from kindred.core.fitting_completion import (
    FitExtensionProgress,
    FitTerminalExtension,
)
from kindred.core.result_repository import SimulationResultValue
from kindred.core.simulator.solver_types import SimulationCancellationControl
from kindred.core.simulation_failure import (
    FactualCause,
    factual_cause_from_exception,
)
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
    _FitDispatchCompilation,
    FitExecutionRequest,
    FitMappedDatasetSnapshot,
    FitOriginSnapshot,
    FitSelectedObservation,
    FitSetupSnapshot,
    _replace_fit_parameter,
    selected_observations,
)
from kindred.gui.parameter_enumeration import admit_gui_parameter_value
from kindred.core.species import validate_concentration
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


def _fit_coordinate_display_label(value: FitCoordinate) -> str:
    """Return a human label that cannot alias typed coordinate scopes."""

    if value.scope == "centred_overall":
        return f"Overall · {value.name}"
    if value.scope == "centred_deviation":
        return f"Deviation · {value.dataset_id}::{value.name}"
    if value.scope in {"shared", "coupled"}:
        return value.display_name
    return value.display_name


def _fit_coordinate_machine_identity(value: object) -> object:
    """Serialize typed coordinates recursively without reconstructing names.

    Centred overall and deviation coordinates may share the same parameter
    name, so the machine boundary retains the complete discriminant and a
    scope-qualified display label. Nested settlement/provenance structures
    are walked here because the core result intentionally retains typed
    ``FitCoordinate`` values in its immutable details.
    """

    if isinstance(value, FitCoordinate):
        return value.to_machine_payload()
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, raw_item in value.items():
            if isinstance(raw_key, FitCoordinate):
                key = raw_key.to_machine_key()
            else:
                key = str(raw_key)
            normalized[key] = _fit_coordinate_machine_identity(raw_item)
        return normalized
    if isinstance(value, (tuple, list)):
        return [
            _fit_coordinate_machine_identity(item)
            for item in value
        ]
    if isinstance(value, (set, frozenset)):
        return [
            _fit_coordinate_machine_identity(item)
            for item in sorted(value, key=repr)
        ]
    if isinstance(value, np.generic):
        return _fit_coordinate_machine_identity(value.item())
    return value


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
    """Scientifically valid terminal result, still local to the Fit window."""

    request: FitExecutionRequest
    result: GlobalFitResult
    run_identity: FitRunIdentity
    resolved_auto_modes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, FitExecutionRequest):
            raise TypeError("Fit candidate request must be retained.")
        if not isinstance(self.result, GlobalFitResult):
            raise TypeError("Fit candidate result must be canonical.")
        nominal_setup = FitSetupSnapshot.from_request(self.request)
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
        refit_execution = dispatch.shared_refit_execution
        admissions = (
            ()
            if refit_execution is None
            else refit_execution.promotion_admissions
        )
        admission_names = {
            item.shared_variable.name for item in admissions
        }
        result_refit_provenance = self.result.shared_refit_provenance
        result_promotions = (
            result_refit_provenance.get("promotions")
            if isinstance(result_refit_provenance, Mapping)
            else None
        )
        if admission_names:
            if (
                result_refit_provenance.get("execution")
                != "Promote to Shared and Refit"
                or not isinstance(result_promotions, Mapping)
                or set(result_promotions) != admission_names
                or any(
                    result_promotions.get(item.shared_variable.name)
                    != item.provenance
                    for item in admissions
                )
            ):
                raise ValueError(
                    "Fit candidate lost its admitted Shared-refit provenance."
                )
        elif result_refit_provenance:
            raise ValueError(
                "Ordinary Fit candidate cannot carry Shared-refit provenance."
            )
        scalar_dataset_ids = {
            dataset_id
            for dataset_id in self.result.dataset_params
        }
        result_dataset_ids = set(self.result.dataset_params)
        simulation_dataset_ids = set(
            self.result.simulation_results_by_dataset
        )
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
        return self.request.origin

    @property
    def nominal_setup_snapshot(self) -> FitSetupSnapshot:
        return FitSetupSnapshot.from_request(self.request)

    @property
    def parameters(self) -> tuple[tuple[str, float], ...]:
        """Project shared fitted values solely from the canonical result."""

        return _finite_items(self.result.shared_params, field="Fit parameters")

    @property
    def scalar_values_by_dataset(
        self,
    ) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
        """Project dataset scalar values solely from the canonical result."""

        return tuple(
            (
                dataset_id,
                _finite_items(
                    {
                        name: value for name, value in values.items()
                        if not name.startswith(INITIAL_PREFIX)
                    },
                    field=f"Fit scalar values {dataset_id}",
                ),
            )
            for dataset_id, values in sorted(self.result.dataset_params.items())
        )

    @property
    def initial_values_by_set(
        self,
    ) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
        """Project fitted initial concentrations solely from the result."""

        return _initial_values_from_result(
            self.nominal_setup_snapshot,
            self.origin,
            self.result,
        )

    @property
    def local_observables(self) -> tuple[FitLocalObservable, ...]:
        return self.nominal_setup_snapshot.local_observables

    @property
    def setup_snapshot(self) -> FitSetupSnapshot:
        """Build the post-terminal next-run projection from result authority."""

        resolved = dict(self.resolved_auto_modes)
        effective = {
            item.dataset_id: (
                replace(
                    item,
                    parametric=replace(
                        item.parametric,
                        resolved_mode=resolved[item.dataset_id],
                    ),
                )
                if item.dataset_id in resolved and item.parametric is not None
                else item
            )
            for item in self.nominal_setup_snapshot.datasets
        }
        return _terminal_setup_snapshot(
            self.nominal_setup_snapshot,
            origin=self.origin,
            result=self.result,
            dispatch=self.request.dispatch,
            effective_datasets=effective,
            initial_values_by_set=self.initial_values_by_set,
        )


@dataclass(frozen=True, slots=True)
class FitWorkerTerminal:
    """Immutable worker-to-owner factual terminal contract."""

    candidate: FitCandidate | None
    failure: FactualCause | None = None
    outcome: FitEvaluationOutcome | None = None
    lifecycle_cause: str = "failed"

    def __post_init__(self) -> None:
        if self.candidate is not None and not isinstance(self.candidate, FitCandidate):
            raise TypeError("Fit worker candidate must be FitCandidate.")
        if self.failure is not None and not isinstance(self.failure, FactualCause):
            raise TypeError("Fit worker failure must be FactualCause.")
        if self.outcome is not None and not isinstance(self.outcome, FitEvaluationOutcome):
            raise TypeError("Fit worker outcome must be FitEvaluationOutcome.")
        if self.lifecycle_cause not in {
            "completed", "user_stop", "cancelled", "failed", "scientific_unavailable",
        }:
            raise ValueError("Fit worker lifecycle cause is invalid.")
        if self.lifecycle_cause == "failed" and self.failure is None:
            raise ValueError("Failed Fit terminal requires a factual failure.")
        if self.lifecycle_cause == "scientific_unavailable":
            if self.candidate is not None or self.failure is not None:
                raise ValueError("Scientifically unavailable Fit terminal cannot publish a candidate or failure.")
            if self.outcome is None or self.outcome.status != "scientific_unavailable":
                raise ValueError("Scientifically unavailable Fit terminal requires its typed outcome.")
        elif self.outcome is not None and self.outcome.status == "scientific_unavailable":
            raise ValueError("Scientific-unavailability outcome requires its dedicated terminal lifecycle.")
        if self.lifecycle_cause in {"completed", "user_stop"} and self.candidate is None:
            raise ValueError(
                f"Fit terminal {self.lifecycle_cause!r} requires a candidate."
            )
        if self.lifecycle_cause in {"completed", "user_stop"} and self.failure is not None:
            raise ValueError(
                f"Fit terminal {self.lifecycle_cause!r} cannot carry a failure."
            )

    @property
    def message(self) -> str:
        """Compact status text; structured facts remain in ``failure``."""

        if self.failure is not None:
            return self.failure.message
        if self.lifecycle_cause == "scientific_unavailable":
            cause = None if self.outcome is None else self.outcome.to_factual_cause()
            return (
                "Fit result scientifically unavailable"
                if cause is None
                else f"Fit result scientifically unavailable: {cause.message}"
            )
        return (
            "Fit stopped by user."
            if self.lifecycle_cause == "user_stop"
            else "Fit cancelled."
            if self.lifecycle_cause == "cancelled"
            else ""
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


def _parametric_x_observation(
    mapped: FitMappedDatasetSnapshot,
    local: FitDatasetLocalSettings,
) -> FitObservationSeries | None:
    parametric = local.parametric
    if parametric is None:
        return None
    x_raw_id = next(
        (raw_id for raw_id, target in local.observation_targets if target == parametric.x_name),
        None,
    )
    if x_raw_id is None:
        raise ValueError(
            f"Dataset {mapped.display_name!r} requires exactly one observed "
            f"source for X={parametric.x_name!r}."
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
    return FitObservationSeries(
        raw_observation_id=source.raw_observation_id,
        raw_column=source.raw_column,
        t=np.asarray(source.t, dtype=float),
        y=np.asarray(source.y, dtype=float),
        source_rows=source.source_rows,
    )


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
        item.model_target: FitObservationSeries(
            raw_observation_id=item.raw_observation_id,
            raw_column=item.raw_column,
            t=np.asarray(item.t, dtype=float),
            y=np.asarray(item.y, dtype=float),
            source_rows=item.source_rows,
        )
        for item in selected
    }
    x_observations = _parametric_x_observation(mapped, local)
    if x_observations is not None:
        x_rows = set(x_observations.source_rows)
        if any(
            not set(item.source_rows).issubset(x_rows)
            for item in observation_map.values()
        ):
            raise ValueError(
                f"Dataset {mapped.display_name!r}: Parametric-X capture is missing a selected Y source row."
            )
    return FitDatasetSpec(
        dataset_id=mapped.dataset_id,
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
    scopes = dict(setup.initial_scopes)
    fitted_by_key: dict[tuple[str, str], object] = {
        (initial.set_id, initial.species): initial
        for initial in setup.initials
        if initial.fit and scopes.get(initial.species) == "local"
    }
    included_ids = {
        item.dataset_id for item in setup.datasets if item.included
    }
    datasets_by_set: dict[str, list[str]] = {}
    for mapped in origin.mapped_datasets:
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
            coupled_values.add(
                validate_concentration(
                    value,
                    field=f"fitted initial {(set_id, species)!r}",
                )
            )
        if len(coupled_values) != 1:
            raise ValueError(
                f"Fit result disagrees across coupled datasets for initial "
                f"{species!r} in set {set_id!r}."
            )
        value = coupled_values.pop()
        values_by_set.setdefault(set_id, {})[species] = value
    for global_initial in setup.global_initials:
        if not global_initial.fit or scopes.get(global_initial.species) != "global":
            continue
        parameter_name = f"{INITIAL_PREFIX}{global_initial.species}"
        values: set[float] = set()
        applicable_ids = sorted(
            mapped.dataset_id
            for mapped in origin.mapped_datasets
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
            value = validate_concentration(
                value,
                field=f"fitted Global initial {global_initial.species!r}",
            )
            values.add(value)
            set_ids = {
                mapped.set_id
                for mapped in origin.mapped_datasets
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
    refit_execution = dispatch.shared_refit_execution
    promotions = {
        item.shared_variable.name: item
        for item in (
            ()
            if refit_execution is None
            else refit_execution.promotion_admissions
        )
    }
    if not set(promotions).issubset(request_parameters):
        raise ValueError(
            "Fit terminal dispatch references an unknown promoted parameter."
        )
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
        admission = promotions.get(parameter.name)
        if admission is not None:
            if requested.scope != "per_dataset":
                raise ValueError(
                    "Shared-refit promotion source must remain Per-dataset."
                )
            if parameter.name not in shared_results:
                raise ValueError(
                    f"Shared refit omitted promoted parameter {parameter.name!r}."
                )
            value = shared_results[parameter.name]
            if not math.isfinite(value):
                raise ValueError(
                    f"Shared refit contains non-finite promoted parameter "
                    f"{parameter.name!r}."
                )
            parameters.append(
                _replace_fit_parameter(
                    parameter,
                    scope="shared",
                    initial=value,
                    minimum=admission.shared_variable.minimum,
                    maximum=admission.shared_variable.maximum,
                    fit=True,
                    log10=admission.shared_variable.log10,
                    dataset_values=(),
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
    scopes = dict(setup_snapshot.initial_scopes)
    global_initial_results: dict[str, float] = {}
    for global_initial in setup_snapshot.global_initials:
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
        shared_refit_promotions=(),
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
        x_observations = _parametric_x_observation(mapped, local)
        observed_x, observed_y = _observed_curve_parts(
            selected,
            x_observations=x_observations,
        )
        x_name = local.parametric.x_name if local.parametric is not None else "t"
        model_names = tuple(item.model_target for item in selected)
        objective_model_series = result.model_series[dataset_id]
        model_x_values = (
            simulation.t
            if x_name == "t"
            else fit_result_target_values(simulation, x_name)
        )
        curves.append(
            FitCurveDataset(
                dataset_id=dataset_id,
                display_name=mapped.display_name,
                y_semantics=_fit_curve_y_semantics(
                    setup_snapshot.prepared_dsl,
                    model_names,
                ),
                model_x=tuple(
                    (
                        name,
                        tuple(float(value) for value in model_x_values),
                    )
                    for name in model_names
                ),
                model_y=tuple(
                    (
                        name,
                        tuple(
                            float(value)
                            for value in fit_result_target_values(simulation, name)
                        ),
                    )
                    for name in model_names
                ),
                observed_x=observed_x,
                observed_y=observed_y,
                model_segment_ids=tuple(
                    (
                        name,
                        tuple(
                            int(value) for value in simulation.segment_ids
                        ),
                    )
                    for name in model_names
                ),
                model_segment_keys=tuple(
                    (
                        name,
                        tuple(simulation.segment_keys),
                    )
                    for name in model_names
                ),
                model_discontinuity_ids=tuple(
                    (
                        name,
                        tuple(
                            int(value) for value in simulation.discontinuity_ids
                        ),
                    )
                    for name in model_names
                ),
                observed_source_rows=tuple(
                    (item.model_target, item.source_rows) for item in selected
                ),
                observed_raw_ids=tuple(
                    (item.model_target, item.raw_observation_id)
                    for item in selected
                ),
                observed_raw_columns=tuple(
                    (item.model_target, item.raw_column) for item in selected
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
        x_observations = _parametric_x_observation(mapped, local)
        observed_x, observed_y = _observed_curve_parts(
            selected,
            x_observations=x_observations,
        )
        x_name = (
            local.parametric.x_name
            if local.parametric is not None
            else "t"
        )
        sim_time = np.asarray(evaluation.t, dtype=float)
        projection = None
        if x_name != "t":
            if x_observations is None or local.parametric is None:
                raise FitSimulationError(
                    f"Dataset {dataset_id!r}: Parametric-X capture is missing."
                )
            retained_rows = {
                int(row)
                for item in selected
                for row in item.source_rows
            }
            x_rows = tuple(
                int(row)
                for row in x_observations.source_rows
                if int(row) in retained_rows
            )
            x_by_row = {
                int(row): (float(time), float(value))
                for row, time, value in zip(
                    x_observations.source_rows,
                    x_observations.t,
                    x_observations.y,
                    strict=True,
                )
            }
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
                model_x=fit_result_target_values(evaluation, x_name),
                segment_ids=np.asarray(evaluation.segment_ids, dtype=np.int64),
                segment_keys=evaluation.segment_keys,
                discontinuity_ids=np.asarray(
                    evaluation.discontinuity_ids,
                    dtype=np.int64,
                ),
                mode=local.parametric.resolved_mode,
                dataset_label=dataset_id,
                x_name=x_name,
            )

        projected_rows: dict[str, tuple[object, ...]] = {}
        for item in selected:
            model_y = np.asarray(
                fit_result_target_values(evaluation, item.model_target),
                dtype=float,
            )
            if x_name == "t":
                rows = _select_fit_rows(
                    model_y=model_y,
                    sim_time=sim_time,
                    observation_time=np.asarray(item.t, dtype=float),
                    observation_id=item.raw_observation_id,
                    raw_column=item.raw_column,
                    source_rows=item.source_rows,
                    segment_ids=np.asarray(evaluation.segment_ids, dtype=np.int64),
                    segment_keys=evaluation.segment_keys,
                    discontinuity_ids=np.asarray(
                        evaluation.discontinuity_ids,
                        dtype=np.int64,
                    ),
                )
            else:
                assert projection is not None
                y_projection = projection.subset(item.source_rows)
                rows = (
                    y_projection.selected_model_x,
                    project_parametric_y(
                        projection=y_projection,
                        model_time=sim_time,
                        model_y=model_y,
                        segment_ids=np.asarray(evaluation.segment_ids, dtype=np.int64),
                        segment_keys=evaluation.segment_keys,
                        discontinuity_ids=np.asarray(
                            evaluation.discontinuity_ids,
                            dtype=np.int64,
                        ),
                    ),
                    y_projection.segment_ids,
                    y_projection.segment_keys,
                    y_projection.discontinuity_ids,
                    y_projection.sample_kinds,
                    tuple(item.raw_observation_id for _ in item.source_rows),
                    tuple(item.raw_column for _ in item.source_rows),
                    tuple(int(row) for row in item.source_rows),
                )
            projected_rows[item.model_target] = rows
        curves.append(
            FitCurveDataset(
                dataset_id=dataset_id,
                display_name=mapped.display_name,
                y_semantics=_fit_curve_y_semantics(
                    request.prepared_dsl,
                    tuple(item.model_target for item in selected),
                ),
                model_x=tuple(
                    (
                        item.model_target,
                        tuple(
                            float(value)
                            for value in projected_rows[item.model_target][0]
                        ),
                    )
                    for item in selected
                ),
                model_y=tuple(
                    (
                        item.model_target,
                        tuple(
                            float(value)
                            for value in projected_rows[item.model_target][1]
                        ),
                    )
                    for item in selected
                ),
                observed_x=observed_x,
                observed_y=observed_y,
                model_segment_ids=tuple(
                    (
                        item.model_target,
                        tuple(
                            int(value)
                            for value in projected_rows[item.model_target][2]
                        ),
                    )
                    for item in selected
                ),
                model_segment_keys=tuple(
                    (
                        item.model_target,
                        tuple(projected_rows[item.model_target][3]),
                    )
                    for item in selected
                ),
                model_discontinuity_ids=tuple(
                    (
                        item.model_target,
                        tuple(
                            int(value)
                            for value in projected_rows[item.model_target][4]
                        ),
                    )
                    for item in selected
                ),
                observed_source_rows=tuple(
                    (item.model_target, item.source_rows)
                    for item in selected
                ),
                observed_raw_ids=tuple(
                    (
                        item.model_target,
                        item.raw_observation_id,
                    )
                    for item in selected
                ),
                observed_raw_columns=tuple(
                    (item.model_target, item.raw_column)
                    for item in selected
                ),
                objective_model_x=tuple(
                    (
                        item.model_target,
                        tuple(
                            float(value)
                            for value in projected_rows[item.model_target][0]
                        ),
                    )
                    for item in selected
                ),
                objective_model_y=tuple(
                    (
                        item.model_target,
                        tuple(
                            float(value)
                            for value in projected_rows[item.model_target][1]
                        ),
                    )
                    for item in selected
                ),
                objective_model_segment_ids=tuple(
                    (
                        item.model_target,
                        tuple(
                            int(value)
                            for value in projected_rows[item.model_target][2]
                        ),
                    )
                    for item in selected
                ),
                objective_model_segment_keys=tuple(
                    (
                        item.model_target,
                        tuple(projected_rows[item.model_target][3]),
                    )
                    for item in selected
                ),
                objective_model_discontinuity_ids=tuple(
                    (
                        item.model_target,
                        tuple(
                            int(value)
                            for value in projected_rows[item.model_target][4]
                        ),
                    )
                    for item in selected
                ),
                objective_selection_provenance=tuple(
                    (
                        item.model_target,
                        tuple(projected_rows[item.model_target][5]),
                    )
                    for item in selected
                ),
                objective_observation_ids=tuple(
                    (
                        item.model_target,
                        tuple(projected_rows[item.model_target][6]),
                    )
                    for item in selected
                ),
                objective_raw_columns=tuple(
                    (
                        item.model_target,
                        tuple(projected_rows[item.model_target][7]),
                    )
                    for item in selected
                ),
                objective_source_rows=tuple(
                    (
                        item.model_target,
                        tuple(projected_rows[item.model_target][8]),
                    )
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

    if isinstance(value, FactualCause):
        return _compact_metadata(value.to_payload())
    if isinstance(value, FitCoordinate):
        return _fit_coordinate_machine_identity(value)
    if isinstance(value, CentredFixedEffectsSettlement):
        return {
            "parameter_name": value.parameter_name,
            "dataset_ids": value.dataset_ids,
            "reference_dataset_id": value.reference_dataset_id,
            "overall_coordinate": _compact_metadata(value.overall_coordinate),
            "overall_optimizer_value": value.overall_optimizer_value,
            "overall_decoded_value": value.overall_decoded_value,
            "deviations": [
                {
                    "coordinate": _compact_metadata(item.coordinate),
                    "optimizer_value": item.optimizer_value,
                    "decoded_value": item.decoded_value,
                }
                for item in value.deviations
            ],
            "dataset_values": _compact_metadata(value.dataset_values),
        }
    if isinstance(value, np.ndarray):
        return _array_identity(value)
    if isinstance(value, np.generic):
        return _compact_metadata(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Fit terminal metadata cannot transport a non-finite number."
            )
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, raw_item in sorted(value.items(), key=lambda entry: str(entry[0])):
            if isinstance(raw_key, FitCoordinate):
                key = raw_key.to_machine_key()
            else:
                key = str(raw_key)
            normalized[key] = _compact_metadata(raw_item)
        return normalized
    if isinstance(value, (tuple, list)):
        return [_compact_metadata(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_compact_metadata(item) for item in sorted(value, key=repr)]
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


def _primary_outcome_identity(outcome: object) -> object:
    """Serialize the complete typed primary outcome for run identity."""

    if outcome is None:
        return None
    cause = getattr(outcome, "cause", None)
    if cause is None:
        return None
    return {
        "classification": outcome.classification,
        "publication": outcome.publication,
        "apply": outcome.apply,
        "details": _compact_metadata(outcome.details),
        "cause": _compact_metadata(cause.to_payload()),
    }


def _terminal_extensions_identity(extensions: Sequence[object]) -> list[object]:
    """Serialize optional outcomes, including every typed factual cause."""

    records: list[dict[str, object]] = []
    for item in extensions:
        record: dict[str, object] = {
            "kind": item.kind,
            "status": item.status,
            "method": item.method,
            "provenance": _compact_metadata(item.provenance),
            "details": _compact_metadata(item.details),
            "unavailable_reason": item.unavailable_reason,
        }
        if item.cause is not None:
            record["cause"] = _compact_metadata(item.cause.to_payload())
        records.append(record)
    return records


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
    info_by_dataset = {
        str(info.dataset_id): info for info in result.dataset_info
    }
    for dataset_id in sorted(result.model_series):
        info = info_by_dataset.get(dataset_id)
        row: dict[str, object] = {
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
        if info is not None and info.x_identity_hash is not None:
            row["x_terminal"] = {
                "identity_hash": info.x_identity_hash,
                "point_count": int(info.x_point_count),
                "delta_x": _array_identity(info.x_delta_x),
                "match_kinds": tuple(info.x_match_kinds),
                "sample_kinds": tuple(info.x_sample_kinds),
                "weighted_sse": (
                    None if info.x_weighted_residual_sse is None
                    else info.x_weighted_residual_sse.to_payload()
                ),
                "weighted_mse": (
                    None if info.x_weighted_residual_mse is None
                    else info.x_weighted_residual_mse.to_payload()
                ),
                "weighted_rmse": (
                    None if info.x_weighted_rmse is None
                    else info.x_weighted_rmse.to_payload()
                ),
                "weighted_mae": (
                    None if info.x_weighted_mae is None
                    else info.x_weighted_mae.to_payload()
                ),
            }
        projection = result.model_x_projection_by_dataset.get(dataset_id)
        if projection is not None:
            row["x_projection"] = {
                "source_rows": projection.source_rows,
                "observation_time": _array_identity(projection.observation_time),
                "observed_x": _array_identity(projection.observed_x),
                "selected_time": _array_identity(projection.selected_time),
                "selected_model_x": _array_identity(projection.selected_model_x),
                "delta_x": _array_identity(projection.delta_x),
                "segment_ids": _array_identity(projection.segment_ids),
                "segment_keys": projection.segment_keys,
                "discontinuity_ids": _array_identity(projection.discontinuity_ids),
                "match_kinds": projection.match_kinds,
                "sample_kinds": projection.sample_kinds,
                "model_rows": projection.model_rows,
            }
        rows.append(row)
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

    included_ids = {item.dataset_id for item in included}
    included_set_ids = {
        mapped_by_id[dataset_id].set_id
        for dataset_id in included_ids
    }
    included_initial_species = {
        species
        for dataset_id in included_ids
        for species, _value in mapped_by_id[dataset_id].initial_values
    }
    fit_local_admission: AdmittedAlgebraProgram | None = request.fit_local_admission
    if fit_local_admission is not None and not isinstance(
        fit_local_admission,
        AdmittedAlgebraProgram,
    ):
        raise TypeError("Fit request local admission must be a typed value.")
    if (
        {item.dataset_id for item in datasets} != included_ids
        or set(effective_local) != included_ids
        or set(prepared_ids) != included_ids
        or set(dict(resolved_modes))
        - included_ids
    ):
        raise ValueError(
            "Fit run identity inputs must exactly match the included datasets."
        )
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
            }
            for name, series in item.observations.items()
        }
        x_identity = None
        if item.x_name != "t":
            x_source = item.x_observations
            if x_source is None:
                raise ValueError("Parametric Fit identity omitted its canonical X capture.")
            x_identity = {
                "raw_observation_id": x_source.raw_observation_id,
                "raw_column": x_source.raw_column,
                "t": _array_identity(x_source.t),
                "x": _array_identity(x_source.y),
                "source_rows": _array_identity(x_source.source_rows),
                "retained_source_rows": tuple(item.x_source_rows),
                "retained_point_count": len(item.x_source_rows),
                "configured_target_weight": float(item.target_weights[item.x_name]),
                "projection_rule": item.x_mode,
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
                "x": x_identity,
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
    identity_material = {
        "kind": "global_fit_run",
        "schema_version": 2,
        "mechanism_text": request.mechanism_source.reactions_text,
        "fit_solver": request.solver,
        "dataset_weight_mode": request.dataset_weight_mode,
        "fit_local_admission": None
        if fit_local_admission is None
        else {
            "identity": fit_local_admission.identity,
            "base_analysis_identity": fit_local_admission.base_analysis_identity,
            "base_executable_fingerprint": fit_local_admission.base_executable_fingerprint,
            "source_text": fit_local_admission.source_text,
            "dependency_plan": fit_local_admission.dependency_plan,
            "observables": [
            {
                "name": observable.name,
                "expression": observable.expression,
                "scalar_names": observable.scalar_names,
            }
            for observable in fit_local_admission.observables
            ],
            "scalar_names": fit_local_admission.scalar_names,
        },
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
                "dataset_values": {
                    dataset_id: value
                    for dataset_id, value in item.selected_config.dataset_values
                    if dataset_id in included_ids
                },
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
            if item.set_id in included_set_ids
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
            if item.species in included_initial_species
        ],
        "settings": {
            "method": request.defaults.method,
            "max_nfev": request.defaults.max_nfev,
            "ftol": request.defaults.ftol,
            "xtol": request.defaults.xtol,
            "use_seed": request.defaults.use_seed,
            "seed": request.defaults.seed if request.defaults.use_seed else None,
        },
        "advanced_analysis": {
            "profile_likelihood": (
                {
                    "parameter_names": tuple(
                        _fit_coordinate_machine_identity(item)
                        for item in request.profile_likelihood.parameter_names
                    ),
                    "confidence_level": request.profile_likelihood.confidence_level,
                    "max_bracketing_steps_per_side": (
                        request.profile_likelihood.max_bracketing_steps_per_side
                    ),
                    "max_nfev_per_point": request.profile_likelihood.max_nfev_per_point,
                    "optimizer_method": request.profile_likelihood.optimizer_method,
                    "likelihood_ratio_threshold": request.profile_likelihood.likelihood_ratio_threshold,
                }
                if request.profile_likelihood is not None
                else None
            ),
            "residual_bootstrap": (
                {
                    "parameter_names": tuple(
                        _fit_coordinate_machine_identity(item)
                        for item in request.residual_bootstrap.parameter_names
                    ),
                    "seed": request.residual_bootstrap.seed,
                    "replicates": request.residual_bootstrap.replicates,
                    "confidence_level": request.residual_bootstrap.confidence_level,
                    "minimum_successful_replicates": request.residual_bootstrap.minimum_successful_replicates,
                    "minimum_success_fraction": request.residual_bootstrap.minimum_success_fraction,
                    "optimizer_method": request.residual_bootstrap.optimizer_method,
                }
                if request.residual_bootstrap is not None
                else None
            ),
            "centred_fixed_effects": [
                {
                    "parameter_name": item.parameter_name,
                    "dataset_ids": tuple(item.dataset_ids),
                    "initial_values": tuple(item.initial_values),
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                    "log10": item.log10,
                }
                for item in request.dispatch.ordinary_centred_variables
            ],
            "shared_refit_execution": (
                None
                if request.dispatch.shared_refit_execution is None
                else {
                    "dataset_ids": list(
                        request.dispatch.shared_refit_execution.dataset_ids
                    ),
                    "promotions": [
                        {
                            "parameter_name": item.shared_variable.name,
                            "initial": item.shared_variable.initial,
                            "minimum": item.shared_variable.minimum,
                            "maximum": item.shared_variable.maximum,
                            "log10": item.shared_variable.log10,
                            "provenance": _compact_metadata(item.provenance),
                        }
                        for item in (
                            request.dispatch.shared_refit_execution.promotion_admissions
                        )
                    ],
                }
            ),
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
            if dataset_id in included_ids
        },
        "resolved_auto_modes": dict(resolved_modes),
    }
    return {
        "kind": "global_fit_run_identity",
        "schema_version": 3,
        "request": {
            "sha256": _metadata_digest(identity_material),
        },
        "prepared": {
            dataset_id: {"sha256": _metadata_digest(metadata)}
            for dataset_id, metadata in sorted(prepared_ids.items())
            if dataset_id in included_ids
        },
        "resolved_auto_modes": {
            "sha256": _metadata_digest(dict(resolved_modes))
        },
    }


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
) -> FitCandidate:
    """Validate provenance and build candidate/curves without worker custody."""

    admitted_layout = (
        request.dispatch.ordinary_optimizer_layout
        if request.dispatch.shared_refit_execution is None
        else request.dispatch.shared_refit_execution.optimizer_layout
    )
    if not isinstance(admitted_layout, FitParameterLayout):
        raise ValueError("Fit terminal omitted its admitted optimizer layout.")
    if not isinstance(result.optimizer_vector, np.ndarray):
        raise TypeError("Fit terminal optimizer vector must be a NumPy array.")
    admitted_layout.validate_optimizer_vector(result.optimizer_vector)
    if result.optimizer_vector.flags.writeable:
        raise ValueError("Fit terminal optimizer vector must be read-only custody.")

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
    refit_execution = request.dispatch.shared_refit_execution
    promoted_names = {
        item.shared_variable.name
        for item in (
            ()
            if refit_execution is None
            else refit_execution.promotion_admissions
        )
    }
    fixed_by_dataset = {
        dataset_id: dict(values)
        for dataset_id, values in request.dispatch.fixed_by_dataset
    }
    for parameter in request.parameters:
        if parameter.scope == "shared" or parameter.name in promoted_names:
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
    if set(evaluators) != included_id_set or set(prepared_ids) != included_id_set:
        raise ValueError(
            "Fit terminal prepared identities do not exactly match the included datasets."
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
    initial_values_by_set = _initial_values_from_result(
        setup_snapshot,
        request.origin,
        result,
    )
    _terminal_setup_snapshot(
        setup_snapshot,
        origin=request.origin,
        result=result,
        dispatch=request.dispatch,
        effective_datasets=effective_local,
        initial_values_by_set=initial_values_by_set,
    )
    terminal_material = {
        "fit_solver": request.solver,
        "status": result.completion.status,
        "optimizer_converged": bool(result.completion.optimizer_converged),
        "message": str(result.message),
        "scipy_nfev": int(result.scipy_nfev),
        "terminal_replay_count": int(result.terminal_replay_count),
        "objective_call_count": (
            None
            if result.objective_call_count is None
            else int(result.objective_call_count)
        ),
        "weighted_residual_sse": result.weighted_residual_sse.to_payload(),
        "weighted_residual_mse": result.weighted_residual_mse.to_payload(),
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
            if dataset_id in included_id_set
        },
        "primary_model": str(getattr(result.completion, "primary_model", "ordinary")),
        "primary_model_details": _compact_metadata(
            result.completion.primary_model_details
        ),
        "shared_refit_provenance": _compact_metadata(
            result.shared_refit_provenance
        ),
        "terminal_extensions": _terminal_extensions_identity(
            result.completion.terminal_extensions
        ),
        "result_array_identity": _result_array_identity(result),
        "simulation_result_array_identity": (
            _simulation_result_array_identity(result)
        ),
    }
    primary_outcome_identity = _primary_outcome_identity(
        result.completion.primary_outcome
    )
    if primary_outcome_identity is not None:
        terminal_material["primary_outcome"] = primary_outcome_identity
    terminal_material["centred_fixed_effects_settlements"] = _compact_metadata(
        result.centred_fixed_effects_settlements
    )
    run_identity_payload["result"] = {
        "sha256": _metadata_digest(terminal_material)
    }
    if result.centred_fixed_effects_settlements:
        # Centred identity is self-describing at the terminal boundary: the
        # compact digest remains the ordinary result identity while the typed
        # settlements remain available to machine consumers.
        run_identity_payload["result"].update(
            {
                "centred_fixed_effects_settlements": _compact_metadata(
                    result.centred_fixed_effects_settlements
                )
            }
        )
    run_identity_payload["canonical_replay"] = {
        dataset_id: {
            "sha256": _metadata_digest(
                solver_provenance_by_dataset[dataset_id]
            )
        }
        for dataset_id in sorted(solver_provenance_by_dataset)
    }
    run_identity = FitRunIdentity.build(run_identity_payload)
    candidate = FitCandidate(
        request=request,
        result=result,
        run_identity=run_identity,
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
) -> tuple[GlobalFitResult | None, FitEvaluationOutcome]:
    """Dispatch one exact normal, advanced, or promotion Fit execution."""

    common = {
        "fit_evaluators": evaluators,
        "datasets": datasets,
        "weights": weights,
        "method": request.defaults.method,
        "max_nfev": int(request.defaults.max_nfev),
        "ftol": float(request.defaults.ftol),
        "xtol": float(request.defaults.xtol),
        "seed": (
            int(request.defaults.seed)
            if request.defaults.use_seed
            else None
        ),
        "progress_callback": progress_callback,
        "extension_progress_callback": extension_progress_callback,
        "cancellation_check": cancellation_check,
        "cancellation_control": cancellation_control,
        "profile_likelihood": request.profile_likelihood,
        "residual_bootstrap": request.residual_bootstrap,
    }
    dispatch = request.dispatch
    refit_execution = dispatch.shared_refit_execution
    if refit_execution is None:
        layout = dispatch.ordinary_optimizer_layout
        if not isinstance(layout, FitParameterLayout):
            raise RuntimeError("Ordinary Fit omitted its admitted optimizer layout.")
        return fit_global(
            shared_variables=dispatch.ordinary_shared_variables,
            dataset_overrides=dispatch.ordinary_dataset_overrides,
            coupled_variables=dispatch.ordinary_coupled_variables,
            centred_variables=dispatch.ordinary_centred_variables,
            optimizer_layout=layout,
            **common,
        )
    return fit_global_shared_refit(
        execution=refit_execution,
        **common,
    )


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
        self._candidate_custody: FitCandidate | None = None
        self._resolved_auto_modes: tuple[tuple[str, str], ...] = ()
        self._detailed_balance_condition = Condition()
        self._terminal_result: FitWorkerTerminal | None = None
        self._terminal_outcome: FitEvaluationOutcome | None = None
        self._publish_best_on_cancel = False
        self._publication_fenced = False

    def pause(self) -> None:
        with self._pause_condition:
            self._paused = True

    @property
    def resolved_auto_modes(self) -> tuple[tuple[str, str], ...]:
        return self._resolved_auto_modes

    @property
    def publication_fenced(self) -> bool:
        return self._publication_fenced

    @property
    def terminal_result(self) -> FitWorkerTerminal | None:
        """Return the retained terminal after the worker's ``run`` exits."""

        return self._terminal_result

    def resume(self) -> None:
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()

    def request_cancel(
        self,
        *,
        publish_best: bool = False,
    ) -> None:
        """Request cancellation without waiting for worker settlement."""

        self._publish_best_on_cancel = bool(publish_best)
        self._publication_fenced = not self._publish_best_on_cancel
        self._cancelled.set()
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()
        with self._detailed_balance_condition:
            self._detailed_balance_condition.notify_all()
        self.requestInterruption()

    def _build_terminal(
        self,
        *,
        candidate: FitCandidate | None,
        failure: FactualCause | None = None,
        outcome: FitEvaluationOutcome | None = None,
    ) -> FitWorkerTerminal:
        """Settle only factual candidate, failure, and lifecycle custody."""

        if outcome is not None and outcome.status == "scientific_unavailable":
            lifecycle = "scientific_unavailable"
            failure = None
            candidate = None
        elif failure is not None:
            lifecycle = "failed"
        elif self._cancelled.is_set():
            lifecycle = (
                "user_stop"
                if self._publish_best_on_cancel and candidate is not None
                else "cancelled"
            )
        elif candidate is not None:
            lifecycle = "completed"
        else:
            lifecycle = "failed"
            failure = factual_cause_from_exception(
                RuntimeError("Fit worker exited without a candidate or failure."),
                domain="fit",
                origin="fit_worker",
            )
        return FitWorkerTerminal(
            candidate=candidate,
            failure=failure,
            outcome=outcome,
            lifecycle_cause=lifecycle,
        )

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
                if self._checkpoint():
                    raise FittingCancelled()
                self.detailed_balance_correction_required.emit(
                    correction.request
                )
                if self._checkpoint():
                    raise FittingCancelled()
                with self._detailed_balance_condition:
                    while not self._cancelled.is_set():
                        self._detailed_balance_condition.wait()
                raise FittingCancelled()
        except FittingCancelled:
            terminal = self._build_terminal(candidate=self._candidate_custody)
        except BaseException as exc:  # noqa: BLE001
            terminal = self._build_terminal(
                candidate=self._candidate_custody,
                failure=factual_cause_from_exception(
                    exc,
                    domain="fit",
                    origin="fit_worker",
                ),
            )
        else:
            if (
                self._checkpoint()
                and not self._publish_best_on_cancel
                and self._terminal_outcome is None
            ):
                terminal = self._build_terminal(candidate=self._candidate_custody)
            else:
                terminal = self._build_terminal(
                    candidate=candidate,
                    outcome=self._terminal_outcome,
                )
        self._terminal_result = terminal

    def _execute(
        self,
        *,
        allow_detailed_balance_correction: bool,
    ) -> FitCandidate | None:
        request = self._request
        if self._checkpoint():
            raise FittingCancelled()
        included = tuple(item for item in request.datasets if item.included)
        mapped_by_id = {item.dataset_id: item for item in request.origin.mapped_datasets}
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
        evaluators: dict[str, SerialFittingEvaluator] = {}
        prepared_ids: dict[str, dict[str, object]] = {}
        for local in included:
            if self._checkpoint():
                raise FittingCancelled()
            science = request.origin.scientific_settings
            dataset_context = prepared_datasets[local.dataset_id]
            try:
                context = prepare_fitting_execution_context(
                    prepared_dsl=request.prepared_dsl,
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
                    fit_local_admission=request.fit_local_admission,
                )
            except BaseException as exc:
                unresolved = _unresolved_detailed_balance_error(exc)
                if (
                    allow_detailed_balance_correction
                    and unresolved is not None
                ):
                    issue = str(unresolved)
                    admission_report = unresolved.report
                    if not isinstance(
                        admission_report,
                        WegscheiderCyclicityReport,
                    ):
                        raise RuntimeError(
                            "Fit Wegscheider correction lost its exact "
                            "admission report."
                        ) from exc
                    identity = FitRunIdentity.build(
                        {
                            "kind": "detailed_balance_correction",
                            "schema_version": 1,
                            "origin_capture_id": request.origin.capture_id,
                            "tab_id": request.origin.tab_id,
                            "source_sha256": hashlib.sha256(
                                request.origin.mechanism_source.reactions_text.encode(
                                    "utf-8"
                                )
                            ).hexdigest(),
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
                            report=admission_report,
                            source_energy_unit=str(
                                request.origin.prepared_dsl.analysis.energy_unit
                            ),
                            accepted_analysis=request.origin.prepared_dsl.analysis,
                        )
                    ) from exc
                if isinstance(exc, FitSimulationError):
                    exc.details.setdefault("fit_solver", request.solver)
                raise
            evaluator = SerialFittingEvaluator(
                context,
            )
            evaluator.set_cancellation_control(self._cancellation_control)
            evaluators[local.dataset_id] = evaluator
            metadata = evaluator.prepared_metadata
            prepared_ids[local.dataset_id] = {
                "prepared_simulation": metadata.to_serializable_dict(),
                "declarative_analysis_identity": (
                    request.prepared_dsl.analysis.identity
                ),
                "fit_local_admission_identity": (
                    None
                    if request.fit_local_admission is None
                    else request.fit_local_admission.identity
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
        starting_projection_failed = False
        refit_execution = dispatch.shared_refit_execution
        if refit_execution is None:
            execution_shared_variables = dispatch.ordinary_shared_variables
            execution_dataset_overrides = dispatch.ordinary_dataset_overrides
            execution_coupled_variables = dispatch.ordinary_coupled_variables
        else:
            execution_shared_variables = refit_execution.shared_variables
            execution_dataset_overrides = refit_execution.dataset_overrides
            execution_coupled_variables = refit_execution.coupled_variables
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
        for local in included:
            chosen = local
            dataset_id = local.dataset_id
            mapped = mapped_by_id[dataset_id]
            starting_values = {
                parameter.name: (
                    dict(parameter.selected_config.dataset_values)[dataset_id]
                    if parameter.scope in {"per_dataset", "centred_fixed_effects"}
                    else parameter.selected_config.value
                )
                for parameter in request.parameters
            }
            starting_values.update(
                {
                    item.name: item.initial
                    for item in execution_shared_variables
                }
            )
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
            try:
                evaluation_outcome = evaluators[
                    dataset_id
                ].evaluate_series_outcome_with_parameter_origins(
                    starting_values,
                    starting_origins,
                )
            except (FittingCancelled,):
                raise
            if not isinstance(evaluation_outcome, FittingSeriesOutcome):
                raise TypeError("Fit evaluator returned an invalid starting outcome.")
            if evaluation_outcome.status == "scientific_unavailable":
                evaluation = None
                starting_projection_failed = True
            elif evaluation_outcome.status == "cancelled":
                raise FittingCancelled()
            elif evaluation_outcome.status == "fatal":
                if evaluation_outcome.cause is None:
                    raise RuntimeError("Starting Fit evaluation omitted its fatal cause.")
                raise evaluation_outcome.cause
            else:
                evaluation = evaluation_outcome.result
            if evaluation is not None:
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
                if evaluation is None:
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
                        x_source = provisional.x_observations
                        if x_source is None:
                            raise ValueError("Parametric-X capture is missing.")
                        x_rows = provisional.x_source_rows
                        x_by_row = dict(
                            zip(x_source.source_rows, x_source.t, strict=True)
                        )
                        align_parametric_x(
                            observation_time=np.asarray(
                                [x_by_row[row] for row in x_rows],
                                dtype=float,
                            ),
                            observed_x=provisional.x_values_for_rows(x_rows),
                            source_rows=x_rows,
                            model_time=np.asarray(series.t, dtype=float),
                            model_x=fit_result_target_values(series, provisional.x_name),
                            segment_ids=np.asarray(series.segment_ids, dtype=np.int64),
                            segment_keys=series.segment_keys,
                            discontinuity_ids=np.asarray(
                                series.discontinuity_ids,
                                dtype=np.int64,
                            ),
                            mode="monotone",
                            dataset_label=dataset_id,
                            x_name=provisional.x_name,
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
            initial_curves = (
                ()
                if starting_projection_failed
                else _curves_from_evaluations(
                    request,
                    starting_evaluations,
                    effective_local,
                )
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
            if not isinstance(update, FitProgress):
                raise TypeError("Fit progress callback requires FitProgress.")
            if update.improved:
                # Best-candidate custody is atomic with the objective's
                # accepted improvement and precedes every Stop checkpoint.
                self._best_progress = update.compact()
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
                            / max(1, request.defaults.max_nfev)
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
                    self.best_updated.emit(self._live_update())
                    if self._checkpoint():
                        raise FittingCancelled()
                    self._last_best_emit = now
                if now - self._last_curve_emit >= _CURVE_INTERVAL_SECONDS:
                    try:
                        live_curves = (
                            _curves_from_evaluations(
                                request,
                                update.transient_simulation_results_by_dataset,
                                effective_local,
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
            if request.dataset_weight_mode == "user"
            else None
        )
        execution_stop_observed = False
        extension_progress_value = 95

        def execution_cancellation_check() -> bool:
            nonlocal execution_stop_observed
            requested = self._checkpoint()
            execution_stop_observed = execution_stop_observed or requested
            return requested

        def execution_cancellation_poll() -> bool:
            nonlocal execution_stop_observed
            requested = self._cancelled.is_set()
            execution_stop_observed = execution_stop_observed or requested
            return requested

        def extension_progress(update: FitExtensionProgress) -> None:
            nonlocal extension_progress_value
            if not isinstance(update, FitExtensionProgress):
                raise TypeError("Fit extension progress callback requires FitExtensionProgress.")
            if self._checkpoint():
                raise FittingCancelled()
            has_profile = request.profile_likelihood is not None
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
                        f"{request.profile_likelihood.max_bracketing_steps_per_side}"
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
        stopped_result = False
        try:
            result, terminal_outcome = _execute_global_fit_model(
                request=request,
                evaluators=evaluators,
                datasets=datasets,
                weights=weights,
                progress_callback=progress,
                extension_progress_callback=extension_progress,
                cancellation_check=execution_cancellation_check,
                cancellation_control=execution_cancellation_control,
            )
            if result is None:
                if terminal_outcome.status != "scientific_unavailable":
                    raise RuntimeError(
                        "Fit core omitted a result without scientific-unavailability outcome."
                    )
                self._terminal_outcome = terminal_outcome
                return None
        except FittingCancelled as cancellation:
            if not self._publish_best_on_cancel or self._best_progress is None:
                raise
            result = self._assemble_stopped_best_result(
                datasets=datasets,
                evaluators=evaluators,
                fixed_by_dataset=fixed_by_dataset,
                weights=weights,
                scipy_nfev=getattr(cancellation, "scipy_nfev", None),
                objective_call_count=getattr(
                    cancellation,
                    "objective_call_count",
                    None,
                ),
            )
            stopped_result = True
        if self._checkpoint() and not stopped_result:
            if not self._publish_best_on_cancel:
                raise FittingCancelled()
            if execution_stop_observed:
                # A returned evaluator result is already the complete primary
                # candidate. Never rebuild it through a second replay route.
                stopped_result = True
            else:
                result = self._assemble_stopped_best_result(
                    datasets=datasets,
                    evaluators=evaluators,
                    fixed_by_dataset=fixed_by_dataset,
                    weights=weights,
                    scipy_nfev=result.scipy_nfev,
                    objective_call_count=result.objective_call_count,
                )
                stopped_result = True
        candidate = _assemble_fit_terminal(
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
        # Transfer complete candidate custody before any post-assembly
        # checkpoint or presentation signal can observe the Stop request.
        self._candidate_custody = candidate
        run_identity = candidate.run_identity
        if self._checkpoint() and not stopped_result:
            raise FittingCancelled()
        if not stopped_result:
            self.progress.emit(100, f"Fit complete · {run_identity.short_hash}")
        if self._checkpoint() and not stopped_result:
            raise FittingCancelled()
        return candidate

    def _assemble_stopped_best_result(
        self,
        *,
        datasets: Sequence[FitDatasetSpec],
        evaluators: Mapping[str, SerialFittingEvaluator],
        fixed_by_dataset: Mapping[str, Mapping[str, float]],
        weights: Mapping[str, float] | None,
        scipy_nfev: int | None,
        objective_call_count: int | None,
    ) -> GlobalFitResult:
        """Replay the best complete valid candidate exactly once for user Stop."""

        best = self._best_progress
        if (
            best is None
            or best.completeness is None
            or not best.completeness.is_complete
            or set(best.completeness.expected_dataset_ids)
            != {item.dataset_id for item in datasets}
        ):
            raise FittingCancelled()
        if scipy_nfev is None or objective_call_count is None:
            raise RuntimeError(
                "Stopped Fit lost its exact optimizer/objective counters."
            )
        dispatch = self._request.dispatch
        refit_execution = dispatch.shared_refit_execution
        layout = (
            dispatch.ordinary_optimizer_layout
            if refit_execution is None
            else refit_execution.optimizer_layout
        )
        if not isinstance(layout, FitParameterLayout):
            raise RuntimeError("Stopped Fit omitted its admitted optimizer layout.")
        candidate_vector = np.asarray(best.optimizer_vector, dtype=float).reshape(-1)
        layout.validate_optimizer_vector(candidate_vector)
        replay_control = SimulationCancellationControl(
            lambda: self._cancelled.is_set()
            and not self._publish_best_on_cancel
        )
        for evaluator in evaluators.values():
            evaluator.set_cancellation_control(replay_control)
        replay_state = _assemble_fit_replay_state(
            fit_evaluators=evaluators,
            payloads=list(datasets),
            layout=layout,
            candidate_vector=candidate_vector,
            fixed_dataset_params={
                dataset_id: dict(values)
                for dataset_id, values in fixed_by_dataset.items()
            },
            weights=normalize_weights(list(datasets), None if weights is None else dict(weights)),
            cancellation_check=None,
            cancellation_control=replay_control,
            success=False,
            message="Stopped by user; published the best fully evaluated valid candidate.",
            scipy_nfev=int(scipy_nfev),
            objective_call_count=int(objective_call_count),
        )
        shared_refit_provenance = None
        stopped_extensions: list[FitTerminalExtension] = []
        if self._request.profile_likelihood is not None:
            stopped_extensions.append(
                FitTerminalExtension(
                    kind="profile_likelihood",
                    status="stopped",
                    method="profile likelihood with nuisance reoptimization",
                    provenance={"settlement": "stopped", "primary_fit_preserved": True},
                    details={"primary_fit_preserved": True},
                    cause=factual_cause_from_exception(
                        FittingCancelled(
                            "Profile likelihood was not started after primary Stop."
                        ),
                        domain="fit",
                        origin="profile_likelihood",
                    ),
                )
            )
        if self._request.residual_bootstrap is not None:
            stopped_extensions.append(
                FitTerminalExtension(
                    kind="bootstrap",
                    status="stopped",
                    method="stratified residual bootstrap with full refits",
                    provenance={"settlement": "stopped", "primary_fit_preserved": True},
                    details={"primary_fit_preserved": True},
                    cause=factual_cause_from_exception(
                        FittingCancelled(
                            "Bootstrap was not started after primary Stop."
                        ),
                        domain="fit",
                        origin="bootstrap",
                    ),
                )
            )
        if refit_execution is not None:
            _validate_shared_refit_replay(
                shared_params=replay_state.shared_params,
                dataset_params=replay_state.dataset_params,
                terminal_replay_count=replay_state.terminal_replay_count,
                completion=replay_state.completion,
                execution=refit_execution,
                terminal_route="user_stop_best_valid_bounded_replay",
            )
            shared_refit_provenance = _shared_refit_provenance(
                refit_execution,
                terminal_route="user_stop_best_valid_bounded_replay",
            )
        return _construct_global_fit_result(
            replay_state,
            terminal_extensions=stopped_extensions,
            shared_refit_provenance=shared_refit_provenance,
        )

    def _live_update(self) -> FitLiveUpdate:
        best_progress = self._best_progress
        if best_progress is None:
            raise RuntimeError("Live Fit update requested before a best candidate.")
        dispatch = self._request.dispatch
        layout = (
            dispatch.ordinary_optimizer_layout
            if dispatch.shared_refit_execution is None
            else dispatch.shared_refit_execution.optimizer_layout
        )
        if not isinstance(layout, FitParameterLayout):
            raise RuntimeError("Live Fit update omitted its admitted optimizer layout.")
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
            value = coordinate_values.get(FitCoordinate("coupled", variable_id))
            if value is not None and math.isfinite(float(value)):
                initials_by_set.setdefault(initial.set_id, {})[
                    initial.species
                ] = float(value)
        for initial in self._request.global_initials:
            if not initial.fit or scopes.get(initial.species) != "global":
                continue
            value = coordinate_values.get(
                FitCoordinate("coupled", f"initial:global:{initial.species}")
            )
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
                            for parameter in self._request.parameters
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
