"""
Shared serial fitting evaluation boundary.

This module owns the fitting-side execution seam below the optimizer and above
the solver. It builds a serial evaluator from structured execution-request data
so the GUI/global-fit path no longer depends on the mutable prepared closure as
its architectural contract.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import numpy as np

from kindred.core.exceptions import FitSimulationError, FittingCancelled, SimulationCancelled
from kindred.core.mechanism_metadata import MechanismMetadataKeys
from kindred.core.runtime_defaults import (
    USE_SPARSE_JACOBIAN_DEFAULT,
    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
)
from kindred.core.intervention_schedule import (
    normalized_intervention_schedule_identity_fingerprints,
)
from kindred.core.intervention_schedule_compiler import (
    InterventionExecutionContract,
)
from kindred.core.simulation_preparation import (
    PreparedSimulationMetadata,
    SimulationExecutionRequest,
    _available_energy_binding_names,
    _apply_execution_parameter_algebra_and_cyclicity,
    _bind_requested_parameters_to_prepared_mechanism,
    _bind_symbolic_jacobian_for_current_mechanism,
    _build_solver_config,
    _fit_simulation_error_from_preparation_error,
    _prepare_preparation_failure,
    _raise_unowned_request_parameter_values,
    _prepared_metadata_with_symbolic_jacobian,
    _reject_requested_algebra_owned_mechanism_parameters_for_fitting,
    _scalar_parameter_is_derived,
    _symbolic_jacobian_for_bind_failure,
    build_simulation_request_from_prepared_run,
    canonicalize_request_parameter_names,
    partition_simulation_parameter_values,
    prepare_bound_mechanism,
    prepare_simulation_worker_run,
    resolve_prepared_run_intervention_execution,
    SimulationPreparationError,
)
from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
from kindred.core.result_repository import SimulationResultValue
from kindred.core.simulation_result_finalization import finalize_simulation_result
from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, solve_ode
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution

logger = logging.getLogger(__name__)

__all__ = [
    "FITTING_PARAM_ORIGIN_CONFIGURED_DATASET",
    "FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR",
    "FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET",
    "FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED",
    "PreparedFittingExecutionContext",
    "SerialFittingEvaluator",
    "evaluate_fitting_series",
    "prepare_fitting_execution_context",
]

FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED = "optimizer_shared"
FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET = "optimizer_dataset"
FITTING_PARAM_ORIGIN_CONFIGURED_DATASET = "configured_dataset"
FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR = "configured_evaluator"

_OPTIMIZER_PARAMETER_ORIGINS = frozenset(
    {
        FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
        FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
    }
)


def _parameter_origin_for(name: str, origins: Optional[Mapping[str, str]]) -> str:
    origin = str((origins or {}).get(name, FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR) or "").strip()
    return origin or FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR


def _coerce_consumed_parameter_value(
    *,
    name: str,
    raw_value: object,
    origins: Optional[Mapping[str, str]],
    failed_params: Optional[Dict[str, float]],
) -> float:
    try:
        value = float(raw_value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise FitSimulationError(
            f"Invalid parameter value for {name!r}: {raw_value!r}",
            failed_params=dict(failed_params or {}) or None,
            details={
                "fatal": True,
                "parameter_origin": _parameter_origin_for(name, origins),
            },
        ) from exc
    if np.isfinite(value):
        return value
    origin = _parameter_origin_for(name, origins)
    raise FitSimulationError(
        f"Non-finite parameter value for {name!r}: {raw_value!r}",
        failed_params=dict(failed_params or {}) or None,
        details={
            "fatal": origin not in _OPTIMIZER_PARAMETER_ORIGINS,
            "candidate_domain": origin in _OPTIMIZER_PARAMETER_ORIGINS,
            "parameter_origin": origin,
        },
    )


def _reject_requested_derived_scalars_for_fitting(mechanism: object, names: object) -> None:
    for raw_name in names or ():
        name = str(raw_name or "").strip()
        if not name:
            continue
        if _scalar_parameter_is_derived(mechanism, name):
            raise _prepare_preparation_failure(
                "parameter_binding",
                f"derived scalar parameter {name!r} cannot be used as fitting write authority.",
            )


def _build_fitting_cancellation_event(cancellation_check):
    if cancellation_check is None:
        return None
    cancel_requested = getattr(cancellation_check, "_kindred_nonblocking_cancelled", cancellation_check)

    def _cancel_event(_t, _y):
        return -1.0 if bool(cancel_requested()) else 1.0

    _cancel_event.terminal = True
    _cancel_event.direction = -1.0
    _cancel_event._kindred_cancel_event = True
    _cancel_event._kindred_cancelled = cancel_requested
    return _cancel_event


def _fitting_cancel_requested(cancellation_check) -> bool:
    if cancellation_check is None:
        return False
    cancel_requested = getattr(cancellation_check, "_kindred_nonblocking_cancelled", cancellation_check)
    return bool(cancel_requested())


class FittingSeriesEvaluator(Protocol):
    """Exact evaluator contract consumed by the Fit optimizer."""

    def evaluate_series_with_parameter_origins(
        self,
        params: Mapping[str, float],
        origins: Optional[Mapping[str, str]] = None,
        *,
        failed_params: Optional[Dict[str, float]] = None,
    ) -> SimulationResultValue: ...


def evaluate_fitting_series(
    evaluator: FittingSeriesEvaluator,
    params: Mapping[str, float],
    *,
    origins: Optional[Mapping[str, str]] = None,
    failed_params: Optional[Dict[str, float]] = None,
) -> SimulationResultValue:
    if not callable(
        getattr(evaluator, "evaluate_series_with_parameter_origins", None)
    ):
        raise TypeError(
            "Fit evaluator must implement "
            "evaluate_series_with_parameter_origins(...)."
        )
    value = evaluator.evaluate_series_with_parameter_origins(
        dict(params or {}),
        dict(origins or {}),
        failed_params=failed_params,
    )
    if not isinstance(value, SimulationResultValue):
        raise TypeError("Fit evaluator must return a SimulationResultValue.")
    return value


def _fitting_plan_from_execution_request(
    execution_request: SimulationExecutionRequest,
) -> SimulationPlan:
    return SimulationPlan.from_execution_request(
        execution_request,
        execution_mode="fitting",
        algebra_policy=SimulationAlgebraPolicy.FITTING_STRICT,
    )


def _coerce_fitting_simulation_plan(
    value: SimulationPlan,
) -> SimulationPlan:
    from kindred.core.simulation_algebra_policy import ensure_fitting_strict_algebra_policy

    if not isinstance(value, SimulationPlan):
        raise TypeError("simulation_plan must be SimulationPlan.")
    plan = value
    if plan.execution_mode != "fitting":
        raise ValueError("Fitting execution context requires a fitting simulation plan.")
    ensure_fitting_strict_algebra_policy(plan.algebra_policy)
    return SimulationPlan.from_payload(plan.to_payload())


@dataclass(frozen=True)
class PreparedFittingExecutionContext:
    """Required final fitting plan and its typed preparation metadata."""

    simulation_plan: SimulationPlan
    requested_param_names: tuple[str, ...]
    prepared_metadata: PreparedSimulationMetadata
    temperature_K: float
    initial_prefix: str
    requested_sample_times: tuple[float, ...]
    fit_sampling_policy: str = "fit_internal_regular_grid"
    fit_sampling_identity: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        plan = _coerce_fitting_simulation_plan(self.simulation_plan)
        if not isinstance(self.prepared_metadata, PreparedSimulationMetadata):
            raise TypeError(
                "prepared_metadata must be PreparedSimulationMetadata."
            )
        names = tuple(
            dict.fromkeys(
                str(name)
                for name in self.requested_param_names
                if str(name).strip()
            )
        )
        object.__setattr__(self, "simulation_plan", plan)
        object.__setattr__(
            self,
            "requested_param_names",
            names,
        )
        object.__setattr__(self, "temperature_K", float(self.temperature_K))
        object.__setattr__(
            self,
            "initial_prefix",
            str(self.initial_prefix or "init:"),
        )
        sample_times = tuple(float(value) for value in self.requested_sample_times)
        t_start, t_end = plan.execution_request.t_span
        if (
            not sample_times
            or any(not np.isfinite(value) for value in sample_times)
            or any(right <= left for left, right in zip(sample_times, sample_times[1:]))
            or sample_times[0] < float(t_start)
            or sample_times[-1] > float(t_end)
        ):
            raise ValueError(
                "Fitting requested sample times must be finite, strictly "
                "increasing, and inside the execution span."
            )
        object.__setattr__(self, "requested_sample_times", sample_times)
        object.__setattr__(
            self,
            "fit_sampling_policy",
            str(self.fit_sampling_policy or "fit_internal_regular_grid"),
        )
        object.__setattr__(
            self,
            "fit_sampling_identity",
            dict(self.fit_sampling_identity or {}),
        )

    def clone(
        self,
        *,
        prepared_metadata: PreparedSimulationMetadata | None = None,
    ) -> "PreparedFittingExecutionContext":
        return type(self)(
            simulation_plan=SimulationPlan.from_payload(copy.deepcopy(self.simulation_plan.to_payload())),
            requested_param_names=self.requested_param_names,
            prepared_metadata=self.prepared_metadata if prepared_metadata is None else prepared_metadata,
            temperature_K=float(self.temperature_K),
            initial_prefix=str(self.initial_prefix),
            requested_sample_times=self.requested_sample_times,
            fit_sampling_policy=self.fit_sampling_policy,
            fit_sampling_identity=dict(self.fit_sampling_identity),
        )


def prepare_fitting_execution_context(
    *,
    prepared_dsl: PreparedDslRequestSet,
    param_names: List[str],
    t_start: float = 0.0,
    t_end: float,
    num_points: int,
    objective_sample_times: Sequence[float],
    solver: str = DEFAULT_SOLVER_NAME,
    rtol: float = 1e-6,
    atol: float = 1e-12,
    use_sparse_jacobian: bool = USE_SPARSE_JACOBIAN_DEFAULT,
    wegscheider_cyclicity_enabled: bool = WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
    initial_prefix: str = "init:",
    fit_sampling_policy: str = "fit_internal_regular_grid",
    fit_sampling_identity: Mapping[str, object] | None = None,
) -> PreparedFittingExecutionContext:
    """Prepare structured execution data for serial fitting evaluation."""

    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
    t_start = float(t_start)
    t_end = float(t_end)
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        raise ValueError("Fitting execution requires finite t_end > t_start.")
    grid_n = max(2, int(num_points))
    initial_prefix = str(initial_prefix or "init:")
    mechanism_param_names = [
        str(name)
        for name in (param_names or ())
        if str(name).strip()
        and not str(name).startswith(initial_prefix)
    ]
    temperature_K = float(prepared_dsl.analysis.structure["temperature_K"])
    objective_times = np.asarray(tuple(objective_sample_times), dtype=float).reshape(-1)
    if (
        np.any(~np.isfinite(objective_times))
        or np.any(objective_times < t_start)
        or np.any(objective_times > t_end)
    ):
        raise ValueError(
            "Objective observation times must be finite and inside the Fit span."
        )
    requested_sample_times = tuple(
        float(value)
        for value in np.unique(
            np.concatenate(
                (
                    np.linspace(t_start, t_end, grid_n, dtype=float),
                    objective_times,
                )
            )
        )
    )

    prepared_solver_config = _build_solver_config(
        solver_input=str(solver or DEFAULT_SOLVER_NAME),
        rtol=rtol,
        atol=atol,
        grid={"N": int(grid_n)},
        use_sparse_jacobian=bool(use_sparse_jacobian),
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
    )
    bound = prepare_bound_mechanism(
        prepared_dsl,
        mechanism_param_names,
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
    )
    try:
        _reject_requested_derived_scalars_for_fitting(
            bound.mechanism,
            mechanism_param_names,
        )
        requested_partition = partition_simulation_parameter_values(
            mechanism=bound.mechanism,
            parameter_overrides=None,
            unresolved_intervention_schedule=bound.unresolved_intervention_schedule,
            requested_parameter_names=mechanism_param_names,
            runtime_parameter_names=_available_energy_binding_names(bound.mechanism),
        )
        _raise_unowned_request_parameter_values(requested_partition)
        canonical_requested_param_names = canonicalize_request_parameter_names(
            requested_partition,
            mechanism_param_names,
        )
        _reject_requested_algebra_owned_mechanism_parameters_for_fitting(
            bound.mechanism,
            requested_partition.bindable_mechanism_parameter_names,
        )
    except SimulationPreparationError as exc:
        raise _fit_simulation_error_from_preparation_error(exc) from exc
    except ValueError as exc:
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("parameter_binding", exc)
        ) from exc
    prepared_payload = dict(bound.as_serializable_execution_payload())
    prepared_payload["requested_parameter_names"] = list(canonical_requested_param_names)
    intervention_schedule = bound.unresolved_intervention_schedule
    from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

    (
        intervention_schedule_declarative_fingerprint,
        intervention_schedule_executable_fingerprint,
    ) = normalized_intervention_schedule_identity_fingerprints(
        intervention_schedule,
        mechanism_namespace=build_namespace_from_mechanism(bound.mechanism),
    )

    prepared_meta = PreparedSimulationMetadata(
        version=1,
        mechanism_text_sha256=hashlib.sha256(
            prepared_dsl.analysis.normalized_text.encode("utf-8")
        ).hexdigest(),
        mechanism_text_len=len(prepared_dsl.analysis.normalized_text),
        param_names=sorted({str(x) for x in canonical_requested_param_names if str(x).strip()}),
        t_end=float(t_end),
        num_points=int(grid_n),
        temperature_K=float(temperature_K),
        solver_requested=str(prepared_solver_config.solver_input),
        solver_normalized=str(prepared_solver_config.solver),
        rtol=float(prepared_solver_config.rtol),
        atol=float(prepared_solver_config.atol),
        use_sparse_jacobian=bool(prepared_solver_config.use_sparse_jacobian),
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
        initial_prefix=str(initial_prefix),
        intervention_schedule_declarative_fingerprint=str(intervention_schedule_declarative_fingerprint),
        intervention_schedule_executable_fingerprint=str(intervention_schedule_executable_fingerprint),
    )

    execution_request = SimulationExecutionRequest(
        prepared_payload=prepared_payload,
        initials={},
        t_span=(float(t_start), float(t_end)),
        solver_config={
            "solver": str(prepared_solver_config.solver),
            "rtol": float(prepared_solver_config.rtol),
            "atol": float(prepared_solver_config.atol),
            "grid": {"N": int(grid_n)},
            "use_sparse_jacobian": bool(prepared_solver_config.use_sparse_jacobian),
            MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED: bool(
                prepared_solver_config.wegscheider_cyclicity_enabled
            ),
            MechanismMetadataKeys.TEMPERATURE_K: float(temperature_K),
        },
        intervention_schedule=intervention_schedule,
    )
    return PreparedFittingExecutionContext(
        simulation_plan=_fitting_plan_from_execution_request(
            execution_request
        ),
        requested_param_names=tuple(
            sorted(
                {
                    str(x)
                    for x in canonical_requested_param_names
                    if str(x).strip()
                }
            )
        ),
        prepared_metadata=prepared_meta,
        temperature_K=float(temperature_K),
        initial_prefix=str(initial_prefix),
        requested_sample_times=requested_sample_times,
        fit_sampling_policy=str(fit_sampling_policy or "fit_internal_regular_grid"),
        fit_sampling_identity=dict(fit_sampling_identity or {}),
    )


class SerialFittingEvaluator:
    """Serial in-process evaluator built from structured fitting execution data."""

    def __init__(
        self,
        context: PreparedFittingExecutionContext,
        *,
        fixed_params: Optional[Mapping[str, float]] = None,
        fixed_param_origins: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._context = context.clone()
        self._fixed_params = {
            str(name): float(value) for name, value in dict(fixed_params or {}).items() if str(name).strip()
        }
        raw_fixed_origins = dict(fixed_param_origins or {})
        self._fixed_param_origins = {
            name: _parameter_origin_for(name, raw_fixed_origins)
            for name in self._fixed_params
        }
        self._prepared_run = None
        self._species_index: Dict[str, int] = {}
        self._prepared_solver_config = None
        self._parameter_algebra_spec = None
        self._compiled_algebra = None
        self._cancellation_check = None

    @property
    def prepared_metadata(self) -> PreparedSimulationMetadata:
        return self._context.prepared_metadata

    def set_cancellation_check(
        self,
        cancellation_check,
    ) -> "SerialFittingEvaluator":
        self._cancellation_check = cancellation_check
        return self

    def _raise_if_cancel_requested(self) -> None:
        if _fitting_cancel_requested(self._cancellation_check):
            raise FittingCancelled()

    def _raise_unrequested_schedule_candidate_authority(
        self,
        parameter_partition: Any,
        candidate_parameter_names: set[str],
    ) -> None:
        if not candidate_parameter_names:
            return
        schedule_names = {
            str(name)
            for name in getattr(parameter_partition, "schedule_parameter_names", ())
            if str(name).strip()
        }
        if not schedule_names:
            return
        requested_names = {
            str(name)
            for name in (self._context.requested_param_names or ())
            if str(name).strip()
        }
        schedule_name_by_raw = dict(
            getattr(parameter_partition, "schedule_parameter_name_by_raw", {}) or {}
        )
        owner_by_raw = {
            **dict(getattr(parameter_partition, "mechanism_parameter_name_by_raw", {}) or {}),
            **dict(getattr(parameter_partition, "unbound_mechanism_parameter_name_by_raw", {}) or {}),
            **dict(getattr(parameter_partition, "runtime_parameter_name_by_raw", {}) or {}),
        }
        schedule_owner_by_raw = {
            raw_name: str(schedule_name_by_raw.get(raw_name, raw_name))
            for raw_name in schedule_names
        }
        for candidate_name in sorted(str(name) for name in candidate_parameter_names if str(name).strip()):
            candidate_owner = str(owner_by_raw.get(candidate_name, candidate_name))
            for schedule_raw_name, schedule_owner in sorted(schedule_owner_by_raw.items()):
                if candidate_name != schedule_raw_name and candidate_owner != schedule_owner:
                    continue
                if (
                    candidate_name in requested_names
                    or candidate_owner in requested_names
                    or schedule_raw_name in requested_names
                    or schedule_owner in requested_names
                ):
                    continue
                raise FitSimulationError(
                    (
                        f"Unrequested intervention schedule parameter {candidate_name!r} "
                        "cannot be used as fitting candidate authority."
                    ),
                    details={"fatal": True, "stage": "parameter_binding"},
                )

    def __call__(self, params: Dict[str, float]) -> SimulationResultValue:
        return self.evaluate_series(params)

    def evaluate_series(self, params: Mapping[str, float]) -> SimulationResultValue:
        configured_origins = {
            str(name): FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR
            for name in dict(params or {})
            if str(name).strip()
        }
        return self.evaluate_series_with_parameter_origins(
            params,
            configured_origins,
            failed_params=None,
        )

    def evaluate_series_with_parameter_origins(
        self,
        params: Mapping[str, float],
        origins: Optional[Mapping[str, str]] = None,
        *,
        failed_params: Optional[Dict[str, float]] = None,
    ) -> SimulationResultValue:
        self._raise_if_cancel_requested()
        self._ensure_prepared()
        # Every candidate starts from the retained immutable execution request.
        # A rejected candidate can therefore never contaminate the next one.
        prepared_run = prepare_simulation_worker_run(
            execution_request=self._context.simulation_plan.execution_request
        )
        try:
            candidate_bindings = _bind_requested_parameters_to_prepared_mechanism(
                prepared_run.mechanism,
                self._context.requested_param_names,
                unresolved_intervention_schedule=(
                    prepared_run.unresolved_intervention_schedule
                ),
            )
        except SimulationPreparationError as exc:
            raise _fit_simulation_error_from_preparation_error(exc) from exc
        except Exception as exc:
            raise FitSimulationError(
                f"Structured fitting candidate preparation failed: {exc}",
                details={"fatal": True, "stage": "parameter_binding"},
            ) from exc

        param_map = {str(name): float(value) for name, value in self._fixed_params.items()}
        origin_map = dict(self._fixed_param_origins)
        runtime_origins = dict(origins or {})
        for raw_name, raw_value in dict(params or {}).items():
            name = str(raw_name)
            if not name.strip():
                continue
            param_map[name] = raw_value
            origin_map[name] = _parameter_origin_for(name, runtime_origins)

        initial_overrides: Dict[str, float] = {}
        shared_values: Dict[str, float] = {}
        from kindred.core.simulation_preparation import partition_simulation_parameter_values

        candidate_parameter_names = {
            str(name)
            for name in dict(params or {})
            if str(name).strip() and not str(name).startswith(self._context.initial_prefix)
        }
        non_initial_parameters = {
            str(name): value
            for name, value in param_map.items()
            if not str(name).startswith(self._context.initial_prefix)
        }
        try:
            parameter_partition = partition_simulation_parameter_values(
                mechanism=prepared_run.mechanism,
                parameter_overrides=non_initial_parameters,
                unresolved_intervention_schedule=prepared_run.unresolved_intervention_schedule,
                runtime_parameter_names=_available_energy_binding_names(prepared_run.mechanism),
                validate_values=False,
            )
            _raise_unowned_request_parameter_values(parameter_partition)
            self._raise_unrequested_schedule_candidate_authority(
                parameter_partition,
                candidate_parameter_names,
            )
        except SimulationPreparationError as exc:
            raise _fit_simulation_error_from_preparation_error(exc) from exc
        except ValueError as exc:
            raise FitSimulationError(
                str(exc),
                failed_params=failed_params,
                details={"fatal": True, "stage": "parameter_binding"},
            ) from exc
        requested_parameter_names = set(self._context.requested_param_names or ())
        for key, raw_val in param_map.items():
            name = str(key)
            if name.startswith(self._context.initial_prefix):
                species_name = name[len(self._context.initial_prefix) :]
                if species_name not in self._species_index:
                    continue
                initial_overrides[species_name] = _coerce_consumed_parameter_value(
                    name=name,
                    raw_value=raw_val,
                    origins=origin_map,
                    failed_params=failed_params,
                )
            else:
                binding_name = parameter_partition.mechanism_parameter_name_by_raw.get(name, name)
                if binding_name not in candidate_bindings:
                    if name in parameter_partition.schedule_only_parameter_names:
                        if name not in requested_parameter_names:
                            raise FitSimulationError(
                                (
                                    f"Unrequested intervention schedule parameter {name!r} "
                                    "cannot be used as fitting candidate authority."
                                ),
                                details={"fatal": True, "stage": "parameter_binding"},
                            )
                        _coerce_consumed_parameter_value(
                            name=name,
                            raw_value=raw_val,
                            origins=origin_map,
                            failed_params=failed_params,
                        )
                        continue
                    if name in parameter_partition.invalid_parameter_identifier_messages:
                        raise FitSimulationError(
                            parameter_partition.invalid_parameter_identifier_messages[name],
                            details={"fatal": True, "stage": "parameter_binding"},
                        )
                    if name in set(self._context.requested_param_names or ()):
                        raise FitSimulationError(
                            (
                                f"Requested fitting parameter {name!r} is unavailable after preparation; "
                                "it may be algebra-derived or unsupported as a fitted dimension."
                            ),
                            details={"fatal": True, "stage": "parameter_binding"},
                        )
                    continue
                shared_values[binding_name] = _coerce_consumed_parameter_value(
                    name=name,
                    raw_value=raw_val,
                    origins=origin_map,
                    failed_params=failed_params,
                )

        prepared_run = self._apply_shared_values(
            prepared_run,
            candidate_bindings,
            shared_values,
            origins=origin_map,
        )

        y0 = np.asarray(prepared_run.y0, dtype=float).copy()
        for species_name, value in initial_overrides.items():
            idx = self._species_index.get(species_name)
            if idx is None:
                continue
            y0[idx] = float(value)

        events = list(prepared_run.request.events or [])
        cancellation_event = _build_fitting_cancellation_event(self._cancellation_check)
        if cancellation_event is not None:
            events.append(cancellation_event)

        try:
            intervention_execution = resolve_prepared_run_intervention_execution(
                prepared_run,
                parameter_partition,
                cancellation_check=(
                    getattr(
                        self._cancellation_check,
                        "_kindred_nonblocking_cancelled",
                        self._cancellation_check,
                    )
                ),
            )
        except SimulationPreparationError as exc:
            schedule_names = {
                str(name)
                for name in getattr(parameter_partition, "schedule_parameter_names", ())
                if str(name).strip()
            }
            candidate_schedule = any(
                _parameter_origin_for(name, origin_map)
                in _OPTIMIZER_PARAMETER_ORIGINS
                for name in schedule_names
            )
            raise FitSimulationError(
                f"Fitting intervention execution failed: {exc}",
                details={
                    "fatal": not candidate_schedule,
                    "candidate_domain": candidate_schedule,
                    "stage": "intervention_execution",
                },
            ) from exc

        symbolic_jacobian = SymbolicJacobianExecution.from_request_fields(
            jacobian_func=prepared_run.request.jacobian_func,
            jac_sparsity=prepared_run.request.jac_sparsity,
            status=prepared_run.request.symbolic_jacobian_status,
        )
        prepared_solver_config = self._prepared_solver_config
        if (
            prepared_solver_config is not None
            and prepared_solver_config.use_sparse_jacobian
            and prepared_solver_config.solver.upper() in {"BDF", "RADAU"}
        ):
            try:
                jacobian_func, symbolic_identity = _bind_symbolic_jacobian_for_current_mechanism(
                    mechanism=prepared_run.mechanism,
                    prepared_solver_config=prepared_solver_config,
                    temperature_schedule=(
                        prepared_run.request.temperature_schedule
                    ),
                )
                symbolic_jacobian = SymbolicJacobianExecution.supported(
                    jacobian_func=jacobian_func,
                    identity=symbolic_identity,
                )
                self._context = self._context.clone(
                    prepared_metadata=_prepared_metadata_with_symbolic_jacobian(
                        self._context.prepared_metadata,
                        symbolic_jacobian,
                    )
                )
            except UnsupportedSymbolicExpressionError as exc:
                symbolic_jacobian = _symbolic_jacobian_for_bind_failure(prepared_run.mechanism, exc)
                self._context = self._context.clone(
                    prepared_metadata=_prepared_metadata_with_symbolic_jacobian(
                        self._context.prepared_metadata,
                        symbolic_jacobian,
                    )
                )

        request = build_simulation_request_from_prepared_run(
            prepared_run,
            y0=y0,
            intervention_execution=intervention_execution,
            symbolic_jacobian=symbolic_jacobian,
            events=events,
        )
        request = replace(
            request,
            t_eval=np.asarray(
                self._context.requested_sample_times,
                dtype=float,
            ),
        )
        try:
            result = solve_ode(request)
        except (FitSimulationError, SimulationCancelled):
            raise
        except Exception as exc:
            raise FitSimulationError(
                f"Fitting simulation failed: {exc}",
                details={"fatal": True, "stage": "solver"},
            ) from exc
        self._raise_if_cancel_requested()

        initials_map = {
            name: float(y0[index])
            for index, name in enumerate(prepared_run.species_names)
        }
        try:
            admitted = finalize_simulation_result(
                mechanism=prepared_run.mechanism,
                result=result,
                species_names=list(prepared_run.species_names),
                initials_for_algebra=initials_map,
                temperature_schedule=prepared_run.request.temperature_schedule,
                simulation_plan=self._context.simulation_plan,
            )
        except Exception as exc:
            if isinstance(exc, (FittingCancelled, SimulationCancelled)):
                raise FittingCancelled() from exc
            raise FitSimulationError(
                f"Fitting result finalization failed: {exc}",
                details={"fatal": True, "stage": "result_finalization"},
            ) from exc
        final_solver_provenance = dict(admitted.provenance.solver)
        final_solver_provenance.update(
            {
                "fit_sampling_policy": self._context.fit_sampling_policy,
                "fit_sampling_identity": dict(
                    self._context.fit_sampling_identity
                ),
                "fit_final_sampling_times": tuple(
                    float(value) for value in np.asarray(admitted.t, dtype=float)
                ),
            }
        )
        admitted = replace(
            admitted,
            provenance=replace(
                admitted.provenance,
                solver=final_solver_provenance,
            ),
        )
        segment_ids = np.asarray(admitted.segment_ids, dtype=np.int64)
        segment_keys = tuple(admitted.segment_keys)
        discontinuity_ids = np.asarray(
            admitted.discontinuity_ids,
            dtype=np.int64,
        )
        times = np.asarray(admitted.t, dtype=float)
        result_execution = admitted.intervention_execution
        if not isinstance(result_execution, InterventionExecutionContract):
            raise FitSimulationError(
                "Fitting result is missing its intervention execution contract.",
                details={"fatal": True, "stage": "intervention_execution"},
            )
        if (
            result_execution.with_trigger_firings(())
            != intervention_execution.with_trigger_firings(())
        ):
            raise FitSimulationError(
                "Fitting result intervention execution conflicts with the "
                "resolved request contract.",
                details={"fatal": True, "stage": "intervention_execution"},
            )
        if not (
            discontinuity_ids.size
            == len(segment_keys)
            == segment_ids.size
            == times.size
        ):
            raise FitSimulationError(
                "Fitting result lost aligned segment/discontinuity identity.",
                details={"fatal": True, "stage": "intervention_execution"},
            )
        if np.any(np.diff(segment_ids) < 0) or np.any(np.diff(discontinuity_ids) < 0):
            raise FitSimulationError(
                "Fitting result segment identity is not ordered.",
                details={"fatal": True, "stage": "intervention_execution"},
            )
        key_by_segment_id: dict[int, str] = {}
        for segment_id, segment_key in zip(
            segment_ids,
            segment_keys,
            strict=True,
        ):
            normalized_key = str(segment_key or "").strip()
            existing = key_by_segment_id.setdefault(
                int(segment_id),
                normalized_key,
            )
            if not normalized_key or existing != normalized_key:
                raise FitSimulationError(
                    "Fitting result segment identity is internally "
                    "inconsistent.",
                    details={
                        "fatal": True,
                        "stage": "intervention_execution",
                    },
                )
        self._raise_if_cancel_requested()
        return admitted

    def _ensure_prepared(self) -> None:
        """Prepare the evaluator's simulation state exactly once for the current owner.

        This method is idempotent: the fast path at the top is a plain attribute check
        against ``self._prepared_run``.

        The current call graph gives this method a single owner at a time. It runs in
        the worker-process initializer before any task dispatch in process-pool workers,
        and from ``evaluate_series_with_parameter_origins()`` on either the fit-worker
        QThread or a process-pool worker thread, where the evaluator instance is not
        shared across threads.

        The method is not thread-safe. If multiple threads ever shared one evaluator
        instance, they could both pass the idempotency check and both execute the
        mutations that follow. No lock is added here because the current call graph does
        not create concurrent callers.
        """
        if self._prepared_run is not None:
            return
        self._raise_if_cancel_requested()
        prepared_run = prepare_simulation_worker_run(
            execution_request=(
                self._context.simulation_plan.execution_request
            )
        )
        self._raise_if_cancel_requested()
        self._prepared_solver_config = _build_solver_config(
            solver_input=str(prepared_run.request.solver),
            rtol=float(prepared_run.request.rtol),
            atol=float(prepared_run.request.atol),
            grid=dict(prepared_run.request.grid or {}),
            use_sparse_jacobian=bool(self._context.prepared_metadata.use_sparse_jacobian),
            wegscheider_cyclicity_enabled=bool(self._context.prepared_metadata.wegscheider_cyclicity_enabled),
        )
        try:
            _bind_requested_parameters_to_prepared_mechanism(
                prepared_run.mechanism,
                self._context.requested_param_names,
                unresolved_intervention_schedule=prepared_run.unresolved_intervention_schedule,
            )
            from kindred.core.ode_builder import build_ode_rhs_from_mechanism

            rhs = build_ode_rhs_from_mechanism(prepared_run.mechanism)
            prepared_run = replace(
                prepared_run,
                rhs=rhs,
                request=replace(prepared_run.request, rhs=rhs),
            )
        except SimulationPreparationError as exc:
            raise _fit_simulation_error_from_preparation_error(exc) from exc
        except Exception as exc:
            raise FitSimulationError(
                f"Structured fitting preparation failed: {exc}",
                details={"fatal": True, "stage": "parameter_binding"},
            ) from exc
        self._prepared_run = prepared_run
        symbolic_jacobian = SymbolicJacobianExecution.from_request_fields(
            jacobian_func=prepared_run.request.jacobian_func,
            jac_sparsity=prepared_run.request.jac_sparsity,
            status=prepared_run.request.symbolic_jacobian_status,
        )
        symbolic_wegscheider_identity = getattr(
            prepared_run.request,
            "symbolic_wegscheider_identity",
            None,
        )
        if symbolic_jacobian.status or symbolic_jacobian.identity:
            self._context = self._context.clone(
                prepared_metadata=_prepared_metadata_with_symbolic_jacobian(
                    self._context.prepared_metadata,
                    symbolic_jacobian,
                )
            )
        if isinstance(symbolic_wegscheider_identity, Mapping) and symbolic_wegscheider_identity:
            self._context = self._context.clone(
                prepared_metadata=replace(
                    self._context.prepared_metadata,
                    symbolic_wegscheider_identity=dict(symbolic_wegscheider_identity),
                )
            )
        self._species_index = {name: idx for idx, name in enumerate(prepared_run.species_names)}
        self._raise_if_cancel_requested()

        from kindred.core.algebra.simulation_series import (
            CompiledAlgebraSeries,
            compiled_algebra_for_mechanism,
        )
        from kindred.core.simulation_algebra_policy import (
            ensure_fitting_strict_algebra_policy,
        )
        from kindred.core.simulator.parameter_algebra import parameter_algebra_spec_from_mechanism

        ensure_fitting_strict_algebra_policy(self._context.simulation_plan.algebra_policy)
        self._parameter_algebra_spec = parameter_algebra_spec_from_mechanism(prepared_run.mechanism)
        self._raise_if_cancel_requested()

        compiled_algebra: Optional[CompiledAlgebraSeries] = None
        compiled_algebra = compiled_algebra_for_mechanism(prepared_run.mechanism)
        self._compiled_algebra = compiled_algebra
        self._raise_if_cancel_requested()

    def _apply_shared_values(
        self,
        prepared_run: object,
        bindings: Mapping[str, Any],
        shared_values: Mapping[str, float],
        *,
        origins: Optional[Mapping[str, str]] = None,
    ) -> object:
        for name, value in shared_values.items():
            binding = bindings.get(name)
            if binding is None:
                continue
            try:
                binding.set(float(value))
            except (ValueError, OverflowError, FloatingPointError) as exc:
                origin = _parameter_origin_for(name, origins)
                raise FitSimulationError(
                    f"Failed to update parameter binding {name!r}: {exc}",
                    details={
                        "fatal": origin not in _OPTIMIZER_PARAMETER_ORIGINS,
                        "candidate_domain": origin in _OPTIMIZER_PARAMETER_ORIGINS,
                        "stage": "parameter_binding",
                    },
                ) from exc
            except Exception as exc:
                raise FitSimulationError(
                    f"Unexpected parameter binding failure for {name!r}: {exc}",
                    details={"fatal": True, "stage": "parameter_binding"},
                ) from exc

        if self._prepared_solver_config is None:
            raise FitSimulationError(
                "Fit candidate preparation is missing its solver config.",
                details={"fatal": True, "stage": "candidate_preparation"},
            )
        try:
            _apply_execution_parameter_algebra_and_cyclicity(
                mechanism=prepared_run.mechanism,
                prepared_solver_config=self._prepared_solver_config,
                require_mutable=True,
            )
        except SimulationPreparationError as exc:
            candidate_domain = bool(
                exc.stage in {"parameter_algebra", "wegscheider_cyclicity"}
                and any(
                    _parameter_origin_for(name, origins)
                    in _OPTIMIZER_PARAMETER_ORIGINS
                    for name in shared_values
                )
            )
            raise FitSimulationError(
                f"Failed to prepare complete fitting candidate: {exc}",
                details={
                    "fatal": not candidate_domain,
                    "candidate_domain": candidate_domain,
                    "stage": "candidate_preparation",
                },
            ) from exc
        except Exception as exc:
            raise FitSimulationError(
                f"Unexpected fitting candidate preparation failure: {exc}",
                details={"fatal": True, "stage": "candidate_preparation"},
            ) from exc
        try:
            from kindred.core.ode_builder import build_ode_rhs_from_mechanism

            rhs = build_ode_rhs_from_mechanism(prepared_run.mechanism)
        except Exception as exc:
            raise FitSimulationError(
                f"Fitting candidate ODE binding failed: {exc}",
                details={"fatal": True, "stage": "ode_build"},
            ) from exc
        return replace(
            prepared_run,
            rhs=rhs,
            request=replace(prepared_run.request, rhs=rhs),
        )
