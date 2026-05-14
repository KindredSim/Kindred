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
import numbers
import pickle
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from kindred.core.exceptions import FitSimulationError, FittingCancelled, SimulationCancelled
from kindred.core.mechanism_metadata import MechanismMetadataKeys
from kindred.core.runtime_defaults import (
    USE_SPARSE_JACOBIAN_DEFAULT,
    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
)
from kindred.core.intervention_schedule import (
    coerce_intervention_schedule,
    normalized_intervention_schedule_fingerprint,
)
from kindred.core.simulation_preparation import (
    PreparedSimulationMetadata,
    SimulationExecutionRequest,
    _bind_symbolic_jacobian_for_current_mechanism,
    _build_solver_config,
    _fit_simulation_error_from_preparation_error,
    _prepare_preparation_failure,
    _raise_unowned_request_parameter_values,
    _mechanism_supports_dynamic_symbolic_snapshot,
    _prepared_metadata_with_symbolic_jacobian,
    _reject_requested_algebra_owned_mechanism_parameters_for_fitting,
    _solve_request,
    _symbolic_jacobian_for_bind_failure,
    assert_simulation_execution_request_schedule_identity,
    build_simulation_request_from_prepared_run,
    canonicalize_request_parameter_names,
    coerce_prepared_simulation_metadata,
    metadata_view_for_mechanism,
    partition_simulation_parameter_values,
    prepare_bound_mechanism,
    prepare_simulation_worker_run,
    resolve_prepared_run_intervention_schedule,
    SimulationPreparationError,
)
from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
from kindred.core.simulation_series_payload import SimulationSeriesPayload, coerce_simulation_series_payload
from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME
from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution

logger = logging.getLogger(__name__)

__all__ = [
    "CallableFittingEvaluator",
    "FITTING_PARAM_ORIGIN_CONFIGURED_DATASET",
    "FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR",
    "FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET",
    "FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED",
    "PreparedFittingExecutionContext",
    "SerialFittingEvaluator",
    "coerce_fitting_series_evaluator",
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


def _prepared_metadata_from_evaluator(value) -> Optional[PreparedSimulationMetadata]:
    try:
        return coerce_prepared_simulation_metadata(
            getattr(value, "prepared_metadata", None)
            or getattr(value, "_kindred_prepared_simulation_meta", None)
        )
    except Exception:
        return None


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
            "parameter_origin": origin,
        },
    )


def _raise_for_forwarded_parameter_values(
    params: Mapping[str, float],
    *,
    origins: Optional[Mapping[str, str]],
    failed_params: Optional[Dict[str, float]],
) -> None:
    for raw_name, raw_value in dict(params or {}).items():
        name = str(raw_name)
        if not name.strip():
            continue
        _coerce_consumed_parameter_value(
            name=name,
            raw_value=raw_value,
            origins=origins,
            failed_params=failed_params,
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


class CallableFittingEvaluator:
    """Adapter that lifts a callable simulation boundary into the evaluator contract."""

    def __init__(self, simulation_func) -> None:
        if not callable(simulation_func):
            raise TypeError("simulation_func must be callable.")
        self._simulation_func = simulation_func

    @property
    def prepared_metadata(self) -> Optional[PreparedSimulationMetadata]:
        return _prepared_metadata_from_evaluator(self._simulation_func)

    def __call__(self, params: Mapping[str, float]) -> SimulationSeriesPayload:
        return self.evaluate_series(params)

    def evaluate_series(self, params: Mapping[str, float]) -> SimulationSeriesPayload:
        return coerce_simulation_series_payload(self._simulation_func(dict(params or {})))

    def evaluate_series_with_parameter_origins(
        self,
        params: Mapping[str, float],
        origins: Optional[Mapping[str, str]] = None,
        *,
        failed_params: Optional[Dict[str, float]] = None,
    ) -> SimulationSeriesPayload:
        forwarded = dict(params or {})
        _raise_for_forwarded_parameter_values(
            forwarded,
            origins=origins,
            failed_params=failed_params,
        )
        return self.evaluate_series(forwarded)


class _EvaluateSeriesMethodAdapter:
    """Adapter that makes evaluate_series-only evaluators callable."""

    def __init__(self, evaluator) -> None:
        self._evaluator = evaluator

    @property
    def prepared_metadata(self) -> Optional[PreparedSimulationMetadata]:
        return _prepared_metadata_from_evaluator(self._evaluator)

    def __call__(self, params: Mapping[str, float]) -> SimulationSeriesPayload:
        return self.evaluate_series(params)

    def evaluate_series(self, params: Mapping[str, float]) -> SimulationSeriesPayload:
        return coerce_simulation_series_payload(self._evaluator.evaluate_series(dict(params or {})))

    def evaluate_series_with_parameter_origins(
        self,
        params: Mapping[str, float],
        origins: Optional[Mapping[str, str]] = None,
        *,
        failed_params: Optional[Dict[str, float]] = None,
    ) -> SimulationSeriesPayload:
        origin_aware_evaluate_series = getattr(
            self._evaluator,
            "evaluate_series_with_parameter_origins",
            None,
        )
        if callable(origin_aware_evaluate_series):
            return coerce_simulation_series_payload(
                origin_aware_evaluate_series(
                    dict(params or {}),
                    dict(origins or {}),
                    failed_params=failed_params,
                )
            )
        forwarded = dict(params or {})
        _raise_for_forwarded_parameter_values(
            forwarded,
            origins=origins,
            failed_params=failed_params,
        )
        return self.evaluate_series(forwarded)


def coerce_fitting_series_evaluator(value):
    if hasattr(value, "evaluate_series") and callable(getattr(value, "evaluate_series")):
        if callable(value):
            return value
        return _EvaluateSeriesMethodAdapter(value)
    if callable(value):
        return CallableFittingEvaluator(value)
    raise TypeError("Fitting evaluator must expose evaluate_series(params) or be callable.")


def _fitting_series_evaluator_lane_clone_method(value):
    normalized = coerce_fitting_series_evaluator(value)
    if isinstance(normalized, _EvaluateSeriesMethodAdapter):
        clone_lane = getattr(normalized._evaluator, "_kindred_clone_fitting_evaluator_lane", None)
    else:
        clone_lane = getattr(normalized, "_kindred_clone_fitting_evaluator_lane", None)
    return clone_lane if callable(clone_lane) else None


def _clone_fitting_series_evaluator_lane(value):
    normalized = coerce_fitting_series_evaluator(value)
    source = normalized._evaluator if isinstance(normalized, _EvaluateSeriesMethodAdapter) else normalized
    clone_lane = _fitting_series_evaluator_lane_clone_method(value)
    if clone_lane is None:
        return None
    try:
        clone = clone_lane()
    except Exception:
        return None
    if clone is source or clone is normalized:
        return None
    try:
        return coerce_fitting_series_evaluator(clone)
    except Exception:
        return None


def _with_fitting_evaluator_cancellation_check(value, cancellation_check):
    normalized = coerce_fitting_series_evaluator(value)
    target = normalized._evaluator if isinstance(normalized, _EvaluateSeriesMethodAdapter) else normalized
    attach = getattr(target, "_kindred_set_fitting_cancellation_check", None)
    if callable(attach):
        attach(cancellation_check)
    return normalized


def _supports_fitting_evaluator_cancellation_check(value) -> bool:
    try:
        normalized = coerce_fitting_series_evaluator(value)
    except Exception:
        return False
    target = normalized._evaluator if isinstance(normalized, _EvaluateSeriesMethodAdapter) else normalized
    return callable(getattr(target, "_kindred_set_fitting_cancellation_check", None))


def _supports_isolated_fitting_evaluator_lanes(value) -> bool:
    try:
        clone_lane = _fitting_series_evaluator_lane_clone_method(value)
    except Exception:
        return False
    return clone_lane is not None


def evaluate_fitting_series(
    evaluator,
    params: Mapping[str, float],
    *,
    origins: Optional[Mapping[str, str]] = None,
    failed_params: Optional[Dict[str, float]] = None,
) -> SimulationSeriesPayload:
    normalized = coerce_fitting_series_evaluator(evaluator)
    origin_method = getattr(normalized, "evaluate_series_with_parameter_origins", None)
    if callable(origin_method):
        return coerce_simulation_series_payload(
            origin_method(
                dict(params or {}),
                dict(origins or {}),
                failed_params=failed_params,
            )
        )
    forwarded = dict(params or {})
    _raise_for_forwarded_parameter_values(
        forwarded,
        origins=origins,
        failed_params=failed_params,
    )
    return coerce_simulation_series_payload(normalized.evaluate_series(forwarded))


def _payload_value_for_comparison(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _payload_value_for_comparison(to_dict())
    fingerprint = getattr(value, "fingerprint", None)
    if callable(fingerprint):
        return str(fingerprint())
    if isinstance(value, Mapping):
        return {str(key): _payload_value_for_comparison(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_payload_value_for_comparison(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_payload_value_for_comparison(item) for item in value)
    return value


def _binding_payload_for_comparison(name: str, binding: Any) -> Dict[str, Any]:
    if isinstance(binding, Mapping):
        binding_name = binding.get("name", name)
        binding_value = binding.get("value")
    elif isinstance(binding, numbers.Real):
        binding_name = name
        binding_value = binding
    else:
        binding_name = getattr(binding, "name", name)
        binding_value = getattr(binding, "value", None)
    if binding_value is None and callable(binding):
        binding_value = binding()
    if binding_value is None:
        binding_value = 0.0
    return {"name": str(binding_name), "value": float(binding_value)}


def _execution_request_payload_for_comparison(value: SimulationExecutionRequest) -> Dict[str, Any]:
    payload = _payload_value_for_comparison(value.to_payload())
    prepared_payload = payload.get("prepared_payload")
    if isinstance(prepared_payload, dict):
        prepared_payload = dict(prepared_payload)
        prepared_payload.pop("mechanism", None)
        bindings = prepared_payload.get("bindings")
        if isinstance(bindings, dict):
            prepared_payload["bindings"] = {
                str(name): _binding_payload_for_comparison(str(name), binding)
                for name, binding in bindings.items()
            }
        payload["prepared_payload"] = prepared_payload
    return payload


def _fitting_plan_from_execution_request(
    execution_request: SimulationExecutionRequest | Mapping[str, Any],
) -> SimulationPlan:
    return SimulationPlan.from_execution_request(
        execution_request,
        execution_mode="fitting",
        algebra_policy=SimulationAlgebraPolicy.FITTING_STRICT,
    )


def _coerce_fitting_simulation_plan(
    value: SimulationPlan | Mapping[str, Any],
) -> SimulationPlan:
    from kindred.core.simulation_algebra_policy import ensure_fitting_strict_algebra_policy

    plan = value if isinstance(value, SimulationPlan) else SimulationPlan.from_payload(value)
    if plan.execution_mode != "fitting":
        raise ValueError("Fitting execution context requires a fitting simulation plan.")
    ensure_fitting_strict_algebra_policy(plan.algebra_policy)
    return SimulationPlan.from_payload(plan.to_payload())


def _assert_matching_fitting_execution_request(
    *,
    simulation_plan: SimulationPlan,
    execution_request: SimulationExecutionRequest | Mapping[str, Any],
) -> None:
    request = (
        execution_request
        if isinstance(execution_request, SimulationExecutionRequest)
        else SimulationExecutionRequest.from_mapping(dict(execution_request))
    )
    if _execution_request_payload_for_comparison(request) != _execution_request_payload_for_comparison(
        simulation_plan.execution_request
    ):
        raise ValueError("Fitting simulation_plan execution_request does not match execution_request.")


@dataclass(frozen=True, init=False)
class PreparedFittingExecutionContext:
    """Structured execution data for the shared fitting evaluation seam."""

    simulation_plan: SimulationPlan
    requested_param_names: List[str]
    prepared_metadata: PreparedSimulationMetadata
    temperature_K: float
    initial_prefix: str

    def __init__(
        self,
        *,
        requested_param_names: List[str],
        prepared_metadata: PreparedSimulationMetadata | Mapping[str, Any],
        temperature_K: float,
        initial_prefix: str,
        simulation_plan: SimulationPlan | Mapping[str, Any] | None = None,
        execution_request: SimulationExecutionRequest | Mapping[str, Any] | None = None,
    ) -> None:
        plan_value = simulation_plan
        if plan_value is None:
            if execution_request is None:
                raise TypeError("simulation_plan or execution_request is required.")
            plan = _fitting_plan_from_execution_request(execution_request)
        else:
            plan = _coerce_fitting_simulation_plan(plan_value)
            if execution_request is not None:
                _assert_matching_fitting_execution_request(
                    simulation_plan=plan,
                    execution_request=execution_request,
                )
        metadata_copy = coerce_prepared_simulation_metadata(prepared_metadata)
        if metadata_copy is None:
            raise TypeError("prepared_metadata must be a PreparedSimulationMetadata or compatible mapping.")
        object.__setattr__(self, "simulation_plan", plan)
        object.__setattr__(
            self,
            "requested_param_names",
            [str(name) for name in list(requested_param_names or []) if str(name).strip()],
        )
        object.__setattr__(self, "prepared_metadata", metadata_copy)
        object.__setattr__(self, "temperature_K", float(temperature_K))
        object.__setattr__(self, "initial_prefix", str(initial_prefix or "init:"))

    @property
    def execution_request(self) -> SimulationExecutionRequest:
        return self.simulation_plan.execution_request

    def clone(
        self,
        *,
        prepared_metadata: PreparedSimulationMetadata | Mapping[str, Any] | None = None,
    ) -> "PreparedFittingExecutionContext":
        return type(self)(
            simulation_plan=SimulationPlan.from_payload(copy.deepcopy(self.simulation_plan.to_payload())),
            requested_param_names=list(self.requested_param_names),
            prepared_metadata=self.prepared_metadata if prepared_metadata is None else prepared_metadata,
            temperature_K=float(self.temperature_K),
            initial_prefix=str(self.initial_prefix),
        )


def prepare_fitting_execution_context(
    *,
    mechanism_text: str,
    param_names: List[str],
    t_end: float,
    num_points: int,
    temperature_K: float = 298.15,
    solver: str = DEFAULT_SOLVER_NAME,
    rtol: float = 1e-6,
    atol: float = 1e-12,
    use_sparse_jacobian: bool = USE_SPARSE_JACOBIAN_DEFAULT,
    wegscheider_cyclicity_enabled: bool = WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
    initial_prefix: str = "init:",
) -> PreparedFittingExecutionContext:
    """Prepare structured execution data for serial fitting evaluation."""

    t_end = float(t_end)
    grid_n = max(2, int(num_points))
    initial_prefix = str(initial_prefix or "init:")

    prepared_solver_config = _build_solver_config(
        solver_input=str(solver or DEFAULT_SOLVER_NAME),
        rtol=rtol,
        atol=atol,
        grid={"N": int(grid_n)},
        use_sparse_jacobian=bool(use_sparse_jacobian),
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
    )
    if prepared_solver_config.solver_warning:
        logger.warning(prepared_solver_config.solver_warning)

    bound = prepare_bound_mechanism(
        mechanism_text=mechanism_text,
        param_names=list(param_names or []),
        temperature_K=float(temperature_K),
        initials={},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
    )
    requested_partition = partition_simulation_parameter_values(
        mechanism=bound.mechanism,
        parameter_overrides=None,
        unresolved_intervention_schedule=bound.unresolved_intervention_schedule,
        requested_parameter_names=param_names or [],
        runtime_parameter_names=bound.bindings.keys(),
    )
    try:
        _raise_unowned_request_parameter_values(requested_partition)
        canonical_requested_param_names = canonicalize_request_parameter_names(
            requested_partition,
            param_names or [],
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
    prepared_payload["bindings"] = dict(bound.bindings)
    intervention_schedule = coerce_intervention_schedule(prepared_payload.get("intervention_schedule"))
    from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

    intervention_schedule_fingerprint = normalized_intervention_schedule_fingerprint(
        intervention_schedule,
        mechanism_namespace=build_namespace_from_mechanism(bound.mechanism),
    )

    prepared_meta = PreparedSimulationMetadata(
        version=1,
        mechanism_text_sha256=hashlib.sha256((mechanism_text or "").encode("utf-8")).hexdigest(),
        mechanism_text_len=len(mechanism_text or ""),
        param_names=sorted({str(x) for x in canonical_requested_param_names if str(x).strip()}),
        t_end=float(t_end),
        num_points=int(grid_n),
        temperature_K=float(temperature_K),
        solver_requested=str(prepared_solver_config.solver_input),
        solver_normalized=str(prepared_solver_config.solver),
        solver_warning=(
            str(prepared_solver_config.solver_warning)
            if prepared_solver_config.solver_warning
            else None
        ),
        rtol=float(prepared_solver_config.rtol),
        atol=float(prepared_solver_config.atol),
        use_sparse_jacobian=bool(prepared_solver_config.use_sparse_jacobian),
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
        initial_prefix=str(initial_prefix),
        intervention_schedule_fingerprint=str(intervention_schedule_fingerprint),
    )

    execution_request = SimulationExecutionRequest(
        prepared_payload=prepared_payload,
        initials={},
        t_span=(0.0, float(t_end)),
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
        mechanism_text=str(mechanism_text or ""),
        intervention_schedule=intervention_schedule,
    )
    return PreparedFittingExecutionContext(
        execution_request=execution_request,
        requested_param_names=sorted({str(x) for x in canonical_requested_param_names if str(x).strip()}),
        prepared_metadata=prepared_meta,
        temperature_K=float(temperature_K),
        initial_prefix=str(initial_prefix),
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
        self._bindings: Dict[str, Any] = {}
        self._species_index: Dict[str, int] = {}
        self._last_shared_fp: Optional[Tuple[Tuple[str, float], ...]] = None
        self._prepared_solver_config = None
        self._parameter_algebra_spec = None
        self._compiled_algebra = None
        self._kindred_fitting_execution_context = self._context
        self._cancellation_check = None

    @property
    def prepared_metadata(self) -> PreparedSimulationMetadata:
        return self._context.prepared_metadata

    @property
    def context(self) -> PreparedFittingExecutionContext:
        return self._context

    def with_fixed_params(self, fixed_params: Mapping[str, float]) -> "SerialFittingEvaluator":
        merged = dict(self._fixed_params)
        merged_origins = dict(self._fixed_param_origins)
        for name, value in dict(fixed_params or {}).items():
            if str(name).strip():
                param_name = str(name)
                merged[param_name] = float(value)
                merged_origins[param_name] = FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR
        return type(self)(self._context, fixed_params=merged, fixed_param_origins=merged_origins)

    def _kindred_clone_fitting_evaluator_lane(self) -> "SerialFittingEvaluator":
        return type(self)(
            self._context,
            fixed_params=dict(self._fixed_params),
            fixed_param_origins=dict(self._fixed_param_origins),
        )

    def to_process_payload(self) -> Dict[str, Any]:
        self._ensure_prepared()
        payload = {
            "simulation_plan": self._context.simulation_plan.to_payload(),
            "requested_param_names": list(self._context.requested_param_names),
            "prepared_metadata": self._context.prepared_metadata.to_serializable_dict(),
            "temperature_K": float(self._context.temperature_K),
            "initial_prefix": str(self._context.initial_prefix),
            "fixed_params": dict(self._fixed_params),
            "fixed_param_origins": dict(self._fixed_param_origins),
        }
        payload_copy = copy.deepcopy(payload)
        pickle.dumps(payload_copy)
        return payload_copy

    @classmethod
    def from_process_payload(cls, payload: Mapping[str, Any]) -> "SerialFittingEvaluator":
        required_keys = (
            "simulation_plan",
            "requested_param_names",
            "prepared_metadata",
            "temperature_K",
            "initial_prefix",
            "fixed_params",
            "fixed_param_origins",
        )
        if not isinstance(payload, Mapping):
            raise TypeError("Process payload must be a mapping.")
        if "execution_request" in payload:
            raise KeyError("Process payload contains legacy execution_request key.")
        missing = [name for name in required_keys if name not in payload]
        if missing:
            raise KeyError(f"Process payload is missing required keys: {', '.join(missing)}")
        simulation_plan = payload["simulation_plan"]
        if not isinstance(simulation_plan, Mapping):
            raise TypeError("Process payload simulation_plan must be a mapping.")
        plan = SimulationPlan.from_payload(simulation_plan)
        plan_payload = plan.to_payload()
        prepared_metadata = PreparedSimulationMetadata.from_mapping(
            dict(payload["prepared_metadata"])
        )
        requested_param_names = sorted(
            {str(name) for name in list(payload["requested_param_names"] or []) if str(name).strip()}
        )
        metadata_param_names = sorted(str(name) for name in list(prepared_metadata.param_names or []))
        if requested_param_names != metadata_param_names:
            raise ValueError("Process payload requested_param_names conflict with simulation_plan metadata.")
        request = plan.to_execution_request()
        solver_config = dict(request.solver_config or {})
        plan_temperature = float(
            solver_config.get(
                MechanismMetadataKeys.TEMPERATURE_K,
                prepared_metadata.temperature_K,
            )
        )
        if abs(float(payload["temperature_K"]) - plan_temperature) > 1e-12:
            raise ValueError("Process payload temperature_K conflicts with simulation_plan.")
        if abs(float(prepared_metadata.temperature_K) - plan_temperature) > 1e-12:
            raise ValueError("Process payload prepared_metadata.temperature_K conflicts with simulation_plan.")
        if str(payload["initial_prefix"] or "init:") != str(prepared_metadata.initial_prefix or "init:"):
            raise ValueError("Process payload initial_prefix conflicts with prepared_metadata.")
        try:
            assert_simulation_execution_request_schedule_identity(
                request,
                expected_fingerprint=str(prepared_metadata.intervention_schedule_fingerprint or ""),
            )
        except SimulationPreparationError as exc:
            raise ValueError(
                f"Process payload intervention_schedule conflicts with simulation_plan: {exc}"
            ) from exc
        return cls(
            PreparedFittingExecutionContext(
                simulation_plan=plan_payload,
                execution_request=None,
                requested_param_names=list(metadata_param_names),
                prepared_metadata=prepared_metadata,
                temperature_K=float(plan_temperature),
                initial_prefix=str(prepared_metadata.initial_prefix),
            ),
            fixed_params=dict(payload["fixed_params"]),
            fixed_param_origins=dict(payload["fixed_param_origins"]),
        )

    def _kindred_set_fitting_cancellation_check(self, cancellation_check) -> "SerialFittingEvaluator":
        self._cancellation_check = cancellation_check
        return self

    def _raise_if_cancel_requested(self) -> None:
        if _fitting_cancel_requested(self._cancellation_check):
            raise FittingCancelled()

    def __call__(self, params: Dict[str, float]) -> SimulationSeriesPayload:
        return self.evaluate_series(params)

    def evaluate_series(self, params: Mapping[str, float]) -> SimulationSeriesPayload:
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
    ) -> SimulationSeriesPayload:
        self._raise_if_cancel_requested()
        self._ensure_prepared()
        prepared_run = self._prepared_run
        if prepared_run is None:
            raise RuntimeError("Prepared fitting lane unavailable.")

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
                runtime_parameter_names=self._bindings.keys(),
                validate_values=False,
            )
            _raise_unowned_request_parameter_values(parameter_partition)
        except SimulationPreparationError as exc:
            raise _fit_simulation_error_from_preparation_error(exc) from exc
        except ValueError as exc:
            raise FitSimulationError(
                str(exc),
                failed_params=failed_params,
                details={"fatal": True, "stage": "parameter_binding"},
            ) from exc
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
                if binding_name not in self._bindings:
                    if name in parameter_partition.schedule_only_parameter_names:
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

        shared_fp = tuple(sorted((name, float(val)) for name, val in shared_values.items()))
        if shared_fp != self._last_shared_fp:
            self._apply_shared_values(shared_values)
            self._last_shared_fp = shared_fp

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
            intervention_schedule = resolve_prepared_run_intervention_schedule(
                prepared_run,
                parameter_partition,
            )
        except SimulationPreparationError as exc:
            raise FitSimulationError(
                f"Fitting intervention schedule failed: {exc}",
                details={"fatal": False, "stage": "intervention_schedule"},
            ) from exc

        symbolic_jacobian = SymbolicJacobianExecution.from_request_fields(
            jacobian_func=prepared_run.request.jacobian_func,
            jac_sparsity=prepared_run.request.jac_sparsity,
            status=prepared_run.request.symbolic_jacobian_status,
        )
        prepared_solver_config = self._prepared_solver_config
        if (
            prepared_solver_config is not None
            and bool(getattr(prepared_solver_config, "use_sparse_jacobian", False))
            and str(getattr(prepared_solver_config, "solver", prepared_run.request.solver)).upper() in {"BDF", "RADAU"}
            and prepared_run.request.temperature_schedule is None
            and _mechanism_supports_dynamic_symbolic_snapshot(prepared_run.mechanism)
        ):
            try:
                jacobian_func, symbolic_identity = _bind_symbolic_jacobian_for_current_mechanism(
                    mechanism=prepared_run.mechanism,
                    prepared_solver_config=prepared_solver_config,
                    temperature_K=float(self._context.temperature_K),
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
                self._kindred_fitting_execution_context = self._context
            except UnsupportedSymbolicExpressionError as exc:
                symbolic_jacobian = _symbolic_jacobian_for_bind_failure(prepared_run.mechanism, exc)
                self._context = self._context.clone(
                    prepared_metadata=_prepared_metadata_with_symbolic_jacobian(
                        self._context.prepared_metadata,
                        symbolic_jacobian,
                    )
                )
                self._kindred_fitting_execution_context = self._context

        request = build_simulation_request_from_prepared_run(
            prepared_run,
            y0=y0,
            intervention_schedule=intervention_schedule,
            symbolic_jacobian=symbolic_jacobian,
            events=events,
        )
        try:
            result = _solve_request(request)
        except (FitSimulationError, SimulationCancelled):
            raise
        except Exception as exc:
            raise FitSimulationError(
                f"Fitting simulation failed: {exc}",
                details={"fatal": False},
            ) from exc

        species_payload = {
            name: np.asarray(result.Y[idx, :], dtype=float).reshape(-1).copy()
            for idx, name in enumerate(prepared_run.species_names)
        }

        algebra_scalars: Dict[str, float] = {}
        if self._compiled_algebra is not None:
            from kindred.core.algebra.simulation_series import (
                evaluate_compiled_algebra_series_for_simulation,
            )
            from kindred.core.simulation_algebra_policy import (
                ensure_fitting_strict_algebra_policy,
                fitting_strict_evaluation_error,
            )

            ensure_fitting_strict_algebra_policy(self._context.simulation_plan.algebra_policy)
            try:
                initials_map = {
                    name: float(y0[idx]) for idx, name in enumerate(prepared_run.species_names)
                }
                species_series = {
                    name: np.asarray(result.Y[idx, :], dtype=float).reshape(-1)
                    for idx, name in enumerate(prepared_run.species_names)
                }
                algebra_series, algebra_scalars = evaluate_compiled_algebra_series_for_simulation(
                    prepared_run.mechanism,
                    self._compiled_algebra,
                    t=np.asarray(result.t, dtype=float).reshape(-1),
                    species_series=species_series,
                    initials=initials_map,
                    temperature_K=float(self._context.temperature_K),
                )
                for name, values in (algebra_series or {}).items():
                    if name in species_payload:
                        continue
                    species_payload[str(name)] = np.asarray(values, dtype=float).reshape(-1).copy()
            except FitSimulationError:
                raise
            except Exception as exc:
                raise fitting_strict_evaluation_error(
                    exc,
                    message_prefix="Algebra evaluation failed during fitting simulation",
                ) from exc

        return coerce_simulation_series_payload(
            SimulationSeriesPayload(
                t=np.asarray(result.t, dtype=float).reshape(-1).copy(),
                species=species_payload,
                algebra_scalars=dict(algebra_scalars),
            )
        )

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
        prepared_run = prepare_simulation_worker_run(execution_request=self._context.execution_request)
        self._raise_if_cancel_requested()
        self._prepared_solver_config = _build_solver_config(
            solver_input=str(prepared_run.request.solver),
            rtol=float(prepared_run.request.rtol),
            atol=float(prepared_run.request.atol),
            grid=dict(prepared_run.request.grid or {}),
            use_sparse_jacobian=bool(self._context.prepared_metadata.use_sparse_jacobian),
            wegscheider_cyclicity_enabled=bool(self._context.prepared_metadata.wegscheider_cyclicity_enabled),
        )
        prepared_payload = dict(self._context.execution_request.prepared_payload or {})
        bindings = prepared_payload.get("bindings") or {}
        if not isinstance(bindings, Mapping):
            raise FitSimulationError("Structured fitting payload is missing mutable bindings.", details={"fatal": True})
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
            self._kindred_fitting_execution_context = self._context
        if isinstance(symbolic_wegscheider_identity, Mapping) and symbolic_wegscheider_identity:
            self._context = self._context.clone(
                prepared_metadata=replace(
                    self._context.prepared_metadata,
                    symbolic_wegscheider_identity=dict(symbolic_wegscheider_identity),
                )
            )
            self._kindred_fitting_execution_context = self._context
        self._bindings = dict(bindings)
        self._species_index = {name: idx for idx, name in enumerate(prepared_run.species_names)}
        self._raise_if_cancel_requested()

        from kindred.core.algebra.simulation_series import (
            CompiledAlgebraSeries,
            compile_algebra_observables,
        )
        from kindred.core.simulation_algebra_policy import (
            ensure_fitting_strict_algebra_policy,
            fitting_strict_parse_error,
            fitting_strict_time_ref_error,
        )
        from kindred.core.simulator.parameter_algebra import parameter_algebra_spec_from_mechanism
        from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

        ensure_fitting_strict_algebra_policy(self._context.simulation_plan.algebra_policy)
        self._parameter_algebra_spec = parameter_algebra_spec_from_mechanism(prepared_run.mechanism)
        self._raise_if_cancel_requested()

        compiled_algebra: Optional[CompiledAlgebraSeries] = None
        algebra_text = metadata_view_for_mechanism(prepared_run.mechanism).algebra_text
        if algebra_text:
            try:
                compiled_algebra = compile_algebra_observables(
                    str(algebra_text),
                    mechanism_namespace=build_namespace_from_mechanism(prepared_run.mechanism),
                )
            except Exception as exc:
                raise fitting_strict_parse_error(
                    exc,
                    message_prefix="Failed to parse Algebra observables for fitting",
                ) from exc
            if compiled_algebra.time_ref_statements:
                stmt = compiled_algebra.time_ref_statements[0]
                raise fitting_strict_time_ref_error(stmt)
        self._compiled_algebra = compiled_algebra
        self._raise_if_cancel_requested()

    def _apply_shared_values(self, shared_values: Mapping[str, float]) -> None:
        prepared_run = self._prepared_run
        if prepared_run is None:
            raise RuntimeError("Prepared fitting lane unavailable.")
        for name, value in shared_values.items():
            binding = self._bindings.get(name)
            if binding is None:
                continue
            try:
                binding.set(float(value))
            except Exception as exc:
                raise FitSimulationError(
                    f"Failed to update parameter binding {name!r}: {exc}",
                    details={"fatal": True},
                ) from exc

        from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_spec_to_mechanism

        try:
            if self._parameter_algebra_spec is not None:
                _ = apply_parameter_algebra_spec_to_mechanism(
                    self._parameter_algebra_spec,
                    mechanism=prepared_run.mechanism,
                    require_mutable=True,
                )
        except Exception as exc:
            raise FitSimulationError(f"Parameter algebra failed during fitting simulation: {exc}") from exc
