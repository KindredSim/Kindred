from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
from kindred.core.simulation_preparation import SimulationExecutionRequest


@dataclass(frozen=True)
class BatchSetDispatchInput:
    set_id: str
    set_name: str
    fast_mode: bool
    t_end: float
    solver_config: Mapping[str, Any]
    cache_key: str
    scope_identity: Mapping[str, Any]
    queue_ids: Sequence[str]
    initials: Mapping[str, Any]
    mechanism_text: str
    simulation_identity: Mapping[str, Any] | None
    plan_payload: Mapping[str, Any] | None
    preview_batch_cache_token: str = ""
    prepared_payload: Mapping[str, Any] | None = None
    parameter_overrides: Mapping[str, Any] | None = None
    intervention_schedule: Mapping[str, Any] | None = None
    contained_owner_identity: Mapping[str, Any] | None = None
    algebra_policy: SimulationAlgebraPolicy = SimulationAlgebraPolicy.BATCH_BEST_EFFORT


@dataclass(frozen=True)
class BatchSetDispatchPlan:
    plan_payload: Dict[str, Any] | None
    execution_request: Dict[str, Any] | None
    simulation_identity: Dict[str, Any]


@dataclass(frozen=True)
class SerialBatchDispatchInput:
    payload: Any
    queue_ids: Sequence[str]
    set_id: str
    set_name: str
    initials: Mapping[str, Any]
    slider_overrides: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SerialBatchDispatchPlan:
    plan_payload: Dict[str, Any] | None
    cache_key: str
    worker_signature: str
    cache_key_rewritten: bool = False


@dataclass(frozen=True)
class ParallelBatchTaskInput:
    payload: Any
    set_id: str
    set_name: str
    queue_ids: Sequence[str]
    initials: Mapping[str, Any]
    include_mechanism_in_result_payload: bool


@dataclass(frozen=True)
class ParallelBatchTaskPlan:
    task: Dict[str, Any]
    simulation_identity: Dict[str, Any]


def _simulation_plan_payload(value: object) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, SimulationPlan):
        return value.to_payload()
    if isinstance(value, Mapping):
        return SimulationPlan.from_payload(value).to_payload()
    return None


def simulation_plan_from_payloadish(value: object) -> Optional[SimulationPlan]:
    if value is None:
        return None
    if isinstance(value, SimulationPlan):
        return value
    if isinstance(value, Mapping):
        return SimulationPlan.from_payload(value)
    return None


def simulation_plan_payload(value: object) -> Optional[Dict[str, Any]]:
    return _simulation_plan_payload(value)


def _execution_request_payload_from_plan(value: object) -> Optional[Dict[str, Any]]:
    payload = _simulation_plan_payload(value)
    if payload is None:
        return None
    return SimulationPlan.from_payload(payload).to_execution_request().to_payload()


def execution_request_payload_from_plan(value: object) -> Optional[Dict[str, Any]]:
    return _execution_request_payload_from_plan(value)


def _simulation_plan_payload_with_execution_request(
    plan_payload: Mapping[str, Any],
    execution_request: Mapping[str, Any],
    *,
    algebra_policy: SimulationAlgebraPolicy,
) -> Dict[str, Any]:
    plan = SimulationPlan.from_payload(plan_payload)
    return SimulationPlan.from_execution_request(
        execution_request,
        execution_mode=plan.execution_mode,
        algebra_policy=algebra_policy,
        cache_identity_payload=plan.cache_identity_payload,
        cache_scope_payload=plan.cache_scope_payload,
        metadata=plan.metadata,
        version=plan.version,
    ).to_payload()


def simulation_plan_payload_with_execution_request(
    plan_payload: Mapping[str, Any],
    execution_request: Mapping[str, Any],
    *,
    algebra_policy: SimulationAlgebraPolicy,
) -> Dict[str, Any]:
    return _simulation_plan_payload_with_execution_request(
        plan_payload,
        execution_request,
        algebra_policy=algebra_policy,
    )


def _new_simulation_plan_payload(
    execution_request: Mapping[str, Any],
    *,
    execution_mode: str,
    algebra_policy: SimulationAlgebraPolicy,
    cache_identity_payload: Mapping[str, Any],
    cache_scope_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    return SimulationPlan.from_execution_request(
        execution_request,
        execution_mode=execution_mode,
        algebra_policy=algebra_policy,
        cache_identity_payload=cache_identity_payload,
        cache_scope_payload=cache_scope_payload,
        metadata=metadata,
    ).to_payload()


def build_batch_set_dispatch_plan(dispatch_input: BatchSetDispatchInput) -> BatchSetDispatchPlan:
    plan_payload = _simulation_plan_payload(dispatch_input.plan_payload)
    execution_request = _execution_request_payload_from_plan(plan_payload)

    if isinstance(execution_request, dict) and not bool(dispatch_input.fast_mode):
        execution_request = dict(execution_request)
        execution_request["prepared_payload"] = None
        if isinstance(plan_payload, dict):
            plan_payload = _simulation_plan_payload_with_execution_request(
                plan_payload,
                execution_request,
                algebra_policy=dispatch_input.algebra_policy,
            )

    simulation_identity = dict(dispatch_input.simulation_identity or {})

    if plan_payload is None:
        if isinstance(execution_request, dict):
            plan_request = dict(execution_request)
        else:
            request_kwargs: Dict[str, Any] = dict(
                prepared_payload=(
                    dict(dispatch_input.prepared_payload)
                    if isinstance(dispatch_input.prepared_payload, Mapping)
                    else None
                ),
                initials=dict(dispatch_input.initials),
                t_span=(0.0, float(dispatch_input.t_end)),
                solver_config=dict(dispatch_input.solver_config or {}),
                mechanism_text=str(dispatch_input.mechanism_text),
                simulation_identity=simulation_identity,
                parameter_overrides=(
                    dict(dispatch_input.parameter_overrides)
                    if isinstance(dispatch_input.parameter_overrides, Mapping)
                    else None
                ),
            )
            if isinstance(dispatch_input.intervention_schedule, Mapping):
                request_kwargs["intervention_schedule"] = dict(dispatch_input.intervention_schedule)
            plan_request = SimulationExecutionRequest(**request_kwargs).to_payload()
        cache_identity_payload = _cache_identity_payload(
            dispatch_input,
            simulation_identity=simulation_identity,
            existing_cache_identity_payload=None,
        )
        metadata = _metadata_payload(dispatch_input, existing_metadata=None)
        plan_payload = _new_simulation_plan_payload(
            plan_request,
            execution_mode="preview" if dispatch_input.fast_mode else "explicit",
            algebra_policy=dispatch_input.algebra_policy,
            cache_identity_payload=cache_identity_payload,
            cache_scope_payload=_cache_scope_payload(
                dispatch_input,
                existing_cache_scope_payload=None,
            ),
            metadata=metadata,
        )

    if isinstance(plan_payload, dict):
        plan_execution_request = (
            dict(execution_request)
            if isinstance(execution_request, dict)
            else _execution_request_payload_from_plan(plan_payload)
        )
        if isinstance(plan_execution_request, dict):
            plan = SimulationPlan.from_payload(plan_payload)
            plan_identity = plan.simulation_identity_payload()
            if plan_identity:
                simulation_identity = dict(plan_identity)
            cache_identity_payload = _cache_identity_payload(
                dispatch_input,
                simulation_identity=simulation_identity,
                existing_cache_identity_payload=plan.cache_identity_payload,
            )
            plan_payload = SimulationPlan.from_execution_request(
                plan_execution_request,
                execution_mode=plan.execution_mode,
                algebra_policy=dispatch_input.algebra_policy,
                cache_identity_payload=cache_identity_payload,
                cache_scope_payload=_cache_scope_payload(
                    dispatch_input,
                    existing_cache_scope_payload=plan.cache_scope_payload,
                ),
                metadata=_metadata_payload(dispatch_input, existing_metadata=plan.metadata),
                version=plan.version,
            ).to_payload()
        plan = SimulationPlan.from_payload(plan_payload)
        plan_identity = plan.simulation_identity_payload()
        if plan_identity:
            simulation_identity = dict(plan_identity)
        return BatchSetDispatchPlan(
            plan_payload=plan.to_payload(),
            execution_request=plan.to_execution_request().to_payload(),
            simulation_identity=simulation_identity,
        )

    return BatchSetDispatchPlan(
        plan_payload=None,
        execution_request=execution_request if isinstance(execution_request, dict) else None,
        simulation_identity=simulation_identity,
    )


def build_serial_batch_dispatch_plan(
    dispatch_input: SerialBatchDispatchInput,
) -> SerialBatchDispatchPlan:
    payload = dispatch_input.payload
    set_id = str(dispatch_input.set_id)
    full_dsl = str(payload.full_dsl)
    solver_config = dict(payload.solver_config)
    t_end = float(payload.t_end)
    cache_key = str(payload.cache_key)
    fast_mode = bool(payload.fast_mode)
    prepared_payload: Optional[Dict[str, Any]] = None
    execution_request: Optional[Dict[str, Any]] = None
    simulation_plan_by_set_id = dict(payload.simulation_plan_by_set_id)
    mechanism_text_by_set_id = dict(payload.mechanism_text_by_set_id)
    mechanism_signature_by_set_id = dict(payload.mechanism_signature_by_set_id)
    simulation_identity_by_set_id = dict(payload.simulation_identity_by_set_id)
    prepared_payloads = dict(payload.prepared_by_set_id)

    candidate = prepared_payloads.get(set_id) if bool(fast_mode) else None
    if isinstance(candidate, dict):
        prepared_payload = candidate

    candidate_plan = simulation_plan_by_set_id.get(set_id)
    plan_payload = simulation_plan_payload(candidate_plan)
    if plan_payload is None:
        raise ValueError(f"Missing simulation plan payload for set {set_id!r}.")
    candidate_request = execution_request_payload_from_plan(candidate_plan)
    if isinstance(candidate_request, dict):
        execution_request = candidate_request
        if not bool(fast_mode):
            execution_request = dict(execution_request)
            execution_request["prepared_payload"] = None
            if isinstance(plan_payload, dict):
                plan_payload = simulation_plan_payload_with_execution_request(
                    plan_payload,
                    execution_request,
                    algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
                )
        initials = dict(candidate_request.get("initials") or dispatch_input.initials)
        solver_config = dict(candidate_request.get("solver_config") or solver_config)
        request_t_span = candidate_request.get("t_span") or (0.0, t_end)
        try:
            t_end = float(request_t_span[1])
        except (TypeError, ValueError, IndexError):
            t_end = float(t_end)
    else:
        initials = dict(dispatch_input.initials)

    if isinstance(execution_request, dict):
        if execution_request.get("prepared_payload") is not None:
            mechanism_text_for_worker = str(execution_request.get("mechanism_text") or "")
        else:
            mechanism_text_for_worker = str(
                execution_request.get("mechanism_text")
                or mechanism_text_by_set_id.get(set_id, full_dsl)
            )
    else:
        mechanism_text_for_worker = mechanism_text_by_set_id.get(set_id, full_dsl)

    worker_signature = str(mechanism_signature_by_set_id.get(set_id) or "")
    cache_key_rewritten = False

    dispatch_plan = build_batch_set_dispatch_plan(
        BatchSetDispatchInput(
            set_id=set_id,
            set_name=str(dispatch_input.set_name),
            fast_mode=bool(fast_mode),
            t_end=float(t_end),
            solver_config=solver_config,
            cache_key=str(cache_key),
            scope_identity=dict(payload.scope_identity),
            queue_ids=[str(item) for item in dispatch_input.queue_ids],
            initials=dict(initials),
            mechanism_text=str(mechanism_text_for_worker),
            simulation_identity=simulation_identity_by_set_id.get(set_id),
            plan_payload=plan_payload,
            preview_batch_cache_token=payload.preview_batch_cache_token_by_set_id.get(set_id, ""),
            prepared_payload=prepared_payload if isinstance(prepared_payload, Mapping) else None,
            algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        )
    )
    plan_payload = dispatch_plan.plan_payload
    if isinstance(plan_payload, dict):
        plan_for_worker = simulation_plan_from_payloadish(plan_payload)
        if plan_for_worker is not None:
            plan_cache_key = plan_for_worker.cache_key()
            if plan_cache_key:
                cache_key = str(plan_cache_key)

    return SerialBatchDispatchPlan(
        plan_payload=plan_payload if isinstance(plan_payload, dict) else None,
        cache_key=str(cache_key),
        worker_signature=worker_signature,
        cache_key_rewritten=bool(cache_key_rewritten),
    )


def build_parallel_batch_task_plan(dispatch_input: ParallelBatchTaskInput) -> ParallelBatchTaskPlan:
    payload = dispatch_input.payload
    set_id = str(dispatch_input.set_id)
    simulation_identity_by_set_id = dict(payload.simulation_identity_by_set_id)
    plan_payload = dict(payload.simulation_plan_by_set_id).get(set_id)
    if plan_payload is None:
        raise ValueError(f"Missing simulation plan payload for set {set_id!r}.")
    dispatch_plan = build_batch_set_dispatch_plan(
        BatchSetDispatchInput(
            set_id=set_id,
            set_name=str(dispatch_input.set_name),
            fast_mode=bool(payload.fast_mode),
            t_end=float(payload.t_end),
            solver_config=dict(payload.solver_config),
            cache_key=str(payload.cache_key),
            scope_identity=dict(payload.scope_identity),
            queue_ids=[str(item) for item in dispatch_input.queue_ids],
            initials=dict(dispatch_input.initials),
            mechanism_text=dict(payload.mechanism_text_by_set_id).get(set_id, str(payload.full_dsl)),
            simulation_identity=simulation_identity_by_set_id.get(set_id),
            plan_payload=plan_payload,
            preview_batch_cache_token=payload.preview_batch_cache_token_by_set_id.get(set_id, ""),
            algebra_policy=SimulationAlgebraPolicy.BATCH_BEST_EFFORT,
        )
    )
    simulation_identity = (
        dict(dispatch_plan.simulation_identity)
        if dispatch_plan.simulation_identity
        else dict(simulation_identity_by_set_id.get(set_id) or {})
    )
    task: Dict[str, Any] = {
        "run_id": int(payload.run_id),
        "request_id": int(payload.request_id),
        "set_id": set_id,
        "set_name": str(dispatch_input.set_name),
        "include_mechanism_in_result_payload": bool(dispatch_input.include_mechanism_in_result_payload),
    }
    if isinstance(dispatch_plan.plan_payload, dict):
        task["simulation_plan"] = dict(dispatch_plan.plan_payload)
    return ParallelBatchTaskPlan(task=task, simulation_identity=simulation_identity)


def _cache_scope_payload(
    dispatch_input: BatchSetDispatchInput,
    *,
    existing_cache_scope_payload: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    existing = dict(existing_cache_scope_payload or {})
    scope_identity = (
        dict(dispatch_input.scope_identity)
        if dispatch_input.scope_identity
        else dict(existing.get("scope_identity") or {})
    )
    queue_ids = [str(item) for item in dispatch_input.queue_ids if str(item)]
    if not queue_ids:
        queue_ids = [str(item) for item in existing.get("queue_ids") or () if str(item)]
    return {
        "scope_identity": scope_identity,
        "queue_ids": queue_ids,
    }


def _cache_identity_payload(
    dispatch_input: BatchSetDispatchInput,
    *,
    simulation_identity: Mapping[str, Any],
    existing_cache_identity_payload: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    existing = dict(existing_cache_identity_payload or {})
    cache_key = str(dispatch_input.cache_key or existing.get("cache_key") or "")
    cache_identity_payload: Dict[str, Any] = {
        "cache_key": cache_key,
        "simulation_identity": dict(simulation_identity),
    }
    preview_token = str(
        dispatch_input.preview_batch_cache_token
        or existing.get("preview_batch_cache_token")
        or ""
    )
    if preview_token:
        cache_identity_payload["preview_batch_cache_token"] = preview_token
    return cache_identity_payload


def _metadata_payload(
    dispatch_input: BatchSetDispatchInput,
    *,
    existing_metadata: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    existing = dict(existing_metadata or {})
    metadata: Dict[str, Any] = {
        "set_id": str(dispatch_input.set_id),
        "set_name": str(dispatch_input.set_name),
        "fast_mode": bool(dispatch_input.fast_mode),
    }
    if isinstance(dispatch_input.contained_owner_identity, Mapping):
        metadata["contained_owner_identity"] = dict(dispatch_input.contained_owner_identity)
    elif isinstance(existing.get("contained_owner_identity"), Mapping):
        metadata["contained_owner_identity"] = dict(existing["contained_owner_identity"])
    return metadata
