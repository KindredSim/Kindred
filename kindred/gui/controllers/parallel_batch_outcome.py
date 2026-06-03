from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Any, Dict, Mapping, Optional

from kindred.core.batch_containment import BatchCompletionRecord, BatchLaneOutcome
from kindred.core.simulation_failure import build_simulation_failure
from kindred.gui.controllers.simulation_runtime_backend import RuntimeCompletionDecision
from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity
from kindred.gui.controllers.simulation_failure_policy import SimulationFailureInput


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParallelBatchOutcomeResolution:
    set_id: str
    set_name: str
    preview_owner_epoch: object | None
    stale: bool
    failed: bool
    payload: Dict[str, Any] | None
    error_payload: Dict[str, Any] | None


@dataclass(frozen=True)
class ParallelBatchOutcomeDependencies:
    record_nonfatal_exception: Any


def resolve_parallel_batch_outcome(
    *,
    set_id: str,
    outcome: BatchLaneOutcome,
    metadata: Mapping[str, Any],
    completion_record: Optional[BatchCompletionRecord] = None,
) -> ParallelBatchOutcomeResolution:
    sid = str(set_id or "")
    meta = dict(metadata or {})
    if completion_record is not None:
        meta.update(dict(completion_record.request_metadata or {}))
        meta.update(
            {
                "set_name": completion_record.set_name,
                "run_id": completion_record.run_id,
                "request_id": completion_record.request_id,
                "preview_owner_epoch": completion_record.preview_owner_epoch,
                "generation": completion_record.generation,
            }
        )
    meta["set_name"] = str(meta.get("set_name") or sid)
    set_name = str(meta["set_name"])
    preview_owner_epoch = meta.get("preview_owner_epoch")
    missing_identity_metadata: list[str] = []
    if meta.get("run_id") is None:
        missing_identity_metadata.append("run_id")
        expected_run_id = -1
    else:
        expected_run_id = int(meta["run_id"])
    if meta.get("request_id") is None:
        missing_identity_metadata.append("request_id")
        expected_request_id = -1
    else:
        expected_request_id = int(meta["request_id"])
    runtime_session_stale = meta.get("runtime_session_stale")
    runtime_session_stale = runtime_session_stale if isinstance(runtime_session_stale, Mapping) else None

    stale = (
        bool(missing_identity_metadata)
        or runtime_session_stale is not None
        or int(outcome.run_id) != int(expected_run_id)
        or int(outcome.request_id) != int(expected_request_id)
        or str(outcome.set_id or "") != sid
    )
    if stale:
        details = {
            "expected_run_id": int(expected_run_id),
            "expected_request_id": int(expected_request_id),
            "expected_set_id": sid,
            "expected_preview_owner_epoch": preview_owner_epoch,
            "actual_run_id": int(outcome.run_id),
            "actual_request_id": int(outcome.request_id),
            "actual_set_id": str(outcome.set_id or ""),
            "lane_owner_epoch": int(outcome.lane_owner_epoch),
        }
        if runtime_session_stale is not None:
            details["runtime_session_stale"] = dict(runtime_session_stale)
        if missing_identity_metadata:
            details["missing_identity_metadata"] = tuple(missing_identity_metadata)
        return ParallelBatchOutcomeResolution(
            set_id=sid,
            set_name=set_name,
            preview_owner_epoch=preview_owner_epoch,
            stale=True,
            failed=True,
            payload=None,
            error_payload=build_simulation_failure(
                "stale_batch_lane_outcome",
                "Rejected stale batch lane outcome.",
                details=details,
            ),
        )

    payload = dict(outcome.payload or {}) if outcome.payload is not None else None
    failed = (not bool(outcome.success)) or (
        isinstance(payload, dict)
        and payload.get("success") is False
        and isinstance(payload.get("error"), dict)
    )
    error_payload = None
    if failed:
        error_payload = (
            dict(payload["error"])
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict)
            else dict(outcome.failure or {})
        )
    return ParallelBatchOutcomeResolution(
        set_id=sid,
        set_name=set_name,
        preview_owner_epoch=preview_owner_epoch,
        stale=False,
        failed=bool(failed),
        payload=payload,
        error_payload=error_payload,
    )


class ParallelBatchOutcomeOwner:
    """Adapter for runtime-lane batch outcomes.

    This owner adapts backend completion records and publishes successful
    completions.  Runtime/preview failure policy is delegated to the canonical
    SimulationFailurePolicyOwner and returned as a typed RuntimeCompletionDecision.
    """

    def __init__(
        self,
        *,
        batch_context_owner: Any,
        completion_callback_owner: Any,
        failure_policy_owner: Any,
        dependencies: ParallelBatchOutcomeDependencies,
    ) -> None:
        self._batch_context_owner = batch_context_owner
        self._completion_callback_owner = completion_callback_owner
        self._failure_policy_owner = failure_policy_owner
        self._deps = dependencies

    def _callback_identity_from_metadata(self, meta: Mapping[str, Any]) -> SimulationCallbackIdentity | None:
        callback_identity = meta.get("callback_identity")
        return callback_identity if isinstance(callback_identity, SimulationCallbackIdentity) else None

    def _context_for_callback_identity(
        self,
        callback_identity: SimulationCallbackIdentity | None,
    ) -> Mapping[str, Any] | None:
        if callback_identity is None or not hasattr(self._batch_context_owner, "context_for_callback_identity"):
            return None
        context_resolution = self._batch_context_owner.context_for_callback_identity(callback_identity)
        if bool(getattr(context_resolution, "matched", False)) and isinstance(
            getattr(context_resolution, "context", None), Mapping
        ):
            return getattr(context_resolution, "context")
        return None

    def _consume_outcome_decision(
        self,
        *,
        set_id: str,
        outcome: BatchLaneOutcome,
        source: str,
        completed_ts: Optional[float] = None,
        completion_record: Optional[BatchCompletionRecord] = None,
        debug_batch_parallel: bool = False,
    ) -> RuntimeCompletionDecision:
        sid = str(set_id or "")
        meta: Dict[str, Any] = {}
        if completion_record is not None:
            meta.update(dict(completion_record.request_metadata or {}))
        resolution = resolve_parallel_batch_outcome(
            set_id=sid,
            outcome=outcome,
            metadata=meta,
            completion_record=completion_record,
        )
        set_name = resolution.set_name
        callback_identity = self._callback_identity_from_metadata(meta)
        resolved_run_context = self._context_for_callback_identity(callback_identity)

        if resolution.stale:
            runtime_session_stale = meta.get("runtime_session_stale")
            expected_run_id = meta.get("run_id") if meta.get("run_id") is not None else getattr(completion_record, "run_id", None)
            expected_request_id = (
                meta.get("request_id")
                if meta.get("request_id") is not None
                else getattr(completion_record, "request_id", None)
            )
            self._deps.record_nonfatal_exception(
                (
                    "Rejected stale batch lane outcome "
                    f"(expected run_id={expected_run_id if expected_run_id is not None else 'missing'} "
                    f"request_id={expected_request_id if expected_request_id is not None else 'missing'} "
                    f"set_id={sid} preview_owner_epoch={resolution.preview_owner_epoch}; "
                    f"got run_id={int(outcome.run_id)} request_id={int(outcome.request_id)} "
                    f"set_id={str(outcome.set_id or '')} lane_owner_epoch={int(outcome.lane_owner_epoch)})"
                ),
                RuntimeError("stale batch lane outcome"),
            )
            decision = self._failure_policy_owner.resolve_failure(
                SimulationFailureInput(
                    error_payload=resolution.error_payload or build_simulation_failure(
                        "stale_batch_lane_outcome",
                        "Rejected stale batch lane outcome.",
                    ),
                    origin="runtime_lane",
                    callback_identity=callback_identity,
                    set_id=sid,
                    set_name=set_name,
                    source=str(source),
                    context=resolved_run_context,
                    active_fast_mode=self._active_completion_fast_mode(),
                    stale_runtime_lane_outcome=True,
                    runtime_session_stale=runtime_session_stale if isinstance(runtime_session_stale, Mapping) else None,
                )
            )
            return RuntimeCompletionDecision.from_failure_decision(decision)

        if callback_identity is None:
            self._deps.record_nonfatal_exception(
                (
                    "Missing callback identity for active parallel batch outcome "
                    f"(run_id={int(getattr(completion_record, 'run_id', outcome.run_id))} "
                    f"request_id={int(getattr(completion_record, 'request_id', outcome.request_id))} "
                    f"set_id={sid} source={str(source)})"
                ),
                RuntimeError("missing parallel batch callback identity"),
            )
            decision = self._failure_policy_owner.resolve_failure(
                SimulationFailureInput(
                    error_payload=build_simulation_failure(
                        "runtime_missing_callback_identity",
                        "Missing callback identity for active parallel batch outcome.",
                    ),
                    origin="runtime_lane",
                    callback_identity=None,
                    set_id=sid,
                    set_name=set_name,
                    source=str(source),
                    active_fast_mode=self._active_completion_fast_mode(),
                )
            )
            return RuntimeCompletionDecision.from_failure_decision(decision)

        if resolution.failed:
            decision = self._failure_policy_owner.resolve_failure(
                SimulationFailureInput(
                    error_payload=dict(resolution.error_payload or {}),
                    origin="runtime_lane",
                    callback_identity=callback_identity,
                    set_id=sid,
                    set_name=set_name,
                    source=str(source),
                    context=resolved_run_context,
                    active_fast_mode=self._active_completion_fast_mode(),
                    allow_scoped_failure=True,
                )
            )
            return RuntimeCompletionDecision.from_failure_decision(decision)

        policy_context = (
            self._batch_context_owner.completion_policy_context(resolved_run_context)
            if isinstance(resolved_run_context, Mapping)
            else None
        )
        if bool(debug_batch_parallel):
            logger.info(
                "BATCH_PAR completion received run_id=%s request_id=%s set_id=%s source=%s completed_at=%.6f received_at=%.6f",
                int(getattr(callback_identity, "run_id", outcome.run_id)),
                int(getattr(callback_identity, "request_id", outcome.request_id)),
                sid,
                str(source),
                float(completed_ts if completed_ts is not None else -1.0),
                float(perf_counter()),
            )
        try:
            self._completion_callback_owner.handle_completion(
                resolution.payload,
                debug_batch_parallel=bool(debug_batch_parallel),
                callback_identity=callback_identity,
                policy_context=policy_context,
            )
        except Exception as exc:
            self._deps.record_nonfatal_exception(
                f"Unhandled exception while handling completed batch lane outcome (set_id={sid}, source={str(source)})",
                exc,
            )
            decision = self._failure_policy_owner.resolve_failure(
                SimulationFailureInput(
                    error_payload=build_simulation_failure(
                        "runtime_completion_exception",
                        str(exc),
                    ),
                    origin="runtime_completion_exception",
                    callback_identity=callback_identity,
                    set_id=sid,
                    set_name=set_name,
                    source=str(source),
                    context=resolved_run_context,
                    active_fast_mode=self._active_completion_fast_mode(),
                )
            )
            return RuntimeCompletionDecision.from_failure_decision(decision)
        return RuntimeCompletionDecision.accepted_current()

    def _active_completion_fast_mode(self) -> bool:
        try:
            state = self._batch_context_owner.completion_state()
        except Exception:
            return False
        return bool(getattr(state, "fast_mode", False))

    def consume_outcome(
        self,
        *,
        set_id: str,
        outcome: BatchLaneOutcome,
        source: str,
        completed_ts: Optional[float] = None,
        completion_record: Optional[BatchCompletionRecord] = None,
        debug_batch_parallel: bool = False,
    ) -> bool:
        return bool(
            self._consume_outcome_decision(
                set_id=set_id,
                outcome=outcome,
                source=source,
                completed_ts=completed_ts,
                completion_record=completion_record,
                debug_batch_parallel=debug_batch_parallel,
            ).accepted
        )

    def consume_runtime_completion(self, event: Any) -> RuntimeCompletionDecision:
        try:
            return self._consume_outcome_decision(
                set_id=str(getattr(event, "set_id", "") or ""),
                outcome=getattr(event, "outcome", None),
                source=str(getattr(event, "source", "") or ""),
                completed_ts=float(getattr(event, "completed_ts", 0.0) or 0.0),
                completion_record=getattr(event, "record", None),
                debug_batch_parallel=False,
            )
        except Exception as exc:
            decision = self._failure_policy_owner.resolve_failure(
                SimulationFailureInput(
                    error_payload=build_simulation_failure("runtime_completion_consumer_exception", str(exc)),
                    origin="runtime_completion_consumer_exception",
                    active_fast_mode=self._active_completion_fast_mode(),
                )
            )
            return RuntimeCompletionDecision.from_failure_decision(decision)
