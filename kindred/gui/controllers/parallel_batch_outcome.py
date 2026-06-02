from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional

from kindred.core.batch_containment import BatchCompletionRecord, BatchLaneOutcome
from kindred.core.simulation_failure import build_simulation_failure, coerce_simulation_failure
from kindred.gui.controllers.simulation_callback_freshness import SimulationCallbackFreshnessOwner
from kindred.gui.controllers.simulation_runtime_backend import RuntimeCompletionDecision
from kindred.gui.ports import DisplayTransitionOutcome


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
    freshness: SimulationCallbackFreshnessOwner
    record_nonfatal_exception: Callable[..., None]
    finalize_scoped_batch_success_subset: Callable[..., DisplayTransitionOutcome | None]
    runtime_display_completed: Callable[..., None]
    show_scoped_batch_failure_summary: Callable[..., None]
    request_terminal_failure_preview_replay: Callable[..., None]
    cancel_runtime_for_parallel_outcome_reset: Callable[..., None]
    set_simulation_running: Callable[[bool], None]
    set_slider_simulation_active: Callable[[bool], None]


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
    def __init__(
        self,
        *,
        ui: Any,
        batch_parallel: Any,
        batch_context_owner: Any,
        batch_cache: Any,
        completion_callback_owner: Any,
        error_handling_owner: Any,
        dependencies: ParallelBatchOutcomeDependencies,
    ) -> None:
        self._ui = ui
        self._batch_parallel = batch_parallel
        self._batch_context_owner = batch_context_owner
        self._batch_cache = batch_cache
        self._completion_callback_owner = completion_callback_owner
        self._error_handling_owner = error_handling_owner
        self._deps = dependencies

    def handle_scoped_failure(
        self,
        *,
        set_id: str,
        set_name: str,
        error_payload: Mapping[str, Any],
    ) -> bool:
        completion_state = self._batch_context_owner.completion_state()
        if (
            completion_state is None
            or not completion_state.active
            or not (completion_state.runtime_task_queue or completion_state.parallel)
        ):
            return False
        if completion_state.fast_mode:
            return False
        total = max(1, int(completion_state.total or len(completion_state.queue_ids) or 1))
        if total <= 1:
            return False

        sid = str(set_id or "")
        failure = coerce_simulation_failure(error_payload)
        transition = self._batch_context_owner.record_scoped_failure(set_id=sid, failure=failure)
        ctx = transition.context
        cache_state = self._batch_context_owner.scoped_failure_cache_state(ctx)
        self._batch_cache.record_explicit_scoped_failure_cache_state(
            cache_key=str(cache_state.cache_key),
            explicit_cache_valid_set_ids=cache_state.explicit_cache_valid_set_ids,
            explicit_cache_invalidated_set_ids=cache_state.explicit_cache_invalidated_set_ids,
        )
        completed_count = int(transition.completed_count)
        if completed_count < total:
            if total > 1:
                self._ui.run_ui.set_sim_progress_value(
                    max(0, min(100, int((completed_count / float(total)) * 100.0)))
                )
            label = str(set_name or sid or "set")
            self._ui.run_ui.set_status_text(f"Failed {label} ({completed_count}/{total})")
            return True

        ctx = self._batch_context_owner.deactivate()
        display_transition = self._deps.finalize_scoped_batch_success_subset(ctx)
        if display_transition is not None and not isinstance(display_transition, DisplayTransitionOutcome):
            raise TypeError("Scoped batch display finalization must return DisplayTransitionOutcome or None")
        self._deps.runtime_display_completed(kind="scoped_failure")
        self._deps.set_simulation_running(False)
        self._deps.set_slider_simulation_active(False)
        self._ui.slider.set_slider_triggered_simulation(False)
        self._ui.run_ui.set_sim_progress_value(100)
        self._ui.run_ui.set_run_button_enabled(True)
        self._ui.run_ui.set_stop_button_enabled(False)
        summary = self._batch_context_owner.completion_summary(ctx)
        self._deps.show_scoped_batch_failure_summary(
            failed_set_ids=summary.failed_set_ids,
            failed_errors=summary.failed_errors,
        )
        self._deps.request_terminal_failure_preview_replay(fast_mode=False)
        return True

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
        callback_identity = meta.get("callback_identity")
        resolved_run_context: Mapping[str, Any] | None = None
        if callback_identity is not None and hasattr(self._batch_context_owner, "context_for_callback_identity"):
            context_resolution = self._batch_context_owner.context_for_callback_identity(callback_identity)
            if bool(getattr(context_resolution, "matched", False)):
                candidate_context = getattr(context_resolution, "context", None)
                if isinstance(candidate_context, Mapping):
                    resolved_run_context = candidate_context

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
            if isinstance(runtime_session_stale, Mapping):
                return RuntimeCompletionDecision.ignored_stale(consumed=False)
            self._deps.cancel_runtime_for_parallel_outcome_reset()
            self._deps.set_simulation_running(False)
            self._deps.set_slider_simulation_active(False)
            self._ui.slider.set_slider_triggered_simulation(False)
            self._ui.run_ui.set_run_button_enabled(True)
            self._ui.run_ui.set_stop_button_enabled(False)
            return RuntimeCompletionDecision.ignored_stale(consumed=False)

        if callback_identity is None or resolved_run_context is None:
            self._deps.record_nonfatal_exception(
                (
                    "Missing callback identity for active parallel batch outcome "
                    f"(run_id={int(getattr(completion_record, 'run_id', outcome.run_id))} "
                    f"request_id={int(getattr(completion_record, 'request_id', outcome.request_id))} "
                    f"set_id={sid} source={str(source)})"
                ),
                RuntimeError("missing parallel batch callback identity"),
            )
            self._deps.cancel_runtime_for_parallel_outcome_reset()
            self._deps.set_simulation_running(False)
            self._deps.set_slider_simulation_active(False)
            self._ui.slider.set_slider_triggered_simulation(False)
            self._ui.run_ui.set_sim_progress_value(0)
            self._ui.run_ui.set_run_button_enabled(True)
            self._ui.run_ui.set_stop_button_enabled(False)
            self._ui.run_ui.set_status_text("Batch simulation failed")
            return RuntimeCompletionDecision.terminal_failure(
                "Missing callback identity for active parallel batch outcome."
            )

        freshness = self._deps.freshness.assess_callback(callback_identity, context=resolved_run_context)
        if freshness.stale_run and int(freshness.active_run_id) > 0:
            return RuntimeCompletionDecision.ignored_stale(consumed=False)
        if freshness.dispatch_identity_stale:
            self._deps.freshness.mark_stale_dispatch_identity_callback_consumed(
                batch_set_id=sid,
                context=resolved_run_context,
            )
            return RuntimeCompletionDecision.ignored_stale()
        if freshness.runtime_input_stale:
            self._deps.freshness.mark_stale_runtime_input_callback_consumed(
                batch_set_id=sid,
                context=resolved_run_context,
            )
            return RuntimeCompletionDecision.ignored_stale()

        if resolution.failed:
            error_payload = dict(resolution.error_payload or {})
            if self.handle_scoped_failure(
                set_id=sid,
                set_name=set_name,
                error_payload=error_payload,
            ):
                return RuntimeCompletionDecision.accepted_current()
            self._error_handling_owner.handle_error(
                error_payload,
                callback_identity=callback_identity,
            )
            self._deps.cancel_runtime_for_parallel_outcome_reset()
            return RuntimeCompletionDecision.terminal_failure(
                "Runtime completion reported a terminal failure."
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
        completion_policy_context = self._batch_context_owner.completion_policy_context(resolved_run_context)
        try:
            self._completion_callback_owner.handle_completion(
                resolution.payload,
                debug_batch_parallel=bool(debug_batch_parallel),
                callback_identity=callback_identity,
                policy_context=completion_policy_context,
            )
        except Exception as exc:
            self._deps.record_nonfatal_exception(
                f"Unhandled exception while handling completed batch lane outcome (set_id={sid}, source={str(source)})",
                exc,
            )
            try:
                self._error_handling_owner.handle_error(
                    f"Simulation failed:\n\n{exc}",
                    callback_identity=callback_identity,
                )
            except Exception as ui_exc:
                self._deps.record_nonfatal_exception(
                    "Failed to surface simulation-complete handling failure to UI",
                    ui_exc,
                )
            self._deps.cancel_runtime_for_parallel_outcome_reset()
            return RuntimeCompletionDecision.terminal_failure(str(exc))
        return RuntimeCompletionDecision.accepted_current()

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
            return RuntimeCompletionDecision.terminal_failure(str(exc))
