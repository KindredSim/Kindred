from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional

from kindred.core.batch_containment import BatchCompletionRecord, BatchLaneOutcome
from kindred.core.simulation_failure import build_simulation_failure, coerce_simulation_failure


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParallelBatchOutcomeResolution:
    set_id: str
    set_name: str
    owner_epoch: object | None
    expected_owner_epoch: object | None
    stale: bool
    owner_epoch_mismatch: bool
    failed: bool
    payload: Dict[str, Any] | None
    error_payload: Dict[str, Any] | None


@dataclass(frozen=True)
class ParallelBatchOutcomeDependencies:
    active_batch_context_runtime_input_stale_for_set: Callable[..., bool]
    mark_stale_runtime_input_callback_consumed: Callable[..., None]
    record_nonfatal_exception: Callable[..., None]
    invalidate_preserved_pending_init_results_after_failed_run: Callable[..., None]
    finalize_scoped_batch_success_subset: Callable[..., None]
    cleanup_parallel_batch_lane_pool_after_run: Callable[..., None]
    show_scoped_batch_failure_summary: Callable[..., None]
    apply_explicit_failure_pending_replay_policy: Callable[..., None]
    reset_parallel_batch_run_and_shutdown_lane_pool: Callable[..., None]
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
                "owner_epoch": completion_record.expected_owner_epoch,
                "generation": completion_record.generation,
            }
        )
    meta["set_name"] = str(meta.get("set_name") or sid)
    set_name = str(meta["set_name"])
    owner_epoch = meta.get("preview_owner_epoch")
    if owner_epoch is None:
        owner_epoch = meta.get("owner_epoch")
    expected_owner_epoch = meta.get("owner_epoch")
    owner_epoch_mismatch = False
    if expected_owner_epoch is not None:
        try:
            owner_epoch_mismatch = int(outcome.owner_epoch) != int(expected_owner_epoch)
        except Exception:
            owner_epoch_mismatch = True
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
        or owner_epoch_mismatch
    )
    if stale:
        details = {
            "expected_run_id": int(expected_run_id),
            "expected_request_id": int(expected_request_id),
            "expected_set_id": sid,
            "expected_owner_epoch": expected_owner_epoch,
            "actual_run_id": int(outcome.run_id),
            "actual_request_id": int(outcome.request_id),
            "actual_set_id": str(outcome.set_id or ""),
            "actual_owner_epoch": int(outcome.owner_epoch),
        }
        if runtime_session_stale is not None:
            details["runtime_session_stale"] = dict(runtime_session_stale)
        if missing_identity_metadata:
            details["missing_identity_metadata"] = tuple(missing_identity_metadata)
        return ParallelBatchOutcomeResolution(
            set_id=sid,
            set_name=set_name,
            owner_epoch=owner_epoch,
            expected_owner_epoch=expected_owner_epoch,
            stale=True,
            owner_epoch_mismatch=bool(owner_epoch_mismatch),
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
        owner_epoch=owner_epoch,
        expected_owner_epoch=expected_owner_epoch,
        stale=False,
        owner_epoch_mismatch=False,
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
        if completion_state is None or not completion_state.active or not completion_state.parallel:
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
        self._deps.invalidate_preserved_pending_init_results_after_failed_run(ctx=ctx)

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
        self._deps.finalize_scoped_batch_success_subset(ctx)
        self._deps.cleanup_parallel_batch_lane_pool_after_run(
            keep_lane_pool_alive=False,
            clear_pending_plot_updates=False,
        )
        self._deps.set_simulation_running(False)
        self._deps.set_slider_simulation_active(False)
        self._ui.slider.set_slider_triggered_simulation(False)
        self._ui.run_ui.set_sim_progress_value(100)
        self._ui.run_ui.set_run_button_enabled(True)
        self._ui.run_ui.set_stop_button_enabled(False)
        failed_count = self._batch_context_owner.scoped_failure_cache_state().failed_count
        self._ui.run_ui.set_status_text(f"Batch completed with {failed_count} failed set(s)")
        summary = self._batch_context_owner.completion_summary(ctx)
        self._deps.show_scoped_batch_failure_summary(
            failed_set_ids=summary.failed_set_ids,
            failed_errors=summary.failed_errors,
        )
        self._deps.apply_explicit_failure_pending_replay_policy(fast_mode=False)
        return True

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
        callback_context = getattr(callback_identity, "callback_context", None)
        callback_context = callback_context if isinstance(callback_context, Mapping) else None

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
                    f"set_id={sid} owner_epoch={resolution.expected_owner_epoch}; "
                    f"got run_id={int(outcome.run_id)} request_id={int(outcome.request_id)} "
                    f"set_id={str(outcome.set_id or '')} owner_epoch={int(outcome.owner_epoch)})"
                ),
                RuntimeError("stale batch lane outcome"),
            )
            if isinstance(runtime_session_stale, Mapping):
                return True
            run_request_set_match = bool(
                expected_run_id is not None
                and expected_request_id is not None
                and int(outcome.run_id) == int(expected_run_id)
                and int(outcome.request_id) == int(expected_request_id)
                and str(outcome.set_id or "") == sid
            )
            if bool(getattr(callback_identity, "fast_mode", False)) and resolution.owner_epoch_mismatch and run_request_set_match:
                return True
            self._deps.reset_parallel_batch_run_and_shutdown_lane_pool()
            self._deps.set_simulation_running(False)
            self._deps.set_slider_simulation_active(False)
            self._ui.slider.set_slider_triggered_simulation(False)
            self._ui.run_ui.set_run_button_enabled(True)
            self._ui.run_ui.set_stop_button_enabled(False)
            return False

        self._batch_parallel.discard_request(sid)

        if callback_identity is None or callback_context is None:
            self._deps.record_nonfatal_exception(
                (
                    "Missing callback identity for active parallel batch outcome "
                    f"(run_id={int(getattr(completion_record, 'run_id', outcome.run_id))} "
                    f"request_id={int(getattr(completion_record, 'request_id', outcome.request_id))} "
                    f"set_id={sid} source={str(source)})"
                ),
                RuntimeError("missing parallel batch callback identity"),
            )
            self._deps.reset_parallel_batch_run_and_shutdown_lane_pool()
            self._deps.set_simulation_running(False)
            self._deps.set_slider_simulation_active(False)
            self._ui.slider.set_slider_triggered_simulation(False)
            self._ui.run_ui.set_sim_progress_value(0)
            self._ui.run_ui.set_run_button_enabled(True)
            self._ui.run_ui.set_stop_button_enabled(False)
            self._ui.run_ui.set_status_text("Batch simulation failed")
            return False

        if self._deps.active_batch_context_runtime_input_stale_for_set(
            batch_set_id=sid,
            context=callback_context,
        ):
            self._deps.mark_stale_runtime_input_callback_consumed(
                batch_set_id=sid,
                context=callback_context,
            )
            return True
        if resolution.failed:
            error_payload = dict(resolution.error_payload or {})
            if self.handle_scoped_failure(
                set_id=sid,
                set_name=set_name,
                error_payload=error_payload,
            ):
                return True
            self._error_handling_owner.handle_error(
                error_payload,
                callback_identity=callback_identity,
            )
            self._deps.reset_parallel_batch_run_and_shutdown_lane_pool()
            return False

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
        completion_policy_context = self._batch_context_owner.completion_policy_context(callback_context)
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
            self._deps.reset_parallel_batch_run_and_shutdown_lane_pool()
            return False
        return True
