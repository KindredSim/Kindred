from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Mapping, Optional

from kindred.core.simulation_failure import (
    coerce_simulation_failure,
    is_cancelled_failure,
    simulation_failure_detail_text,
    simulation_failure_user_message,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationErrorHandlingDependencies:
    active_run_id: Callable[[], int]
    latest_request_id: Callable[[], int]
    current_global_epoch: Callable[[], int]
    active_batch_context_runtime_input_stale_for_set: Callable[..., bool]
    mark_stale_runtime_input_callback_consumed: Callable[..., None]
    effective_preview_owner_epoch_for_callback: Callable[..., int | None]
    missing_preview_owner_epoch_for_current_fast_owner: Callable[..., bool]
    preview_request_matches_current_owner_epoch: Callable[..., bool]
    completion_policy_preview_ownership: Callable[[], Any]
    completion_policy_pending_replay_state: Callable[[], Any]
    apply_completion_policy_state_patch: Callable[..., None]
    apply_lifecycle_effects: Callable[..., None]
    handle_current_preview_simulation_failure: Callable[..., None]
    has_deferred_preview_replay_intent: Callable[[], bool]


class SimulationErrorHandlingOwner:
    def __init__(
        self,
        *,
        ui: Any,
        batch_context_owner: Any,
        completion_policy: Any,
        lifecycle_effect_owner: Any,
        dependencies: SimulationErrorHandlingDependencies,
    ) -> None:
        self._ui = ui
        self._batch_context_owner = batch_context_owner
        self._completion_policy = completion_policy
        self._lifecycle_effect_owner = lifecycle_effect_owner
        self._deps = dependencies

    def handle_error(
        self,
        error_msg: object,
        *,
        run_id: Optional[int],
        fast_mode: Optional[bool],
        request_id: Optional[int],
        owner_epoch: Optional[int],
        batch_set: Optional[str],
        batch_set_id: Optional[str],
    ) -> None:
        error_payload = coerce_simulation_failure(error_msg)
        error_text = simulation_failure_user_message(error_payload)
        error_detail_text = simulation_failure_detail_text(error_payload)
        cancelled = is_cancelled_failure(error_payload)
        active_run_id = int(self._deps.active_run_id())
        if run_id is not None and int(run_id) != active_run_id:
            logger.debug(
                "Ignoring stale simulation error (run_id=%s, active=%s): %s",
                run_id,
                active_run_id,
                error_text,
            )
            return
        latest_request_id = int(self._deps.latest_request_id())
        ctx = None
        if batch_set is None or batch_set_id is None:
            hinted_set, hinted_set_id = self._batch_context_owner.current_queue_item()
            if batch_set is None:
                batch_set = hinted_set
            if batch_set_id is None:
                batch_set_id = hinted_set_id
        if batch_set_id is None and isinstance(batch_set, str):
            batch_set_id = self._ui.batch.batch_set_id_for_name(batch_set)
        if self._deps.active_batch_context_runtime_input_stale_for_set(
            batch_set_id=batch_set_id,
        ):
            logger.debug(
                "Ignoring stale simulation error (batch_set_id=%s, current_global_epoch=%s): %s",
                str(batch_set_id or ""),
                int(self._deps.current_global_epoch()),
                error_text,
            )
            self._deps.mark_stale_runtime_input_callback_consumed(batch_set_id=batch_set_id)
            return
        policy_context = self._batch_context_owner.completion_policy_context()
        callback_owner_epoch = self._deps.effective_preview_owner_epoch_for_callback(
            owner_epoch=owner_epoch,
            context=policy_context,
        )
        missing_owner_epoch = self._deps.missing_preview_owner_epoch_for_current_fast_owner(
            fast_mode=fast_mode,
            request_id=request_id,
            owner_epoch=callback_owner_epoch,
            latest_request_id=latest_request_id,
        )
        is_superseded_fast_request = bool(
            fast_mode
            and request_id is not None
            and (
                bool(missing_owner_epoch)
                or (not self._deps.preview_request_matches_current_owner_epoch(request_id, callback_owner_epoch))
            )
        )
        if is_superseded_fast_request:
            stale_fast_decision = self._completion_policy.resolve_superseded_fast_error(
                preview_ownership=self._deps.completion_policy_preview_ownership(),
                context=policy_context,
                request_id=int(request_id),
                preview_owner_epoch=callback_owner_epoch,
                pending_replay=self._deps.completion_policy_pending_replay_state(),
            )
            logger.debug(
                "Active fast error superseded (request_id=%s, latest=%s, run_id=%s, schedule_pending=%s): %s",
                request_id,
                latest_request_id,
                run_id,
                bool(stale_fast_decision.schedule_pending_preview_run),
                error_text,
            )

            self._deps.apply_completion_policy_state_patch(
                stale_fast_decision.state_patch,
            )
            self._deps.apply_lifecycle_effects(
                self._lifecycle_effect_owner.superseded_fast_error_effects(
                    deactivate_context_immediately=bool(stale_fast_decision.deactivate_context_immediately),
                    schedule_pending_preview_run=bool(stale_fast_decision.schedule_pending_preview_run),
                    reset_status_progress=bool(stale_fast_decision.reset_status_progress),
                )
            )
            return
        preview_failure_kind = str(error_payload.get("kind") or "").strip().lower()
        preview_failure_details = error_payload.get("details")
        preview_failure_source = (
            str(preview_failure_details.get("source") or "").strip().lower()
            if isinstance(preview_failure_details, Mapping)
            else ""
        )
        status_only_preview_failure = (
            preview_failure_kind == "timeout"
            or preview_failure_kind.endswith("_timeout")
            or preview_failure_kind.startswith("simulation_containment")
            or preview_failure_source == "simulation_containment"
        )
        if bool(fast_mode) and not cancelled and status_only_preview_failure:
            self._deps.handle_current_preview_simulation_failure(
                error_payload,
                error_text=error_text,
                error_detail_text=error_detail_text,
                context=ctx if isinstance(ctx, Mapping) else None,
            )
            return
        logger.warning("Simulation error surfaced to UI: %s", error_text)

        ctx = self._batch_context_owner.deactivate_if_active(ctx)

        if not cancelled and error_detail_text:
            logger.warning("%s", error_detail_text)
        if cancelled and self._deps.has_deferred_preview_replay_intent():
            logger.debug("Resuming pending slider update after cancellation")
        self._deps.apply_lifecycle_effects(
            self._lifecycle_effect_owner.terminal_error_effects(
                cancelled=bool(cancelled),
                error_text=str(error_text),
                error_detail_text=str(error_detail_text or ""),
                fast_mode=bool(fast_mode),
                has_deferred_preview_replay=bool(self._deps.has_deferred_preview_replay_intent()),
            ),
            failed_run_context=ctx if isinstance(ctx, Mapping) else None,
        )
