from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Mapping

from kindred.gui.controllers.simulation_callback_freshness import SimulationCallbackFreshnessOwner
from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity
from kindred.core.simulation_failure import (
    coerce_simulation_failure,
    is_cancelled_failure,
    simulation_failure_detail_text,
    simulation_failure_user_message,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationErrorHandlingDependencies:
    freshness: SimulationCallbackFreshnessOwner
    completion_policy_preview_ownership: Callable[[], Any]
    stale_fast_error_replay_decision: Callable[..., Any]
    apply_completion_policy_state_patch: Callable[..., None]
    apply_lifecycle_effects: Callable[..., None]
    apply_runtime_effects: Callable[..., None]
    runtime_cancel_requested: Callable[..., Any]
    capture_terminal_failure_preview_replay_snapshot: Callable[..., Any]
    request_terminal_failure_preview_replay: Callable[..., None]
    request_pending_preview_replay: Callable[..., None]
    handle_current_preview_simulation_failure: Callable[..., None]


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
        callback_identity: SimulationCallbackIdentity,
    ) -> None:
        run_id = callback_identity.run_id
        fast_mode = callback_identity.fast_mode
        request_id = callback_identity.request_id
        batch_set_id = callback_identity.batch_set_id
        error_payload = coerce_simulation_failure(error_msg)
        error_text = simulation_failure_user_message(error_payload)
        error_detail_text = simulation_failure_detail_text(error_payload)
        cancelled = is_cancelled_failure(error_payload)
        context_resolution = self._batch_context_owner.context_for_callback_identity(callback_identity)
        ctx = context_resolution.context if context_resolution.matched else None
        if not isinstance(ctx, Mapping):
            raise ValueError(f"simulation error requires a current batch context ({context_resolution.reason}).")
        freshness = self._deps.freshness.assess_callback(callback_identity, context=ctx)
        if freshness.stale_run:
            logger.debug(
                "Ignoring stale simulation error (run_id=%s, active=%s): %s",
                run_id,
                freshness.active_run_id,
                error_text,
            )
            return
        latest_request_id = freshness.latest_request_id
        if freshness.dispatch_identity_stale:
            logger.debug(
                "Ignoring stale simulation error for mismatched runtime task identity "
                "(batch_set_id=%s, run_id=%s, request_id=%s): %s",
                str(batch_set_id or ""),
                run_id,
                request_id,
                error_text,
            )
            self._deps.freshness.mark_stale_dispatch_identity_callback_consumed(
                batch_set_id=batch_set_id,
                context=ctx,
            )
            return
        if freshness.runtime_input_stale:
            logger.debug(
                "Ignoring stale simulation error (batch_set_id=%s, current_global_epoch=%s): %s",
                str(batch_set_id or ""),
                freshness.current_global_epoch,
                error_text,
            )
            self._deps.freshness.mark_stale_runtime_input_callback_consumed(
                batch_set_id=batch_set_id,
                context=ctx,
            )
            return
        policy_context = (
            self._batch_context_owner.completion_policy_context(ctx)
            if isinstance(ctx, Mapping)
                else None
            )
        callback_preview_owner_epoch = freshness.callback_preview_owner_epoch
        is_superseded_fast_request = freshness.superseded_fast_request
        if is_superseded_fast_request:
            stale_fast_decision = self._completion_policy.resolve_superseded_fast_error(
                preview_ownership=self._deps.completion_policy_preview_ownership(),
                context=policy_context,
                request_id=int(request_id),
                preview_owner_epoch=callback_preview_owner_epoch,
            )
            replay_decision = self._deps.stale_fast_error_replay_decision()
            logger.debug(
                "Active fast error superseded (request_id=%s, latest=%s, run_id=%s, schedule_pending=%s): %s",
                request_id,
                latest_request_id,
                run_id,
                bool(replay_decision.pending_replay_queued),
                error_text,
            )

            self._deps.apply_completion_policy_state_patch(
                stale_fast_decision.state_patch,
                base_context=ctx if isinstance(ctx, Mapping) else None,
            )
            callback_context_matches_current = bool(
                isinstance(ctx, Mapping)
                and self._batch_context_owner.context_matches_current_run_identity(ctx)
            )
            if bool(
                stale_fast_decision.deactivate_context_immediately
                and callback_context_matches_current
            ):
                self._deps.apply_runtime_effects(
                    self._deps.runtime_cancel_requested(kind="soft_shutdown")
                )
            self._deps.apply_lifecycle_effects(
                self._lifecycle_effect_owner.superseded_fast_error_effects(
                    deactivate_context_immediately=bool(
                        stale_fast_decision.deactivate_context_immediately
                        and callback_context_matches_current
                    ),
                    reset_status_progress=bool(
                        stale_fast_decision.reset_status_progress
                        and callback_context_matches_current
                    ),
                )
            )
            self._deps.apply_runtime_effects(replay_decision.effects)
            return
        preview_failure_kind = str(error_payload.get("kind") or "").strip().lower()
        preview_failure_details = error_payload.get("details")
        preview_failure_source = (
            str(preview_failure_details.get("source") or "").strip().lower()
            if isinstance(preview_failure_details, Mapping)
            else ""
        )
        preview_failure_stage = (
            str(preview_failure_details.get("stage") or "").strip().lower()
            if isinstance(preview_failure_details, Mapping)
            else ""
        )
        status_only_preview_failure = (
            preview_failure_kind == "timeout"
            or preview_failure_kind.endswith("_timeout")
            or preview_failure_kind.startswith("simulation_containment")
            or preview_failure_source == "simulation_containment"
            or preview_failure_stage == "wegscheider_cyclicity"
        )
        if bool(fast_mode) and not cancelled and status_only_preview_failure:
            if isinstance(ctx, Mapping):
                preview_error_text = error_text
                if preview_failure_stage == "wegscheider_cyclicity":
                    preview_error_text = str(
                        error_payload.get("message") or "Unresolved Wegscheider cyclicity."
                    )
                self._deps.handle_current_preview_simulation_failure(
                    error_payload,
                    error_text=preview_error_text,
                    error_detail_text=error_detail_text,
                    context=ctx,
                )
                return
        logger.warning("Simulation error surfaced to UI: %s", error_text)

        if not cancelled and error_detail_text:
            logger.warning("%s", error_detail_text)
        replay_snapshot = (
            None
            if bool(cancelled)
            else self._deps.capture_terminal_failure_preview_replay_snapshot()
        )
        self._deps.apply_runtime_effects(
            self._deps.runtime_cancel_requested(
                kind="stop" if bool(cancelled) else "terminal_failure"
            )
        )
        if isinstance(ctx, Mapping):
            ctx = self._batch_context_owner.deactivate_if_active(ctx)
        self._deps.apply_lifecycle_effects(
            self._lifecycle_effect_owner.terminal_error_effects(
                cancelled=bool(cancelled),
                error_text=str(error_text),
                error_detail_text=str(error_detail_text or ""),
                fast_mode=bool(fast_mode),
            ),
            failed_run_context=ctx if isinstance(ctx, Mapping) else None,
        )
        if bool(cancelled):
            self._deps.request_pending_preview_replay(shutdown_requested=False)
        else:
            self._deps.request_terminal_failure_preview_replay(
                fast_mode=bool(fast_mode),
                replay_snapshot=replay_snapshot,
            )
