from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Mapping

from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity
from kindred.gui.controllers.simulation_completion_policy import CompletionPolicyContext
from kindred.gui.controllers.simulation_completion_publication import CompletionCallbackState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationCompletionCallbackDependencies:
    active_run_id: Callable[[], int]
    shutdown_requested: Callable[[], bool]
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


class SimulationCompletionCallbackOwner:
    def __init__(
        self,
        *,
        ui: Any,
        batch_context_owner: Any,
        completion_policy: Any,
        lifecycle_effect_owner: Any,
        publication_owner: Any,
        dependencies: SimulationCompletionCallbackDependencies,
    ) -> None:
        self._ui = ui
        self._batch_context_owner = batch_context_owner
        self._completion_policy = completion_policy
        self._lifecycle_effect_owner = lifecycle_effect_owner
        self._publication_owner = publication_owner
        self._deps = dependencies

    def handle_completion(
        self,
        result: Mapping[str, Any],
        *,
        debug_batch_parallel: bool,
        callback_identity: SimulationCallbackIdentity,
        policy_context: CompletionPolicyContext | None = None,
    ) -> None:
        run_id = callback_identity.run_id
        fast_mode = callback_identity.fast_mode
        request_id = callback_identity.request_id
        owner_epoch = callback_identity.owner_epoch
        batch_set = callback_identity.batch_set
        batch_set_id = callback_identity.batch_set_id
        cache_key = callback_identity.cache_key
        active_run_id = int(self._deps.active_run_id())
        shutdown_requested = bool(self._deps.shutdown_requested())
        if int(run_id) != active_run_id:
            logger.debug(
                "Ignoring stale simulation completion (run_id=%s, active=%s)",
                run_id,
                active_run_id,
            )
            return

        ctx: Mapping[str, Any] | None = (
            callback_identity.callback_context
            if isinstance(callback_identity.callback_context, Mapping)
            else None
        )
        if not isinstance(ctx, Mapping):
            raise ValueError("simulation completion requires callback_identity.callback_context.")
        if isinstance(ctx, Mapping) and self._deps.active_batch_context_runtime_input_stale_for_set(
            batch_set_id=batch_set_id,
            context=ctx,
        ):
            logger.debug(
                "Ignoring stale simulation completion (batch_set_id=%s, current_global_epoch=%s)",
                str(batch_set_id or ""),
                int(self._deps.current_global_epoch()),
            )
            self._deps.mark_stale_runtime_input_callback_consumed(
                batch_set_id=batch_set_id,
                context=ctx,
            )
            return

        if policy_context is None:
            policy_context = (
                self._batch_context_owner.completion_policy_context(ctx)
                if isinstance(ctx, Mapping)
                else None
            )
        latest_request_id = int(self._deps.latest_request_id())
        callback_owner_epoch = self._deps.effective_preview_owner_epoch_for_callback(
            owner_epoch=owner_epoch,
        )
        missing_owner_epoch = self._deps.missing_preview_owner_epoch_for_current_fast_owner(
            fast_mode=fast_mode,
            request_id=request_id,
            owner_epoch=callback_owner_epoch,
            latest_request_id=latest_request_id,
        )
        state = CompletionCallbackState(
            run_id=run_id,
            request_id=request_id,
            batch_set=batch_set,
            batch_set_id=batch_set_id,
            cache_key=cache_key,
            policy_context=policy_context,
            ctx=ctx,
            shutdown_requested=shutdown_requested,
            is_preview=bool(fast_mode),
            slider_triggered=bool(self._ui.slider.slider_triggered_simulation()) or bool(fast_mode),
            explicit_batch_coalescing=False,
            simulation_identity=(
                callback_identity.simulation_identity
                if isinstance(callback_identity.simulation_identity, Mapping)
                else None
            ),
            preview_batch_cache_token=callback_identity.preview_batch_cache_token,
        )
        if isinstance(callback_identity.launch_provenance, Mapping):
            solver_provenance = dict(result.get("provenance") or {})
            launch_provenance = dict(callback_identity.launch_provenance)
            solver_provenance["launch_provenance"] = launch_provenance
            result = dict(result)
            result["provenance"] = solver_provenance
            if "mechanism_text" not in result and launch_provenance.get("mechanism_text") is not None:
                result["mechanism_text"] = str(launch_provenance["mechanism_text"])
        is_superseded_fast_request = bool(
            fast_mode
            and (
                bool(missing_owner_epoch)
                or (not self._deps.preview_request_matches_current_owner_epoch(request_id, callback_owner_epoch))
            )
        )

        if not is_superseded_fast_request:
            state.explicit_batch_coalescing = self._batch_context_owner.explicit_batch_coalescing_for_completion(
                slider_triggered=bool(state.slider_triggered),
                context=state.ctx if isinstance(state.ctx, Mapping) else {},
            )
            self._publication_owner.publish_success(
                result,
                state,
                debug_batch_parallel=bool(debug_batch_parallel),
            )
            return

        stale_fast_decision = self._completion_policy.resolve_superseded_fast_completion(
            preview_ownership=self._deps.completion_policy_preview_ownership(),
            context=policy_context,
            request_id=int(request_id),
            preview_owner_epoch=callback_owner_epoch,
            pending_replay=self._deps.completion_policy_pending_replay_state(),
            shutdown_requested=shutdown_requested,
        )
        logger.debug(
            "Active fast completion superseded (request_id=%s, latest=%s, run_id=%s, schedule_pending=%s, display_current_preview=%s, handoff_after_display=%s)",
            request_id,
            latest_request_id,
            run_id,
            bool(stale_fast_decision.schedule_pending_preview_run),
            bool(stale_fast_decision.display_current_preview),
            bool(stale_fast_decision.defer_context_deactivation_until_after_display),
        )

        self._deps.apply_completion_policy_state_patch(
            stale_fast_decision.state_patch,
            base_context=state.ctx if isinstance(state.ctx, Mapping) else None,
        )
        if stale_fast_decision.display_current_preview:
            state.stale_fast_handoff_after_display = bool(
                stale_fast_decision.defer_context_deactivation_until_after_display
            )
            state.explicit_batch_coalescing = self._batch_context_owner.explicit_batch_coalescing_for_completion(
                slider_triggered=bool(state.slider_triggered),
                context=state.ctx if isinstance(state.ctx, Mapping) else {},
            )
            self._publication_owner.publish_success(
                result,
                state,
                debug_batch_parallel=bool(debug_batch_parallel),
            )
            return

        callback_context_matches_current = bool(
            isinstance(state.ctx, Mapping)
            and self._batch_context_owner.context_matches_current_run_identity(state.ctx)
        )
        self._deps.apply_lifecycle_effects(
            self._lifecycle_effect_owner.superseded_fast_completion_effects(
                deactivate_context_immediately=bool(
                    stale_fast_decision.deactivate_context_immediately
                    and callback_context_matches_current
                ),
                schedule_pending_preview_run=bool(stale_fast_decision.schedule_pending_preview_run),
                reset_status_progress=bool(
                    stale_fast_decision.reset_status_progress
                    and callback_context_matches_current
                ),
                display_current_preview=bool(stale_fast_decision.display_current_preview),
                cleanup_state=self._batch_context_owner.completion_cleanup_state(
                    state.ctx if isinstance(state.ctx, Mapping) else {}
                ),
            )
        )

    def _has_active_context(self) -> bool:
        try:
            state = self._batch_context_owner.completion_state()
        except Exception:
            return False
        return bool(state is not None and state.active)
