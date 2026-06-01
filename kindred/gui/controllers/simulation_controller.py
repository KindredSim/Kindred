from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import os
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from PySide6 import QtCore

from kindred.core.batch_containment import BatchCompletionRecord, BatchLaneOutcome
from kindred.core.simulation_identity import (
    SimulationIdentity,
    contained_simulation_owner_identity,
)
from kindred.core.simulation_plan import SimulationPlan
from kindred.core.simulation_failure import (
    coerce_simulation_failure,
    simulation_failure_user_message,
)
from kindred.gui.controllers.batch_run_context_owner import BatchRunContextOwner
from kindred.gui.controllers.batch_dispatch_plan import (
    simulation_plan_from_payloadish,
)
from kindred.gui.controllers.batch_dispatch_materialization import BatchDispatchMaterializationOwner
from kindred.gui.controllers.simulation_run_preparation import (
    SimulationRunPreparationDependencies,
    SimulationRunPreparationOwner,
    SimulationRunPreparationPorts,
)
from kindred.gui.controllers.runtime_lane_allocation import (
    PreparedRuntimeRequestSet,
    RuntimeCompatibilityKey,
    RuntimeDispatchPlan,
    RuntimeLaneAllocator,
    RuntimeLaunchIntent,
    RuntimePreparationBlockedReason,
    RuntimeTaskDescriptor,
)
from kindred.gui.controllers.simulation_runtime_orchestrator import (
    RuntimePreviewReplaySnapshot,
    RuntimePreviewReplayState,
    RuntimeUiEffect,
    SimulationRuntimeOrchestrator,
)
from kindred.gui.controllers.simulation_runtime_dispatch import (
    SimulationRuntimeDispatchDependencies,
    SimulationRuntimeDispatchOwner,
)
from kindred.gui.controllers.simulation_result_materialization import SimulationResultMaterializationOwner
from kindred.gui.controllers.simulation_completion_policy import (
    CompletionPolicyContext,
    DirtySetState,
    RunActivitySnapshot,
    SimulationCompletionPolicy,
)
from kindred.gui.controllers.simulation_callback_freshness import (
    SimulationCallbackFreshnessDependencies,
    SimulationCallbackFreshnessOwner,
)
from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity
from kindred.gui.controllers.simulation_completion_callback import (
    SimulationCompletionCallbackDependencies,
    SimulationCompletionCallbackOwner,
)
from kindred.gui.controllers.simulation_completion_publication import (
    SimulationCompletionPublicationDependencies,
    SimulationCompletionPublicationOwner,
)
from kindred.gui.controllers.simulation_error_handling import (
    SimulationErrorHandlingDependencies,
    SimulationErrorHandlingOwner,
)
from kindred.gui.controllers.simulation_lifecycle_effects import (
    SimulationLifecycleEffectOwner,
    SimulationLifecycleEffects,
)
from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.core.runtime_defaults import contained_child_blas_thread_env
from kindred.gui.controllers.parallel_batch_executor import ParallelBatchExecutor
from kindred.gui.controllers.parallel_batch_outcome import (
    ParallelBatchOutcomeDependencies,
    ParallelBatchOutcomeOwner,
)
from kindred.gui.controllers.simulation_cache_admin import SimulationCacheAdmin
from kindred.gui.controllers.simulation_run_state import (
    PendingSliderPreviewLaunchState,
    PreviewOwnershipState,
    SimulationRunState,
)
from kindred.gui.controllers.slider_plot_coalescer import SliderPlotCoalescer
from kindred.gui.project_schema import PROJECT_DEFAULTS
from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.gui.ports import (
    DisplayTransitionOutcome,
    FreshPreviewDisplayEntry,
    SimulationCacheOpResult,
    SimulationCompletionDisplayOutcome,
    SimulationUiPorts,
    SliderReplayIntent,
)

logger = logging.getLogger(__name__)

__all__ = ["SimulationController"]

@dataclass(frozen=True)
class SimulationRuntimeInputsChangeOutcome:
    interactive_runtime_refresh_requested: bool

@dataclass
class _SerialBatchDispatchState:
    plan_payload: Dict[str, Any] | None
    cache_key: str
    context: Mapping[str, Any] | None


class SimulationController(QtCore.QObject):
    """
    Simulation execution + batch orchestration controller.

    This keeps MainWindow focused on UI composition while preserving behavior by
    allowing controlled access to UI elements via narrow UI ports.
    """

    runtime_readiness_render_requested = QtCore.Signal(object)

    def __init__(self, ui: SimulationUiPorts, *, parent: QtCore.QObject):
        super().__init__(parent)
        self.ui = ui

        self._run_state = SimulationRunState(on_progress_timeout=self._flush_progress_ui, parent=self)

        # ------------------------------------------------------------------
        # Batch execution + caching (migrated from MainWindow.__init__)
        # ------------------------------------------------------------------
        self._batch_run_queue: List[str] = []
        self._batch_run_results: Dict[str, Dict[str, Any]] = {}

        # Cache identity state (explicit full results vs slider previews)
        self._batch_cache = BatchSimulationCache()
        self._cache_admin = SimulationCacheAdmin(
            cache=self._batch_cache,
            settings_set_value=self.ui.settings.settings_set_value,
            settings_sync=self.ui.settings.settings_sync,
            record_nonfatal_exception=self._record_nonfatal_exception,
        )
        self._run_preparation_owner = SimulationRunPreparationOwner(
            ports=SimulationRunPreparationPorts(
                batch=self.ui.batch,
                dialogs=self.ui.dialogs,
                mechanism=self.ui.mechanism,
                mechanism_helpers=self.ui.mechanism_helpers,
                run_ui=self.ui.run_ui,
                slider=self.ui.slider,
                solver=self.ui.solver,
            ),
            dependencies=SimulationRunPreparationDependencies(
                claim_preview_ownership=self._claim_preview_ownership,
                clear_preview_ownership=self._clear_preview_ownership,
                clear_failed_fast_preview_ownership=self._clear_failed_fast_preview_ownership,
                clear_slider_triggered_preflight_state=self._clear_slider_triggered_preflight_state,
                requeue_preserved_pending_slider_replay_after_preflight_abort=(
                    self._requeue_preserved_pending_slider_replay_after_preflight_abort
                ),
                record_nonfatal_exception=self._record_nonfatal_exception,
                set_simulation_running=self._set_simulation_running,
                set_slider_simulation_active=self._set_slider_simulation_active,
                runtime_parameter_names_for_set=self._runtime_parameter_names_for_set,
                pending_initials_for_run_source_set=self.ui.mechanism.pending_initials_for_run_source_set,
                simulation_identity_for_set=self._simulation_identity_for_set,
                resolved_initials_for_batch_row=self._resolved_initials_for_batch_row,
                runtime_parameter_values_for_set=self._runtime_parameter_values_for_set,
                preview_contained_owner_identity=self._preview_contained_owner_identity,
                ordinary_contained_owner_identity=self._ordinary_contained_owner_identity,
                record_run_cache_key=self._batch_cache.record_run_cache_key,
                runtime_environment_key=self._runtime_lane_environment_key,
                runtime_lane_budget=lambda: int(self.batch_runtime_lane_budget),
                max_parallel_batch_workers=lambda: int(self.batch_runtime_lane_budget),
            ),
        )
        self._batch_context_owner = BatchRunContextOwner()
        self._batch_lane_executor = ParallelBatchExecutor(
            max_parallel_workers=int(PROJECT_DEFAULTS["max_parallel_batch_workers"]),
            limit_blas_threads_per_worker=bool(PROJECT_DEFAULTS["limit_blas_threads_per_worker"]),
            record_nonfatal_exception=self._record_nonfatal_exception,
        )
        self._runtime_lane_allocator = RuntimeLaneAllocator(
            backend_lease_provider=self._batch_parallel,
        )
        self._batch_dispatch_materialization_owner = BatchDispatchMaterializationOwner(
            batch=self.ui.batch,
            slider=self.ui.slider,
        )
        self._authoritative_mechanism_transition_epoch = 0
        self._authoritative_runtime_input_epoch = 0
        self._authoritative_runtime_input_global_epoch = 0
        self._authoritative_runtime_input_set_epoch_by_set_id: Dict[str, int] = {}
        self._symbolic_wegscheider_identity_cache: Dict[tuple[str, str], Dict[str, Any]] = {}

        self._batch_completion_poll_timer = QtCore.QTimer(self)
        self._batch_completion_poll_timer.setInterval(20)
        self._batch_completion_poll_timer.timeout.connect(self._poll_parallel_batch_completions)

        self._plot_coalescer = SliderPlotCoalescer(
            on_timeout=self._flush_slider_plot_updates,
            parent=self,
            slider_interval_ms=24,
            explicit_interval_ms=90,
        )

        self._debug_batch_parallel: bool = bool(os.environ.get("KINDRED_DEBUG_BATCH_PAR"))
        self._nonfatal_exception_count: int = 0
        self._last_nonfatal_exception: Optional[str] = None
        self._shutdown_requested_for_close: bool = False
        self._discarded_slider_preview_generation_id: Optional[int] = None
        self._batch_runtime_lane_budget = int(PROJECT_DEFAULTS["batch_runtime_lane_budget"])
        self._runtime_orchestrator = SimulationRuntimeOrchestrator(
            allocator=self._runtime_lane_allocator,
            backend=self._batch_parallel,
            render=self.runtime_readiness_render_requested.emit,
            current_runtime_input_epochs=self._runtime_input_epochs_for_sets,
            prepared_request_is_current=self._runtime_prepared_request_is_current,
            next_preview_replay_request_id=self._next_slider_preview_request_id,
        )
        self._runtime_dispatch_owner = SimulationRuntimeDispatchOwner(
            ui=self.ui,
            batch_executor=self._batch_parallel,
            parent=self,
            dependencies=SimulationRuntimeDispatchDependencies(
                next_run_id=self._next_runtime_run_id,
                load_context=self._load_runtime_dispatch_context,
                callback_identity_for_descriptor=self._callback_identity_for_runtime_descriptor,
                runtime_lifecycle=self._runtime_orchestrator,
                record_nonfatal_exception=self._record_nonfatal_exception,
                deactivate_dispatch_context=self._deactivate_runtime_dispatch_context,
            ),
        )
        self._completion_policy = SimulationCompletionPolicy()
        self._lifecycle_effect_owner = SimulationLifecycleEffectOwner()
        self._result_materialization_owner = SimulationResultMaterializationOwner(
            ui=self.ui,
            record_nonfatal_exception=self._record_nonfatal_exception,
        )
        self._completion_publication_owner = SimulationCompletionPublicationOwner(
            ui=self.ui,
            batch_context_owner=self._batch_context_owner,
            batch_cache=self._batch_cache,
            cache_admin=self._cache_admin,
            completion_policy=self._completion_policy,
            lifecycle_effect_owner=self._lifecycle_effect_owner,
            result_materialization_owner=self._result_materialization_owner,
            dependencies=SimulationCompletionPublicationDependencies(
                apply_lifecycle_effects=self._apply_simulation_lifecycle_effects,
                record_nonfatal_exception=self._record_nonfatal_exception,
                queue_slider_plot_update=self.queue_slider_plot_update,
                finalize_explicit_batch_dirty_reset=self._finalize_explicit_batch_dirty_reset,
                flush_slider_plot_updates=self.flush_slider_plot_updates,
                show_scoped_batch_failure_summary=self._show_scoped_batch_failure_summary,
                request_completion_preview_replay=self._request_completion_preview_replay_effects,
                clear_pending_progress_status=self._clear_pending_progress_status,
            ),
        )
        self._callback_freshness_owner = SimulationCallbackFreshnessOwner(
            SimulationCallbackFreshnessDependencies(
                run_state=self._run_state,
                batch_context_owner=self._batch_context_owner,
                preview_ownership=self._completion_policy_preview_ownership,
                shutdown_requested=self._shutdown_requested_for_completion_callbacks,
                current_global_epoch=self._current_runtime_input_global_epoch,
                current_epoch=self._current_runtime_input_epoch,
                current_set_epoch=self._runtime_input_set_epoch,
                finalize_batch_queue_done_without_result=self._finalize_batch_queue_done_without_result,
            )
        )
        self._completion_callback_owner = SimulationCompletionCallbackOwner(
            ui=self.ui,
            batch_context_owner=self._batch_context_owner,
            completion_policy=self._completion_policy,
            lifecycle_effect_owner=self._lifecycle_effect_owner,
            publication_owner=self._completion_publication_owner,
            dependencies=SimulationCompletionCallbackDependencies(
                freshness=self._callback_freshness_owner,
                completion_policy_preview_ownership=self._completion_policy_preview_ownership,
                stale_fast_completion_replay_decision=self._stale_fast_completion_replay_decision,
                apply_completion_policy_state_patch=self._apply_completion_policy_state_patch,
                apply_lifecycle_effects=self._apply_simulation_lifecycle_effects,
                apply_runtime_effects=self._apply_runtime_lifecycle_ui_effects,
            ),
        )
        self._error_handling_owner = SimulationErrorHandlingOwner(
            ui=self.ui,
            batch_context_owner=self._batch_context_owner,
            completion_policy=self._completion_policy,
            lifecycle_effect_owner=self._lifecycle_effect_owner,
            dependencies=SimulationErrorHandlingDependencies(
                freshness=self._callback_freshness_owner,
                completion_policy_preview_ownership=self._completion_policy_preview_ownership,
                stale_fast_error_replay_decision=self._stale_fast_error_replay_decision,
                apply_completion_policy_state_patch=self._apply_completion_policy_state_patch,
                apply_lifecycle_effects=self._apply_simulation_lifecycle_effects,
                apply_runtime_effects=self._apply_runtime_lifecycle_ui_effects,
                capture_terminal_failure_preview_replay_snapshot=(
                    self._capture_terminal_failure_preview_replay_snapshot
                ),
                request_terminal_failure_preview_replay=(
                    self._request_terminal_failure_preview_replay_effects
                ),
                request_pending_preview_replay=self._request_completion_preview_replay_effects,
                handle_current_preview_simulation_failure=self._handle_current_preview_simulation_failure,
            ),
        )
        self._parallel_batch_outcome_owner = ParallelBatchOutcomeOwner(
            ui=self.ui,
            batch_parallel=self._batch_parallel,
            batch_context_owner=self._batch_context_owner,
            batch_cache=self._batch_cache,
            completion_callback_owner=self._completion_callback_owner,
            error_handling_owner=self._error_handling_owner,
            dependencies=ParallelBatchOutcomeDependencies(
                freshness=self._callback_freshness_owner,
                record_nonfatal_exception=self._record_nonfatal_exception,
                finalize_scoped_batch_success_subset=self._finalize_scoped_batch_success_subset,
                runtime_display_completed=self._runtime_orchestrator.display_completed,
                show_scoped_batch_failure_summary=self._show_scoped_batch_failure_summary,
                request_terminal_failure_preview_replay=(
                    self._request_terminal_failure_preview_replay_effects
                ),
                reset_parallel_batch_run_and_shutdown_lane_pool=self._cancel_runtime_for_parallel_outcome_reset,
                set_simulation_running=self._set_simulation_running,
                set_slider_simulation_active=self._set_slider_simulation_active,
            ),
        )
        self._runtime_orchestrator.set_completion_consumer(self._parallel_batch_outcome_owner)

    # ------------------------------------------------------------------
    # Public interface (MainWindow boundary)
    # ------------------------------------------------------------------
    @property
    def authoritative_mechanism_transition_epoch(self) -> int:
        return int(self._authoritative_mechanism_transition_epoch)

    @property
    def _batch_parallel(self):
        return self._batch_lane_executor

    @property
    def _simulation_running(self) -> bool:
        return bool(self._run_state.simulation_running)

    @_simulation_running.setter
    def _simulation_running(self, value: bool) -> None:
        self._run_state.simulation_running = bool(value)

    def _set_discarded_slider_preview_generation(self, value: int | None) -> None:
        self._discarded_slider_preview_generation_id = int(value) if value is not None else None

    def _start_batch_completion_poll_timer(self) -> None:
        timer = getattr(self, "_batch_completion_poll_timer", None)
        if timer is not None:
            timer.start()

    @property
    def _processing_progress(self) -> bool:
        return bool(self._run_state.processing_progress)

    @_processing_progress.setter
    def _processing_progress(self, value: bool) -> None:
        self._run_state.processing_progress = bool(value)

    @property
    def _pending_progress_payload(self) -> Optional[Tuple[int, str]]:
        return self._run_state.pending_progress_payload

    @_pending_progress_payload.setter
    def _pending_progress_payload(self, value: Optional[Tuple[int, str]]) -> None:
        self._run_state.pending_progress_payload = value

    @property
    def _progress_flush_interval_ms(self) -> int:
        return int(self._run_state.progress_flush_interval_ms)

    @_progress_flush_interval_ms.setter
    def _progress_flush_interval_ms(self, value: int) -> None:
        self._run_state.progress_flush_interval_ms = int(value)
        self._run_state.progress_flush_timer.setInterval(int(value))

    @property
    def _progress_flush_timer(self) -> QtCore.QTimer:
        return self._run_state.progress_flush_timer

    @property
    def _slider_simulation_active(self) -> bool:
        return bool(self._run_state.slider_simulation_active)

    @_slider_simulation_active.setter
    def _slider_simulation_active(self, value: bool) -> None:
        self._run_state.slider_simulation_active = bool(value)

    @property
    def _pending_slider_simulation(self) -> bool:
        return bool(self._pending_slider_preview_launch.active)

    @property
    def _run_sequence_id(self) -> int:
        return int(self._run_state.run_sequence_id)

    @_run_sequence_id.setter
    def _run_sequence_id(self, value: int) -> None:
        self._run_state.run_sequence_id = int(value)

    @property
    def _active_run_id(self) -> int:
        return int(self._run_state.active_run_id)

    @_active_run_id.setter
    def _active_run_id(self, value: int) -> None:
        self._run_state.active_run_id = int(value)

    @property
    def _sim_request_id(self) -> int:
        return int(self._run_state.sim_request_id)

    @_sim_request_id.setter
    def _sim_request_id(self, value: int) -> None:
        self._run_state.sim_request_id = int(value)

    @property
    def _latest_sim_request_id(self) -> int:
        return int(self._run_state.latest_sim_request_id)

    @_latest_sim_request_id.setter
    def _latest_sim_request_id(self, value: int) -> None:
        value_i = int(value)
        self._run_state.latest_sim_request_id = value_i
        if int(getattr(self._run_state, "sim_request_id", 0) or 0) < value_i:
            self._run_state.sim_request_id = value_i

    @property
    def _pending_slider_sim_request_id(self) -> Optional[int]:
        return self._pending_slider_preview_launch.request_id

    @property
    def _pending_slider_target_set_ids(self) -> Tuple[str, ...]:
        return tuple(
            str(set_id)
            for set_id in (self._pending_slider_preview_launch.target_set_ids or ())
            if str(set_id)
        )

    @property
    def _pending_slider_handoff_queued(self) -> bool:
        return bool(self._pending_slider_preview_launch.handoff_queued)

    @property
    def _pending_slider_preview_launch(self) -> PendingSliderPreviewLaunchState:
        replay = getattr(self._run_state, "pending_slider_preview_launch", None)
        if isinstance(replay, PendingSliderPreviewLaunchState):
            return replay
        normalized = PendingSliderPreviewLaunchState()
        self._run_state.pending_slider_preview_launch = normalized
        return normalized

    def _next_pending_slider_preview_replay_generation(self) -> int:
        current = getattr(self._run_state, "pending_slider_preview_replay_generation", 0)
        try:
            next_generation = max(0, int(current)) + 1
        except (TypeError, ValueError, OverflowError):
            next_generation = 1
        self._run_state.pending_slider_preview_replay_generation = int(next_generation)
        return int(next_generation)

    def _run_target_set_ids_for_rows(self, rows: Sequence[int]) -> Tuple[str, ...]:
        target_set_ids: list[str] = []
        seen: set[str] = set()
        for row in rows or ():
            try:
                set_id = self.ui.batch.batch_set_id_for_row(int(row))
            except Exception:
                set_id = None
            set_id_s = str(set_id or "").strip()
            if not set_id_s or set_id_s in seen:
                continue
            seen.add(set_id_s)
            target_set_ids.append(set_id_s)
        return tuple(target_set_ids)

    def _run_rows_for_target_set_ids(
        self,
        target_set_ids: Sequence[str],
        *,
        fallback_rows: Sequence[int] = (),
    ) -> list[int]:
        target_ids = [str(set_id).strip() for set_id in target_set_ids or () if str(set_id).strip()]
        if target_ids:
            rows_by_set_id: Dict[str, int] = {}
            try:
                row_count = int(self.ui.batch.batch_store_row_count())
            except Exception:
                row_count = 0
            for row in range(max(0, int(row_count))):
                try:
                    set_id = self.ui.batch.batch_set_id_for_row(int(row))
                except Exception:
                    set_id = None
                set_id_s = str(set_id or "").strip()
                if set_id_s and set_id_s not in rows_by_set_id:
                    rows_by_set_id[set_id_s] = int(row)
            if any(set_id not in rows_by_set_id for set_id in target_ids):
                return []
            return [int(rows_by_set_id[set_id]) for set_id in target_ids if set_id in rows_by_set_id]
        try:
            row_count = int(self.ui.batch.batch_store_row_count())
        except Exception:
            row_count = None
        rows: list[int] = []
        seen_rows: set[int] = set()
        for row in fallback_rows or ():
            try:
                row_i = int(row)
            except (TypeError, ValueError, OverflowError):
                continue
            if row_i in seen_rows:
                continue
            if row_count is not None and not (0 <= row_i < int(row_count)):
                continue
            seen_rows.add(row_i)
            rows.append(row_i)
        return rows

    def _runtime_preview_replay_state(
        self,
        replay: Optional[PendingSliderPreviewLaunchState] = None,
    ) -> RuntimePreviewReplayState:
        state = replay if isinstance(replay, PendingSliderPreviewLaunchState) else self._pending_slider_preview_launch
        return RuntimePreviewReplayState.from_pending(state)

    def _stop_deferred_preview_replay_timers(self) -> None:
        for stop_fn, timer_name in (
            (self.ui.slider.stop_variable_update_timer, "_variable_update_timer"),
            (self.ui.slider.stop_species_slider_update_timer, "_species_slider_update_timer"),
        ):
            try:
                stop_fn()
            except Exception as exc:
                self._record_nonfatal_exception(
                    f"Failed to stop debounce timer {str(timer_name)} before deferred replay handoff",
                    exc,
                )

    def _stop_slider_debounce_timers_for_lifecycle_effect(self) -> None:
        for stop_fn, timer_name in (
            (self.ui.slider.stop_variable_update_timer, "_variable_update_timer"),
            (self.ui.slider.stop_species_slider_update_timer, "_species_slider_update_timer"),
        ):
            try:
                stop_fn()
            except Exception as exc:
                self._record_nonfatal_exception(
                    f"Failed to stop debounce timer {str(timer_name)} while applying simulation lifecycle effects",
                    exc,
                )

    def _apply_simulation_lifecycle_effects(
        self,
        effects: SimulationLifecycleEffects,
        *,
        failed_run_context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if bool(effects.shutdown_lane_pool):
            self._apply_runtime_lifecycle_ui_effects(
                self._runtime_orchestrator.cancel_requested(
                    kind=effects.runtime_cancel_kind or "shutdown"
                )
            )
        if bool(effects.cleanup_lane_pool):
            if bool(effects.clear_pending_plot_updates):
                self._clear_pending_slider_plot_updates()
            self._runtime_orchestrator.display_completed(
                kind=effects.runtime_display_kind or "success"
            )
        if bool(effects.clear_shutdown_request):
            self._clear_shutdown_request_after_close_cleanup()
        if bool(effects.clear_pending_preview_plot_updates):
            self._clear_pending_preview_slider_plot_updates()
        if bool(effects.reset_slider_triggered):
            try:
                self.ui.slider.set_slider_triggered_simulation(False)
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to clear slider-triggered state while applying simulation lifecycle effects",
                    exc,
                )
        if effects.simulation_running is not None:
            self._simulation_running = bool(effects.simulation_running)
        if effects.slider_simulation_active is not None:
            self._slider_simulation_active = bool(effects.slider_simulation_active)
        if effects.run_enabled is not None:
            try:
                self.ui.run_ui.set_run_button_enabled(bool(effects.run_enabled))
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to apply Run button simulation lifecycle effect",
                    exc,
                )
        if effects.stop_enabled is not None:
            try:
                self.ui.run_ui.set_stop_button_enabled(bool(effects.stop_enabled))
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to apply Stop button simulation lifecycle effect",
                    exc,
                )
        if effects.status_text is not None:
            self.ui.run_ui.set_status_text(str(effects.status_text))
        if effects.progress_value is not None:
            self.ui.run_ui.set_sim_progress_value(int(effects.progress_value))
        if effects.algebra_status_text is not None:
            try:
                self.ui.run_ui.set_algebra_status_text(
                    str(effects.algebra_status_text),
                    details=effects.algebra_status_details,
                )
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to apply algebra status label simulation lifecycle effect",
                    exc,
                )
        elif bool(effects.clear_algebra_status):
            try:
                self.ui.run_ui.set_algebra_status_text("")
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to clear algebra status label while applying simulation lifecycle effects",
                    exc,
                )
        if bool(effects.repaint_widgets):
            self.ui.run_ui.repaint_simulation_widgets()
        if bool(effects.stop_debounce_timers):
            self._stop_slider_debounce_timers_for_lifecycle_effect()
        if effects.modal_error is not None:
            self.ui.dialogs.message_box_critical(
                effects.modal_error.title,
                effects.modal_error.message,
                details=effects.modal_error.details,
            )

    @property
    def _preview_ownership(self) -> PreviewOwnershipState:
        ownership = getattr(self._run_state, "preview_ownership", None)
        if isinstance(ownership, PreviewOwnershipState):
            return ownership
        normalized = PreviewOwnershipState()
        self._run_state.preview_ownership = normalized
        return normalized

    @_preview_ownership.setter
    def _preview_ownership(self, value: PreviewOwnershipState) -> None:
        self._run_state.preview_ownership = (
            value if isinstance(value, PreviewOwnershipState) else PreviewOwnershipState()
        )

    def _set_preview_ownership(
        self,
        *,
        request_id: Optional[int],
        target_set_ids: Sequence[str],
    ) -> PreviewOwnershipState:
        current = self._preview_ownership
        candidate = PreviewOwnershipState(
            request_id=request_id,
            epoch=current.epoch,
            target_set_ids=tuple(target_set_ids),
        )
        if (
            current.request_id == candidate.request_id
            and current.target_set_ids == candidate.target_set_ids
        ):
            return current
        updated = PreviewOwnershipState(
            request_id=candidate.request_id,
            epoch=int(current.epoch) + 1,
            target_set_ids=candidate.target_set_ids,
        )
        self._preview_ownership = updated
        return updated

    def _claim_preview_ownership(
        self,
        *,
        request_id: int,
        target_set_ids: Sequence[str],
    ) -> PreviewOwnershipState:
        return self._set_preview_ownership(
            request_id=int(request_id),
            target_set_ids=target_set_ids,
        )

    def _clear_preview_ownership(self) -> PreviewOwnershipState:
        return self._set_preview_ownership(request_id=None, target_set_ids=())

    def _mark_request_started(self, request_id: int) -> int:
        request_id_i = int(request_id)
        if request_id_i > int(getattr(self, "_latest_sim_request_id", 0)):
            self._latest_sim_request_id = request_id_i
        return request_id_i

    def queue_pending_slider_preview_replay(
        self,
        *,
        target_set_ids: Sequence[str],
        request_id: Optional[int] = None,
        preserve_existing_request: bool = False,
    ) -> None:
        self._apply_runtime_lifecycle_ui_effects(
            self._runtime_orchestrator.preview_replay_state_requested(
                current_state=self._pending_slider_preview_launch,
                target_set_ids=tuple(target_set_ids),
                request_id=request_id,
                preserve_existing_request=bool(preserve_existing_request),
            )
        )

    def submit_slider_preview_replay_intent(
        self,
        intent: SliderReplayIntent,
        *,
        preserve_existing_request: bool = False,
    ) -> None:
        normalized_intent = intent if isinstance(intent, SliderReplayIntent) else None
        if normalized_intent is None or not normalized_intent.target_set_ids:
            self.clear_pending_slider_preview_replay(clear_plot_updates=False)
            return
        self.queue_pending_slider_preview_replay(
            target_set_ids=normalized_intent.target_set_ids,
            request_id=None,
            preserve_existing_request=bool(preserve_existing_request),
        )

    def _clear_failed_fast_preview_ownership(self) -> None:
        self._clear_preview_ownership()
        self._clear_pending_slider_preview_replay_state(clear_plot_updates=False)

    def clear_pending_slider_preview_replay(self, *, clear_plot_updates: bool = True) -> None:
        self._clear_pending_slider_preview_replay_state(clear_plot_updates=bool(clear_plot_updates))

    def _set_pending_slider_preview_replay_state(self, state: object) -> None:
        self._run_state.pending_slider_preview_launch = PendingSliderPreviewLaunchState(
            active=bool(getattr(state, "active", False)),
            request_id=getattr(state, "request_id", None),
            target_set_ids=tuple(getattr(state, "target_set_ids", ()) or ()),
            handoff_queued=bool(getattr(state, "handoff_queued", False)),
            replay_generation=int(getattr(state, "replay_generation", 0) or 0),
        )
        self._run_state.pending_slider_preview_replay_generation = int(
            self._run_state.pending_slider_preview_launch.replay_generation
        )

    def _clear_pending_slider_preview_replay_state(self, *, clear_plot_updates: bool = True) -> None:
        self._run_state.pending_slider_preview_launch = PendingSliderPreviewLaunchState()
        self._run_state.pending_slider_preview_replay_generation = 0
        if clear_plot_updates:
            self._clear_pending_preview_slider_plot_updates()

    @property
    def _pending_slider_plot_set_ids(self) -> Set[str]:
        return set(self._plot_coalescer.pending.set_ids)

    @_pending_slider_plot_set_ids.setter
    def _pending_slider_plot_set_ids(self, value: Set[str]) -> None:
        self._plot_coalescer.pending.set_ids = set(value or set())

    @property
    def _pending_slider_plot_cache_key(self) -> Optional[str]:
        return self._plot_coalescer.pending.cache_key

    @_pending_slider_plot_cache_key.setter
    def _pending_slider_plot_cache_key(self, value: Optional[str]) -> None:
        self._plot_coalescer.pending.cache_key = str(value) if value is not None else None

    @property
    def _pending_slider_plot_cache_kind(self) -> Optional[str]:
        return self._plot_coalescer.pending.cache_kind

    @_pending_slider_plot_cache_kind.setter
    def _pending_slider_plot_cache_kind(self, value: Optional[str]) -> None:
        self._plot_coalescer.pending.cache_kind = str(value) if value is not None else None

    @property
    def _pending_slider_plot_request_id(self) -> Optional[int]:
        return self._plot_coalescer.pending.request_id

    @_pending_slider_plot_request_id.setter
    def _pending_slider_plot_request_id(self, value: Optional[int]) -> None:
        self._plot_coalescer.pending.request_id = int(value) if value is not None else None

    @property
    def _pending_slider_plot_run_id(self) -> Optional[int]:
        return self._plot_coalescer.pending.run_id

    @_pending_slider_plot_run_id.setter
    def _pending_slider_plot_run_id(self, value: Optional[int]) -> None:
        self._plot_coalescer.pending.run_id = int(value) if value is not None else None

    @property
    def _pending_slider_plot_preview_request_id(self) -> Optional[int]:
        return self._plot_coalescer.pending.accepted_preview_request_id

    @_pending_slider_plot_preview_request_id.setter
    def _pending_slider_plot_preview_request_id(self, value: Optional[int]) -> None:
        self._plot_coalescer.pending.accepted_preview_request_id = int(value) if value is not None else None

    @property
    def _pending_slider_plot_preview_owner_epoch(self) -> Optional[int]:
        return self._plot_coalescer.pending.accepted_preview_owner_epoch

    @_pending_slider_plot_preview_owner_epoch.setter
    def _pending_slider_plot_preview_owner_epoch(self, value: Optional[int]) -> None:
        self._plot_coalescer.pending.accepted_preview_owner_epoch = int(value) if value is not None else None

    @property
    def _slider_plot_coalesce_interval_ms(self) -> int:
        return int(self._plot_coalescer.slider_interval_ms)

    @_slider_plot_coalesce_interval_ms.setter
    def _slider_plot_coalesce_interval_ms(self, value: int) -> None:
        self._plot_coalescer.slider_interval_ms = int(value)

    @property
    def _explicit_plot_coalesce_interval_ms(self) -> int:
        return int(self._plot_coalescer.explicit_interval_ms)

    @_explicit_plot_coalesce_interval_ms.setter
    def _explicit_plot_coalesce_interval_ms(self, value: int) -> None:
        self._plot_coalescer.explicit_interval_ms = int(value)

    @property
    def _slider_plot_coalesce_timer(self) -> QtCore.QTimer:
        return self._plot_coalescer.timer

    @property
    def simulation_running(self) -> bool:
        return bool(self._simulation_running)

    @simulation_running.setter
    def simulation_running(self, value: bool) -> None:
        self._simulation_running = bool(value)

    @property
    def run_state(self) -> SimulationRunState:
        return self._run_state

    @property
    def batch_cache(self) -> BatchSimulationCache:
        return self._batch_cache

    @property
    def batch_context_owner(self) -> BatchRunContextOwner:
        return self._batch_context_owner

    @property
    def parallel_batch(self) -> ParallelBatchExecutor:
        return self._batch_parallel

    @property
    def batch_runtime_lane_budget(self) -> int:
        return max(1, int(getattr(self, "_batch_runtime_lane_budget", 1) or 1))

    @batch_runtime_lane_budget.setter
    def batch_runtime_lane_budget(self, value: object) -> None:
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            parsed = int(PROJECT_DEFAULTS["batch_runtime_lane_budget"])
        self._batch_runtime_lane_budget = min(
            int(MAX_PARALLEL_WORKERS_CEILING),
            max(1, int(parsed)),
        )

    @property
    def plot_coalescer(self) -> SliderPlotCoalescer:
        return self._plot_coalescer

    def _completion_policy_activity_snapshot(self) -> RunActivitySnapshot:
        return RunActivitySnapshot(
            latest_request_id=int(getattr(self, "_latest_sim_request_id", 0)),
            simulation_running=bool(getattr(self, "_simulation_running", False)),
            slider_simulation_active=bool(getattr(self, "_slider_simulation_active", False)),
            worker_running=False,
            worker_fast_mode=None,
            worker_request_id=None,
            discarded_slider_preview_generation_id=getattr(
                self,
                "_discarded_slider_preview_generation_id",
                None,
            ),
        )

    def _completion_policy_preview_ownership(self) -> PreviewOwnershipState:
        return self._preview_ownership

    def _stale_fast_completion_replay_decision(self, **kwargs):
        return self._runtime_orchestrator.stale_fast_completion_replay_decision(
            current_state=self._runtime_preview_replay_state(),
            **kwargs,
        )

    def _stale_fast_error_replay_decision(self, **kwargs):
        return self._runtime_orchestrator.stale_fast_error_replay_decision(
            current_state=self._runtime_preview_replay_state(),
            **kwargs,
        )

    def _shutdown_requested_for_completion_callbacks(self) -> bool:
        return bool(getattr(self, "_shutdown_requested_for_close", False))

    def _current_runtime_input_global_epoch(self) -> int:
        return int(getattr(self, "_authoritative_runtime_input_global_epoch", 0) or 0)

    def _current_runtime_input_epoch(self) -> int:
        return int(getattr(self, "_authoritative_runtime_input_epoch", 0) or 0)

    def _apply_completion_policy_state_patch(
        self,
        patch,
        *,
        base_context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[CompletionPolicyContext]:
        updated_context = None
        if patch.context is not None:
            updated_context = patch.context
            self._batch_context_owner.serialize_completion_policy_context(
                patch.context,
                base_context=base_context,
            )
        if bool(getattr(patch, "clear_discarded_slider_preview_generation", False)):
            self._discarded_slider_preview_generation_id = None
        return updated_context

    def _capture_dirty_state_by_set_id(
        self,
        set_ids: Sequence[str],
    ) -> Dict[str, DirtySetState]:
        state_by_set_id: Dict[str, DirtySetState] = {}
        for set_id in (set_ids or ()):
            set_id_s = str(set_id or "").strip()
            if not set_id_s:
                continue
            is_dirty = False
            generation = None
            try:
                is_dirty = bool(self.ui.slider.has_dirty_state_for_set(set_id_s))
            except Exception as exc:
                self._record_nonfatal_exception(
                    f"Failed to inspect dirty-state ownership for {set_id_s}",
                    exc,
                )
            if is_dirty:
                try:
                    generation = int(self.ui.slider.dirty_state_generation(set_id_s))
                except Exception as exc:
                    self._record_nonfatal_exception(
                        f"Failed to inspect dirty-state generation for {set_id_s}",
                        exc,
                    )
                    generation = None
            state_by_set_id[set_id_s] = DirtySetState(is_dirty=is_dirty, generation=generation)
        return state_by_set_id

    def _runtime_dirty_generation_facts(
        self,
        target_set_ids: Sequence[str],
    ) -> Dict[str, int | None]:
        dirty_state = self._capture_dirty_state_by_set_id(target_set_ids)
        return {
            str(set_id): (
                int(state.generation)
                if state is not None and bool(state.is_dirty) and state.generation is not None
                else None
            )
            for set_id, state in dirty_state.items()
            if str(set_id)
        }

    def _capture_terminal_failure_preview_replay_snapshot(self) -> RuntimePreviewReplaySnapshot:
        pending_state = self._runtime_preview_replay_state()
        return self._runtime_orchestrator.terminal_failure_replay_snapshot(
            pending_state=pending_state,
            dirty_generation_by_set_id=self._runtime_dirty_generation_facts(
                pending_state.target_set_ids,
            ),
        )

    def _request_completion_preview_replay_effects(
        self,
        *,
        shutdown_requested: bool,
    ) -> None:
        self._apply_runtime_lifecycle_ui_effects(
            self._runtime_orchestrator.pending_preview_replay_requested(
                current_state=self._runtime_preview_replay_state(),
                shutdown_requested=bool(shutdown_requested),
                stop_timers=True,
            )
        )

    def _request_terminal_failure_preview_replay_effects(
        self,
        *,
        fast_mode: bool,
        replay_snapshot: RuntimePreviewReplaySnapshot | None = None,
    ) -> None:
        pending_state = self._runtime_preview_replay_state()
        effects = self._runtime_orchestrator.terminal_failure_replay_requested(
            fast_mode=bool(fast_mode),
            pending_state=pending_state,
            replay_snapshot=replay_snapshot,
            dirty_generation_by_set_id=self._runtime_dirty_generation_facts(
                pending_state.target_set_ids,
            ),
        )
        self._apply_runtime_lifecycle_ui_effects(effects)

    def queue_slider_plot_update(
        self,
        *,
        set_id: Optional[str],
        cache_key: Optional[str],
        request_id: Optional[int],
        run_id: Optional[int],
        slider_triggered: bool = True,
        valid_set_ids: Optional[Sequence[str]] = None,
        fresh_preview_entry: Optional[FreshPreviewDisplayEntry] = None,
    ) -> None:
        self._queue_slider_plot_update(
            set_id=set_id,
            cache_key=cache_key,
            request_id=request_id,
            run_id=run_id,
            slider_triggered=slider_triggered,
            valid_set_ids=valid_set_ids,
            fresh_preview_entry=fresh_preview_entry,
        )

    def next_sim_request_id(self) -> int:
        return int(self._next_sim_request_id())

    def next_slider_preview_request_id(self) -> int:
        return int(self._next_slider_preview_request_id())

    def _pending_slider_preview_rows(self) -> tuple[int, ...]:
        pending = self._pending_slider_preview_launch
        if not bool(pending.active) or not pending.target_set_ids:
            return ()
        selected_rows = self.ui.batch.batch_rows_for_scope("selected")
        return tuple(
            self._slider_target_rows_for_dispatch(
                selected_rows,
                target_set_ids=list(pending.target_set_ids),
            )
        )

    def run_simulation(self) -> None:
        rows_to_run = self._run_selected_rows_or_abort()
        if not rows_to_run:
            return
        request_id = self._next_sim_request_id()
        target_set_ids = self._run_target_set_ids_for_rows(rows_to_run)
        self._prepare_ordinary_runtime_launch_context(
            target_set_ids=target_set_ids,
        )
        requested_show_set_ids = self._runtime_requested_show_set_ids(target_set_ids)
        intent = RuntimeLaunchIntent(
            intent_kind="ordinary",
            ui_action="run_selected",
            rows=tuple(rows_to_run),
            set_ids=target_set_ids,
            requested_show_set_ids=requested_show_set_ids,
            requested_show_labels_by_set_id=self._runtime_requested_show_labels_by_set_id(
                requested_show_set_ids
            ),
            request_token=int(request_id),
            runtime_input_epochs=self._runtime_input_epochs_for_sets(
                target_set_ids
            ),
        )
        prepared = self._run_preparation_owner.prepare_runtime_request_set(
            intent=intent,
            fast_mode=False,
        )
        self._accept_and_dispatch_prepared_runtime_request(prepared)

    def _prepare_ordinary_runtime_launch_context(
        self,
        *,
        target_set_ids: Sequence[str],
    ) -> None:
        self._flush_pending_slider_updates_for_run(reset_set_ids=target_set_ids)
        self.clear_pending_slider_preview_replay(clear_plot_updates=False)
        self._clear_preview_ownership()

    def launch_pending_slider_preview_replay(self) -> None:
        pending = self._pending_slider_preview_launch
        if bool(pending.active) and pending.request_id is None:
            self.clear_pending_slider_preview_replay(clear_plot_updates=False)
            return
        rows = self._pending_slider_preview_rows()
        if not rows:
            self.clear_pending_slider_preview_replay(clear_plot_updates=False)
            return
        request_token = pending.request_id
        target_set_ids = tuple(str(set_id) for set_id in pending.target_set_ids or () if str(set_id))
        preview_ownership = getattr(self, "_preview_ownership", None)
        if request_token is not None and target_set_ids:
            preview_ownership = self._claim_preview_ownership(
                request_id=int(request_token),
                target_set_ids=target_set_ids,
            )
        preview_epoch = getattr(preview_ownership, "epoch", None)
        runtime_set_ids = self._run_target_set_ids_for_rows(rows)
        requested_show_set_ids = self._runtime_requested_show_set_ids(target_set_ids or runtime_set_ids)
        intent = RuntimeLaunchIntent(
            intent_kind="preview",
            ui_action="slider_preview",
            rows=tuple(rows),
            set_ids=runtime_set_ids,
            requested_show_set_ids=requested_show_set_ids,
            requested_show_labels_by_set_id=self._runtime_requested_show_labels_by_set_id(
                requested_show_set_ids
            ),
            request_token=int(request_token) if request_token is not None else None,
            preview_request_id=int(request_token) if request_token is not None else None,
            preview_epoch=int(preview_epoch) if preview_epoch is not None else None,
            runtime_input_epochs=self._runtime_input_epochs_for_sets(
                runtime_set_ids
            ),
        )
        prepared = self._run_preparation_owner.prepare_runtime_request_set(
            intent=intent,
            fast_mode=True,
        )
        dispatch_plan = self._runtime_orchestrator.accept_prepared_request(prepared)
        if dispatch_plan is not None:
            self._dispatch_runtime_plan(dispatch_plan)

    def retry_runtime_readiness(self) -> RuntimeDispatchPlan | None:
        dispatch_plan = self._runtime_orchestrator.retry_runtime_readiness()
        if dispatch_plan is not None:
            self._dispatch_runtime_plan(dispatch_plan)
        return dispatch_plan

    def _runtime_requested_show_set_ids(self, fallback_set_ids: Sequence[str]) -> tuple[str, ...]:
        requested: Sequence[str] = ()
        try:
            requested = self.ui.batch.requested_show_batch_set_ids()
        except Exception:
            requested = ()
        ordered: list[str] = []
        for group in (fallback_set_ids or (), requested or ()):
            for raw_set_id in group or ():
                set_id = str(raw_set_id or "").strip()
                if set_id and set_id not in ordered:
                    ordered.append(set_id)
        return tuple(ordered)

    def _runtime_requested_show_labels_by_set_id(
        self,
        set_ids: Sequence[str],
    ) -> dict[str, str]:
        labels: dict[str, str] = {}
        for set_id in set_ids or ():
            sid = str(set_id or "").strip()
            if not sid:
                continue
            try:
                label = self.ui.batch.batch_set_name_for_id(sid)
            except Exception:
                label = None
            labels[sid] = str(label or sid)
        return labels

    def _runtime_input_epochs_for_sets(self, set_ids: Sequence[str]) -> dict[str, int]:
        epochs = {
            "global": int(getattr(self, "_authoritative_runtime_input_global_epoch", 0) or 0),
            "fallback": int(getattr(self, "_authoritative_runtime_input_epoch", 0) or 0),
        }
        for set_id, epoch in self._runtime_input_context_set_epochs(set_ids).items():
            epochs[f"set:{set_id}"] = int(epoch)
        return epochs

    def _runtime_prepared_request_is_current(self, prepared: PreparedRuntimeRequestSet) -> bool:
        intent = prepared.intent
        if intent is None:
            return False
        if str(intent.intent_kind or "") == "preview":
            if intent.preview_request_id is None or intent.preview_epoch is None:
                return False
            ownership = self._preview_ownership
            if ownership.request_id is None:
                return False
            if int(ownership.request_id) != int(intent.preview_request_id):
                return False
            if int(ownership.epoch) != int(intent.preview_epoch):
                return False
            expected_targets = tuple(str(set_id) for set_id in intent.set_ids if str(set_id))
            if expected_targets and tuple(ownership.target_set_ids) != expected_targets:
                return False
            return True
        request_token = intent.request_token
        if request_token is None:
            return True
        latest = int(getattr(self, "_latest_sim_request_id", 0) or 0)
        return int(request_token) >= latest

    def _accept_and_dispatch_prepared_runtime_request(
        self,
        prepared: PreparedRuntimeRequestSet,
    ) -> RuntimeDispatchPlan | None:
        dispatch_plan = self._runtime_orchestrator.accept_prepared_request(prepared)
        if dispatch_plan is None:
            return None
        self._dispatch_runtime_plan(dispatch_plan)
        return dispatch_plan

    def _dispatch_runtime_plan(self, dispatch_plan: RuntimeDispatchPlan) -> bool:
        result = self._runtime_dispatch_owner.dispatch(dispatch_plan)
        self._apply_runtime_lifecycle_ui_effects(result.effects)
        return bool(result.started)

    def _load_runtime_dispatch_context(
        self,
        *,
        dispatch_plan: RuntimeDispatchPlan,
        run_id: int,
        active: bool,
    ) -> Mapping[str, Any]:
        return self._batch_context_owner.load_runtime_dispatch_context(
            dispatch_plan=dispatch_plan,
            run_id=int(run_id),
            active=bool(active),
            runtime_input_epoch=int(getattr(self, "_authoritative_runtime_input_epoch", 0) or 0),
            runtime_input_global_epoch=int(getattr(self, "_authoritative_runtime_input_global_epoch", 0) or 0),
            runtime_input_set_epoch_by_set_id=self._runtime_input_context_set_epochs,
            label_for_set_id=self.ui.batch.batch_set_name_for_id,
        )

    def _deactivate_runtime_dispatch_context(self, context: Mapping[str, Any]) -> None:
        self._batch_context_owner.deactivate_if_active(context)

    def _callback_identity_for_runtime_descriptor(
        self,
        descriptor: RuntimeTaskDescriptor,
        *,
        dispatch_plan: RuntimeDispatchPlan,
        run_id: int,
        context: Mapping[str, Any],
    ) -> SimulationCallbackIdentity:
        try:
            submitted_plan = simulation_plan_from_payloadish(dict(descriptor.plan_payload or {}))
        except Exception:
            submitted_plan = None
        set_name = self.ui.batch.batch_set_name_for_id(descriptor.set_id) or descriptor.set_id
        assignment = dispatch_plan.assignment_for_task(descriptor.task_id)
        return SimulationCallbackIdentity.capture(
            run_id=int(run_id),
            fast_mode=dispatch_plan.launch_allocation.launch_intent.intent_kind == "preview",
            request_id=int(descriptor.request_token or 0),
            preview_owner_epoch=descriptor.preview_epoch,
            batch_set=str(set_name),
            batch_set_id=str(descriptor.set_id),
            cache_key=str(descriptor.cache_key),
            simulation_identity=(
                submitted_plan.simulation_identity_payload()
                if submitted_plan is not None
                else {}
            ),
            preview_batch_cache_token=descriptor.preview_batch_cache_token,
            launch_provenance=self._launch_provenance_for_plan(submitted_plan),
            allocation_id=str(dispatch_plan.launch_allocation.allocation_id),
            lane_id=str(assignment.lane_id if assignment is not None else ""),
            lane_generation=int(assignment.lane_generation if assignment is not None else 0),
            row=int(descriptor.row),
            exact_descriptor_hash=str(descriptor.exact_descriptor_hash),
            compatibility_key=descriptor.compatibility_key.to_payload(),
        )

    def _next_runtime_run_id(self) -> int:
        self._run_sequence_id = int(getattr(self, "_run_sequence_id", 0)) + 1
        self._active_run_id = int(self._run_sequence_id)
        return int(self._run_sequence_id)

    def stop_simulation(self) -> None:
        self._stop_simulation()

    def invalidate_slider_preview_work(self) -> None:
        self._invalidate_slider_preview_work()

    def slider_preview_work_intersects_target_scope(self, target_set_ids: Sequence[str]) -> bool:
        return bool(self._preview_work_intersects_runtime_input_scope(target_set_ids))

    def discard_slider_preview_work_preserving_runtime_owner(self) -> None:
        self._invalidate_slider_preview_work(close_runtime_owner=False)

    def invalidate_active_explicit_simulation_for_authoritative_change(self) -> None:
        self._invalidate_active_explicit_simulation_for_authoritative_change()

    def supersede_active_work_for_authoritative_mechanism_transition(
        self,
        *,
        epoch: int,
        affected_set_ids: Sequence[str] = (),
        close_preview_runtime_owner: bool = True,
    ) -> None:
        self._supersede_active_work_for_authoritative_mechanism_transition(
            epoch=int(epoch),
            affected_set_ids=affected_set_ids,
            close_preview_runtime_owner=bool(close_preview_runtime_owner),
        )

    def prepare_runtime_work_for_project_apply(self, *, epoch: int) -> None:
        self._supersede_active_work_for_authoritative_mechanism_transition(
            epoch=int(epoch),
            close_preview_runtime_owner=True,
        )
        self._apply_runtime_lifecycle_ui_effects(
            self._runtime_orchestrator.project_applied()
        )

    def poll_parallel_batch_completions(self) -> None:
        self._poll_parallel_batch_completions()

    def simulation_runtime_inputs_changed(
        self,
        *,
        batch_runtime_pool_inputs_changed: bool = True,
    ) -> SimulationRuntimeInputsChangeOutcome:
        return self._simulation_runtime_inputs_changed(
            batch_runtime_pool_inputs_changed=bool(batch_runtime_pool_inputs_changed)
        )

    def prepare_simulation_shutdown_for_close(self) -> bool:
        return self._prepare_simulation_shutdown_for_close()

    def flush_slider_plot_updates(
        self,
        *,
        force: bool = False,
        cache_key: Optional[str] = None,
        request_id: Optional[int] = None,
        run_id: Optional[int] = None,
    ) -> bool:
        return bool(
            self._flush_slider_plot_updates(
                force=force,
                cache_key=cache_key,
                request_id=request_id,
                run_id=run_id,
            )
        )


    def on_simulation_progress(self, percent: int, message: str) -> None:
        self._on_simulation_progress(percent, message)

    def on_simulation_complete(
        self,
        result: dict,
        *,
        callback_identity: SimulationCallbackIdentity,
    ):
        return self._on_simulation_complete(
            result,
            callback_identity=callback_identity,
        )

    def on_simulation_error(
        self,
        error_msg: object,
        *,
        callback_identity: SimulationCallbackIdentity,
    ) -> None:
        self._on_simulation_error(
            error_msg,
            callback_identity=callback_identity,
        )

    def _launch_provenance_for_plan(self, plan: SimulationPlan | None) -> dict[str, Any]:
        if plan is None:
            return {}
        request = plan.to_execution_request()
        solver_config = dict(request.solver_config or {})
        provenance: dict[str, Any] = {}
        if solver_config.get("temperature_K") is not None:
            provenance["temperature_K"] = float(solver_config["temperature_K"])
        provenance["mechanism_text"] = str(request.mechanism_text or "")
        try:
            provenance["simulation_time"] = float(request.t_span[1])
        except Exception:
            pass
        grid = solver_config.get("grid")
        if isinstance(grid, Mapping) and grid.get("N") is not None:
            provenance["num_points_requested"] = int(grid["N"])
        try:
            provenance["temperature_source"] = (
                "dsl"
                if self.ui.solver.dsl_global_temperature_K(str(request.mechanism_text or "")) is not None
                else "ui"
            )
        except Exception:
            provenance["temperature_source"] = "ui"
        return provenance

    def _record_nonfatal_exception(self, context: str, exc: BaseException) -> None:
        self._nonfatal_exception_count += 1
        self._last_nonfatal_exception = f"{context}: {type(exc).__name__}: {exc}"
        logger.exception("%s", self._last_nonfatal_exception)

    # ------------------------------------------------------------------
    # Cache API
    # ------------------------------------------------------------------
    def set_simulation_cache_caps(
        self,
        *,
        result_cap: int,
        preview_cap: int,
        persist: bool = True,
    ) -> SimulationCacheOpResult:
        return self._cache_admin.set_caps(result_cap=result_cap, preview_cap=preview_cap, persist=persist)

    def simulation_cache_stats(self) -> SimulationCacheOpResult:
        """Return cache usage stats for UI display."""
        return self._cache_admin.stats()

    def purge_simulation_result_cache(self) -> SimulationCacheOpResult:
        return self._cache_admin.purge_result_cache()

    def purge_simulation_preview_cache(self) -> SimulationCacheOpResult:
        return self._cache_admin.purge_preview_cache()

    def purge_simulation_all_caches(self) -> SimulationCacheOpResult:
        return self._cache_admin.purge_all_caches()

    # ------------------------------------------------------------------
    # Lane-pool lifecycle
    # ------------------------------------------------------------------

    def _has_active_explicit_simulation(self) -> bool:
        return self._completion_policy.has_active_explicit_simulation(
            activity=self._completion_policy_activity_snapshot(),
            context=self._batch_context_owner.completion_policy_context(),
        )

    def _has_active_fast_preview_in_flight(self) -> bool:
        return self._completion_policy.has_active_fast_preview_in_flight(
            activity=self._completion_policy_activity_snapshot(),
            context=self._batch_context_owner.completion_policy_context(),
        )

    def _stale_fast_request_still_owns_current_state(self, request_id: int) -> bool:
        return self._completion_policy.stale_fast_request_still_owns_current_state(
            preview_ownership=self._completion_policy_preview_ownership(),
            request_id=int(request_id),
        )

    def _prepare_simulation_shutdown_for_close(self) -> bool:
        self._apply_runtime_lifecycle_ui_effects(
            self._runtime_orchestrator.close_requested(force_terminate=True)
        )
        self._shutdown_requested_for_close = False
        return True

    def _clear_shutdown_request_after_close_cleanup(self) -> None:
        self._shutdown_requested_for_close = False

    def _contained_child_blas_threads_limited(self) -> bool:
        return bool(self.parallel_batch.limit_blas_threads_per_worker)

    def _contained_child_handler_env(self) -> Dict[str, str]:
        return contained_child_blas_thread_env(
            enabled=self._contained_child_blas_threads_limited()
        )

    def _runtime_lane_environment_key(self) -> str:
        blas_limited = bool(self._contained_child_blas_threads_limited())
        return f"contained-child-blas:{'limited' if blas_limited else 'unlimited'}"

    def _new_contained_simulation_owner(
        self,
        *,
        fast_mode: bool,
        simulation_plan_payload: Mapping[str, Any],
    ):
        owner_plan_payload = dict(simulation_plan_payload or {})
        factory = getattr(self, "_contained_simulation_owner_factory", None)
        if callable(factory):
            try:
                return factory(
                    fast_mode=bool(fast_mode),
                    simulation_plan_payload=owner_plan_payload,
                )
            except TypeError:
                return factory(fast_mode=bool(fast_mode))
        from kindred.core.simulation_containment import WarmSimulationOwner

        timeout_s = getattr(self, "_contained_simulation_timeout_s", None)
        kwargs: Dict[str, Any] = {}
        if timeout_s is not None:
            kwargs["active_timeout_s"] = float(timeout_s)
        kwargs["handler_env"] = self._contained_child_handler_env()
        return WarmSimulationOwner(owner_plan_payload, **kwargs)

    def _interactive_runtime_rows(self, rows: Optional[Sequence[int]] = None) -> list[int]:
        if rows is not None:
            normalized: list[int] = []
            seen: set[int] = set()
            try:
                row_count = int(self.ui.batch.batch_store_row_count())
            except Exception:
                row_count = 0
            for row in rows or ():
                try:
                    row_i = int(row)
                except (TypeError, ValueError):
                    continue
                if not (0 <= row_i < row_count) or row_i in seen:
                    continue
                normalized.append(row_i)
                seen.add(row_i)
            return normalized
        try:
            rows = list(self.ui.batch.batch_rows_for_scope("selected"))
        except Exception:
            rows = []
        return [int(row) for row in rows]

    def _parallel_batch_pool_settings_changed(self) -> None:
        self._apply_runtime_lifecycle_ui_effects(
            self._runtime_orchestrator.pool_settings_changed()
        )

    def _cancel_runtime_for_parallel_outcome_reset(self) -> None:
        self._apply_runtime_lifecycle_ui_effects(
            self._runtime_orchestrator.cancel_requested(kind="parallel_outcome_reset")
        )

    def _simulation_runtime_inputs_changed(
        self,
        *,
        batch_runtime_pool_inputs_changed: bool = True,
    ) -> SimulationRuntimeInputsChangeOutcome:
        if bool(batch_runtime_pool_inputs_changed):
            self._parallel_batch_pool_settings_changed()
        self.refresh_interactive_runtime_readiness()
        return SimulationRuntimeInputsChangeOutcome(
            interactive_runtime_refresh_requested=True,
        )

    def refresh_interactive_runtime_readiness(
        self,
        rows: Optional[Sequence[int]] = None,
    ) -> bool:
        runtime_rows = self._interactive_runtime_rows(rows)
        target_set_ids = self._run_target_set_ids_for_rows(runtime_rows)
        if not runtime_rows or not target_set_ids:
            return bool(
                self._runtime_orchestrator.refresh_readiness(
                    self._blocked_interactive_runtime_readiness_request(
                        runtime_rows=runtime_rows,
                        target_set_ids=target_set_ids,
                    )
                )
            )
        intent = RuntimeLaunchIntent(
            intent_kind="ordinary",
            ui_action="runtime_readiness",
            rows=tuple(runtime_rows),
            set_ids=target_set_ids,
            requested_show_set_ids=self._runtime_requested_show_set_ids(target_set_ids),
            requested_show_labels_by_set_id=self._runtime_requested_show_labels_by_set_id(target_set_ids),
            request_token=None,
            runtime_input_epochs=self._runtime_input_epochs_for_sets(target_set_ids),
        )
        prepared = self._run_preparation_owner.prepare_runtime_request_set(
            intent=intent,
            fast_mode=False,
        )
        launch_available = self._runtime_orchestrator.refresh_readiness(prepared)
        self._prewarm_interactive_preview_runtime_readiness(
            runtime_rows=runtime_rows,
            target_set_ids=target_set_ids,
        )
        return bool(launch_available)

    def _blocked_interactive_runtime_readiness_request(
        self,
        *,
        runtime_rows: Sequence[int],
        target_set_ids: Sequence[str],
    ) -> PreparedRuntimeRequestSet:
        normalized_targets = tuple(str(set_id) for set_id in target_set_ids or () if str(set_id))
        intent = RuntimeLaunchIntent(
            intent_kind="ordinary",
            ui_action="runtime_readiness",
            rows=tuple(runtime_rows or ()),
            set_ids=normalized_targets,
            requested_show_set_ids=(),
            requested_show_labels_by_set_id={},
            request_token=None,
            runtime_input_epochs=(
                self._runtime_input_epochs_for_sets(normalized_targets)
                if normalized_targets
                else {}
            ),
        )
        return PreparedRuntimeRequestSet(
            intent=intent,
            compatibility_key=RuntimeCompatibilityKey(
                structural_digest="",
                execution_profile="explicit",
                environment_key=self._runtime_lane_environment_key(),
                schema_key="",
            ),
            blocked_reason=RuntimePreparationBlockedReason(
                source="selection",
                code="no_targets",
                message="Select at least one set before running.",
                rows=tuple(runtime_rows or ()),
                set_ids=normalized_targets,
                retryable=False,
            ),
        )

    def _prewarm_interactive_preview_runtime_readiness(
        self,
        *,
        runtime_rows: Sequence[int],
        target_set_ids: Sequence[str],
    ) -> bool:
        if not runtime_rows or not target_set_ids:
            return False
        prewarm = getattr(
            self._runtime_orchestrator,
            "prewarm_compatible_runtime_lanes",
            None,
        )
        if not callable(prewarm):
            return False
        intent = RuntimeLaunchIntent(
            intent_kind="preview",
            ui_action="slider_preview_readiness",
            rows=tuple(runtime_rows),
            set_ids=tuple(str(set_id) for set_id in target_set_ids if str(set_id)),
            requested_show_set_ids=self._runtime_requested_show_set_ids(target_set_ids),
            requested_show_labels_by_set_id=self._runtime_requested_show_labels_by_set_id(target_set_ids),
            request_token=None,
            runtime_input_epochs=self._runtime_input_epochs_for_sets(target_set_ids),
        )
        try:
            prepared = self._run_preparation_owner.prepare_runtime_request_set(
                intent=intent,
                fast_mode=True,
            )
        except Exception as exc:
            self._record_nonfatal_exception(
                "Failed to prepare preview runtime readiness prewarm",
                exc,
            )
            return False
        return bool(prewarm(prepared))

    def _apply_runtime_lifecycle_ui_effects(
        self,
        effects: RuntimeUiEffect | Sequence[RuntimeUiEffect] | None,
    ) -> None:
        if effects is None:
            return
        effect_items = tuple(effects) if isinstance(effects, (list, tuple)) else (effects,)
        for effect in effect_items:
            if bool(getattr(effect, "stop_completion_polling", False)):
                timer = getattr(self, "_batch_completion_poll_timer", None)
                if timer is not None:
                    timer.stop()
            if bool(getattr(effect, "start_completion_polling", False)):
                self._start_batch_completion_poll_timer()
            render_state = getattr(effect, "render_state", None)
            if render_state is not None:
                self.runtime_readiness_render_requested.emit(render_state)
            surface_failure = str(getattr(effect, "surface_failure", "") or "")
            if surface_failure:
                try:
                    self._surface_current_parallel_batch_pool_failure_to_ui(
                        f"Simulation failed:\n\n{surface_failure}"
                    )
                except Exception as exc:
                    self._record_nonfatal_exception("Failed to surface runtime failure effect to UI", exc)
            if getattr(effect, "simulation_running", None) is not None:
                self._simulation_running = bool(effect.simulation_running)
            if getattr(effect, "slider_simulation_active", None) is not None:
                self._slider_simulation_active = bool(effect.slider_simulation_active)
            if getattr(effect, "run_enabled", None) is not None:
                self.ui.run_ui.set_run_button_enabled(bool(effect.run_enabled))
            if getattr(effect, "stop_enabled", None) is not None:
                self.ui.run_ui.set_stop_button_enabled(bool(effect.stop_enabled))
            if getattr(effect, "progress_value", None) is not None:
                self.ui.run_ui.set_sim_progress_value(int(effect.progress_value))
            if getattr(effect, "status_text", None) is not None:
                self.ui.run_ui.set_status_text(str(effect.status_text))
            if bool(getattr(effect, "stop_debounce_timers", False)):
                self._stop_slider_debounce_timers_for_lifecycle_effect()
            if bool(getattr(effect, "clear_preview_replay", False)):
                self._clear_pending_slider_preview_replay_state(
                    clear_plot_updates=bool(getattr(effect, "clear_preview_plot_updates", False))
                )
            replay_update = getattr(effect, "set_preview_replay", None)
            if replay_update is not None:
                self._set_pending_slider_preview_replay_state(replay_update.state)
                if bool(getattr(replay_update, "clear_plot_updates", False)):
                    self._clear_pending_preview_slider_plot_updates()
            replay = getattr(effect, "queue_preview_replay", None)
            if replay is not None:
                self._set_pending_slider_preview_replay_state(
                    replace(
                        self._pending_slider_preview_launch,
                        active=True,
                        request_id=int(replay.request_id),
                        target_set_ids=tuple(replay.target_set_ids or ()),
                        handoff_queued=True,
                    )
                )
                if bool(replay.stop_timers):
                    self._stop_deferred_preview_replay_timers()
                QtCore.QTimer.singleShot(0, self.launch_pending_slider_preview_replay)

    def _supersede_parallel_batch_run_soft(self) -> None:
        """
        Supersede the active parallel run without destroying the process pool.

        Used by slider-triggered restarts to preserve worker processes and avoid
        pool recreation on every minor parameter update.
        """
        state = self._batch_context_owner.active_batch_state()
        if state is not None and state.active and (state.runtime_task_queue or state.parallel):
            self._batch_context_owner.deactivate()

        self._clear_pending_slider_plot_updates()

        self._apply_runtime_lifecycle_ui_effects(
            self._runtime_orchestrator.cancel_requested(kind="soft_supersede")
        )

        if bool(getattr(self, "_debug_batch_parallel", False)):
            logger.info("BATCH_PAR soft-supersede requested")

    # ------------------------------------------------------------------
    # Plot coalescing (cache-backed)
    # ------------------------------------------------------------------
    def _clear_pending_slider_plot_updates(self) -> None:
        self._plot_coalescer.clear()

    def _clear_pending_preview_slider_plot_updates(self) -> None:
        pending = getattr(self._plot_coalescer, "pending", None)
        if pending is None:
            self._plot_coalescer.clear()
            return
        cache_kind = str(getattr(pending, "cache_kind", "") or "").strip().lower()
        if cache_kind in ("", "preview"):
            self._plot_coalescer.clear()

    def _invalidate_slider_preview_work(self, *, close_runtime_owner: bool = True) -> None:
        invalidation_request_id = int(self._next_sim_request_id())
        self._discarded_slider_preview_generation_id = int(invalidation_request_id)
        self._clear_preview_ownership()
        self.clear_pending_slider_preview_replay(clear_plot_updates=False)
        self._clear_pending_preview_slider_plot_updates()
        self.ui.batch.clear_active_preview_cache_identity_state()
        state = self._batch_context_owner.active_batch_state()
        if (
            state is not None
            and state.active
            and (state.runtime_task_queue or state.parallel)
            and state.fast_mode
        ):
            self._supersede_parallel_batch_run_soft()
        has_active_explicit_simulation = self._has_active_explicit_simulation()
        self.ui.slider.set_slider_triggered_simulation(False)
        self._slider_simulation_active = False
        if has_active_explicit_simulation:
            return
        self._simulation_running = False
        try:
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self.ui.run_ui.set_status_text("Ready")
            self.ui.run_ui.set_sim_progress_value(0)
        except Exception as exc:
            self._record_nonfatal_exception(
                "Failed to reset Run/Stop/status/progress after invalidating slider preview work",
                exc,
            )

    def _invalidate_active_explicit_simulation_for_authoritative_change(self) -> None:
        if not self._has_active_explicit_simulation():
            return
        self._run_sequence_id = int(getattr(self, "_run_sequence_id", 0)) + 1
        self._active_run_id = int(self._run_sequence_id)
        self._cancel_active_run_for_restart()

    @staticmethod
    def _normalize_runtime_input_set_ids(set_ids: Sequence[str] | None) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(set_id) for set_id in (set_ids or ()) if str(set_id)))

    def _runtime_input_set_epoch(self, set_id: str) -> int:
        set_id_s = str(set_id or "").strip()
        if not set_id_s:
            return 0
        epochs = getattr(self, "_authoritative_runtime_input_set_epoch_by_set_id", {}) or {}
        try:
            return int(epochs.get(set_id_s, 0) or 0)
        except Exception:
            return 0

    def _runtime_input_context_set_epochs(self, set_ids: Sequence[str]) -> Dict[str, int]:
        return {
            str(set_id): self._runtime_input_set_epoch(str(set_id))
            for set_id in self._normalize_runtime_input_set_ids(set_ids)
        }

    def _active_explicit_runtime_task_set_ids(self) -> tuple[str, ...]:
        state = self._batch_context_owner.active_batch_state()
        if state is None or not state.active or state.fast_mode:
            return ()
        return tuple(str(set_id) for set_id in state.queue_ids if str(set_id))

    def _preview_work_intersects_runtime_input_scope(self, affected_set_ids: Sequence[str]) -> bool:
        affected_scope = set(self._normalize_runtime_input_set_ids(affected_set_ids))
        if not affected_scope:
            return True

        preview_ownership = self._preview_ownership
        if preview_ownership.request_id is not None:
            owner_targets = set(preview_ownership.target_set_ids)
            if not owner_targets:
                return True
            if affected_scope & owner_targets:
                return True

        pending_preview = self._pending_slider_preview_launch
        if pending_preview.active:
            pending_targets = set(pending_preview.target_set_ids)
            if not pending_targets:
                return True
            if affected_scope & pending_targets:
                return True

        if self._pending_slider_plot_cache_kind == "preview":
            pending_plot_targets = set(self._pending_slider_plot_set_ids)
            if not pending_plot_targets:
                return True
            if affected_scope & pending_plot_targets:
                return True

        state = self._batch_context_owner.active_batch_state()
        if state is not None and state.active and state.fast_mode:
            context_targets = set(self._batch_context_owner.active_fast_preview_scope_set_ids() or ())
            if not context_targets:
                return True
            if affected_scope & context_targets:
                return True

        return False

    def _supersede_active_work_for_authoritative_mechanism_transition(
        self,
        *,
        epoch: int,
        affected_set_ids: Sequence[str] = (),
        close_preview_runtime_owner: bool = True,
    ) -> None:
        self._authoritative_mechanism_transition_epoch = int(epoch)
        self._authoritative_runtime_input_epoch = int(epoch)
        affected_scope = self._normalize_runtime_input_set_ids(affected_set_ids)
        if affected_scope:
            epochs = dict(getattr(self, "_authoritative_runtime_input_set_epoch_by_set_id", {}) or {})
            for set_id in affected_scope:
                epochs[str(set_id)] = int(epoch)
            self._authoritative_runtime_input_set_epoch_by_set_id = epochs
            active_set_ids = set(self._active_explicit_runtime_task_set_ids())
            if active_set_ids and active_set_ids.intersection(set(affected_scope)):
                self._invalidate_active_explicit_simulation_for_authoritative_change()
        else:
            self._authoritative_runtime_input_global_epoch = int(epoch)
            self._invalidate_active_explicit_simulation_for_authoritative_change()
        if (not affected_scope) or self._preview_work_intersects_runtime_input_scope(affected_scope):
            self._invalidate_slider_preview_work(close_runtime_owner=bool(close_preview_runtime_owner))

    def _queue_slider_plot_update(
        self,
        *,
        set_id: Optional[str],
        cache_key: Optional[str],
        request_id: Optional[int],
        run_id: Optional[int],
        slider_triggered: bool = True,
        valid_set_ids: Optional[Sequence[str]] = None,
        fresh_preview_entry: Optional[FreshPreviewDisplayEntry] = None,
    ) -> None:
        preview_ownership = self._preview_ownership
        self._plot_coalescer.queue(
            set_id=set_id,
            cache_key=cache_key,
            request_id=request_id,
            run_id=run_id,
            slider_triggered=slider_triggered,
            preview_request_id=preview_ownership.request_id,
            preview_owner_epoch=preview_ownership.epoch,
            preview_target_set_ids=preview_ownership.target_set_ids,
            latest_request_id=int(getattr(self, "_latest_sim_request_id", 0)),
            valid_set_ids=valid_set_ids,
            fresh_preview_entry=fresh_preview_entry,
            active_run_id=int(self._active_run_id),
            record_nonfatal_exception=self._record_nonfatal_exception,
        )

    def _flush_slider_plot_updates(
        self,
        *,
        force: bool = False,
        cache_key: Optional[str] = None,
        request_id: Optional[int] = None,
        run_id: Optional[int] = None,
    ) -> bool:
        pending = self._plot_coalescer.take_pending()
        pending_set_ids = set(pending.set_ids)

        refresh = self.ui.results.publish_runtime_slider_replay_display_from_pending(
            cache_admin=self._cache_admin,
            pending=pending,
            cache_key=cache_key,
            request_id=request_id,
            run_id=run_id,
            current_preview_request_id=self._preview_ownership.request_id,
            current_preview_owner_epoch=self._preview_ownership.epoch,
            latest_request_id=int(getattr(self, "_latest_sim_request_id", 0)),
            active_run_id=int(getattr(self, "_active_run_id", 0)),
        )
        displayed = bool(getattr(refresh, "displayed", False))
        focused_controls_use_workspace = getattr(refresh, "focused_controls_use_workspace", None)
        if focused_controls_use_workspace is not None:
            try:
                self.ui.mechanism_helpers.sync_mechanism_controls_to_focused_batch_set(
                    use_workspace=bool(focused_controls_use_workspace)
                )
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to resync focused mechanism controls after preview display refresh",
                    exc,
                )
        if bool(getattr(self, "_debug_batch_parallel", False)):
            logger.info(
                "BATCH_PAR plot flush run_id=%s request_id=%s changed_sets=%s forced=%s displayed=%s reason=%s ts=%.6f",
                int(run_id or 0),
                int(request_id or 0),
                int(len(pending_set_ids)),
                bool(force),
                bool(displayed),
                str(getattr(refresh, "log_reason", "")),
                float(perf_counter()),
            )
        return bool(displayed)

    # ------------------------------------------------------------------
    # Batch lane outcome polling/consumption
    # ------------------------------------------------------------------
    def _surface_current_parallel_batch_pool_failure_to_ui(self, error_msg: object) -> None:
        dispatch_context = self._batch_context_owner.active_parallel_error_dispatch_context()
        if dispatch_context is None:
            return
        callback_identity = SimulationCallbackIdentity.capture(
            run_id=int(dispatch_context.run_id),
            fast_mode=bool(dispatch_context.fast_mode),
            request_id=int(dispatch_context.request_id),
            preview_owner_epoch=dispatch_context.preview_owner_epoch,
            batch_set="",
            batch_set_id="",
            cache_key=str(dispatch_context.cache_key),
            simulation_identity=dispatch_context.simulation_identity,
            preview_batch_cache_token="",
        )
        self._on_simulation_error(
            error_msg,
            callback_identity=callback_identity,
        )

    def _scoped_batch_failure_detail_lines(
        self,
        *,
        failed_set_ids: Iterable[str],
        failed_errors: Mapping[str, Any],
    ) -> list[str]:
        detail_lines: list[str] = []
        for failed_id in sorted(str(set_id) for set_id in failed_set_ids if str(set_id)):
            failed = coerce_simulation_failure(failed_errors.get(failed_id) or {})
            try:
                failed_name = str(self.ui.batch.batch_set_name_for_id(failed_id) or failed_id)
            except Exception:
                failed_name = str(failed_id)
            detail_lines.append(f"{failed_name}: {simulation_failure_user_message(failed)}")
        return detail_lines

    def _show_scoped_batch_failure_summary(
        self,
        *,
        failed_set_ids: Iterable[str],
        failed_errors: Mapping[str, Any],
    ) -> None:
        failed_ids = [str(set_id) for set_id in failed_set_ids if str(set_id)]
        failed_count = len(failed_ids)
        if failed_count <= 0:
            return
        detail_lines = self._scoped_batch_failure_detail_lines(
            failed_set_ids=failed_ids,
            failed_errors=failed_errors,
        )
        self.ui.dialogs.message_box_critical(
            "Batch Simulation Error",
            f"Batch completed with {failed_count} failed set(s).",
            details="\n".join(detail_lines) if detail_lines else None,
        )

    def _current_mechanism_species_for_batch_sync(self) -> list[str]:
        try:
            last_mech = self.ui.mechanism_helpers.last_mechanism()
            if last_mech is not None and hasattr(last_mech, "species_names"):
                species_names = [str(name) for name in (last_mech.species_names() or ()) if str(name)]
                if species_names:
                    return species_names
        except Exception:
            pass
        try:
            mechanism_text = str(self.ui.mechanism.get_mechanism_text() or "")
            if not mechanism_text.strip():
                return []
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism
            from kindred.core.units import UnitsModel

            temperature_K = self.ui.solver.dsl_global_temperature_K(mechanism_text)
            if temperature_K is None:
                temperature_K = float(self.ui.solver.temperature_spinbox_value())
            mechanism = parse_dsl_to_mechanism(
                mechanism_text,
                initials={},
                units=UnitsModel(temperature_K=float(temperature_K), energy_unit="kJ/mol"),
            )
            return [str(name) for name in (mechanism.species_names() or ()) if str(name)]
        except Exception as exc:
            self._record_nonfatal_exception(
                "Failed to resolve mechanism species after partial staged overlay reset",
                exc,
            )
            return []

    def _finalize_explicit_batch_dirty_reset(
        self,
        ctx: Mapping[str, Any],
        *,
        mechanism: object = None,
        species_names: Sequence[str] = (),
    ) -> dict[str, Any]:
        policy_context = self._batch_context_owner.completion_policy_context(ctx)
        reset_target_set_ids = self._batch_context_owner.pending_dirty_reset_state(ctx).set_ids
        dirty_reset_decision = self._completion_policy.resolve_explicit_dirty_reset(
            context=policy_context or self._batch_context_owner.completion_policy_context(ctx),
            dirty_state_by_set_id=self._capture_dirty_state_by_set_id(reset_target_set_ids),
        )
        eligible_reset_set_ids = list(dirty_reset_decision.eligible_reset_set_ids)
        workspaces_cleared = False
        if eligible_reset_set_ids:
            try:
                workspaces_cleared = bool(
                    self.ui.slider.reset_mechanism_workspaces(eligible_reset_set_ids)
                )
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to clear targeted slider workspaces after canonical explicit run",
                    exc,
                )
        overlays_cleared = False
        if eligible_reset_set_ids:
            try:
                overlays_cleared = bool(
                    self.ui.slider.discard_concentration_overlays_for_set_ids(
                        eligible_reset_set_ids
                    )
                )
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to clear staged concentration overlays after canonical explicit run",
                    exc,
                )
        policy_context = self._apply_completion_policy_state_patch(
            dirty_reset_decision.state_patch,
            base_context=ctx if isinstance(ctx, Mapping) else None,
        ) or policy_context
        if overlays_cleared:
            try:
                species_for_sync = (
                    list(mechanism.species_names())
                    if mechanism is not None and hasattr(mechanism, "species_names")
                    else list(species_names)
                )
                if species_for_sync:
                    self.ui.batch.sync_batch_species_columns(
                        species_for_sync,
                        retain_active_cache_identity=True,
                    )
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to refresh batch/species surfaces after clearing staged concentration overlays",
                    exc,
                )
        if workspaces_cleared or overlays_cleared:
            self._apply_runtime_lifecycle_ui_effects(
                self._runtime_orchestrator.preview_replay_after_canonical_reset(
                    current_state=self._runtime_preview_replay_state(),
                    reset_set_ids=tuple(eligible_reset_set_ids),
                )
            )
        if eligible_reset_set_ids:
            try:
                self.ui.mechanism_helpers.sync_mechanism_controls_to_focused_batch_set(
                    use_workspace=True
                )
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to resync focused mechanism controls after canonical explicit run",
                    exc,
                )
        if policy_context is not None:
            return self._batch_context_owner.serialize_completion_policy_context(
                policy_context,
                base_context=ctx if isinstance(ctx, Mapping) else None,
            )
        return dict(ctx or {})

    def _finalize_batch_queue_done_without_result(
        self,
        ctx: Mapping[str, Any],
        *,
        status_text: str = "Simulation complete",
    ) -> None:
        shutdown_requested = bool(getattr(self, "_shutdown_requested_for_close", False))
        ctx = self._batch_context_owner.deactivate_if_active(ctx)
        try:
            cleanup_state = self._batch_context_owner.completion_cleanup_state(ctx)
            if not cleanup_state.fast_mode:
                ctx = self._finalize_explicit_batch_dirty_reset(
                    ctx,
                    species_names=self._current_mechanism_species_for_batch_sync(),
                )
            summary = self._batch_context_owner.completion_summary(ctx)
            self._apply_simulation_lifecycle_effects(
                self._lifecycle_effect_owner.completion_without_result_ui_effects(
                    summary=summary,
                    status_text=str(status_text),
                )
            )
            if summary.failed_set_ids and not summary.fast_mode:
                self._show_scoped_batch_failure_summary(
                    failed_set_ids=summary.failed_set_ids,
                    failed_errors=summary.failed_errors,
                )
        finally:
            cleanup_state = self._batch_context_owner.completion_cleanup_state(ctx)
            self._apply_simulation_lifecycle_effects(
                SimulationLifecycleEffects(reset_slider_triggered=True)
            )
            replay_effects = self._runtime_orchestrator.completion_without_result_finalized(
                pending_state=self._runtime_preview_replay_state(),
                shutdown_requested=bool(shutdown_requested),
                display_kind=self._lifecycle_effect_owner.runtime_display_kind_for_cleanup(
                    cleanup_state
                ),
            )
            if replay_effects:
                logger.debug("Processing pending slider update after completion")
                self._apply_runtime_lifecycle_ui_effects(replay_effects)
            self._clear_shutdown_request_after_close_cleanup()

    def _finalize_scoped_batch_success_subset(
        self,
        ctx: Mapping[str, Any],
    ) -> DisplayTransitionOutcome | None:
        if not isinstance(ctx, Mapping):
            return None
        policy_context = self._batch_context_owner.completion_policy_context(ctx)
        if policy_context is None:
            return None
        ctx = self._finalize_explicit_batch_dirty_reset(
            ctx,
            species_names=self._current_mechanism_species_for_batch_sync(),
        )
        outcome = self._completion_publication_owner.publish_completed_run_display_for_context(ctx)
        if not isinstance(outcome, SimulationCompletionDisplayOutcome):
            raise TypeError("Completed-run display publication must return SimulationCompletionDisplayOutcome")
        return outcome.transition_outcome

    def _try_handle_scoped_batch_failure(
        self,
        *,
        set_id: str,
        set_name: str,
        error_payload: Mapping[str, Any],
    ) -> bool:
        return self._parallel_batch_outcome_owner.handle_scoped_failure(
            set_id=set_id,
            set_name=set_name,
            error_payload=error_payload,
        )

    def _consume_parallel_batch_outcome(
        self,
        *,
        set_id: str,
        outcome: BatchLaneOutcome,
        source: str,
        completed_ts: Optional[float] = None,
        completion_record: Optional[BatchCompletionRecord] = None,
    ) -> bool:
        return self._parallel_batch_outcome_owner.consume_outcome(
            set_id=set_id,
            outcome=outcome,
            source=str(source),
            completed_ts=completed_ts,
            completion_record=completion_record,
            debug_batch_parallel=bool(getattr(self, "_debug_batch_parallel", False)),
        )

    def _poll_parallel_batch_completions(self) -> None:
        try:
            effects = self._runtime_orchestrator.consume_progress_tick()
        except Exception as exc:
            self._record_nonfatal_exception("Unhandled exception while polling parallel batch completions", exc)
            self._apply_runtime_lifecycle_ui_effects(
                self._runtime_orchestrator.polling_failed(str(exc))
            )
            return
        try:
            self._apply_runtime_lifecycle_ui_effects(effects)
        except Exception as exc:
            self._record_nonfatal_exception(
                "Unhandled exception while applying runtime progress effects",
                exc,
            )

    # ------------------------------------------------------------------
    # Simulation request ids
    # ------------------------------------------------------------------
    def _next_sim_request_id(self) -> int:
        """Return a new monotonically increasing simulation request id."""
        return int(self._run_state.next_request_id())

    def _next_slider_preview_request_id(self) -> int:
        if not self._has_active_fast_preview_in_flight():
            return int(self._next_sim_request_id())

        latest_request_id = int(getattr(self, "_latest_sim_request_id", 0) or 0)
        pending_request_id = getattr(self, "_pending_slider_sim_request_id", None)
        if pending_request_id is not None and int(pending_request_id) > latest_request_id:
            return int(pending_request_id)

        reserve_request_id = getattr(self._run_state, "reserve_request_id", None)
        if callable(reserve_request_id):
            reserved_request_id = int(reserve_request_id())
            if reserved_request_id > latest_request_id:
                return reserved_request_id
        synced_request_id = int(max(int(getattr(self._run_state, "sim_request_id", 0) or 0), latest_request_id) + 1)
        self._sim_request_id = synced_request_id
        return synced_request_id

    def _flush_pending_slider_updates_for_run(self, *, reset_set_ids: Sequence[str] = ()) -> None:
        """
        Ensure Run starts from the latest committed slider state and does not leave
        stale slider-triggered simulations queued behind it.
        """
        _ = reset_set_ids
        self.ui.slider.stop_slider_release_commit_timer()
        if self.ui.slider.has_pending_slider_values():
            try:
                self.ui.slider.finalize_slider_release_commit()
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to finalize slider release commit before Run",
                    exc,
                )

        self.ui.slider.stop_variable_update_timer()
        self.ui.slider.stop_species_slider_update_timer()
        self._clear_pending_preview_slider_plot_updates()

        self.ui.slider.set_slider_triggered_simulation(False)

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    def _slider_target_rows_for_dispatch(
        self,
        candidate_rows: Sequence[int],
        *,
        target_set_ids: Optional[Sequence[str]] = None,
    ) -> list[int]:
        _ = candidate_rows
        snapshot_set_ids = [str(set_id) for set_id in (target_set_ids or ()) if str(set_id)]
        if not snapshot_set_ids:
            return []

        rows_by_set_id: Dict[str, int] = {}
        try:
            row_count = int(self.ui.batch.batch_store_row_count())
        except Exception:
            row_count = 0
        for row in range(max(0, row_count)):
            try:
                set_id = self.ui.batch.batch_set_id_for_row(int(row))
            except Exception:
                continue
            set_id_s = str(set_id or "").strip()
            if set_id_s and set_id_s not in rows_by_set_id:
                rows_by_set_id[set_id_s] = int(row)

        resolved_rows: list[int] = []
        seen_rows: set[int] = set()
        for set_id in snapshot_set_ids:
            row = rows_by_set_id.get(str(set_id))
            if row is None or row in seen_rows:
                continue
            seen_rows.add(int(row))
            resolved_rows.append(int(row))
        return resolved_rows

    def _execution_identity_flags(self, *, fast_mode: bool) -> tuple[str, ...]:
        return ("fast_mode",) if bool(fast_mode) else ()

    def _runtime_symbolic_jacobian_identity(
        self,
        *,
        set_id: str,
        solver_config: Mapping[str, Any],
        fast_mode: bool,
    ) -> Mapping[str, Any]:
        solver_name = str(dict(solver_config or {}).get("solver") or "").strip().lower()
        if solver_name not in {"bdf", "radau"}:
            return {}
        if not bool(dict(solver_config or {}).get("use_sparse_jacobian", False)):
            return {}
        try:
            mechanism_source = self.ui.mechanism.mechanism_source_for_run_set(
                self.ui.mechanism.mechanism_source_for_run(fast_mode=bool(fast_mode)),
                set_id=str(set_id),
                apply_parameter_overrides=False,
                strip_initial_concentrations=True,
            )
            mechanism_text = mechanism_source.full_dsl
            parameter_overrides = (
                self._runtime_parameter_values_for_set(set_id=str(set_id))
                if bool(fast_mode) and self.ui.mechanism.has_local_runtime_parameter_values()
                else {}
            )
            from kindred.core.simulation_preparation import (
                symbolic_jacobian_identity_for_execution_text,
            )

            payload = symbolic_jacobian_identity_for_execution_text(
                mechanism_text=str(mechanism_text or ""),
                solver_config=dict(solver_config or {}),
                parameter_overrides=parameter_overrides,
            )
            if not payload:
                return {}
            return dict(payload)
        except (ValueError, TypeError):
            raise
        except Exception as exc:
            if (
                exc.__class__.__name__ == "SimulationPreparationError"
                and str(getattr(exc, "stage", "")) == "parameter_overrides"
            ):
                raise
            return {}

    def _symbolic_wegscheider_identity_for_set(
        self,
        *,
        set_id: str,
        solver_config: Mapping[str, Any],
        fast_mode: bool,
    ) -> Mapping[str, Any]:
        if not bool(dict(solver_config or {}).get("wegscheider_cyclicity_enabled", True)):
            return {}
        try:
            mechanism_source = self.ui.mechanism.mechanism_source_for_run_set(
                self.ui.mechanism.mechanism_source_for_run(fast_mode=bool(fast_mode)),
                set_id=str(set_id),
                apply_parameter_overrides=bool(fast_mode) and self.ui.mechanism.has_local_runtime_parameter_values(),
                strip_initial_concentrations=True,
            )
            mechanism_text = mechanism_source.full_dsl
            solver_identity = repr(
                {
                    "temperature_K": dict(solver_config or {}).get("temperature_K"),
                    "wegscheider_cyclicity_enabled": bool(
                        dict(solver_config or {}).get("wegscheider_cyclicity_enabled", True)
                    ),
                }
            )
            cache_key = (str(mechanism_text or ""), solver_identity)
            cached = self._symbolic_wegscheider_identity_cache.get(cache_key)
            if cached is not None:
                return dict(cached)
            from kindred.core.simulation_preparation import (
                symbolic_wegscheider_identity_for_execution_text,
            )

            payload = symbolic_wegscheider_identity_for_execution_text(
                mechanism_text=str(mechanism_text or ""),
                solver_config=dict(solver_config or {}),
            )
            if not payload:
                return {}
            self._symbolic_wegscheider_identity_cache[cache_key] = dict(payload)
            return dict(payload)
        except Exception:
            return {}

    def _simulation_identity_for_set(
        self,
        *,
        set_id: str,
        solver_config: Mapping[str, Any],
        t_end: float,
        canonical_initials_fingerprint: str = "",
        preview_batch_cache_token: str = "",
        intervention_schedule_declarative_fingerprint: str = "",
        intervention_schedule_executable_fingerprint: str = "",
        fast_mode: bool,
    ) -> SimulationIdentity:
        param_fingerprint = ""
        preview_token = ""
        if bool(fast_mode):
            param_fingerprint = self.ui.mechanism.runtime_parameter_fingerprint_for_set(
                set_id=str(set_id),
                fast_mode=True,
            )
            preview_token = str(preview_batch_cache_token or "")
        return SimulationIdentity.build(
            schema_id=self.ui.mechanism.simulation_schema_id(fast_mode=bool(fast_mode)),
            param_fingerprint=param_fingerprint,
            canonical_initials_fingerprint=str(canonical_initials_fingerprint or ""),
            solver_config=solver_config,
            t_end=float(t_end),
            intervention_schedule_declarative_fingerprint=str(
                intervention_schedule_declarative_fingerprint or ""
            ),
            intervention_schedule_executable_fingerprint=str(
                intervention_schedule_executable_fingerprint or ""
            ),
            preview_batch_cache_token=preview_token,
            execution_flags=self._execution_identity_flags(fast_mode=bool(fast_mode)),
            symbolic_jacobian_identity=self._runtime_symbolic_jacobian_identity(
                set_id=str(set_id),
                solver_config=solver_config,
                fast_mode=bool(fast_mode),
            ),
            symbolic_wegscheider_identity=self._symbolic_wegscheider_identity_for_set(
                set_id=str(set_id),
                solver_config=solver_config,
                fast_mode=bool(fast_mode),
            ),
        )

    def _resolved_initials_for_batch_row(
        self,
        *,
        row: int,
        include_preview_initials: bool,
        pending_initials: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, float]:
        return self._batch_dispatch_materialization_owner.materialize_initials(
            row=int(row),
            fast_mode=bool(include_preview_initials),
            pending_initials=dict(pending_initials or {}),
        )

    def _requeue_preserved_pending_slider_replay_after_preflight_abort(self) -> None:
        self._apply_runtime_lifecycle_ui_effects(
            self._runtime_orchestrator.preview_replay_after_preflight_abort(
                current_state=self._runtime_preview_replay_state(),
                explicit_run=True,
            )
        )

    def _runtime_parameter_names_for_set(self, *, set_id: Optional[str]) -> list[str]:
        try:
            return [
                str(name)
                for name in self.ui.mechanism.runtime_parameter_names_for_set(
                    set_id=set_id,
                    fast_mode=True,
                )
                if str(name)
            ]
        except Exception:
            return []

    def _runtime_parameter_values_for_set(self, *, set_id: Optional[str]) -> Dict[str, float]:
        try:
            values = dict(
                self.ui.mechanism.runtime_parameter_values_for_set(
                    set_id=set_id,
                    fast_mode=True,
                )
            )
        except Exception:
            return {}
        return {name: value for name, value in values.items() if str(name)}

    def _preview_contained_owner_identity(
        self,
        *,
        owner_mechanism_text: str,
        solver_config: Mapping[str, Any],
        t_end: float,
        set_id: str,
        parameter_names: Optional[Sequence[str]] = None,
        simulation_identity: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        names = (
            list(parameter_names)
            if parameter_names is not None
            else self._runtime_parameter_names_for_set(set_id=str(set_id or ""))
        )
        return contained_simulation_owner_identity(
            execution_mode="preview",
            owner_mechanism_text=str(owner_mechanism_text or ""),
            solver_config=solver_config,
            t_end=float(t_end),
            set_id=str(set_id or ""),
            parameter_names=names,
            simulation_identity=simulation_identity,
            contained_child_blas_threads_limited=self._contained_child_blas_threads_limited(),
        )

    def _ordinary_contained_owner_identity(
        self,
        *,
        owner_mechanism_text: str,
        solver_config: Mapping[str, Any],
        t_end: float,
        set_id: str,
        simulation_identity: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        return contained_simulation_owner_identity(
            execution_mode="explicit",
            owner_mechanism_text=str(owner_mechanism_text or ""),
            solver_config=solver_config,
            t_end=float(t_end),
            set_id=str(set_id or ""),
            simulation_identity=simulation_identity,
            contained_child_blas_threads_limited=self._contained_child_blas_threads_limited(),
        )

    def _run_selected_rows_or_abort(self) -> list[int] | None:
        selected_rows = list(self.ui.batch.batch_rows_for_scope("selected"))
        selected_target_set_ids = self._run_target_set_ids_for_rows(selected_rows)
        auto_lock_result = self.ui.mechanism.auto_lock_for_run()
        if not auto_lock_result:
            self.ui.run_ui.set_status_text("Cannot run: mechanism has errors. Fix and try again.")
            return None
        if not self.ui.mechanism.is_mechanism_ready_for_run():
            self.ui.run_ui.set_status_text("Cannot run: mechanism has errors. Fix and try again.")
            return None
        self._discarded_slider_preview_generation_id = None
        if bool(getattr(self, "_simulation_running", False)):
            logger.info("Superseding active simulation with new Run Selected request")
            self._cancel_active_run_for_restart()

        rows_to_run = self._run_rows_for_target_set_ids(
            selected_target_set_ids,
            fallback_rows=selected_rows,
        )
        if not rows_to_run:
            reason = self.ui.batch.run_selected_empty_target_reason()
            self.ui.dialogs.message_box_warning("No Sets", reason)
            return None

        try:
            self.ui.solver.parse_sim_time_seconds()
        except ValueError as exc:
            self.ui.dialogs.message_box_warning("Invalid t_end", f"Fix t_end before running:\n\n{exc}")
            return None
        if not self._resolve_wegscheider_cyclicity_for_run_or_abort():
            return None
        return [int(row) for row in rows_to_run]

    def _run_simulation(self):
        self.run_simulation()

    def _resolve_wegscheider_cyclicity_for_run_or_abort(self) -> bool:
        if not bool(self.ui.solver.wegscheider_cyclicity_enabled()):
            return True

        from kindred.gui.wegscheider_resolution import resolve_wegscheider_cyclicity_for_gui

        prompt_shown = {"value": False}

        def _choose_resolution(title: str, message: str, choices: Mapping[str, Any]) -> Mapping[str, str] | None:
            prompt_shown["value"] = True
            return self.ui.dialogs.choose_wegscheider_resolution(
                str(title),
                str(message),
                choices,
            )

        try:
            resolution = resolve_wegscheider_cyclicity_for_gui(
                self.ui.mechanism.mechanism_reactions_text_raw(),
                enabled=True,
                choose_resolution=_choose_resolution,
            )
        except Exception as exc:
            self.ui.dialogs.message_box_warning(
                "Wegscheider Cyclicity",
                f"Cannot resolve Wegscheider cyclicity automatically:\n\n{exc}",
            )
            self.ui.run_ui.set_status_text("Cannot run: unresolved Wegscheider cyclicity.")
            return False

        if resolution is None:
            if bool(prompt_shown["value"]):
                self.ui.run_ui.set_status_text("Run cancelled: unresolved Wegscheider cyclicity.")
                return False
            return True

        self.ui.mechanism.apply_wegscheider_resolution_reactions_rewrite(
            resolution.rewritten_reactions_text
        )
        self.ui.run_ui.set_status_text("Applied Wegscheider cyclicity resolution.")
        return True

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------
    def _cancel_active_run_for_restart(self) -> None:
        state = self._batch_context_owner.active_batch_state()
        if state is not None and state.active:
            self._batch_context_owner.deactivate()
        self._apply_runtime_lifecycle_ui_effects(
            self._runtime_orchestrator.cancel_requested(kind="shutdown")
        )
        self._simulation_running = False
        self._slider_simulation_active = False
        self.ui.run_ui.set_run_button_enabled(True)
        self.ui.run_ui.set_stop_button_enabled(False)

    def _clear_slider_triggered_preflight_state(self, *, fast_mode: bool) -> None:
        if bool(fast_mode):
            self.ui.slider.set_slider_triggered_simulation(False)

    def _set_simulation_running(self, value: bool) -> None:
        self._simulation_running = bool(value)

    def _set_slider_simulation_active(self, value: bool) -> None:
        self._slider_simulation_active = bool(value)

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------
    def _clear_pending_progress_status(self) -> None:
        self._pending_progress_payload = None
        if self._progress_flush_timer.isActive():
            self._progress_flush_timer.stop()

    def _flush_progress_ui(self) -> None:
        payload = self._pending_progress_payload
        if payload is None:
            if self._progress_flush_timer.isActive():
                self._progress_flush_timer.stop()
            return

        self._pending_progress_payload = None
        percent, message = payload

        self.ui.run_ui.set_sim_progress_value(int(percent))
        self.ui.run_ui.set_status_text(str(message))
        self.ui.run_ui.repaint_simulation_widgets()

        if self._pending_progress_payload is None and self._progress_flush_timer.isActive():
            self._progress_flush_timer.stop()

    def _on_simulation_progress(self, percent: int, message: str):
        try:
            sender = self.sender()
            if sender is not None:
                sender_run_id = getattr(sender, "_run_id", None)
                if sender_run_id is not None and int(sender_run_id) != int(getattr(self, "_active_run_id", 0)):
                    return
        except Exception as exc:
            self._record_nonfatal_exception(
                "Failed to validate sender run_id in simulation progress callback",
                exc,
            )
        if self._processing_progress:
            return

        self._processing_progress = True
        try:
            completion_state = self._batch_context_owner.completion_state()
            if completion_state is not None and completion_state.active:
                queue = list(completion_state.queue_names)
                pos = int(completion_state.pos)
                total = max(1, int(completion_state.total or len(queue) or 1))
                if total > 1:
                    frac = max(0.0, min(1.0, float(percent) / 100.0))
                    overall = int(((pos + frac) / float(total)) * 100.0)
                    percent = max(0, min(100, overall))
                    if 0 <= pos < len(queue):
                        message = f"{queue[pos]} ({pos + 1}/{total}) • {message}"

            self._pending_progress_payload = (int(percent), str(message))
            if not self._progress_flush_timer.isActive():
                self._progress_flush_timer.start()
        finally:
            self._processing_progress = False

    def _on_simulation_complete(
        self,
        result: dict,
        *,
        callback_identity: SimulationCallbackIdentity,
    ):
        self._completion_callback_owner.handle_completion(
            result,
            debug_batch_parallel=bool(getattr(self, "_debug_batch_parallel", False)),
            callback_identity=callback_identity,
        )

    def _on_simulation_error(
        self,
        error_msg: object,
        *,
        callback_identity: SimulationCallbackIdentity,
    ):
        self._error_handling_owner.handle_error(
            error_msg,
            callback_identity=callback_identity,
        )

    def _handle_current_preview_simulation_failure(
        self,
        error_payload: Mapping[str, Any],
        *,
        error_text: str,
        error_detail_text: str,
        context: Optional[Mapping[str, Any]],
    ) -> None:
        kind = str(error_payload.get("kind") or "").strip().lower()
        details = error_payload.get("details")
        stage = (
            str(details.get("stage") or "").strip().lower()
            if isinstance(details, Mapping)
            else ""
        )
        if error_detail_text:
            logger.warning("%s", error_detail_text)
        if kind == "timeout":
            status_text = "Preview timed out. Adjust sliders or run again."
        elif stage == "wegscheider_cyclicity":
            status_text = str(error_payload.get("message") or "Unresolved Wegscheider cyclicity.")
        else:
            status_text = "Preview unavailable. Adjust sliders or run again."
        logger.warning("Preview simulation failed without modal: %s", error_text)

        if isinstance(context, Mapping):
            self._batch_context_owner.deactivate_if_active(context)
        self._apply_simulation_lifecycle_effects(
            self._lifecycle_effect_owner.current_preview_failure_effects(
                status_text=str(status_text),
            )
        )

    def _stop_simulation(self):
        if not self._simulation_running:
            return

        logger.info("Stop simulation requested")

        state = self._batch_context_owner.active_batch_state()
        if state is not None and state.active:
            self._batch_context_owner.deactivate()
        self._apply_runtime_lifecycle_ui_effects(
            self._runtime_orchestrator.cancel_requested(kind="shutdown")
        )

        self._simulation_running = False
        self.ui.run_ui.set_run_button_enabled(True)
        self.ui.run_ui.set_stop_button_enabled(False)
        self.ui.run_ui.set_status_text("Ready")
        self.ui.run_ui.set_sim_progress_value(0)
