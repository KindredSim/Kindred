from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import os
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from PySide6 import QtCore
import shiboken6

from kindred.core.batch_parallel import compute_effective_batch_workers
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
from kindred.gui.controllers.batch_run_context_owner import BatchContextSeed, BatchRunContextOwner
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
    RuntimeDispatchPlan,
    RuntimeLaneAllocator,
    RuntimeLaunchIntent,
    RuntimeTaskDescriptor,
)
from kindred.gui.controllers.simulation_runtime_readiness_lifecycle import (
    SimulationRuntimeReadinessLifecycle,
)
from kindred.gui.controllers.simulation_runtime_dispatch import (
    SimulationRuntimeDispatchDependencies,
    SimulationRuntimeDispatchOwner,
)
from kindred.gui.controllers.simulation_result_materialization import SimulationResultMaterializationOwner
from kindred.gui.controllers.simulation_completion_policy import (
    CompletionPolicyContext,
    DirtySetState,
    PendingReplayDirective,
    PendingReplayState,
    PolicyStatePatch,
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
    CompletedRunDisplayIntent,
    DisplayRefreshSource,
    DisplayTransitionCause,
    DisplayTransitionOutcome,
    DisplayTransitionOutcomeKind,
    FreshPreviewDisplayEntry,
    FreshPreviewDisplayTransaction,
    SimulationCacheOpResult,
    SimulationCompletionDisplayOutcome,
    SimulationUiPorts,
    SliderReplayIntent,
)

logger = logging.getLogger(__name__)

__all__ = ["SimulationController"]

_WORKER_APPLICATION_SIGNAL_HANDLERS_ATTR = "_kindred_controller_worker_signal_handlers"


@dataclass(frozen=True)
class SimulationRuntimeInputsChangeOutcome:
    interactive_runtime_refresh_requested: bool
    batch_pool_shut_down: bool
    batch_pool_marked_stale_draining: bool

@dataclass
class _SerialBatchDispatchState:
    plan_payload: Dict[str, Any] | None
    cache_key: str
    context: Mapping[str, Any] | None


@dataclass(frozen=True)
class _TerminalFailureReplaySnapshot:
    active: bool
    request_id: Optional[int]
    target_set_ids: tuple[str, ...]
    replay_generation: int = 0
    dirty_generation_by_set_id: tuple[tuple[str, int], ...] = ()


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
            ),
        )
        self._batch_context_owner = BatchRunContextOwner()
        self._batch_lane_executor = ParallelBatchExecutor(
            max_parallel_workers=int(PROJECT_DEFAULTS["max_parallel_batch_workers"]),
            limit_blas_threads_per_worker=bool(PROJECT_DEFAULTS["limit_blas_threads_per_worker"]),
            record_nonfatal_exception=self._record_nonfatal_exception,
        )
        self._runtime_lane_allocator = RuntimeLaneAllocator(
            lane_warmer=self._warm_runtime_lane_pool,
            backend_lane_is_live=self._runtime_lane_is_live,
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
        self._retained_simulation_workers: List[object] = []
        self._shutdown_requested_for_close: bool = False
        self._discarded_slider_preview_generation_id: Optional[int] = None
        self._batch_runtime_lane_budget = int(PROJECT_DEFAULTS["batch_runtime_lane_budget"])
        self._runtime_readiness_lifecycle = SimulationRuntimeReadinessLifecycle(
            allocator=self._runtime_lane_allocator,
            render=self.runtime_readiness_render_requested.emit,
            current_runtime_input_epochs=self._runtime_input_epochs_for_sets,
            prepared_request_is_current=self._runtime_prepared_request_is_current,
            dispatch_ready=self._dispatch_runtime_plan,
            parent=self,
        )
        self._active_runtime_dispatch_plan: RuntimeDispatchPlan | None = None
        self._runtime_dispatch_owner = SimulationRuntimeDispatchOwner(
            ui=self.ui,
            batch_executor=self._batch_parallel,
            parent=self,
            dependencies=SimulationRuntimeDispatchDependencies(
                next_run_id=self._next_runtime_run_id,
                load_context=self._load_runtime_dispatch_context,
                callback_identity_for_descriptor=self._callback_identity_for_runtime_descriptor,
                set_simulation_running=self._set_simulation_running,
                set_slider_simulation_active=self._set_slider_simulation_active,
                release_dispatch_plan=self._runtime_readiness_lifecycle.release_dispatch_plan,
                render_failure=self._runtime_readiness_lifecycle.render_failure,
                set_active_dispatch_plan=self._set_active_runtime_dispatch_plan,
                record_nonfatal_exception=self._record_nonfatal_exception,
                start_completion_poll_timer=self._start_batch_completion_poll_timer,
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
                has_deferred_preview_replay_intent=self._has_deferred_preview_replay_intent,
                start_next_batch_simulation=self._start_next_batch_simulation,
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
                completion_policy_pending_replay_state=self._completion_policy_pending_replay_state,
                apply_completion_policy_state_patch=self._apply_completion_policy_state_patch,
                apply_lifecycle_effects=self._apply_simulation_lifecycle_effects,
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
                completion_policy_pending_replay_state=self._completion_policy_pending_replay_state,
                apply_completion_policy_state_patch=self._apply_completion_policy_state_patch,
                apply_lifecycle_effects=self._apply_simulation_lifecycle_effects,
                handle_current_preview_simulation_failure=self._handle_current_preview_simulation_failure,
                has_deferred_preview_replay_intent=self._has_deferred_preview_replay_intent,
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
                cleanup_parallel_batch_lane_pool_after_run=self._cleanup_parallel_batch_lane_pool_after_run,
                show_scoped_batch_failure_summary=self._show_scoped_batch_failure_summary,
                apply_explicit_failure_pending_replay_policy=self._apply_explicit_failure_pending_replay_policy,
                reset_parallel_batch_run_and_shutdown_lane_pool=self._reset_parallel_batch_run_and_shutdown_lane_pool,
                set_simulation_running=self._set_simulation_running,
                set_slider_simulation_active=self._set_slider_simulation_active,
            ),
        )

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

    @property
    def _simulation_worker(self):
        return self._run_state.simulation_worker

    @_simulation_worker.setter
    def _simulation_worker(self, value) -> None:
        self._run_state.simulation_worker = value

    def _set_discarded_slider_preview_generation(self, value: int | None) -> None:
        self._discarded_slider_preview_generation_id = int(value) if value is not None else None

    def _set_simulation_worker(self, worker) -> None:
        self._simulation_worker = worker

    def _set_active_runtime_dispatch_plan(self, dispatch_plan: RuntimeDispatchPlan | None) -> None:
        self._active_runtime_dispatch_plan = dispatch_plan

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

    def _has_deferred_preview_replay_intent(
        self,
        replay: Optional[PendingSliderPreviewLaunchState] = None,
    ) -> bool:
        state = replay if isinstance(replay, PendingSliderPreviewLaunchState) else self._pending_slider_preview_launch
        return bool(state.active or state.target_set_ids)

    def _has_deferred_preview_replay_launch_state(
        self,
        replay: Optional[PendingSliderPreviewLaunchState] = None,
    ) -> bool:
        state = replay if isinstance(replay, PendingSliderPreviewLaunchState) else self._pending_slider_preview_launch
        return bool(state.active or state.request_id is not None or state.target_set_ids)

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

    def _schedule_deferred_preview_replay_handoff_once(
        self,
        *,
        stop_timers: bool = True,
    ) -> bool:
        replay = self._pending_slider_preview_launch
        if not self._has_deferred_preview_replay_intent(replay):
            return False
        if replay.handoff_queued:
            return False
        request_id = replay.request_id
        if request_id is None:
            request_id = self._next_slider_preview_request_id()
        self._run_state.pending_slider_preview_launch = replace(
            replay,
            active=True,
            request_id=int(request_id),
            handoff_queued=True,
        )
        if stop_timers:
            self._stop_deferred_preview_replay_timers()
        QtCore.QTimer.singleShot(0, self.launch_pending_slider_preview_replay)
        return True

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
        explicit_failure_replay_snapshot = (
            self._capture_terminal_failure_replay_snapshot()
            if bool(effects.apply_explicit_failure_pending_replay)
            else None
        )
        if bool(effects.release_worker):
            self._release_current_simulation_worker()
        if bool(effects.shutdown_lane_pool):
            self._shutdown_batch_lane_pool(force_terminate=bool(effects.lane_pool_force_terminate))
        if bool(effects.cleanup_lane_pool):
            self._cleanup_parallel_batch_lane_pool_after_run(
                keep_lane_pool_alive=bool(effects.keep_lane_pool_alive),
                clear_pending_plot_updates=bool(effects.clear_pending_plot_updates),
                stale_fast_handoff_after_display=bool(effects.stale_fast_handoff_after_display),
            )
        if bool(effects.close_contained_owner):
            self._close_contained_simulation_owner(
                fast_mode=bool(effects.close_contained_fast_mode),
                kill=bool(effects.close_contained_kill),
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
        if bool(effects.schedule_deferred_preview_replay):
            self._schedule_deferred_preview_replay_handoff_once(
                stop_timers=bool(effects.deferred_replay_stop_timers),
            )
        if effects.modal_error is not None:
            self.ui.dialogs.message_box_critical(
                effects.modal_error.title,
                effects.modal_error.message,
                details=effects.modal_error.details,
            )
        if bool(effects.apply_explicit_failure_pending_replay):
            self._apply_post_modal_explicit_failure_pending_replay_policy(
                fast_mode=bool(effects.close_contained_fast_mode),
                replay_snapshot=explicit_failure_replay_snapshot,
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

    def _queued_preview_update_still_matches_current_preview_owner(
        self,
        *,
        request_id: Optional[int],
        accepted_preview_request_id: Optional[int],
        accepted_preview_owner_epoch: Optional[int],
    ) -> bool:
        if request_id is None:
            return True
        current = self._preview_ownership
        if accepted_preview_request_id is None or accepted_preview_owner_epoch is None:
            return False
        return (
            current.request_id is not None
            and int(current.request_id) == int(request_id)
            and int(accepted_preview_request_id) == int(request_id)
            and int(current.epoch) == int(accepted_preview_owner_epoch)
        )

    def queue_pending_slider_preview_replay(
        self,
        *,
        target_set_ids: Sequence[str],
        request_id: Optional[int] = None,
        preserve_existing_request: bool = False,
    ) -> None:
        current = self._pending_slider_preview_launch
        normalized_targets = PendingSliderPreviewLaunchState(target_set_ids=target_set_ids).target_set_ids
        next_request_id: Optional[int]
        if request_id is not None:
            next_request_id = int(request_id)
        elif bool(preserve_existing_request):
            preserved_request_id = current.request_id
            if preserved_request_id is None:
                preserved_request_id = self._next_slider_preview_request_id()
            next_request_id = int(preserved_request_id)
        else:
            next_request_id = None
        preserve_handoff_queued = bool(
            current.handoff_queued
            and current.target_set_ids == normalized_targets
            and current.request_id == next_request_id
        )
        self._run_state.pending_slider_preview_launch = PendingSliderPreviewLaunchState(
            active=True,
            request_id=next_request_id,
            target_set_ids=normalized_targets,
            handoff_queued=preserve_handoff_queued,
            replay_generation=self._next_pending_slider_preview_replay_generation(),
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
        self._run_state.pending_slider_preview_launch = PendingSliderPreviewLaunchState()

    def clear_pending_slider_preview_replay(self, *, clear_plot_updates: bool = True) -> None:
        self._run_state.pending_slider_preview_launch = PendingSliderPreviewLaunchState()
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
        worker = getattr(self, "_simulation_worker", None)
        worker_running = bool(worker is not None and self._worker_is_running(worker))
        return RunActivitySnapshot(
            latest_request_id=int(getattr(self, "_latest_sim_request_id", 0)),
            simulation_running=bool(getattr(self, "_simulation_running", False)),
            slider_simulation_active=bool(getattr(self, "_slider_simulation_active", False)),
            worker_running=worker_running,
            worker_fast_mode=(
                bool(getattr(worker, "_fast_mode", False))
                if worker is not None
                else None
            ),
            worker_request_id=(getattr(worker, "_request_id", None) if worker is not None else None),
            discarded_slider_preview_generation_id=getattr(
                self,
                "_discarded_slider_preview_generation_id",
                None,
            ),
        )

    def _completion_policy_pending_replay_state(self) -> PendingReplayState:
        return self._pending_slider_preview_launch

    def _completion_policy_preview_ownership(self) -> PreviewOwnershipState:
        return self._preview_ownership

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
        if patch.pending_replay is not None:
            directive: PendingReplayDirective = patch.pending_replay
            if directive.action == "clear":
                self.clear_pending_slider_preview_replay(clear_plot_updates=bool(directive.clear_plot_updates))
            elif directive.action == "preserve":
                self.queue_pending_slider_preview_replay(
                    target_set_ids=directive.target_set_ids,
                    request_id=None,
                    preserve_existing_request=True,
                )
                if directive.clear_plot_updates:
                    self._clear_pending_preview_slider_plot_updates()
            elif directive.action == "queue_fresh":
                self.queue_pending_slider_preview_replay(
                    target_set_ids=directive.target_set_ids,
                    request_id=self._next_slider_preview_request_id(),
                )
            elif directive.action == "arm_existing":
                target_set_ids = directive.target_set_ids or self._pending_slider_target_set_ids
                self.queue_pending_slider_preview_replay(
                    target_set_ids=target_set_ids,
                    request_id=None,
                    preserve_existing_request=bool(directive.preserve_existing_request),
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

    def _capture_terminal_failure_replay_snapshot(self) -> _TerminalFailureReplaySnapshot:
        pending = self._pending_slider_preview_launch
        target_set_ids = tuple(str(set_id) for set_id in (pending.target_set_ids or ()) if str(set_id))
        if not self._has_deferred_preview_replay_intent(pending) or not target_set_ids:
            return _TerminalFailureReplaySnapshot(
                active=False,
                request_id=pending.request_id,
                target_set_ids=target_set_ids,
                replay_generation=pending.replay_generation,
            )
        dirty_state = self._capture_dirty_state_by_set_id(target_set_ids)
        generations: list[tuple[str, int]] = []
        for set_id in target_set_ids:
            state = dirty_state.get(str(set_id))
            if state is None or not bool(state.is_dirty) or state.generation is None:
                return _TerminalFailureReplaySnapshot(
                    active=False,
                    request_id=pending.request_id,
                    target_set_ids=target_set_ids,
                    replay_generation=pending.replay_generation,
                )
            generations.append((str(set_id), int(state.generation)))
        return _TerminalFailureReplaySnapshot(
            active=True,
            request_id=pending.request_id,
            target_set_ids=target_set_ids,
            replay_generation=pending.replay_generation,
            dirty_generation_by_set_id=tuple(generations),
        )

    def _terminal_failure_replay_snapshot_still_current(
        self,
        snapshot: _TerminalFailureReplaySnapshot,
    ) -> bool:
        if not bool(snapshot.active) or not snapshot.target_set_ids:
            return False
        current = self._pending_slider_preview_launch
        if tuple(current.target_set_ids or ()) != tuple(snapshot.target_set_ids):
            return False
        if current.request_id != snapshot.request_id:
            return False
        expected_generations = {
            str(set_id): int(generation)
            for set_id, generation in (snapshot.dirty_generation_by_set_id or ())
            if str(set_id)
        }
        if set(expected_generations) != set(snapshot.target_set_ids):
            return False
        current_dirty_state = self._capture_dirty_state_by_set_id(snapshot.target_set_ids)
        for set_id, expected_generation in expected_generations.items():
            state = current_dirty_state.get(str(set_id))
            if state is None or not bool(state.is_dirty) or state.generation is None:
                return False
            if int(state.generation) != int(expected_generation):
                return False
        return True

    def _terminal_failure_replay_snapshot_matches_current(
        self,
        snapshot: _TerminalFailureReplaySnapshot,
    ) -> bool:
        if not bool(snapshot.active) or not snapshot.target_set_ids:
            return False
        current = self._pending_slider_preview_launch
        return (
            tuple(str(set_id) for set_id in (current.target_set_ids or ()) if str(set_id))
            == tuple(snapshot.target_set_ids)
            and current.request_id == snapshot.request_id
            and int(current.replay_generation) == int(snapshot.replay_generation)
        )

    def _apply_post_modal_explicit_failure_pending_replay_policy(
        self,
        *,
        fast_mode: bool,
        replay_snapshot: Optional[_TerminalFailureReplaySnapshot],
    ) -> None:
        if (
            replay_snapshot is not None
            and self._terminal_failure_replay_snapshot_matches_current(replay_snapshot)
            and not self._terminal_failure_replay_snapshot_still_current(replay_snapshot)
        ):
            self.clear_pending_slider_preview_replay(clear_plot_updates=False)
            return
        self._apply_explicit_failure_pending_replay_policy(fast_mode=bool(fast_mode))

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
            preferred_lane_capacity=self._lane_capacity_for_rows(rows_to_run),
        )
        self._accept_and_dispatch_prepared_runtime_request(prepared)

    def launch_pending_slider_preview_replay(self) -> None:
        pending = self._pending_slider_preview_launch
        if bool(pending.active) and pending.request_id is None:
            pending = replace(
                pending,
                request_id=int(self._next_slider_preview_request_id()),
                handoff_queued=False,
            )
            self._pending_slider_preview_launch = pending
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
        dispatch_plan = self._runtime_readiness_lifecycle.accept_preview_replay_intent(
            intent,
            prepare=lambda replay_intent: self._run_preparation_owner.prepare_runtime_request_set(
                intent=replay_intent,
                fast_mode=True,
                preferred_lane_capacity=self._lane_capacity_for_rows(rows),
            ),
        )
        if dispatch_plan is not None:
            self._dispatch_runtime_plan(dispatch_plan)

    def retry_runtime_readiness(self) -> RuntimeDispatchPlan | None:
        dispatch_plan = self._runtime_readiness_lifecycle.retry_runtime_readiness()
        if dispatch_plan is not None:
            self._dispatch_runtime_plan(dispatch_plan)
        return dispatch_plan

    def invalidate_interactive_simulation_runtimes(self, *, kill: bool = False) -> None:
        self._runtime_readiness_lifecycle.release_all(failed=bool(kill))

    def _runtime_requested_show_set_ids(self, fallback_set_ids: Sequence[str]) -> tuple[str, ...]:
        requested: Sequence[str] = ()
        try:
            requested = self.ui.batch.requested_show_batch_set_ids()
        except Exception:
            requested = ()
        normalized = tuple(str(set_id) for set_id in (requested or ()) if str(set_id))
        if normalized:
            return normalized
        return tuple(str(set_id) for set_id in (fallback_set_ids or ()) if str(set_id))

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

    def _lane_capacity_for_rows(self, rows: Sequence[int]) -> int:
        return max(1, min(int(self.batch_runtime_lane_budget), int(self._effective_batch_worker_count(len(rows)))))

    def _accept_and_dispatch_prepared_runtime_request(
        self,
        prepared: PreparedRuntimeRequestSet,
    ) -> RuntimeDispatchPlan | None:
        dispatch_plan = self._runtime_readiness_lifecycle.accept_prepared_request(prepared)
        if dispatch_plan is None:
            return None
        self._dispatch_runtime_plan(dispatch_plan)
        return dispatch_plan

    def _dispatch_runtime_plan(self, dispatch_plan: RuntimeDispatchPlan) -> bool:
        return self._runtime_dispatch_owner.dispatch(dispatch_plan)

    def _load_runtime_dispatch_context(
        self,
        *,
        dispatch_plan: RuntimeDispatchPlan,
        run_id: int,
        active: bool,
    ) -> Mapping[str, Any]:
        descriptors = tuple(dispatch_plan.ordered_task_descriptors or ())
        intent = dispatch_plan.launch_allocation.launch_intent
        queue_ids = tuple(descriptor.set_id for descriptor in descriptors)
        queue_names = tuple(
            str(
                descriptor.set_label
                or self.ui.batch.batch_set_name_for_id(descriptor.set_id)
                or descriptor.set_id
            )
            for descriptor in descriptors
        )
        requested_show_ids = tuple(
            str(set_id)
            for set_id in (
                intent.requested_show_set_ids
                if intent is not None and intent.requested_show_set_ids
                else queue_ids
            )
            if str(set_id)
        )
        requested_labels = {
            str(set_id): str(label)
            for set_id, label in dict(
                getattr(intent, "requested_show_labels_by_set_id", {}) or {}
            ).items()
            if str(set_id)
        }
        queue_labels = {
            str(set_id): str(queue_names[index])
            for index, set_id in enumerate(queue_ids)
            if str(set_id)
        }
        for set_id in requested_show_ids:
            requested_labels.setdefault(
                str(set_id),
                str(queue_labels.get(str(set_id)) or self.ui.batch.batch_set_name_for_id(str(set_id)) or set_id),
            )
        owned_species_by_set_id = {
            str(descriptor.set_id): tuple(str(name) for name in descriptor.owned_species if str(name))
            for descriptor in descriptors
            if str(descriptor.set_id)
        }
        plan_by_set_id = {
            str(descriptor.set_id): dict(descriptor.plan_payload or {})
            for descriptor in descriptors
            if str(descriptor.set_id) and isinstance(descriptor.plan_payload, Mapping)
        }
        mechanism_text_by_set_id = {
            str(descriptor.set_id): str(descriptor.mechanism_text)
            for descriptor in descriptors
            if str(descriptor.set_id) and str(descriptor.mechanism_text)
        }
        mechanism_signature_by_set_id = {
            str(descriptor.set_id): str(descriptor.mechanism_signature)
            for descriptor in descriptors
            if str(descriptor.set_id) and str(descriptor.mechanism_signature)
        }
        simulation_identity_by_set_id = {
            str(descriptor.set_id): dict(descriptor.simulation_identity or {})
            for descriptor in descriptors
            if str(descriptor.set_id) and isinstance(descriptor.simulation_identity, Mapping)
        }
        preview_token_by_set_id = {
            str(descriptor.set_id): str(descriptor.preview_batch_cache_token)
            for descriptor in descriptors
            if str(descriptor.set_id) and str(descriptor.preview_batch_cache_token)
        }
        runtime_task_identity_by_set_id = {}
        for descriptor in descriptors:
            set_id = str(descriptor.set_id)
            if not set_id:
                continue
            assignment = dispatch_plan.assignment_for_task(descriptor.task_id)
            runtime_task_identity_by_set_id[set_id] = {
                "allocation_id": str(dispatch_plan.launch_allocation.allocation_id),
                "lane_id": str(assignment.lane_id if assignment is not None else ""),
                "lane_generation": int(assignment.lane_generation if assignment is not None else 0),
                "row": int(descriptor.row),
                "task_id": str(descriptor.task_id),
                "exact_descriptor_hash": str(descriptor.exact_descriptor_hash),
                "compatibility_key": descriptor.compatibility_key.to_payload(),
                "cache_key": str(descriptor.cache_key),
            }
        display_primary_set_id = ""
        queue_members = set(str(set_id) for set_id in queue_ids if str(set_id))
        for set_id in requested_show_ids:
            if str(set_id) in queue_members:
                display_primary_set_id = str(set_id)
                break
        if not display_primary_set_id:
            display_primary_set_id = str(requested_show_ids[0]) if requested_show_ids else ""
        display_intent = CompletedRunDisplayIntent(
            requested_show_set_ids=requested_show_ids,
            labels_by_set_id=requested_labels,
            primary_set_id=display_primary_set_id,
            cache_key=str(descriptors[0].cache_key if descriptors else ""),
            run_id=int(run_id),
            request_id=int(intent.request_token or 0),
            owned_species_by_set_id=owned_species_by_set_id,
            run_target_set_ids=queue_ids,
        )
        context = BatchContextSeed(
            active=bool(active),
            request_id=int(intent.request_token or 0),
            run_id=int(run_id),
            runtime_input_epoch=int(getattr(self, "_authoritative_runtime_input_epoch", 0) or 0),
            runtime_input_global_epoch=int(getattr(self, "_authoritative_runtime_input_global_epoch", 0) or 0),
            runtime_input_set_epoch_by_set_id=self._runtime_input_context_set_epochs(queue_ids),
            fast_mode=intent.intent_kind == "preview",
            completion_mode="runtime_task_queue",
            keep_lane_pool_alive=bool(dispatch_plan.launch_allocation.retain_lanes_after_success),
            effective_workers=max(1, int(dispatch_plan.launch_allocation.accepted_capacity or 1)),
            cache_key=str(descriptors[0].cache_key if descriptors else ""),
            simulation_plan_by_set_id=plan_by_set_id,
            mechanism_text_by_set_id=mechanism_text_by_set_id,
            mechanism_signature_by_set_id=mechanism_signature_by_set_id,
            simulation_identity_by_set_id=simulation_identity_by_set_id,
            rows=tuple(descriptor.row for descriptor in descriptors),
            queue_ids=queue_ids,
            queue_names=queue_names,
            primary_set_id=str(queue_ids[0]) if queue_ids else None,
            preview_owner_epoch=int(intent.preview_epoch) if intent.preview_epoch is not None else None,
            preview_batch_cache_token_by_set_id=preview_token_by_set_id,
            runtime_task_identity_by_set_id=runtime_task_identity_by_set_id,
            total=len(descriptors),
            completed_run_display_intent=display_intent,
            computed_owned_species_by_set_id=owned_species_by_set_id,
        ).to_context()
        self._batch_context_owner.load_context(BatchContextSeed(**context))
        return context

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
            callback_context=context,
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
        self._reset_parallel_batch_run_and_shutdown_lane_pool()

    def poll_parallel_batch_completions(self) -> None:
        self._poll_parallel_batch_completions()

    def shutdown_batch_lane_pool(self, *, force_terminate: bool) -> None:
        self._shutdown_batch_lane_pool(force_terminate=force_terminate)

    def simulation_runtime_inputs_changed(
        self,
        *,
        batch_runtime_pool_inputs_changed: bool = True,
    ) -> SimulationRuntimeInputsChangeOutcome:
        return self._simulation_runtime_inputs_changed(
            batch_runtime_pool_inputs_changed=bool(batch_runtime_pool_inputs_changed)
        )

    def release_current_simulation_worker(self) -> None:
        self._release_current_simulation_worker()

    def has_running_owned_simulation_workers(self) -> bool:
        return self._has_running_owned_simulation_workers()

    def prepare_simulation_shutdown_for_close(self) -> bool:
        return self._prepare_simulation_shutdown_for_close()

    def cleanup_worker_safely(self, worker, worker_name: str = "worker") -> None:
        self._cleanup_worker_safely(worker, worker_name)

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

    def start_next_batch_simulation(self) -> None:
        self._start_next_batch_simulation()

    def _start_next_batch_simulation(self) -> None:
        return None

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

    def _capture_simulation_callback_identity(
        self,
        *,
        run_id: int,
        fast_mode: bool,
        request_id: int,
        preview_owner_epoch: Optional[int],
        batch_set: Optional[str],
        batch_set_id: Optional[str],
        cache_key: str,
        callback_context: Mapping[str, Any],
        simulation_identity: Mapping[str, Any],
        preview_batch_cache_token: Optional[str] = None,
        launch_provenance: Mapping[str, Any] | None = None,
    ) -> SimulationCallbackIdentity:
        if not isinstance(callback_context, Mapping):
            raise ValueError("callback identity capture requires supplied callback context.")
        if not isinstance(simulation_identity, Mapping):
            raise ValueError("callback identity capture requires supplied simulation identity.")
        resolved_context = callback_context
        set_id = str(batch_set_id or "").strip()
        resolved_batch_set = batch_set
        resolved_simulation_identity = simulation_identity
        resolved_preview_token = preview_batch_cache_token
        return SimulationCallbackIdentity.capture(
            run_id=run_id,
            fast_mode=fast_mode,
            request_id=request_id,
            preview_owner_epoch=preview_owner_epoch,
            batch_set=resolved_batch_set,
            batch_set_id=set_id or batch_set_id,
            cache_key=cache_key,
            callback_context=resolved_context,
            simulation_identity=resolved_simulation_identity,
            preview_batch_cache_token=resolved_preview_token,
            launch_provenance=dict(launch_provenance or {}),
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
    # Worker / lane-pool lifecycle
    # ------------------------------------------------------------------
    @staticmethod
    def _worker_is_valid(worker) -> bool:
        if worker is None:
            return False
        if isinstance(worker, QtCore.QObject):
            try:
                return bool(shiboken6.isValid(worker))
            except Exception:
                return False
        return True

    @staticmethod
    def _worker_is_running(worker) -> bool:
        if worker is None or (not SimulationController._worker_is_valid(worker)) or not hasattr(worker, "isRunning"):
            return False
        try:
            return bool(worker.isRunning())
        except Exception:
            return False

    def _forget_retained_simulation_worker(self, worker) -> None:
        if worker is None:
            return
        self._retained_simulation_workers = [
            item for item in self._retained_simulation_workers if item is not worker
        ]

    def _delete_worker_if_stopped(self, worker, worker_name: str) -> None:
        if worker is None or self._worker_is_running(worker):
            return
        self._forget_retained_simulation_worker(worker)
        if getattr(self, "_simulation_worker", None) is worker:
            self._simulation_worker = None
        if not self._worker_is_valid(worker):
            return
        if hasattr(worker, "deleteLater"):
            try:
                worker.deleteLater()
            except Exception as exc:
                self._record_nonfatal_exception(
                    f"Failed to schedule deleteLater() for {str(worker_name)}",
                    exc,
                )
                return
        try:
            QtCore.QCoreApplication.sendPostedEvents(worker, QtCore.QEvent.DeferredDelete)
        except Exception as exc:
            self._record_nonfatal_exception(
                f"Failed to send deferred delete events for {str(worker_name)}",
                exc,
            )

    def _prune_stopped_owned_simulation_workers(self) -> None:
        seen_ids: set[int] = set()
        owned_workers = []
        current_worker = getattr(self, "_simulation_worker", None)
        if current_worker is not None:
            owned_workers.append(current_worker)
            seen_ids.add(id(current_worker))
        for worker in list(self._retained_simulation_workers):
            if id(worker) in seen_ids:
                continue
            owned_workers.append(worker)
            seen_ids.add(id(worker))
        for worker in owned_workers:
            if not self._worker_is_running(worker):
                self._delete_worker_if_stopped(worker, "simulation worker")

    def _on_retained_simulation_worker_finished(self, worker, worker_name: str = "simulation worker") -> None:
        self._forget_retained_simulation_worker(worker)
        if getattr(self, "_simulation_worker", None) is worker:
            self._simulation_worker = None
        shutdown_requested = bool(getattr(self, "_shutdown_requested_for_close", False))
        self._delete_worker_if_stopped(worker, worker_name)
        if (
            self._has_deferred_preview_replay_intent()
            and (not shutdown_requested)
            and (not self._has_running_owned_simulation_workers())
        ):
            self._schedule_deferred_preview_replay_handoff_once()
        self._clear_shutdown_request_after_close_cleanup()

    def _retain_simulation_worker(self, worker, worker_name: str = "simulation worker") -> None:
        if worker is None:
            return
        if not self._worker_is_valid(worker):
            self._delete_worker_if_stopped(worker, worker_name)
            return
        if any(item is worker for item in self._retained_simulation_workers):
            return
        self._retained_simulation_workers.append(worker)
        finished_signal = None
        try:
            finished_signal = worker.finished
        except Exception:
            finished_signal = None
        if finished_signal is not None and hasattr(finished_signal, "connect"):
            try:
                finished_signal.connect(
                    lambda *_args, _worker=worker, _name=str(worker_name): self._on_retained_simulation_worker_finished(
                        _worker,
                        _name,
                    )
                )
            except Exception as exc:
                self._record_nonfatal_exception(
                    f"Failed to connect retained-worker release hook for {str(worker_name)}",
                    exc,
                )
        if not self._worker_is_running(worker):
            self._on_retained_simulation_worker_finished(worker, worker_name)

    def _has_running_owned_simulation_workers(self) -> bool:
        seen_ids: set[int] = set()
        owned_workers = []
        current_worker = getattr(self, "_simulation_worker", None)
        if current_worker is not None:
            owned_workers.append(current_worker)
            seen_ids.add(id(current_worker))
        for worker in list(self._retained_simulation_workers):
            if id(worker) in seen_ids:
                continue
            owned_workers.append(worker)
            seen_ids.add(id(worker))
        for worker in owned_workers:
            if self._worker_is_running(worker):
                return True
        return False

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

    def _preview_request_can_display(self, request_id: Optional[int]) -> bool:
        return self._completion_policy.preview_request_can_display(
            preview_ownership=self._completion_policy_preview_ownership(),
            request_id=request_id,
        )

    def _prepare_simulation_shutdown_for_close(self) -> bool:
        seen_ids: set[int] = set()
        owned_workers = []
        current_worker = getattr(self, "_simulation_worker", None)
        self._close_contained_simulation_owner(kill=True)
        if current_worker is not None:
            owned_workers.append(current_worker)
            seen_ids.add(id(current_worker))
        for worker in list(self._retained_simulation_workers):
            if id(worker) in seen_ids:
                continue
            owned_workers.append(worker)
            seen_ids.add(id(worker))

        for worker in owned_workers:
            still_running = self._cleanup_worker_safely(
                worker,
                "simulation worker (closeEvent)",
                retain_if_running=True,
                preserve_handlers_if_running=True,
            )
            if (not still_running) and getattr(self, "_simulation_worker", None) is worker:
                self._simulation_worker = None
        self._shutdown_batch_lane_pool(force_terminate=True)
        self._prune_stopped_owned_simulation_workers()
        has_running_workers = self._has_running_owned_simulation_workers()
        if has_running_workers:
            self.clear_pending_slider_preview_replay(clear_plot_updates=False)
        self._shutdown_requested_for_close = bool(has_running_workers)
        return not has_running_workers

    def _clear_shutdown_request_after_close_cleanup(self) -> None:
        self._prune_stopped_owned_simulation_workers()
        if bool(getattr(self, "_shutdown_requested_for_close", False)) and (not self._has_running_owned_simulation_workers()):
            self._shutdown_requested_for_close = False

    def _cleanup_worker_safely(
        self,
        worker,
        worker_name: str = "worker",
        *,
        retain_if_running: bool = False,
        preserve_handlers_if_running: bool = False,
    ) -> bool:
        if worker is None:
            return False
        if not self._worker_is_valid(worker):
            self._delete_worker_if_stopped(worker, worker_name)
            return False
        is_running = self._worker_is_running(worker)

        if is_running:
            logger.warning(f"Previous {worker_name} still running, requesting cancellation")
            if hasattr(worker, "cancel"):
                try:
                    worker.cancel()
                except Exception as exc:
                    self._record_nonfatal_exception(
                        f"Failed to request cancellation for {str(worker_name)}",
                        exc,
                    )

        still_running = self._worker_is_running(worker)
        should_disconnect_application_signals = not (
            bool(still_running) and bool(retain_if_running) and bool(preserve_handlers_if_running)
        )

        if should_disconnect_application_signals:
            self._disconnect_simulation_worker_application_signals(worker)

        if still_running:
            finished_signal = None
            try:
                finished_signal = worker.finished
            except Exception:
                finished_signal = None
            if finished_signal is not None and hasattr(finished_signal, "connect") and hasattr(worker, "deleteLater"):
                try:
                    finished_signal.connect(worker.deleteLater)
                    try:
                        if not self._worker_is_running(worker):
                            worker.deleteLater()
                    except Exception as exc:
                        self._record_nonfatal_exception(
                            f"Failed to deleteLater() for finished {str(worker_name)}",
                            exc,
                        )
                except Exception as exc:
                    self._record_nonfatal_exception(
                        f"Failed to connect finished->deleteLater for {str(worker_name)}",
                        exc,
                    )
            if retain_if_running:
                self._retain_simulation_worker(worker, worker_name)
        else:
            self._delete_worker_if_stopped(worker, worker_name)
        logger.debug(f"{worker_name} cleaned up successfully")
        return bool(still_running)

    def _release_current_simulation_worker(self) -> None:
        worker = getattr(self, "_simulation_worker", None)
        if worker is None:
            return
        self._cleanup_worker_safely(worker, "simulation worker", retain_if_running=True)
        if getattr(self, "_simulation_worker", None) is worker:
            self._simulation_worker = None

    def _contained_child_blas_threads_limited(self) -> bool:
        return bool(self.parallel_batch.limit_blas_threads_per_worker)

    def _contained_child_handler_env(self) -> Dict[str, str]:
        return contained_child_blas_thread_env(
            enabled=self._contained_child_blas_threads_limited()
        )

    def _runtime_lane_environment_key(self) -> str:
        blas_limited = bool(self._contained_child_blas_threads_limited())
        return f"contained-child-blas:{'limited' if blas_limited else 'unlimited'}"

    def _warm_runtime_lane_pool(self, capacity: int, *, wait: bool = False) -> tuple[tuple[str, int, str], ...]:
        requested = max(1, int(capacity or 1))
        self._batch_parallel.ensure_warm_lane_pool(max_lanes=requested, wait=bool(wait))
        if not self._batch_parallel.has_ready_lane_pool(max_lanes=requested):
            return ()
        snapshot = self._batch_parallel.runtime_snapshot()
        token = self._batch_parallel.lane_pool_token()
        token_s = str(token if token is not None else id(self._batch_parallel))
        generation = int(getattr(snapshot, "current_generation", 0) or 0)
        return tuple((f"runtime-lane-{token_s}-{index + 1}", generation, token_s) for index in range(requested))

    def _runtime_lane_is_live(self, lane) -> bool:
        token = self._batch_parallel.lane_pool_token()
        if token is None:
            return False
        token_s = str(token)
        lane_token = str(getattr(lane, "backend_pool_token", "") or "")
        if lane_token and lane_token != token_s:
            return False
        try:
            snapshot = self._batch_parallel.runtime_snapshot()
        except Exception:
            return False
        if bool(getattr(snapshot, "pool_stale", False)):
            return False
        lane_generation = int(getattr(lane, "generation", 0) or 0)
        current_generation = int(getattr(snapshot, "current_generation", 0) or 0)
        if lane_generation and current_generation and lane_generation != current_generation:
            return False
        return bool(self._batch_parallel.has_ready_lane_pool(max_lanes=1))

    def _release_active_runtime_dispatch_plan(self, *, failed: bool = False) -> None:
        dispatch_plan = getattr(self, "_active_runtime_dispatch_plan", None)
        if isinstance(dispatch_plan, RuntimeDispatchPlan):
            self._runtime_readiness_lifecycle.release_dispatch_plan(
                dispatch_plan,
                failed=bool(failed),
            )
        self._active_runtime_dispatch_plan = None

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

    def _close_contained_simulation_owner(
        self,
        *,
        fast_mode: Optional[bool] = None,
        kill: bool = False,
    ) -> None:
        self._runtime_readiness_lifecycle.release_all(failed=bool(kill))

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

    def _connect_simulation_worker_application_signals(
        self,
        worker,
        *,
        callback_identity: SimulationCallbackIdentity,
    ) -> None:
        if worker is None:
            return
        self._disconnect_simulation_worker_application_signals(worker)
        setattr(worker, "_kindred_runtime_dispatch_failed", False)
        connected_handlers = list(getattr(worker, _WORKER_APPLICATION_SIGNAL_HANDLERS_ATTR, ()) or ())
        progress_handler = self.on_simulation_progress

        def result_handler(
            payload,
            _identity=callback_identity,
        ):
            return self._on_simulation_complete(
                payload,
                callback_identity=_identity,
            )

        def error_handler(
            msg,
            _identity=callback_identity,
            _worker=worker,
        ):
            setattr(_worker, "_kindred_runtime_dispatch_failed", True)
            return self._on_simulation_error(
                msg,
                callback_identity=_identity,
            )

        for signal_name, handler in (
            ("progress", progress_handler),
            ("result_ready", result_handler),
            ("error", error_handler),
        ):
            signal = getattr(worker, signal_name, None)
            if signal is None or not hasattr(signal, "connect"):
                continue
            signal.connect(handler)
            connected_handlers.append((signal_name, handler))
        setattr(worker, _WORKER_APPLICATION_SIGNAL_HANDLERS_ATTR, tuple(connected_handlers))

    def _disconnect_simulation_worker_application_signals(self, worker) -> None:
        if worker is None:
            return
        connections = tuple(getattr(worker, _WORKER_APPLICATION_SIGNAL_HANDLERS_ATTR, ()) or ())
        remaining_connections: list[tuple[str, Any]] = []
        for signal_name, handler in connections:
            signal = getattr(worker, signal_name, None)
            if signal is None or not hasattr(signal, "disconnect"):
                continue
            try:
                signal.disconnect(handler)
            except TypeError:
                continue
            except RuntimeError as exc:
                remaining_connections.append((signal_name, handler))
                self._record_nonfatal_exception(
                    f"Failed to disconnect tracked simulation worker {signal_name} handler",
                    exc,
                )
        setattr(worker, _WORKER_APPLICATION_SIGNAL_HANDLERS_ATTR, tuple(remaining_connections))

    def _effective_batch_worker_count(self, num_sets: int) -> int:
        return min(
            int(self.batch_runtime_lane_budget),
            int(
                compute_effective_batch_workers(
                    num_sets=max(0, int(num_sets)),
                    max_parallel_workers=max(1, int(self._batch_parallel.max_parallel_workers)),
                )
            ),
        )

    def _shutdown_batch_lane_pool(self, *, force_terminate: bool) -> None:
        timer = getattr(self, "_batch_completion_poll_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        self._clear_pending_slider_plot_updates()
        prior_requests = int(self._batch_parallel.active_request_count())
        self._batch_parallel.shutdown(
            force_terminate=bool(force_terminate),
            record_nonfatal_exception=self._record_nonfatal_exception,
        )
        if bool(getattr(self, "_debug_batch_parallel", False)):
            logger.info(
                "BATCH_PAR shutdown lane owner force=%s pending_requests=%s",
                bool(force_terminate),
                int(prior_requests),
            )

    def _has_active_parallel_batch_work(self) -> bool:
        runtime_snapshot = self._batch_parallel.runtime_snapshot()
        if runtime_snapshot.active:
            return True
        return bool(self._batch_parallel.has_active_requests())

    def _parallel_batch_pool_settings_changed(self) -> str:
        if self._has_active_parallel_batch_work():
            self._batch_parallel.mark_pool_stale()
            return "marked_stale_draining"
        had_pool = bool(self._batch_parallel.has_lane_pool())
        self._shutdown_batch_lane_pool(force_terminate=False)
        return "shut_down" if had_pool else "idle_no_pool"

    def _simulation_runtime_inputs_changed(
        self,
        *,
        batch_runtime_pool_inputs_changed: bool = True,
    ) -> SimulationRuntimeInputsChangeOutcome:
        if bool(batch_runtime_pool_inputs_changed):
            batch_outcome = self._parallel_batch_pool_settings_changed()
        else:
            batch_outcome = "unchanged"
        return SimulationRuntimeInputsChangeOutcome(
            interactive_runtime_refresh_requested=True,
            batch_pool_shut_down=batch_outcome == "shut_down",
            batch_pool_marked_stale_draining=batch_outcome == "marked_stale_draining",
        )

    def _interactive_batch_runtime_capacity(self) -> int:
        max_workers = max(1, int(getattr(self._batch_parallel, "max_parallel_workers", 1) or 1))
        return max(
            1,
            min(
                int(self.batch_runtime_lane_budget),
                int(
                    compute_effective_batch_workers(
                        num_sets=max_workers,
                        max_parallel_workers=max_workers,
                    )
                ),
            ),
        )

    def _cleanup_parallel_batch_lane_pool_after_run(
        self,
        *,
        keep_lane_pool_alive: bool,
        clear_pending_plot_updates: bool = False,
        stale_fast_handoff_after_display: bool = False,
    ) -> None:
        if bool(keep_lane_pool_alive):
            self._batch_parallel.finish_after_run(
                keep_lane_pool_alive=True,
                record_nonfatal_exception=self._record_nonfatal_exception,
            )
            if bool(clear_pending_plot_updates):
                self._clear_pending_slider_plot_updates()
            self._release_active_runtime_dispatch_plan(failed=False)
            return
        self._shutdown_batch_lane_pool(force_terminate=False)
        self._release_active_runtime_dispatch_plan(failed=True)

    def _supersede_parallel_batch_run_soft(self) -> tuple[int, int]:
        """
        Supersede the active parallel run without destroying the process pool.

        Used by slider-triggered restarts to preserve worker processes and avoid
        pool recreation on every minor parameter update.
        """
        state = self._batch_context_owner.active_batch_state()
        if state is not None and state.active and (state.runtime_task_queue or state.parallel):
            self._batch_context_owner.deactivate()

        timer = getattr(self, "_batch_completion_poll_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()

        cancelled, running = self._batch_parallel.soft_supersede()
        self._clear_pending_slider_plot_updates()
        if running > 0 and timer is not None:
            timer.start()

        if bool(getattr(self, "_debug_batch_parallel", False)):
            logger.info(
                "BATCH_PAR soft-supersede cancelled=%s running=%s",
                int(cancelled),
                int(running),
            )
        return int(cancelled), int(running)

    # ------------------------------------------------------------------
    # Parallel completion queue helpers
    # ------------------------------------------------------------------
    def _drain_batch_completion_queue(self) -> None:
        self._batch_parallel.drain_completion_queue()

    def _enqueue_parallel_batch_completion(self, set_id: str) -> None:
        self._batch_parallel.enqueue_completion(set_id)

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
        if bool(close_runtime_owner):
            self._close_contained_simulation_owner(fast_mode=True, kill=True)
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

    def _active_explicit_worker_set_id(self) -> str:
        worker = getattr(self, "_simulation_worker", None)
        if worker is None:
            return ""
        return str(getattr(worker, "_batch_set_id", "") or "").strip()

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
            active_set_id = self._active_explicit_worker_set_id()
            if active_set_id and active_set_id in set(affected_scope):
                if not self._try_supersede_active_serial_runtime_input_set(
                    affected_set_ids=affected_scope,
                ):
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
        request_accepted = (
            self._preview_request_can_display(request_id)
            if bool(slider_triggered)
            else (request_id is None or int(request_id) == int(getattr(self, "_latest_sim_request_id", 0)))
        )
        accepted_target_set_ids = (
            preview_ownership.target_set_ids if bool(slider_triggered) and bool(request_accepted) else ()
        )
        self._plot_coalescer.queue(
            set_id=set_id,
            cache_key=cache_key,
            request_id=request_id,
            request_accepted=bool(request_accepted),
            run_id=run_id,
            accepted_preview_request_id=(
                preview_ownership.request_id if bool(slider_triggered) and bool(request_accepted) else None
            ),
            accepted_preview_owner_epoch=(
                int(preview_ownership.epoch) if bool(slider_triggered) and bool(request_accepted) else None
            ),
            slider_triggered=slider_triggered,
            valid_set_ids=valid_set_ids,
            fresh_preview_entry=fresh_preview_entry,
            target_set_ids=accepted_target_set_ids,
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
        pending_cache_key = str(pending.cache_key or "")
        pending_cache_kind = str(pending.cache_kind or "")
        pending_request_id = pending.request_id
        pending_run_id = pending.run_id
        pending_preview_request_id = pending.accepted_preview_request_id
        pending_preview_owner_epoch = pending.accepted_preview_owner_epoch
        pending_target_set_ids = tuple(str(set_id) for set_id in getattr(pending, "target_set_ids", ()) if str(set_id))
        pending_valid_set_ids = pending.valid_set_ids
        pending_fresh_preview_entries = dict(getattr(pending, "fresh_preview_entries", {}) or {})

        cache_key = str(cache_key or pending_cache_key or "")
        request_id = pending_request_id if request_id is None else request_id
        run_id = pending_run_id if run_id is None else run_id
        if not cache_key:
            return False
        request_accepted = (
            self._queued_preview_update_still_matches_current_preview_owner(
                request_id=request_id,
                accepted_preview_request_id=pending_preview_request_id,
                accepted_preview_owner_epoch=pending_preview_owner_epoch,
            )
            if pending_cache_kind == "preview"
            else (request_id is None or int(request_id) == int(getattr(self, "_latest_sim_request_id", 0)))
        )
        if request_id is not None and not bool(request_accepted):
            return False
        if run_id is not None and int(run_id) != int(getattr(self, "_active_run_id", 0)):
            return False

        live_requested_show_set_ids = list(self.ui.batch.requested_show_batch_set_ids())
        live_requested_show_ids = [str(set_id) for set_id in live_requested_show_set_ids if str(set_id)]
        requested_show_ids = (
            [str(set_id) for set_id in pending_target_set_ids if str(set_id)]
            if pending_cache_kind == "preview" and pending_target_set_ids
            else live_requested_show_ids
        )
        if not requested_show_ids:
            return False

        prefer = None
        current_row = self.ui.batch.batch_current_row()
        if current_row is not None:
            prefer = self.ui.batch.batch_set_id_for_row(int(current_row))

        display_outcome: object | None = None
        if pending_cache_kind == "preview":
            if set(live_requested_show_ids) == set(requested_show_ids):
                display_outcome = self.ui.results.refresh_display_from_request_scope(
                    display_source=DisplayRefreshSource.SLIDER_REPLAY,
                )
                if display_outcome.focused_controls_use_workspace is not None:
                    try:
                        self.ui.mechanism_helpers.sync_mechanism_controls_to_focused_batch_set(
                            use_workspace=bool(display_outcome.focused_controls_use_workspace)
                        )
                    except Exception as exc:
                        self._record_nonfatal_exception(
                            "Failed to resync focused mechanism controls after preview display refresh",
                            exc,
                        )
        else:
            if pending_valid_set_ids:
                self._batch_cache.apply_explicit_cache_reconciliation(
                    clear_active_cache_identity_state=False,
                    active_cache_key=str(cache_key),
                    active_cache_preview_token=None,
                    active_cache_preview_scope_set_ids=None,
                    active_cache_valid_set_ids=pending_valid_set_ids,
                    active_cache_invalidated_set_ids=None,
                )
            display_outcome = self.ui.results.publish_cached_batch_display_scope(
                cache_key=str(cache_key),
                requested_show_set_ids=requested_show_ids,
                prefer_set=prefer,
                display_source=DisplayRefreshSource.SLIDER_REPLAY,
            )
        displayed = self._display_transition_published(display_outcome)
        if pending_cache_kind == "preview" and not displayed:
            fresh_outcome = self._publish_fresh_preview_plot_update(
                fresh_preview_entries=pending_fresh_preview_entries,
                requested_show_set_ids=requested_show_ids,
                target_set_ids=pending_target_set_ids,
                prefer_set=prefer,
                cache_key=str(cache_key),
                request_id=request_id,
                run_id=run_id,
            )
            if fresh_outcome is not None:
                display_outcome = fresh_outcome
                displayed = self._display_transition_published(display_outcome)
        if bool(getattr(self, "_debug_batch_parallel", False)):
            logger.info(
                "BATCH_PAR plot flush run_id=%s request_id=%s changed_sets=%s forced=%s displayed=%s reason=%s ts=%.6f",
                int(run_id or 0),
                int(request_id or 0),
                int(len(pending_set_ids)),
                bool(force),
                bool(displayed),
                self._display_transition_log_reason(display_outcome),
                float(perf_counter()),
            )
        return bool(displayed)

    def _publish_fresh_preview_plot_update(
        self,
        *,
        fresh_preview_entries: Mapping[str, FreshPreviewDisplayEntry],
        requested_show_set_ids: Sequence[str],
        target_set_ids: Sequence[str],
        prefer_set: Optional[str],
        cache_key: str,
        request_id: Optional[int],
        run_id: Optional[int],
    ) -> Optional[SimulationCompletionDisplayOutcome]:
        entries_by_id = {
            str(set_id): entry
            for set_id, entry in dict(fresh_preview_entries or {}).items()
            if str(set_id) and isinstance(entry, FreshPreviewDisplayEntry)
        }
        if not entries_by_id:
            return None
        requested_show_ids = tuple(str(set_id) for set_id in (requested_show_set_ids or ()) if str(set_id))
        target_ids = tuple(dict.fromkeys(str(set_id) for set_id in (target_set_ids or ()) if str(set_id)))
        if not target_ids:
            return None
        display_ids = requested_show_ids or target_ids
        if (
            not display_ids
            or set(display_ids) != set(target_ids)
            or set(target_ids) != set(entries_by_id)
        ):
            return None
        primary_id = str(prefer_set or "").strip()
        if primary_id not in display_ids:
            primary_id = str(display_ids[0])
        transaction = FreshPreviewDisplayTransaction(
            entries=tuple(entries_by_id[set_id] for set_id in display_ids),
            display_set_ids=display_ids,
            target_set_ids=target_ids,
            display_primary_set_id=primary_id,
            cache_key=str(cache_key or ""),
            display_source=DisplayRefreshSource.SLIDER_REPLAY,
            requested_show_set_ids=requested_show_ids or display_ids,
            requested_labels_by_set_id={
                str(set_id): str(self.ui.batch.batch_set_name_for_id(str(set_id)) or set_id)
                for set_id in (requested_show_ids or display_ids)
                if str(set_id)
            },
            request_id=(int(request_id) if request_id is not None else None),
            run_id=(int(run_id) if run_id is not None else None),
        )
        outcome = self.ui.results.publish_fresh_preview_display(transaction)
        if not isinstance(outcome, SimulationCompletionDisplayOutcome):
            raise RuntimeError("ResultsController returned an invalid fresh-preview display outcome")
        return outcome

    @staticmethod
    def _display_transition_published(
        outcome: SimulationCompletionDisplayOutcome | None,
    ) -> bool:
        transition_outcome = outcome.transition_outcome if outcome is not None else None
        return (
            isinstance(transition_outcome, DisplayTransitionOutcome)
            and transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
        )

    @staticmethod
    def _display_transition_log_reason(
        outcome: SimulationCompletionDisplayOutcome | None,
    ) -> str:
        transition_outcome = outcome.transition_outcome if outcome is not None else None
        if isinstance(transition_outcome, DisplayTransitionOutcome):
            cause = transition_outcome.cause
            if cause is not None:
                return str(cause.value)
            return str(transition_outcome.kind.value)
        return ""

    # ------------------------------------------------------------------
    # Batch lane outcome polling/consumption
    # ------------------------------------------------------------------
    def _clear_stale_parallel_batch_requests(self) -> None:
        self._batch_parallel.clear_stale_requests()

    def _reset_parallel_batch_run_and_shutdown_lane_pool(self) -> None:
        state = self._batch_context_owner.active_batch_state()
        if state is not None and state.active and (state.runtime_task_queue or state.parallel):
            self._batch_context_owner.deactivate()
        self.shutdown_batch_lane_pool(force_terminate=True)
        self._clear_stale_parallel_batch_requests()
        self._drain_batch_completion_queue()

    def _surface_current_parallel_batch_pool_failure_to_ui(self, error_msg: object) -> None:
        dispatch_context = self._batch_context_owner.active_parallel_error_dispatch_context()
        if dispatch_context is None:
            return
        if not self._batch_parallel.has_lane_pool():
            return
        callback_identity = self._capture_simulation_callback_identity(
            run_id=int(dispatch_context.run_id),
            fast_mode=bool(dispatch_context.fast_mode),
            request_id=int(dispatch_context.request_id),
            preview_owner_epoch=dispatch_context.preview_owner_epoch,
            batch_set="",
            batch_set_id="",
            cache_key=str(dispatch_context.cache_key),
            callback_context=dispatch_context.callback_context,
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

    def _apply_explicit_failure_pending_replay_policy(self, *, fast_mode: bool) -> None:
        pending_replay_directive = self._completion_policy.resolve_explicit_error_pending_replay(
            fast_mode=bool(fast_mode),
            pending_replay=self._completion_policy_pending_replay_state(),
        )
        if pending_replay_directive.action in {"queue_fresh", "arm_existing"}:
            logger.debug("Replaying pending slider update after explicit failure")
            self._apply_completion_policy_state_patch(
                PolicyStatePatch(pending_replay=pending_replay_directive)
            )
            self._schedule_deferred_preview_replay_handoff_once()
        else:
            self._apply_completion_policy_state_patch(
                PolicyStatePatch(pending_replay=pending_replay_directive)
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
            pending_replay_directive = self._completion_policy.resolve_pending_replay_after_canonical_reset(
                pending_replay=self._completion_policy_pending_replay_state(),
                reset_set_ids=tuple(eligible_reset_set_ids),
            )
            self._apply_completion_policy_state_patch(
                PolicyStatePatch(pending_replay=pending_replay_directive)
            )
            if pending_replay_directive.action == "clear":
                for stop_fn, timer_name in (
                    (self.ui.slider.stop_variable_update_timer, "_variable_update_timer"),
                    (self.ui.slider.stop_species_slider_update_timer, "_species_slider_update_timer"),
                ):
                    try:
                        stop_fn()
                    except Exception as exc:
                        self._record_nonfatal_exception(
                            f"Failed to stop debounce timer {str(timer_name)} after canonical explicit reset",
                            exc,
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
            if summary.failed_set_ids and not summary.fast_mode:
                self.ui.run_ui.set_sim_progress_value(100)
                failed_count = len(summary.failed_set_ids)
                self.ui.run_ui.set_status_text(f"Batch completed with {failed_count} failed set(s)")
                self._show_scoped_batch_failure_summary(
                    failed_set_ids=summary.failed_set_ids,
                    failed_errors=summary.failed_errors,
                )
            elif summary.has_truthful_success:
                self.ui.run_ui.set_sim_progress_value(100)
                self.ui.run_ui.set_status_text(str(status_text))
            else:
                self.ui.run_ui.set_sim_progress_value(0)
                self.ui.run_ui.set_status_text("Ready")
            self.ui.run_ui.repaint_simulation_widgets()
        finally:
            self._release_current_simulation_worker()
            cleanup_state = self._batch_context_owner.completion_cleanup_state(ctx)
            keep_lane_pool_alive = bool(
                (cleanup_state.runtime_task_queue or cleanup_state.parallel)
                and cleanup_state.keep_lane_pool_alive
            )
            self._cleanup_parallel_batch_lane_pool_after_run(
                keep_lane_pool_alive=keep_lane_pool_alive,
            )
            self.ui.slider.set_slider_triggered_simulation(False)
            self._simulation_running = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self._slider_simulation_active = False
            if self._has_deferred_preview_replay_intent():
                logger.debug("Processing pending slider update after completion")
                if not shutdown_requested:
                    self._schedule_deferred_preview_replay_handoff_once()
            self._clear_shutdown_request_after_close_cleanup()

    def _continue_or_finish_serial_batch_after_stale_runtime_input(
        self,
        ctx: Mapping[str, Any],
    ) -> None:
        state = self._batch_context_owner.active_batch_state()
        transition = self._batch_context_owner.consume_stale_serial_queue_prefix_for_current_epochs(
            current_global_epoch=int(getattr(self, "_authoritative_runtime_input_global_epoch", 0) or 0),
            current_set_epoch_by_set_id=self._runtime_input_context_set_epochs(
                state.queue_ids if state is not None else ()
            ),
            current_epoch=int(getattr(self, "_authoritative_runtime_input_epoch", 0) or 0),
        )
        ctx = transition.context
        if transition.batch_done:
            self._finalize_batch_queue_done_without_result(ctx)
            return

        completion_state = self._batch_context_owner.completion_state(ctx)
        if completion_state is not None and len(completion_state.queue_ids) > 1:
            pos = int(transition.completed_count)
            total = max(1, int(completion_state.total or len(completion_state.queue_ids) or 1))
            overall = int((pos / float(total)) * 100.0)
            self.ui.run_ui.set_sim_progress_value(max(0, min(100, overall)))
        QtCore.QTimer.singleShot(0, self._start_next_batch_simulation)

    def _try_supersede_active_serial_runtime_input_set(
        self,
        *,
        affected_set_ids: Sequence[str],
    ) -> bool:
        affected_scope = set(self._normalize_runtime_input_set_ids(affected_set_ids))
        if not affected_scope:
            return False
        state = self._batch_context_owner.active_batch_state()
        if (
            state is None
            or not state.active
            or state.runtime_task_queue
            or state.parallel
            or state.fast_mode
        ):
            return False
        active_set_id = self._active_explicit_worker_set_id()
        if not active_set_id or active_set_id not in affected_scope:
            return False
        updated = self._batch_context_owner.record_active_serial_runtime_input_superseded(
            active_set_id=active_set_id,
        )
        worker = getattr(self, "_simulation_worker", None)
        if worker is not None:
            try:
                if hasattr(worker, "cancel"):
                    worker.cancel()
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to cancel stale serial worker during scoped runtime-input supersede",
                    exc,
                )
        self._release_current_simulation_worker()
        self._simulation_running = True
        self._slider_simulation_active = False
        self._continue_or_finish_serial_batch_after_stale_runtime_input(updated)
        return True

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
        coverage = self._batch_context_owner.completed_run_display_coverage(ctx)
        if coverage.transaction is not None:
            outcome = self._completion_publication_owner.publish_completed_run_display_transaction(
                coverage.transaction
            )
            if not isinstance(outcome, SimulationCompletionDisplayOutcome):
                raise TypeError("Completed-run display publication must return SimulationCompletionDisplayOutcome")
            return outcome.transition_outcome
        cause = coverage.cause
        if not isinstance(cause, DisplayTransitionCause):
            raise TypeError("Completed-run display coverage requires DisplayTransitionCause")
        affected_set_ids = tuple(
            str(set_id)
            for set_id in (
                coverage.unresolved_intent_set_ids
                or coverage.unavailable_set_ids
                or coverage.missing_set_ids
            )
            if str(set_id)
        )
        if not affected_set_ids and coverage.intent is not None:
            affected_set_ids = tuple(
                str(set_id) for set_id in coverage.intent.requested_show_set_ids if str(set_id)
            )
        outcome = self._completion_publication_owner.publish_completed_run_display_unavailable(
            cause=cause,
            affected_set_ids=affected_set_ids,
            requested_show_set_ids=(
                tuple(str(set_id) for set_id in coverage.intent.requested_show_set_ids if str(set_id))
                if coverage.intent is not None
                else affected_set_ids
            ),
            requested_labels_by_set_id=(
                {
                    str(set_id): str(label)
                    for set_id, label in dict(coverage.intent.labels_by_set_id or {}).items()
                    if str(set_id)
                }
                if coverage.intent is not None
                else {}
            ),
            unresolved_intent_set_ids=tuple(
                str(set_id) for set_id in coverage.unresolved_intent_set_ids if str(set_id)
            ),
            missing_intent_set_ids=tuple(
                str(set_id) for set_id in coverage.missing_set_ids if str(set_id)
            ),
            failed_intent_set_ids=tuple(
                str(set_id) for set_id in coverage.failed_intent_set_ids if str(set_id)
            ),
            semantic_unavailable_set_ids=tuple(
                str(set_id) for set_id in coverage.semantic_unavailable_set_ids if str(set_id)
            ),
        )
        if not isinstance(outcome, SimulationCompletionDisplayOutcome):
            raise TypeError("Completed-run display unavailable publication must return SimulationCompletionDisplayOutcome")
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

    def _stop_batch_completion_poll_timer_if_idle(self) -> None:
        state = self._batch_context_owner.active_batch_state()
        active_parallel = bool(
            state is not None
            and state.active
            and (state.runtime_task_queue or state.parallel)
        )
        if active_parallel or self._batch_parallel.has_active_requests():
            return
        timer = getattr(self, "_batch_completion_poll_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()

    def _poll_parallel_batch_completions(self) -> None:
        try:
            runtime_snapshot = self._batch_parallel.runtime_snapshot()
            active_parallel = bool(runtime_snapshot.active)
            if not active_parallel and not self._batch_parallel.has_active_requests():
                if self._batch_parallel.is_pool_stale:
                    self._shutdown_batch_lane_pool(force_terminate=False)
                self._stop_batch_completion_poll_timer_if_idle()
                if self._has_deferred_preview_replay_intent():
                    self._schedule_deferred_preview_replay_handoff_once()
                return

            for polled in self._batch_parallel.poll_completed_records():
                sid = str(polled.set_id or "")
                completion_record = polled.record
                if not self._consume_parallel_batch_outcome(
                    set_id=sid,
                    outcome=completion_record.outcome,
                    source=polled.source,
                    completed_ts=float(polled.completed_ts),
                    completion_record=completion_record,
                ):
                    return

            if (
                self._batch_parallel.is_pool_stale
                and (not self._batch_parallel.has_active_requests())
            ):
                self._shutdown_batch_lane_pool(force_terminate=False)
        except Exception as exc:
            # Architecture note (polling safety net):
            # This broad catch is a last-resort guard for the QTimer-driven poll
            # loop. If polling raises unexpectedly, letting the exception escape
            # can leave the GUI in a silent "stuck" state with a live lane owner
            # and no further timer ticks. We log, surface an error to the UI when
            # possible, and forcefully terminate the parallel lane owner to keep
            # the application recoverable.
            self._record_nonfatal_exception("Unhandled exception while polling parallel batch completions", exc)
            try:
                self._surface_current_parallel_batch_pool_failure_to_ui(f"Simulation failed:\n\n{exc}")
            except Exception as ui_exc:
                self._record_nonfatal_exception("Failed to surface polling exception to UI", ui_exc)
            self._shutdown_batch_lane_pool(force_terminate=True)
            return

        self._stop_batch_completion_poll_timer_if_idle()

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
        directive = self._completion_policy.resolve_preflight_abort_pending_replay(
            pending_replay=self._completion_policy_pending_replay_state(),
            explicit_run=True,
        )
        if directive is not None:
            self._apply_completion_policy_state_patch(PolicyStatePatch(pending_replay=directive))

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
        self._prune_stopped_owned_simulation_workers()
        if self._has_running_owned_simulation_workers():
            logger.warning(
                "Run Selected blocked while previous simulation worker shutdown remains in progress"
            )
            self._simulation_running = False
            self._slider_simulation_active = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self.ui.run_ui.set_status_text("Cancelling previous simulation...")
            return None

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
        active_fast = bool(state.fast_mode) if state is not None else False
        if state is not None and state.active:
            self._batch_context_owner.deactivate()
        self._shutdown_batch_lane_pool(force_terminate=True)
        worker = getattr(self, "_simulation_worker", None)
        if worker is not None:
            try:
                if hasattr(worker, "cancel"):
                    worker.cancel()
            except Exception as exc:
                self._record_nonfatal_exception("Failed to cancel active worker during restart", exc)
        self._close_contained_simulation_owner(fast_mode=active_fast, kill=True)
        if worker is not None:
            self._release_current_simulation_worker()
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
        ctx = None
        if state is not None and state.active:
            ctx = self._batch_context_owner.deactivate()
        self._shutdown_batch_lane_pool(force_terminate=True)
        active_fast = self._batch_context_owner.active_fast_mode(ctx)

        if self._worker_is_running(self._simulation_worker):
            self._simulation_worker.cancel()
            self._close_contained_simulation_owner(fast_mode=active_fast, kill=True)
            logger.info("Cancellation requested from simulation worker")
            self.ui.run_ui.set_status_text("Cancelling simulation...")
        else:
            self._close_contained_simulation_owner(fast_mode=active_fast, kill=True)
            self._simulation_running = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self.ui.run_ui.set_status_text("Ready")
            self.ui.run_ui.set_sim_progress_value(0)
