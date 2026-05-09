from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import suppress
import hashlib
import json
import logging
import os
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from PySide6 import QtCore
import shiboken6

from kindred.core.batch_parallel import (
    batch_mechanism_signature,
    compute_effective_batch_workers,
)
from kindred.core.batch_containment import BatchCompletionRecord, BatchLaneOutcome, BatchLanePool
from kindred.core.simulation_identity import (
    SimulationIdentity,
    contained_simulation_owner_identity,
)
from kindred.core.simulation_runtime_readiness import RuntimeReadinessSnapshot, SimulationRuntimeApplication
from kindred.core.simulation_failure import (
    coerce_simulation_failure,
    simulation_failure_from_exception,
    simulation_failure_user_message,
)
from kindred.gui.controllers.batch_run_context_owner import BatchRunContextOwner
from kindred.gui.controllers.batch_dispatch_plan import (
    ParallelBatchTaskInput,
    SerialBatchDispatchInput,
    build_fallback_cache_key as _build_fallback_cache_key,
    build_parallel_batch_task_plan,
    build_serial_batch_dispatch_plan,
)
from kindred.gui.controllers.batch_dispatch_materialization import BatchDispatchMaterializationOwner
from kindred.gui.controllers.serial_worker_launch import (
    ContainedSerialWorkerLaunchOwner,
    ContainedSerialWorkerLaunchRequest,
)
from kindred.gui.controllers.simulation_run_preparation import (
    RunDispatchContext,
    RunMechanismContext,
    RunSolverContext,
    SimulationRunPreparationDependencies,
    SimulationRunPreparationOwner,
    SimulationRunPreparationPorts,
    build_run_start_context,
)
from kindred.gui.controllers.simulation_result_materialization import SimulationResultMaterializationOwner
from kindred.gui.controllers.simulation_completion_policy import (
    CacheAuthorityState,
    CompletionPolicyContext,
    DirtySetState,
    PendingReplayDirective,
    PendingReplayState,
    PolicyStatePatch,
    RunActivitySnapshot,
    SimulationCompletionPolicy,
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
from kindred.gui.controllers.simulation_slider_preview_launch import (
    SimulationSliderPreviewLaunchDependencies,
    SimulationSliderPreviewLaunchOwner,
)
from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.gui.controllers.parallel_batch_executor import ParallelBatchExecutor
from kindred.gui.controllers.parallel_batch_runtime_readiness_owner import (
    ParallelBatchRuntimeReadinessOwner,
)
from kindred.gui.controllers.parallel_batch_outcome import (
    ParallelBatchOutcomeDependencies,
    ParallelBatchOutcomeOwner,
)
from kindred.gui.controllers.simulation_cache_admin import SimulationCacheAdmin
from kindred.gui.controllers.simulation_run_state import (
    PendingRunAfterRuntimeReadyState,
    PendingSliderPreviewLaunchState,
    PreviewOwnershipState,
    SimulationRunState,
)
from kindred.gui.controllers.slider_plot_coalescer import SliderPlotCoalescer
from kindred.gui.project_schema import PROJECT_DEFAULTS
from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.core.batch_initial_conditions import strip_reaction_dsl_initial_concentrations
from kindred.gui.ports import SimulationCacheOpResult, SimulationUiPorts, SliderReplayIntent

logger = logging.getLogger(__name__)

__all__ = ["SimulationController"]

_WORKER_APPLICATION_SIGNAL_HANDLERS_ATTR = "_kindred_controller_worker_signal_handlers"


@dataclass
class _SerialBatchDispatchState:
    plan_payload: Dict[str, Any] | None
    cache_key: str
    worker_signature: str | None
    context: Mapping[str, Any] | None


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _runtime_readiness_snapshot(
    *,
    mode: str,
    status: str,
    ready: bool = False,
    generation: int = 0,
    failure: Optional[str] = None,
    message: Optional[str] = None,
    required: bool = True,
    controls_ready: Optional[bool] = None,
    polling: Optional[bool] = None,
) -> RuntimeReadinessSnapshot:
    status_text = str(status or "missing")
    ready_value = bool(ready)
    required_value = bool(required)
    if controls_ready is None:
        controls_ready_value = bool(ready_value or not required_value)
    else:
        controls_ready_value = bool(controls_ready)
    if polling is None:
        polling_value = bool(
            required_value
            and not ready_value
            and status_text in {"missing", "warming", "not_ready", "stale", "rebuilding"}
        )
    else:
        polling_value = bool(polling)
    return RuntimeReadinessSnapshot(
        mode=str(mode or "ordinary"),
        status=status_text,
        ready=ready_value,
        generation=int(generation),
        failure=failure,
        message=message,
        required=required_value,
        controls_ready=controls_ready_value,
        polling=polling_value,
    )


def build_fallback_cache_key(
    mechanism_text: str = "",
    t_end: float = 0.0,
    solver_config: dict | None = None,
    *,
    simulation_identity: object | None = None,
) -> str:
    return _build_fallback_cache_key(
        mechanism_text,
        t_end,
        solver_config,
        simulation_identity=simulation_identity,
    )


def _default_batch_lane_pool_factory(max_workers: int, limit_blas_threads: bool):
    """Create a warm lane pool for batch simulations (injectable in tests)."""
    return BatchLanePool(
        max_lanes=max(1, int(max_workers)),
        limit_blas_threads_per_worker=bool(limit_blas_threads),
    )


class SimulationController(QtCore.QObject):
    """
    Simulation execution + batch orchestration controller.

    This keeps MainWindow focused on UI composition while preserving behavior by
    allowing controlled access to UI elements via a narrow UI port adapter.
    """

    def __init__(self, ui: SimulationUiPorts, *, parent: QtCore.QObject):
        super().__init__(parent)
        self.ui = ui

        self._run_state = SimulationRunState(on_progress_timeout=self._flush_progress_ui, parent=self)

        # ------------------------------------------------------------------
        # Batch execution + caching (migrated from MainWindow.__init__)
        # ------------------------------------------------------------------
        self._batch_run_queue: List[str] = []
        self._batch_run_results: Dict[str, Dict[str, Any]] = {}

        # Cache + selection state (explicit full results vs slider previews)
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
                apply_parameter_override_fallback_to_dsl=self._apply_parameter_override_fallback_to_dsl,
                invalidate_preserved_pending_init_results_after_failed_run=(
                    self._invalidate_preserved_pending_init_results_after_failed_run
                ),
                clear_failed_fast_preview_ownership=self._clear_failed_fast_preview_ownership,
                clear_slider_triggered_preflight_state=self._clear_slider_triggered_preflight_state,
                requeue_preserved_pending_slider_replay_after_preflight_abort=(
                    self._requeue_preserved_pending_slider_replay_after_preflight_abort
                ),
                record_nonfatal_exception=self._record_nonfatal_exception,
                set_simulation_running=self._set_simulation_running,
                set_slider_simulation_active=self._set_slider_simulation_active,
                sync_batch_species_columns_for_run=self._sync_batch_species_columns_for_run,
                slider_runtime_parameter_names=self._slider_runtime_parameter_names,
                simulation_identity_for_set=self._simulation_identity_for_set,
                request_mechanism_text_for_set=self._request_mechanism_text_for_set,
                resolved_initials_for_batch_row=self._resolved_initials_for_batch_row,
                slider_execution_parameter_values=self._slider_execution_parameter_values,
                preview_contained_owner_identity=self._preview_contained_owner_identity,
                ordinary_contained_owner_identity=self._ordinary_contained_owner_identity,
                record_run_cache_key=self._batch_cache.record_run_cache_key,
                batch_mechanism_signature=lambda **kwargs: batch_mechanism_signature(**kwargs),
            ),
        )
        self._batch_context_owner = BatchRunContextOwner()
        self._batch_dispatch_materialization_owner = BatchDispatchMaterializationOwner(
            batch=self.ui.batch,
            slider=self.ui.slider,
        )
        self._authoritative_mechanism_transition_epoch = 0
        self._authoritative_runtime_input_epoch = 0
        self._authoritative_runtime_input_global_epoch = 0
        self._authoritative_runtime_input_set_epoch_by_set_id: Dict[str, int] = {}

        # Parallel batch orchestration (warm lane owner adapter)
        self._batch_parallel = ParallelBatchExecutor(
            lane_pool_factory=_default_batch_lane_pool_factory,
            max_parallel_workers=int(PROJECT_DEFAULTS["max_parallel_batch_workers"]),
            limit_blas_threads_per_worker=bool(PROJECT_DEFAULTS["limit_blas_threads_per_worker"]),
            record_nonfatal_exception=self._record_nonfatal_exception,
        )
        self._parallel_batch_runtime_readiness_owner = ParallelBatchRuntimeReadinessOwner(
            batch_parallel=self._batch_parallel,
            capacity_getter=self._interactive_batch_runtime_capacity,
        )

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
        self._pending_reaction_init_stub: Optional[str] = None
        self._pending_reaction_init_stub_request_id: Optional[int] = None
        self._nonfatal_exception_count: int = 0
        self._last_nonfatal_exception: Optional[str] = None
        self._retained_simulation_workers: List[object] = []
        self._shutdown_requested_for_close: bool = False
        self._discarded_slider_preview_generation_id: Optional[int] = None
        self._batch_runtime_lane_budget = int(PROJECT_DEFAULTS["batch_runtime_lane_budget"])
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
            dependencies=SimulationCompletionPublicationDependencies(
                completion_policy_cache_state=lambda *args, **kwargs: self._completion_policy_cache_state(
                    *args, **kwargs
                ),
                resolve_completion_mechanism=self._result_materialization_owner.resolve_completion_mechanism,
                update_primary_result_materialization_contract=(
                    self._result_materialization_owner.update_primary_result_materialization_contract
                ),
                remember_primary_result_mechanism=self._result_materialization_owner.remember_primary_result_mechanism,
                include_mechanism_in_result_payload=lambda *args, **kwargs: self._include_mechanism_in_result_payload(
                    *args, **kwargs
                ),
                apply_lifecycle_effects=lambda *args, **kwargs: self._apply_simulation_lifecycle_effects(
                    *args, **kwargs
                ),
                record_nonfatal_exception=lambda *args, **kwargs: self._record_nonfatal_exception(*args, **kwargs),
                queue_slider_plot_update=lambda *args, **kwargs: self.queue_slider_plot_update(*args, **kwargs),
                finalize_explicit_batch_dirty_reset=(
                    lambda *args, **kwargs: self._finalize_explicit_batch_dirty_reset(*args, **kwargs)
                ),
                flush_slider_plot_updates=lambda *args, **kwargs: self.flush_slider_plot_updates(*args, **kwargs),
                show_scoped_batch_failure_summary=lambda *args, **kwargs: self._show_scoped_batch_failure_summary(
                    *args, **kwargs
                ),
                refresh_primary_result_controls=self._result_materialization_owner.refresh_primary_result_controls,
                has_deferred_preview_replay_intent=lambda *args, **kwargs: self._has_deferred_preview_replay_intent(
                    *args, **kwargs
                ),
                start_next_batch_simulation=self._start_next_batch_simulation,
            ),
        )
        self._completion_callback_owner = SimulationCompletionCallbackOwner(
            ui=self.ui,
            batch_context_owner=self._batch_context_owner,
            completion_policy=self._completion_policy,
            lifecycle_effect_owner=self._lifecycle_effect_owner,
            publication_owner=self._completion_publication_owner,
            dependencies=SimulationCompletionCallbackDependencies(
                active_run_id=lambda: int(getattr(self, "_active_run_id", 0)),
                shutdown_requested=lambda: bool(getattr(self, "_shutdown_requested_for_close", False)),
                latest_request_id=lambda: int(getattr(self, "_latest_sim_request_id", 0)),
                current_global_epoch=lambda: int(
                    getattr(self, "_authoritative_runtime_input_global_epoch", 0) or 0
                ),
                active_batch_context_runtime_input_stale_for_set=self._active_batch_context_runtime_input_stale_for_set,
                mark_stale_runtime_input_callback_consumed=self._mark_stale_runtime_input_callback_consumed,
                effective_preview_owner_epoch_for_callback=self._effective_preview_owner_epoch_for_callback,
                missing_preview_owner_epoch_for_current_fast_owner=(
                    self._missing_preview_owner_epoch_for_current_fast_owner
                ),
                preview_request_matches_current_owner_epoch=self._preview_request_matches_current_owner_epoch,
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
                active_run_id=lambda: int(getattr(self, "_active_run_id", 0)),
                latest_request_id=lambda: int(getattr(self, "_latest_sim_request_id", 0)),
                current_global_epoch=lambda: int(
                    getattr(self, "_authoritative_runtime_input_global_epoch", 0) or 0
                ),
                active_batch_context_runtime_input_stale_for_set=self._active_batch_context_runtime_input_stale_for_set,
                mark_stale_runtime_input_callback_consumed=self._mark_stale_runtime_input_callback_consumed,
                effective_preview_owner_epoch_for_callback=self._effective_preview_owner_epoch_for_callback,
                missing_preview_owner_epoch_for_current_fast_owner=(
                    self._missing_preview_owner_epoch_for_current_fast_owner
                ),
                preview_request_matches_current_owner_epoch=self._preview_request_matches_current_owner_epoch,
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
            dependencies=ParallelBatchOutcomeDependencies(
                active_batch_context_runtime_input_stale_for_set=(
                    lambda **kwargs: self._active_batch_context_runtime_input_stale_for_set(**kwargs)
                ),
                mark_stale_runtime_input_callback_consumed=(
                    lambda **kwargs: self._mark_stale_runtime_input_callback_consumed(**kwargs)
                ),
                record_nonfatal_exception=lambda *args, **kwargs: self._record_nonfatal_exception(*args, **kwargs),
                invalidate_preserved_pending_init_results_after_failed_run=(
                    lambda **kwargs: self._invalidate_preserved_pending_init_results_after_failed_run(**kwargs)
                ),
                finalize_scoped_batch_success_subset=lambda ctx: self._finalize_scoped_batch_success_subset(ctx),
                cleanup_parallel_batch_lane_pool_after_run=(
                    lambda **kwargs: self._cleanup_parallel_batch_lane_pool_after_run(**kwargs)
                ),
                show_scoped_batch_failure_summary=lambda **kwargs: self._show_scoped_batch_failure_summary(**kwargs),
                apply_explicit_failure_pending_replay_policy=(
                    lambda **kwargs: self._apply_explicit_failure_pending_replay_policy(**kwargs)
                ),
                reset_parallel_batch_run_and_shutdown_lane_pool=(
                    lambda: self._reset_parallel_batch_run_and_shutdown_lane_pool()
                ),
                dispatch_simulation_error=lambda *args, **kwargs: self._dispatch_simulation_error(*args, **kwargs),
                dispatch_simulation_complete=lambda *args, **kwargs: self._dispatch_simulation_complete(*args, **kwargs),
                set_simulation_running=self._set_simulation_running,
                set_slider_simulation_active=self._set_slider_simulation_active,
            ),
        )
        self._slider_preview_launch_owner = SimulationSliderPreviewLaunchOwner(
            ui=self.ui,
            run_state=self._run_state,
            batch_context_owner=self._batch_context_owner,
            dependencies=SimulationSliderPreviewLaunchDependencies(
                preview_owner_request_id=lambda: self._preview_ownership.request_id,
                set_discarded_preview_generation=self._set_discarded_slider_preview_generation,
                worker_is_running=lambda worker: self._worker_is_running(worker),
                clear_pending_slider_preview_replay=(
                    lambda *args, **kwargs: self.clear_pending_slider_preview_replay(*args, **kwargs)
                ),
                next_slider_preview_request_id=lambda: self._next_slider_preview_request_id(),
                queue_pending_slider_preview_replay=(
                    lambda *args, **kwargs: self.queue_pending_slider_preview_replay(*args, **kwargs)
                ),
                has_active_explicit_simulation=lambda: self._has_active_explicit_simulation(),
                has_active_parallel_batch_work=lambda: self._has_active_parallel_batch_work(),
                supersede_parallel_batch_run_soft=lambda: self._supersede_parallel_batch_run_soft(),
                prune_stopped_owned_simulation_workers=lambda: self._prune_stopped_owned_simulation_workers(),
                has_running_owned_simulation_workers=lambda: self._has_running_owned_simulation_workers(),
                slider_target_rows_for_dispatch=(
                    lambda *args, **kwargs: self._slider_target_rows_for_dispatch(*args, **kwargs)
                ),
                slider_preview_uses_parallel_batch_runtime=(
                    lambda *args, **kwargs: self._slider_preview_uses_parallel_batch_runtime(*args, **kwargs)
                ),
                slider_preview_runtime_snapshot=(
                    lambda *args, **kwargs: self._slider_preview_runtime_snapshot(*args, **kwargs)
                ),
                ensure_parallel_batch_pool_eagerly_created=(
                    lambda *args, **kwargs: self._ensure_parallel_batch_pool_eagerly_created(*args, **kwargs)
                ),
                ensure_interactive_simulation_runtime_available_for_mode=(
                    lambda *args, **kwargs: self._ensure_interactive_simulation_runtime_available_for_mode(
                        *args, **kwargs
                    )
                ),
                mark_request_started=lambda request_id: self._mark_request_started(request_id),
                run_simulation_internal=lambda **kwargs: self.run_simulation_internal(**kwargs),
                retry_slider_preview_launch=self._run_simulation_from_slider,
            ),
        )
        self._runtime_application = SimulationRuntimeApplication()
        self._contained_serial_worker_launch_owner = ContainedSerialWorkerLaunchOwner(
            acquire_ready_owner_for_plan=lambda **kwargs: self._acquire_ready_contained_simulation_owner_for_plan(
                **kwargs
            ),
            release_owner=lambda owner, *, kill=False: self._runtime_application.release_owner(owner, kill=kill),
            record_nonfatal_exception=self._record_nonfatal_exception,
        )

    # ------------------------------------------------------------------
    # Public interface (MainWindow boundary)
    # ------------------------------------------------------------------
    @property
    def _batch_parallel(self):
        return self._batch_parallel_adapter

    @_batch_parallel.setter
    def _batch_parallel(self, value) -> None:
        self._batch_parallel_adapter = value
        readiness_owner = getattr(self, "_parallel_batch_runtime_readiness_owner", None)
        if readiness_owner is not None:
            readiness_owner.batch_parallel = value
        outcome_owner = getattr(self, "_parallel_batch_outcome_owner", None)
        if outcome_owner is not None:
            outcome_owner.batch_parallel = value

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

    @property
    def _ordinary_simulation_owner(self):
        return self._runtime_application.current_owner(mode="ordinary")

    @_ordinary_simulation_owner.setter
    def _ordinary_simulation_owner(self, value) -> None:
        self._runtime_application.adopt_owner(mode="ordinary", owner=value)

    @property
    def _preview_simulation_owner(self):
        return self._runtime_application.current_owner(mode="preview")

    @_preview_simulation_owner.setter
    def _preview_simulation_owner(self, value) -> None:
        self._runtime_application.adopt_owner(mode="preview", owner=value)

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

    @_pending_slider_simulation.setter
    def _pending_slider_simulation(self, value: object) -> None:
        self._run_state.pending_slider_preview_launch = replace(
            self._pending_slider_preview_launch,
            active=value,
        )

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

    @_pending_slider_sim_request_id.setter
    def _pending_slider_sim_request_id(self, value: Optional[int]) -> None:
        self._run_state.pending_slider_preview_launch = replace(
            self._pending_slider_preview_launch,
            request_id=(int(value) if value is not None else None),
        )

    @property
    def _pending_slider_target_set_ids(self) -> Tuple[str, ...]:
        return tuple(
            str(set_id)
            for set_id in (self._pending_slider_preview_launch.target_set_ids or ())
            if str(set_id)
        )

    @_pending_slider_target_set_ids.setter
    def _pending_slider_target_set_ids(self, value: Sequence[str]) -> None:
        normalized: list[str] = []
        seen: set[str] = set()
        values = (value,) if isinstance(value, str) else value
        for set_id in values or ():
            set_id_s = str(set_id or "").strip()
            if not set_id_s or set_id_s in seen:
                continue
            seen.add(set_id_s)
            normalized.append(set_id_s)
        self._run_state.pending_slider_preview_launch = replace(
            self._pending_slider_preview_launch,
            target_set_ids=tuple(normalized),
        )

    @property
    def _pending_slider_handoff_queued(self) -> bool:
        return bool(self._pending_slider_preview_launch.handoff_queued)

    @_pending_slider_handoff_queued.setter
    def _pending_slider_handoff_queued(self, value: bool) -> None:
        self._run_state.pending_slider_preview_launch = replace(
            self._pending_slider_preview_launch,
            handoff_queued=bool(value),
        )

    @property
    def _pending_slider_preview_launch(self) -> PendingSliderPreviewLaunchState:
        replay = getattr(self._run_state, "pending_slider_preview_launch", None)
        if isinstance(replay, PendingSliderPreviewLaunchState):
            return replay
        normalized = PendingSliderPreviewLaunchState()
        self._run_state.pending_slider_preview_launch = normalized
        return normalized

    @property
    def _pending_run_after_runtime_ready(self) -> PendingRunAfterRuntimeReadyState:
        pending = getattr(self._run_state, "pending_run_after_runtime_ready", None)
        if isinstance(pending, PendingRunAfterRuntimeReadyState):
            return pending
        normalized = PendingRunAfterRuntimeReadyState()
        self._run_state.pending_run_after_runtime_ready = normalized
        return normalized

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

    def _run_intent_signature_for_rows(self, rows: Sequence[int]) -> str:
        rows_tuple = tuple(int(row) for row in rows or ())
        try:
            payloads = self._build_runtime_readiness_plan_payloads(
                fast_mode=False,
                batch_rows=rows_tuple,
            )
        except Exception as exc:
            payloads = [{"error": type(exc).__name__, "message": str(exc)}]
        material = {
            "rows": list(rows_tuple),
            "target_set_ids": list(self._run_target_set_ids_for_rows(rows_tuple)),
            "parallel_batch_runtime": bool(self._selected_run_uses_parallel_batch_runtime()),
            "payloads": payloads,
        }
        try:
            return json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            return repr(material)

    def _clear_pending_run_after_runtime_ready(self) -> None:
        self._run_state.pending_run_after_runtime_ready = PendingRunAfterRuntimeReadyState()

    def _ensure_selected_run_runtime_warming(self) -> None:
        if self._selected_run_uses_parallel_batch_runtime():
            self._ensure_parallel_batch_pool_eagerly_created(wait=False)
        else:
            self._ensure_interactive_simulation_runtime_available_for_mode(fast_mode=False, wait=False)

    def _queue_run_after_runtime_ready(
        self,
        *,
        rows_to_run: Sequence[int],
        runtime_snapshot: RuntimeReadinessSnapshot,
    ) -> None:
        if not bool(runtime_snapshot.should_poll):
            self._clear_pending_run_after_runtime_ready()
            return
        self._ensure_selected_run_runtime_warming()
        self._run_state.pending_run_after_runtime_ready = PendingRunAfterRuntimeReadyState(
            active=True,
            rows=tuple(int(row) for row in rows_to_run or ()),
            target_set_ids=self._run_target_set_ids_for_rows(rows_to_run),
            intent_signature=self._run_intent_signature_for_rows(rows_to_run),
        )
        QtCore.QTimer.singleShot(50, self._retry_pending_run_after_runtime_ready)

    def _restore_run_controls_after_pending_run_cancelled(self) -> None:
        runtime_snapshot = self._selected_run_runtime_snapshot()
        if bool(runtime_snapshot.required) and not bool(runtime_snapshot.ready):
            self._ensure_selected_run_runtime_warming()
            self.ui.run_ui.set_runtime_backed_run_controls_ready(False)
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self.ui.run_ui.set_status_text(
                str(runtime_snapshot.message or "Preparing simulation runtime...")
            )
            if bool(runtime_snapshot.should_poll):
                self.ui.run_ui.schedule_runtime_availability_refresh()
            return
        self.ui.run_ui.set_runtime_backed_run_controls_ready(True)
        self.ui.run_ui.set_run_button_enabled(True)
        self.ui.run_ui.set_stop_button_enabled(False)
        self.ui.run_ui.set_status_text("Ready.")

    def _retry_pending_run_after_runtime_ready(self) -> None:
        pending = self._pending_run_after_runtime_ready
        if not pending.active:
            return
        if bool(getattr(self, "_simulation_running", False)) or self._has_running_owned_simulation_workers():
            QtCore.QTimer.singleShot(50, self._retry_pending_run_after_runtime_ready)
            return
        current_rows = tuple(self.ui.batch.batch_rows_for_scope("selected") or ())
        if current_rows != tuple(pending.rows) or self._run_target_set_ids_for_rows(current_rows) != tuple(
            pending.target_set_ids
        ) or self._run_intent_signature_for_rows(current_rows) != str(
            pending.intent_signature or ""
        ):
            self._clear_pending_run_after_runtime_ready()
            self._restore_run_controls_after_pending_run_cancelled()
            return

        runtime_snapshot = self._selected_run_runtime_snapshot()
        if bool(runtime_snapshot.required) and not bool(runtime_snapshot.ready):
            self._ensure_selected_run_runtime_warming()
            self.ui.run_ui.set_runtime_backed_run_controls_ready(False)
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self.ui.run_ui.set_status_text(
                str(runtime_snapshot.message or "Preparing simulation runtime...")
            )
            if bool(runtime_snapshot.should_poll):
                QtCore.QTimer.singleShot(50, self._retry_pending_run_after_runtime_ready)
            else:
                self._clear_pending_run_after_runtime_ready()
            return

        self._clear_pending_run_after_runtime_ready()
        self.ui.run_ui.set_runtime_backed_run_controls_ready(True)
        self._run_simulation()

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
        QtCore.QTimer.singleShot(0, self._run_simulation_from_slider)
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
        if effects.modal_error is not None:
            self.ui.dialogs.message_box_critical(
                effects.modal_error.title,
                effects.modal_error.message,
                details=effects.modal_error.details,
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
                self.ui.run_ui.set_algebra_status_text(str(effects.algebra_status_text))
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
        if effects.show_preview_unavailable_status is not None:
            try:
                self.ui.slider.show_preview_unavailable_for_dirty_state(
                    str(effects.show_preview_unavailable_status)
                )
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to show dirty no-preview state after preview failure",
                    exc,
                )
                self.ui.run_ui.set_status_text(str(effects.show_preview_unavailable_status))
        if bool(effects.schedule_deferred_preview_replay):
            self._schedule_deferred_preview_replay_handoff_once(
                stop_timers=bool(effects.deferred_replay_stop_timers),
            )
        if bool(effects.apply_explicit_failure_pending_replay):
            self._apply_explicit_failure_pending_replay_policy(
                fast_mode=bool(effects.close_contained_fast_mode)
            )
        if bool(effects.invalidate_failed_pending_init_results):
            self._invalidate_preserved_pending_init_results_after_failed_run(
                ctx=failed_run_context if isinstance(failed_run_context, Mapping) else None,
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

    def _preview_request_matches_current_owner(self, request_id: Optional[int]) -> bool:
        if request_id is None:
            return True
        owner_request_id = self._preview_ownership.request_id
        if owner_request_id is None:
            return False
        return int(owner_request_id) == int(request_id)

    def _preview_request_matches_current_owner_epoch(
        self,
        request_id: Optional[int],
        owner_epoch: Optional[int],
    ) -> bool:
        if not self._preview_request_matches_current_owner(request_id):
            return False
        if owner_epoch is None:
            return True
        return int(self._preview_ownership.epoch) == int(owner_epoch)

    def _queued_preview_update_still_matches_current_owner(
        self,
        *,
        request_id: Optional[int],
        accepted_owner_request_id: Optional[int],
        accepted_owner_epoch: Optional[int],
    ) -> bool:
        if request_id is None:
            return True
        current = self._preview_ownership
        if accepted_owner_request_id is None or accepted_owner_epoch is None:
            return False
        return (
            current.request_id is not None
            and int(current.request_id) == int(request_id)
            and int(accepted_owner_request_id) == int(request_id)
            and int(current.epoch) == int(accepted_owner_epoch)
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
    def _pending_slider_plot_owner_request_id(self) -> Optional[int]:
        return self._plot_coalescer.pending.accepted_owner_request_id

    @_pending_slider_plot_owner_request_id.setter
    def _pending_slider_plot_owner_request_id(self, value: Optional[int]) -> None:
        self._plot_coalescer.pending.accepted_owner_request_id = int(value) if value is not None else None

    @property
    def _pending_slider_plot_owner_epoch(self) -> Optional[int]:
        return self._plot_coalescer.pending.accepted_owner_epoch

    @_pending_slider_plot_owner_epoch.setter
    def _pending_slider_plot_owner_epoch(self, value: Optional[int]) -> None:
        self._plot_coalescer.pending.accepted_owner_epoch = int(value) if value is not None else None

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
    def parallel_batch_runtime_readiness_owner(self) -> ParallelBatchRuntimeReadinessOwner:
        return self._parallel_batch_runtime_readiness_owner

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

    def _completion_policy_cache_state(self) -> CacheAuthorityState:
        return CacheAuthorityState(
            active_cache_key=self._batch_cache.active_cache_key,
            active_cache_preview_token=self._batch_cache.active_cache_preview_token,
            active_cache_preview_scope_set_ids=self._batch_cache.active_cache_preview_scope_set_ids,
            active_cache_valid_set_ids=self._batch_cache.active_cache_valid_set_ids,
            active_cache_invalidated_set_ids=self._batch_cache.active_cache_invalidated_set_ids,
        )

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

    def queue_slider_plot_update(
        self,
        *,
        set_id: Optional[str],
        cache_key: Optional[str],
        request_id: Optional[int],
        run_id: Optional[int],
        slider_triggered: bool = True,
        valid_set_ids: Optional[Sequence[str]] = None,
        allow_fallback: bool = True,
    ) -> None:
        self._queue_slider_plot_update(
            set_id=set_id,
            cache_key=cache_key,
            request_id=request_id,
            run_id=run_id,
            slider_triggered=slider_triggered,
            valid_set_ids=valid_set_ids,
            allow_fallback=allow_fallback,
        )

    def next_sim_request_id(self) -> int:
        return int(self._next_sim_request_id())

    def next_slider_preview_request_id(self) -> int:
        return int(self._next_slider_preview_request_id())

    def launch_pending_slider_preview_replay(self) -> None:
        self._run_simulation_from_slider()

    def run_simulation(self) -> None:
        self._run_simulation()

    def stop_simulation(self) -> None:
        self._stop_simulation()

    def ensure_interactive_simulation_runtimes_available(self, *, wait: bool = False) -> None:
        for fast_mode in (False, True):
            try:
                self._ensure_interactive_simulation_runtime_available_for_mode(
                    fast_mode=bool(fast_mode),
                    wait=bool(wait),
                )
            except Exception as exc:
                mode_label = "preview" if bool(fast_mode) else "ordinary"
                self._record_nonfatal_exception(
                    f"Failed to make {mode_label} contained simulation runtime available",
                    exc,
                )

    def invalidate_interactive_simulation_runtimes(self, *, kill: bool = False) -> None:
        self._runtime_application.close(kill=bool(kill))

    def interactive_simulation_runtimes_ready(self) -> bool:
        return bool(
            self.interactive_simulation_runtime_ready(fast_mode=False)
            and self.interactive_simulation_runtime_ready(fast_mode=True)
        )

    def interactive_simulation_runtime_snapshot(self, *, fast_mode: bool) -> RuntimeReadinessSnapshot:
        return self._interactive_simulation_runtime_snapshot(fast_mode=bool(fast_mode))

    def interactive_simulation_runtime_ready(self, *, fast_mode: bool) -> bool:
        return bool(self._interactive_simulation_runtime_snapshot(fast_mode=bool(fast_mode)).ready)

    def slider_preview_runtime_snapshot(self) -> RuntimeReadinessSnapshot:
        return self._slider_preview_runtime_snapshot()

    def invalidate_slider_preview_work(self) -> None:
        self._invalidate_slider_preview_work()

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

    def run_simulation_internal(
        self,
        *,
        fast_mode: bool = False,
        request_id: Optional[int] = None,
        batch_rows: Optional[Sequence[int]] = None,
        reuse_parallel_lane_pool: bool = False,
    ) -> None:
        self._run_simulation_internal(
            fast_mode=fast_mode,
            request_id=request_id,
            batch_rows=batch_rows,
            reuse_parallel_lane_pool=reuse_parallel_lane_pool,
        )

    def poll_parallel_batch_completions(self) -> None:
        self._poll_parallel_batch_completions()

    def shutdown_batch_lane_pool(self, *, force_terminate: bool) -> None:
        self._shutdown_batch_lane_pool(force_terminate=force_terminate)

    def parallel_batch_pool_settings_changed(self) -> None:
        self._parallel_batch_pool_settings_changed()

    def ensure_parallel_batch_pool_eagerly_created(self, *, wait: bool = False) -> None:
        self._ensure_parallel_batch_pool_eagerly_created(wait=bool(wait))

    def parallel_batch_runtime_ready(self) -> bool:
        return self._parallel_batch_runtime_ready()

    def selected_run_uses_parallel_batch_runtime(self) -> bool:
        return bool(self._selected_run_uses_parallel_batch_runtime())

    def selected_run_runtime_snapshot(self) -> RuntimeReadinessSnapshot:
        return self._selected_run_runtime_snapshot()

    def selected_run_runtime_ready(self) -> bool:
        return bool(self._selected_run_runtime_snapshot().ready)

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

    def start_parallel_batch_simulations(self) -> None:
        self._start_parallel_batch_simulations()

    def start_next_batch_simulation(self) -> None:
        self._start_next_batch_simulation()

    def on_simulation_progress(self, percent: int, message: str) -> None:
        self._on_simulation_progress(percent, message)

    def on_simulation_complete(
        self,
        result: dict,
        *,
        run_id: Optional[int] = None,
        fast_mode: Optional[bool] = None,
        request_id: Optional[int] = None,
        batch_set: Optional[str] = None,
        batch_set_id: Optional[str] = None,
        cache_key: Optional[str] = None,
        callback_identity: SimulationCallbackIdentity | None = None,
    ):
        return self._on_simulation_complete(
            result,
            run_id=run_id,
            fast_mode=fast_mode,
            request_id=request_id,
            batch_set=batch_set,
            batch_set_id=batch_set_id,
            cache_key=cache_key,
            callback_identity=callback_identity,
        )

    def on_simulation_error(
        self,
        error_msg: object,
        *,
        run_id: Optional[int] = None,
        fast_mode: Optional[bool] = None,
        request_id: Optional[int] = None,
        batch_set: Optional[str] = None,
        batch_set_id: Optional[str] = None,
        cache_key: Optional[str] = None,
        callback_identity: SimulationCallbackIdentity | None = None,
    ) -> None:
        self._on_simulation_error(
            error_msg,
            run_id=run_id,
            fast_mode=fast_mode,
            request_id=request_id,
            batch_set=batch_set,
            batch_set_id=batch_set_id,
            cache_key=cache_key,
            callback_identity=callback_identity,
        )

    def _dispatch_simulation_complete(
        self,
        result: dict,
        *,
        run_id: Optional[int] = None,
        fast_mode: Optional[bool] = None,
        request_id: Optional[int] = None,
        owner_epoch: Optional[int] = None,
        batch_set: Optional[str] = None,
        batch_set_id: Optional[str] = None,
        cache_key: Optional[str] = None,
        callback_identity: SimulationCallbackIdentity | None = None,
    ):
        identity = callback_identity or self._capture_simulation_callback_identity(
            run_id=run_id,
            fast_mode=fast_mode,
            request_id=request_id,
            owner_epoch=owner_epoch,
            batch_set=batch_set,
            batch_set_id=batch_set_id,
            cache_key=cache_key,
        )
        if identity.owner_epoch is None:
            return self.on_simulation_complete(
                result,
                run_id=identity.run_id,
                fast_mode=identity.fast_mode,
                request_id=identity.request_id,
                batch_set=identity.batch_set,
                batch_set_id=identity.batch_set_id,
                cache_key=identity.cache_key,
                callback_identity=identity,
            )
        return self._on_simulation_complete(
            result,
            run_id=identity.run_id,
            fast_mode=identity.fast_mode,
            request_id=identity.request_id,
            owner_epoch=identity.owner_epoch,
            batch_set=identity.batch_set,
            batch_set_id=identity.batch_set_id,
            cache_key=identity.cache_key,
            callback_identity=identity,
        )

    def _dispatch_simulation_error(
        self,
        error_msg: object,
        *,
        run_id: Optional[int] = None,
        fast_mode: Optional[bool] = None,
        request_id: Optional[int] = None,
        owner_epoch: Optional[int] = None,
        batch_set: Optional[str] = None,
        batch_set_id: Optional[str] = None,
        cache_key: Optional[str] = None,
        callback_identity: SimulationCallbackIdentity | None = None,
    ) -> None:
        identity = callback_identity or self._capture_simulation_callback_identity(
            run_id=run_id,
            fast_mode=fast_mode,
            request_id=request_id,
            owner_epoch=owner_epoch,
            batch_set=batch_set,
            batch_set_id=batch_set_id,
            cache_key=cache_key,
        )
        if identity.owner_epoch is None:
            self.on_simulation_error(
                error_msg,
                run_id=identity.run_id,
                fast_mode=identity.fast_mode,
                request_id=identity.request_id,
                batch_set=identity.batch_set,
                batch_set_id=identity.batch_set_id,
                cache_key=identity.cache_key,
                callback_identity=identity,
            )
            return
        self._on_simulation_error(
            error_msg,
            run_id=identity.run_id,
            fast_mode=identity.fast_mode,
            request_id=identity.request_id,
            owner_epoch=identity.owner_epoch,
            batch_set=identity.batch_set,
            batch_set_id=identity.batch_set_id,
            cache_key=identity.cache_key,
            callback_identity=identity,
        )

    def _capture_simulation_callback_identity(
        self,
        *,
        run_id: Optional[int],
        fast_mode: Optional[bool],
        request_id: Optional[int],
        owner_epoch: Optional[int],
        batch_set: Optional[str],
        batch_set_id: Optional[str],
        cache_key: Optional[str],
    ) -> SimulationCallbackIdentity:
        return SimulationCallbackIdentity.capture(
            run_id=run_id,
            fast_mode=fast_mode,
            request_id=request_id,
            owner_epoch=owner_epoch,
            batch_set=batch_set,
            batch_set_id=batch_set_id,
            cache_key=cache_key,
            policy_context=self._batch_context_owner.completion_policy_context(),
            context_snapshot=self._batch_context_owner.current_context_snapshot(),
        )

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
        self._release_runtime_owner_from_worker(worker)
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

    def _effective_preview_owner_epoch_for_callback(
        self,
        *,
        owner_epoch: Optional[int],
        context: Optional[CompletionPolicyContext],
    ) -> Optional[int]:
        if owner_epoch is not None:
            return int(owner_epoch)
        if context is not None and context.preview_owner_epoch is not None:
            return int(context.preview_owner_epoch)
        return None

    def _missing_preview_owner_epoch_for_current_fast_owner(
        self,
        *,
        fast_mode: Optional[bool],
        request_id: Optional[int],
        owner_epoch: Optional[int],
        latest_request_id: int,
    ) -> bool:
        if (not bool(fast_mode)) or request_id is None or owner_epoch is not None:
            return False
        owner_request_id = self._preview_ownership.request_id
        if owner_request_id is None:
            return False
        return (
            int(owner_request_id) == int(request_id)
            and int(request_id) != int(latest_request_id)
        )

    def _prepare_simulation_shutdown_for_close(self) -> bool:
        seen_ids: set[int] = set()
        owned_workers = []
        current_worker = getattr(self, "_simulation_worker", None)
        current_worker_running = self._worker_is_running(current_worker)
        state = self._batch_context_owner.active_batch_state()
        active_fast = bool(state.fast_mode) if state is not None else False
        detached_active_owner = None
        if current_worker_running:
            detached_active_owner = self._detach_contained_simulation_owner(fast_mode=active_fast)
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
            if (
                worker is current_worker
                and (not still_running)
                and detached_active_owner is not None
                and hasattr(detached_active_owner, "close")
            ):
                try:
                    detached_active_owner.close(kill=True)
                except Exception as exc:
                    self._record_nonfatal_exception(
                        "Failed to close detached contained simulation owner during closeEvent cleanup",
                        exc,
                    )
                detached_active_owner = None
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
        still_running = self._cleanup_worker_safely(worker, "simulation worker", retain_if_running=True)
        if not bool(still_running):
            self._release_runtime_owner_from_worker(worker)
        if getattr(self, "_simulation_worker", None) is worker:
            self._simulation_worker = None

    def _release_runtime_owner_from_worker(self, worker) -> None:
        if worker is None or bool(getattr(worker, "_kindred_runtime_owner_released", False)):
            return
        owner = getattr(worker, "_owner", None)
        if owner is None:
            return
        try:
            self._runtime_application.release_owner(
                owner,
                kill=bool(getattr(worker, "_owner_closed", False)),
            )
            setattr(worker, "_kindred_runtime_owner_released", True)
        except Exception as exc:
            self._record_nonfatal_exception("Failed to release active simulation runtime owner", exc)

    def _contained_owner_attr(self, *, fast_mode: bool) -> str:
        return "_preview_simulation_owner" if bool(fast_mode) else "_ordinary_simulation_owner"

    def _contained_owner_mode(self, *, fast_mode: bool) -> str:
        return "preview" if bool(fast_mode) else "ordinary"

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
        return WarmSimulationOwner(owner_plan_payload, **kwargs)

    def _contained_simulation_owner(
        self,
        *,
        fast_mode: bool,
        simulation_plan_payload: Optional[Mapping[str, Any]] = None,
    ):
        mode = self._contained_owner_mode(fast_mode=bool(fast_mode))
        owner_plan_payload = dict(simulation_plan_payload or {})
        owner = self._runtime_application.current_owner(mode=mode)
        if owner is not None:
            return owner
        owner = self._new_contained_simulation_owner(
            fast_mode=bool(fast_mode),
            simulation_plan_payload=owner_plan_payload,
        )
        self._runtime_application.adopt_owner(mode=mode, owner=owner, payload=owner_plan_payload)
        return owner

    def _ready_contained_simulation_owner_for_plan(
        self,
        *,
        fast_mode: bool,
        simulation_plan_payload: Mapping[str, Any],
    ):
        plan_payload = dict(simulation_plan_payload or {})
        if not plan_payload:
            return None
        mode = self._contained_owner_mode(fast_mode=bool(fast_mode))
        return self._runtime_application.ready_owner(mode=mode, payload=plan_payload)

    def _acquire_ready_contained_simulation_owner_for_plan(
        self,
        *,
        fast_mode: bool,
        simulation_plan_payload: Mapping[str, Any],
    ):
        plan_payload = dict(simulation_plan_payload or {})
        if not plan_payload:
            return None
        mode = self._contained_owner_mode(fast_mode=bool(fast_mode))
        return self._runtime_application.acquire_ready_owner(mode=mode, payload=plan_payload)

    def _warm_contained_simulation_owner_for_plan(
        self,
        *,
        fast_mode: bool,
        simulation_plan_payload: Mapping[str, Any],
        wait: bool = True,
    ) -> None:
        plan_payload = dict(simulation_plan_payload or {})
        if not plan_payload:
            return
        mode = self._contained_owner_mode(fast_mode=bool(fast_mode))

        def _owner_factory(payload: Mapping[str, object]):
            owner = self._new_contained_simulation_owner(
                fast_mode=bool(fast_mode),
                simulation_plan_payload=dict(payload),
            )
            return owner

        self._runtime_application.ensure_ready(
            mode=mode,
            payload=plan_payload,
            owner_factory=_owner_factory,
            wait=bool(wait),
        )

    def _interactive_runtime_rows(self) -> list[int]:
        try:
            rows = list(self.ui.batch.batch_rows_for_scope("selected"))
        except Exception:
            rows = []
        if not rows:
            try:
                row_count = int(self.ui.batch.batch_store_row_count())
            except Exception:
                row_count = 0
            if row_count > 0:
                rows = [0]
        return [int(row) for row in rows]

    def _interactive_runtime_plan_payloads_for_mode(self, *, fast_mode: bool) -> list[dict[str, Any]]:
        return self._build_runtime_readiness_plan_payloads(
            fast_mode=bool(fast_mode),
            batch_rows=self._interactive_runtime_rows(),
        )

    def _interactive_simulation_runtime_snapshot(self, *, fast_mode: bool) -> RuntimeReadinessSnapshot:
        mode = self._contained_owner_mode(fast_mode=bool(fast_mode))
        payloads = self._interactive_runtime_plan_payloads_for_mode(fast_mode=bool(fast_mode))
        if not payloads:
            return _runtime_readiness_snapshot(
                mode=mode,
                status="not_applicable",
                ready=False,
                required=False,
                controls_ready=True,
                polling=False,
                message="No runnable simulation runtime is required for the current state.",
            )
        app_snapshot = self._runtime_application.snapshot(mode=mode)
        all_ready = True
        for payload in payloads:
            if self._runtime_application.ready_owner(mode=mode, payload=payload) is None:
                all_ready = False
                break
        if all_ready:
            return _runtime_readiness_snapshot(
                mode=mode,
                status="ready",
                ready=True,
                generation=int(app_snapshot.generation),
                required=True,
                controls_ready=True,
                polling=False,
            )
        status = str(app_snapshot.status or "missing")
        failure = app_snapshot.failure
        if status == "failed":
            message = f"{'Preview' if bool(fast_mode) else 'Simulation'} runtime failed to start."
            if failure:
                message = f"{message} {failure}"
            return _runtime_readiness_snapshot(
                mode=mode,
                status="failed",
                ready=False,
                generation=int(app_snapshot.generation),
                failure=failure,
                message=message,
                required=True,
                controls_ready=False,
                polling=False,
            )
        return _runtime_readiness_snapshot(
            mode=mode,
            status=status if status else "warming",
            ready=False,
            generation=int(app_snapshot.generation),
            failure=failure,
            message=f"Preparing {'preview' if bool(fast_mode) else 'simulation'} runtime...",
            required=True,
            controls_ready=False,
            polling=True,
        )

    def _build_runtime_readiness_plan_payloads(
        self,
        *,
        fast_mode: bool,
        batch_rows: Sequence[int],
    ) -> list[dict[str, Any]]:
        try:
            rows = self._run_rows_or_abort(
                batch_rows=batch_rows,
                fast_mode=bool(fast_mode),
                runtime_readiness_only=True,
            )
            if rows is None:
                return []
            mechanism_context = self._run_mechanism_context_or_abort(
                fast_mode=bool(fast_mode),
                request_id=0,
                batch_rows=rows,
                runtime_readiness_only=True,
            )
            if mechanism_context is None:
                return []
            solver_context = self._run_solver_context_or_abort(
                fast_mode=bool(fast_mode),
                runtime_readiness_only=True,
                mechanism_context=mechanism_context,
            )
            if solver_context is None:
                return []
            dispatch_context = self._build_run_dispatch_context_or_abort(
                fast_mode=bool(fast_mode),
                runtime_readiness_only=True,
                mechanism_context=mechanism_context,
                solver_context=solver_context,
            )
            if dispatch_context is None:
                return []
        except Exception:
            return []

        from kindred.core.simulation_containment import build_contained_simulation_plan_payload

        return [
            build_contained_simulation_plan_payload(dispatch_context.simulation_plan_by_set_id[str(set_id)])
            for set_id in mechanism_context.queue_ids
            if str(set_id) in dispatch_context.simulation_plan_by_set_id
        ]

    def _ensure_interactive_simulation_runtime_available_for_mode(
        self,
        *,
        fast_mode: bool,
        wait: bool = False,
    ) -> None:
        payloads = self._interactive_runtime_plan_payloads_for_mode(fast_mode=bool(fast_mode))
        if not payloads:
            return
        mode = self._contained_owner_mode(fast_mode=bool(fast_mode))

        def _owner_factory(payload: Mapping[str, object]):
            return self._new_contained_simulation_owner(
                fast_mode=bool(fast_mode),
                simulation_plan_payload=dict(payload),
            )

        self._runtime_application.ensure_ready_many(
            mode=mode,
            payloads=[dict(payload) for payload in payloads],
            owner_factory=_owner_factory,
            wait=bool(wait),
        )

    def _detach_contained_simulation_owner(self, *, fast_mode: bool):
        return self._runtime_application.detach_owner(
            mode=self._contained_owner_mode(fast_mode=bool(fast_mode))
        )

    def _close_contained_simulation_owner(
        self,
        *,
        fast_mode: Optional[bool] = None,
        kill: bool = False,
    ) -> None:
        modes = (False, True) if fast_mode is None else (bool(fast_mode),)
        for mode in modes:
            try:
                self._runtime_application.close(
                    mode=self._contained_owner_mode(fast_mode=mode),
                    kill=bool(kill),
                )
            except Exception as exc:
                self._record_nonfatal_exception(
                    f"Failed to close {'preview' if mode else 'ordinary'} contained simulation owner",
                    exc,
                )

    def _connect_simulation_worker_application_signals(
        self,
        worker,
        *,
        run_id: int,
        fast_mode: bool,
        request_id: int,
        owner_epoch: Optional[int] = None,
        set_name: str,
        set_id: str,
        cache_key: str,
    ) -> None:
        if worker is None:
            return
        self._disconnect_simulation_worker_application_signals(worker)
        connected_handlers = list(getattr(worker, _WORKER_APPLICATION_SIGNAL_HANDLERS_ATTR, ()) or ())
        progress_handler = self.on_simulation_progress
        callback_identity = self._capture_simulation_callback_identity(
            run_id=run_id,
            fast_mode=fast_mode,
            request_id=request_id,
            owner_epoch=owner_epoch,
            batch_set=set_name,
            batch_set_id=set_id,
            cache_key=cache_key,
        )

        def result_handler(
            payload,
            _identity=callback_identity,
        ):
            return self._dispatch_simulation_complete(
                payload,
                run_id=_identity.run_id,
                fast_mode=_identity.fast_mode,
                request_id=_identity.request_id,
                owner_epoch=_identity.owner_epoch,
                batch_set=_identity.batch_set,
                batch_set_id=_identity.batch_set_id,
                cache_key=_identity.cache_key,
                callback_identity=_identity,
            )

        def error_handler(
            msg,
            _identity=callback_identity,
        ):
            return self._dispatch_simulation_error(
                msg,
                run_id=_identity.run_id,
                fast_mode=_identity.fast_mode,
                request_id=_identity.request_id,
                owner_epoch=_identity.owner_epoch,
                batch_set=_identity.batch_set,
                batch_set_id=_identity.batch_set_id,
                cache_key=_identity.cache_key,
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

    def _selected_run_uses_parallel_batch_runtime(self) -> bool:
        rows = self._interactive_runtime_rows()
        if len(rows) <= 1:
            return False
        return bool(self._effective_batch_worker_count(len(rows)) > 1)

    def _selected_run_runtime_snapshot(self) -> RuntimeReadinessSnapshot:
        if self._selected_run_uses_parallel_batch_runtime():
            return self._parallel_batch_runtime_snapshot()
        return self._interactive_simulation_runtime_snapshot(fast_mode=False)

    def _slider_preview_uses_parallel_batch_runtime(self, rows: Optional[Sequence[int]] = None) -> bool:
        if rows is None:
            rows = self._interactive_runtime_rows()
        row_count = len(list(rows or ()))
        return bool(row_count > 1 and self._effective_batch_worker_count(row_count) > 1)

    def _slider_preview_runtime_snapshot(self, rows: Optional[Sequence[int]] = None) -> RuntimeReadinessSnapshot:
        if self._slider_preview_uses_parallel_batch_runtime(rows):
            return self._parallel_batch_runtime_snapshot(rows=rows)
        return self._interactive_simulation_runtime_snapshot(fast_mode=True)

    def _parallel_batch_runtime_snapshot(self, rows: Optional[Sequence[int]] = None) -> RuntimeReadinessSnapshot:
        if rows is None:
            rows = self._interactive_runtime_rows()
        rows = list(rows or ())
        if len(rows) <= 1 or self._effective_batch_worker_count(len(rows)) <= 1:
            return _runtime_readiness_snapshot(
                mode="batch",
                status="not_applicable",
                ready=False,
                required=False,
                controls_ready=True,
                polling=False,
                message="Parallel batch runtime is not required for the current selection.",
            )
        required_lanes = max(1, int(self._effective_batch_worker_count(len(rows))))
        try:
            ready = bool(self._batch_parallel.has_ready_lane_pool(max_lanes=required_lanes))
            snapshot = self._batch_parallel.runtime_snapshot()
        except Exception as exc:
            return _runtime_readiness_snapshot(
                mode="batch",
                status="failed",
                ready=False,
                failure=f"{type(exc).__name__}: {exc}",
                message=f"Batch runtime readiness check failed: {exc}",
                required=True,
                controls_ready=False,
                polling=False,
            )
        if ready:
            self._parallel_batch_runtime_readiness_owner.mark_ready()
            return _runtime_readiness_snapshot(
                mode="batch",
                status="ready",
                ready=True,
                generation=int(getattr(snapshot, "current_generation", 0) or 0),
                required=True,
                controls_ready=True,
                polling=False,
            )
        self._parallel_batch_runtime_readiness_owner.mark_not_ready()
        failure = getattr(snapshot, "warm_failure", None)
        if failure:
            return _runtime_readiness_snapshot(
                mode="batch",
                status="failed",
                ready=False,
                generation=int(getattr(snapshot, "current_generation", 0) or 0),
                failure=str(failure),
                message=f"Batch runtime failed to prepare. {failure}",
                required=True,
                controls_ready=False,
                polling=False,
            )
        if bool(getattr(snapshot, "pool_stale", False)):
            status = "stale"
            message = "Rebuilding batch runtime..."
        elif bool(getattr(snapshot, "has_lane_pool", False)):
            status = "warming"
            message = "Preparing batch runtime..."
        else:
            status = "missing"
            message = "Preparing batch runtime..."
        return _runtime_readiness_snapshot(
            mode="batch",
            status=status,
            ready=False,
            generation=int(getattr(snapshot, "current_generation", 0) or 0),
            message=message,
            required=True,
            controls_ready=False,
            polling=True,
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
        self._parallel_batch_runtime_readiness_owner.mark_not_ready()
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

    def _parallel_batch_pool_settings_changed(self) -> None:
        if self._has_active_parallel_batch_work():
            self._batch_parallel.mark_pool_stale()
            self._parallel_batch_runtime_readiness_owner.mark_not_ready()
            return
        self._shutdown_batch_lane_pool(force_terminate=False)

    def _ensure_parallel_batch_pool_eagerly_created(self, *, wait: bool = False) -> None:
        self._parallel_batch_runtime_readiness_owner.ensure(wait=bool(wait))

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

    def _parallel_batch_runtime_ready(self) -> bool:
        return self._parallel_batch_runtime_readiness_owner.ready()

    def _cleanup_parallel_batch_lane_pool_after_run(
        self,
        *,
        keep_lane_pool_alive: bool,
        clear_pending_plot_updates: bool = False,
        stale_fast_handoff_after_display: bool = False,
    ) -> None:
        if bool(keep_lane_pool_alive) and (not self._batch_parallel.is_pool_stale):
            if stale_fast_handoff_after_display:
                cancelled, running = self._batch_parallel.soft_supersede()
                timer = getattr(self, "_batch_completion_poll_timer", None)
                if running > 0 and timer is not None:
                    timer.start()
                if bool(getattr(self, "_debug_batch_parallel", False)):
                    logger.info(
                        "BATCH_PAR soft handoff after stale preview display cancelled=%s running=%s",
                        int(cancelled),
                        int(running),
                    )
            else:
                self._batch_parallel.finish_after_run(
                    keep_lane_pool_alive=True,
                    record_nonfatal_exception=self._record_nonfatal_exception,
                )
            self._stop_batch_completion_poll_timer_if_idle()
            if bool(clear_pending_plot_updates):
                self._clear_pending_slider_plot_updates()
            if bool(getattr(self, "_debug_batch_parallel", False)):
                logger.info("BATCH_PAR keeping lane pool alive after slider batch completion")
            return
        self._shutdown_batch_lane_pool(force_terminate=False)

    def _supersede_parallel_batch_run_soft(self) -> tuple[int, int]:
        """
        Supersede the active parallel run without destroying the process pool.

        Used by slider-triggered restarts to preserve worker processes and avoid
        pool recreation on every minor parameter update.
        """
        state = self._batch_context_owner.active_batch_state()
        if state is not None and state.active and state.parallel:
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
        clear_preview = getattr(self._batch_cache, "clear_active_preview_selection_state", None)
        if callable(clear_preview):
            clear_preview()
        state = self._batch_context_owner.active_batch_state()
        if state is not None and state.active and state.parallel and state.fast_mode:
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

    def _runtime_input_context_stale_for_set(
        self,
        context: Mapping[str, Any],
        *,
        batch_set_id: Optional[str],
    ) -> bool:
        set_id = str(batch_set_id or "").strip()
        return self._batch_context_owner.runtime_input_stale_for_set(
            context,
            batch_set_id=set_id,
            current_global_epoch=int(getattr(self, "_authoritative_runtime_input_global_epoch", 0) or 0),
            current_set_epoch=self._runtime_input_set_epoch(set_id),
            current_epoch=int(getattr(self, "_authoritative_runtime_input_epoch", 0) or 0),
        )

    def _active_batch_context_runtime_input_stale_for_set(
        self,
        *,
        batch_set_id: Optional[str],
        context: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if isinstance(context, Mapping):
            return self._runtime_input_context_stale_for_set(
                context,
                batch_set_id=batch_set_id,
            )
        set_id = str(batch_set_id or "").strip()
        return self._batch_context_owner.active_runtime_input_stale_for_set(
            batch_set_id=set_id,
            current_global_epoch=int(getattr(self, "_authoritative_runtime_input_global_epoch", 0) or 0),
            current_set_epoch=self._runtime_input_set_epoch(set_id),
            current_epoch=int(getattr(self, "_authoritative_runtime_input_epoch", 0) or 0),
        )

    def _mark_stale_runtime_input_callback_consumed(
        self,
        *,
        batch_set_id: Optional[str],
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        set_id = str(batch_set_id or "").strip()
        if not set_id:
            return
        if isinstance(context, Mapping) and not self._batch_context_owner.context_matches_current_run_identity(context):
            return
        transition = self._batch_context_owner.record_parallel_stale_callback_consumed_if_active(set_id=set_id)
        if transition is None:
            return
        if transition.batch_done:
            context = transition.context
            self._finalize_batch_queue_done_without_result(context)

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
        allow_fallback: bool = True,
    ) -> None:
        preview_ownership = self._preview_ownership
        request_accepted = (
            self._preview_request_can_display(request_id)
            if bool(slider_triggered)
            else (request_id is None or int(request_id) == int(getattr(self, "_latest_sim_request_id", 0)))
        )
        self._plot_coalescer.queue(
            set_id=set_id,
            cache_key=cache_key,
            request_id=request_id,
            request_accepted=bool(request_accepted),
            run_id=run_id,
            accepted_owner_request_id=(
                preview_ownership.request_id if bool(slider_triggered) and bool(request_accepted) else None
            ),
            accepted_owner_epoch=(
                int(preview_ownership.epoch) if bool(slider_triggered) and bool(request_accepted) else None
            ),
            slider_triggered=slider_triggered,
            valid_set_ids=valid_set_ids,
            allow_fallback=allow_fallback,
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
        pending_owner_request_id = pending.accepted_owner_request_id
        pending_owner_epoch = pending.accepted_owner_epoch
        pending_valid_set_ids = pending.valid_set_ids
        pending_allow_fallback = bool(pending.allow_fallback)

        cache_key = str(cache_key or pending_cache_key or "")
        request_id = pending_request_id if request_id is None else request_id
        run_id = pending_run_id if run_id is None else run_id
        if not cache_key:
            return False
        request_accepted = (
            self._queued_preview_update_still_matches_current_owner(
                request_id=request_id,
                accepted_owner_request_id=pending_owner_request_id,
                accepted_owner_epoch=pending_owner_epoch,
            )
            if pending_cache_kind == "preview"
            else (request_id is None or int(request_id) == int(getattr(self, "_latest_sim_request_id", 0)))
        )
        if request_id is not None and not bool(request_accepted):
            return False
        if run_id is not None and int(run_id) != int(getattr(self, "_active_run_id", 0)):
            return False

        cache_store = self._batch_cache.store_for_kind(pending_cache_kind)

        shown_sets = list(self.ui.batch.shown_batch_set_ids())
        selected_sets = [str(set_id) for set_id in shown_sets if str(set_id)]
        if not selected_sets:
            selected_sets = list(self.ui.batch.batch_set_ids_for_scope("selected"))
        if not selected_sets:
            selected_sets = sorted(pending_set_ids)
        else:
            selected_sets = [str(set_id) for set_id in selected_sets if str(set_id)]
        if force and not selected_sets:
            cached_ids = {
                set_id
                for set_id, _entry in self._batch_cache.entries_for_cache_key(
                    cache_key=str(cache_key),
                    is_preview=(pending_cache_kind == "preview"),
                )
                if str(set_id)
            }
            if cached_ids:
                selected_sets = sorted(cached_ids)
        if not selected_sets:
            return False

        prefer = None
        current_row = self.ui.batch.batch_current_row()
        if current_row is not None:
            prefer = self.ui.batch.batch_set_id_for_row(int(current_row))

        displayed = self.ui.batch.display_cached_batch_selection(
            cache_key=str(cache_key),
            selected_sets=selected_sets,
            prefer_set=prefer,
            cache_store=cache_store,
            valid_set_ids=pending_valid_set_ids,
            allow_fallback=pending_allow_fallback,
        )
        if bool(getattr(self, "_debug_batch_parallel", False)):
            logger.info(
                "BATCH_PAR plot flush run_id=%s request_id=%s changed_sets=%s forced=%s displayed=%s ts=%.6f",
                int(run_id or 0),
                int(request_id or 0),
                int(len(pending_set_ids)),
                bool(force),
                bool(displayed),
                float(perf_counter()),
            )
        return bool(displayed)

    # ------------------------------------------------------------------
    # Batch lane outcome polling/consumption
    # ------------------------------------------------------------------
    def _clear_stale_parallel_batch_requests(self) -> None:
        self._batch_parallel.clear_stale_requests()

    def _reset_parallel_batch_run_and_shutdown_lane_pool(self) -> None:
        state = self._batch_context_owner.active_batch_state()
        if state is not None and state.active and state.parallel:
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
        self._dispatch_simulation_error(
            error_msg,
            run_id=int(dispatch_context.run_id),
            fast_mode=bool(dispatch_context.fast_mode),
            request_id=int(dispatch_context.request_id),
            owner_epoch=dispatch_context.owner_epoch,
            batch_set="",
            batch_set_id="",
            cache_key=str(dispatch_context.cache_key),
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
                        preserve_active_cache=True,
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
            keep_lane_pool_alive = bool(cleanup_state.parallel and cleanup_state.keep_lane_pool_alive)
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
        if state is None or not state.active or state.parallel or state.fast_mode:
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

    def _finalize_scoped_batch_success_subset(self, ctx: Mapping[str, Any]) -> tuple[str, ...]:
        if not isinstance(ctx, Mapping):
            return ()
        policy_context = self._batch_context_owner.completion_policy_context(ctx)
        if policy_context is None:
            return ()
        eligible_reset_set_ids = tuple(policy_context.pending_workspace_reset_set_ids)
        ctx = self._finalize_explicit_batch_dirty_reset(
            ctx,
            species_names=self._current_mechanism_species_for_batch_sync(),
        )
        flush_context = self._batch_context_owner.completion_flush_context(ctx)
        self.flush_slider_plot_updates(
            force=True,
            cache_key=str(flush_context.cache_key),
            request_id=flush_context.request_id,
            run_id=flush_context.run_id,
        )
        return tuple(eligible_reset_set_ids)

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
        run_id: int,
        request_id: int,
        fast_mode: bool,
        cache_key: str,
        source: str,
        completed_ts: Optional[float] = None,
        completion_record: Optional[BatchCompletionRecord] = None,
    ) -> bool:
        return self._parallel_batch_outcome_owner.consume_outcome(
            set_id=set_id,
            outcome=outcome,
            run_id=int(run_id),
            request_id=int(request_id),
            fast_mode=bool(fast_mode),
            cache_key=str(cache_key),
            source=str(source),
            completed_ts=completed_ts,
            completion_record=completion_record,
            debug_batch_parallel=bool(getattr(self, "_debug_batch_parallel", False)),
        )

    def _stop_batch_completion_poll_timer_if_idle(self) -> None:
        state = self._batch_context_owner.active_batch_state()
        active_parallel = bool(state is not None and state.active and state.parallel)
        if active_parallel or self._batch_parallel.has_active_requests():
            return
        timer = getattr(self, "_batch_completion_poll_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()

    def _poll_parallel_batch_completions(self) -> None:
        runtime_snapshot = self._batch_parallel.runtime_snapshot()
        active_parallel = bool(runtime_snapshot.active)
        if not active_parallel and not self._batch_parallel.has_active_requests():
            if self._batch_parallel.is_pool_stale:
                self._shutdown_batch_lane_pool(force_terminate=False)
            self._stop_batch_completion_poll_timer_if_idle()
            if self._has_deferred_preview_replay_intent():
                self._schedule_deferred_preview_replay_handoff_once()
            return

        run_id = int(runtime_snapshot.run_id) if active_parallel else 0
        request_id = int(runtime_snapshot.request_id) if active_parallel else 0
        fast_mode = bool(runtime_snapshot.fast_mode) if active_parallel else False
        cache_key = str(runtime_snapshot.cache_key or "") if active_parallel else ""

        try:
            for polled in self._batch_parallel.poll_completed_records():
                sid = str(polled.set_id or "")
                completion_record = polled.record
                if not self._consume_parallel_batch_outcome(
                    set_id=sid,
                    outcome=completion_record.outcome,
                    run_id=run_id,
                    request_id=request_id,
                    fast_mode=fast_mode,
                    cache_key=cache_key,
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
                self._on_simulation_error(
                    f"Simulation failed:\n\n{exc}",
                    run_id=run_id,
                    fast_mode=fast_mode,
                    request_id=request_id,
                    batch_set="",
                    batch_set_id="",
                    cache_key=cache_key,
                )
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
        fallback_rows: Sequence[int],
        *,
        target_set_ids: Optional[Sequence[str]] = None,
    ) -> list[int]:
        _ = fallback_rows
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

    def _apply_parameter_override_fallback_to_dsl(self, dsl_text: str, *, set_id: Optional[str]) -> str:
        mechanism_text = str(dsl_text or "")
        overrides = self.ui.mechanism.slider_overrides(set_id=set_id)
        if not overrides:
            return mechanism_text
        try:
            return self.ui.mechanism.apply_parameter_overrides_to_dsl(mechanism_text, overrides)
        except Exception as exc:
            self._record_nonfatal_exception(
                f"Failed to apply parameter override fallback to slider DSL for set_id={str(set_id or '')}",
                exc,
            )
            return mechanism_text

    def _execution_identity_flags(self, *, fast_mode: bool) -> tuple[str, ...]:
        return ("fast_mode",) if bool(fast_mode) else ()

    def _intervention_schedule_fingerprint_for_set(self, *, set_id: str, fast_mode: bool) -> str:
        try:
            mechanism_text = self._request_mechanism_text_for_set(
                set_id=str(set_id),
                has_slider_overrides=bool(fast_mode) and self.ui.mechanism.has_slider_overrides(),
            )
        except Exception as exc:
            self._record_nonfatal_exception(
                f"Failed to resolve intervention schedule identity text for set_id={str(set_id or '')}",
                exc,
            )
            try:
                mechanism_text = self.ui.mechanism.get_mechanism_text()
            except Exception:
                mechanism_text = ""
        try:
            from kindred.core.intervention_schedule import intervention_schedule_fingerprint_from_dsl_text

            return str(intervention_schedule_fingerprint_from_dsl_text(str(mechanism_text or "")) or "")
        except Exception:
            return hashlib.sha256(str(mechanism_text or "").encode("utf-8", "surrogatepass")).hexdigest()

    def _simulation_identity_for_set(
        self,
        *,
        set_id: str,
        solver_config: Mapping[str, Any],
        t_end: float,
        canonical_initials_fingerprint: str = "",
        preview_batch_cache_token: str = "",
        fast_mode: bool,
    ) -> SimulationIdentity:
        param_fingerprint = ""
        preview_token = ""
        if bool(fast_mode):
            param_fingerprint = self.ui.mechanism.simulation_param_fingerprint(set_id=str(set_id))
            preview_token = str(preview_batch_cache_token or "")
        return SimulationIdentity.build(
            schema_id=self.ui.mechanism.simulation_schema_id(),
            param_fingerprint=param_fingerprint,
            canonical_initials_fingerprint=str(canonical_initials_fingerprint or ""),
            solver_config=solver_config,
            t_end=float(t_end),
            intervention_schedule_fingerprint=self._intervention_schedule_fingerprint_for_set(
                set_id=str(set_id),
                fast_mode=bool(fast_mode),
            ),
            preview_batch_cache_token=preview_token,
            execution_flags=self._execution_identity_flags(fast_mode=bool(fast_mode)),
        )

    def _include_mechanism_in_result_payload(
        self,
        *,
        fast_mode: bool,
        batch_set_id: Optional[str],
        context: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if bool(fast_mode):
            return False
        set_id = str(batch_set_id or "").strip()
        if not set_id:
            return True
        primary_set = self._batch_context_owner.primary_set_id(
            context if isinstance(context, Mapping) else None
        )
        if primary_set:
            return set_id == primary_set
        return True

    def _resolved_initials_for_batch_row(
        self,
        *,
        row: int,
        set_name: str,
        pending_init_seed: Optional[Mapping[str, object]],
        pending_init_applied: bool,
        include_preview_initials: bool,
    ) -> Dict[str, float]:
        return self._batch_dispatch_materialization_owner.materialize_initials(
            row=int(row),
            set_name=str(set_name),
            fast_mode=bool(include_preview_initials),
            pending_init_seed=pending_init_seed,
            pending_init_applied=bool(pending_init_applied),
        )

    def _invalidate_preserved_pending_init_results_after_failed_run(
        self,
        *,
        pending_init_applied: bool = False,
        ctx: Optional[Mapping[str, Any]] = None,
    ) -> None:
        policy_context = self._batch_context_owner.completion_policy_context(ctx)
        if bool(pending_init_applied) and policy_context is not None and (not policy_context.pending_init_applied):
            policy_context = policy_context.evolve(pending_init_applied=True)
        decision = self._completion_policy.resolve_pending_init_failure(policy_context)
        if not decision.should_invalidate_preserved_results:
            return
        try:
            self.ui.mechanism_helpers.invalidate_pending_init_preserved_results_after_failed_run()
        except Exception as exc:
            self._record_nonfatal_exception(
                "Failed to invalidate preserved pending-init results after explicit run failure",
                exc,
            )
        self._apply_completion_policy_state_patch(
            decision.state_patch,
            base_context=(ctx if isinstance(ctx, Mapping) else None),
        )

    def _requeue_preserved_pending_slider_replay_after_preflight_abort(self) -> None:
        directive = self._completion_policy.resolve_preflight_abort_pending_replay(
            pending_replay=self._completion_policy_pending_replay_state(),
            explicit_run=True,
        )
        if directive is not None:
            self._apply_completion_policy_state_patch(PolicyStatePatch(pending_replay=directive))

    def _request_mechanism_text_for_set(
        self,
        *,
        set_id: str,
        has_slider_overrides: bool,
    ) -> str:
        set_reactions_text = self.ui.mechanism.mechanism_reactions_text_raw()
        if has_slider_overrides:
            set_reactions_text = self.ui.mechanism.apply_overrides_to_text(
                set_reactions_text,
                set_id=str(set_id),
            )
        set_reactions_text = strip_reaction_dsl_initial_concentrations(set_reactions_text)

        set_state_network_dsl = self.ui.mechanism.mechanism_state_network_dsl_raw()
        if has_slider_overrides:
            set_state_network_dsl = self.ui.mechanism.apply_overrides_to_state_network_dsl(
                set_state_network_dsl,
                set_id=str(set_id),
            )

        request_mechanism_text = set_reactions_text
        if set_state_network_dsl.strip():
            request_mechanism_text += "\n\n# State Network\n" + set_state_network_dsl
        if has_slider_overrides:
            request_mechanism_text = self._apply_parameter_override_fallback_to_dsl(
                request_mechanism_text,
                set_id=str(set_id),
            )
        return str(request_mechanism_text)

    def _slider_runtime_parameter_names(self, *, set_id: Optional[str]) -> list[str]:
        names: set[str] = set()
        try:
            names.update(str(name) for name in self.ui.mechanism.slider_overrides(set_id=set_id).keys())
        except Exception:
            pass
        try:
            names.update(str(name) for name in self.ui.mechanism.variable_slider_values().keys())
        except Exception:
            pass
        try:
            names.update(str(name) for name in self.ui.mechanism.variable_metadata().keys())
        except Exception:
            pass
        if not names:
            names.update(self._slider_parameter_names_from_current_mechanism())
        return sorted(name for name in names if name)

    def _slider_parameter_names_from_current_mechanism(self) -> list[str]:
        try:
            from kindred.core.batch_initial_conditions import strip_named_reaction_dsl_initial_concentration_sets
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism
            from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism
            from kindred.core.units import UnitsModel
            from kindred.gui.parameter_enumeration import enumerate_step_parameters_for_gui

            reactions_text = strip_named_reaction_dsl_initial_concentration_sets(
                self.ui.mechanism.mechanism_reactions_text_raw()
            )
            state_network_dsl = self.ui.mechanism.mechanism_state_network_dsl_raw()
            full_dsl = str(reactions_text or "")
            if str(state_network_dsl or "").strip():
                full_dsl += "\n\n# State Network\n" + str(state_network_dsl).strip("\n")
            if not full_dsl.strip():
                return []
            try:
                temperature_K = float(self.ui.solver.temperature_spinbox_value())
            except Exception:
                temperature_K = 298.15
            units = UnitsModel(temperature_K=float(temperature_K))
            wegscheider_enabled = bool(self.ui.solver.wegscheider_cyclicity_enabled())

            def _build_structure_snapshot(full_text: str) -> object:
                mechanism_obj = parse_dsl_to_mechanism(full_text, initials={}, units=units)
                if isinstance(getattr(mechanism_obj, "metadata", None), dict):
                    mechanism_obj.metadata["wegscheider_cyclicity_enabled"] = wegscheider_enabled
                apply_parameter_algebra_to_mechanism(
                    full_text,
                    mechanism=mechanism_obj,
                    require_mutable=False,
                )
                return mechanism_obj

            authoritative_structure_snapshot = getattr(
                self.ui.mechanism_helpers,
                "authoritative_structure_snapshot",
                None,
            )
            if callable(authoritative_structure_snapshot):
                structure_snapshot = authoritative_structure_snapshot(
                    reactions_text=str(reactions_text or ""),
                    state_network_text=state_network_dsl,
                    units_identity=(
                        "temperature_K",
                        f"{float(temperature_K):.17g}",
                        "wegscheider",
                        str(wegscheider_enabled),
                    ),
                    builder=_build_structure_snapshot,
                )
                mechanism = structure_snapshot.mechanism
            else:
                mechanism = _build_structure_snapshot(full_dsl)
            variables, _metadata = enumerate_step_parameters_for_gui(mechanism)
            names = {str(name) for name in dict(variables or {}).keys() if str(name)}
            scalar_params = (getattr(mechanism, "metadata", {}) or {}).get("scalar_params") or {}
            if isinstance(scalar_params, Mapping):
                names.update(str(name) for name in scalar_params.keys() if str(name))
            return sorted(names)
        except Exception:
            return []

    def _slider_execution_parameter_values(self, *, set_id: Optional[str]) -> Dict[str, float]:
        values: Dict[str, float] = {}
        try:
            values.update(
                {
                    str(name): float(value)
                    for name, value in self.ui.mechanism.variable_slider_values().items()
                }
            )
        except Exception:
            pass
        try:
            values.update(
                {
                    str(name): float(value)
                    for name, value in self.ui.mechanism.slider_overrides(set_id=set_id).items()
                }
            )
        except Exception:
            pass
        return {name: value for name, value in values.items() if str(name)}

    def _preview_contained_owner_identity(
        self,
        *,
        owner_mechanism_text: str,
        solver_config: Mapping[str, Any],
        t_end: float,
        set_id: str,
        parameter_names: Sequence[str],
        simulation_identity: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        return contained_simulation_owner_identity(
            execution_mode="preview",
            owner_mechanism_text=str(owner_mechanism_text or ""),
            solver_config=solver_config,
            t_end=float(t_end),
            set_id=str(set_id or ""),
            parameter_names=parameter_names,
            simulation_identity=simulation_identity,
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
        )

    def _run_simulation_from_slider(self):
        self._slider_preview_launch_owner.run_from_slider()

    def _run_simulation(self):
        if not self.ui.mechanism.auto_lock_for_run():
            self.ui.run_ui.set_status_text("Cannot run: mechanism has errors. Fix and try again.")
            return
        if not self.ui.mechanism.is_mechanism_ready_for_run():
            self.ui.run_ui.set_status_text("Cannot run: mechanism has errors. Fix and try again.")
            return
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
            return

        rows_to_run = self.ui.batch.batch_rows_for_scope("selected")
        if not rows_to_run:
            self.ui.dialogs.message_box_warning("No Sets", "Add at least one set before running.")
            return

        try:
            self.ui.solver.parse_sim_time_seconds()
        except ValueError as exc:
            self.ui.dialogs.message_box_warning("Invalid t_end", f"Fix t_end before running:\n\n{exc}")
            return

        runtime_snapshot = self._selected_run_runtime_snapshot()
        if bool(runtime_snapshot.required) and not bool(runtime_snapshot.ready):
            self._queue_run_after_runtime_ready(
                rows_to_run=rows_to_run,
                runtime_snapshot=runtime_snapshot,
            )
            self.ui.run_ui.set_runtime_backed_run_controls_ready(False)
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self.ui.run_ui.set_status_text(
                str(runtime_snapshot.message or "Preparing simulation runtime...")
            )
            return

        self._clear_pending_run_after_runtime_ready()
        reset_set_ids: list[str] = []
        for row in rows_to_run:
            try:
                set_id = self.ui.batch.batch_set_id_for_row(int(row))
            except Exception:
                set_id = None
            set_id_s = str(set_id or "").strip()
            if set_id_s:
                reset_set_ids.append(set_id_s)

        self._flush_pending_slider_updates_for_run(reset_set_ids=reset_set_ids)
        self._clear_preview_ownership()
        request_id = self._next_sim_request_id()

        self.ui.run_ui.set_run_button_enabled(False)
        self.ui.run_ui.set_stop_button_enabled(True)
        self._simulation_running = True

        logger.info("Starting simulation")
        self.ui.run_ui.set_status_text("Running simulation...")
        self.ui.run_ui.set_sim_progress_value(0)

        self.run_simulation_internal(
            fast_mode=False,
            request_id=int(request_id),
            batch_rows=rows_to_run,
            reuse_parallel_lane_pool=bool(len(rows_to_run) > 1),
        )

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
        if self._worker_is_running(worker):
            self._detach_contained_simulation_owner(fast_mode=active_fast)
        else:
            self._close_contained_simulation_owner(fast_mode=active_fast, kill=True)
        if worker is not None:
            self._release_current_simulation_worker()
        self._simulation_running = False
        self._slider_simulation_active = False
        self.ui.run_ui.set_run_button_enabled(True)
        self.ui.run_ui.set_stop_button_enabled(False)

    def _start_parallel_batch_simulations(self) -> None:
        payload = self._batch_context_owner.parallel_start_payload()
        if payload is None:
            return

        rows = list(payload.rows)
        queue_ids = list(payload.queue_ids)
        queue_names = list(payload.queue_names)
        run_id = int(payload.run_id)
        request_id = int(payload.request_id)
        fast_mode = bool(payload.fast_mode)
        max_workers = max(1, int(payload.effective_workers))

        prior_lane_pool_token = self._batch_parallel.lane_pool_token()
        availability = self._parallel_batch_runtime_readiness_owner.run_start_availability(
            required_lanes=int(max_workers)
        )
        lane_pool = availability.lane_pool
        if not availability.ready:
            if availability.error is not None:
                logger.warning("Parallel batch lane pool unavailable: %s", availability.error)
                self._handle_parallel_batch_runtime_check_failed(availability.error)
            else:
                self._handle_parallel_batch_runtime_waiting(
                    rows=rows,
                    queue_ids=queue_ids,
                    runtime_snapshot=availability.snapshot,
                )
            return
        waiting_state = self._batch_context_owner.active_batch_state()
        if waiting_state is not None and waiting_state.runtime_waiting:
            ctx = self._batch_context_owner.clear_runtime_waiting()
        else:
            ctx = None
        payload = self._batch_context_owner.parallel_start_payload(ctx)
        if payload is None:
            return
        if bool(getattr(self, "_debug_batch_parallel", False)):
            lane_pool_token = None if lane_pool is None else int(id(lane_pool))
            if prior_lane_pool_token is None:
                action = "created"
            elif prior_lane_pool_token == lane_pool_token:
                action = "reused"
            else:
                action = "resized"
            logger.info("BATCH_PAR lane pool %s workers=%s run_id=%s", action, int(max_workers), int(run_id))

        timer = getattr(self, "_batch_completion_poll_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        self._clear_pending_slider_plot_updates()
        self._batch_parallel.begin_run(
            run_id=int(run_id),
            request_id=int(request_id),
            fast_mode=fast_mode,
            queue_ids=[str(item) for item in queue_ids],
            queue_names=[str(item) for item in queue_names],
            keep_lane_pool_alive=bool(payload.keep_lane_pool_alive),
            preview_owner_epoch=payload.preview_owner_epoch,
            active_timeout_s=float(payload.active_timeout_s),
            cache_key=str(payload.cache_key),
        )

        if not self._submit_parallel_batch_tasks(
            payload=payload,
            context=ctx if isinstance(ctx, Mapping) else None,
        ):
            return

        if not self._batch_parallel.has_active_requests():
            self._finish_parallel_batch_with_no_active_requests()
            return

        total = self._batch_parallel.active_request_count()
        if bool(getattr(self, "_debug_batch_parallel", False)):
            logger.info(
                "BATCH_PAR submitted run_id=%s sets=%s workers=%s",
                int(run_id),
                int(total),
                int(max_workers),
            )
        self.ui.run_ui.set_sim_progress_value(0)
        self.ui.run_ui.set_status_text(f"Running {total} sets in parallel ({max_workers} workers)...")
        if hasattr(self, "_batch_completion_poll_timer"):
            self._batch_completion_poll_timer.start()

    def _finish_parallel_batch_with_no_active_requests(self) -> None:
        self._batch_context_owner.deactivate()
        self._shutdown_batch_lane_pool(force_terminate=False)
        self._simulation_running = False
        self.ui.run_ui.set_run_button_enabled(True)
        self.ui.run_ui.set_stop_button_enabled(False)

    def _handle_parallel_batch_runtime_waiting(
        self,
        *,
        rows: Sequence[int],
        queue_ids: Sequence[str],
        runtime_snapshot: RuntimeReadinessSnapshot,
    ) -> None:
        ctx = self._batch_context_owner.mark_runtime_waiting()
        waiting_state = self._batch_context_owner.active_batch_state(ctx)
        self._simulation_running = False
        self._slider_simulation_active = False
        if waiting_state is not None and bool(waiting_state.fast_mode):
            self._ensure_parallel_batch_pool_eagerly_created(wait=False)
            self.queue_pending_slider_preview_replay(
                target_set_ids=[str(set_id) for set_id in queue_ids if str(set_id)],
                request_id=int(waiting_state.request_id or self._next_slider_preview_request_id()),
            )
            if bool(runtime_snapshot.should_poll):
                QtCore.QTimer.singleShot(50, self._run_simulation_from_slider)
            else:
                self.clear_pending_slider_preview_replay(clear_plot_updates=False)
        else:
            self._queue_run_after_runtime_ready(
                rows_to_run=rows,
                runtime_snapshot=runtime_snapshot,
            )
        self.ui.run_ui.set_runtime_backed_run_controls_ready(False)
        self.ui.run_ui.set_run_button_enabled(True)
        self.ui.run_ui.set_stop_button_enabled(False)
        self.ui.run_ui.set_sim_progress_value(0)
        self.ui.run_ui.set_status_text("Batch runtime is not ready.")
        self.ui.run_ui.schedule_runtime_availability_refresh()

    def _handle_parallel_batch_runtime_check_failed(self, exc: Exception) -> None:
        ctx = self._batch_context_owner.mark_runtime_waiting()
        waiting_state = self._batch_context_owner.active_batch_state(ctx)
        self._simulation_running = False
        self._slider_simulation_active = False
        self.ui.run_ui.set_runtime_backed_run_controls_ready(False)
        self.ui.run_ui.set_run_button_enabled(True)
        self.ui.run_ui.set_stop_button_enabled(False)
        self.ui.run_ui.set_sim_progress_value(0)
        self.ui.run_ui.set_status_text(f"Batch runtime readiness check failed: {exc}")
        if waiting_state is not None and bool(waiting_state.fast_mode):
            self.clear_pending_slider_preview_replay(clear_plot_updates=False)
            with suppress(Exception):
                self.ui.slider.show_preview_unavailable_for_dirty_state(
                    f"Batch runtime readiness check failed: {exc}"
                )

    def _submit_parallel_batch_tasks(
        self,
        *,
        payload: Any,
        context: Mapping[str, Any] | None,
    ) -> bool:
        rows = list(payload.rows)
        queue_ids = list(payload.queue_ids)
        queue_names = list(payload.queue_names)
        run_id = int(payload.run_id)
        request_id = int(payload.request_id)
        pending_seed = payload.pending_init_seed
        pending_init_applied = bool(payload.pending_init_applied)

        for idx, set_id in enumerate(queue_ids):
            if not (0 <= idx < len(rows)):
                continue
            row = int(rows[idx])
            set_name = str(queue_names[idx]) if 0 <= idx < len(queue_names) else str(set_id)
            try:
                initials_dict = self._batch_dispatch_materialization_owner.materialize_initials(
                    row=row,
                    set_name=str(set_name),
                    fast_mode=bool(payload.fast_mode),
                    pending_init_seed=pending_seed,
                    pending_init_applied=bool(pending_init_applied),
                )
            except Exception as exc:
                self.ui.dialogs.message_box_warning(
                    "Invalid Initial Conditions",
                    f"Set '{set_name}' has invalid initial conditions:\n\n{exc}",
                )
                if payload.fast_mode:
                    self._clear_failed_fast_preview_ownership()
                ctx = self._batch_context_owner.deactivate()
                self._shutdown_batch_lane_pool(force_terminate=True)
                self._simulation_running = False
                self.ui.run_ui.set_run_button_enabled(True)
                self.ui.run_ui.set_stop_button_enabled(False)
                self._slider_simulation_active = False
                self._invalidate_preserved_pending_init_results_after_failed_run(ctx=ctx)
                return False

            task_plan = build_parallel_batch_task_plan(
                ParallelBatchTaskInput(
                    payload=payload,
                    set_id=str(set_id),
                    set_name=str(set_name),
                    queue_ids=[str(item) for item in queue_ids],
                    initials=dict(initials_dict),
                    include_mechanism_in_result_payload=self._include_mechanism_in_result_payload(
                        fast_mode=bool(payload.fast_mode),
                        batch_set_id=str(set_id),
                        context=context,
                    ),
                )
            )
            sid = str(set_id)
            try:
                callback_identity = self._capture_simulation_callback_identity(
                    run_id=payload.run_id,
                    fast_mode=payload.fast_mode,
                    request_id=payload.request_id,
                    owner_epoch=payload.preview_owner_epoch,
                    batch_set=str(set_name),
                    batch_set_id=sid,
                    cache_key=payload.cache_key,
                )
                self._batch_parallel.submit_task(
                    task_plan.task,
                    set_id=sid,
                    set_name=str(set_name),
                    callback_identity=callback_identity,
                )
            except Exception as exc:
                self._record_nonfatal_exception(
                    f"Failed to submit batch lane request (set_id={sid})",
                    exc,
                )
                error_payload = simulation_failure_from_exception(exc, kind="simulation_containment_submission")
                details = dict(error_payload.get("details") or {})
                details.setdefault("source", "simulation_containment")
                error_payload["details"] = details
                if self._try_handle_scoped_batch_failure(
                    set_id=sid,
                    set_name=str(set_name),
                    error_payload=error_payload,
                ):
                    continue
                self._dispatch_simulation_error(
                    error_payload,
                    fast_mode=bool(payload.fast_mode),
                    run_id=int(run_id),
                    request_id=int(request_id),
                    owner_epoch=payload.preview_owner_epoch,
                    batch_set=str(set_name),
                    batch_set_id=sid,
                    cache_key=str(payload.cache_key),
                )
                return False
        return True

    def _abort_for_unready_interactive_runtime(self, *, fast_mode: bool, context: Mapping[str, Any]) -> None:
        self._batch_context_owner.deactivate_if_active(context)
        self._simulation_running = False
        self._slider_simulation_active = False
        self.ui.run_ui.set_runtime_backed_run_controls_ready(False)
        self.ui.run_ui.set_run_button_enabled(True)
        self.ui.run_ui.set_stop_button_enabled(False)
        self.ui.run_ui.set_sim_progress_value(0)
        if bool(fast_mode):
            message = "Preview runtime is not ready."
            self._clear_failed_fast_preview_ownership()
            self.ui.slider.set_slider_triggered_simulation(False)
            with suppress(Exception):
                self.ui.slider.show_preview_unavailable_for_dirty_state(message)
        else:
            message = "Simulation runtime is not ready."
            self._requeue_preserved_pending_slider_replay_after_preflight_abort()
        self.ui.run_ui.set_status_text(message)
        self.ui.run_ui.schedule_runtime_availability_refresh()

    def _abort_serial_batch_for_invalid_initials(
        self,
        *,
        row: int,
        set_id: str,
        set_name: str,
        fast_mode: bool,
        context: Mapping[str, Any] | None,
        exc: Exception,
    ) -> None:
        try:
            self.ui.batch.batch_model_validate_rows([int(row)])
        except Exception as validate_exc:
            self._record_nonfatal_exception(
                f"Failed to validate batch model rows after invalid initials (row={int(row)} set_id={str(set_id)})",
                validate_exc,
            )
        self.ui.dialogs.message_box_warning(
            "Invalid Initial Conditions",
            f"Set '{set_name}' has invalid initial conditions:\n\n{exc}",
        )
        if bool(fast_mode):
            self._clear_failed_fast_preview_ownership()
        ctx = self._batch_context_owner.deactivate()
        self._simulation_running = False
        self.ui.run_ui.set_run_button_enabled(True)
        self.ui.run_ui.set_stop_button_enabled(False)
        self._slider_simulation_active = False
        self._invalidate_preserved_pending_init_results_after_failed_run(ctx=ctx or context)

    def _start_contained_serial_batch_worker(
        self,
        *,
        plan_payload: Mapping[str, Any] | None,
        run_id: int,
        request_id: int,
        fast_mode: bool,
        owner_epoch: int | None,
        set_name: str,
        set_id: str,
        cache_key: str,
        context: Mapping[str, Any] | None,
        include_mechanism_in_result_payload: bool,
        worker_signature: str | None,
    ) -> bool:
        callback_identity = self._capture_simulation_callback_identity(
            run_id=int(run_id),
            fast_mode=bool(fast_mode),
            request_id=int(request_id),
            owner_epoch=owner_epoch,
            batch_set=str(set_name),
            batch_set_id=str(set_id),
            cache_key=str(cache_key),
        )
        if not isinstance(plan_payload, dict):
            self._dispatch_simulation_error(
                simulation_failure_from_exception(
                    ValueError("Missing simulation plan payload for contained batch dispatch"),
                    kind="simulation_plan_payload",
                ),
                run_id=int(run_id),
                fast_mode=bool(fast_mode),
                request_id=int(request_id),
                owner_epoch=owner_epoch,
                batch_set=str(set_name),
                batch_set_id=str(set_id),
                cache_key=str(cache_key),
                callback_identity=callback_identity,
            )
            return False

        try:
            self._simulation_worker = self._contained_serial_worker_launch_owner.create_worker(
                ContainedSerialWorkerLaunchRequest(
                    plan_payload=plan_payload,
                    callback_identity=callback_identity,
                    include_mechanism_in_result_payload=bool(include_mechanism_in_result_payload),
                    worker_signature=worker_signature,
                    parent=self,
                )
            )
            if self._simulation_worker is None:
                self._abort_for_unready_interactive_runtime(
                    fast_mode=bool(fast_mode),
                    context=context or {},
                )
                return False
        except Exception as exc:
            self._dispatch_simulation_error(
                simulation_failure_from_exception(exc, kind="simulation_containment_payload"),
                run_id=int(run_id),
                fast_mode=bool(fast_mode),
                request_id=int(request_id),
                owner_epoch=owner_epoch,
                batch_set=str(set_name),
                batch_set_id=str(set_id),
                cache_key=str(cache_key),
                callback_identity=callback_identity,
            )
            return False

        self._connect_simulation_worker_application_signals(
            self._simulation_worker,
            run_id=int(run_id),
            fast_mode=bool(fast_mode),
            request_id=int(request_id),
            owner_epoch=owner_epoch,
            set_name=str(set_name),
            set_id=str(set_id),
            cache_key=str(cache_key),
        )

        self._simulation_worker.start()
        return True

    def _serial_batch_dispatch_state(
        self,
        *,
        payload: Any,
        context: Mapping[str, Any] | None,
        queue_ids: Sequence[str],
        set_id: str,
        set_name: str,
        initials_dict: Mapping[str, Any],
    ) -> _SerialBatchDispatchState:
        dispatch_plan = build_serial_batch_dispatch_plan(
            SerialBatchDispatchInput(
                payload=payload,
                queue_ids=[str(item) for item in queue_ids],
                set_id=str(set_id),
                set_name=str(set_name),
                initials=dict(initials_dict),
                slider_overrides=self.ui.mechanism.slider_overrides(set_id=str(set_id)),
            ),
            apply_parameter_overrides_to_dsl=self.ui.mechanism.apply_parameter_overrides_to_dsl,
            record_nonfatal_exception=self._record_nonfatal_exception,
        )
        if dispatch_plan.cache_key_rewritten and isinstance(context, dict):
            context = self._batch_context_owner.record_cache_key(str(dispatch_plan.cache_key))

        return _SerialBatchDispatchState(
            plan_payload=dispatch_plan.plan_payload,
            cache_key=str(dispatch_plan.cache_key),
            worker_signature=str(dispatch_plan.worker_signature or ""),
            context=context if isinstance(context, Mapping) else None,
        )

    def _start_next_batch_simulation(self) -> None:
        state = self._batch_context_owner.active_batch_state()
        if state is None or not state.active:
            return

        state = self._batch_context_owner.active_batch_state()
        transition = self._batch_context_owner.consume_stale_serial_queue_prefix_for_current_epochs(
            current_global_epoch=int(getattr(self, "_authoritative_runtime_input_global_epoch", 0) or 0),
            current_set_epoch_by_set_id=self._runtime_input_context_set_epochs(
                state.queue_ids if state is not None else ()
            ),
            current_epoch=int(getattr(self, "_authoritative_runtime_input_epoch", 0) or 0),
        )
        ctx = transition.context
        pos = int(transition.completed_count)
        if transition.batch_done:
            self._finalize_batch_queue_done_without_result(ctx)
            return

        payload = self._batch_context_owner.serial_next_payload(ctx)
        if payload is None:
            self._finalize_batch_queue_done_without_result(ctx)
            return
        queue_ids = list(payload.queue_ids)
        set_id = str(payload.set_id)
        set_name = str(payload.set_name or self.ui.batch.batch_set_name_for_id(set_id) or set_id)
        row = int(payload.row)

        try:
            initials_dict = self._batch_dispatch_materialization_owner.materialize_initials(
                row=row,
                set_name=str(set_name),
                fast_mode=bool(payload.fast_mode),
                pending_init_seed=payload.pending_init_seed,
                pending_init_applied=bool(payload.pending_init_applied),
            )
        except Exception as exc:
            self._abort_serial_batch_for_invalid_initials(
                row=int(row),
                set_id=str(set_id),
                set_name=str(set_name),
                fast_mode=bool(payload.fast_mode),
                context=ctx if isinstance(ctx, Mapping) else None,
                exc=exc,
            )
            return
        fast_mode = bool(payload.fast_mode)

        request_id = int(payload.request_id)
        dispatch_state = self._serial_batch_dispatch_state(
            payload=payload,
            context=ctx if isinstance(ctx, Mapping) else None,
            queue_ids=[str(item) for item in queue_ids],
            set_id=str(set_id),
            set_name=str(set_name),
            initials_dict=dict(initials_dict),
        )
        ctx = dispatch_state.context

        self._release_current_simulation_worker()

        self._run_sequence_id = int(getattr(self, "_run_sequence_id", 0)) + 1
        run_id = int(self._run_sequence_id)
        self._active_run_id = run_id

        include_mechanism_in_result_payload = self._include_mechanism_in_result_payload(
            fast_mode=bool(fast_mode),
            batch_set_id=set_id,
            context=ctx,
        )

        total = len(queue_ids)
        self.ui.run_ui.set_status_text(f"Running {set_name} ({pos + 1}/{total})...")
        started = self._start_contained_serial_batch_worker(
            plan_payload=dispatch_state.plan_payload,
            run_id=int(run_id),
            request_id=int(request_id),
            fast_mode=bool(fast_mode),
            owner_epoch=payload.preview_owner_epoch,
            set_name=str(set_name),
            set_id=str(set_id),
            cache_key=str(dispatch_state.cache_key),
            context=ctx if isinstance(ctx, Mapping) else None,
            include_mechanism_in_result_payload=bool(include_mechanism_in_result_payload),
            worker_signature=str(dispatch_state.worker_signature or ""),
        )
        if not started:
            return

    def _clear_slider_triggered_preflight_state(self, *, fast_mode: bool) -> None:
        if bool(fast_mode):
            self.ui.slider.set_slider_triggered_simulation(False)

    def _set_simulation_running(self, value: bool) -> None:
        self._simulation_running = bool(value)

    def _set_slider_simulation_active(self, value: bool) -> None:
        self._slider_simulation_active = bool(value)

    def _defer_active_fast_run_if_needed(
        self,
        *,
        fast_mode: bool,
        batch_rows: Optional[Sequence[int]],
        request_id: Optional[int],
        runtime_readiness_only: bool,
    ) -> bool:
        if not bool(fast_mode):
            return False
        active_fast_worker = False
        worker = getattr(self, "_simulation_worker", None)
        if worker is not None and hasattr(worker, "isRunning"):
            try:
                active_fast_worker = self._worker_is_running(worker) and bool(getattr(worker, "_fast_mode", False))
            except Exception:
                active_fast_worker = False

        state = self._batch_context_owner.active_batch_state()
        active_fast_parallel = bool(state is not None and state.active and state.fast_mode)
        if not (active_fast_worker or active_fast_parallel):
            return False
        if bool(runtime_readiness_only):
            return True
        logger.debug("Fast slider run already in flight; recording latest-only pending request")
        self._pending_slider_simulation = True
        deferred_target_set_ids: list[str] = []
        for row in list(batch_rows or []):
            try:
                set_id = self.ui.batch.batch_set_id_for_row(int(row))
            except Exception:
                continue
            set_id_s = str(set_id or "").strip()
            if set_id_s and set_id_s not in deferred_target_set_ids:
                deferred_target_set_ids.append(set_id_s)
        if deferred_target_set_ids:
            self._pending_slider_target_set_ids = deferred_target_set_ids
        if request_id is not None:
            self._pending_slider_sim_request_id = int(request_id)
        return True

    def _run_request_id(self, *, request_id: Optional[int], runtime_readiness_only: bool) -> int:
        if bool(runtime_readiness_only):
            return int(request_id or 0)
        if request_id is None:
            return int(self._next_sim_request_id())
        return int(self._mark_request_started(int(request_id)))

    def _run_rows_or_abort(
        self,
        *,
        batch_rows: Optional[Sequence[int]],
        fast_mode: bool,
        runtime_readiness_only: bool,
    ) -> List[int] | None:
        if batch_rows is None:
            batch_rows = self.ui.batch.batch_rows_for_scope("selected")
        row_count = int(self.ui.batch.batch_store_row_count())
        rows = [int(r) for r in (batch_rows or []) if 0 <= int(r) < int(row_count)]
        if not rows:
            if int(row_count) > 0:
                return [0]
            if bool(runtime_readiness_only):
                return None
            self.ui.dialogs.message_box_warning("No Sets", "Add at least one set before running.")
            if bool(fast_mode):
                self._clear_failed_fast_preview_ownership()
            self._simulation_running = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self._clear_slider_triggered_preflight_state(fast_mode=bool(fast_mode))
            if not bool(fast_mode):
                self._requeue_preserved_pending_slider_replay_after_preflight_abort()
            return None
        invalid = self.ui.batch.batch_model_validate_rows(rows)
        if invalid:
            if bool(runtime_readiness_only):
                return None
            examples = sorted(invalid)[:8]
            details = "\n".join(f"  • row {r+1}: {sp}" for r, sp in examples)
            more = "" if len(invalid) <= len(examples) else f"\n  • ... and {len(invalid) - len(examples)} more"
            self._invalidate_preserved_pending_init_results_after_failed_run(ctx=None)
            self.ui.dialogs.message_box_warning(
                "Invalid Initial Conditions",
                "Fix invalid numeric cells in the Initial Conditions table before running:\n\n" + details + more,
            )
            if bool(fast_mode):
                self._clear_failed_fast_preview_ownership()
            self._simulation_running = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self._clear_slider_triggered_preflight_state(fast_mode=bool(fast_mode))
            if not bool(fast_mode):
                self._requeue_preserved_pending_slider_replay_after_preflight_abort()
            return None
        return rows

    def _run_mechanism_context_or_abort(
        self,
        *,
        fast_mode: bool,
        request_id: int,
        batch_rows: Sequence[int],
        runtime_readiness_only: bool,
    ) -> RunMechanismContext | None:
        return self._run_preparation_owner.build_mechanism_context_or_abort(
            fast_mode=bool(fast_mode),
            request_id=int(request_id),
            batch_rows=batch_rows,
            runtime_readiness_only=bool(runtime_readiness_only),
        )

    def _run_solver_context_or_abort(
        self,
        *,
        fast_mode: bool,
        runtime_readiness_only: bool,
        mechanism_context: RunMechanismContext,
    ) -> RunSolverContext | None:
        return self._run_preparation_owner.build_solver_context_or_abort(
            fast_mode=bool(fast_mode),
            runtime_readiness_only=bool(runtime_readiness_only),
            mechanism_context=mechanism_context,
        )

    def _sync_batch_species_columns_for_run(
        self,
        *,
        fast_mode: bool,
        slider_runtime: object | None,
        full_dsl: str,
        temperature_K: float,
    ) -> None:
        species_for_sync: List[str] = []
        if fast_mode and slider_runtime is not None:
            species_for_sync = list(getattr(slider_runtime, "species_names", []) or [])
        else:
            try:
                last_mech = self.ui.mechanism_helpers.last_mechanism()
                last_ctx = self.ui.mechanism_helpers.last_mechanism_context()
                if last_mech is not None and str(last_ctx.get("dsl_text") or "") == str(full_dsl):
                    species_for_sync = list(last_mech.species_names())
                else:
                    from kindred.core.simulator.dsl import parse_dsl_to_mechanism
                    from kindred.core.units import UnitsModel

                    mech_tmp = parse_dsl_to_mechanism(
                        full_dsl,
                        initials={},
                        units=UnitsModel(temperature_K=float(temperature_K), energy_unit="kJ/mol"),
                    )
                    species_for_sync = list(mech_tmp.species_names())
            except Exception:
                species_for_sync = []
        if species_for_sync:
            try:
                self.ui.batch.sync_batch_species_columns(species_for_sync)
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to sync batch species columns from mechanism species list",
                    exc,
                )

    def _build_run_dispatch_context_or_abort(
        self,
        *,
        fast_mode: bool,
        runtime_readiness_only: bool,
        mechanism_context: RunMechanismContext,
        solver_context: RunSolverContext,
    ) -> RunDispatchContext | None:
        return self._run_preparation_owner.build_dispatch_context_or_abort(
            fast_mode=bool(fast_mode),
            runtime_readiness_only=bool(runtime_readiness_only),
            mechanism_context=mechanism_context,
            solver_context=solver_context,
        )

    def _warm_runtime_for_dispatch_context(
        self,
        *,
        fast_mode: bool,
        wait: bool,
        dispatch_context: RunDispatchContext,
    ) -> None:
        from kindred.core.simulation_containment import build_contained_simulation_plan_payload

        contained_payloads = [
            build_contained_simulation_plan_payload(plan_payload)
            for plan_payload in dispatch_context.simulation_plan_by_set_id.values()
            if isinstance(plan_payload, dict)
        ]
        if not contained_payloads:
            return
        mode = self._contained_owner_mode(fast_mode=bool(fast_mode))

        def _owner_factory(payload: Mapping[str, object]):
            return self._new_contained_simulation_owner(
                fast_mode=bool(fast_mode),
                simulation_plan_payload=dict(payload),
            )

        if len(contained_payloads) == 1:
            self._runtime_application.ensure_ready(
                mode=mode,
                payload=dict(contained_payloads[0]),
                owner_factory=_owner_factory,
                wait=bool(wait),
            )
        else:
            self._runtime_application.ensure_ready_many(
                mode=mode,
                payloads=[dict(payload) for payload in contained_payloads],
                owner_factory=_owner_factory,
                wait=bool(wait),
            )

    def _start_run_context_and_dispatch(
        self,
        *,
        fast_mode: bool,
        request_id: int,
        reuse_parallel_lane_pool: bool,
        mechanism_context: RunMechanismContext,
        solver_context: RunSolverContext,
        dispatch_context: RunDispatchContext,
    ) -> None:
        if bool(reuse_parallel_lane_pool):
            self._supersede_parallel_batch_run_soft()
        else:
            self._shutdown_batch_lane_pool(force_terminate=True)

        self._release_current_simulation_worker()
        run_start_cache_decision = self._completion_policy.build_run_start_cache_decision(
            fast_mode=bool(fast_mode),
            queue_ids=tuple(mechanism_context.queue_ids),
        )
        explicit_valid_set_ids = run_start_cache_decision.explicit_cache_valid_set_ids
        self._batch_cache.apply_run_start_cache_decision(
            fast_mode=bool(fast_mode),
            explicit_cache_valid_set_ids=explicit_valid_set_ids,
            explicit_cache_invalidated_set_ids=run_start_cache_decision.explicit_cache_invalidated_set_ids,
            preview_scope_set_ids=run_start_cache_decision.preview_scope_set_ids,
        )
        effective_workers = self._effective_batch_worker_count(len(mechanism_context.queue_ids))
        dirty_reset_tracking = self._completion_policy.capture_dirty_reset_tracking(
            fast_mode=bool(fast_mode),
            queue_ids=tuple(mechanism_context.queue_ids),
            dirty_state_by_set_id=self._capture_dirty_state_by_set_id(mechanism_context.queue_ids),
        )
        run_start_context = build_run_start_context(
            request_id=int(request_id),
            current_run_sequence_id=int(getattr(self, "_run_sequence_id", 0)),
            runtime_input_epoch=int(getattr(self, "_authoritative_runtime_input_epoch", 0) or 0),
            runtime_input_global_epoch=int(getattr(self, "_authoritative_runtime_input_global_epoch", 0) or 0),
            runtime_input_set_epoch_by_set_id=self._runtime_input_context_set_epochs(mechanism_context.queue_ids),
            fast_mode=bool(fast_mode),
            reuse_parallel_lane_pool=bool(reuse_parallel_lane_pool),
            effective_workers=int(effective_workers),
            mechanism_context=mechanism_context,
            solver_context=solver_context,
            dispatch_context=dispatch_context,
            run_start_cache_decision=run_start_cache_decision,
            dirty_reset_tracking=dirty_reset_tracking,
        )
        self._run_sequence_id = int(run_start_context.run_sequence_id)
        if run_start_context.run_id is not None:
            self._active_run_id = int(run_start_context.run_id)
        self._batch_context_owner.start_run(run_start_context.request)

        self._slider_simulation_active = bool(fast_mode)
        if bool(getattr(self, "_debug_batch_parallel", False)):
            logger.info(
                "BATCH_PAR run prepared request_id=%s run_id=%s sets=%s workers=%s parallel=%s slider=%s",
                int(request_id),
                int(run_start_context.run_id or 0),
                int(len(mechanism_context.queue_ids)),
                int(run_start_context.effective_workers),
                bool(run_start_context.parallel_mode),
                bool(reuse_parallel_lane_pool),
            )
        if run_start_context.parallel_mode:
            self._start_parallel_batch_simulations()
        else:
            self._start_next_batch_simulation()

    def _run_simulation_internal(
        self,
        fast_mode: bool = False,
        *,
        request_id: Optional[int] = None,
        batch_rows: Optional[Sequence[int]] = None,
        reuse_parallel_lane_pool: bool = False,
        runtime_readiness_only: bool = False,
        runtime_readiness_wait: bool = False,
    ):
        if self._defer_active_fast_run_if_needed(
            fast_mode=bool(fast_mode),
            batch_rows=batch_rows,
            request_id=request_id,
            runtime_readiness_only=bool(runtime_readiness_only),
        ):
            return

        request_id_value = self._run_request_id(
            request_id=request_id,
            runtime_readiness_only=bool(runtime_readiness_only),
        )
        rows = self._run_rows_or_abort(
            batch_rows=batch_rows,
            fast_mode=bool(fast_mode),
            runtime_readiness_only=bool(runtime_readiness_only),
        )
        if rows is None:
            return
        mechanism_context = self._run_mechanism_context_or_abort(
            fast_mode=bool(fast_mode),
            request_id=int(request_id_value),
            batch_rows=rows,
            runtime_readiness_only=bool(runtime_readiness_only),
        )
        if mechanism_context is None:
            return
        solver_context = self._run_solver_context_or_abort(
            fast_mode=bool(fast_mode),
            runtime_readiness_only=bool(runtime_readiness_only),
            mechanism_context=mechanism_context,
        )
        if solver_context is None:
            return
        dispatch_context = self._build_run_dispatch_context_or_abort(
            fast_mode=bool(fast_mode),
            runtime_readiness_only=bool(runtime_readiness_only),
            mechanism_context=mechanism_context,
            solver_context=solver_context,
        )
        if dispatch_context is None:
            return
        if bool(runtime_readiness_only):
            self._warm_runtime_for_dispatch_context(
                fast_mode=bool(fast_mode),
                wait=bool(runtime_readiness_wait),
                dispatch_context=dispatch_context,
            )
            return
        self._start_run_context_and_dispatch(
            fast_mode=bool(fast_mode),
            request_id=int(request_id_value),
            reuse_parallel_lane_pool=bool(reuse_parallel_lane_pool),
            mechanism_context=mechanism_context,
            solver_context=solver_context,
            dispatch_context=dispatch_context,
        )

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------
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
        run_id: Optional[int] = None,
        fast_mode: Optional[bool] = None,
        request_id: Optional[int] = None,
        owner_epoch: Optional[int] = None,
        *,
        batch_set: Optional[str] = None,
        batch_set_id: Optional[str] = None,
        cache_key: Optional[str] = None,
        callback_identity: SimulationCallbackIdentity | None = None,
    ):
        identity = callback_identity or self._capture_simulation_callback_identity(
            run_id=run_id,
            fast_mode=fast_mode,
            request_id=request_id,
            owner_epoch=owner_epoch,
            batch_set=batch_set,
            batch_set_id=batch_set_id,
            cache_key=cache_key,
        )
        self._completion_callback_owner.handle_completion(
            result,
            run_id=identity.run_id,
            fast_mode=identity.fast_mode,
            request_id=identity.request_id,
            owner_epoch=identity.owner_epoch,
            batch_set=identity.batch_set,
            batch_set_id=identity.batch_set_id,
            cache_key=identity.cache_key,
            debug_batch_parallel=bool(getattr(self, "_debug_batch_parallel", False)),
            callback_identity=identity,
        )

    def _on_simulation_error(
        self,
        error_msg: object,
        run_id: Optional[int] = None,
        fast_mode: Optional[bool] = None,
        request_id: Optional[int] = None,
        owner_epoch: Optional[int] = None,
        *,
        batch_set: Optional[str] = None,
        batch_set_id: Optional[str] = None,
        cache_key: Optional[str] = None,
        callback_identity: SimulationCallbackIdentity | None = None,
    ):
        identity = callback_identity or self._capture_simulation_callback_identity(
            run_id=run_id,
            fast_mode=fast_mode,
            request_id=request_id,
            owner_epoch=owner_epoch,
            batch_set=batch_set,
            batch_set_id=batch_set_id,
            cache_key=cache_key,
        )
        self._error_handling_owner.handle_error(
            error_msg,
            run_id=identity.run_id,
            fast_mode=identity.fast_mode,
            request_id=identity.request_id,
            owner_epoch=identity.owner_epoch,
            batch_set=identity.batch_set,
            batch_set_id=identity.batch_set_id,
            cache_key=identity.cache_key,
            callback_identity=identity,
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
        if error_detail_text:
            logger.warning("%s", error_detail_text)
        if kind == "timeout":
            status_text = "Preview timed out. Adjust sliders or run again."
        else:
            status_text = "Preview unavailable. Adjust sliders or run again."
        logger.warning("Preview simulation failed without modal: %s", error_text)

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
            self._detach_contained_simulation_owner(fast_mode=active_fast)
            logger.info("Cancellation requested from simulation worker")
            self.ui.run_ui.set_status_text("Cancelling simulation...")
        else:
            self._close_contained_simulation_owner(fast_mode=active_fast, kill=True)
            self._simulation_running = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self.ui.run_ui.set_status_text("Ready")
            self.ui.run_ui.set_sim_progress_value(0)
