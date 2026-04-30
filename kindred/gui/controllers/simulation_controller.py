from __future__ import annotations

from dataclasses import replace
from contextlib import suppress
import hashlib
import json
import logging
import os
import platform
import threading
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from PySide6 import QtCore
import shiboken6

from kindred import __version__ as KINDRED_VERSION
from kindred.core.batch_parallel import (
    batch_mechanism_signature,
    compute_effective_batch_workers,
)
from kindred.core.batch_containment import BatchCompletionRecord, BatchLaneOutcome, BatchLanePool
from kindred.core.simulation_identity import (
    SimulationIdentity,
    SimulationScopeIdentity,
    canonical_initials_fingerprint,
    contained_simulation_owner_identity,
    coerce_simulation_identity,
)
from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
from kindred.core.simulation_runtime_readiness import RuntimeReadinessSnapshot, SimulationRuntimeApplication
from kindred.core.simulation_failure import (
    build_simulation_failure,
    coerce_simulation_failure,
    is_cancelled_failure,
    simulation_failure_from_exception,
    simulation_failure_detail_text,
    simulation_failure_user_message,
)
from kindred.core.simulation_preparation import SimulationExecutionRequest
from kindred.gui.controllers.cache_contracts import build_batch_cache_entry
from kindred.gui.controllers.simulation_completion_policy import (
    CacheAuthorityState,
    CompletionPolicyContext,
    DirtySetState,
    PendingReplayDirective,
    PendingReplayState,
    PolicyStatePatch,
    RunActivitySnapshot,
    SimulationCompletionPolicy,
    pending_initial_seed_for_set,
)
from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.gui.controllers.parallel_batch_executor import ParallelBatchExecutor
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
from kindred.core.batch_initial_conditions import (
    migrate_reaction_dsl_initial_concentration_sets,
    strip_reaction_dsl_initial_concentrations,
)
from kindred.gui.ports import SimulationCacheOpResult, SimulationUiPorts, SliderReplayIntent

logger = logging.getLogger(__name__)

__all__ = ["SimulationController"]

_WORKER_APPLICATION_SIGNAL_HANDLERS_ATTR = "_kindred_controller_worker_signal_handlers"


def _try_float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _simulation_plan_payload(value: object) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, SimulationPlan):
        return value.to_payload()
    if isinstance(value, Mapping):
        return SimulationPlan.from_payload(value).to_payload()
    return None


def _simulation_plan(value: object) -> Optional[SimulationPlan]:
    if value is None:
        return None
    if isinstance(value, SimulationPlan):
        return value
    if isinstance(value, Mapping):
        return SimulationPlan.from_payload(value)
    return None


def _simulation_plan_for_set_from_context(
    context: object,
    *,
    batch_set_id: Optional[str],
) -> Optional[SimulationPlan]:
    if not batch_set_id or not isinstance(context, Mapping):
        return None
    plan_by_set_id = context.get("simulation_plan_by_set_id")
    if not isinstance(plan_by_set_id, Mapping):
        return None
    try:
        return _simulation_plan(plan_by_set_id.get(str(batch_set_id)))
    except (TypeError, ValueError) as exc:
        logger.debug("Ignoring invalid simulation plan payload for cache identity: %s", exc, exc_info=True)
        return None


def _execution_request_payload_from_plan(value: object) -> Optional[Dict[str, Any]]:
    payload = _simulation_plan_payload(value)
    if payload is None:
        return None
    return SimulationPlan.from_payload(payload).to_execution_request().to_payload()


def _simulation_plan_payload_with_execution_request(
    plan_payload: Mapping[str, Any],
    execution_request: Mapping[str, Any],
    *,
    algebra_policy: Optional[SimulationAlgebraPolicy] = None,
) -> Dict[str, Any]:
    plan = SimulationPlan.from_payload(plan_payload)
    return SimulationPlan.from_execution_request(
        execution_request,
        execution_mode=plan.execution_mode,
        algebra_policy=algebra_policy or plan.algebra_policy,
        cache_identity_payload=plan.cache_identity_payload,
        cache_scope_payload=plan.cache_scope_payload,
        metadata=plan.metadata,
        version=plan.version,
    ).to_payload()


def _new_simulation_plan_payload(
    execution_request: Mapping[str, Any],
    *,
    execution_mode: str,
    algebra_policy: SimulationAlgebraPolicy = SimulationAlgebraPolicy.GUI_BEST_EFFORT,
    cache_identity_payload: Optional[Mapping[str, Any]] = None,
    cache_scope_payload: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return SimulationPlan.from_execution_request(
        execution_request,
        execution_mode=execution_mode,
        algebra_policy=algebra_policy,
        cache_identity_payload=cache_identity_payload,
        cache_scope_payload=cache_scope_payload,
        metadata=metadata,
    ).to_payload()


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
    identity = coerce_simulation_identity(simulation_identity)
    if identity is not None:
        return identity.cache_key()
    solver_config_parts = []
    for key, value in sorted((solver_config or {}).items(), key=lambda kv: str(kv[0])):
        solver_config_parts.append(f"{key}={value!r}")
    solver_config_for_key = "|".join(solver_config_parts)
    cache_key_material = "\x00".join(
        [
            str(mechanism_text),
            f"{float(t_end)!r}",
            str(solver_config_for_key),
        ]
    )
    return hashlib.sha256(cache_key_material.encode("utf-8")).hexdigest()


def build_fast_preview_solver_grid_context(
    *,
    initial_solver_name: Optional[str],
    num_points: int,
    fast_mode: bool,
    slider_points_override: Optional[int],
    slider_solver_override: Optional[str],
    slider_drag_active: bool,
    last_slider_change_name: str,
) -> Dict[str, Any]:
    from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, normalize_solver_name

    solver_label = str(initial_solver_name or DEFAULT_SOLVER_NAME).strip() or DEFAULT_SOLVER_NAME
    solver, solver_warning = normalize_solver_name(solver_label)
    n_points = int(num_points)

    if fast_mode:
        if slider_points_override is not None:
            n_points = max(50, int(slider_points_override))
        else:
            n_points = max(50, n_points)
        if slider_solver_override is not None:
            solver_label = str(slider_solver_override).strip() or solver_label
            solver, solver_warning = normalize_solver_name(solver_label)

    preview_mode = bool(
        fast_mode
        and slider_drag_active
        and isinstance(last_slider_change_name, str)
        and last_slider_change_name.startswith("Keq")
        and last_slider_change_name[3:].isdigit()
    )
    if preview_mode:
        n_points = min(int(n_points), 120)

    return {
        "solver": str(solver),
        "solver_label": str(solver_label),
        "solver_warning": str(solver_warning) if solver_warning else None,
        "grid": {"N": int(n_points)},
    }


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
        self._batch_run_context: Dict[str, Any] = {}

        # Parallel batch orchestration (warm lane owner adapter)
        self._batch_parallel = ParallelBatchExecutor(
            lane_pool_factory=_default_batch_lane_pool_factory,
            max_parallel_workers=int(PROJECT_DEFAULTS["max_parallel_batch_workers"]),
            limit_blas_threads_per_worker=bool(PROJECT_DEFAULTS["limit_blas_threads_per_worker"]),
            record_nonfatal_exception=self._record_nonfatal_exception,
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
        self._pool_eagerly_created: bool = False
        self._pool_eager_creation_thread: Optional[threading.Thread] = None
        self._pool_eager_creation_lock = threading.RLock()
        self._batch_runtime_lane_budget = int(PROJECT_DEFAULTS["batch_runtime_lane_budget"])
        self._completion_policy = SimulationCompletionPolicy()
        self._runtime_application = SimulationRuntimeApplication()

    # ------------------------------------------------------------------
    # Public interface (MainWindow boundary)
    # ------------------------------------------------------------------
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

    @property
    def batch_run_context(self) -> Dict[str, Any]:
        return self._batch_run_context

    @batch_run_context.setter
    def batch_run_context(self, value: Dict[str, Any]) -> None:
        self._batch_run_context = dict(value or {})

    def _completion_policy_context_from_raw(
        self,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[CompletionPolicyContext]:
        ctx = context if isinstance(context, Mapping) else getattr(self, "_batch_run_context", {}) or {}
        if not isinstance(ctx, Mapping):
            return None
        return CompletionPolicyContext(
            active=ctx.get("active"),
            request_id=ctx.get("request_id"),
            run_id=ctx.get("run_id"),
            fast_mode=ctx.get("fast_mode"),
            parallel=ctx.get("parallel"),
            keep_lane_pool_alive=ctx.get("keep_lane_pool_alive"),
            queue_ids=ctx.get("queue_ids"),
            queue_names=ctx.get("queue_names"),
            total=ctx.get("total"),
            pos=ctx.get("pos"),
            primary_set_id=ctx.get("primary_set_id"),
            completed_set_ids=ctx.get("completed_set_ids"),
            pending_workspace_reset_set_ids=ctx.get("pending_workspace_reset_set_ids"),
            pending_dirty_reset_generation_by_set_id=ctx.get("pending_dirty_reset_generation_by_set_id"),
            pending_init_seed=ctx.get("pending_init_seed"),
            pending_init_rewrite=ctx.get("pending_init_rewrite"),
            pending_init_applied=ctx.get("pending_init_applied", False),
            explicit_cache_preview_token=ctx.get("explicit_cache_preview_token"),
            explicit_cache_preview_scope_set_ids=ctx.get("explicit_cache_preview_scope_set_ids"),
            explicit_cache_valid_set_ids=ctx.get("explicit_cache_valid_set_ids"),
            explicit_cache_invalidated_set_ids=ctx.get("explicit_cache_invalidated_set_ids"),
            preview_scope_set_ids=ctx.get("preview_scope_set_ids"),
            preview_owner_epoch=ctx.get("preview_owner_epoch"),
        )

    def _serialize_completion_policy_context(
        self,
        context: CompletionPolicyContext,
        *,
        base_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw = dict(base_context or getattr(self, "_batch_run_context", {}) or {})
        raw["active"] = bool(context.active)
        raw["request_id"] = context.request_id
        raw["run_id"] = context.run_id
        raw["fast_mode"] = bool(context.fast_mode)
        raw["parallel"] = bool(context.parallel)
        raw["keep_lane_pool_alive"] = bool(context.keep_lane_pool_alive)
        raw["queue_ids"] = list(context.queue_ids)
        raw["queue_names"] = list(context.queue_names)
        raw["total"] = int(context.total)
        raw["pos"] = int(context.pos)
        raw["primary_set_id"] = context.primary_set_id
        raw["completed_set_ids"] = list(context.completed_set_ids)
        raw["pending_workspace_reset_set_ids"] = list(context.pending_workspace_reset_set_ids)
        raw["pending_dirty_reset_generation_by_set_id"] = dict(context.pending_dirty_reset_generation_by_set_id)
        raw["pending_init_seed"] = {
            str(set_name): {str(species): float(value) for species, value in dict(seed).items()}
            for set_name, seed in context.pending_init_seed.items()
        }
        raw["pending_init_rewrite"] = context.pending_init_rewrite
        raw["pending_init_applied"] = bool(context.pending_init_applied)
        raw["explicit_cache_preview_token"] = context.explicit_cache_preview_token
        raw["explicit_cache_preview_scope_set_ids"] = context.explicit_cache_preview_scope_set_ids
        raw["explicit_cache_valid_set_ids"] = context.explicit_cache_valid_set_ids
        raw["explicit_cache_invalidated_set_ids"] = context.explicit_cache_invalidated_set_ids
        raw["preview_scope_set_ids"] = context.preview_scope_set_ids
        raw["preview_owner_epoch"] = context.preview_owner_epoch
        return raw

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
            self._batch_run_context = self._serialize_completion_policy_context(
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

    def _preview_batch_cache_token_for_cached_result(
        self,
        *,
        batch_set_id: Optional[str],
        context: Dict[str, Any] | None,
    ) -> str:
        if not batch_set_id or not isinstance(context, dict):
            return ""
        token_by_set_id = context.get("preview_batch_cache_token_by_set_id")
        if not isinstance(token_by_set_id, dict):
            return ""
        token = token_by_set_id.get(str(batch_set_id))
        return str(token or "")

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
    ):
        return self._on_simulation_complete(
            result,
            run_id=run_id,
            fast_mode=fast_mode,
            request_id=request_id,
            batch_set=batch_set,
            batch_set_id=batch_set_id,
            cache_key=cache_key,
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
    ) -> None:
        self._on_simulation_error(
            error_msg,
            run_id=run_id,
            fast_mode=fast_mode,
            request_id=request_id,
            batch_set=batch_set,
            batch_set_id=batch_set_id,
            cache_key=cache_key,
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
    ):
        if owner_epoch is None:
            return self.on_simulation_complete(
                result,
                run_id=run_id,
                fast_mode=fast_mode,
                request_id=request_id,
                batch_set=batch_set,
                batch_set_id=batch_set_id,
                cache_key=cache_key,
            )
        return self._on_simulation_complete(
            result,
            run_id=run_id,
            fast_mode=fast_mode,
            request_id=request_id,
            owner_epoch=owner_epoch,
            batch_set=batch_set,
            batch_set_id=batch_set_id,
            cache_key=cache_key,
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
    ) -> None:
        if owner_epoch is None:
            self.on_simulation_error(
                error_msg,
                run_id=run_id,
                fast_mode=fast_mode,
                request_id=request_id,
                batch_set=batch_set,
                batch_set_id=batch_set_id,
                cache_key=cache_key,
            )
            return
        self._on_simulation_error(
            error_msg,
            run_id=run_id,
            fast_mode=fast_mode,
            request_id=request_id,
            owner_epoch=owner_epoch,
            batch_set=batch_set,
            batch_set_id=batch_set_id,
            cache_key=cache_key,
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
            context=self._completion_policy_context_from_raw(),
        )

    def _has_active_fast_preview_in_flight(self) -> bool:
        return self._completion_policy.has_active_fast_preview_in_flight(
            activity=self._completion_policy_activity_snapshot(),
            context=self._completion_policy_context_from_raw(),
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
        ctx = getattr(self, "_batch_run_context", {}) or {}
        active_fast = bool(isinstance(ctx, dict) and ctx.get("fast_mode"))
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
            row_count = int(self.ui.batch.batch_store_row_count())
        except Exception:
            row_count = 0
        rows = [int(row) for row in (batch_rows or []) if 0 <= int(row) < int(row_count)]
        if not rows and row_count > 0:
            rows = [0]
        if not rows:
            return []
        try:
            invalid = self.ui.batch.batch_model_validate_rows(rows)
        except Exception:
            invalid = set()
        if invalid:
            return []

        any_slider_workspace = bool(self.ui.mechanism.has_slider_overrides())
        has_slider_overrides = bool(fast_mode) and bool(any_slider_workspace)
        primary = self.ui.batch.batch_preferred_primary_set_id(rows)
        primary_set_id = str(primary) if primary is not None else None

        owner_reactions_text_raw = self.ui.mechanism.mechanism_reactions_text_raw()
        owner_state_network_dsl_raw = self.ui.mechanism.mechanism_state_network_dsl_raw()
        reactions_text_raw = owner_reactions_text_raw
        if has_slider_overrides:
            reactions_text_raw = self.ui.mechanism.apply_overrides_to_text(
                reactions_text_raw,
                set_id=primary_set_id,
            )

        pending_init_seed: Dict[str, Dict[str, float]] = {}
        migrated = reactions_text_raw
        try:
            pending_init_seed, migrated = migrate_reaction_dsl_initial_concentration_sets(
                reactions_text_raw,
                default_set_name="set1",
            )
        except Exception:
            pending_init_seed = {}
            migrated = reactions_text_raw

        names = list(self.ui.batch.batch_store_set_names())
        queue_names = [str(names[row]) for row in rows if 0 <= int(row) < len(names)]
        queue_ids = [
            str(self.ui.batch.batch_set_id_for_row(int(row)) or str(names[int(row)]))
            for row in rows
            if 0 <= int(row) < len(names)
        ]
        if not queue_ids:
            return []

        reactions_text = strip_reaction_dsl_initial_concentrations(migrated)
        state_network_dsl = owner_state_network_dsl_raw
        if has_slider_overrides:
            state_network_dsl = self.ui.mechanism.apply_overrides_to_state_network_dsl(
                state_network_dsl,
                set_id=primary_set_id,
            )
        full_dsl = reactions_text
        if state_network_dsl.strip():
            full_dsl += "\n\n# State Network\n" + state_network_dsl
        if has_slider_overrides:
            full_dsl = self._apply_parameter_override_fallback_to_dsl(full_dsl, set_id=primary_set_id)
        if not full_dsl.strip():
            return []

        owner_reactions_text = strip_reaction_dsl_initial_concentrations(owner_reactions_text_raw)
        owner_full_dsl = owner_reactions_text
        if owner_state_network_dsl_raw.strip():
            owner_full_dsl += "\n\n# State Network\n" + owner_state_network_dsl_raw

        solver_grid_context = build_fast_preview_solver_grid_context(
            initial_solver_name=self.ui.solver.initial_solver_name(),
            num_points=int(self.ui.solver.num_points_spinbox_value()),
            fast_mode=bool(fast_mode),
            slider_points_override=self.ui.mechanism.mechanism_slider_points_value(),
            slider_solver_override=self.ui.mechanism.mechanism_slider_solver_value(),
            slider_drag_active=bool(self.ui.slider.slider_drag_active()),
            last_slider_change_name=str(self.ui.slider.last_slider_change_name() or ""),
        )
        solver_label = str(solver_grid_context.get("solver_label") or "")
        solver = str(solver_grid_context.get("solver") or "")
        solver_warning = solver_grid_context.get("solver_warning")
        rtol = self.ui.solver.initial_rtol() or 1e-6
        atol = self.ui.solver.initial_atol() or 1e-12
        temperature_K = float(self.ui.solver.temperature_spinbox_value())
        T_override = self.ui.solver.dsl_global_temperature_K(full_dsl)
        if T_override is not None:
            temperature_K = float(T_override)
        solver_config = {
            "solver": solver,
            "solver_label": solver_label,
            "solver_warning": str(solver_warning) if solver_warning else None,
            "rtol": rtol,
            "atol": atol,
            "grid": {"N": int((solver_grid_context.get("grid") or {}).get("N") or 0)},
            "temperature_K": temperature_K,
            "use_sparse_jacobian": bool(self.ui.solver.use_sparse_jacobian()),
            "wegscheider_cyclicity_enabled": bool(self.ui.solver.wegscheider_cyclicity_enabled()),
        }
        try:
            t_end = float(self.ui.solver.parse_sim_time_seconds())
        except ValueError:
            return []

        preview_batch_cache_token_by_set_id: Dict[str, str] = {}
        for index, set_id in enumerate(queue_ids):
            token = ""
            if bool(fast_mode) and index < len(rows):
                try:
                    token = self.ui.slider.preview_batch_cache_token([int(rows[index])])
                except Exception:
                    token = ""
            preview_batch_cache_token_by_set_id[str(set_id)] = str(token or "")

        simulation_identity_by_set_id: Dict[str, dict[str, Any]] = {}
        simulation_plan_by_set_id: Dict[str, Dict[str, Any]] = {}
        owner_parameter_names_by_set_id: Dict[str, list[str]] = {}
        for index, set_id in enumerate(queue_ids):
            identity = self._simulation_identity_for_set(
                set_id=str(set_id),
                solver_config=solver_config,
                t_end=float(t_end),
                preview_batch_cache_token=preview_batch_cache_token_by_set_id.get(str(set_id), ""),
                fast_mode=bool(fast_mode),
            )
            simulation_identity_by_set_id[str(set_id)] = identity.to_payload()
            if index >= len(rows):
                continue
            row = int(rows[index])
            set_name = str(queue_names[index]) if index < len(queue_names) else str(set_id)
            request_mechanism_text = self._request_mechanism_text_for_set(
                set_id=str(set_id),
                has_slider_overrides=has_slider_overrides,
            )
            try:
                initials_dict = self._resolved_initials_for_batch_row(
                    row=row,
                    set_name=set_name,
                    pending_init_seed=pending_init_seed,
                    pending_init_applied=False,
                    include_preview_initials=bool(fast_mode),
                )
            except Exception:
                return []
            identity = self._simulation_identity_for_set(
                set_id=str(set_id),
                solver_config=solver_config,
                t_end=float(t_end),
                canonical_initials_fingerprint=canonical_initials_fingerprint(initials_dict),
                preview_batch_cache_token=preview_batch_cache_token_by_set_id.get(str(set_id), ""),
                fast_mode=bool(fast_mode),
            )
            simulation_identity_by_set_id[str(set_id)] = identity.to_payload()
            request_payload = SimulationExecutionRequest(
                prepared_payload=None,
                initials=dict(initials_dict),
                t_span=(0.0, float(t_end)),
                solver_config=dict(solver_config),
                mechanism_text=str(request_mechanism_text),
                simulation_identity=identity.to_payload(),
                parameter_overrides=(
                    self._slider_execution_parameter_values(set_id=str(set_id))
                    if bool(fast_mode)
                    else None
                ),
            ).to_payload()
            preview_token = preview_batch_cache_token_by_set_id.get(str(set_id), "")
            cache_identity_payload: Dict[str, Any] = {
                "cache_key": "",
                "simulation_identity": identity.to_payload(),
            }
            if preview_token:
                cache_identity_payload["preview_batch_cache_token"] = str(preview_token)
            simulation_plan_by_set_id[str(set_id)] = _new_simulation_plan_payload(
                request_payload,
                execution_mode="preview" if bool(fast_mode) else "explicit",
                algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
                cache_identity_payload=cache_identity_payload,
                metadata={
                    "set_id": str(set_id),
                    "set_name": set_name,
                    "fast_mode": bool(fast_mode),
                },
            )
            if bool(fast_mode):
                owner_parameter_names_by_set_id[str(set_id)] = self._slider_runtime_parameter_names(set_id=str(set_id))

        scope_identity = SimulationScopeIdentity.build(
            queue_ids=queue_ids,
            identity_by_set_id={
                set_id: payload
                for set_id, payload in simulation_identity_by_set_id.items()
            },
        )
        cache_key = self.ui.batch.batch_cache_key(scope_identity=scope_identity)
        set_name_by_set_id = {
            str(set_id): str(queue_names[index]) if index < len(queue_names) else str(set_id)
            for index, set_id in enumerate(queue_ids)
        }
        for set_id, plan_payload in list(simulation_plan_by_set_id.items()):
            request_payload = _execution_request_payload_from_plan(plan_payload)
            if request_payload is None:
                continue
            preview_token = preview_batch_cache_token_by_set_id.get(str(set_id), "")
            cache_identity_payload = {
                "cache_key": cache_key,
                "simulation_identity": dict(simulation_identity_by_set_id.get(str(set_id)) or {}),
            }
            if preview_token:
                cache_identity_payload["preview_batch_cache_token"] = str(preview_token)
            metadata: Dict[str, Any] = {
                "set_id": str(set_id),
                "set_name": set_name_by_set_id.get(str(set_id), str(set_id)),
                "fast_mode": bool(fast_mode),
            }
            if bool(fast_mode):
                metadata["contained_owner_identity"] = self._preview_contained_owner_identity(
                    owner_mechanism_text=str(request_payload.get("mechanism_text") or owner_full_dsl),
                    solver_config=solver_config,
                    t_end=float(t_end),
                    set_id=str(set_id),
                    parameter_names=owner_parameter_names_by_set_id.get(str(set_id), []),
                    simulation_identity=simulation_identity_by_set_id.get(str(set_id)),
                )
            else:
                metadata["contained_owner_identity"] = self._ordinary_contained_owner_identity(
                    owner_mechanism_text=owner_full_dsl,
                    solver_config=solver_config,
                    t_end=float(t_end),
                    set_id=str(set_id),
                    simulation_identity=simulation_identity_by_set_id.get(str(set_id)),
                )
            simulation_plan_by_set_id[str(set_id)] = _new_simulation_plan_payload(
                request_payload,
                execution_mode="preview" if bool(fast_mode) else "explicit",
                algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
                cache_identity_payload=cache_identity_payload,
                cache_scope_payload={
                    "scope_identity": scope_identity.to_payload(),
                    "queue_ids": [str(queue_id) for queue_id in queue_ids],
                },
                metadata=metadata,
            )

        from kindred.core.simulation_containment import build_contained_simulation_plan_payload

        return [
            build_contained_simulation_plan_payload(simulation_plan_by_set_id[str(set_id)])
            for set_id in queue_ids
            if str(set_id) in simulation_plan_by_set_id
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

        def result_handler(
            payload,
            _rid=run_id,
            _fast=bool(fast_mode),
            _req=int(request_id),
            _owner_epoch=owner_epoch,
            _set=set_name,
            _sid=set_id,
            _key=cache_key,
        ):
            return self._dispatch_simulation_complete(
                payload,
                run_id=_rid,
                fast_mode=_fast,
                request_id=_req,
                owner_epoch=_owner_epoch,
                batch_set=_set,
                batch_set_id=_sid,
                cache_key=_key,
            )

        def error_handler(
            msg,
            _rid=run_id,
            _fast=bool(fast_mode),
            _req=int(request_id),
            _owner_epoch=owner_epoch,
            _set=set_name,
            _sid=set_id,
            _key=cache_key,
        ):
            return self._dispatch_simulation_error(
                msg,
                run_id=_rid,
                fast_mode=_fast,
                request_id=_req,
                owner_epoch=_owner_epoch,
                batch_set=_set,
                batch_set_id=_sid,
                cache_key=_key,
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
            self._pool_eagerly_created = True
            return _runtime_readiness_snapshot(
                mode="batch",
                status="ready",
                ready=True,
                generation=int(getattr(snapshot, "current_generation", 0) or 0),
                required=True,
                controls_ready=True,
                polling=False,
            )
        self._pool_eagerly_created = False
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
        self._pool_eagerly_created = False
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
            self._pool_eagerly_created = False
            return
        self._shutdown_batch_lane_pool(force_terminate=False)

    def _ensure_parallel_batch_pool_eagerly_created(self, *, wait: bool = False) -> None:
        effective_workers = self._interactive_batch_runtime_capacity()
        try:
            if self._batch_parallel.has_ready_lane_pool(max_lanes=max(1, int(effective_workers))):
                self._pool_eagerly_created = True
                return
        except Exception:
            self._pool_eagerly_created = False
        self._pool_eagerly_created = False
        if not bool(wait):
            with self._pool_eager_creation_lock:
                existing = self._pool_eager_creation_thread
                if existing is not None and existing.is_alive():
                    return
                thread = threading.Thread(
                    target=self._ensure_parallel_batch_pool_eagerly_created,
                    kwargs={"wait": True},
                    name="kindred-batch-runtime-readiness",
                    daemon=True,
                )
                self._pool_eager_creation_thread = thread
                thread.start()
            return
        try:
            self._batch_parallel.ensure_warm_lane_pool(
                max_lanes=max(1, int(effective_workers)),
                wait=bool(wait),
            )
        except Exception:
            self._pool_eagerly_created = False
            return
        self._pool_eagerly_created = bool(
            self._batch_parallel.has_ready_lane_pool(max_lanes=max(1, int(effective_workers)))
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

    def _parallel_batch_runtime_ready(self) -> bool:
        effective_workers = self._interactive_batch_runtime_capacity()
        try:
            ready = bool(
                self._batch_parallel.has_ready_lane_pool(max_lanes=max(1, int(effective_workers)))
            )
        except Exception:
            ready = False
        if ready:
            self._pool_eagerly_created = True
        else:
            self._pool_eagerly_created = False
        return ready

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
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel"):
            ctx["active"] = False
            self._batch_run_context = dict(ctx)

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
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel") and bool(ctx.get("fast_mode")):
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

    def _supersede_active_work_for_authoritative_mechanism_transition(
        self,
        *,
        epoch: int,
        affected_set_ids: Sequence[str] = (),
        close_preview_runtime_owner: bool = True,
    ) -> None:
        self._authoritative_mechanism_transition_epoch = int(epoch)
        self._authoritative_runtime_input_epoch = int(epoch)
        self._authoritative_runtime_input_invalidated_set_ids = tuple(
            dict.fromkeys(str(set_id) for set_id in (affected_set_ids or ()) if str(set_id))
        )
        self._invalidate_active_explicit_simulation_for_authoritative_change()
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

        cache_store = (
            self._batch_cache.preview_cache if pending_cache_kind == "preview" else self._batch_cache.result_cache
        )

        shown_sets = list(self.ui.batch.shown_batch_set_ids())
        selected_sets = [str(set_id) for set_id in shown_sets if str(set_id)]
        if not selected_sets:
            selected_sets = list(self.ui.batch.batch_set_ids_for_scope("selected"))
        if not selected_sets:
            selected_sets = sorted(pending_set_ids)
        else:
            selected_sets = [str(set_id) for set_id in selected_sets if str(set_id)]
        if force and not selected_sets:
            prefix = f"{str(cache_key)}::"
            cached_ids: Set[str] = set()
            for k in list((cache_store or {}).keys()):
                token = str(k or "")
                if not token.startswith(prefix):
                    continue
                sid = token[len(prefix) :].strip()
                if sid:
                    cached_ids.add(sid)
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
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel"):
            ctx["active"] = False
            self._batch_run_context = dict(ctx)
        self.shutdown_batch_lane_pool(force_terminate=True)
        self._clear_stale_parallel_batch_requests()
        self._drain_batch_completion_queue()

    def _surface_current_parallel_batch_pool_failure_to_ui(self, error_msg: object) -> None:
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if not (isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel")):
            return
        if not self._batch_parallel.has_lane_pool():
            return
        self._dispatch_simulation_error(
            error_msg,
            run_id=int(ctx.get("run_id") or 0),
            fast_mode=bool(ctx.get("fast_mode")),
            request_id=int(ctx.get("request_id") or 0),
            owner_epoch=ctx.get("preview_owner_epoch"),
            batch_set="",
            batch_set_id="",
            cache_key=str(ctx.get("cache_key") or ""),
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

    def _record_scoped_batch_failure_cache_state(self, ctx: dict[str, Any], failed_set_id: str) -> None:
        sid = str(failed_set_id or "")
        if not sid:
            return
        valid_raw = ctx.get("explicit_cache_valid_set_ids")
        if valid_raw is None:
            return
        valid_set_ids = tuple(str(item) for item in (valid_raw or ()) if str(item) and str(item) != sid)
        invalidated_seen = {
            str(item)
            for item in (ctx.get("explicit_cache_invalidated_set_ids") or ())
            if str(item)
        }
        invalidated_seen.add(sid)
        queue_order = [str(item) for item in (ctx.get("queue_ids") or ()) if str(item)]
        invalidated_set_ids = tuple(
            item for item in queue_order if item in invalidated_seen
        ) + tuple(sorted(item for item in invalidated_seen if item not in set(queue_order)))
        ctx["explicit_cache_valid_set_ids"] = valid_set_ids
        ctx["explicit_cache_invalidated_set_ids"] = invalidated_set_ids
        if str(getattr(self._batch_cache, "active_cache_key", "") or "") == str(ctx.get("cache_key") or ""):
            self._batch_cache.active_cache_valid_set_ids = valid_set_ids
            self._batch_cache.active_cache_invalidated_set_ids = invalidated_set_ids

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

    def _finalize_scoped_batch_success_subset(self, ctx: Mapping[str, Any]) -> tuple[str, ...]:
        if not isinstance(ctx, Mapping):
            return ()
        policy_context = self._completion_policy_context_from_raw(ctx)
        if policy_context is None:
            return ()
        reset_target_set_ids = tuple(policy_context.pending_workspace_reset_set_ids)
        dirty_reset_decision = self._completion_policy.resolve_explicit_dirty_reset(
            context=policy_context,
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
                    "Failed to clear targeted slider workspaces after partial canonical explicit run",
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
                    "Failed to clear staged concentration overlays after partial canonical explicit run",
                    exc,
                )
        updated_context = self._apply_completion_policy_state_patch(
            dirty_reset_decision.state_patch,
            base_context=ctx,
        )
        if updated_context is not None:
            ctx = self._batch_run_context
        if overlays_cleared:
            species_for_sync = self._current_mechanism_species_for_batch_sync()
            if species_for_sync:
                try:
                    self.ui.batch.sync_batch_species_columns(
                        species_for_sync,
                        preserve_active_cache=True,
                    )
                except Exception as exc:
                    self._record_nonfatal_exception(
                        "Failed to refresh batch/species surfaces after partial staged overlay reset",
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
        if eligible_reset_set_ids:
            try:
                self.ui.mechanism_helpers.sync_mechanism_controls_to_focused_batch_set(
                    use_workspace=True
                )
            except Exception as exc:
                self._record_nonfatal_exception(
                    "Failed to resync focused mechanism controls after partial canonical explicit run",
                    exc,
                )
        self.flush_slider_plot_updates(
            force=True,
            cache_key=str(ctx.get("cache_key") or ""),
            request_id=ctx.get("request_id"),
            run_id=ctx.get("run_id"),
        )
        return tuple(eligible_reset_set_ids)

    def _try_handle_scoped_batch_failure(
        self,
        *,
        set_id: str,
        set_name: str,
        error_payload: Mapping[str, Any],
    ) -> bool:
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if not (isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel")):
            return False
        if bool(ctx.get("fast_mode")):
            return False
        queue_ids = [str(item) for item in (ctx.get("queue_ids") or ()) if str(item)]
        total = _safe_int(ctx.get("total"), default=max(1, len(queue_ids)))
        total = max(1, total or len(queue_ids) or 1)
        if total <= 1:
            return False

        sid = str(set_id or "")
        failure = coerce_simulation_failure(error_payload)
        completed_ids = set(str(item) for item in (ctx.get("completed_set_ids") or ()) if str(item))
        failed_set_ids = set(str(item) for item in (ctx.get("failed_set_ids") or ()) if str(item))
        failed_errors = dict(ctx.get("failed_set_errors") or {})
        completed_ids.add(sid)
        failed_set_ids.add(sid)
        failed_errors[sid] = failure
        pending_reset_ids = [
            str(item)
            for item in (ctx.get("pending_workspace_reset_set_ids") or ())
            if str(item) and str(item) != sid
        ]
        pending_reset_generations = dict(ctx.get("pending_dirty_reset_generation_by_set_id") or {})
        pending_reset_generations.pop(sid, None)
        self._record_scoped_batch_failure_cache_state(ctx, sid)
        ctx["completed_set_ids"] = sorted(completed_ids)
        ctx["failed_set_ids"] = sorted(failed_set_ids)
        ctx["failed_set_errors"] = failed_errors
        ctx["pending_workspace_reset_set_ids"] = pending_reset_ids
        ctx["pending_dirty_reset_generation_by_set_id"] = pending_reset_generations
        self._batch_run_context = dict(ctx)
        self._invalidate_preserved_pending_init_results_after_failed_run(ctx=ctx)
        ctx = dict(getattr(self, "_batch_run_context", {}) or ctx)

        completed_count = len(completed_ids)
        if completed_count < total:
            if total > 1:
                self.ui.run_ui.set_sim_progress_value(
                    max(0, min(100, int((completed_count / float(total)) * 100.0)))
                )
            label = str(set_name or sid or "set")
            self.ui.run_ui.set_status_text(f"Failed {label} ({completed_count}/{total})")
            return True

        ctx["active"] = False
        self._batch_run_context = dict(ctx)
        self._finalize_scoped_batch_success_subset(ctx)
        ctx = dict(getattr(self, "_batch_run_context", {}) or ctx)
        self._cleanup_parallel_batch_lane_pool_after_run(
            keep_lane_pool_alive=False,
            clear_pending_plot_updates=False,
        )
        self._simulation_running = False
        self._slider_simulation_active = False
        self.ui.slider.set_slider_triggered_simulation(False)
        self.ui.run_ui.set_sim_progress_value(100)
        self.ui.run_ui.set_run_button_enabled(True)
        self.ui.run_ui.set_stop_button_enabled(False)
        failed_count = len(failed_set_ids)
        self.ui.run_ui.set_status_text(f"Batch completed with {failed_count} failed set(s)")
        self._show_scoped_batch_failure_summary(
            failed_set_ids=failed_set_ids,
            failed_errors=failed_errors,
        )
        self._apply_explicit_failure_pending_replay_policy(fast_mode=False)
        return True

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
        sid = str(set_id or "")
        if completion_record is not None:
            meta = {
                "set_name": completion_record.set_name,
                "preview_owner_epoch": completion_record.preview_owner_epoch,
                "owner_epoch": completion_record.expected_owner_epoch,
                "generation": completion_record.generation,
            }
        else:
            meta = self._batch_parallel.active_request_metadata(sid)
        meta["set_name"] = str(meta.get("set_name") or sid)
        set_name = meta["set_name"]
        owner_epoch = meta.get("preview_owner_epoch")
        if owner_epoch is None:
            owner_epoch = meta.get("owner_epoch")
        self._batch_parallel.discard_request(sid)

        expected_owner_epoch = meta.get("owner_epoch")
        owner_epoch_mismatch = False
        if expected_owner_epoch is not None:
            try:
                owner_epoch_mismatch = int(outcome.owner_epoch) != int(expected_owner_epoch)
            except Exception:
                owner_epoch_mismatch = True

        if (
            int(outcome.run_id) != int(run_id)
            or int(outcome.request_id) != int(request_id)
            or str(outcome.set_id or "") != sid
            or owner_epoch_mismatch
        ):
            self._record_nonfatal_exception(
                (
                    "Rejected stale batch lane outcome "
                    f"(expected run_id={int(run_id)} request_id={int(request_id)} set_id={sid} owner_epoch={expected_owner_epoch}; "
                    f"got run_id={int(outcome.run_id)} request_id={int(outcome.request_id)} "
                    f"set_id={str(outcome.set_id or '')} owner_epoch={int(outcome.owner_epoch)})"
                ),
                RuntimeError("stale batch lane outcome"),
            )
            stale_failure = build_simulation_failure(
                "stale_batch_lane_outcome",
                "Rejected stale batch lane outcome.",
                details={
                    "expected_run_id": int(run_id),
                    "expected_request_id": int(request_id),
                    "expected_set_id": sid,
                    "expected_owner_epoch": expected_owner_epoch,
                    "actual_run_id": int(outcome.run_id),
                    "actual_request_id": int(outcome.request_id),
                    "actual_set_id": str(outcome.set_id or ""),
                    "actual_owner_epoch": int(outcome.owner_epoch),
                },
            )
            if self._try_handle_scoped_batch_failure(
                set_id=sid,
                set_name=set_name,
                error_payload=stale_failure,
            ):
                return True
            self._reset_parallel_batch_run_and_shutdown_lane_pool()
            self._simulation_running = False
            self._slider_simulation_active = False
            self.ui.slider.set_slider_triggered_simulation(False)
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            return False

        payload = dict(outcome.payload or {}) if outcome.payload is not None else None
        if (not bool(outcome.success)) or (
            isinstance(payload, dict) and payload.get("success") is False and isinstance(payload.get("error"), dict)
        ):
            error_payload = (
                dict(payload["error"])
                if isinstance(payload, dict) and isinstance(payload.get("error"), dict)
                else dict(outcome.failure or {})
            )
            if self._try_handle_scoped_batch_failure(
                set_id=sid,
                set_name=set_name,
                error_payload=error_payload,
            ):
                return True
            self._dispatch_simulation_error(
                error_payload,
                run_id=run_id,
                fast_mode=fast_mode,
                request_id=request_id,
                owner_epoch=owner_epoch,
                batch_set=set_name,
                batch_set_id=sid,
                cache_key=cache_key,
            )
            self._reset_parallel_batch_run_and_shutdown_lane_pool()
            return False

        if bool(getattr(self, "_debug_batch_parallel", False)):
            logger.info(
                "BATCH_PAR completion received run_id=%s request_id=%s set_id=%s source=%s completed_at=%.6f received_at=%.6f",
                int(run_id),
                int(request_id),
                sid,
                str(source),
                float(completed_ts if completed_ts is not None else -1.0),
                float(perf_counter()),
            )
        try:
            self._dispatch_simulation_complete(
                payload,
                run_id=run_id,
                fast_mode=fast_mode,
                request_id=request_id,
                owner_epoch=owner_epoch,
                batch_set=set_name,
                batch_set_id=sid,
                cache_key=cache_key,
            )
        except Exception as exc:
            self._record_nonfatal_exception(
                f"Unhandled exception while handling completed batch lane outcome (set_id={sid}, source={str(source)})",
                exc,
            )
            try:
                self._dispatch_simulation_error(
                    f"Simulation failed:\n\n{exc}",
                    run_id=run_id,
                    fast_mode=fast_mode,
                    request_id=request_id,
                    owner_epoch=owner_epoch,
                    batch_set=set_name,
                    batch_set_id=sid,
                    cache_key=cache_key,
                )
            except Exception as ui_exc:
                self._record_nonfatal_exception(
                    "Failed to surface simulation-complete handling failure to UI",
                    ui_exc,
                )
            self._reset_parallel_batch_run_and_shutdown_lane_pool()
            return False
        return True

    def _stop_batch_completion_poll_timer_if_idle(self) -> None:
        ctx = getattr(self, "_batch_run_context", {}) or {}
        active_parallel = bool(isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel"))
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
        ctx = context if isinstance(context, Mapping) else getattr(self, "_batch_run_context", {}) or {}
        primary_set = str(ctx.get("primary_set_id") or "").strip() if isinstance(ctx, Mapping) else ""
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
        initials_dict = self.ui.batch.batch_initials_for_row(int(row))
        pending_seed_for_set = pending_initial_seed_for_set(pending_init_seed, set_name=str(set_name))
        if pending_seed_for_set and (not bool(pending_init_applied)):
            for sp, val in pending_seed_for_set.items():
                float_val = _try_float(val)
                if float_val is None:
                    continue
                initials_dict[str(sp)] = float_val
        if bool(include_preview_initials):
            return self.ui.slider.preview_initials_for_row(int(row), initials_dict)
        return initials_dict

    def _invalidate_preserved_pending_init_results_after_failed_run(
        self,
        *,
        pending_init_applied: bool = False,
        ctx: Optional[Mapping[str, Any]] = None,
    ) -> None:
        policy_context = self._completion_policy_context_from_raw(ctx)
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
            mechanism = parse_dsl_to_mechanism(full_dsl, initials={}, units=units)
            if isinstance(getattr(mechanism, "metadata", None), dict):
                mechanism.metadata["wegscheider_cyclicity_enabled"] = bool(
                    self.ui.solver.wegscheider_cyclicity_enabled()
                )
            apply_parameter_algebra_to_mechanism(full_dsl, mechanism=mechanism, require_mutable=False)
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
        replay = self._pending_slider_preview_launch
        if not self._has_deferred_preview_replay_launch_state(replay):
            if replay.handoff_queued:
                self._pending_slider_handoff_queued = False
            return
        if replay.handoff_queued:
            replay = replace(replay, active=True, handoff_queued=False)
            self._run_state.pending_slider_preview_launch = replay
        worker = self._simulation_worker
        request_id = replay.request_id
        pending_target_set_ids = list(replay.target_set_ids)
        owner_request_id = self._preview_ownership.request_id
        ctx = getattr(self, "_batch_run_context", {}) or {}
        active_fast_parallel = bool(
            isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel") and ctx.get("fast_mode")
        )
        active_fast_request_id = None
        if active_fast_parallel:
            raw_active_fast_request_id = ctx.get("request_id") if isinstance(ctx, dict) else None
            try:
                if raw_active_fast_request_id is not None:
                    active_fast_request_id = int(raw_active_fast_request_id)
            except Exception:
                active_fast_request_id = None
        if request_id is not None and owner_request_id is not None and int(request_id) < int(owner_request_id):
            logger.debug(
                "Discarding stale slider simulation request (request_id=%s, preview_owner=%s)",
                request_id,
                owner_request_id,
            )
            self.clear_pending_slider_preview_replay(clear_plot_updates=False)
            return
        if request_id is None:
            request_id = self._next_slider_preview_request_id()
            self._run_state.pending_slider_preview_launch = replace(
                self._pending_slider_preview_launch,
                active=True,
                request_id=int(request_id),
                handoff_queued=False,
            )
        self._discarded_slider_preview_generation_id = None
        self.ui.slider.set_slider_triggered_simulation(True)

        def _defer_current_slider_replay() -> None:
            target_set_ids = [str(set_id) for set_id in pending_target_set_ids if str(set_id)]
            if not target_set_ids:
                try:
                    target_set_ids = [
                        str(set_id)
                        for set_id in (self.ui.batch.batch_set_ids_for_scope("selected") or ())
                        if str(set_id)
                    ]
                except Exception:
                    target_set_ids = []
            self.queue_pending_slider_preview_replay(
                target_set_ids=target_set_ids,
                request_id=int(request_id),
            )

        if self._has_active_explicit_simulation() and (
            worker is None or not getattr(worker, "_fast_mode", False)
        ):
            logger.debug("Full simulation in progress; deferring slider update")
            _defer_current_slider_replay()
            return

        # Guard against a race where a fast-mode worker has been constructed but has not yet
        # transitioned to "running" at the moment we check `isRunning()`. In that window,
        # starting another slider run will cancel the previous worker and can force a re-parse.
        if bool(getattr(self, "_simulation_running", False)):
            if (
                active_fast_parallel
                and request_id is not None
                and active_fast_request_id is not None
                and int(request_id) != int(active_fast_request_id)
            ):
                logger.debug(
                    "Superseding active fast parallel slider batch (active_request_id=%s, pending_request_id=%s)",
                    active_fast_request_id,
                    request_id,
                )
                _defer_current_slider_replay()
                supersede_result = self._supersede_parallel_batch_run_soft()
                try:
                    _cancelled, running = supersede_result
                except (TypeError, ValueError):
                    running = 0
                self._simulation_running = False
                self._slider_simulation_active = False
                if int(running) > 0:
                    return
                ctx = getattr(self, "_batch_run_context", {}) or {}
            else:
                logger.debug("Simulation already active; deferring slider update")
                _defer_current_slider_replay()
                return

        if self._worker_is_running(worker):
            logger.debug("Simulation currently running; deferring slider update")
            _defer_current_slider_replay()
            return

        active_fast_batch_work = bool(
            isinstance(ctx, dict)
            and bool(ctx.get("fast_mode"))
            and (bool(ctx.get("active")) or self._has_active_parallel_batch_work())
        )
        if active_fast_batch_work:
            logger.debug("Fast slider run currently running; deferring slider update")
            _defer_current_slider_replay()
            return
        self._prune_stopped_owned_simulation_workers()
        if self._has_running_owned_simulation_workers():
            logger.warning(
                "Slider-triggered run blocked while previous simulation worker shutdown remains in progress"
            )
            _defer_current_slider_replay()
            self.ui.slider.set_slider_triggered_simulation(False)
            self._simulation_running = False
            self._slider_simulation_active = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self.ui.run_ui.set_status_text("Cancelling previous simulation...")
            return

        selected_rows = self.ui.batch.batch_rows_for_scope("selected")
        selected_rows = self._slider_target_rows_for_dispatch(
            selected_rows,
            target_set_ids=pending_target_set_ids,
        )
        if not selected_rows:
            logger.debug(
                "Discarding slider replay launch with no resolvable target rows (target_set_ids=%s)",
                pending_target_set_ids,
            )
            self.ui.slider.set_slider_triggered_simulation(False)
            self.clear_pending_slider_preview_replay(clear_plot_updates=False)
            return

        uses_parallel_batch_runtime = self._slider_preview_uses_parallel_batch_runtime(selected_rows)
        preview_snapshot = self._slider_preview_runtime_snapshot(selected_rows)
        if bool(preview_snapshot.required) and not bool(preview_snapshot.ready):
            if uses_parallel_batch_runtime:
                self._ensure_parallel_batch_pool_eagerly_created(wait=False)
                self.ui.slider.set_slider_triggered_simulation(False)
                self.ui.run_ui.set_runtime_backed_run_controls_ready(False)
                self.ui.run_ui.set_status_text(str(preview_snapshot.message or "Preparing batch runtime..."))
                if bool(preview_snapshot.should_poll):
                    self.ui.run_ui.schedule_runtime_availability_refresh()
                    QtCore.QTimer.singleShot(50, self._run_simulation_from_slider)
                else:
                    self.clear_pending_slider_preview_replay(clear_plot_updates=False)
                    with suppress(Exception):
                        self.ui.slider.show_preview_unavailable_for_dirty_state(
                            str(preview_snapshot.message or "Batch runtime is not ready.")
                        )
                return
            self._ensure_interactive_simulation_runtime_available_for_mode(
                fast_mode=True,
                wait=False,
            )
            self.ui.slider.set_slider_triggered_simulation(False)
            self.ui.run_ui.set_status_text(str(preview_snapshot.message or "Preparing preview runtime..."))
            if bool(preview_snapshot.should_poll):
                QtCore.QTimer.singleShot(50, self._run_simulation_from_slider)
            else:
                self.clear_pending_slider_preview_replay(clear_plot_updates=False)
                with suppress(Exception):
                    self.ui.slider.show_preview_unavailable_for_dirty_state(
                        str(preview_snapshot.message or "Preview runtime is not ready.")
                    )
            return

        self._run_state.pending_slider_preview_launch = PendingSliderPreviewLaunchState()

        self._simulation_running = True
        self.ui.run_ui.set_stop_button_enabled(True)
        self._slider_simulation_active = True
        request_id = self._mark_request_started(int(request_id))

        logger.info("Starting slider-triggered simulation")
        self.ui.run_ui.set_status_text("Updating simulation...")
        self.ui.run_ui.set_sim_progress_value(0)

        self.run_simulation_internal(
            fast_mode=True,
            request_id=int(request_id),
            batch_rows=selected_rows,
            reuse_parallel_lane_pool=True,
        )

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
        ctx = getattr(self, "_batch_run_context", {}) or {}
        active_fast = bool(isinstance(ctx, dict) and ctx.get("fast_mode"))
        if isinstance(ctx, dict) and ctx.get("active"):
            ctx["active"] = False
            self._batch_run_context = dict(ctx)
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
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if not (isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel")):
            return

        rows = list(ctx.get("rows") or [])
        queue_ids = list(ctx.get("queue_ids") or [])
        queue_names = list(ctx.get("queue_names") or [])
        run_id = int(ctx.get("run_id") or 0)
        max_workers = max(1, int(ctx.get("effective_workers") or 1))

        prior_lane_pool_token = self._batch_parallel.lane_pool_token()
        try:
            existing_capacity = self._batch_parallel.current_max_workers
            pool_already_warmed = bool(
                self._batch_parallel.has_lane_pool()
                and (not self._batch_parallel.is_pool_stale)
                and existing_capacity is not None
                and int(existing_capacity) >= int(max_workers)
                and self._batch_parallel.has_ready_lane_pool(max_lanes=int(max_workers))
            )
            if pool_already_warmed:
                lane_pool = self._batch_parallel.ensure_lane_pool(max_lanes=int(max_workers))
            else:
                batch_runtime_state = self._batch_parallel.runtime_snapshot()
                runtime_snapshot = _runtime_readiness_snapshot(
                    mode="batch",
                    status="warming",
                    ready=False,
                    generation=int(getattr(batch_runtime_state, "current_generation", 0) or 0),
                    message="Preparing batch runtime...",
                    required=True,
                    controls_ready=False,
                    polling=True,
                )
                ctx["runtime_waiting"] = True
                ctx["active"] = False
                self._batch_run_context = dict(ctx)
                self._simulation_running = False
                self._slider_simulation_active = False
                if bool(ctx.get("fast_mode")):
                    self._ensure_parallel_batch_pool_eagerly_created(wait=False)
                    self.queue_pending_slider_preview_replay(
                        target_set_ids=[str(set_id) for set_id in queue_ids if str(set_id)],
                        request_id=int(ctx.get("request_id") or self._next_slider_preview_request_id()),
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
                return
        except Exception as exc:
            logger.warning("Parallel batch lane pool unavailable: %s", exc)
            ctx["runtime_waiting"] = True
            ctx["active"] = False
            self._batch_run_context = dict(ctx)
            self._simulation_running = False
            self._slider_simulation_active = False
            self.ui.run_ui.set_runtime_backed_run_controls_ready(False)
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self.ui.run_ui.set_sim_progress_value(0)
            self.ui.run_ui.set_status_text(f"Batch runtime readiness check failed: {exc}")
            if bool(ctx.get("fast_mode")):
                self.clear_pending_slider_preview_replay(clear_plot_updates=False)
                with suppress(Exception):
                    self.ui.slider.show_preview_unavailable_for_dirty_state(
                        f"Batch runtime readiness check failed: {exc}"
                    )
            return
        if ctx.get("runtime_waiting"):
            ctx.pop("runtime_waiting", None)
            self._batch_run_context = dict(ctx)
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
            request_id=int(ctx.get("request_id") or 0),
            fast_mode=bool(ctx.get("fast_mode")),
            queue_ids=[str(item) for item in queue_ids],
            queue_names=[str(item) for item in queue_names],
            keep_lane_pool_alive=bool(ctx.get("keep_lane_pool_alive")),
            preview_owner_epoch=ctx.get("preview_owner_epoch"),
            active_timeout_s=float(ctx.get("active_timeout_s") or 60.0),
            cache_key=str(ctx.get("cache_key") or ""),
        )

        mechanism_text = str(ctx.get("full_dsl") or "")
        solver_config = dict(ctx.get("solver_config") or {})
        t_end = float(ctx.get("t_end") or 0.0)
        request_id = int(ctx.get("request_id") or 0)
        simulation_plan_by_set_id = {
            str(set_id): dict(payload)
            for set_id, payload in dict(ctx.get("simulation_plan_by_set_id") or {}).items()
            if str(set_id) and isinstance(payload, dict)
        }
        mechanism_text_by_set_id = {
            str(set_id): str(text)
            for set_id, text in dict(ctx.get("mechanism_text_by_set_id") or {}).items()
            if str(set_id)
        }
        simulation_identity_by_set_id = {
            str(set_id): dict(payload)
            for set_id, payload in dict(ctx.get("simulation_identity_by_set_id") or {}).items()
            if str(set_id) and isinstance(payload, dict)
        }

        pending_seed = ctx.get("pending_init_seed") if isinstance(ctx, dict) else None
        pending_init_applied = bool(ctx.get("pending_init_applied", False))

        for idx, set_id in enumerate(queue_ids):
            if not (0 <= idx < len(rows)):
                continue
            row = int(rows[idx])
            set_name = str(queue_names[idx]) if 0 <= idx < len(queue_names) else str(set_id)
            try:
                initials_dict = self.ui.batch.batch_initials_for_row(row)
            except Exception as exc:
                self.ui.dialogs.message_box_warning(
                    "Invalid Initial Conditions",
                    f"Set '{set_name}' has invalid initial conditions:\n\n{exc}",
                )
                if bool(ctx.get("fast_mode")):
                    self._clear_failed_fast_preview_ownership()
                ctx["active"] = False
                self._batch_run_context = dict(ctx)
                self._shutdown_batch_lane_pool(force_terminate=True)
                self._simulation_running = False
                self.ui.run_ui.set_run_button_enabled(True)
                self.ui.run_ui.set_stop_button_enabled(False)
                self._slider_simulation_active = False
                self._invalidate_preserved_pending_init_results_after_failed_run(ctx=ctx)
                return
            pending_seed_for_set = pending_initial_seed_for_set(pending_seed, set_name=str(set_name))
            if pending_seed_for_set and (not pending_init_applied):
                for sp, val in pending_seed_for_set.items():
                    float_val = _try_float(val)
                    if float_val is None:
                        continue
                    initials_dict[str(sp)] = float_val
            if bool(ctx.get("fast_mode")):
                initials_dict = self.ui.slider.preview_initials_for_row(row, initials_dict)

            task = {
                "run_id": int(run_id),
                "request_id": int(request_id),
                "set_id": str(set_id),
                "set_name": str(set_name),
                "include_mechanism_in_result_payload": self._include_mechanism_in_result_payload(
                    fast_mode=bool(ctx.get("fast_mode")),
                    batch_set_id=str(set_id),
                    context=ctx,
                ),
            }
            plan_payload = simulation_plan_by_set_id.get(str(set_id))
            execution_request = _execution_request_payload_from_plan(plan_payload)
            if isinstance(execution_request, dict):
                if not bool(ctx.get("fast_mode")):
                    execution_request = dict(execution_request)
                    execution_request["prepared_payload"] = None
                    if isinstance(plan_payload, dict):
                        plan_payload = _simulation_plan_payload_with_execution_request(
                            plan_payload,
                            execution_request,
                            algebra_policy=SimulationAlgebraPolicy.BATCH_BEST_EFFORT,
                        )
            if plan_payload is None:
                plan_request = (
                    dict(execution_request)
                    if isinstance(execution_request, dict)
                    else SimulationExecutionRequest(
                        prepared_payload=None,
                        initials=dict(initials_dict),
                        t_span=(0.0, float(t_end)),
                        solver_config=dict(solver_config),
                        mechanism_text=mechanism_text_by_set_id.get(str(set_id), mechanism_text),
                        simulation_identity=simulation_identity_by_set_id.get(str(set_id)),
                    ).to_payload()
                )
                cache_identity_payload: Dict[str, Any] = {
                    "cache_key": str(ctx.get("cache_key") or ""),
                    "simulation_identity": dict(simulation_identity_by_set_id.get(str(set_id)) or {}),
                }
                preview_token = dict(ctx.get("preview_batch_cache_token_by_set_id") or {}).get(str(set_id))
                if preview_token:
                    cache_identity_payload["preview_batch_cache_token"] = str(preview_token)
                plan_payload = _new_simulation_plan_payload(
                    plan_request,
                    execution_mode="preview" if bool(ctx.get("fast_mode")) else "explicit",
                    algebra_policy=SimulationAlgebraPolicy.BATCH_BEST_EFFORT,
                    cache_identity_payload=cache_identity_payload,
                    cache_scope_payload={
                        "scope_identity": dict(ctx.get("scope_identity") or {}),
                        "queue_ids": [str(item) for item in queue_ids if str(item)],
                    },
                    metadata={
                        "set_id": str(set_id),
                        "set_name": str(set_name),
                        "fast_mode": bool(ctx.get("fast_mode")),
                    },
                )
            if isinstance(plan_payload, dict):
                plan_execution_request = (
                    dict(execution_request)
                    if isinstance(execution_request, dict)
                    else _execution_request_payload_from_plan(plan_payload)
                )
                if isinstance(plan_execution_request, dict):
                    plan_payload = _simulation_plan_payload_with_execution_request(
                        plan_payload,
                        plan_execution_request,
                        algebra_policy=SimulationAlgebraPolicy.BATCH_BEST_EFFORT,
                    )
                    plan_for_task = _simulation_plan(plan_payload)
                    if plan_for_task is not None:
                        plan_identity = plan_for_task.simulation_identity_payload()
                        if plan_identity:
                            simulation_identity_by_set_id[str(set_id)] = plan_identity
                task["simulation_plan"] = dict(plan_payload)
            sid = str(set_id)
            try:
                self._batch_parallel.submit_task(
                    task,
                    set_id=sid,
                    set_name=str(set_name),
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
                    run_id=int(run_id),
                    fast_mode=bool(ctx.get("fast_mode")),
                    request_id=int(request_id),
                    owner_epoch=ctx.get("preview_owner_epoch"),
                    batch_set=str(set_name),
                    batch_set_id=sid,
                    cache_key=str(ctx.get("cache_key") or ""),
                )
                return

        if not self._batch_parallel.has_active_requests():
            ctx["active"] = False
            self._batch_run_context = dict(ctx)
            self._shutdown_batch_lane_pool(force_terminate=False)
            self._simulation_running = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
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

    def _abort_for_unready_interactive_runtime(self, *, fast_mode: bool, context: Mapping[str, Any]) -> None:
        ctx = dict(context or {})
        ctx["active"] = False
        self._batch_run_context = ctx
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

    def _start_next_batch_simulation(self) -> None:
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if not (isinstance(ctx, dict) and ctx.get("active")):
            return

        rows = list(ctx.get("rows") or [])
        queue_ids = list(ctx.get("queue_ids") or [])
        queue_names = list(ctx.get("queue_names") or [])
        try:
            pos = int(ctx.get("pos", 0))
        except Exception:
            pos = 0

        if pos >= len(queue_ids) or pos >= len(rows):
            return

        set_id = str(queue_ids[pos])
        if 0 <= pos < len(queue_names):
            set_name = str(queue_names[pos])
        else:
            set_name = self.ui.batch.batch_set_name_for_id(set_id) or str(set_id)
        try:
            row = int(rows[pos])
        except Exception:
            row = 0

        try:
            initials_dict = self.ui.batch.batch_initials_for_row(row)
        except Exception as exc:
            try:
                self.ui.batch.batch_model_validate_rows([row])
            except Exception as validate_exc:
                self._record_nonfatal_exception(
                    f"Failed to validate batch model rows after invalid initials (row={int(row)} set_id={str(set_id)})",
                    validate_exc,
                )
            self.ui.dialogs.message_box_warning(
                "Invalid Initial Conditions",
                f"Set '{set_name}' has invalid initial conditions:\n\n{exc}",
            )
            if bool(ctx.get("fast_mode")):
                self._clear_failed_fast_preview_ownership()
            ctx["active"] = False
            self._batch_run_context = dict(ctx)
            self._simulation_running = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self._slider_simulation_active = False
            self._invalidate_preserved_pending_init_results_after_failed_run(ctx=ctx)
            return
        pending_seed = ctx.get("pending_init_seed") if isinstance(ctx, dict) else None
        pending_seed_for_set = pending_initial_seed_for_set(pending_seed, set_name=str(set_name))
        if pending_seed_for_set and (not bool(ctx.get("pending_init_applied", False))):
            for sp, val in pending_seed_for_set.items():
                float_val = _try_float(val)
                if float_val is None:
                    continue
                initials_dict[str(sp)] = float_val
        fast_mode = bool(ctx.get("fast_mode"))
        if bool(fast_mode):
            initials_dict = self.ui.slider.preview_initials_for_row(row, initials_dict)

        full_dsl = str(ctx.get("full_dsl") or "")
        solver_config = dict(ctx.get("solver_config") or {})
        t_end = float(ctx.get("t_end") or 0.0)
        request_id = int(ctx.get("request_id") or 0)
        cache_key = str(ctx.get("cache_key") or "")
        allow_batch_global_fallback = not bool(fast_mode)
        prepared_payload: Optional[Dict[str, Any]] = None
        execution_request: Optional[Dict[str, Any]] = None
        simulation_plan_by_set_id = {
            str(candidate_set_id): dict(payload)
            for candidate_set_id, payload in dict(ctx.get("simulation_plan_by_set_id") or {}).items()
            if str(candidate_set_id) and isinstance(payload, dict)
        }
        mechanism_text_by_set_id = {
            str(candidate_set_id): str(text)
            for candidate_set_id, text in dict(ctx.get("mechanism_text_by_set_id") or {}).items()
            if str(candidate_set_id)
        }
        mechanism_signature_by_set_id = {
            str(candidate_set_id): str(sig)
            for candidate_set_id, sig in dict(ctx.get("mechanism_signature_by_set_id") or {}).items()
            if str(candidate_set_id)
        }
        simulation_identity_by_set_id = {
            str(candidate_set_id): dict(payload)
            for candidate_set_id, payload in dict(ctx.get("simulation_identity_by_set_id") or {}).items()
            if str(candidate_set_id) and isinstance(payload, dict)
        }
        prepared_payloads = {
            str(candidate_set_id): dict(payload)
            for candidate_set_id, payload in dict(ctx.get("prepared_by_set_id") or {}).items()
            if str(candidate_set_id) and isinstance(payload, dict)
        }
        candidate = prepared_payloads.get(set_id) if bool(fast_mode) else None
        if allow_batch_global_fallback and candidate is None and bool(fast_mode):
            candidate = ctx.get("prepared")
        if isinstance(candidate, dict):
            prepared_payload = candidate
        candidate_plan = simulation_plan_by_set_id.get(set_id)
        plan_payload = _simulation_plan_payload(candidate_plan)
        candidate_request = _execution_request_payload_from_plan(candidate_plan)
        if allow_batch_global_fallback and candidate_request is None:
            candidate_plan = ctx.get("simulation_plan")
            plan_payload = _simulation_plan_payload(candidate_plan)
            candidate_request = _execution_request_payload_from_plan(candidate_plan)
        if isinstance(candidate_request, dict):
            execution_request = candidate_request
            if not bool(fast_mode):
                execution_request = dict(execution_request)
                execution_request["prepared_payload"] = None
                if isinstance(plan_payload, dict):
                    plan_payload = _simulation_plan_payload_with_execution_request(
                        plan_payload,
                        execution_request,
                        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
                    )
            initials_dict = dict(candidate_request.get("initials") or initials_dict)
            solver_config = dict(candidate_request.get("solver_config") or solver_config)
            request_t_span = candidate_request.get("t_span") or (0.0, t_end)
            try:
                t_end = float(request_t_span[1])
            except (TypeError, ValueError, IndexError):
                t_end = float(t_end)

        self._release_current_simulation_worker()

        self._run_sequence_id = int(getattr(self, "_run_sequence_id", 0)) + 1
        run_id = int(self._run_sequence_id)
        self._active_run_id = run_id

        if isinstance(execution_request, dict):
            if execution_request.get("prepared_payload") is not None:
                mechanism_text_for_worker = str(execution_request.get("mechanism_text") or "")
            else:
                mechanism_text_for_worker = str(
                    execution_request.get("mechanism_text") or mechanism_text_by_set_id.get(set_id, full_dsl)
                )
        else:
            mechanism_text_for_worker = mechanism_text_by_set_id.get(set_id, full_dsl)
        worker_signature = mechanism_signature_by_set_id.get(set_id)
        if bool(fast_mode) and prepared_payload is None:
            overrides = self.ui.mechanism.slider_overrides(set_id=set_id)
            if overrides:
                try:
                    mechanism_text_for_worker = self.ui.mechanism.apply_parameter_overrides_to_dsl(
                        mechanism_text_for_worker,
                        overrides,
                    )
                except Exception as exc:
                    self._record_nonfatal_exception(
                        "Failed to apply slider overrides to worker DSL; falling back to baseline DSL",
                        exc,
                    )
                    mechanism_text_for_worker = full_dsl
            if len(queue_ids) <= 1:
                cache_key = build_fallback_cache_key(
                    str(mechanism_text_for_worker),
                    float(t_end),
                    dict(solver_config or {}),
                    simulation_identity=simulation_identity_by_set_id.get(set_id),
                )
                if isinstance(ctx, dict):
                    ctx["cache_key"] = str(cache_key)
                    self._batch_run_context = dict(ctx)

        if plan_payload is None:
            plan_request = SimulationExecutionRequest(
                prepared_payload=dict(prepared_payload) if isinstance(prepared_payload, dict) else None,
                initials=dict(initials_dict),
                t_span=(0.0, float(t_end)),
                solver_config=dict(solver_config),
                mechanism_text=str(mechanism_text_for_worker),
                simulation_identity=simulation_identity_by_set_id.get(set_id),
            ).to_payload()
            cache_identity_payload: Dict[str, Any] = {
                "cache_key": str(cache_key),
                "simulation_identity": dict(simulation_identity_by_set_id.get(set_id) or {}),
            }
            preview_token = dict(ctx.get("preview_batch_cache_token_by_set_id") or {}).get(str(set_id))
            if preview_token:
                cache_identity_payload["preview_batch_cache_token"] = str(preview_token)
            plan_payload = _new_simulation_plan_payload(
                plan_request,
                execution_mode="preview" if bool(fast_mode) else "explicit",
                algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
                cache_identity_payload=cache_identity_payload,
                cache_scope_payload={
                    "scope_identity": dict(ctx.get("scope_identity") or {}),
                    "queue_ids": [str(item) for item in queue_ids if str(item)],
                },
                metadata={
                    "set_id": str(set_id),
                    "set_name": str(set_name),
                    "fast_mode": bool(fast_mode),
                },
            )

        if isinstance(plan_payload, dict):
            plan_execution_request = (
                dict(execution_request)
                if isinstance(execution_request, dict)
                else _execution_request_payload_from_plan(plan_payload)
            )
            if isinstance(plan_execution_request, dict):
                plan_payload = _simulation_plan_payload_with_execution_request(
                    plan_payload,
                    plan_execution_request,
                    algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
                )
            plan_for_worker = _simulation_plan(plan_payload)
            if plan_for_worker is not None:
                plan_cache_key = plan_for_worker.cache_key()
                if plan_cache_key:
                    cache_key = str(plan_cache_key)
                plan_identity = plan_for_worker.simulation_identity_payload()
                if plan_identity:
                    simulation_identity_by_set_id[set_id] = plan_identity

        include_mechanism_in_result_payload = self._include_mechanism_in_result_payload(
            fast_mode=bool(fast_mode),
            batch_set_id=set_id,
            context=ctx,
        )

        contained_owner = None
        if isinstance(plan_payload, dict):
            try:
                from kindred.core.simulation_containment import build_contained_simulation_plan_payload
                from kindred.gui.simulation_worker import ContainedSimulationWorker

                contained_plan_payload = build_contained_simulation_plan_payload(plan_payload)
                contained_owner = self._acquire_ready_contained_simulation_owner_for_plan(
                    fast_mode=bool(fast_mode),
                    simulation_plan_payload=contained_plan_payload,
                )
                if contained_owner is None:
                    self._abort_for_unready_interactive_runtime(
                        fast_mode=bool(fast_mode),
                        context=ctx,
                    )
                    return
                self._simulation_worker = ContainedSimulationWorker(
                    owner=contained_owner,
                    simulation_plan_payload=contained_plan_payload,
                    include_mechanism_in_result_payload=include_mechanism_in_result_payload,
                    parent=self,
                )
            except Exception as exc:
                if contained_owner is not None:
                    try:
                        self._runtime_application.release_owner(contained_owner, kill=False)
                    except Exception as release_exc:
                        self._record_nonfatal_exception(
                            "Failed to release acquired simulation runtime owner after worker construction failure",
                            release_exc,
                        )
                self._dispatch_simulation_error(
                    simulation_failure_from_exception(exc, kind="simulation_containment_payload"),
                    run_id=int(run_id),
                    fast_mode=bool(fast_mode),
                    request_id=int(request_id),
                    owner_epoch=ctx.get("preview_owner_epoch"),
                    batch_set=str(set_name),
                    batch_set_id=str(set_id),
                    cache_key=str(cache_key),
                )
                return
        else:
            from kindred.gui.simulation_worker import SimulationWorker

            self._simulation_worker = SimulationWorker(
                mechanism_text=mechanism_text_for_worker,
                initials=initials_dict,
                t_span=(0.0, t_end),
                solver_config=solver_config,
                parent=self,
                prepared=prepared_payload,
                include_mechanism_in_result_payload=include_mechanism_in_result_payload,
            )
        self._simulation_worker._run_id = run_id  # type: ignore[attr-defined]
        self._simulation_worker._request_id = int(request_id)  # type: ignore[attr-defined]
        self._simulation_worker._fast_mode = bool(fast_mode)  # type: ignore[attr-defined]
        self._simulation_worker._batch_set_name = set_name  # type: ignore[attr-defined]
        self._simulation_worker._batch_set_id = set_id  # type: ignore[attr-defined]
        self._simulation_worker._batch_cache_key = cache_key  # type: ignore[attr-defined]
        if worker_signature:
            self._simulation_worker._batch_mechanism_signature = str(worker_signature)  # type: ignore[attr-defined]
        if isinstance(plan_payload, dict):
            self._simulation_worker._simulation_plan = dict(plan_payload)  # type: ignore[attr-defined]

        self._connect_simulation_worker_application_signals(
            self._simulation_worker,
            run_id=int(run_id),
            fast_mode=bool(fast_mode),
            request_id=int(request_id),
            owner_epoch=ctx.get("preview_owner_epoch"),
            set_name=str(set_name),
            set_id=str(set_id),
            cache_key=str(cache_key),
        )

        total = len(queue_ids)
        self.ui.run_ui.set_status_text(f"Running {set_name} ({pos + 1}/{total})...")
        self._simulation_worker.start()

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
        if bool(fast_mode):
            active_fast_worker = False
            worker = getattr(self, "_simulation_worker", None)
            if worker is not None and hasattr(worker, "isRunning"):
                try:
                    active_fast_worker = self._worker_is_running(worker) and bool(getattr(worker, "_fast_mode", False))
                except Exception:
                    active_fast_worker = False

            ctx = getattr(self, "_batch_run_context", {}) or {}
            active_fast_parallel = bool(isinstance(ctx, dict) and ctx.get("active") and ctx.get("fast_mode"))
            if active_fast_worker or active_fast_parallel:
                if bool(runtime_readiness_only):
                    return
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
                    request_id = int(request_id)
                    self._pending_slider_sim_request_id = int(request_id)
                return

        def _clear_slider_triggered_preflight_state() -> None:
            if bool(fast_mode):
                self.ui.slider.set_slider_triggered_simulation(False)

        if bool(runtime_readiness_only):
            request_id = int(request_id or 0)
        elif request_id is None:
            request_id = self._next_sim_request_id()
        else:
            request_id = self._mark_request_started(int(request_id))
        if batch_rows is None:
            batch_rows = self.ui.batch.batch_rows_for_scope("selected")
        row_count = int(self.ui.batch.batch_store_row_count())
        batch_rows = [int(r) for r in (batch_rows or []) if 0 <= int(r) < int(row_count)]
        if not batch_rows:
            if int(row_count) > 0:
                batch_rows = [0]
            else:
                if bool(runtime_readiness_only):
                    return
                self.ui.dialogs.message_box_warning("No Sets", "Add at least one set before running.")
                if bool(fast_mode):
                    self._clear_failed_fast_preview_ownership()
                self._simulation_running = False
                self.ui.run_ui.set_run_button_enabled(True)
                self.ui.run_ui.set_stop_button_enabled(False)
                _clear_slider_triggered_preflight_state()
                if not bool(fast_mode):
                    self._requeue_preserved_pending_slider_replay_after_preflight_abort()
                return
        invalid = self.ui.batch.batch_model_validate_rows(batch_rows)
        if invalid:
            if bool(runtime_readiness_only):
                return
            examples = sorted(invalid)[:8]
            details = "\n".join(f"  • row {r+1}: {sp}" for r, sp in examples)
            more = "" if len(invalid) <= len(examples) else f"\n  • ... and {len(invalid) - len(examples)} more"
            self._invalidate_preserved_pending_init_results_after_failed_run(
                ctx=getattr(self, "_batch_run_context", {}) or None,
            )
            self.ui.dialogs.message_box_warning(
                "Invalid Initial Conditions",
                "Fix invalid numeric cells in the Initial Conditions table before running:\n\n" + details + more,
            )
            if bool(fast_mode):
                self._clear_failed_fast_preview_ownership()
            self._simulation_running = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            _clear_slider_triggered_preflight_state()
            if not bool(fast_mode):
                self._requeue_preserved_pending_slider_replay_after_preflight_abort()
            return

        # Uncommitted slider workspace is preview-only. Explicit runs must use
        # the canonical mechanism/editor state unless the user commits first.
        any_slider_workspace = bool(self.ui.mechanism.has_slider_overrides())
        has_slider_overrides = bool(fast_mode) and bool(any_slider_workspace)
        primary = self.ui.batch.batch_preferred_primary_set_id(batch_rows)
        primary_set_id = str(primary) if primary is not None else None

        owner_reactions_text_raw = self.ui.mechanism.mechanism_reactions_text_raw()
        owner_state_network_dsl_raw = self.ui.mechanism.mechanism_state_network_dsl_raw()
        reactions_text_raw = owner_reactions_text_raw
        if has_slider_overrides:
            reactions_text_raw = self.ui.mechanism.apply_overrides_to_text(
                reactions_text_raw,
                set_id=primary_set_id,
            )

        pending_init_seed: Dict[str, Dict[str, float]] = {}
        pending_init_rewrite: Optional[str] = None
        pending_init_applied = False
        migrated = reactions_text_raw
        try:
            pending_init_seed, migrated = migrate_reaction_dsl_initial_concentration_sets(
                reactions_text_raw,
                default_set_name="set1",
            )
            if pending_init_seed:
                pending_init_rewrite = migrated
        except Exception:
            pending_init_seed = {}
            pending_init_rewrite = None
            migrated = reactions_text_raw

        if (
            (not bool(fast_mode))
            and pending_init_seed
            and pending_init_rewrite
            and not bool(runtime_readiness_only)
        ):
            try:
                pending_init_applied = bool(
                    self.ui.mechanism_helpers.apply_pending_init_migration(
                        seed_sets=dict(pending_init_seed),
                        rewrite=str(pending_init_rewrite),
                    )
                )
            except Exception:
                pending_init_applied = False
            if pending_init_applied:
                reactions_text_raw = str(pending_init_rewrite)
                imported_names = [str(name) for name in pending_init_seed.keys() if str(name)]
                materialized_names = list(self.ui.batch.batch_store_set_names())
                materialized_rows: List[int] = []
                for imported_name in imported_names:
                    try:
                        row_idx = materialized_names.index(str(imported_name))
                    except ValueError:
                        continue
                    materialized_rows.append(int(row_idx))
                if materialized_rows:
                    batch_rows = materialized_rows
                    primary = self.ui.batch.batch_preferred_primary_set_id(batch_rows)
                    primary_set_id = str(primary) if primary is not None else None

        names = list(self.ui.batch.batch_store_set_names())
        queue_names = [str(names[r]) for r in batch_rows if 0 <= int(r) < len(names)]
        queue_ids = [
            str(self.ui.batch.batch_set_id_for_row(int(r)) or str(names[int(r)]))
            for r in batch_rows
            if 0 <= int(r) < len(names)
        ]
        preview_owner_epoch = None
        if bool(fast_mode) and not bool(runtime_readiness_only):
            preview_ownership = self._claim_preview_ownership(
                request_id=int(request_id),
                target_set_ids=queue_ids,
            )
            preview_owner_epoch = int(preview_ownership.epoch)
        elif (not bool(fast_mode)) and (not bool(runtime_readiness_only)):
            self._clear_preview_ownership()
        reactions_text = strip_reaction_dsl_initial_concentrations(
            reactions_text_raw if pending_init_applied else migrated
        )
        state_network_dsl = owner_state_network_dsl_raw
        if has_slider_overrides:
            state_network_dsl = self.ui.mechanism.apply_overrides_to_state_network_dsl(
                state_network_dsl,
                set_id=primary_set_id,
            )

        full_dsl = reactions_text
        if state_network_dsl.strip():
            full_dsl += "\n\n# State Network\n" + state_network_dsl
        if has_slider_overrides:
            full_dsl = self._apply_parameter_override_fallback_to_dsl(full_dsl, set_id=primary_set_id)
        owner_reactions_text = strip_reaction_dsl_initial_concentrations(owner_reactions_text_raw)
        owner_full_dsl = owner_reactions_text
        if owner_state_network_dsl_raw.strip():
            owner_full_dsl += "\n\n# State Network\n" + owner_state_network_dsl_raw

        if not full_dsl.strip():
            if bool(runtime_readiness_only):
                return
            self._invalidate_preserved_pending_init_results_after_failed_run(
                pending_init_applied=bool(pending_init_applied)
            )
            self.ui.dialogs.message_box_warning(
                "No Mechanism",
                "Please define reactions or state network in the Mechanism editor first.",
            )
            if bool(fast_mode):
                self._clear_failed_fast_preview_ownership()
            self.ui.run_ui.set_status_text("Ready")
            self._simulation_running = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self.ui.batch.update_batch_row_controls_state()
            _clear_slider_triggered_preflight_state()
            self._requeue_preserved_pending_slider_replay_after_preflight_abort()
            return

        solver_grid_context = build_fast_preview_solver_grid_context(
            initial_solver_name=self.ui.solver.initial_solver_name(),
            num_points=int(self.ui.solver.num_points_spinbox_value()),
            fast_mode=bool(fast_mode),
            slider_points_override=self.ui.mechanism.mechanism_slider_points_value(),
            slider_solver_override=self.ui.mechanism.mechanism_slider_solver_value(),
            slider_drag_active=bool(self.ui.slider.slider_drag_active()),
            last_slider_change_name=str(self.ui.slider.last_slider_change_name() or ""),
        )
        solver_label = str(solver_grid_context.get("solver_label") or "")
        solver = str(solver_grid_context.get("solver") or "")
        solver_warning = solver_grid_context.get("solver_warning")
        if solver_warning and not bool(runtime_readiness_only):
            logger.warning("Solver normalization: %s (requested=%r)", solver_warning, solver_label)
            with suppress(Exception):
                self.ui.run_ui.set_status_text(str(solver_warning))
        rtol = self.ui.solver.initial_rtol() or 1e-6
        atol = self.ui.solver.initial_atol() or 1e-12
        temperature_K = float(self.ui.solver.temperature_spinbox_value())
        T_override = self.ui.solver.dsl_global_temperature_K(full_dsl)
        if T_override is not None:
            temperature_K = float(T_override)

        prepared_payload: Optional[Dict[str, Any]] = None
        prepared_payload_by_set_id: Dict[str, Dict[str, Any]] = {}
        execution_prepared_payload_by_set_id: Dict[str, Dict[str, Any]] = {}
        owner_parameter_names_by_set_id: Dict[str, list[str]] = {}
        slider_runtime = None
        if bool(fast_mode):
            target_runtime_set_ids = list(queue_ids)
            if (not target_runtime_set_ids) and primary_set_id:
                target_runtime_set_ids = [str(primary_set_id)]
            for set_id in target_runtime_set_ids:
                runtime_parameter_names = self._slider_runtime_parameter_names(set_id=str(set_id))
                owner_parameter_names_by_set_id[str(set_id)] = list(runtime_parameter_names)
            if primary_set_id:
                prepared_payload = prepared_payload_by_set_id.get(str(primary_set_id))
            if prepared_payload is None and prepared_payload_by_set_id:
                prepared_payload = dict(next(iter(prepared_payload_by_set_id.values())))

        if (not bool(runtime_readiness_only)) and (
            (not bool(fast_mode)) or (not list(self.ui.batch.batch_store_visible_species()))
        ):
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

        n_points = int((solver_grid_context.get("grid") or {}).get("N") or 0)
        if fast_mode:
            logger.debug("Fast mode: using %s points and %s solver for slider update", n_points, solver)

        solver_config = {
            "solver": solver,
            "solver_label": solver_label,
            "solver_warning": str(solver_warning) if solver_warning else None,
            "rtol": rtol,
            "atol": atol,
            "grid": {"N": n_points},
            "temperature_K": temperature_K,
            "use_sparse_jacobian": bool(self.ui.solver.use_sparse_jacobian()),
            "wegscheider_cyclicity_enabled": bool(self.ui.solver.wegscheider_cyclicity_enabled()),
        }
        try:
            t_end = float(self.ui.solver.parse_sim_time_seconds())
        except ValueError as exc:
            if bool(runtime_readiness_only):
                return
            self._invalidate_preserved_pending_init_results_after_failed_run(
                pending_init_applied=bool(pending_init_applied)
            )
            self.ui.dialogs.message_box_warning("Invalid t_end", f"Fix t_end before running:\n\n{exc}")
            if bool(fast_mode):
                self._clear_failed_fast_preview_ownership()
            self._simulation_running = False
            try:
                self.ui.run_ui.set_run_button_enabled(True)
            except Exception as exc:
                self._record_nonfatal_exception("Failed to re-enable Run button after invalid t_end", exc)
            try:
                self.ui.run_ui.set_stop_button_enabled(False)
            except Exception as exc:
                self._record_nonfatal_exception("Failed to disable Stop button after invalid t_end", exc)
            self._slider_simulation_active = False
            _clear_slider_triggered_preflight_state()
            if not bool(fast_mode):
                self._requeue_preserved_pending_slider_replay_after_preflight_abort()
            return

        simulation_plan_by_set_id: Dict[str, Dict[str, Any]] = {}
        mechanism_text_by_set_id: Dict[str, str] = {}
        mechanism_signature_by_set_id: Dict[str, str] = {}
        preview_batch_cache_token_by_set_id: Dict[str, str] = {}
        for index, set_id in enumerate(queue_ids):
            token = ""
            if bool(fast_mode) and index < len(batch_rows):
                try:
                    token = self.ui.slider.preview_batch_cache_token([int(batch_rows[index])])
                except Exception:
                    token = ""
            preview_batch_cache_token_by_set_id[str(set_id)] = str(token or "")

        simulation_identity_by_set_id: Dict[str, dict[str, Any]] = {}
        for index, set_id in enumerate(queue_ids):
            identity = self._simulation_identity_for_set(
                set_id=str(set_id),
                solver_config=solver_config,
                t_end=float(t_end),
                preview_batch_cache_token=preview_batch_cache_token_by_set_id.get(str(set_id), ""),
                fast_mode=bool(fast_mode),
            )
            simulation_identity_by_set_id[str(set_id)] = identity.to_payload()
            if index >= len(batch_rows):
                continue
            row = int(batch_rows[index])
            set_name = str(queue_names[index]) if index < len(queue_names) else str(set_id)
            request_mechanism_text = self._request_mechanism_text_for_set(
                set_id=str(set_id),
                has_slider_overrides=has_slider_overrides,
            )
            mechanism_text_by_set_id[str(set_id)] = str(request_mechanism_text)
            prepared_execution_payload = execution_prepared_payload_by_set_id.get(str(set_id))
            if bool(fast_mode):
                if isinstance(prepared_execution_payload, dict):
                    try:
                        initials_dict = self._resolved_initials_for_batch_row(
                            row=row,
                            set_name=set_name,
                            pending_init_seed=pending_init_seed,
                            pending_init_applied=False,
                            include_preview_initials=True,
                        )
                    except Exception as exc:
                        if bool(runtime_readiness_only):
                            return
                        self.ui.dialogs.message_box_warning(
                            "Invalid Initial Conditions",
                            f"Set '{set_name}' has invalid initial conditions:\n\n{exc}",
                        )
                        self._clear_failed_fast_preview_ownership()
                        self._simulation_running = False
                        self.ui.run_ui.set_run_button_enabled(True)
                        self.ui.run_ui.set_stop_button_enabled(False)
                        self._slider_simulation_active = False
                        _clear_slider_triggered_preflight_state()
                        return
                    identity = self._simulation_identity_for_set(
                        set_id=str(set_id),
                        solver_config=solver_config,
                        t_end=float(t_end),
                        canonical_initials_fingerprint=canonical_initials_fingerprint(initials_dict),
                        preview_batch_cache_token=preview_batch_cache_token_by_set_id.get(str(set_id), ""),
                        fast_mode=bool(fast_mode),
                    )
                    simulation_identity_by_set_id[str(set_id)] = identity.to_payload()
                    mechanism_signature_by_set_id[str(set_id)] = batch_mechanism_signature(
                        simulation_identity=identity,
                    )
                    request_payload = SimulationExecutionRequest(
                        prepared_payload=dict(prepared_execution_payload),
                        initials=dict(initials_dict),
                        t_span=(0.0, float(t_end)),
                        solver_config=dict(solver_config),
                        mechanism_text=str(request_mechanism_text),
                        simulation_identity=identity.to_payload(),
                        parameter_overrides=self._slider_execution_parameter_values(set_id=str(set_id)),
                    ).to_payload()
                    preview_token = preview_batch_cache_token_by_set_id.get(str(set_id), "")
                    cache_identity_payload: Dict[str, Any] = {
                        "cache_key": "",
                        "simulation_identity": identity.to_payload(),
                    }
                    if preview_token:
                        cache_identity_payload["preview_batch_cache_token"] = str(preview_token)
                    simulation_plan_by_set_id[str(set_id)] = _new_simulation_plan_payload(
                        request_payload,
                        execution_mode="preview",
                        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
                        cache_identity_payload=cache_identity_payload,
                        metadata={
                            "set_id": str(set_id),
                            "set_name": set_name,
                            "fast_mode": True,
                        },
                    )
                else:
                    mechanism_signature_by_set_id[str(set_id)] = batch_mechanism_signature(
                        mechanism_text=str(request_mechanism_text),
                        temperature_K=float(solver_config.get("temperature_K") or 298.15),
                        use_sparse_jacobian=bool(
                            solver_config.get(
                                "use_sparse_jacobian",
                                PROJECT_DEFAULTS["use_sparse_jacobian"],
                            )
                        ),
                        wegscheider_cyclicity_enabled=bool(
                            solver_config.get(
                                "wegscheider_cyclicity_enabled",
                                PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"],
                            )
                        ),
                    )
                    try:
                        initials_dict = self._resolved_initials_for_batch_row(
                            row=row,
                            set_name=set_name,
                            pending_init_seed=pending_init_seed,
                            pending_init_applied=False,
                            include_preview_initials=True,
                        )
                    except Exception as exc:
                        if bool(runtime_readiness_only):
                            return
                        self.ui.dialogs.message_box_warning(
                            "Invalid Initial Conditions",
                            f"Set '{set_name}' has invalid initial conditions:\n\n{exc}",
                        )
                        self._clear_failed_fast_preview_ownership()
                        self._simulation_running = False
                        self.ui.run_ui.set_run_button_enabled(True)
                        self.ui.run_ui.set_stop_button_enabled(False)
                        self._slider_simulation_active = False
                        _clear_slider_triggered_preflight_state()
                        return
                    identity = self._simulation_identity_for_set(
                        set_id=str(set_id),
                        solver_config=solver_config,
                        t_end=float(t_end),
                        canonical_initials_fingerprint=canonical_initials_fingerprint(initials_dict),
                        preview_batch_cache_token=preview_batch_cache_token_by_set_id.get(str(set_id), ""),
                        fast_mode=bool(fast_mode),
                    )
                    simulation_identity_by_set_id[str(set_id)] = identity.to_payload()
                    request_payload = SimulationExecutionRequest(
                        prepared_payload=None,
                        initials=dict(initials_dict),
                        t_span=(0.0, float(t_end)),
                        solver_config=dict(solver_config),
                        mechanism_text=str(request_mechanism_text),
                        simulation_identity=identity.to_payload(),
                        parameter_overrides=self._slider_execution_parameter_values(set_id=str(set_id)),
                    ).to_payload()
                    preview_token = preview_batch_cache_token_by_set_id.get(str(set_id), "")
                    cache_identity_payload: Dict[str, Any] = {
                        "cache_key": "",
                        "simulation_identity": identity.to_payload(),
                    }
                    if preview_token:
                        cache_identity_payload["preview_batch_cache_token"] = str(preview_token)
                    simulation_plan_by_set_id[str(set_id)] = _new_simulation_plan_payload(
                        request_payload,
                        execution_mode="preview",
                        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
                        cache_identity_payload=cache_identity_payload,
                        metadata={
                            "set_id": str(set_id),
                            "set_name": set_name,
                            "fast_mode": True,
                        },
                    )
                continue

            try:
                initials_dict = self._resolved_initials_for_batch_row(
                    row=row,
                    set_name=set_name,
                    pending_init_seed=pending_init_seed,
                    pending_init_applied=False,
                    include_preview_initials=False,
                )
            except Exception as exc:
                if bool(runtime_readiness_only):
                    return
                self._invalidate_preserved_pending_init_results_after_failed_run(
                    pending_init_applied=bool(pending_init_applied)
                )
                self.ui.dialogs.message_box_warning(
                    "Invalid Initial Conditions",
                    f"Set '{set_name}' has invalid initial conditions:\n\n{exc}",
                )
                self._simulation_running = False
                self.ui.run_ui.set_run_button_enabled(True)
                self.ui.run_ui.set_stop_button_enabled(False)
                self._slider_simulation_active = False
                self._requeue_preserved_pending_slider_replay_after_preflight_abort()
                return

            identity = self._simulation_identity_for_set(
                set_id=str(set_id),
                solver_config=solver_config,
                t_end=float(t_end),
                canonical_initials_fingerprint=canonical_initials_fingerprint(initials_dict),
                preview_batch_cache_token=preview_batch_cache_token_by_set_id.get(str(set_id), ""),
                fast_mode=bool(fast_mode),
            )
            simulation_identity_by_set_id[str(set_id)] = identity.to_payload()
            mechanism_signature_by_set_id[str(set_id)] = batch_mechanism_signature(
                simulation_identity=identity,
            )

            request_payload = SimulationExecutionRequest(
                prepared_payload=dict(prepared_execution_payload) if isinstance(prepared_execution_payload, dict) else None,
                initials=dict(initials_dict),
                t_span=(0.0, float(t_end)),
                solver_config=dict(solver_config),
                mechanism_text=str(request_mechanism_text),
                simulation_identity=identity.to_payload(),
            ).to_payload()
            simulation_plan_by_set_id[str(set_id)] = _new_simulation_plan_payload(
                request_payload,
                execution_mode="explicit",
                algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
                cache_identity_payload={
                    "cache_key": "",
                    "simulation_identity": identity.to_payload(),
                },
                metadata={
                    "set_id": str(set_id),
                    "set_name": set_name,
                    "fast_mode": False,
                },
            )

        scope_identity = SimulationScopeIdentity.build(
            queue_ids=queue_ids,
            identity_by_set_id={
                set_id: payload
                for set_id, payload in simulation_identity_by_set_id.items()
            },
        )
        cache_key = self.ui.batch.batch_cache_key(scope_identity=scope_identity)
        if not bool(runtime_readiness_only):
            if bool(fast_mode):
                self._batch_cache.active_preview_cache_key = cache_key
            else:
                self._batch_cache.active_cache_key = cache_key
                self._batch_cache.active_cache_preview_token = None

        set_name_by_set_id = {
            str(set_id): str(queue_names[index]) if index < len(queue_names) else str(set_id)
            for index, set_id in enumerate(queue_ids)
        }
        for set_id, plan_payload in list(simulation_plan_by_set_id.items()):
            preview_token = preview_batch_cache_token_by_set_id.get(str(set_id), "")
            cache_identity_payload: Dict[str, Any] = {
                "cache_key": cache_key,
                "simulation_identity": dict(simulation_identity_by_set_id.get(str(set_id)) or {}),
            }
            if preview_token:
                cache_identity_payload["preview_batch_cache_token"] = str(preview_token)
            request_payload = _execution_request_payload_from_plan(plan_payload)
            if request_payload is None:
                continue
            metadata: Dict[str, Any] = {
                "set_id": str(set_id),
                "set_name": set_name_by_set_id.get(str(set_id), str(set_id)),
                "fast_mode": bool(fast_mode),
            }
            if bool(fast_mode):
                parameter_names = owner_parameter_names_by_set_id.get(str(set_id))
                if parameter_names is None:
                    parameter_names = self._slider_runtime_parameter_names(set_id=str(set_id))
                metadata["contained_owner_identity"] = self._preview_contained_owner_identity(
                    owner_mechanism_text=str(request_payload.get("mechanism_text") or owner_full_dsl),
                    solver_config=solver_config,
                    t_end=float(t_end),
                    set_id=str(set_id),
                    parameter_names=parameter_names,
                    simulation_identity=simulation_identity_by_set_id.get(str(set_id)),
                )
            else:
                metadata["contained_owner_identity"] = self._ordinary_contained_owner_identity(
                    owner_mechanism_text=owner_full_dsl,
                    solver_config=solver_config,
                    t_end=float(t_end),
                    set_id=str(set_id),
                    simulation_identity=simulation_identity_by_set_id.get(str(set_id)),
                )
            simulation_plan_by_set_id[str(set_id)] = _new_simulation_plan_payload(
                request_payload,
                execution_mode="preview" if bool(fast_mode) else "explicit",
                algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
                cache_identity_payload=cache_identity_payload,
                cache_scope_payload={
                    "scope_identity": scope_identity.to_payload(),
                    "queue_ids": [str(queue_id) for queue_id in queue_ids],
                },
                metadata=metadata,
            )

        if bool(runtime_readiness_only):
            from kindred.core.simulation_containment import build_contained_simulation_plan_payload

            contained_payloads = [
                build_contained_simulation_plan_payload(plan_payload)
                for plan_payload in simulation_plan_by_set_id.values()
                if isinstance(plan_payload, dict)
            ]
            if contained_payloads:
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
                        wait=bool(runtime_readiness_wait),
                    )
                else:
                    self._runtime_application.ensure_ready_many(
                        mode=mode,
                        payloads=[dict(payload) for payload in contained_payloads],
                        owner_factory=_owner_factory,
                        wait=bool(runtime_readiness_wait),
                    )
            return

        if bool(reuse_parallel_lane_pool):
            self._supersede_parallel_batch_run_soft()
        else:
            self._shutdown_batch_lane_pool(force_terminate=True)

        self._release_current_simulation_worker()

        run_start_cache_decision = self._completion_policy.build_run_start_cache_decision(
            fast_mode=bool(fast_mode),
            queue_ids=tuple(queue_ids),
        )
        explicit_valid_set_ids = run_start_cache_decision.explicit_cache_valid_set_ids
        if not bool(fast_mode):
            self._batch_cache.active_cache_preview_scope_set_ids = None
        if not bool(fast_mode):
            self._batch_cache.active_cache_valid_set_ids = explicit_valid_set_ids
            self._batch_cache.active_cache_invalidated_set_ids = (
                run_start_cache_decision.explicit_cache_invalidated_set_ids
            )
        else:
            self._batch_cache.active_preview_scope_set_ids = run_start_cache_decision.preview_scope_set_ids
        effective_workers = self._effective_batch_worker_count(len(queue_ids))
        parallel_mode = bool(effective_workers > 1 and len(queue_ids) > 1)
        retain_prepared_payloads_in_context = not (bool(parallel_mode) and not bool(fast_mode))
        dirty_reset_tracking = self._completion_policy.capture_dirty_reset_tracking(
            fast_mode=bool(fast_mode),
            queue_ids=tuple(queue_ids),
            dirty_state_by_set_id=self._capture_dirty_state_by_set_id(queue_ids),
        )
        run_id = None
        if parallel_mode:
            self._run_sequence_id = int(getattr(self, "_run_sequence_id", 0)) + 1
            run_id = int(self._run_sequence_id)
            self._active_run_id = int(run_id)

        primary_simulation_plan = None
        if not bool(fast_mode):
            if primary_set_id:
                primary_simulation_plan = simulation_plan_by_set_id.get(str(primary_set_id))
            if primary_simulation_plan is None and simulation_plan_by_set_id:
                primary_simulation_plan = dict(next(iter(simulation_plan_by_set_id.values())))

        self._batch_run_context = {
            "active": True,
            "request_id": int(request_id),
            "run_id": run_id,
            "runtime_input_epoch": int(getattr(self, "_authoritative_runtime_input_epoch", 0) or 0),
            "fast_mode": bool(fast_mode),
            "reuse_parallel_lane_pool": bool(reuse_parallel_lane_pool),
            "keep_lane_pool_alive": bool(reuse_parallel_lane_pool and parallel_mode),
            "parallel": bool(parallel_mode),
            "effective_workers": int(effective_workers),
            "prepared": (
                dict(prepared_payload)
                if (
                    (not bool(fast_mode))
                    and retain_prepared_payloads_in_context
                    and isinstance(prepared_payload, dict)
                )
                else None
            ),
            "prepared_by_set_id": (
                {str(set_id): dict(payload) for set_id, payload in prepared_payload_by_set_id.items()}
                if retain_prepared_payloads_in_context
                else {}
            ),
            "simulation_plan": (
                dict(primary_simulation_plan)
                if ((not bool(fast_mode)) and isinstance(primary_simulation_plan, dict))
                else None
            ),
            "simulation_plan_by_set_id": {
                str(set_id): dict(payload) for set_id, payload in simulation_plan_by_set_id.items()
            },
            "cache_key": cache_key,
            "scope_identity": scope_identity.to_payload(),
            "full_dsl": full_dsl,
            "mechanism_text_by_set_id": {str(set_id): str(text) for set_id, text in mechanism_text_by_set_id.items()},
            "mechanism_signature": (
                str(mechanism_signature_by_set_id.get(str(primary_set_id or "")) or "")
                if primary_set_id
                else ""
            )
            or batch_mechanism_signature(
                simulation_identity=(
                    coerce_simulation_identity(simulation_identity_by_set_id.get(str(primary_set_id or "")))
                    if primary_set_id
                    else None
                ),
            ),
            "mechanism_signature_by_set_id": {
                str(set_id): str(signature) for set_id, signature in mechanism_signature_by_set_id.items()
            },
            "simulation_identity_by_set_id": dict(simulation_identity_by_set_id),
            "solver_config": dict(solver_config),
            "t_end": float(t_end),
            "rows": list(batch_rows),
            "queue_ids": list(queue_ids),
            "queue_names": list(queue_names),
            "pending_workspace_reset_set_ids": list(dirty_reset_tracking.pending_workspace_reset_set_ids),
            "pending_dirty_reset_generation_by_set_id": dict(dirty_reset_tracking.pending_dirty_reset_generation_by_set_id),
            "pos": 0,
            "primary_set_id": primary,
            "total": len(queue_ids),
            "completed_set_ids": [],
            "pending_init_seed": {
                str(set_name): {str(species): float(value) for species, value in dict(seed).items()}
                for set_name, seed in dict(pending_init_seed).items()
            }
            if pending_init_seed
            else {},
            "pending_init_rewrite": pending_init_rewrite,
            "pending_init_applied": bool(pending_init_applied),
            "explicit_cache_preview_token": None,
            "explicit_cache_preview_scope_set_ids": run_start_cache_decision.explicit_preview_scope_set_ids,
            "explicit_cache_valid_set_ids": explicit_valid_set_ids,
            "explicit_cache_invalidated_set_ids": run_start_cache_decision.explicit_cache_invalidated_set_ids,
            "preview_scope_set_ids": run_start_cache_decision.preview_scope_set_ids,
            "preview_owner_epoch": preview_owner_epoch,
            "preview_batch_cache_token_by_set_id": dict(preview_batch_cache_token_by_set_id),
        }

        self._slider_simulation_active = bool(fast_mode)
        if bool(getattr(self, "_debug_batch_parallel", False)):
            logger.info(
                "BATCH_PAR run prepared request_id=%s run_id=%s sets=%s workers=%s parallel=%s slider=%s",
                int(request_id),
                int(run_id or 0),
                int(len(queue_ids)),
                int(effective_workers),
                bool(parallel_mode),
                bool(reuse_parallel_lane_pool),
            )
        if parallel_mode:
            self._start_parallel_batch_simulations()
        else:
            self._start_next_batch_simulation()

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
            ctx = getattr(self, "_batch_run_context", {}) or {}
            if isinstance(ctx, dict) and ctx.get("active"):
                queue = list(ctx.get("queue_names") or [])
                try:
                    pos = int(ctx.get("pos", 0))
                except Exception:
                    pos = 0
                try:
                    total = int(ctx.get("total") or len(queue) or 1)
                except Exception:
                    total = max(1, len(queue))
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

    def _resolve_completion_mechanism(
        self,
        *,
        mechanism: object | None,
        mechanism_text: str,
        solver_config: Mapping[str, Any],
        is_preview: bool,
        is_primary: bool,
    ) -> object | None:
        if mechanism is not None:
            return mechanism
        if bool(is_preview) or (not bool(is_primary)):
            return None
        mechanism_text_s = str(mechanism_text or "")
        if not mechanism_text_s.strip():
            return None
        try:
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism
            from kindred.core.units import UnitsModel

            temp_for_parse = float((solver_config or {}).get("temperature_K") or self.ui.solver.temperature_spinbox_value())
            return parse_dsl_to_mechanism(
                mechanism_text_s,
                initials={},
                units=UnitsModel(temperature_K=temp_for_parse, energy_unit="kJ/mol"),
            )
        except Exception:
            return None

    def _update_primary_result_materialization_contract(
        self,
        *,
        mechanism: object | None,
        mechanism_text: str,
        solver_config: Mapping[str, Any],
        is_preview: bool,
        is_primary: bool,
    ) -> bool:
        energy_mode = bool(
            mechanism is not None
            and (
                self.ui.mechanism_helpers.is_energy_mode_mechanism(mechanism)
                or self.ui.mechanism_helpers.dsl_has_computational_mode_generated_block(str(mechanism_text))
            )
        )
        if (not bool(is_primary)) or bool(is_preview):
            return energy_mode
        if energy_mode and mechanism is not None and self.ui.solver.dsl_global_temperature_K(str(mechanism_text)) is not None:
            self.ui.mechanism_helpers.sync_energy_mode_temperature_from_mechanism(mechanism)
        elif energy_mode:
            self.ui.mechanism_helpers.set_temperature_override_state(
                enabled=True,
                tooltip="Temperature for thermodynamic calculations (energy mode: add T=... to override).",
            )
            T_spin = float(self.ui.solver.temperature_spinbox_value())
            self.ui.mechanism_helpers.set_temperature_mode_indicator_text(
                f"Temperature: {T_spin:.2f} K (energy mode: set T=... in DSL)"
            )
        else:
            self.ui.mechanism_helpers.set_temperature_override_state(
                enabled=True,
                tooltip="Temperature for thermodynamic calculations",
            )
            self.ui.mechanism_helpers.update_temperature_mode_indicator()
        return energy_mode

    def _remember_primary_result_mechanism(
        self,
        *,
        mechanism: object,
        mechanism_text: str,
        solver_config: Mapping[str, Any],
    ) -> None:
        self.ui.mechanism_helpers.remember_last_mechanism(mechanism, str(mechanism_text), dict(solver_config or {}))
        try:
            self.ui.batch.sync_batch_species_columns(
                mechanism.species_names(),
                preserve_active_cache=True,
            )
        except Exception as exc:
            self._record_nonfatal_exception(
                "Failed to sync batch species columns after primary simulation completion",
                exc,
            )

    def _refresh_primary_result_controls(
        self,
        *,
        mechanism: object | None,
        energy_mode: bool,
        slider_triggered: bool,
        is_primary: bool,
    ) -> None:
        if not bool(is_primary):
            return
        if bool(energy_mode) and mechanism is not None:
            self.ui.mechanism_helpers.populate_energy_mode_variables_from_mechanism(
                mechanism,
                refresh_sliders=bool((not self.ui.slider.suppress_slider_refresh()) and (not bool(slider_triggered))),
                preserve_visibility=True,
            )
            if bool(slider_triggered):
                self.ui.slider.set_slider_triggered_simulation(False)
            return
        if self.ui.slider.suppress_slider_refresh():
            if bool(slider_triggered):
                logger.debug("Suppressed slider refresh during live drag")
                self.ui.slider.set_slider_triggered_simulation(False)
            return
        if not bool(slider_triggered):
            self.ui.mechanism_helpers.extract_and_populate_variables(
                preserve_visibility=True
            )
            return
        self.ui.slider.set_slider_triggered_simulation(False)
        logger.debug("Skipped variable extraction (slider-triggered simulation)")

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
    ):
        active_run_id = int(getattr(self, "_active_run_id", 0))
        shutdown_requested = bool(getattr(self, "_shutdown_requested_for_close", False))
        stale_fast_handoff_after_display = False
        if run_id is not None and int(run_id) != active_run_id:
            logger.debug(
                "Ignoring stale simulation completion (run_id=%s, active=%s)",
                run_id,
                active_run_id,
            )
            return
        latest_request_id = int(getattr(self, "_latest_sim_request_id", 0))
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if isinstance(ctx, Mapping) and ctx.get("runtime_input_epoch") is not None:
            try:
                completion_runtime_input_epoch = int(ctx.get("runtime_input_epoch") or 0)
            except Exception:
                completion_runtime_input_epoch = 0
            current_runtime_input_epoch = int(getattr(self, "_authoritative_runtime_input_epoch", 0) or 0)
            if completion_runtime_input_epoch != current_runtime_input_epoch:
                logger.debug(
                    "Ignoring stale simulation completion (runtime_input_epoch=%s, current=%s)",
                    completion_runtime_input_epoch,
                    current_runtime_input_epoch,
                )
                return
        policy_context = self._completion_policy_context_from_raw(ctx)
        callback_owner_epoch = self._effective_preview_owner_epoch_for_callback(
            owner_epoch=owner_epoch,
            context=policy_context,
        )
        missing_owner_epoch = self._missing_preview_owner_epoch_for_current_fast_owner(
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
                or (not self._preview_request_matches_current_owner_epoch(request_id, callback_owner_epoch))
            )
        )
        if is_superseded_fast_request:
            stale_fast_decision = self._completion_policy.resolve_superseded_fast_completion(
                preview_ownership=self._completion_policy_preview_ownership(),
                context=policy_context,
                request_id=int(request_id),
                preview_owner_epoch=callback_owner_epoch,
                pending_replay=self._completion_policy_pending_replay_state(),
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

            if stale_fast_decision.display_current_preview:
                self._apply_completion_policy_state_patch(stale_fast_decision.state_patch)
                if stale_fast_decision.defer_context_deactivation_until_after_display:
                    stale_fast_handoff_after_display = True
            else:
                updated_policy_context = self._apply_completion_policy_state_patch(
                    stale_fast_decision.state_patch,
                    base_context=ctx if isinstance(ctx, Mapping) else None,
                )
                if stale_fast_decision.deactivate_context_immediately:
                    ctx = self._batch_run_context

                if stale_fast_decision.deactivate_context_immediately:
                    self._release_current_simulation_worker()

                    keep_lane_pool_alive = bool(isinstance(ctx, dict) and ctx.get("parallel") and ctx.get("keep_lane_pool_alive"))
                    self._cleanup_parallel_batch_lane_pool_after_run(
                        keep_lane_pool_alive=keep_lane_pool_alive,
                        clear_pending_plot_updates=True,
                    )

                    self.ui.slider.set_slider_triggered_simulation(False)
                    self._simulation_running = False
                    try:
                        self.ui.run_ui.set_run_button_enabled(True)
                        self.ui.run_ui.set_stop_button_enabled(False)
                    except Exception as exc:
                        self._record_nonfatal_exception(
                            "Failed to reset Run/Stop button state after superseded fast completion",
                            exc,
                        )
                    self._slider_simulation_active = False

                    for stop_fn, timer_name in (
                        (self.ui.slider.stop_variable_update_timer, "_variable_update_timer"),
                        (self.ui.slider.stop_species_slider_update_timer, "_species_slider_update_timer"),
                    ):
                        try:
                            stop_fn()
                        except Exception as exc:
                            self._record_nonfatal_exception(
                                f"Failed to stop debounce timer {str(timer_name)} after superseded fast completion",
                                exc,
                            )

                if stale_fast_decision.schedule_pending_preview_run and (not stale_fast_decision.display_current_preview):
                    self._schedule_deferred_preview_replay_handoff_once(stop_timers=False)
                else:
                    if stale_fast_decision.reset_status_progress:
                        try:
                            self.ui.run_ui.set_status_text("Ready")
                            self.ui.run_ui.set_sim_progress_value(0)
                        except Exception as exc:
                            self._record_nonfatal_exception(
                                "Failed to reset status/progress after superseded fast completion",
                                exc,
                            )
                self._clear_shutdown_request_after_close_cleanup()
                return

        is_preview = bool(fast_mode)
        slider_triggered = bool(self.ui.slider.slider_triggered_simulation()) or bool(fast_mode)
        batch_queue_done = True
        try:
            logger.info("Simulation completed successfully")
            if bool(getattr(self, "_debug_batch_parallel", False)):
                logger.info(
                    "BATCH_PAR completion handler run_id=%s request_id=%s set_id=%s slider=%s ts=%.6f",
                    int(run_id or 0),
                    int(request_id or 0),
                    str(batch_set_id or ""),
                    bool(slider_triggered),
                    float(perf_counter()),
                )

            t = result["t"]
            Y = result["Y"]
            species_names = result["species_names"]
            algebra_scalars = result.get("algebra_scalars") or {}
            algebra_errors = result.get("algebra_errors") or []
            mechanism = result.get("mechanism")
            base_species_count = None
            raw_base_species_count = result.get("base_species_count")
            try:
                if raw_base_species_count is not None:
                    base_species_count = max(0, int(raw_base_species_count))
            except Exception:
                base_species_count = None
            if base_species_count is None and mechanism is not None:
                try:
                    base_species_count = len(list(mechanism.species_names()))
                except Exception:
                    base_species_count = None
            mechanism_text = result.get("mechanism_text", self.ui.mechanism.get_mechanism_text())
            solver_config = result.get("solver_config", {})

            explicit_batch_coalescing = bool(
                isinstance(ctx, dict)
                and ctx.get("parallel")
                and (not slider_triggered)
                and int(ctx.get("total") or 0) > 1
            )
            if cache_key is None and isinstance(ctx, dict):
                cache_key = ctx.get("cache_key")  # type: ignore[assignment]
            if cache_key is None:
                if is_preview:
                    cache_key = self._batch_cache.active_preview_cache_key
                else:
                    cache_key = self._batch_cache.active_cache_key
            cache_key = str(cache_key) if cache_key else None
            if cache_key:
                if is_preview:
                    self._batch_cache.active_preview_cache_key = cache_key
                    preview_scope_ids = (
                        policy_context.preview_scope_set_ids
                        if policy_context is not None and policy_context.preview_scope_set_ids
                        else None
                    )
                    self._batch_cache.active_preview_scope_set_ids = preview_scope_ids
                else:
                    cache_reconciliation = self._completion_policy.build_explicit_cache_reconciliation(
                        context=policy_context or self._completion_policy_context_from_raw(ctx),
                        cache_state=self._completion_policy_cache_state(),
                        cache_key=cache_key,
                    )
                    if cache_reconciliation.clear_active_selection_state:
                        self._batch_cache.clear_active_selection_state()
                    else:
                        self._batch_cache.active_cache_key = cache_reconciliation.active_cache_key
                        self._batch_cache.active_cache_preview_token = (
                            cache_reconciliation.active_cache_preview_token
                        )
                        self._batch_cache.active_cache_preview_scope_set_ids = (
                            cache_reconciliation.active_cache_preview_scope_set_ids
                        )
                        self._batch_cache.active_cache_valid_set_ids = cache_reconciliation.active_cache_valid_set_ids
                        self._batch_cache.active_cache_invalidated_set_ids = (
                            cache_reconciliation.active_cache_invalidated_set_ids
                        )

            cache_reconciliation = None
            redraw_valid_set_ids = None
            has_redraw_subset = False
            if not is_preview and policy_context is not None:
                cache_reconciliation = self._completion_policy.build_explicit_cache_reconciliation(
                    context=policy_context,
                    cache_state=self._completion_policy_cache_state(),
                    cache_key=cache_key,
                )
                redraw_valid_set_ids = cache_reconciliation.redraw_valid_set_ids
                has_redraw_subset = bool(cache_reconciliation.has_redraw_subset)

            if (batch_set is None or batch_set_id is None) and isinstance(ctx, dict) and ctx.get("active"):
                queue_names = list(ctx.get("queue_names") or [])
                queue_ids = list(ctx.get("queue_ids") or [])
                try:
                    pos_hint = int(ctx.get("pos", 0))
                except Exception:
                    pos_hint = 0
                if 0 <= pos_hint < len(queue_names):
                    batch_set = str(queue_names[pos_hint])
                if 0 <= pos_hint < len(queue_ids):
                    batch_set_id = str(queue_ids[pos_hint])
            if batch_set_id is None and isinstance(batch_set, str):
                batch_set_id = self.ui.batch.batch_set_id_for_name(batch_set)
            if batch_set is None and isinstance(batch_set_id, str):
                batch_set = self.ui.batch.batch_set_name_for_id(batch_set_id)
            cache_plan_for_completion = _simulation_plan_for_set_from_context(ctx, batch_set_id=batch_set_id)

            primary_set = ctx.get("primary_set_id") if isinstance(ctx, dict) else None
            is_primary = True
            if primary_set:
                is_primary = bool(batch_set_id is not None and str(batch_set_id) == str(primary_set))

            mechanism = self._resolve_completion_mechanism(
                mechanism=mechanism,
                mechanism_text=str(mechanism_text),
                solver_config=solver_config,
                is_preview=bool(is_preview),
                is_primary=bool(is_primary),
            )
            energy_mode = self._update_primary_result_materialization_contract(
                mechanism=mechanism,
                mechanism_text=str(mechanism_text),
                solver_config=solver_config,
                is_preview=bool(is_preview),
                is_primary=bool(is_primary),
            )

            fallback_occurred = bool(result.get("fallback_occurred"))
            fallback_message = result.get("fallback_message")
            if (not is_preview) and is_primary and fallback_occurred:
                solver_name = solver_config.get("solver", "selected solver")
                warning_text = f"The requested stiff solver {solver_name} failed"
                if fallback_message:
                    warning_text += f" ({fallback_message})."
                else:
                    warning_text += "."
                warning_text += " The simulation retried with an alternative stiff SciPy solver."
                logger.warning("Displaying solver fallback warning: %s", warning_text)
                self.ui.dialogs.message_box_warning("Solver fallback", warning_text)

            if (not is_preview) and is_primary and mechanism is not None:
                self._remember_primary_result_mechanism(
                    mechanism=mechanism,
                    mechanism_text=str(mechanism_text),
                    solver_config=solver_config,
                )
                if policy_context is not None and cache_key:
                    policy_context = self._completion_policy.build_context_update_from_cache_truth(
                        context=policy_context,
                        cache_state=self._completion_policy_cache_state(),
                        cache_key=cache_key,
                    )
                    ctx = self._serialize_completion_policy_context(
                        policy_context,
                        base_context=ctx if isinstance(ctx, Mapping) else None,
                    )
                    self._batch_run_context = dict(ctx)

            series: Dict[str, Any] = {}
            for i, species_name in enumerate(species_names):
                series[species_name] = Y[i, :]

            try:
                if isinstance(algebra_errors, list) and algebra_errors:
                    ok = max(0, len(species_names) - int(base_species_count or 0))
                    err = len([e for e in algebra_errors if isinstance(e, dict)])
                    self.ui.run_ui.set_algebra_status_text(
                        f"Algebra: {ok} ok, {err} error" + ("s" if err != 1 else "")
                    )
                else:
                    self.ui.run_ui.set_algebra_status_text("")
            except Exception:
                self.ui.run_ui.set_algebra_status_text("")

            cache_plan = cache_plan_for_completion
            cache_token = (cache_plan.cache_key() if cache_plan is not None else "") or str(cache_key or "")
            if cache_token and batch_set_id:
                cache_store = self._batch_cache.preview_cache if bool(is_preview) else self._batch_cache.result_cache
                composite_key = BatchSimulationCache.entry_key(cache_token, str(batch_set_id))
                cached_mechanism = (
                    mechanism
                    if self._include_mechanism_in_result_payload(
                        fast_mode=bool(is_preview),
                        batch_set_id=batch_set_id,
                        context=ctx,
                    )
                    else None
                )
                cache_simulation_identity = (
                    cache_plan.simulation_identity_payload() if cache_plan is not None else {}
                )
                if not cache_simulation_identity and isinstance(ctx, dict):
                    cache_simulation_identity = dict(
                        (ctx.get("simulation_identity_by_set_id") or {}).get(str(batch_set_id)) or {}
                    )
                cache_preview_token = None
                if bool(is_preview):
                    cache_preview_token = (
                        cache_plan.preview_batch_cache_token() if cache_plan is not None else ""
                    ) or self._preview_batch_cache_token_for_cached_result(batch_set_id=batch_set_id, context=ctx)
                payload = build_batch_cache_entry(
                    t=t,
                    series=series,
                    algebra_scalars=(algebra_scalars if isinstance(algebra_scalars, dict) else None),
                    mechanism=cached_mechanism,
                    mechanism_text=str(mechanism_text),
                    simulation_identity=cache_simulation_identity,
                    solver_config=(solver_config if isinstance(solver_config, dict) else None),
                    preview_batch_cache_token=cache_preview_token,
                    fallback_occurred=bool(fallback_occurred),
                    fallback_message=fallback_message,
                )
                cache_store.put(composite_key, payload)

            if policy_context is not None:
                pending_init_completion = self._completion_policy.resolve_pending_init_completion(
                    context=policy_context,
                    batch_set=batch_set,
                    is_preview=bool(is_preview),
                    is_primary=bool(is_primary),
                )
                if pending_init_completion.should_attempt_apply:
                    applied = False
                    try:
                        applied = bool(
                            self.ui.mechanism_helpers.apply_pending_init_migration(
                                seed_sets={str(batch_set): dict(pending_init_completion.seed_for_ui)},
                                rewrite=str(pending_init_completion.rewrite or ""),
                            )
                        )
                    except Exception:
                        applied = False
                    if applied:
                        policy_context = self._completion_policy.note_pending_init_apply_result(
                            context=policy_context,
                            applied=True,
                        )
                        ctx = self._serialize_completion_policy_context(
                            policy_context,
                            base_context=ctx if isinstance(ctx, Mapping) else None,
                        )
                        self._batch_run_context = dict(ctx)

            displayed = False
            if cache_key and (slider_triggered or explicit_batch_coalescing):
                self.queue_slider_plot_update(
                    set_id=batch_set_id,
                    cache_key=str(cache_key),
                    request_id=request_id,
                    run_id=run_id,
                    slider_triggered=bool(slider_triggered),
                    valid_set_ids=(
                        redraw_valid_set_ids
                        if (explicit_batch_coalescing and has_redraw_subset)
                        else None
                    ),
                    allow_fallback=(not bool(explicit_batch_coalescing)),
                )
                displayed = True
            elif cache_key:
                selected_sets = self.ui.batch.batch_set_ids_for_scope("selected")
                prefer = None
                current_row = self.ui.batch.batch_current_row()
                if current_row is not None:
                    prefer = self.ui.batch.batch_set_id_for_row(int(current_row))
                displayed = self.ui.batch.display_cached_batch_selection(
                    cache_key=str(cache_key),
                    selected_sets=selected_sets,
                    prefer_set=prefer,
                    cache_store=None,
                    valid_set_ids=(redraw_valid_set_ids if has_redraw_subset else None),
                    allow_fallback=False,
                )
            if not displayed:
                owned_species = None
                if mechanism is not None:
                    try:
                        owned_species = list(mechanism.species_names())
                    except Exception:
                        owned_species = None
                if str(batch_set_id or "").strip():
                    self.ui.batch.set_active_batch_selection(str(batch_set_id), str(batch_set or ""), [str(batch_set_id)])
                else:
                    self.ui.batch.clear_display_selection_state()
                self.ui.results.set_data(
                    t,
                    series,
                    label=(str(batch_set) if batch_set else None),
                    overlays=[],
                    owned_species=owned_species,
                )
                if hasattr(self.ui.results, "sync_main_plot_copy_labels"):
                    self.ui.results.sync_main_plot_copy_labels(
                        str(batch_set_id or ""),
                        [str(batch_set_id)] if str(batch_set_id or "").strip() else [],
                    )
                plot = self.ui.results.main_plot()
                if hasattr(plot, "set_scalar_values"):
                    plot.set_scalar_values(algebra_scalars)

                display_label = str(batch_set) if batch_set else (str(batch_set_id) if batch_set_id else "Results")
                try:
                    if hasattr(plot, "set_statistics_results"):
                        plot.set_statistics_results({display_label: {"t": t, "series": series}}, prefer=display_label)
                    else:
                        plot.update_statistics(t, series)
                except Exception as exc:
                        self._record_nonfatal_exception(
                            f"Failed to update plot statistics after simulation completion (label={display_label})",
                            exc,
                        )
                try:
                    self.ui.results.set_results_table(plot.stats_table())
                except Exception as exc:
                    self._record_nonfatal_exception("Failed to update results table after simulation completion", exc)

            if policy_context is not None:
                pending_init_guard_rewrite = self._completion_policy.should_arm_pending_init_guard(
                    context=policy_context,
                    is_preview=bool(is_preview),
                    is_primary=bool(is_primary),
                )
                if pending_init_guard_rewrite:
                    self.ui.mechanism_helpers.arm_pending_init_result_invalidation_guard(
                        rewrite=str(pending_init_guard_rewrite)
                    )

            self._refresh_primary_result_controls(
                mechanism=mechanism,
                energy_mode=bool(energy_mode),
                slider_triggered=bool(slider_triggered),
                is_primary=bool(is_primary),
            )

            if is_primary:
                temperature_used = float(self.ui.solver.temperature_spinbox_value())
                energy_unit_used = None
                if mechanism is not None:
                    try:
                        mmeta = getattr(mechanism, "metadata", {}) or {}
                        if isinstance(mmeta, dict) and mmeta.get("temperature_K") is not None:
                            temperature_used = float(mmeta.get("temperature_K"))
                        if isinstance(mmeta, dict) and mmeta.get("energy_unit"):
                            energy_unit_used = str(mmeta.get("energy_unit"))
                    except Exception as exc:
                        self._record_nonfatal_exception(
                            "Failed to read mechanism metadata for simulation provenance",
                            exc,
                        )

                sim_time_prov: float | str
                sim_time_prov = str(self.ui.solver.sim_time_spinbox_text()).strip()
                try:
                    sim_time_prov = float(self.ui.solver.parse_sim_time_seconds())
                except Exception as exc:
                    self._record_nonfatal_exception(
                        "Failed to parse simulation time for provenance; keeping text value",
                        exc,
                    )

                from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, normalize_solver_name

                solver_label = str(
                    solver_config.get("solver_label") or self.ui.solver.initial_solver_name() or DEFAULT_SOLVER_NAME
                ).strip() or DEFAULT_SOLVER_NAME
                solver_method, solver_warning = normalize_solver_name(str(solver_config.get("solver") or solver_label))

                provenance: Dict[str, Any] = {
                    "timestamp": datetime.now().isoformat(),
                    "kindred_version": KINDRED_VERSION,
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "mechanism_dsl": mechanism_text,
                    "solver": str(solver_method),
                    "solver_label": str(solver_label),
                    "solver_warning": str(solver_warning) if solver_warning else None,
                    "rtol": solver_config.get("rtol", self.ui.solver.initial_rtol() or 1e-6),
                    "atol": solver_config.get("atol", self.ui.solver.initial_atol() or 1e-12),
                    "temperature_K": float(temperature_used),
                    "temperature_source": (
                        "dsl"
                        if self.ui.solver.dsl_global_temperature_K(mechanism_text) is not None
                        else "ui"
                    ),
                    "energy_unit": energy_unit_used,
                    "energy_mode": bool(energy_mode),
                    "simulation_time": sim_time_prov,
                    "num_points_requested": int(self.ui.solver.num_points_spinbox_value()),
                    "num_species": len(species_names),
                    "num_points": len(t),
                    "species_names": list(species_names),
                    "datasets": self.ui.provenance.snapshot_datasets(),
                }
                if algebra_scalars:
                    provenance["algebra_scalars"] = dict(algebra_scalars)
                overlay_snapshot = getattr(self.ui.results.main_plot(), "overlay_snapshot", None)
                if callable(overlay_snapshot):
                    provenance["dataset_overlays"] = overlay_snapshot()
                fit_meta = self.ui.provenance.last_fit_metadata()
                if fit_meta:
                    provenance["fit"] = fit_meta

                ctc_values: Dict[str, float] = {}
                ctc_metadata: Dict[str, Any] = {}
                for species_name, conc_array in series.items():
                    final_conc = conc_array[-1]
                    max_conc = np.max(np.abs(conc_array))
                    threshold = max(1e-10, 0.01 * max_conc)

                    if abs(final_conc) < threshold:
                        ctc_value, method, is_uniform, eps_used, tail_used = self.ui.provenance.integrate_ctc(
                            t,
                            conc_array,
                            uniformity_eps=1e-6,
                            tail_strategy="38",
                        )
                    else:
                        deviation = np.abs(conc_array - final_conc)
                        ctc_value, method, is_uniform, eps_used, tail_used = self.ui.provenance.integrate_ctc(
                            t,
                            deviation,
                            uniformity_eps=1e-6,
                            tail_strategy="38",
                        )

                    ctc_values[species_name] = float(ctc_value)
                    ctc_metadata = {
                        "integration_method": method,
                        "uniform_grid_detected": is_uniform,
                        "uniformity_eps": eps_used,
                        "tail_strategy": tail_used,
                    }

                self.ui.provenance.set_last_simulation_ctc(ctc_values)
                if ctc_metadata:
                    provenance["ctc"] = ctc_metadata
                self.ui.provenance.set_last_simulation_provenance(provenance)

            if isinstance(ctx, dict) and ctx.get("active"):
                policy_context = self._completion_policy_context_from_raw(ctx)
                queue_ids = list(policy_context.queue_ids if policy_context is not None else (ctx.get("queue_ids") or []))
                pos = int(policy_context.pos) if policy_context is not None else _safe_int(ctx.get("pos", 0), default=0)
                total = (
                    int(policy_context.total)
                    if policy_context is not None
                    else _safe_int(ctx.get("total"), default=max(1, len(queue_ids)))
                )
                total = max(1, total or len(queue_ids) or 1)

                if stale_fast_handoff_after_display:
                    ctx["active"] = False
                    self._batch_run_context = dict(ctx)
                    batch_queue_done = True
                elif bool(policy_context.parallel if policy_context is not None else ctx.get("parallel")):
                    completed_ids = set(policy_context.completed_set_ids if policy_context is not None else ())
                    if batch_set_id:
                        completed_ids.add(str(batch_set_id))
                    ctx["completed_set_ids"] = sorted(completed_ids)
                    completed = len(completed_ids)
                    if completed < total:
                        batch_queue_done = False
                        self._batch_run_context = dict(ctx)
                        if total > 1:
                            overall = int((completed / float(total)) * 100.0)
                            self.ui.run_ui.set_sim_progress_value(max(0, min(100, overall)))
                        if batch_set:
                            self.ui.run_ui.set_status_text(f"Completed {batch_set} ({completed}/{total})")
                    else:
                        ctx["active"] = False
                        self._batch_run_context = dict(ctx)
                        batch_queue_done = True
                else:
                    if 0 <= pos < len(queue_ids):
                        expected = str(queue_ids[pos])
                        if batch_set_id is None or str(batch_set_id) == expected:
                            ctx["pos"] = pos + 1

                    next_pos = int(ctx.get("pos", 0))
                    if next_pos < len(queue_ids):
                        if shutdown_requested:
                            ctx["active"] = False
                            self._batch_run_context = dict(ctx)
                            self._release_current_simulation_worker()
                            batch_queue_done = True
                        else:
                            batch_queue_done = False
                            self._batch_run_context = dict(ctx)
                            if total > 1:
                                overall = int((next_pos / float(total)) * 100.0)
                                self.ui.run_ui.set_sim_progress_value(max(0, min(100, overall)))
                            if batch_set:
                                self.ui.run_ui.set_status_text(f"Completed {batch_set} ({next_pos}/{total})")
                            self._release_current_simulation_worker()
                            QtCore.QTimer.singleShot(0, self._start_next_batch_simulation)
                    else:
                        ctx["active"] = False
                        self._batch_run_context = dict(ctx)
                        self._release_current_simulation_worker()
                        batch_queue_done = True

            if batch_queue_done:
                if (not is_preview) and isinstance(ctx, dict):
                    policy_context = self._completion_policy_context_from_raw(ctx)
                    reset_target_set_ids = (
                        tuple(policy_context.pending_workspace_reset_set_ids)
                        if policy_context is not None
                        else tuple(str(set_id) for set_id in (ctx.get("pending_workspace_reset_set_ids") or ()) if str(set_id))
                    )
                    dirty_reset_decision = self._completion_policy.resolve_explicit_dirty_reset(
                        context=policy_context or self._completion_policy_context_from_raw(ctx),
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
                    if policy_context is not None:
                        ctx = self._batch_run_context
                    if overlays_cleared:
                        try:
                            species_for_sync = (
                                list(mechanism.species_names())
                                if mechanism is not None and hasattr(mechanism, "species_names")
                                else list(species_names)
                            )
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
                if slider_triggered or explicit_batch_coalescing:
                    self.flush_slider_plot_updates(
                        force=bool(explicit_batch_coalescing),
                        cache_key=cache_key,
                        request_id=request_id,
                        run_id=run_id,
                    )
                failed_set_ids = []
                failed_errors: Mapping[str, Any] = {}
                if isinstance(ctx, Mapping):
                    failed_set_ids = [
                        str(set_id)
                        for set_id in (ctx.get("failed_set_ids") or ())
                        if str(set_id)
                    ]
                    raw_failed_errors = ctx.get("failed_set_errors")
                    if isinstance(raw_failed_errors, Mapping):
                        failed_errors = raw_failed_errors

                self.ui.run_ui.set_sim_progress_value(100)
                if failed_set_ids and not bool(is_preview):
                    failed_count = len(failed_set_ids)
                    self.ui.run_ui.set_status_text(f"Batch completed with {failed_count} failed set(s)")
                    self._show_scoped_batch_failure_summary(
                        failed_set_ids=failed_set_ids,
                        failed_errors=failed_errors,
                    )
                else:
                    self.ui.run_ui.set_status_text(
                        f"Simulation complete: {len(species_names)} species, {len(t)} points"
                    )

                self.ui.run_ui.repaint_simulation_widgets()

                logger.info(f"Displayed results: {len(t)} time points")
                logger.info("Captured simulation provenance and CTC metadata")

        except Exception as e:
            logger.error(f"Error displaying results: {e}", exc_info=True)
            self.ui.dialogs.message_box_critical(
                "Display Error",
                f"Failed to display results:\n\n{e}",
            )
            self.ui.run_ui.set_status_text("Display failed")
        finally:
            if batch_queue_done:
                self._release_current_simulation_worker()
                ctx_for_cleanup = locals().get("ctx", {}) or {}
                keep_lane_pool_alive = bool(
                    isinstance(ctx_for_cleanup, dict)
                    and ctx_for_cleanup.get("parallel")
                    and ctx_for_cleanup.get("keep_lane_pool_alive")
                )
                self._cleanup_parallel_batch_lane_pool_after_run(
                    keep_lane_pool_alive=keep_lane_pool_alive,
                    stale_fast_handoff_after_display=stale_fast_handoff_after_display,
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
    ):
        error_payload = coerce_simulation_failure(error_msg)
        error_text = simulation_failure_user_message(error_payload)
        error_detail_text = simulation_failure_detail_text(error_payload)
        cancelled = is_cancelled_failure(error_payload)
        active_run_id = int(getattr(self, "_active_run_id", 0))
        if run_id is not None and int(run_id) != active_run_id:
            logger.debug(
                "Ignoring stale simulation error (run_id=%s, active=%s): %s",
                run_id,
                active_run_id,
                error_text,
            )
            return
        latest_request_id = int(getattr(self, "_latest_sim_request_id", 0))
        ctx = getattr(self, "_batch_run_context", {}) or {}
        policy_context = self._completion_policy_context_from_raw(ctx)
        callback_owner_epoch = self._effective_preview_owner_epoch_for_callback(
            owner_epoch=owner_epoch,
            context=policy_context,
        )
        missing_owner_epoch = self._missing_preview_owner_epoch_for_current_fast_owner(
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
                or (not self._preview_request_matches_current_owner_epoch(request_id, callback_owner_epoch))
            )
        )
        if is_superseded_fast_request:
            stale_fast_decision = self._completion_policy.resolve_superseded_fast_error(
                preview_ownership=self._completion_policy_preview_ownership(),
                context=policy_context,
                request_id=int(request_id),
                preview_owner_epoch=callback_owner_epoch,
                pending_replay=self._completion_policy_pending_replay_state(),
            )
            logger.debug(
                "Active fast error superseded (request_id=%s, latest=%s, run_id=%s, schedule_pending=%s): %s",
                request_id,
                latest_request_id,
                run_id,
                bool(stale_fast_decision.schedule_pending_preview_run),
                error_text,
            )

            self._apply_completion_policy_state_patch(
                stale_fast_decision.state_patch,
                base_context=ctx if isinstance(ctx, Mapping) else None,
            )
            ctx = self._batch_run_context
            if stale_fast_decision.deactivate_context_immediately:

                self._release_current_simulation_worker()
                self._shutdown_batch_lane_pool(force_terminate=False)
                self._clear_shutdown_request_after_close_cleanup()

                self.ui.slider.set_slider_triggered_simulation(False)
                self._simulation_running = False
                try:
                    self.ui.run_ui.set_run_button_enabled(True)
                    self.ui.run_ui.set_stop_button_enabled(False)
                except Exception as exc:
                    self._record_nonfatal_exception(
                        "Failed to reset Run/Stop button state after superseded fast error",
                        exc,
                    )
                self._slider_simulation_active = False

                for stop_fn, timer_name in (
                    (self.ui.slider.stop_variable_update_timer, "_variable_update_timer"),
                    (self.ui.slider.stop_species_slider_update_timer, "_species_slider_update_timer"),
                ):
                    try:
                        stop_fn()
                    except Exception as exc:
                        self._record_nonfatal_exception(
                            f"Failed to stop debounce timer {str(timer_name)} after superseded fast error",
                            exc,
                        )

            if stale_fast_decision.schedule_pending_preview_run:
                self._schedule_deferred_preview_replay_handoff_once(stop_timers=False)
            else:
                if stale_fast_decision.reset_status_progress:
                    try:
                        self.ui.run_ui.set_status_text("Ready")
                        self.ui.run_ui.set_sim_progress_value(0)
                    except Exception as exc:
                        self._record_nonfatal_exception(
                            "Failed to reset status/progress after superseded fast error",
                            exc,
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
            self._handle_current_preview_simulation_failure(
                error_payload,
                error_text=error_text,
                error_detail_text=error_detail_text,
                context=ctx if isinstance(ctx, Mapping) else None,
            )
            return
        logger.warning("Simulation error surfaced to UI: %s", error_text)

        if isinstance(ctx, dict) and ctx.get("active"):
            ctx["active"] = False
            self._batch_run_context = dict(ctx)
        self._release_current_simulation_worker()
        self._shutdown_batch_lane_pool(force_terminate=True)
        self._close_contained_simulation_owner(fast_mode=bool(fast_mode), kill=True)
        self._clear_shutdown_request_after_close_cleanup()

        if not cancelled:
            if error_detail_text:
                logger.warning("%s", error_detail_text)
            self.ui.dialogs.message_box_critical(
                "Simulation Error",
                f"Simulation failed:\n\n{error_text}",
                details=error_detail_text or None,
            )
            self.ui.run_ui.set_status_text("Simulation failed")
        else:
            self.ui.run_ui.set_status_text("Simulation cancelled by user")
        try:
            self.ui.run_ui.set_algebra_status_text("")
        except Exception as exc:
            self._record_nonfatal_exception(
                "Failed to clear algebra status label after simulation error",
                exc,
            )

        self.ui.run_ui.set_sim_progress_value(0)

        self._simulation_running = False
        self.ui.run_ui.set_run_button_enabled(True)
        self.ui.run_ui.set_stop_button_enabled(False)
        self._slider_simulation_active = False
        self.ui.slider.set_slider_triggered_simulation(False)

        if cancelled:
            if self._has_deferred_preview_replay_intent():
                logger.debug("Resuming pending slider update after cancellation")
                self._schedule_deferred_preview_replay_handoff_once()
        else:
            self._apply_explicit_failure_pending_replay_policy(fast_mode=bool(fast_mode))

        self._invalidate_preserved_pending_init_results_after_failed_run(
            ctx=ctx if isinstance(ctx, Mapping) else None,
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

        ctx = dict(context or {})
        if ctx.get("active"):
            ctx["active"] = False
            self._batch_run_context = dict(ctx)
        self._release_current_simulation_worker()
        self._shutdown_batch_lane_pool(force_terminate=True)
        self._close_contained_simulation_owner(fast_mode=True, kill=True)
        self._clear_shutdown_request_after_close_cleanup()
        self._clear_pending_preview_slider_plot_updates()
        try:
            self.ui.slider.set_slider_triggered_simulation(False)
        except Exception as exc:
            self._record_nonfatal_exception(
                "Failed to clear slider-triggered state after preview failure",
                exc,
            )
        self._simulation_running = False
        self._slider_simulation_active = False
        self.ui.run_ui.set_run_button_enabled(True)
        self.ui.run_ui.set_stop_button_enabled(False)
        self.ui.run_ui.set_sim_progress_value(0)
        try:
            self.ui.run_ui.set_algebra_status_text("")
        except Exception as exc:
            self._record_nonfatal_exception(
                "Failed to clear algebra status label after preview failure",
                exc,
            )

        try:
            self.ui.slider.show_preview_unavailable_for_dirty_state(status_text)
        except Exception as exc:
            self._record_nonfatal_exception(
                "Failed to show dirty no-preview state after preview failure",
                exc,
            )
            self.ui.run_ui.set_status_text(status_text)
        self.ui.run_ui.set_status_text(status_text)

    def _stop_simulation(self):
        if not self._simulation_running:
            return

        logger.info("Stop simulation requested")

        ctx = getattr(self, "_batch_run_context", {}) or {}
        if isinstance(ctx, dict) and ctx.get("active"):
            ctx["active"] = False
            self._batch_run_context = dict(ctx)
        self._shutdown_batch_lane_pool(force_terminate=True)
        active_fast = bool(isinstance(ctx, dict) and ctx.get("fast_mode"))

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
