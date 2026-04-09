from __future__ import annotations

from contextlib import suppress
import hashlib
import logging
import os
import platform
from datetime import datetime
from queue import Empty
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from PySide6 import QtCore
import shiboken6

from kindred import __version__ as KINDRED_VERSION
from kindred.core.batch_parallel import (
    batch_mechanism_signature,
    compute_effective_batch_workers,
    initialize_batch_worker,
    run_batch_simulation_task,
)
from kindred.core.simulation_identity import (
    SimulationIdentity,
    SimulationScopeIdentity,
    coerce_simulation_identity,
)
from kindred.core.simulation_failure import (
    coerce_simulation_failure,
    is_cancelled_failure,
    simulation_failure_user_message,
)
from kindred.core.simulation_preparation import BoundMechanism, SimulationExecutionRequest
from kindred.gui.controllers.cache_contracts import build_batch_cache_entry
from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.gui.controllers.parallel_batch_executor import ParallelBatchExecutor
from kindred.gui.controllers.simulation_cache_admin import SimulationCacheAdmin
from kindred.gui.controllers.simulation_run_state import SimulationRunState
from kindred.gui.controllers.slider_plot_coalescer import SliderPlotCoalescer
from kindred.core.batch_initial_conditions import (
    migrate_reaction_dsl_initial_concentration_sets,
    strip_reaction_dsl_initial_concentrations,
)
from kindred.gui.ports import SimulationCacheOpResult, SimulationUiPorts

logger = logging.getLogger(__name__)

__all__ = ["SimulationController"]


def _try_float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _pending_initial_seed_for_set(
    pending_init_seed: object,
    *,
    set_name: str,
) -> Dict[str, object]:
    if not isinstance(pending_init_seed, Mapping) or not pending_init_seed:
        return {}

    nested_seed = pending_init_seed.get(str(set_name))
    if isinstance(nested_seed, Mapping):
        return {str(species): value for species, value in nested_seed.items()}

    # Backward-compatibility for legacy tests/context payloads that still store a
    # flat seed dict targeting the default set1 row.
    if str(set_name) == "set1" and all(not isinstance(value, Mapping) for value in pending_init_seed.values()):
        return {str(species): value for species, value in pending_init_seed.items()}
    return {}

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


def _default_batch_executor_factory(max_workers: int, limit_blas_threads: bool):
    """Create a process pool for batch simulations (injectable in tests)."""
    # Keep multiprocessing/process-pool imports lazy so startup paths never
    # touch process-launch machinery until an actual parallel batch run.
    import multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor

    return _ProcessPoolExecutor(
        max_workers=max(1, int(max_workers)),
        initializer=initialize_batch_worker,
        initargs=(bool(limit_blas_threads),),
        mp_context=_mp.get_context("spawn"),
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
        self._batch_cache = BatchSimulationCache(result_cache_cap=100, preview_cache_cap=3)
        self._cache_admin = SimulationCacheAdmin(
            cache=self._batch_cache,
            settings_set_value=self.ui.settings.settings_set_value,
            settings_sync=self.ui.settings.settings_sync,
            record_nonfatal_exception=self._record_nonfatal_exception,
        )
        self._batch_run_context: Dict[str, Any] = {}

        # Parallel batch orchestration (process pool + futures bookkeeping)
        self._batch_parallel = ParallelBatchExecutor(
            executor_factory=_default_batch_executor_factory,
            max_parallel_workers=12,
            limit_blas_threads_per_worker=True,
            record_nonfatal_exception=self._record_nonfatal_exception,
        )

        self._batch_future_poll_timer = QtCore.QTimer(self)
        self._batch_future_poll_timer.setInterval(20)
        self._batch_future_poll_timer.timeout.connect(self._poll_parallel_batch_futures)

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
        return bool(self._run_state.pending_slider_simulation)

    @_pending_slider_simulation.setter
    def _pending_slider_simulation(self, value: bool) -> None:
        self._run_state.pending_slider_simulation = bool(value)

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
        return self._run_state.pending_slider_sim_request_id

    @_pending_slider_sim_request_id.setter
    def _pending_slider_sim_request_id(self, value: Optional[int]) -> None:
        self._run_state.pending_slider_sim_request_id = int(value) if value is not None else None

    @property
    def _pending_slider_target_set_ids(self) -> Tuple[str, ...]:
        return tuple(
            str(set_id)
            for set_id in (getattr(self._run_state, "pending_slider_target_set_ids", ()) or ())
            if str(set_id)
        )

    @_pending_slider_target_set_ids.setter
    def _pending_slider_target_set_ids(self, value: Sequence[str]) -> None:
        normalized: list[str] = []
        seen: set[str] = set()
        for set_id in value or ():
            set_id_s = str(set_id or "").strip()
            if not set_id_s or set_id_s in seen:
                continue
            seen.add(set_id_s)
            normalized.append(set_id_s)
        self._run_state.pending_slider_target_set_ids = tuple(normalized)

    def queue_pending_slider_preview_replay(
        self,
        *,
        target_set_ids: Sequence[str],
        request_id: Optional[int] = None,
        preserve_existing_request: bool = False,
    ) -> None:
        self._pending_slider_target_set_ids = target_set_ids
        if request_id is not None:
            self._pending_slider_sim_request_id = int(request_id)
        elif not bool(preserve_existing_request):
            self._pending_slider_sim_request_id = None
        self._pending_slider_simulation = True

    def clear_pending_slider_preview_replay(self, *, clear_plot_updates: bool = True) -> None:
        self._pending_slider_simulation = False
        self._pending_slider_sim_request_id = None
        self._pending_slider_target_set_ids = ()
        if clear_plot_updates:
            self._clear_pending_preview_slider_plot_updates()

    def _preserve_pending_slider_preview_replay_excluding(self, reset_set_ids: Sequence[str]) -> bool:
        reset_set_id_set = {str(set_id) for set_id in (reset_set_ids or ()) if str(set_id)}
        surviving_target_set_ids = [
            str(set_id)
            for set_id in self._pending_slider_target_set_ids
            if str(set_id) and str(set_id) not in reset_set_id_set
        ]
        if not surviving_target_set_ids:
            self.clear_pending_slider_preview_replay(clear_plot_updates=True)
            return False
        self.queue_pending_slider_preview_replay(
            target_set_ids=surviving_target_set_ids,
            request_id=None,
        )
        self._clear_pending_preview_slider_plot_updates()
        return True

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
    def plot_coalescer(self) -> SliderPlotCoalescer:
        return self._plot_coalescer

    @property
    def batch_run_context(self) -> Dict[str, Any]:
        return self._batch_run_context

    @batch_run_context.setter
    def batch_run_context(self, value: Dict[str, Any]) -> None:
        self._batch_run_context = dict(value or {})

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

    def run_simulation(self) -> None:
        self._run_simulation()

    def stop_simulation(self) -> None:
        self._stop_simulation()

    def run_simulation_from_slider(self) -> None:
        self._run_simulation_from_slider()

    def invalidate_slider_preview_work(self) -> None:
        self._invalidate_slider_preview_work()

    def invalidate_active_explicit_simulation_for_authoritative_change(self) -> None:
        self._invalidate_active_explicit_simulation_for_authoritative_change()

    def run_simulation_internal(
        self,
        *,
        fast_mode: bool = False,
        request_id: Optional[int] = None,
        batch_rows: Optional[Sequence[int]] = None,
        reuse_parallel_executor: bool = False,
    ) -> None:
        self._run_simulation_internal(
            fast_mode=fast_mode,
            request_id=request_id,
            batch_rows=batch_rows,
            reuse_parallel_executor=reuse_parallel_executor,
        )

    def poll_parallel_batch_futures(self) -> None:
        self._poll_parallel_batch_futures()

    def shutdown_batch_executor(self, *, force_terminate: bool) -> None:
        self._shutdown_batch_executor(force_terminate=force_terminate)

    def parallel_batch_pool_settings_changed(self) -> None:
        self._parallel_batch_pool_settings_changed()

    def ensure_parallel_batch_pool_eagerly_created(self) -> None:
        self._ensure_parallel_batch_pool_eagerly_created()

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
    # Worker / executor lifecycle
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
        pending_request_id = getattr(self, "_pending_slider_sim_request_id", None)
        latest_request_id = int(getattr(self, "_latest_sim_request_id", 0))
        if pending_request_id is not None and int(pending_request_id) < latest_request_id:
            self._pending_slider_simulation = False
            self._pending_slider_sim_request_id = None
            self._pending_slider_target_set_ids = ()
        elif (
            self._pending_slider_simulation
            and (not shutdown_requested)
            and (not self._has_running_owned_simulation_workers())
        ):
            self._pending_slider_simulation = False
            self.ui.slider.stop_variable_update_timer()
            self.ui.slider.stop_species_slider_update_timer()
            QtCore.QTimer.singleShot(0, self._run_simulation_from_slider)
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
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if isinstance(ctx, dict) and ctx.get("active"):
            return not bool(ctx.get("fast_mode"))
        worker = getattr(self, "_simulation_worker", None)
        if worker is not None and self._worker_is_running(worker):
            return not bool(getattr(worker, "_fast_mode", False))
        return bool(getattr(self, "_simulation_running", False)) and (not bool(getattr(self, "_slider_simulation_active", False)))

    def _has_active_fast_preview_in_flight(self) -> bool:
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if isinstance(ctx, dict) and ctx.get("active") and bool(ctx.get("fast_mode")):
            return True
        worker = getattr(self, "_simulation_worker", None)
        if worker is not None and self._worker_is_running(worker):
            return bool(getattr(worker, "_fast_mode", False))
        return bool(getattr(self, "_simulation_running", False)) and bool(getattr(self, "_slider_simulation_active", False))

    def _stale_fast_request_still_owns_current_state(self, request_id: int) -> bool:
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if isinstance(ctx, dict) and ctx.get("active"):
            if not bool(ctx.get("fast_mode")):
                return False
            ctx_request_id = ctx.get("request_id")
            if ctx_request_id is None:
                return True
            return int(ctx_request_id) == int(request_id)
        worker = getattr(self, "_simulation_worker", None)
        if worker is not None:
            if not bool(getattr(worker, "_fast_mode", False)):
                return False
            worker_request_id = getattr(worker, "_request_id", None)
            if worker_request_id is None:
                return True
            return int(worker_request_id) == int(request_id)
        return bool(getattr(self, "_slider_simulation_active", False))

    def _preview_request_can_display(self, request_id: Optional[int]) -> bool:
        if request_id is None:
            return True
        latest_request_id = int(getattr(self, "_latest_sim_request_id", 0))
        if int(request_id) == latest_request_id:
            return True
        suppress_discard_ui = int(getattr(self, "_discarded_slider_preview_generation_id", 0) or 0) == latest_request_id
        if suppress_discard_ui:
            return False
        return self._stale_fast_request_still_owns_current_state(int(request_id))

    def _prepare_simulation_shutdown_for_close(self) -> bool:
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
            still_running = self._cleanup_worker_safely(
                worker,
                "simulation worker (closeEvent)",
                retain_if_running=True,
                preserve_handlers_if_running=True,
            )
            if (not still_running) and getattr(self, "_simulation_worker", None) is worker:
                self._simulation_worker = None
        self._shutdown_batch_executor(force_terminate=True)
        self._prune_stopped_owned_simulation_workers()
        has_running_workers = self._has_running_owned_simulation_workers()
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
            # Disconnect SimulationWorker application-level signals safely. Do
            # not disconnect native QThread completion signals used for
            # lifecycle hooks.
            for signal_name in ("progress", "result_ready", "error"):
                if hasattr(worker, signal_name):
                    signal = getattr(worker, signal_name)
                    with suppress(TypeError):
                        signal.disconnect()

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

    def _effective_batch_worker_count(self, num_sets: int) -> int:
        return int(
            compute_effective_batch_workers(
                num_sets=max(0, int(num_sets)),
                max_parallel_workers=max(1, int(self._batch_parallel.max_parallel_workers)),
            )
        )

    def _shutdown_batch_executor(self, *, force_terminate: bool) -> None:
        timer = getattr(self, "_batch_future_poll_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        self._clear_pending_slider_plot_updates()
        prior_futures = int(len(self._batch_parallel.future_map or {})) + int(
            len(self._batch_parallel.superseded_future_map or {})
        )
        self._batch_parallel.shutdown(
            force_terminate=bool(force_terminate),
            record_nonfatal_exception=self._record_nonfatal_exception,
        )
        self._pool_eagerly_created = False
        if bool(getattr(self, "_debug_batch_parallel", False)):
            logger.info(
                "BATCH_PAR shutdown executor force=%s pending_futures=%s",
                bool(force_terminate),
                int(prior_futures),
            )

    def _has_active_parallel_batch_work(self) -> bool:
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel"):
            return True
        return bool(self._batch_parallel.future_map) or bool(self._batch_parallel.superseded_future_map)

    def _parallel_batch_pool_settings_changed(self) -> None:
        if self._has_active_parallel_batch_work():
            self._batch_parallel.mark_pool_stale()
            return
        self._shutdown_batch_executor(force_terminate=False)

    def _ensure_parallel_batch_pool_eagerly_created(self) -> None:
        if self._pool_eagerly_created:
            return
        try:
            effective_workers = compute_effective_batch_workers(
                num_sets=max(1, int(self._batch_parallel.max_parallel_workers)),
                max_parallel_workers=max(1, int(self._batch_parallel.max_parallel_workers)),
            )
            self._batch_parallel.ensure_executor(
                max_workers=max(1, int(effective_workers))
            )
        except Exception:
            self._pool_eagerly_created = False
            return
        self._pool_eagerly_created = True

    def _cleanup_parallel_batch_executor_after_run(
        self,
        *,
        keep_executor_alive: bool,
        clear_pending_plot_updates: bool = False,
        stale_fast_handoff_after_display: bool = False,
    ) -> None:
        if bool(keep_executor_alive) and (not self._batch_parallel.is_pool_stale):
            if stale_fast_handoff_after_display:
                cancelled, running = self._batch_parallel.soft_supersede()
                timer = getattr(self, "_batch_future_poll_timer", None)
                if running > 0 and timer is not None:
                    timer.start()
                if bool(getattr(self, "_debug_batch_parallel", False)):
                    logger.info(
                        "BATCH_PAR soft handoff after stale preview display cancelled=%s running=%s",
                        int(cancelled),
                        int(running),
                    )
            else:
                self._batch_parallel.reset_active_run_state()
            self._stop_batch_future_poll_timer_if_idle()
            if bool(clear_pending_plot_updates):
                self._clear_pending_slider_plot_updates()
            if bool(getattr(self, "_debug_batch_parallel", False)):
                logger.info("BATCH_PAR keeping executor alive after slider batch completion")
            return
        self._shutdown_batch_executor(force_terminate=False)

    def _supersede_parallel_batch_run_soft(self) -> None:
        """
        Supersede the active parallel run without destroying the process pool.

        Used by slider-triggered restarts to preserve worker processes and avoid
        pool recreation on every minor parameter update.
        """
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel"):
            ctx["active"] = False
            self._batch_run_context = dict(ctx)

        timer = getattr(self, "_batch_future_poll_timer", None)
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

    def _invalidate_slider_preview_work(self) -> None:
        previous_latest_request_id = int(getattr(self, "_latest_sim_request_id", 0))
        invalidation_request_id = int(self._next_sim_request_id())
        self._discarded_slider_preview_generation_id = int(invalidation_request_id)
        self._pending_slider_simulation = False
        self._pending_slider_target_set_ids = ()
        if previous_latest_request_id > 0:
            self._pending_slider_sim_request_id = int(previous_latest_request_id)
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
        pending_valid_set_ids = pending.valid_set_ids
        pending_allow_fallback = bool(pending.allow_fallback)

        cache_key = str(cache_key or pending_cache_key or "")
        request_id = pending_request_id if request_id is None else request_id
        run_id = pending_run_id if run_id is None else run_id
        if not cache_key:
            return False
        request_accepted = (
            self._preview_request_can_display(request_id)
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
    # Batch futures polling/consumption
    # ------------------------------------------------------------------
    def _pop_all_stale_parallel_batch_futures(self) -> None:
        self._batch_parallel.future_map.clear()
        self._batch_parallel.future_meta.clear()
        self._batch_parallel.superseded_future_map.clear()
        self._batch_parallel.superseded_future_meta.clear()

    def _reset_parallel_batch_run_and_shutdown_executor(self) -> None:
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel"):
            ctx["active"] = False
            self._batch_run_context = dict(ctx)
        self.shutdown_batch_executor(force_terminate=True)
        self._pop_all_stale_parallel_batch_futures()
        self._drain_batch_completion_queue()

    def _surface_current_parallel_batch_pool_failure_to_ui(self, error_msg: object) -> None:
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if not (isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel")):
            return
        if self._batch_parallel.executor is None:
            return
        self.on_simulation_error(
            error_msg,
            run_id=int(ctx.get("run_id") or 0),
            fast_mode=bool(ctx.get("fast_mode")),
            request_id=int(ctx.get("request_id") or 0),
            batch_set="",
            batch_set_id="",
            cache_key=str(ctx.get("cache_key") or ""),
        )

    def _consume_parallel_batch_future(
        self,
        *,
        set_id: str,
        fut: Any,
        run_id: int,
        request_id: int,
        fast_mode: bool,
        cache_key: str,
        source: str,
        completed_ts: Optional[float] = None,
    ) -> bool:
        sid = str(set_id or "")
        meta = dict((self._batch_parallel.future_meta or {}).get(sid) or {})
        meta["set_name"] = str(meta.get("set_name") or sid)
        set_name = meta["set_name"]
        self._batch_parallel.future_map.pop(sid, None)
        self._batch_parallel.future_meta.pop(sid, None)

        try:
            payload = fut.result()
        except Exception as exc:
            self._record_nonfatal_exception(
                f"Parallel batch future failed while retrieving result (set_id={sid}, source={str(source)})",
                exc,
            )
            self.on_simulation_error(
                f"Simulation failed:\n\n{exc}",
                run_id=run_id,
                fast_mode=fast_mode,
                request_id=request_id,
                batch_set=set_name,
                batch_set_id=sid,
                cache_key=cache_key,
            )
            self._reset_parallel_batch_run_and_shutdown_executor()
            return False

        if isinstance(payload, dict) and payload.get("success") is False and isinstance(payload.get("error"), dict):
            self.on_simulation_error(
                payload["error"],
                run_id=run_id,
                fast_mode=fast_mode,
                request_id=request_id,
                batch_set=set_name,
                batch_set_id=sid,
                cache_key=cache_key,
            )
            self._reset_parallel_batch_run_and_shutdown_executor()
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
            self.on_simulation_complete(
                payload,
                run_id=run_id,
                fast_mode=fast_mode,
                request_id=request_id,
                batch_set=set_name,
                batch_set_id=sid,
                cache_key=cache_key,
            )
        except Exception as exc:
            self._record_nonfatal_exception(
                f"Unhandled exception while handling completed batch future (set_id={sid}, source={str(source)})",
                exc,
            )
            try:
                self.on_simulation_error(
                    f"Simulation failed:\n\n{exc}",
                    run_id=run_id,
                    fast_mode=fast_mode,
                    request_id=request_id,
                    batch_set=set_name,
                    batch_set_id=sid,
                    cache_key=cache_key,
                )
            except Exception as ui_exc:
                self._record_nonfatal_exception(
                    "Failed to surface simulation-complete handling failure to UI",
                    ui_exc,
                )
            self._reset_parallel_batch_run_and_shutdown_executor()
            return False
        return True

    def _consume_superseded_parallel_batch_future(
        self,
        *,
        owner_key: str,
        fut: Any,
    ) -> bool:
        owner = str(owner_key or "")
        meta = dict((self._batch_parallel.superseded_future_meta or {}).get(owner) or {})
        self._batch_parallel.superseded_future_map.pop(owner, None)
        self._batch_parallel.superseded_future_meta.pop(owner, None)

        set_id = str(meta.get("set_id") or owner)
        set_name = str(meta.get("set_name") or set_id)
        try:
            payload = fut.result()
        except Exception as exc:
            self._record_nonfatal_exception(
                f"Superseded parallel batch future failed after soft supersede (set_id={set_id}, set_name={set_name})",
                exc,
            )
            logger.debug(
                "Stale superseded future failed after active run moved on (set_id=%s, set_name=%s)",
                set_id,
                set_name,
                exc_info=True,
            )
            return True

        if isinstance(payload, dict) and payload.get("success") is False and isinstance(payload.get("error"), dict):
            error_payload = coerce_simulation_failure(payload["error"])
            error_text = simulation_failure_user_message(error_payload) or "Unknown superseded batch error"
            self._record_nonfatal_exception(
                f"Superseded parallel batch future returned error payload after soft supersede (set_id={set_id}, set_name={set_name})",
                RuntimeError(error_text),
            )
        return True

    def _stop_batch_future_poll_timer_if_idle(self) -> None:
        ctx = getattr(self, "_batch_run_context", {}) or {}
        active_parallel = bool(isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel"))
        if active_parallel or bool(self._batch_parallel.future_map) or bool(self._batch_parallel.superseded_future_map):
            return
        timer = getattr(self, "_batch_future_poll_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()

    def _poll_parallel_batch_futures(self) -> None:
        ctx = getattr(self, "_batch_run_context", {}) or {}
        active_parallel = bool(isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel"))
        has_superseded = bool(self._batch_parallel.superseded_future_map)
        if not active_parallel and not has_superseded:
            self._stop_batch_future_poll_timer_if_idle()
            return

        run_id = int(ctx.get("run_id") or 0) if active_parallel else 0
        request_id = int(ctx.get("request_id") or 0) if active_parallel else 0
        fast_mode = bool(ctx.get("fast_mode")) if active_parallel else False
        cache_key = str(ctx.get("cache_key") or "") if active_parallel else ""
        processed_ids: Set[str] = set()

        try:
            completion_queue = self._batch_parallel.completed_queue
            while True:
                try:
                    sid_raw, completed_ts = completion_queue.get_nowait()
                except Empty:
                    break
                except Exception as exc:
                    logger.debug("Failed reading batch completion queue: %s", exc, exc_info=True)
                    break
                sid = str(sid_raw or "")
                if not sid or sid in processed_ids:
                    continue
                fut = (self._batch_parallel.future_map or {}).get(sid) if active_parallel else None
                if fut is None or not fut.done():
                    continue
                processed_ids.add(sid)
                if not self._consume_parallel_batch_future(
                    set_id=sid,
                    fut=fut,
                    run_id=run_id,
                    request_id=request_id,
                    fast_mode=fast_mode,
                    cache_key=cache_key,
                    source="callback",
                    completed_ts=float(completed_ts),
                ):
                    return
                ctx = getattr(self, "_batch_run_context", {}) or {}
                active_parallel = bool(isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel"))
                if not active_parallel:
                    run_id = 0
                    request_id = 0
                    fast_mode = False
                    cache_key = ""

            if active_parallel:
                for set_id, fut in list((self._batch_parallel.future_map or {}).items()):
                    sid = str(set_id or "")
                    if not sid or sid in processed_ids:
                        continue
                    if not fut.done():
                        continue
                    processed_ids.add(sid)
                    if not self._consume_parallel_batch_future(
                        set_id=sid,
                        fut=fut,
                        run_id=run_id,
                        request_id=request_id,
                        fast_mode=fast_mode,
                        cache_key=cache_key,
                        source="scan",
                    ):
                        return

                    ctx = getattr(self, "_batch_run_context", {}) or {}
                    active_parallel = bool(isinstance(ctx, dict) and ctx.get("active") and ctx.get("parallel"))
                    if not active_parallel:
                        break

            for owner_key, fut in list((self._batch_parallel.superseded_future_map or {}).items()):
                if not fut.done():
                    continue
                if not self._consume_superseded_parallel_batch_future(owner_key=str(owner_key), fut=fut):
                    return
            if (
                self._batch_parallel.is_pool_stale
                and (not self._batch_parallel.future_map)
                and (not self._batch_parallel.superseded_future_map)
            ):
                self._shutdown_batch_executor(force_terminate=False)
        except Exception as exc:
            # Architecture note (polling safety net):
            # This broad catch is a last-resort guard for the QTimer-driven poll
            # loop. If polling raises unexpectedly, letting the exception escape
            # can leave the GUI in a silent "stuck" state with a live executor
            # and no further timer ticks. We log, surface an error to the UI when
            # possible, and forcefully terminate the parallel executor to keep
            # the application recoverable.
            self._record_nonfatal_exception("Unhandled exception while polling parallel batch futures", exc)
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
            self._shutdown_batch_executor(force_terminate=True)
            return

        self._stop_batch_future_poll_timer_if_idle()

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
        last_name = str(self.ui.slider.last_slider_change_name() or "").strip()
        snapshot_set_ids = [str(set_id) for set_id in (target_set_ids or ()) if str(set_id)]
        if not last_name or (last_name.startswith("init:") and not snapshot_set_ids):
            return [int(row) for row in (fallback_rows or [])]
        if not snapshot_set_ids:
            try:
                snapshot_set_ids = [
                    str(set_id)
                    for set_id in (self.ui.slider.slider_gesture_target_set_ids_snapshot() or [])
                    if str(set_id)
                ]
            except Exception:
                snapshot_set_ids = []
        if not snapshot_set_ids:
            return [int(row) for row in (fallback_rows or [])]

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
        if resolved_rows:
            return resolved_rows
        return [int(row) for row in (fallback_rows or [])]

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
        pending_seed_for_set = _pending_initial_seed_for_set(pending_init_seed, set_name=str(set_name))
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
        if not bool(pending_init_applied):
            pending_init_applied = bool(ctx.get("pending_init_applied", False)) if isinstance(ctx, Mapping) else False
        if not bool(pending_init_applied):
            return
        try:
            self.ui.mechanism_helpers.invalidate_pending_init_preserved_results_after_failed_run()
        except Exception as exc:
            self._record_nonfatal_exception(
                "Failed to invalidate preserved pending-init results after explicit run failure",
                exc,
            )
        if isinstance(ctx, dict) and ctx.get("pending_init_applied", False):
            updated_ctx = dict(ctx)
            updated_ctx["pending_init_applied"] = False
            self._batch_run_context = updated_ctx

    def _requeue_preserved_pending_slider_replay_after_preflight_abort(self) -> None:
        pending_target_set_ids = [str(set_id) for set_id in self._pending_slider_target_set_ids if str(set_id)]
        if (not bool(self._pending_slider_simulation)) and (not pending_target_set_ids):
            return
        self.queue_pending_slider_preview_replay(
            target_set_ids=pending_target_set_ids,
            request_id=self._next_slider_preview_request_id(),
        )

    def _cancel_pending_slider_preview_replay_after_canonical_reset(self, reset_set_ids: Sequence[str]) -> None:
        if self._preserve_pending_slider_preview_replay_excluding(reset_set_ids):
            return
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

    def _run_simulation_from_slider(self):
        worker = self._simulation_worker
        request_id = getattr(self, "_pending_slider_sim_request_id", None)
        pending_target_set_ids = list(self._pending_slider_target_set_ids)
        latest_id = int(getattr(self, "_latest_sim_request_id", 0))
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
        if request_id is not None and int(request_id) > latest_id:
            self._latest_sim_request_id = int(request_id)
            latest_id = int(request_id)
        elif request_id is not None and int(request_id) != latest_id:
            logger.debug(
                "Discarding stale slider simulation request (request_id=%s, latest=%s)",
                request_id,
                latest_id,
            )
            self._pending_slider_simulation = False
            self._pending_slider_sim_request_id = None
            self._pending_slider_target_set_ids = ()
            return
        if request_id is None:
            request_id = self._next_sim_request_id()
            self._pending_slider_sim_request_id = request_id
        self._discarded_slider_preview_generation_id = None
        self.ui.slider.set_slider_triggered_simulation(True)

        if (not self.ui.run_ui.run_button_is_enabled()) and (
            worker is None or not getattr(worker, "_fast_mode", False)
        ):
            logger.debug("Full simulation in progress; deferring slider update")
            self._pending_slider_simulation = True
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
                self._supersede_parallel_batch_run_soft()
                self._simulation_running = False
                self._slider_simulation_active = False
                ctx = getattr(self, "_batch_run_context", {}) or {}
            else:
                logger.debug("Simulation already active; deferring slider update")
                self._pending_slider_simulation = True
                return

        if self._worker_is_running(worker):
            logger.debug("Simulation currently running; deferring slider update")
            self._pending_slider_simulation = True
            return

        if isinstance(ctx, dict) and ctx.get("active") and bool(ctx.get("fast_mode")):
            logger.debug("Fast slider run currently running; deferring slider update")
            self._pending_slider_simulation = True
            return
        self._prune_stopped_owned_simulation_workers()
        if self._has_running_owned_simulation_workers():
            logger.warning(
                "Slider-triggered run blocked while previous simulation worker shutdown remains in progress"
            )
            self._pending_slider_simulation = True
            self.ui.slider.set_slider_triggered_simulation(False)
            self._simulation_running = False
            self._slider_simulation_active = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self.ui.run_ui.set_status_text("Cancelling previous simulation...")
            return

        self._pending_slider_simulation = False
        self._pending_slider_target_set_ids = ()

        self._simulation_running = True
        self.ui.run_ui.set_stop_button_enabled(True)
        self._slider_simulation_active = True

        logger.info("Starting slider-triggered simulation")
        self.ui.run_ui.set_status_text("Updating simulation...")
        self.ui.run_ui.set_sim_progress_value(0)
        selected_rows = self.ui.batch.batch_rows_for_scope("selected")
        selected_rows = self._slider_target_rows_for_dispatch(
            selected_rows,
            target_set_ids=pending_target_set_ids,
        )

        self.run_simulation_internal(
            fast_mode=True,
            request_id=int(request_id),
            batch_rows=selected_rows,
            reuse_parallel_executor=True,
        )

    def _run_simulation(self):
        if not self.ui.mechanism.auto_lock_for_run():
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
            reuse_parallel_executor=bool(len(rows_to_run) > 1),
        )

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------
    def _cancel_active_run_for_restart(self) -> None:
        ctx = getattr(self, "_batch_run_context", {}) or {}
        if isinstance(ctx, dict) and ctx.get("active"):
            ctx["active"] = False
            self._batch_run_context = dict(ctx)
        self._shutdown_batch_executor(force_terminate=True)
        worker = getattr(self, "_simulation_worker", None)
        if worker is not None:
            try:
                if hasattr(worker, "cancel"):
                    worker.cancel()
            except Exception as exc:
                self._record_nonfatal_exception("Failed to cancel active worker during restart", exc)
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

        executor = self._batch_parallel.executor
        if executor is None:
            try:
                executor = self._batch_parallel.ensure_executor(max_workers=int(max_workers))
            except Exception as exc:
                logger.warning("Parallel batch executor unavailable; falling back to serial path: %s", exc)
                ctx["parallel"] = False
                self._batch_run_context = dict(ctx)
                self._start_next_batch_simulation()
                return
            if bool(getattr(self, "_debug_batch_parallel", False)):
                logger.info("BATCH_PAR executor created workers=%s run_id=%s", int(max_workers), int(run_id))
        elif bool(getattr(self, "_debug_batch_parallel", False)):
            logger.info("BATCH_PAR executor reused workers=%s run_id=%s", int(max_workers), int(run_id))

        timer = getattr(self, "_batch_future_poll_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        self._clear_pending_slider_plot_updates()
        self._batch_parallel.reset_active_run_state()

        mechanism_text = str(ctx.get("full_dsl") or "")
        solver_config = dict(ctx.get("solver_config") or {})
        t_end = float(ctx.get("t_end") or 0.0)
        signature = str(ctx.get("mechanism_signature") or "")
        request_id = int(ctx.get("request_id") or 0)
        execution_request_by_set_id = {
            str(set_id): dict(payload)
            for set_id, payload in dict(ctx.get("execution_request_by_set_id") or {}).items()
            if str(set_id) and isinstance(payload, dict)
        }
        mechanism_text_by_set_id = {
            str(set_id): str(text)
            for set_id, text in dict(ctx.get("mechanism_text_by_set_id") or {}).items()
            if str(set_id)
        }
        mechanism_signature_by_set_id = {
            str(set_id): str(sig)
            for set_id, sig in dict(ctx.get("mechanism_signature_by_set_id") or {}).items()
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
                ctx["active"] = False
                self._batch_run_context = dict(ctx)
                self._shutdown_batch_executor(force_terminate=True)
                self._simulation_running = False
                self.ui.run_ui.set_run_button_enabled(True)
                self.ui.run_ui.set_stop_button_enabled(False)
                self._slider_simulation_active = False
                self._invalidate_preserved_pending_init_results_after_failed_run(ctx=ctx)
                return
            pending_seed_for_set = _pending_initial_seed_for_set(pending_seed, set_name=str(set_name))
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
                "mechanism_text": mechanism_text_by_set_id.get(str(set_id), mechanism_text),
                "mechanism_signature": mechanism_signature_by_set_id.get(str(set_id), signature),
                "simulation_identity": simulation_identity_by_set_id.get(str(set_id)),
                "initials": dict(initials_dict),
                "t_span": (0.0, float(t_end)),
                "solver_config": dict(solver_config),
                "include_mechanism_in_result_payload": self._include_mechanism_in_result_payload(
                    fast_mode=bool(ctx.get("fast_mode")),
                    batch_set_id=str(set_id),
                    context=ctx,
                ),
            }
            execution_request = execution_request_by_set_id.get(str(set_id))
            if isinstance(execution_request, dict):
                if not bool(ctx.get("fast_mode")):
                    execution_request = dict(execution_request)
                    execution_request["prepared_payload"] = None
                task["execution_request"] = dict(execution_request)
            fut = executor.submit(run_batch_simulation_task, task)
            sid = str(set_id)
            self._batch_parallel.future_map[sid] = fut
            self._batch_parallel.future_meta[sid] = {"set_name": str(set_name)}
            try:
                fut.add_done_callback(lambda _fut, sid=sid: self._enqueue_parallel_batch_completion(sid))
            except Exception as exc:
                self._record_nonfatal_exception(
                    f"Failed to attach completion callback for batch future (set_id={sid})",
                    exc,
                )

        if not self._batch_parallel.future_map:
            ctx["active"] = False
            self._batch_run_context = dict(ctx)
            self._shutdown_batch_executor(force_terminate=False)
            self._simulation_running = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            return

        total = len(self._batch_parallel.future_map)
        if bool(getattr(self, "_debug_batch_parallel", False)):
            logger.info(
                "BATCH_PAR submitted run_id=%s sets=%s workers=%s",
                int(run_id),
                int(total),
                int(max_workers),
            )
        self.ui.run_ui.set_sim_progress_value(0)
        self.ui.run_ui.set_status_text(f"Running {total} sets in parallel ({max_workers} workers)...")
        if hasattr(self, "_batch_future_poll_timer"):
            self._batch_future_poll_timer.start()

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
            ctx["active"] = False
            self._batch_run_context = dict(ctx)
            self._simulation_running = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self._slider_simulation_active = False
            self._invalidate_preserved_pending_init_results_after_failed_run(ctx=ctx)
            return
        pending_seed = ctx.get("pending_init_seed") if isinstance(ctx, dict) else None
        pending_seed_for_set = _pending_initial_seed_for_set(pending_seed, set_name=str(set_name))
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
        execution_request_by_set_id = {
            str(candidate_set_id): dict(payload)
            for candidate_set_id, payload in dict(ctx.get("execution_request_by_set_id") or {}).items()
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
        candidate_request = execution_request_by_set_id.get(set_id)
        if allow_batch_global_fallback and candidate_request is None:
            candidate_request = ctx.get("execution_request")
        if isinstance(candidate_request, dict):
            execution_request = candidate_request
            if not bool(fast_mode):
                execution_request = dict(execution_request)
                execution_request["prepared_payload"] = None
            initials_dict = dict(candidate_request.get("initials") or initials_dict)
            solver_config = dict(candidate_request.get("solver_config") or solver_config)
            request_t_span = candidate_request.get("t_span") or (0.0, t_end)
            try:
                t_end = float(request_t_span[1])
            except (TypeError, ValueError, IndexError):
                t_end = float(t_end)

        from kindred.gui.simulation_worker import SimulationWorker

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

        include_mechanism_in_result_payload = self._include_mechanism_in_result_payload(
            fast_mode=bool(fast_mode),
            batch_set_id=set_id,
            context=ctx,
        )

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
        if isinstance(execution_request, dict):
            self._simulation_worker._execution_request = dict(execution_request)  # type: ignore[attr-defined]

        self._simulation_worker.progress.connect(self.on_simulation_progress)
        self._simulation_worker.result_ready.connect(
            lambda payload, _rid=run_id, _fast=bool(fast_mode), _req=int(request_id), _set=set_name, _sid=set_id, _key=cache_key: self.on_simulation_complete(
                payload, run_id=_rid, fast_mode=_fast, request_id=_req, batch_set=_set, batch_set_id=_sid, cache_key=_key
            )
        )
        self._simulation_worker.error.connect(
            lambda msg, _rid=run_id, _fast=bool(fast_mode), _req=int(request_id), _set=set_name, _sid=set_id, _key=cache_key: self.on_simulation_error(
                msg, run_id=_rid, fast_mode=_fast, request_id=_req, batch_set=_set, batch_set_id=_sid, cache_key=_key
            )
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
        reuse_parallel_executor: bool = False,
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
                logger.debug("Fast slider run already in flight; recording latest-only pending request")
                self._pending_slider_simulation = True
                if request_id is not None:
                    self._pending_slider_sim_request_id = int(request_id)
                return

        def _clear_slider_triggered_preflight_state() -> None:
            if bool(fast_mode):
                self.ui.slider.set_slider_triggered_simulation(False)

        if request_id is None:
            request_id = self._next_sim_request_id()
        if batch_rows is None:
            batch_rows = self.ui.batch.batch_rows_for_scope("selected")
        row_count = int(self.ui.batch.batch_store_row_count())
        batch_rows = [int(r) for r in (batch_rows or []) if 0 <= int(r) < int(row_count)]
        if not batch_rows:
            if int(row_count) > 0:
                batch_rows = [0]
            else:
                self.ui.dialogs.message_box_warning("No Sets", "Add at least one set before running.")
                self._simulation_running = False
                self.ui.run_ui.set_run_button_enabled(True)
                self.ui.run_ui.set_stop_button_enabled(False)
                _clear_slider_triggered_preflight_state()
                if not bool(fast_mode):
                    self._requeue_preserved_pending_slider_replay_after_preflight_abort()
                return
        invalid = self.ui.batch.batch_model_validate_rows(batch_rows)
        if invalid:
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

        reactions_text_raw = self.ui.mechanism.mechanism_reactions_text_raw()
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

        if (not bool(fast_mode)) and pending_init_seed and pending_init_rewrite:
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
        reactions_text = strip_reaction_dsl_initial_concentrations(
            reactions_text_raw if pending_init_applied else migrated
        )
        state_network_dsl = self.ui.mechanism.mechanism_state_network_dsl_raw()
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
            self._invalidate_preserved_pending_init_results_after_failed_run(
                pending_init_applied=bool(pending_init_applied)
            )
            self.ui.dialogs.message_box_warning(
                "No Mechanism",
                "Please define reactions or state network in the Mechanism editor first.",
            )
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
        if solver_warning:
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
        slider_runtime: Optional[BoundMechanism] = None
        if bool(fast_mode):
            target_runtime_set_ids = list(queue_ids)
            if (not target_runtime_set_ids) and primary_set_id:
                target_runtime_set_ids = [str(primary_set_id)]
            for index, set_id in enumerate(target_runtime_set_ids):
                candidate_runtime = self.ui.runtime.prepare_slider_runtime(set_id=str(set_id))
                if candidate_runtime is None:
                    continue
                if slider_runtime is None:
                    slider_runtime = candidate_runtime
                try:
                    if self.ui.runtime.apply_slider_overrides_to_bindings(candidate_runtime, set_id=str(set_id)):
                        try:
                            worker_payload = dict(candidate_runtime.as_worker_payload())
                            prepared_payload_by_set_id[str(set_id)] = worker_payload
                            if hasattr(candidate_runtime, "as_serializable_execution_payload"):
                                execution_payload = dict(candidate_runtime.as_serializable_execution_payload())
                            else:
                                execution_payload = dict(worker_payload)
                                execution_payload.pop("rhs", None)
                                execution_payload["version"] = 2
                            execution_prepared_payload_by_set_id[str(set_id)] = execution_payload
                        except Exception:
                            prepared_payload_by_set_id.pop(str(set_id), None)
                            execution_prepared_payload_by_set_id.pop(str(set_id), None)
                    else:
                        prepared_payload_by_set_id.pop(str(set_id), None)
                        execution_prepared_payload_by_set_id.pop(str(set_id), None)
                finally:
                    # Mark dirty between iterations so each set gets a fresh
                    # runtime, and also after the final iteration so later
                    # single-set interactions do not reuse the last set's
                    # bindings.
                    if len(target_runtime_set_ids) > 1:
                        self.ui.runtime.set_slider_runtime_dirty(True)
            if primary_set_id:
                prepared_payload = prepared_payload_by_set_id.get(str(primary_set_id))
            if prepared_payload is None and prepared_payload_by_set_id:
                prepared_payload = dict(next(iter(prepared_payload_by_set_id.values())))

        if (not bool(fast_mode)) or (not list(self.ui.batch.batch_store_visible_species())):
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
            self._invalidate_preserved_pending_init_results_after_failed_run(
                pending_init_applied=bool(pending_init_applied)
            )
            self.ui.dialogs.message_box_warning("Invalid t_end", f"Fix t_end before running:\n\n{exc}")
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

        execution_request_by_set_id: Dict[str, Dict[str, Any]] = {}
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
                        self.ui.dialogs.message_box_warning(
                            "Invalid Initial Conditions",
                            f"Set '{set_name}' has invalid initial conditions:\n\n{exc}",
                        )
                        self._simulation_running = False
                        self.ui.run_ui.set_run_button_enabled(True)
                        self.ui.run_ui.set_stop_button_enabled(False)
                        self._slider_simulation_active = False
                        _clear_slider_triggered_preflight_state()
                        return
                    mechanism_signature_by_set_id[str(set_id)] = batch_mechanism_signature(
                        simulation_identity=identity,
                    )
                    execution_request_by_set_id[str(set_id)] = SimulationExecutionRequest(
                        prepared_payload=dict(prepared_execution_payload),
                        initials=dict(initials_dict),
                        t_span=(0.0, float(t_end)),
                        solver_config=dict(solver_config),
                        mechanism_text=str(request_mechanism_text),
                        simulation_identity=identity.to_payload(),
                    ).to_payload()
                else:
                    mechanism_signature_by_set_id[str(set_id)] = batch_mechanism_signature(
                        mechanism_text=str(request_mechanism_text),
                        temperature_K=float(solver_config.get("temperature_K") or 298.15),
                        use_sparse_jacobian=bool(solver_config.get("use_sparse_jacobian")),
                        wegscheider_cyclicity_enabled=bool(
                            solver_config.get("wegscheider_cyclicity_enabled", False)
                        ),
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

            mechanism_signature_by_set_id[str(set_id)] = batch_mechanism_signature(
                simulation_identity=identity,
            )

            execution_request_by_set_id[str(set_id)] = SimulationExecutionRequest(
                prepared_payload=dict(prepared_execution_payload) if isinstance(prepared_execution_payload, dict) else None,
                initials=dict(initials_dict),
                t_span=(0.0, float(t_end)),
                solver_config=dict(solver_config),
                mechanism_text=str(request_mechanism_text),
                simulation_identity=identity.to_payload(),
            ).to_payload()

        scope_identity = SimulationScopeIdentity.build(
            queue_ids=queue_ids,
            identity_by_set_id={
                set_id: payload
                for set_id, payload in simulation_identity_by_set_id.items()
            },
        )
        cache_key = self.ui.batch.batch_cache_key(scope_identity=scope_identity)
        explicit_preview_scope_set_ids = None
        if bool(fast_mode):
            self._batch_cache.active_preview_cache_key = cache_key
        else:
            self._batch_cache.active_cache_key = cache_key
            self._batch_cache.active_cache_preview_token = None

        if bool(reuse_parallel_executor):
            self._supersede_parallel_batch_run_soft()
        else:
            self._shutdown_batch_executor(force_terminate=True)

        self._release_current_simulation_worker()

        explicit_valid_set_ids = tuple(str(set_id) for set_id in queue_ids) if queue_ids else None
        if not bool(fast_mode):
            self._batch_cache.active_cache_preview_scope_set_ids = None
        if not bool(fast_mode):
            self._batch_cache.active_cache_valid_set_ids = explicit_valid_set_ids
            self._batch_cache.active_cache_invalidated_set_ids = None
        elif explicit_valid_set_ids:
            self._batch_cache.active_preview_scope_set_ids = explicit_valid_set_ids
        effective_workers = self._effective_batch_worker_count(len(queue_ids))
        parallel_mode = bool(effective_workers > 1 and len(queue_ids) > 1)
        retain_prepared_payloads_in_context = not (bool(parallel_mode) and not bool(fast_mode))
        pending_dirty_reset_generation_by_set_id: Dict[str, int] = {}
        if not bool(fast_mode):
            for set_id in queue_ids:
                set_id_s = str(set_id or "").strip()
                if not set_id_s:
                    continue
                try:
                    if not bool(self.ui.slider.has_dirty_state_for_set(set_id_s)):
                        continue
                    pending_dirty_reset_generation_by_set_id[set_id_s] = int(
                        self.ui.slider.dirty_state_generation(set_id_s)
                    )
                except Exception as exc:
                    self._record_nonfatal_exception(
                        f"Failed to snapshot dirty-state generation for explicit run reset candidate {set_id_s}",
                        exc,
                    )
        run_id = None
        if parallel_mode:
            self._run_sequence_id = int(getattr(self, "_run_sequence_id", 0)) + 1
            run_id = int(self._run_sequence_id)
            self._active_run_id = int(run_id)

        primary_execution_request = None
        if not bool(fast_mode):
            if primary_set_id:
                primary_execution_request = execution_request_by_set_id.get(str(primary_set_id))
            if primary_execution_request is None and execution_request_by_set_id:
                primary_execution_request = dict(next(iter(execution_request_by_set_id.values())))

        self._batch_run_context = {
            "active": True,
            "request_id": int(request_id),
            "run_id": run_id,
            "fast_mode": bool(fast_mode),
            "reuse_parallel_executor": bool(reuse_parallel_executor),
            "keep_executor_alive": bool(reuse_parallel_executor and parallel_mode),
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
            "execution_request": (
                dict(primary_execution_request)
                if ((not bool(fast_mode)) and isinstance(primary_execution_request, dict))
                else None
            ),
            "execution_request_by_set_id": {
                str(set_id): dict(payload) for set_id, payload in execution_request_by_set_id.items()
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
            "pending_workspace_reset_set_ids": (
                list(pending_dirty_reset_generation_by_set_id.keys()) if not bool(fast_mode) else []
            ),
            "pending_dirty_reset_generation_by_set_id": (
                dict(pending_dirty_reset_generation_by_set_id) if not bool(fast_mode) else {}
            ),
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
            "explicit_cache_preview_scope_set_ids": explicit_preview_scope_set_ids,
            "explicit_cache_valid_set_ids": explicit_valid_set_ids,
            "explicit_cache_invalidated_set_ids": None,
            "preview_scope_set_ids": (tuple(str(set_id) for set_id in queue_ids) if bool(fast_mode) and queue_ids else None),
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
                bool(reuse_parallel_executor),
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
        if request_id is not None and int(request_id) != latest_request_id and bool(fast_mode):
            pending_request_id = getattr(self, "_pending_slider_sim_request_id", None)
            should_schedule_pending = bool(getattr(self, "_pending_slider_simulation", False)) or (
                pending_request_id is not None and int(pending_request_id) == latest_request_id
            )
            suppress_discard_ui = int(getattr(self, "_discarded_slider_preview_generation_id", 0) or 0) == latest_request_id
            stale_fast_owns_current_state = self._stale_fast_request_still_owns_current_state(int(request_id))
            display_current_preview = bool(stale_fast_owns_current_state) and (not bool(suppress_discard_ui))
            logger.debug(
                "Active fast completion superseded (request_id=%s, latest=%s, run_id=%s, schedule_pending=%s, suppress_discard_ui=%s, owns_current_state=%s, display_current_preview=%s)",
                request_id,
                latest_request_id,
                run_id,
                bool(should_schedule_pending),
                bool(suppress_discard_ui),
                bool(stale_fast_owns_current_state),
                bool(display_current_preview),
            )

            if display_current_preview:
                if should_schedule_pending:
                    self._pending_slider_simulation = True
                    ctx_check = getattr(self, "_batch_run_context", {}) or {}
                    active_parallel_batch = bool(
                        isinstance(ctx_check, dict) and ctx_check.get("active") and ctx_check.get("parallel")
                    )
                    if not active_parallel_batch:
                        stale_fast_handoff_after_display = bool(stale_fast_owns_current_state)
            else:
                ctx = getattr(self, "_batch_run_context", {}) or {}
                if stale_fast_owns_current_state:
                    if isinstance(ctx, dict) and ctx.get("active"):
                        ctx["active"] = False
                        self._batch_run_context = dict(ctx)

                    self._release_current_simulation_worker()

                    keep_executor_alive = bool(isinstance(ctx, dict) and ctx.get("parallel") and ctx.get("keep_executor_alive"))
                    self._cleanup_parallel_batch_executor_after_run(
                        keep_executor_alive=keep_executor_alive,
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

                if should_schedule_pending and (not shutdown_requested):
                    self._pending_slider_simulation = False
                    self._discarded_slider_preview_generation_id = None
                    QtCore.QTimer.singleShot(0, self._run_simulation_from_slider)
                else:
                    if stale_fast_owns_current_state and (not bool(suppress_discard_ui)):
                        try:
                            self.ui.run_ui.set_status_text("Ready")
                            self.ui.run_ui.set_sim_progress_value(0)
                        except Exception as exc:
                            self._record_nonfatal_exception(
                                "Failed to reset status/progress after superseded fast completion",
                                exc,
                            )
                    self._discarded_slider_preview_generation_id = None
                    self.clear_pending_slider_preview_replay(clear_plot_updates=False)
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

            ctx = getattr(self, "_batch_run_context", {}) or {}
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
                    preview_scope_ids = None
                    if isinstance(ctx, dict):
                        scope_ids = ctx.get("preview_scope_set_ids") or ()
                        preview_scope_ids = tuple(str(set_id) for set_id in scope_ids) if scope_ids else None
                    self._batch_cache.active_preview_scope_set_ids = preview_scope_ids
                else:
                    preview_token = None
                    preview_scope_set_ids = None
                    valid_set_ids = None
                    invalidated_set_ids = None
                    has_valid_subset = False
                    if isinstance(ctx, dict):
                        token = ctx.get("explicit_cache_preview_token")
                        preview_token = str(token) if token else None
                        scope_ids = ctx.get("explicit_cache_preview_scope_set_ids") or ()
                        preview_scope_set_ids = tuple(str(set_id) for set_id in scope_ids) if scope_ids else None
                        if "explicit_cache_valid_set_ids" in ctx:
                            has_valid_subset = True
                            subset_ids = ctx.get("explicit_cache_valid_set_ids") or ()
                            valid_set_ids = tuple(str(set_id) for set_id in subset_ids) if subset_ids else ()
                            invalid_subset_ids = ctx.get("explicit_cache_invalidated_set_ids") or ()
                            invalidated_set_ids = (
                                tuple(str(set_id) for set_id in invalid_subset_ids)
                                if invalid_subset_ids
                                else ()
                            )
                        else:
                            queue_ids = ctx.get("queue_ids") or ()
                            valid_set_ids = tuple(str(set_id) for set_id in queue_ids) if queue_ids else None
                    if has_valid_subset and valid_set_ids == ():
                        self._batch_cache.clear_active_selection_state()
                    else:
                        self._batch_cache.active_cache_key = cache_key
                        self._batch_cache.active_cache_preview_token = preview_token
                        self._batch_cache.active_cache_preview_scope_set_ids = preview_scope_set_ids
                        self._batch_cache.active_cache_valid_set_ids = valid_set_ids
                        self._batch_cache.active_cache_invalidated_set_ids = (
                            invalidated_set_ids if has_valid_subset else None
                        )

            redraw_valid_set_ids = None
            has_redraw_subset = False
            if isinstance(ctx, dict) and "explicit_cache_valid_set_ids" in ctx:
                has_redraw_subset = True
                subset_ids = ctx.get("explicit_cache_valid_set_ids") or ()
                redraw_valid_set_ids = tuple(str(set_id) for set_id in subset_ids) if subset_ids else ()
            elif self._batch_cache.active_cache_valid_set_ids is not None:
                has_redraw_subset = True
                redraw_valid_set_ids = tuple(str(set_id) for set_id in self._batch_cache.active_cache_valid_set_ids)

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
                if isinstance(ctx, dict) and cache_key:
                    updated_ctx = dict(ctx)
                    if str(self._batch_cache.active_cache_key or "") == str(cache_key):
                        preview_scope_ids_after = self._batch_cache.active_cache_preview_scope_set_ids
                        valid_set_ids_after = self._batch_cache.active_cache_valid_set_ids
                        updated_ctx["explicit_cache_preview_token"] = self._batch_cache.active_cache_preview_token
                        updated_ctx["explicit_cache_preview_scope_set_ids"] = (
                            tuple(str(set_id) for set_id in preview_scope_ids_after)
                            if preview_scope_ids_after is not None
                            else None
                        )
                        updated_ctx["explicit_cache_valid_set_ids"] = (
                            tuple(str(set_id) for set_id in valid_set_ids_after)
                            if valid_set_ids_after is not None
                            else None
                        )
                        invalidated_set_ids_after = self._batch_cache.active_cache_invalidated_set_ids
                        updated_ctx["explicit_cache_invalidated_set_ids"] = (
                            tuple(str(set_id) for set_id in invalidated_set_ids_after)
                            if invalidated_set_ids_after is not None
                            else None
                        )
                    else:
                        updated_ctx["explicit_cache_preview_token"] = None
                        updated_ctx["explicit_cache_preview_scope_set_ids"] = ()
                        updated_ctx["explicit_cache_valid_set_ids"] = ()
                        updated_ctx["explicit_cache_invalidated_set_ids"] = ()
                    ctx = updated_ctx
                    self._batch_run_context = updated_ctx

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

            if cache_key and batch_set_id:
                cache_store = self._batch_cache.preview_cache if bool(is_preview) else self._batch_cache.result_cache
                cache_token = str(cache_key)
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
                payload = build_batch_cache_entry(
                    t=t,
                    series=series,
                    algebra_scalars=(algebra_scalars if isinstance(algebra_scalars, dict) else None),
                    mechanism=cached_mechanism,
                    mechanism_text=str(mechanism_text),
                    simulation_identity=(
                        dict((ctx.get("simulation_identity_by_set_id") or {}).get(str(batch_set_id)) or {})
                        if isinstance(ctx, dict)
                        else None
                    ),
                    solver_config=(solver_config if isinstance(solver_config, dict) else None),
                    preview_batch_cache_token=(
                        self._preview_batch_cache_token_for_cached_result(batch_set_id=batch_set_id, context=ctx)
                        if bool(is_preview)
                        else None
                    ),
                    fallback_occurred=bool(fallback_occurred),
                    fallback_message=fallback_message,
                )
                cache_store.put(composite_key, payload)

            if (not is_preview) and is_primary and isinstance(ctx, dict):
                pending_seed = ctx.get("pending_init_seed")
                pending_rewrite = ctx.get("pending_init_rewrite")
                pending_applied = bool(ctx.get("pending_init_applied", False))
                if (
                    isinstance(pending_seed, dict)
                    and pending_seed
                    and isinstance(pending_rewrite, str)
                    and pending_rewrite
                    and not pending_applied
                ):
                    seed_for_ui: Dict[str, float] = {}
                    for sp, val in _pending_initial_seed_for_set(
                        pending_seed,
                        set_name=str(batch_set),
                    ).items():
                        float_val = _try_float(val)
                        if float_val is None:
                            continue
                        seed_for_ui[str(sp)] = float_val
                    applied = False
                    try:
                        applied = bool(
                            self.ui.mechanism_helpers.apply_pending_init_migration(
                                seed_sets={str(batch_set): dict(seed_for_ui)},
                                rewrite=str(pending_rewrite),
                            )
                        )
                    except Exception:
                        applied = False
                    if applied:
                        ctx["pending_init_applied"] = True
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

            if (not is_preview) and is_primary and isinstance(ctx, dict) and bool(ctx.get("pending_init_applied", False)):
                self.ui.mechanism_helpers.arm_pending_init_result_invalidation_guard(
                    rewrite=str(ctx.get("pending_init_rewrite") or "")
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
                queue_ids = list(ctx.get("queue_ids") or [])
                try:
                    pos = int(ctx.get("pos", 0))
                except Exception:
                    pos = 0
                try:
                    total = int(ctx.get("total") or len(queue_ids) or 1)
                except Exception:
                    total = max(1, len(queue_ids))

                if stale_fast_handoff_after_display:
                    ctx["active"] = False
                    self._batch_run_context = dict(ctx)
                    batch_queue_done = True
                elif bool(ctx.get("parallel")):
                    completed_ids = {str(s) for s in (ctx.get("completed_set_ids") or []) if str(s)}
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
                    pending_workspace_reset_set_ids = [
                        str(set_id) for set_id in (ctx.get("pending_workspace_reset_set_ids") or ()) if str(set_id)
                    ]
                    pending_dirty_reset_generation_by_set_id = {
                        str(set_id): int(generation)
                        for set_id, generation in dict(
                            ctx.get("pending_dirty_reset_generation_by_set_id") or {}
                        ).items()
                        if str(set_id)
                    }
                    eligible_reset_set_ids: list[str] = []
                    for set_id in pending_workspace_reset_set_ids:
                        expected_generation = pending_dirty_reset_generation_by_set_id.get(str(set_id))
                        if expected_generation is None:
                            continue
                        try:
                            if not bool(self.ui.slider.has_dirty_state_for_set(str(set_id))):
                                continue
                            current_generation = int(self.ui.slider.dirty_state_generation(str(set_id)))
                        except Exception as exc:
                            self._record_nonfatal_exception(
                                f"Failed to compare dirty-state generation for explicit reset candidate {set_id}",
                                exc,
                            )
                            continue
                        if current_generation != int(expected_generation):
                            continue
                        eligible_reset_set_ids.append(str(set_id))
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
                    if pending_workspace_reset_set_ids or pending_dirty_reset_generation_by_set_id:
                        updated_ctx = dict(ctx)
                        updated_ctx["pending_workspace_reset_set_ids"] = []
                        updated_ctx["pending_dirty_reset_generation_by_set_id"] = {}
                        ctx = updated_ctx
                        self._batch_run_context = updated_ctx
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
                        self._cancel_pending_slider_preview_replay_after_canonical_reset(
                            eligible_reset_set_ids
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
                self.ui.run_ui.set_sim_progress_value(100)
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
                keep_executor_alive = bool(
                    isinstance(ctx_for_cleanup, dict)
                    and ctx_for_cleanup.get("parallel")
                    and ctx_for_cleanup.get("keep_executor_alive")
                )
                self._cleanup_parallel_batch_executor_after_run(
                    keep_executor_alive=keep_executor_alive,
                    stale_fast_handoff_after_display=stale_fast_handoff_after_display,
                )
                self.ui.slider.set_slider_triggered_simulation(False)
                self._simulation_running = False
                self.ui.run_ui.set_run_button_enabled(True)
                self.ui.run_ui.set_stop_button_enabled(False)
                self._slider_simulation_active = False

                if self._pending_slider_simulation:
                    logger.debug("Processing pending slider update after completion")
                    self._pending_slider_simulation = False
                    for stop_fn in (
                        self.ui.slider.stop_variable_update_timer,
                        self.ui.slider.stop_species_slider_update_timer,
                    ):
                        stop_fn()
                    if not shutdown_requested:
                        QtCore.QTimer.singleShot(0, self._run_simulation_from_slider)
                self._clear_shutdown_request_after_close_cleanup()

    def _on_simulation_error(
        self,
        error_msg: object,
        run_id: Optional[int] = None,
        fast_mode: Optional[bool] = None,
        request_id: Optional[int] = None,
        *,
        batch_set: Optional[str] = None,
        batch_set_id: Optional[str] = None,
        cache_key: Optional[str] = None,
    ):
        error_payload = coerce_simulation_failure(error_msg)
        error_text = simulation_failure_user_message(error_payload)
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
        if request_id is not None and int(request_id) != latest_request_id and bool(fast_mode):
            pending_request_id = getattr(self, "_pending_slider_sim_request_id", None)
            should_schedule_pending = bool(getattr(self, "_pending_slider_simulation", False)) or (
                pending_request_id is not None and int(pending_request_id) == latest_request_id
            )
            suppress_discard_ui = int(getattr(self, "_discarded_slider_preview_generation_id", 0) or 0) == latest_request_id
            stale_fast_owns_current_state = self._stale_fast_request_still_owns_current_state(int(request_id))
            logger.debug(
                "Active fast error superseded (request_id=%s, latest=%s, run_id=%s, schedule_pending=%s, suppress_discard_ui=%s, owns_current_state=%s): %s",
                request_id,
                latest_request_id,
                run_id,
                bool(should_schedule_pending),
                bool(suppress_discard_ui),
                bool(stale_fast_owns_current_state),
                error_text,
            )

            ctx = getattr(self, "_batch_run_context", {}) or {}
            if stale_fast_owns_current_state:
                if isinstance(ctx, dict) and ctx.get("active"):
                    ctx["active"] = False
                    self._batch_run_context = dict(ctx)

                self._release_current_simulation_worker()
                self._shutdown_batch_executor(force_terminate=False)
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

            if should_schedule_pending:
                self._pending_slider_simulation = False
                self._discarded_slider_preview_generation_id = None
                QtCore.QTimer.singleShot(0, self._run_simulation_from_slider)
            else:
                if stale_fast_owns_current_state and (not bool(suppress_discard_ui)):
                    try:
                        self.ui.run_ui.set_status_text("Ready")
                        self.ui.run_ui.set_sim_progress_value(0)
                    except Exception as exc:
                        self._record_nonfatal_exception(
                            "Failed to reset status/progress after superseded fast error",
                            exc,
                        )
                self._discarded_slider_preview_generation_id = None
                self.clear_pending_slider_preview_replay(clear_plot_updates=False)
            return
        logger.warning("Simulation error surfaced to UI: %s", error_text)

        ctx = getattr(self, "_batch_run_context", {}) or {}
        if isinstance(ctx, dict) and ctx.get("active"):
            ctx["active"] = False
            self._batch_run_context = dict(ctx)
        self._release_current_simulation_worker()
        self._shutdown_batch_executor(force_terminate=True)
        self._clear_shutdown_request_after_close_cleanup()

        if not cancelled:
            self.ui.dialogs.message_box_critical(
                "Simulation Error",
                f"Simulation failed:\n\n{error_text}",
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
            if self._pending_slider_simulation:
                logger.debug("Resuming pending slider update after cancellation")
                self._pending_slider_simulation = False
                self.ui.slider.stop_variable_update_timer()
                self.ui.slider.stop_species_slider_update_timer()
                QtCore.QTimer.singleShot(0, self._run_simulation_from_slider)
        else:
            if (not bool(fast_mode)) and (
                bool(self._pending_slider_simulation) or bool(self._pending_slider_target_set_ids)
            ):
                logger.debug("Replaying pending slider update after explicit failure")
                self.queue_pending_slider_preview_replay(
                    target_set_ids=self._pending_slider_target_set_ids,
                    request_id=self._next_slider_preview_request_id(),
                )
                self.ui.slider.stop_variable_update_timer()
                self.ui.slider.stop_species_slider_update_timer()
                QtCore.QTimer.singleShot(0, self._run_simulation_from_slider)
            else:
                self.clear_pending_slider_preview_replay(clear_plot_updates=False)

        self._invalidate_preserved_pending_init_results_after_failed_run(
            ctx=ctx if isinstance(ctx, Mapping) else None,
        )

    def _stop_simulation(self):
        if not self._simulation_running:
            return

        logger.info("Stop simulation requested")

        ctx = getattr(self, "_batch_run_context", {}) or {}
        if isinstance(ctx, dict) and ctx.get("active"):
            ctx["active"] = False
            self._batch_run_context = dict(ctx)
        self._shutdown_batch_executor(force_terminate=True)

        if self._worker_is_running(self._simulation_worker):
            self._simulation_worker.cancel()
            logger.info("Cancellation requested from simulation worker")
            self.ui.run_ui.set_status_text("Cancelling simulation...")
        else:
            self._simulation_running = False
            self.ui.run_ui.set_run_button_enabled(True)
            self.ui.run_ui.set_stop_button_enabled(False)
            self.ui.run_ui.set_status_text("Ready")
            self.ui.run_ui.set_sim_progress_value(0)
