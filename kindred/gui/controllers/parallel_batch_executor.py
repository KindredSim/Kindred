from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

from kindred.core.batch_parallel import prewarm_worker_imports
from kindred.gui.project_schema import PROJECT_DEFAULTS


def _noop_record_nonfatal_exception(_message: str, _exc: BaseException) -> None:
    return None


@dataclass
class ParallelBatchExecutor:
    """
    Owns process-pool lifecycle and per-run future bookkeeping for batch simulations.

    This is kept Qt-free; SimulationController remains responsible for QTimer wiring
    and UI updates.
    """

    executor_factory: Callable[[int, bool], Any]
    max_parallel_workers: int = int(PROJECT_DEFAULTS["max_parallel_batch_workers"])
    limit_blas_threads_per_worker: bool = bool(PROJECT_DEFAULTS["limit_blas_threads_per_worker"])
    record_nonfatal_exception: Callable[[str, BaseException], None] = _noop_record_nonfatal_exception
    executor: Any = None
    future_map: Dict[str, Any] = field(default_factory=dict)
    future_meta: Dict[str, Dict[str, str]] = field(default_factory=dict)
    superseded_future_map: Dict[str, Any] = field(default_factory=dict)
    superseded_future_meta: Dict[str, Dict[str, str]] = field(default_factory=dict)
    completed_queue: SimpleQueue[Tuple[str, float]] = field(default_factory=SimpleQueue)
    _current_max_workers: Optional[int] = None
    _pool_stale: bool = False

    def reset_active_run_state(self) -> None:
        self.future_map = {}
        self.future_meta = {}

    def reset_run_state(self) -> None:
        self.reset_active_run_state()
        self.superseded_future_map = {}
        self.superseded_future_meta = {}
        self.drain_completion_queue()

    def drain_completion_queue(self) -> None:
        while True:
            try:
                self.completed_queue.get_nowait()
            except Empty:
                break
            except Exception:
                break

    def enqueue_completion(self, set_id: str) -> None:
        sid = str(set_id or "")
        if not sid:
            return
        ts = float(perf_counter())
        with suppress(Exception):
            self.completed_queue.put((sid, ts))

    def ensure_executor(self, *, max_workers: int) -> Any:
        requested_workers = max(1, int(max_workers))
        executor = self.executor
        if executor is None:
            return self._create_and_prewarm_executor(max_workers=requested_workers)

        current_workers = self._current_max_workers
        if current_workers is None:
            return executor

        if requested_workers <= current_workers:
            return executor

        self.shutdown(
            force_terminate=True,
            record_nonfatal_exception=self.record_nonfatal_exception,
        )
        return self._create_and_prewarm_executor(max_workers=requested_workers)

    def _create_and_prewarm_executor(self, *, max_workers: int) -> Any:
        try:
            executor = self.executor_factory(int(max_workers), bool(self.limit_blas_threads_per_worker))
            self.executor = executor
            self._current_max_workers = int(max_workers)
            self._pool_stale = False
            self._submit_prewarm_tasks(executor, max_workers=int(max_workers))
        except Exception as exc:
            self.shutdown(
                force_terminate=True,
                record_nonfatal_exception=self.record_nonfatal_exception,
            )
            self.executor = None
            self.record_nonfatal_exception("Failed to create and prewarm batch executor", exc)
            raise
        return self.executor

    def _submit_prewarm_tasks(self, executor: Any, *, max_workers: int) -> None:
        for _ in range(max(1, int(max_workers))):
            executor.submit(prewarm_worker_imports)

    def shutdown(self, *, force_terminate: bool, record_nonfatal_exception: Callable[[str, BaseException], None]) -> None:
        prior_futures = int(len(self.future_map or {}))
        self.reset_run_state()
        self._current_max_workers = None
        self._pool_stale = False
        executor = self.executor
        self.executor = None
        if executor is None:
            return
        terminate_target = ()
        if bool(force_terminate):
            processes = getattr(executor, "_processes", None)
            if isinstance(processes, dict):
                terminate_target = tuple(processes.values())
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            with suppress(Exception):
                executor.shutdown(wait=False)
        except Exception as exc:
            record_nonfatal_exception("Failed executor.shutdown during batch executor shutdown", exc)

        if terminate_target:
            self._terminate_processes_best_effort(terminate_target, record_nonfatal_exception)

        _ = prior_futures

    def soft_supersede(self) -> tuple[int, int]:
        cancelled = 0
        running = 0
        active_futures = dict(self.future_map or {})
        active_meta = dict(self.future_meta or {})
        for set_id, fut in active_futures.items():
            try:
                if fut.cancel():
                    cancelled += 1
                else:
                    running += 1
                    retained_key = self._reserve_superseded_key(str(set_id or ""))
                    meta = dict(active_meta.get(str(set_id or "")) or {})
                    meta["superseded"] = "1"
                    meta["set_id"] = str(set_id or "")
                    self.superseded_future_map[retained_key] = fut
                    self.superseded_future_meta[retained_key] = meta
            except Exception:
                running += 1
                retained_key = self._reserve_superseded_key(str(set_id or ""))
                meta = dict(active_meta.get(str(set_id or "")) or {})
                meta["superseded"] = "1"
                meta["set_id"] = str(set_id or "")
                self.superseded_future_map[retained_key] = fut
                self.superseded_future_meta[retained_key] = meta
        self.reset_active_run_state()
        return cancelled, running

    @property
    def is_pool_stale(self) -> bool:
        return bool(self._pool_stale)

    def mark_pool_stale(self) -> None:
        self._pool_stale = True

    def _reserve_superseded_key(self, set_id: str) -> str:
        sid = str(set_id or "").strip() or "superseded"
        if sid not in self.superseded_future_map:
            return sid
        suffix = 2
        while True:
            candidate = f"{sid}#superseded#{suffix}"
            if candidate not in self.superseded_future_map:
                return candidate
            suffix += 1

    @staticmethod
    def _terminate_processes_best_effort(processes: tuple[Any, ...], record_nonfatal_exception: Callable[[str, BaseException], None]) -> None:
        for proc in tuple(processes):
            try:
                if proc is not None and hasattr(proc, "is_alive") and proc.is_alive():
                    proc.terminate()
            except Exception as exc:
                record_nonfatal_exception("Failed to terminate batch executor process", exc)
