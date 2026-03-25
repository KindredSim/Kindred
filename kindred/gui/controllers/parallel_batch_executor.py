from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue
from time import perf_counter
from typing import Any, Dict, Tuple


@dataclass
class ParallelBatchExecutor:
    """
    Owns process-pool lifecycle and per-run future bookkeeping for batch simulations.

    This is kept Qt-free; SimulationController remains responsible for QTimer wiring
    and UI updates.
    """

    executor_factory: Callable[[int, bool], Any]
    max_parallel_workers: int = 12
    limit_blas_threads_per_worker: bool = True
    executor: Any = None
    future_map: Dict[str, Any] = field(default_factory=dict)
    future_meta: Dict[str, Dict[str, str]] = field(default_factory=dict)
    superseded_future_map: Dict[str, Any] = field(default_factory=dict)
    superseded_future_meta: Dict[str, Dict[str, str]] = field(default_factory=dict)
    completed_queue: SimpleQueue[Tuple[str, float]] = field(default_factory=SimpleQueue)

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
        if self.executor is not None:
            return self.executor
        self.executor = self.executor_factory(int(max_workers), bool(self.limit_blas_threads_per_worker))
        return self.executor

    def shutdown(self, *, force_terminate: bool, record_nonfatal_exception: Callable[[str, BaseException], None]) -> None:
        prior_futures = int(len(self.future_map or {}))
        self.reset_run_state()
        executor = self.executor
        self.executor = None
        if executor is None:
            return
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            with suppress(Exception):
                executor.shutdown(wait=False)
        except Exception as exc:
            record_nonfatal_exception("Failed executor.shutdown during batch executor shutdown", exc)

        if bool(force_terminate):
            self._terminate_processes_best_effort(executor, record_nonfatal_exception)

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
    def _terminate_processes_best_effort(executor: Any, record_nonfatal_exception: Callable[[str, BaseException], None]) -> None:
        processes = getattr(executor, "_processes", None)
        if not isinstance(processes, dict):
            return
        for proc in list(processes.values()):
            try:
                if proc is not None and hasattr(proc, "is_alive") and proc.is_alive():
                    proc.terminate()
            except Exception as exc:
                record_nonfatal_exception("Failed to terminate batch executor process", exc)
