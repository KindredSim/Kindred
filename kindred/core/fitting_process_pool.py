from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import threading
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import suppress
from typing import Any, Callable, Mapping, MutableMapping, Optional

from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING

BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_WORKER_EVALUATOR: Any = None
_WORKER_CANCEL_EVENT: Any = None

__all__ = [
    "BLAS_THREAD_ENV_VARS",
    "FittingProcessPool",
    "apply_worker_blas_limits",
    "initialize_fitting_worker",
    "run_fitting_evaluation_task",
]

_PREWARM_POLL_SECONDS = 0.05


def apply_worker_blas_limits(*, enabled: bool, environ: MutableMapping[str, str] | None = None) -> None:
    if not bool(enabled):
        return
    env = os.environ if environ is None else environ
    for name in BLAS_THREAD_ENV_VARS:
        env[str(name)] = "1"


def _worker_cancel_requested() -> bool:
    if _WORKER_CANCEL_EVENT is None:
        return False
    try:
        return bool(_WORKER_CANCEL_EVENT.is_set())
    except Exception:
        return False


def initialize_fitting_worker(
    evaluator_payload: Mapping[str, Any],
    cancel_event: Any,
    limit_blas_threads: bool = True,
) -> None:
    from kindred.core.exceptions import FittingCancelled
    global _WORKER_CANCEL_EVENT
    global _WORKER_EVALUATOR

    apply_worker_blas_limits(enabled=bool(limit_blas_threads))

    from kindred.core.fitting_evaluation import SerialFittingEvaluator

    _WORKER_CANCEL_EVENT = cancel_event
    evaluator = SerialFittingEvaluator.from_process_payload(dict(evaluator_payload or {}))
    evaluator._kindred_set_fitting_cancellation_check(_worker_cancel_requested)
    if _worker_cancel_requested():
        _WORKER_EVALUATOR = evaluator
        return
    try:
        evaluator._ensure_prepared()
    except FittingCancelled:
        _WORKER_EVALUATOR = evaluator
        return
    _WORKER_EVALUATOR = evaluator


def _serialize_error_context(context: Any) -> Optional[dict[str, Any]]:
    if context is None:
        return None
    return {
        "line": getattr(context, "line", None),
        "col": getattr(context, "col", None),
        "line_text": getattr(context, "line_text", None),
        "file_path": getattr(context, "file_path", None),
        "stack_trace": getattr(context, "stack_trace", None),
    }


def _marshal_exception(exc: BaseException) -> dict[str, Any]:
    from kindred.core.exceptions import FitSimulationError, FittingCancelled, SimulationCancelled

    if isinstance(exc, FitSimulationError):
        return {
            "kind": "fit_simulation",
            "message": str(exc),
            "code": getattr(exc, "code", None),
            "details": dict(getattr(exc, "details", None) or {}),
            "failed_params": dict(getattr(exc, "failed_params", None) or {}),
            "context": _serialize_error_context(getattr(exc, "context", None)),
        }
    if isinstance(exc, FittingCancelled):
        return {
            "kind": "fitting_cancelled",
            "message": str(exc) or "Fit cancelled by user",
            "code": getattr(exc, "code", None),
            "details": dict(getattr(exc, "details", None) or {}),
            "context": _serialize_error_context(getattr(exc, "context", None)),
        }
    if isinstance(exc, SimulationCancelled):
        return {
            "kind": "simulation_cancelled",
            "message": str(exc) or "Simulation cancelled by user",
            "code": getattr(exc, "code", None),
            "details": dict(getattr(exc, "details", None) or {}),
            "context": _serialize_error_context(getattr(exc, "context", None)),
        }
    return {
        "kind": "generic",
        "message": str(exc) or exc.__class__.__name__,
        "exc_type": exc.__class__.__name__,
    }


def _worker_identity() -> int:
    return int(os.getpid())


def run_fitting_evaluation_task(item: Any) -> dict[str, Any]:
    from kindred.core.exceptions import FitSimulationError, FittingCancelled, SimulationCancelled
    from kindred.core.fitting_evaluation import evaluate_fitting_series

    if _WORKER_EVALUATOR is None:
        raise RuntimeError("Fitting process worker is not initialized.")

    dataset_id = str(getattr(getattr(item, "payload", None), "dataset_id", ""))
    try:
        if _worker_cancel_requested():
            raise FittingCancelled()
        result = evaluate_fitting_series(
            _WORKER_EVALUATOR,
            getattr(item, "full_params", {}),
            origins=getattr(item, "parameter_origins", {}),
            failed_params=getattr(item, "failed_param_snapshot", {}),
        )
        return {
            "index": int(getattr(item, "index")),
            "dataset_id": dataset_id,
            "worker_pid": int(os.getpid()),
            "ok": True,
            "series_payload": result,
            "error": None,
            "error_provenance": None,
            "final_error_message": None,
        }
    except FitSimulationError as exc:
        return {
            "index": int(getattr(item, "index")),
            "dataset_id": dataset_id,
            "worker_pid": int(os.getpid()),
            "ok": False,
            "series_payload": None,
            "error": _marshal_exception(exc),
            "error_provenance": {"dataset": dataset_id, "provenance": getattr(exc, "provenance", None)},
            "final_error_message": str(exc),
        }
    except (FittingCancelled, SimulationCancelled) as exc:
        return {
            "index": int(getattr(item, "index")),
            "dataset_id": dataset_id,
            "worker_pid": int(os.getpid()),
            "ok": False,
            "series_payload": None,
            "error": _marshal_exception(exc),
            "error_provenance": {"dataset": dataset_id},
            "final_error_message": str(exc),
        }
    except Exception as exc:
        wrapped = FitSimulationError(
            f"Simulation failed for dataset '{dataset_id}': {exc}",
            failed_params=dict(getattr(item, "failed_param_snapshot", {}) or {}) or None,
        )
        return {
            "index": int(getattr(item, "index")),
            "dataset_id": dataset_id,
            "worker_pid": int(os.getpid()),
            "ok": False,
            "series_payload": None,
            "error": _marshal_exception(wrapped),
            "error_provenance": {"dataset": dataset_id},
            "final_error_message": str(wrapped),
        }


class FittingProcessPool:
    def __init__(
        self,
        evaluator_payload: Mapping[str, Any],
        *,
        max_workers: int,
        limit_blas_threads: bool = True,
        publish_callback: Optional[Callable[[Optional["FittingProcessPool"]], None]] = None,
    ) -> None:
        requested_workers = min(
            int(MAX_PARALLEL_WORKERS_CEILING),
            max(1, int(max_workers)),
        )
        ctx = mp.get_context("spawn")
        payload = dict(evaluator_payload or {})
        payload_copy = dict(payload)
        pickle.dumps(payload_copy)
        self._manager = None
        self._cancel_event = None
        self._executor = None
        self._closed = False
        self._max_workers = requested_workers
        self._startup_cancelled = False
        self._prewarm_in_progress = False
        self._shutdown_in_progress = False
        self._state_lock = threading.Lock()
        published = False
        if publish_callback is not None:
            publish_callback(self)
            published = True
        try:
            self._raise_if_startup_cancelled()
            self._manager = ctx.Manager()
            self._raise_if_startup_cancelled()
            self._cancel_event = self._manager.Event()
            self._raise_if_startup_cancelled()
            self._executor = ProcessPoolExecutor(
                max_workers=requested_workers,
                mp_context=ctx,
                initializer=initialize_fitting_worker,
                initargs=(payload_copy, self._cancel_event, bool(limit_blas_threads)),
            )
            self._raise_if_startup_cancelled()
        except Exception:
            with suppress(Exception):
                self.shutdown(force_terminate=True)
            if published and publish_callback is not None:
                with suppress(Exception):
                    publish_callback(None)
            raise
        try:
            self._prewarm()
        except Exception:
            self.shutdown(force_terminate=True)
            if published and publish_callback is not None:
                with suppress(Exception):
                    publish_callback(None)
            raise

    @property
    def max_workers(self) -> int:
        return int(self._max_workers)

    def submit(self, item: Any):
        return self._submit_executor(run_fitting_evaluation_task, item)

    def cancel(self) -> None:
        self._startup_cancelled = True
        terminate_target = ()
        with self._state_lock:
            cancel_event = self._cancel_event
            if self._prewarm_in_progress and not self._closed:
                processes = getattr(self._executor, "_processes", None)
                if isinstance(processes, dict):
                    terminate_target = tuple(processes.values())
        with suppress(Exception):
            if cancel_event is not None:
                cancel_event.set()
        if terminate_target:
            self._terminate_processes_best_effort(terminate_target)

    def worker_pids(self) -> tuple[int, ...]:
        with self._state_lock:
            executor = self._executor
            if executor is None:
                return ()
            processes = getattr(executor, "_processes", None)
            if not isinstance(processes, dict):
                return ()
            try:
                process_snapshot = tuple(processes.values())
            except RuntimeError:
                return ()
        pids = []
        for proc in process_snapshot:
            pid = getattr(proc, "pid", None)
            if isinstance(pid, int) and pid > 0:
                pids.append(int(pid))
        return tuple(sorted(set(pids)))

    def shutdown(self, *, force_terminate: bool) -> None:
        self.cancel()
        execute_shutdown = False
        executor = None
        manager = None
        terminate_target = ()
        with self._state_lock:
            if self._closed and self._executor is None and self._manager is None:
                return
            executor = self._executor
            manager = self._manager
            if self._shutdown_in_progress:
                if bool(force_terminate) and executor is not None:
                    processes = getattr(executor, "_processes", None)
                    if isinstance(processes, dict):
                        terminate_target = tuple(processes.values())
            else:
                self._shutdown_in_progress = True
                execute_shutdown = True
                if bool(force_terminate) and executor is not None:
                    processes = getattr(executor, "_processes", None)
                    if isinstance(processes, dict):
                        terminate_target = tuple(processes.values())
        if not execute_shutdown:
            if terminate_target:
                self._terminate_processes_best_effort(terminate_target)
            return
        try:
            if executor is not None:
                try:
                    executor.shutdown(wait=not bool(force_terminate), cancel_futures=True)
                except TypeError:
                    with suppress(Exception):
                        executor.shutdown(wait=not bool(force_terminate))
        finally:
            if terminate_target:
                self._terminate_processes_best_effort(terminate_target)
            with suppress(Exception):
                if manager is not None:
                    manager.shutdown()
            with self._state_lock:
                if self._executor is executor:
                    self._executor = None
                if self._manager is manager:
                    self._manager = None
                self._closed = True
                self._shutdown_in_progress = False

    def _prewarm(self) -> None:
        if self._executor is None:
            raise RuntimeError("Fitting process pool is not initialized.")
        with self._state_lock:
            self._prewarm_in_progress = True
        try:
            futures = [self._submit_executor(_worker_identity) for _ in range(self._max_workers)]
            for future in futures:
                while True:
                    self._raise_if_startup_cancelled()
                    try:
                        future.result(timeout=_PREWARM_POLL_SECONDS)
                        break
                    except FutureTimeoutError:
                        self._raise_if_startup_cancelled()
        finally:
            with self._state_lock:
                self._prewarm_in_progress = False

    def _raise_if_startup_cancelled(self) -> None:
        if not self._startup_cancelled:
            return
        from kindred.core.exceptions import FittingCancelled

        raise FittingCancelled()

    def _submit_executor(self, fn: Callable[..., Any], *args: Any):
        from kindred.core.exceptions import FittingCancelled

        with self._state_lock:
            if self._closed or self._shutdown_in_progress:
                raise FittingCancelled()
            executor = self._executor
            if executor is None:
                raise RuntimeError("Fitting process pool is not initialized.")
            try:
                return executor.submit(fn, *args)
            except RuntimeError as exc:
                if self._closed or self._shutdown_in_progress or self._startup_cancelled:
                    raise FittingCancelled() from exc
                raise

    def _terminate_processes_best_effort(self, processes: tuple[Any, ...]) -> None:
        for proc in tuple(processes):
            with suppress(Exception):
                if proc is not None and hasattr(proc, "is_alive") and proc.is_alive():
                    proc.terminate()
