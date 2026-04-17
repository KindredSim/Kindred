"""Fitting process-pool wrapper with cancellation, termination, and shutdown reporting.

This module wraps :class:`concurrent.futures.ProcessPoolExecutor` for fitting dataset
evaluation. The wrapper layers fitting-specific worker initialization, cooperative
cancel propagation, best-effort process termination, and shutdown observability on top
of the stdlib executor.

The cancellation model has four layers:

1. ``multiprocessing.Manager().Event()`` is passed into each worker process and polled
   by the worker evaluator. This is cooperative cancellation for work that has already
   started. It does not preempt a worker between poll sites.
2. ``ProcessPoolExecutor.shutdown(cancel_futures=True)`` is used during pool shutdown to
   cancel executor-managed futures that have not started yet. It does not stop a worker
   that is already running.
3. The pool snapshots ``ProcessPoolExecutor._processes`` and calls ``terminate()`` on
   the known children as a best-effort escalation path. This depends on CPython private
   API and is therefore a documented, bounded limitation rather than a stdlib-guaranteed
   contract.
4. The dispatch layer also calls ``Future.cancel()`` on queued dataset futures before
   shutdown. This is queued-only cancellation at the call-site boundary and does not
   replace the cooperative worker event or forced termination path.

The pool is intended to be used as a context manager so shutdown ownership stays with
the call site that creates it. Direct construction plus an explicit ``shutdown()`` call
is also supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import multiprocessing as mp
import os
import pickle
import threading
import traceback
import weakref
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Mapping, MutableMapping, Optional

from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING

logger = logging.getLogger(__name__)

BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_WORKER_EVALUATOR: Any = None
_WORKER_CANCEL_EVENT: Any = None
_ACTIVE_POOL_REGISTRY_LOCK = threading.Lock()
_ACTIVE_POOLS_BY_OWNER_THREAD: dict[int, weakref.WeakSet["FittingProcessPool"]] = {}

__all__ = [
    "BLAS_THREAD_ENV_VARS",
    "FittingProcessPool",
    "ShutdownOutcome",
    "ShutdownStatus",
    "apply_worker_blas_limits",
    "initialize_fitting_worker",
    "run_fitting_evaluation_task",
]

_PREWARM_POLL_SECONDS = 0.05
_PROCESS_SNAPSHOT_ATTEMPTS = 3


class ShutdownStatus(str, Enum):
    GRACEFUL_COMPLETION = "graceful_completion"
    GRACEFUL_WITH_ERRORS = "graceful_with_errors"
    FORCED_TERMINATION_CLEAN = "forced_termination_clean"
    FORCED_TERMINATION_WITH_ERRORS = "forced_termination_with_errors"
    NEVER_STARTED = "never_started"


@dataclass(frozen=True)
class ShutdownOutcome:
    """Observed result of one pool shutdown attempt.

    ``status`` records whether shutdown was graceful, forced, error-bearing, or invoked
    before pool startup reached prewarm.
    ``executor_shutdown_error`` stores the exception raised by
    ``ProcessPoolExecutor.shutdown(...)``, if any.
    ``manager_shutdown_error`` stores the exception raised by ``manager.shutdown()``,
    if any.
    ``termination_errors`` stores best-effort ``proc.terminate()`` failures, labeled by
    worker PID when available or by iteration index when no PID was readable.
    ``cancel_event_error`` stores the exception raised while setting the cooperative
    cancel event, if that boundary failed.
    ``final_worker_process_count`` records how many worker processes were present in the
    last executor-process snapshot taken during this shutdown call.
    ``final_snapshot_nonempty`` records whether that last snapshot returned one or more
    processes. Snapshot failures during forced termination are surfaced through
    ``termination_errors`` with a ``snapshot`` label.
    """

    status: ShutdownStatus
    executor_shutdown_error: BaseException | None = None
    manager_shutdown_error: BaseException | None = None
    termination_errors: tuple[tuple[str, BaseException], ...] = ()
    cancel_event_error: BaseException | None = None
    final_worker_process_count: int = 0
    final_snapshot_nonempty: bool = False

    @property
    def has_errors(self) -> bool:
        return any(
            (
                self.executor_shutdown_error is not None,
                self.manager_shutdown_error is not None,
                self.cancel_event_error is not None,
                bool(self.termination_errors),
            )
        )


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


def _format_marshaled_stack_trace(exc: BaseException) -> Optional[str]:
    if exc.__traceback__ is None:
        raise RuntimeError("FitSimulationError must be marshaled from a caught exception with traceback.")
    stack_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if not stack_trace.strip():
        raise RuntimeError("FitSimulationError traceback formatting produced empty output.")
    return stack_trace


def _error_context_with_marshaled_stack_trace(exc: BaseException) -> Any:
    from kindred.core.exceptions import ErrorContext

    context = getattr(exc, "context", None)
    stack_trace = _format_marshaled_stack_trace(exc)
    if stack_trace is None:
        return context
    if context is None:
        return ErrorContext(stack_trace=stack_trace)
    return ErrorContext(
        line=getattr(context, "line", None),
        col=getattr(context, "col", None),
        line_text=getattr(context, "line_text", None),
        file_path=getattr(context, "file_path", None),
        stack_trace=stack_trace,
    )


def _marshal_exception(exc: BaseException) -> dict[str, Any]:
    from kindred.core.exceptions import FitSimulationError, FittingCancelled, SimulationCancelled

    if isinstance(exc, FitSimulationError):
        return {
            "kind": "fit_simulation",
            "message": str(exc),
            "code": getattr(exc, "code", None),
            "details": dict(getattr(exc, "details", None) or {}),
            "failed_params": dict(getattr(exc, "failed_params", None) or {}),
            "context": _serialize_error_context(_error_context_with_marshaled_stack_trace(exc)),
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
        try:
            raise FitSimulationError(
                f"Simulation failed for dataset '{dataset_id}': {exc}",
                failed_params=dict(getattr(item, "failed_param_snapshot", {}) or {}) or None,
            ) from exc
        except FitSimulationError as wrapped:
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
    """Process-backed fitting evaluator pool with cooperative and forced shutdown paths.

    Lifecycle:
    construction validates and pickles the evaluator payload, builds the manager event
    and executor, then prewarms all workers before returning. Submission runs through the
    executor under the pool lock. ``cancel()`` is idempotent, sets the cooperative cancel
    event, and terminates known workers only when startup is still in prewarm. ``shutdown()``
    is idempotent, returns a ``ShutdownOutcome``, and escalates to best-effort child
    termination when ``force_terminate=True``.

    Context-manager protocol:
    ``__enter__`` returns the fully prewarmed pool. ``__exit__`` always calls
    ``shutdown(force_terminate=exc_type is not None)`` and never suppresses the original
    exception.

    ``cancel()`` is safe to call repeatedly after ``__enter__`` returns. ``worker_pids()``
    returns a lock-synchronized snapshot of currently known worker PIDs taken from the
    executor's internal process registry and returns ``()`` after shutdown clears the
    executor reference.
    """

    def __init__(
        self,
        evaluator_payload: Mapping[str, Any],
        *,
        max_workers: int,
        limit_blas_threads: bool = True,
        cancellation_check: Optional[Callable[[], bool]] = None,
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
        self._owner_thread_ident = threading.get_ident()
        self._startup_cancelled = False
        self._cancellation_check = cancellation_check
        self._prewarm_in_progress = False
        self._prewarm_started = False
        self._entered = False
        self._shutdown_in_progress = False
        self._last_shutdown_outcome: ShutdownOutcome | None = None
        self._state_lock = threading.Lock()
        self._shutdown_condition = threading.Condition(self._state_lock)
        self._register_active_pool()
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
            outcome = self.shutdown(force_terminate=True)
            if outcome.has_errors:
                self._log_shutdown_outcome_errors(outcome, context="pool construction cleanup")
            raise
        try:
            self._prewarm_started = True
            self._prewarm()
        except Exception:
            outcome = self.shutdown(force_terminate=True)
            if outcome.has_errors:
                self._log_shutdown_outcome_errors(outcome, context="pool prewarm cleanup")
            raise

    @property
    def max_workers(self) -> int:
        return int(self._max_workers)

    def __enter__(self) -> "FittingProcessPool":
        with self._state_lock:
            if bool(getattr(self, "_entered", False)):
                raise RuntimeError("Fitting process pool context manager is not re-entrant.")
            if (
                getattr(self, "_manager", None) is None
                or getattr(self, "_cancel_event", None) is None
                or getattr(self, "_executor", None) is None
                or bool(getattr(self, "_prewarm_in_progress", False))
            ):
                raise RuntimeError("Fitting process pool is not fully initialized.")
            self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            try:
                outcome = self.shutdown(force_terminate=exc_type is not None)
            except Exception as exc:
                logger.warning("pool context-manager exit: shutdown raised unexpectedly.", exc_info=exc)
                if exc_type is None:
                    raise
                return None
            if outcome.has_errors:
                self._log_shutdown_outcome_errors(outcome, context="pool context-manager exit")
            return None
        finally:
            with self._state_lock:
                self._entered = False

    def submit(self, item: Any):
        """Submit one dataset-evaluation item to the executor.

        Thread-safe via the pool state lock. Raises ``FittingCancelled`` when shutdown or
        startup cancellation has already begun. Raises ``RuntimeError`` if the executor
        was never initialized.
        """
        return self._submit_executor(run_fitting_evaluation_task, item)

    def cancel(self) -> None:
        """Request cooperative cancellation and prewarm-time worker termination.

        Thread-safe and idempotent. The method sets the cooperative cancel event for
        running workers and, while prewarm is still in progress, best-effort terminates
        workers discovered in the executor snapshot. Failures to set the cancel event or
        terminate workers are logged at DEBUG and do not raise.
        """
        cancel_event, terminate_target, _snapshot_attempted = self._prepare_cancel()
        if cancel_event is not None:
            try:
                cancel_event.set()
            except Exception as exc:
                logger.debug("Failed to set fitting process-pool cancel event during cancel().", exc_info=exc)
        if terminate_target:
            termination_errors = self._terminate_processes_best_effort(terminate_target)
            for label, exc in termination_errors:
                logger.debug(
                    "Failed to terminate fitting worker %s during cancel().",
                    label,
                    exc_info=exc,
                )

    def _snapshot_processes_locked(self, executor: Any) -> tuple[Any, ...]:
        if executor is None:
            return ()
        processes = getattr(executor, "_processes", None)
        if not isinstance(processes, dict):
            return ()
        attempts = max(1, int(_PROCESS_SNAPSHOT_ATTEMPTS))
        for _ in range(attempts):
            try:
                return tuple(processes.values())
            except RuntimeError:
                continue
        return ()

    def worker_pids(self) -> tuple[int, ...]:
        """Return a lock-synchronized snapshot of currently known worker PIDs.

        Thread-safe via the pool state lock. The snapshot is derived from the executor's
        internal process registry. If the executor reference has already been cleared, or
        if the registry snapshot is empty, this method returns ``()``.
        """
        with self._state_lock:
            process_snapshot = self._snapshot_processes_locked(self._executor)
            if not process_snapshot:
                return ()
        pids = []
        for proc in process_snapshot:
            pid = getattr(proc, "pid", None)
            if isinstance(pid, int) and pid > 0:
                pids.append(int(pid))
        return tuple(sorted(set(pids)))

    def shutdown(self, *, force_terminate: bool) -> ShutdownOutcome:
        """Shut down the pool and report what the shutdown path observed.

        Thread-safe and idempotent. ``force_terminate=True`` keeps executor shutdown
        non-blocking and best-effort terminates workers from the final process snapshot.
        ``force_terminate=False`` performs graceful executor shutdown and relies on the
        cooperative cancel event plus queued-future cancellation. The returned
        ``ShutdownOutcome`` records shutdown-boundary exceptions without raising them.
        """
        cancel_event, cancel_terminate_target, cancel_snapshot_attempted = self._prepare_cancel()
        cancel_event_error: BaseException | None = None
        if cancel_event is not None:
            try:
                cancel_event.set()
            except Exception as exc:
                cancel_event_error = exc
        shutdown_condition = self._ensure_shutdown_condition()
        never_started = not bool(getattr(self, "_prewarm_started", False))
        force_escalated = False
        while True:
            need_force_escalation = False
            closed_outcome: ShutdownOutcome | None = None
            with shutdown_condition:
                last_shutdown_outcome = getattr(self, "_last_shutdown_outcome", None)
                if self._closed:
                    closed_outcome = (
                        last_shutdown_outcome
                        if last_shutdown_outcome is not None
                        else ShutdownOutcome(status=ShutdownStatus.NEVER_STARTED)
                    )
                elif not self._shutdown_in_progress:
                    self._shutdown_in_progress = True
                    break
                elif bool(force_terminate) and not force_escalated:
                    need_force_escalation = True
                    force_escalated = True
                else:
                    shutdown_condition.wait()
            if closed_outcome is not None:
                return closed_outcome
            if need_force_escalation:
                escalated_outcome = self._force_shutdown_out_of_band()
                if escalated_outcome is not None:
                    return escalated_outcome

        with shutdown_condition:
            if self._closed:
                last_shutdown_outcome = getattr(self, "_last_shutdown_outcome", None)
                return (
                    last_shutdown_outcome
                    if last_shutdown_outcome is not None
                    else ShutdownOutcome(status=ShutdownStatus.NEVER_STARTED)
                )
            terminate_target = tuple(cancel_terminate_target) if cancel_snapshot_attempted else ()
            return self._complete_shutdown_locked(
                shutdown_condition=shutdown_condition,
                force_terminate=bool(force_terminate),
                never_started=never_started,
                cancel_event_error=cancel_event_error,
                terminate_target=terminate_target,
            )

    def _prewarm(self) -> None:
        """Start all workers and wait until each initializer has run.

        This method is not thread-safe and is intended to run only during construction.
        It polls ``_raise_if_startup_cancelled()`` between timed waits so construction-time
        cancellation can abort startup quickly. Raises ``FittingCancelled`` when startup
        cancellation is requested, or propagates executor and initializer failures.
        """
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
        cancelled = bool(self._startup_cancelled)
        if not cancelled:
            cancellation_check = getattr(self, "_cancellation_check", None)
            if cancellation_check is not None:
                # Prefer the callable's _kindred_nonblocking_cancelled helper so
                # startup treats pause as "not cancelled, proceed" until readiness.
                cancel_requested = getattr(
                    cancellation_check,
                    "_kindred_nonblocking_cancelled",
                    cancellation_check,
                )
                if bool(cancel_requested()):
                    self._startup_cancelled = True
                    cancelled = True
        if not cancelled:
            return
        from kindred.core.exceptions import FittingCancelled

        raise FittingCancelled()

    def _submit_executor(self, fn: Callable[..., Any], *args: Any):
        """Submit raw executor work under the pool lock.

        Thread-safe via the pool state lock. Raises ``FittingCancelled`` if shutdown is
        already in progress or startup cancellation has been requested. Raises
        ``RuntimeError`` if the executor reference is missing. Propagates executor submit
        errors unless shutdown state changed concurrently, in which case they are coerced
        to ``FittingCancelled``.
        """
        from kindred.core.exceptions import FittingCancelled

        with self._state_lock:
            if self._closed or self._shutdown_in_progress or self._startup_cancelled:
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

    def _prepare_cancel(self) -> tuple[Any, tuple[Any, ...], bool]:
        self._startup_cancelled = True
        terminate_target = ()
        snapshot_attempted = False
        with self._state_lock:
            cancel_event = self._cancel_event
            if self._prewarm_in_progress and not self._closed:
                snapshot_attempted = True
                terminate_target = self._snapshot_processes_locked(self._executor)
        return cancel_event, terminate_target, snapshot_attempted

    @classmethod
    def force_shutdown_registered_pools_for_owner_thread(
        cls,
        owner_thread_ident: int | None,
    ) -> tuple[ShutdownOutcome, ...]:
        if not isinstance(owner_thread_ident, int):
            return ()
        with _ACTIVE_POOL_REGISTRY_LOCK:
            pools = tuple(_ACTIVE_POOLS_BY_OWNER_THREAD.get(owner_thread_ident, ()))
        outcomes: list[ShutdownOutcome] = []
        for pool in pools:
            outcome = pool._force_shutdown_out_of_band()
            if outcome is not None:
                outcomes.append(outcome)
        return tuple(outcomes)

    def _force_shutdown_out_of_band(self) -> ShutdownOutcome | None:
        shutdown_condition = self._ensure_shutdown_condition()
        cancel_event, terminate_target, _snapshot_attempted = self._prepare_cancel()
        cancel_event_error: BaseException | None = None
        if cancel_event is not None:
            try:
                cancel_event.set()
            except Exception as exc:
                cancel_event_error = exc
        with shutdown_condition:
            if self._closed:
                return self._last_shutdown_outcome
            if not self._shutdown_in_progress:
                self._shutdown_in_progress = True
            executor = self._executor
            final_snapshot, snapshot_error = self._snapshot_processes_observed_locked(executor)
            termination_errors: tuple[tuple[str, BaseException], ...] = ()
            if final_snapshot:
                termination_errors = self._terminate_processes_best_effort(final_snapshot)
            if self._closed:
                return self._last_shutdown_outcome
            return self._complete_shutdown_locked(
                shutdown_condition=shutdown_condition,
                force_terminate=True,
                never_started=not bool(getattr(self, "_prewarm_started", False)),
                cancel_event_error=cancel_event_error,
                terminate_target=tuple(final_snapshot),
                termination_errors=termination_errors,
                snapshot_error=snapshot_error,
                terminate_target_already_handled=True,
            )

    def _ensure_shutdown_condition(self) -> threading.Condition:
        condition = getattr(self, "_shutdown_condition", None)
        if condition is not None:
            return condition
        state_lock = getattr(self, "_state_lock", None)
        if state_lock is None:
            state_lock = threading.Lock()
            self._state_lock = state_lock
        condition = threading.Condition(state_lock)
        self._shutdown_condition = condition
        return condition

    @staticmethod
    def _resolve_shutdown_status(*, force_terminate: bool, never_started: bool, has_errors: bool) -> ShutdownStatus:
        if force_terminate:
            if has_errors:
                return ShutdownStatus.FORCED_TERMINATION_WITH_ERRORS
            if never_started:
                return ShutdownStatus.NEVER_STARTED
            return ShutdownStatus.FORCED_TERMINATION_CLEAN
        if has_errors:
            return ShutdownStatus.GRACEFUL_WITH_ERRORS
        if never_started:
            return ShutdownStatus.NEVER_STARTED
        return ShutdownStatus.GRACEFUL_COMPLETION

    def _complete_shutdown_locked(
        self,
        *,
        shutdown_condition: threading.Condition,
        force_terminate: bool,
        never_started: bool,
        cancel_event_error: BaseException | None,
        terminate_target: tuple[Any, ...] = (),
        termination_errors: tuple[tuple[str, BaseException], ...] = (),
        snapshot_error: BaseException | None = None,
        terminate_target_already_handled: bool = False,
    ) -> ShutdownOutcome:
        executor = self._executor
        manager = self._manager
        executor_shutdown_error: BaseException | None = None
        manager_shutdown_error: BaseException | None = None
        resolved_termination_errors = tuple(termination_errors)
        if force_terminate and not terminate_target:
            terminate_target, snapshot_error = self._snapshot_processes_observed_locked(executor)
        if snapshot_error is not None:
            resolved_termination_errors = tuple(resolved_termination_errors) + (("snapshot", snapshot_error),)

        final_snapshot = tuple(terminate_target)
        final_snapshot_error: BaseException | None = snapshot_error
        try:
            if executor is not None:
                try:
                    executor.shutdown(wait=not bool(force_terminate), cancel_futures=True)
                except Exception as exc:
                    executor_shutdown_error = exc
        finally:
            if terminate_target and not terminate_target_already_handled:
                resolved_termination_errors = tuple(resolved_termination_errors) + self._terminate_processes_best_effort(
                    terminate_target
                )
            if bool(force_terminate):
                final_snapshot, final_snapshot_error = self._snapshot_processes_observed_locked(executor)
            else:
                final_snapshot = self._snapshot_processes_locked(executor)
                final_snapshot_error = None
            if final_snapshot_error is not None:
                resolved_termination_errors = tuple(resolved_termination_errors) + (("snapshot", final_snapshot_error),)
            if manager is not None:
                try:
                    manager.shutdown()
                except Exception as exc:
                    manager_shutdown_error = exc

        outcome = ShutdownOutcome(
            status=self._resolve_shutdown_status(
                force_terminate=bool(force_terminate),
                never_started=bool(never_started),
                has_errors=bool(
                    executor_shutdown_error is not None
                    or manager_shutdown_error is not None
                    or cancel_event_error is not None
                    or resolved_termination_errors
                ),
            ),
            executor_shutdown_error=executor_shutdown_error,
            manager_shutdown_error=manager_shutdown_error,
            termination_errors=resolved_termination_errors,
            cancel_event_error=cancel_event_error,
            final_worker_process_count=len(tuple(final_snapshot)),
            final_snapshot_nonempty=bool(final_snapshot),
        )
        self._executor = None
        self._manager = None
        self._closed = True
        self._shutdown_in_progress = False
        self._last_shutdown_outcome = outcome
        shutdown_condition.notify_all()
        self._unregister_active_pool()
        return outcome

    def _snapshot_processes_observed_locked(self, executor: Any) -> tuple[tuple[Any, ...], BaseException | None]:
        snapshot = self._snapshot_processes_locked(executor)
        if snapshot:
            return tuple(snapshot), None
        if executor is None:
            return (), None
        processes = getattr(executor, "_processes", None)
        if not isinstance(processes, dict):
            return (), RuntimeError("Fitting worker process registry is unavailable on the executor.")
        try:
            if len(processes) <= 0:
                return (), None
        except Exception:
            return (), RuntimeError("Failed to inspect fitting worker process registry size.")
        return (), RuntimeError("Failed to snapshot fitting worker processes from the executor registry.")

    def _register_active_pool(self) -> None:
        owner_thread_ident = getattr(self, "_owner_thread_ident", None)
        if not isinstance(owner_thread_ident, int):
            return
        with _ACTIVE_POOL_REGISTRY_LOCK:
            pools = _ACTIVE_POOLS_BY_OWNER_THREAD.get(owner_thread_ident)
            if pools is None:
                pools = weakref.WeakSet()
                _ACTIVE_POOLS_BY_OWNER_THREAD[owner_thread_ident] = pools
            pools.add(self)

    def _unregister_active_pool(self) -> None:
        owner_thread_ident = getattr(self, "_owner_thread_ident", None)
        if not isinstance(owner_thread_ident, int):
            return
        with _ACTIVE_POOL_REGISTRY_LOCK:
            pools = _ACTIVE_POOLS_BY_OWNER_THREAD.get(owner_thread_ident)
            if pools is None:
                return
            pools.discard(self)
            if not pools:
                _ACTIVE_POOLS_BY_OWNER_THREAD.pop(owner_thread_ident, None)

    @staticmethod
    def _log_shutdown_outcome_errors(outcome: ShutdownOutcome, *, context: str) -> None:
        if outcome.cancel_event_error is not None:
            logger.warning(
                "%s: failed to set fitting process-pool cancel event.",
                context,
                exc_info=outcome.cancel_event_error,
            )
        if outcome.executor_shutdown_error is not None:
            logger.warning(
                "%s: executor shutdown reported an error.",
                context,
                exc_info=outcome.executor_shutdown_error,
            )
        if outcome.manager_shutdown_error is not None:
            logger.warning(
                "%s: manager shutdown reported an error.",
                context,
                exc_info=outcome.manager_shutdown_error,
            )
        for label, exc in outcome.termination_errors:
            logger.warning("%s: failed to terminate fitting worker %s.", context, label, exc_info=exc)

    def _terminate_processes_best_effort(self, processes: tuple[Any, ...]) -> tuple[tuple[str, BaseException], ...]:
        errors: list[tuple[str, BaseException]] = []
        for index, proc in enumerate(tuple(processes)):
            try:
                if proc is not None and hasattr(proc, "is_alive") and proc.is_alive():
                    proc.terminate()
            except Exception as exc:
                pid = getattr(proc, "pid", None)
                if isinstance(pid, int) and pid > 0:
                    label = f"pid={pid}"
                else:
                    label = f"index={index}"
                errors.append((label, exc))
        return tuple(errors)
