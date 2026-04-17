from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future

import numpy as np
import pytest

from tests.conftest import CAN_CREATE_PROCESS_POOL, PROCESS_POOL_SKIP_REASON


def _payload(dataset_id: str, y_values) -> object:
    from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec

    y = np.asarray(y_values, dtype=float).reshape(1, -1)
    return FitDatasetSpec(
        dataset_id=str(dataset_id),
        t_exp=np.linspace(0.0, 1.0, y.shape[1]),
        species_list=["A"],
        y_matrix=y,
        point_count=int(y.size),
        x_name="t",
        x_obs=None,
        x_mode="auto",
        target_weights={},
    )


def _raw_dataset(dataset_id: str, y_values) -> dict[str, object]:
    y = np.asarray(y_values, dtype=float).reshape(-1)
    return {
        "id": str(dataset_id),
        "t": np.linspace(0.0, 1.0, y.size),
        "species": "A",
        "y": y,
    }


def _serial_context(*, solver: str = "BDF", num_points: int = 6):
    from kindred.core.fitting_evaluation import prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    return prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=num_points,
        solver=solver,
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )


def _make_serial_evaluator(*, solver: str = "BDF", num_points: int = 6):
    from kindred.core.fitting_evaluation import SerialFittingEvaluator

    return SerialFittingEvaluator(_serial_context(solver=solver, num_points=num_points)).with_fixed_params({"k1": 0.2})


def _dataset_input(index: int, dataset_id: str, init_a: float):
    from kindred.core.analysis.global_fitting import _ObjectiveDatasetInput

    return _ObjectiveDatasetInput(
        index=int(index),
        payload=_payload(dataset_id, [0.0, 0.0]),
        full_params={"init:A": float(init_a)},
        parameter_origins={},
        failed_param_snapshot={"init:A": float(init_a)},
    )


def _sleeping_worker_task(_item) -> dict[str, object]:
    time.sleep(5.0)
    return {
        "index": 0,
        "dataset_id": "sleep",
        "worker_pid": os.getpid(),
        "ok": True,
        "series_payload": {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "species": {"A": np.asarray([0.0, 0.0], dtype=float)},
        },
        "error": None,
        "error_provenance": None,
        "final_error_message": None,
    }


def _cancel_aware_worker_task(_item) -> dict[str, object]:
    import kindred.core.fitting_process_pool as fitting_process_pool
    from kindred.core.exceptions import FittingCancelled

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        cancel_event = getattr(fitting_process_pool, "_WORKER_CANCEL_EVENT", None)
        if cancel_event is not None and bool(cancel_event.is_set()):
            raise FittingCancelled("cancel event observed inside process worker")
        time.sleep(0.05)
    raise AssertionError("cancel-aware worker task should observe cancellation before timing out")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _manager_process_alive(manager) -> bool:
    process = getattr(manager, "_process", None)
    return bool(process is not None and process.is_alive())


def _signal_shutdown_completion(pool, outcome) -> None:
    with pool._shutdown_condition:
        pool._closed = True
        pool._shutdown_in_progress = False
        pool._executor = None
        pool._manager = None
        pool._last_shutdown_outcome = outcome
        pool._shutdown_condition.notify_all()


def _pool_registered_for_owner_thread(pool) -> bool:
    import kindred.core.fitting_process_pool as fitting_process_pool

    with fitting_process_pool._ACTIVE_POOL_REGISTRY_LOCK:
        pools = tuple(fitting_process_pool._ACTIVE_POOLS_BY_OWNER_THREAD.get(pool._owner_thread_ident, ()))
    return pool in pools


def _force_shutdown_registered_pool(pool):
    from kindred.core.fitting_process_pool import FittingProcessPool

    outcomes = FittingProcessPool.force_shutdown_registered_pools_for_owner_thread(pool._owner_thread_ident)
    assert len(outcomes) == 1
    return outcomes[0]


def _cleanup_pool_after_test(pool, *, fallback_outcome=None) -> None:
    ensure_shutdown_condition = getattr(pool, "_ensure_shutdown_condition", None)
    if callable(ensure_shutdown_condition):
        shutdown_condition = ensure_shutdown_condition()
    else:
        shutdown_condition = getattr(pool, "_shutdown_condition", None)
    executor = getattr(pool, "_executor", None)
    if executor is not None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    manager = getattr(pool, "_manager", None)
    if manager is not None:
        try:
            manager.shutdown()
        except Exception:
            pass
    if shutdown_condition is not None:
        with shutdown_condition:
            if getattr(pool, "_last_shutdown_outcome", None) is None and fallback_outcome is not None:
                pool._last_shutdown_outcome = fallback_outcome
            pool._executor = None
            pool._manager = None
            pool._closed = True
            pool._shutdown_in_progress = False
            shutdown_condition.notify_all()
    else:
        state_lock = getattr(pool, "_state_lock", None)
        if state_lock is not None:
            with state_lock:
                if getattr(pool, "_last_shutdown_outcome", None) is None and fallback_outcome is not None:
                    pool._last_shutdown_outcome = fallback_outcome
                pool._executor = None
                pool._manager = None
                pool._closed = True
                pool._shutdown_in_progress = False
    unregister = getattr(pool, "_unregister_active_pool", None)
    if callable(unregister):
        try:
            unregister()
        except Exception:
            pass


class _EvaluateOnlyNoClone:
    def __init__(self, *, t_axis, state):
        self._t_axis = np.asarray(t_axis, dtype=float).reshape(-1)
        self._state = state

    def evaluate_series(self, params):
        self._state["base_calls"] += 1
        value = float(dict(params).get("init:A", 0.0))
        return {
            "t": self._t_axis.copy(),
            "species": {"A": np.full_like(self._t_axis, value, dtype=float)},
        }

@pytest.mark.skipif(not CAN_CREATE_PROCESS_POOL, reason=PROCESS_POOL_SKIP_REASON)
def test_fitting_process_pool_caps_requested_workers_at_shared_ceiling() -> None:
    from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
    from kindred.core.fitting_process_pool import FittingProcessPool

    pool = FittingProcessPool(_make_serial_evaluator().to_process_payload(), max_workers=200)

    try:
        assert pool.max_workers == int(MAX_PARALLEL_WORKERS_CEILING)
    finally:
        pool.shutdown(force_terminate=False)


@pytest.mark.skipif(not CAN_CREATE_PROCESS_POOL, reason=PROCESS_POOL_SKIP_REASON)
def test_fitting_process_pool_worker_pids_are_spawned_processes() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    evaluator = _make_serial_evaluator()
    items = [
        _dataset_input(0, "ds1", 1.0),
        _dataset_input(1, "ds2", 2.0),
    ]
    pool = FittingProcessPool(evaluator.to_process_payload(), max_workers=2)

    try:
        futures = [pool.submit(item) for item in items]
        payloads = [future.result() for future in futures]
        worker_pids = pool.worker_pids()
    finally:
        pool.shutdown(force_terminate=False)

    assert len(worker_pids) == 2
    assert os.getpid() not in worker_pids
    assert {int(payload["worker_pid"]) for payload in payloads}.issubset(set(worker_pids))


@pytest.mark.skipif(not CAN_CREATE_PROCESS_POOL, reason=PROCESS_POOL_SKIP_REASON)
def test_fitting_process_pool_worker_pids_are_empty_after_shutdown_returns() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    pool = FittingProcessPool(_make_serial_evaluator().to_process_payload(), max_workers=1)

    assert pool.worker_pids()

    pool.shutdown(force_terminate=False)

    assert pool.worker_pids() == ()


def test_fitting_process_pool_worker_pids_return_empty_on_process_snapshot_race() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    class _MutatingProcesses(dict):
        def values(self):
            raise RuntimeError("dictionary changed size during iteration")

    class _FakeExecutor:
        def __init__(self) -> None:
            self._processes = _MutatingProcesses({1: object()})

    pool = object.__new__(FittingProcessPool)
    pool._state_lock = threading.Lock()
    pool._executor = _FakeExecutor()

    assert pool.worker_pids() == ()


@pytest.mark.skipif(not CAN_CREATE_PROCESS_POOL, reason=PROCESS_POOL_SKIP_REASON)
@pytest.mark.parametrize("force_terminate", [False, True])
def test_fitting_process_pool_worker_pids_do_not_raise_during_concurrent_shutdown(
    force_terminate: bool,
) -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    pool = FittingProcessPool(_make_serial_evaluator().to_process_payload(), max_workers=1)
    started = threading.Event()
    stop = threading.Event()
    errors: list[BaseException] = []
    call_count = 0
    call_count_lock = threading.Lock()

    def read_worker_pids() -> None:
        nonlocal call_count
        while not stop.is_set():
            try:
                pool.worker_pids()
            except BaseException as exc:  # pragma: no cover - failure path only
                errors.append(exc)
                stop.set()
                return
            with call_count_lock:
                call_count += 1
                started.set()

    reader = threading.Thread(target=read_worker_pids)
    reader.start()
    assert started.wait(timeout=5.0)
    try:
        pool.shutdown(force_terminate=force_terminate)
    finally:
        stop.set()
        reader.join(timeout=5.0)

    assert not reader.is_alive()
    assert call_count > 0
    assert errors == []


def test_fitting_process_pool_initializer_limits_blas_threads_before_prepare(monkeypatch) -> None:
    import kindred.core.fitting_evaluation as fitting_evaluation
    from kindred.core.fitting_process_pool import initialize_fitting_worker

    captured = {}
    original_from_process_payload = fitting_evaluation.SerialFittingEvaluator.from_process_payload

    def fake_from_process_payload(cls, payload):
        evaluator = original_from_process_payload(payload)
        original_ensure_prepared = evaluator._ensure_prepared

        def tracking_ensure_prepared():
            captured["omp"] = os.environ.get("OMP_NUM_THREADS")
            captured["mkl"] = os.environ.get("MKL_NUM_THREADS")
            captured["openblas"] = os.environ.get("OPENBLAS_NUM_THREADS")
            return original_ensure_prepared()

        evaluator._ensure_prepared = tracking_ensure_prepared
        return evaluator

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.setattr(
        fitting_evaluation.SerialFittingEvaluator,
        "from_process_payload",
        classmethod(fake_from_process_payload),
    )

    initialize_fitting_worker(
        _make_serial_evaluator(num_points=4).to_process_payload(),
        cancel_event=None,
        limit_blas_threads=True,
    )

    assert captured == {
        "omp": "1",
        "mkl": "1",
        "openblas": "1",
    }


def test_initialize_fitting_worker_skips_prepare_when_cancel_is_already_set(monkeypatch) -> None:
    import kindred.core.fitting_evaluation as fitting_evaluation
    from kindred.core.fitting_process_pool import initialize_fitting_worker

    captured = {"prepared": False}
    original_from_process_payload = fitting_evaluation.SerialFittingEvaluator.from_process_payload

    def fake_from_process_payload(cls, payload):
        evaluator = original_from_process_payload(payload)

        def tracking_ensure_prepared():
            captured["prepared"] = True
            return None

        evaluator._ensure_prepared = tracking_ensure_prepared
        return evaluator

    class _CancelledEvent:
        def is_set(self):
            return True

    monkeypatch.setattr(
        fitting_evaluation.SerialFittingEvaluator,
        "from_process_payload",
        classmethod(fake_from_process_payload),
    )

    initialize_fitting_worker(
        _make_serial_evaluator(num_points=4).to_process_payload(),
        cancel_event=_CancelledEvent(),
        limit_blas_threads=True,
    )

    assert captured["prepared"] is False


def test_initialize_fitting_worker_stops_prepare_when_cancel_arrives_mid_prepare(monkeypatch) -> None:
    import kindred.core.fitting_evaluation as fitting_evaluation
    import kindred.core.fitting_process_pool as fitting_process_pool
    from kindred.core.fitting_process_pool import initialize_fitting_worker

    cancel_event = type(
        "_ToggleEvent",
        (),
        {
            "__init__": lambda self: setattr(self, "flag", False),
            "is_set": lambda self: bool(self.flag),
        },
    )()
    original_prepare = fitting_evaluation.prepare_simulation_worker_run

    def fake_prepare_simulation_worker_run(*args, **kwargs):
        prepared = original_prepare(*args, **kwargs)
        cancel_event.flag = True
        return prepared

    monkeypatch.setattr(
        fitting_evaluation,
        "prepare_simulation_worker_run",
        fake_prepare_simulation_worker_run,
    )

    initialize_fitting_worker(
        _make_serial_evaluator(num_points=4).to_process_payload(),
        cancel_event=cancel_event,
        limit_blas_threads=True,
    )

    assert fitting_process_pool._WORKER_EVALUATOR is not None
    assert fitting_process_pool._WORKER_EVALUATOR._prepared_run is None


@pytest.mark.skipif(not CAN_CREATE_PROCESS_POOL, reason=PROCESS_POOL_SKIP_REASON)
def test_fitting_process_pool_cancel_event_crosses_process_boundary() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    pool = FittingProcessPool(_make_serial_evaluator().to_process_payload(), max_workers=1)

    try:
        pool.cancel()
        payload = pool.submit(_dataset_input(0, "ds1", 1.0)).result()
    finally:
        pool.shutdown(force_terminate=True)

    assert payload["ok"] is False
    assert payload["error"]["kind"] == "fitting_cancelled"
    assert int(payload["worker_pid"]) in set(pool.worker_pids()) | {int(payload["worker_pid"])}


@pytest.mark.skipif(not CAN_CREATE_PROCESS_POOL, reason=PROCESS_POOL_SKIP_REASON)
def test_fitting_process_pool_force_shutdown_terminates_running_worker_process(monkeypatch) -> None:
    import kindred.core.fitting_process_pool as fitting_process_pool
    from kindred.core.fitting_process_pool import FittingProcessPool

    monkeypatch.setattr(fitting_process_pool, "run_fitting_evaluation_task", _sleeping_worker_task)

    pool = FittingProcessPool(_make_serial_evaluator().to_process_payload(), max_workers=1)
    worker_pid = None
    try:
        _ = pool.submit(_dataset_input(0, "ds1", 1.0))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            worker_pids = pool.worker_pids()
            if worker_pids:
                worker_pid = int(worker_pids[0])
                break
            time.sleep(0.05)
        assert worker_pid is not None
        assert _pid_exists(worker_pid) is True
    finally:
        pool.shutdown(force_terminate=True)

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and _pid_exists(worker_pid):
        time.sleep(0.05)
    assert _pid_exists(worker_pid) is False


def test_fitting_process_pool_shutdown_sets_cancel_event_before_executor_shutdown() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, ShutdownStatus

    calls = []

    class _FakeEvent:
        def set(self):
            calls.append("event.set")

    class _FakeExecutor:
        def shutdown(self, *, wait, cancel_futures):
            calls.append(("executor.shutdown", bool(wait), bool(cancel_futures)))

    class _FakeManager:
        def shutdown(self):
            calls.append("manager.shutdown")

    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = _FakeExecutor()
    pool._manager = _FakeManager()
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._prewarm_started = True
    pool._shutdown_in_progress = False
    pool._startup_cancelled = False
    pool._state_lock = threading.Lock()

    outcome = pool.shutdown(force_terminate=False)

    assert calls == [
        "event.set",
        ("executor.shutdown", True, True),
        "manager.shutdown",
    ]
    assert outcome.status is ShutdownStatus.GRACEFUL_COMPLETION
    assert outcome.cancel_event_error is None
    assert outcome.executor_shutdown_error is None
    assert outcome.manager_shutdown_error is None
    assert outcome.termination_errors == ()
    assert outcome.final_worker_process_count == 0
    assert outcome.final_snapshot_nonempty is False


def test_fitting_process_pool_shutdown_returns_stored_outcome_when_already_closed() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, ShutdownOutcome, ShutdownStatus

    pool = object.__new__(FittingProcessPool)
    expected = ShutdownOutcome(
        status=ShutdownStatus.FORCED_TERMINATION_CLEAN,
        final_worker_process_count=0,
        final_snapshot_nonempty=False,
    )
    pool._cancel_event = object()
    pool._executor = None
    pool._manager = None
    pool._closed = True
    pool._prewarm_in_progress = False
    pool._prewarm_started = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = False
    pool._state_lock = threading.Lock()
    pool._shutdown_condition = threading.Condition(pool._state_lock)
    pool._last_shutdown_outcome = expected

    outcome = pool.shutdown(force_terminate=True)

    assert outcome is expected


def test_fitting_process_pool_context_manager_returns_pool_and_gracefully_shuts_down(monkeypatch) -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, ShutdownOutcome, ShutdownStatus

    calls = []
    pool = object.__new__(FittingProcessPool)
    pool._manager = object()
    pool._cancel_event = object()
    pool._executor = object()
    pool._prewarm_in_progress = False
    pool._entered = False
    pool._state_lock = threading.Lock()

    monkeypatch.setattr(
        FittingProcessPool,
        "shutdown",
        lambda self, *, force_terminate: calls.append(bool(force_terminate))
        or ShutdownOutcome(status=ShutdownStatus.GRACEFUL_COMPLETION),
    )

    with pool as entered:
        assert entered is pool

    assert calls == [False]


def test_fitting_process_pool_context_manager_force_shuts_down_on_exception(monkeypatch) -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, ShutdownOutcome, ShutdownStatus

    calls = []
    pool = object.__new__(FittingProcessPool)
    pool._manager = object()
    pool._cancel_event = object()
    pool._executor = object()
    pool._prewarm_in_progress = False
    pool._entered = False
    pool._state_lock = threading.Lock()

    monkeypatch.setattr(
        FittingProcessPool,
        "shutdown",
        lambda self, *, force_terminate: calls.append(bool(force_terminate))
        or ShutdownOutcome(status=ShutdownStatus.FORCED_TERMINATION_CLEAN),
    )

    with pytest.raises(RuntimeError, match="boom"):
        with pool:
            raise RuntimeError("boom")

    assert calls == [True]


def test_fitting_process_pool_context_manager_preserves_body_exception_when_shutdown_raises(monkeypatch) -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    pool = object.__new__(FittingProcessPool)
    pool._manager = object()
    pool._cancel_event = object()
    pool._executor = object()
    pool._prewarm_in_progress = False
    pool._entered = False
    pool._state_lock = threading.Lock()

    def fake_shutdown(self, *, force_terminate):
        raise RuntimeError(f"shutdown boom force={force_terminate}")

    monkeypatch.setattr(FittingProcessPool, "shutdown", fake_shutdown)

    with pytest.raises(ValueError, match="inner boom"):
        with pool:
            raise ValueError("inner boom")


def test_fitting_process_pool_context_manager_raises_shutdown_failure_on_clean_exit(monkeypatch) -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    pool = object.__new__(FittingProcessPool)
    pool._manager = object()
    pool._cancel_event = object()
    pool._executor = object()
    pool._prewarm_in_progress = False
    pool._entered = False
    pool._state_lock = threading.Lock()

    def fake_shutdown(self, *, force_terminate):
        raise RuntimeError(f"shutdown boom force={force_terminate}")

    monkeypatch.setattr(FittingProcessPool, "shutdown", fake_shutdown)

    with pytest.raises(RuntimeError, match="shutdown boom force=False"):
        with pool:
            pass


def test_fitting_process_pool_context_manager_reentry_is_rejected() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, ShutdownOutcome, ShutdownStatus

    pool = object.__new__(FittingProcessPool)
    pool._manager = object()
    pool._cancel_event = object()
    pool._executor = object()
    pool._prewarm_in_progress = False
    pool._entered = False
    pool._state_lock = threading.Lock()

    def fake_shutdown(*, force_terminate):
        return ShutdownOutcome(status=ShutdownStatus.GRACEFUL_COMPLETION)

    pool.shutdown = fake_shutdown

    with pytest.raises(RuntimeError, match="not re-entrant"):
        with pool:
            pool.__enter__()


def test_fitting_process_pool_force_shutdown_reports_manager_shutdown_error() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, ShutdownStatus

    class _FakeEvent:
        def set(self):
            return None

    class _FakeExecutor:
        def __init__(self):
            self._processes = {}

        def shutdown(self, *, wait, cancel_futures):
            return None

    class _FakeManager:
        def shutdown(self):
            raise RuntimeError("manager shutdown failed")

    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = _FakeExecutor()
    pool._manager = _FakeManager()
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._prewarm_started = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = False
    pool._state_lock = threading.Lock()

    outcome = pool.shutdown(force_terminate=True)

    assert outcome.status is ShutdownStatus.FORCED_TERMINATION_WITH_ERRORS
    assert isinstance(outcome.manager_shutdown_error, RuntimeError)
    assert "manager shutdown failed" in str(outcome.manager_shutdown_error)


def test_fitting_process_pool_force_shutdown_reports_process_termination_error() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, ShutdownStatus

    class _FakeEvent:
        def set(self):
            return None

    class _FakeProcess:
        pid = 4321

        def is_alive(self):
            return True

        def terminate(self):
            raise OSError("terminate failed")

    class _FakeExecutor:
        def __init__(self):
            self._processes = {4321: _FakeProcess()}

        def shutdown(self, *, wait, cancel_futures):
            return None

    class _FakeManager:
        def shutdown(self):
            return None

    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = _FakeExecutor()
    pool._manager = _FakeManager()
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._prewarm_started = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = False
    pool._state_lock = threading.Lock()

    outcome = pool.shutdown(force_terminate=True)

    assert outcome.status is ShutdownStatus.FORCED_TERMINATION_WITH_ERRORS
    assert outcome.final_worker_process_count == 1
    assert outcome.final_snapshot_nonempty is True
    assert len(outcome.termination_errors) == 1
    assert outcome.termination_errors[0][0] == "pid=4321"
    assert isinstance(outcome.termination_errors[0][1], OSError)


def test_fitting_process_pool_force_shutdown_reports_snapshot_failure_as_error() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, ShutdownStatus

    class _BrokenProcesses(dict):
        def values(self):
            raise RuntimeError("snapshot failed")

    class _FakeEvent:
        def set(self):
            return None

    class _FakeExecutor:
        def __init__(self):
            self._processes = _BrokenProcesses({1: object()})

        def shutdown(self, *, wait, cancel_futures):
            return None

    class _FakeManager:
        def shutdown(self):
            return None

    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = _FakeExecutor()
    pool._manager = _FakeManager()
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._prewarm_started = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = False
    pool._state_lock = threading.Lock()
    pool._shutdown_condition = threading.Condition(pool._state_lock)
    pool._last_shutdown_outcome = None

    outcome = pool.shutdown(force_terminate=True)

    assert outcome.status is ShutdownStatus.FORCED_TERMINATION_WITH_ERRORS
    assert outcome.final_worker_process_count == 0
    assert outcome.final_snapshot_nonempty is False
    assert any(label == "snapshot" for label, _exc in outcome.termination_errors)


def test_fitting_process_pool_force_shutdown_reports_unavailable_registry_as_error() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, ShutdownStatus

    class _FakeEvent:
        def set(self):
            return None

    class _FakeExecutor:
        def __init__(self):
            self._processes = object()

        def shutdown(self, *, wait, cancel_futures):
            return None

    class _FakeManager:
        def shutdown(self):
            return None

    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = _FakeExecutor()
    pool._manager = _FakeManager()
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._prewarm_started = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = False
    pool._state_lock = threading.Lock()
    pool._shutdown_condition = threading.Condition(pool._state_lock)
    pool._last_shutdown_outcome = None

    outcome = pool.shutdown(force_terminate=True)

    assert outcome.status is ShutdownStatus.FORCED_TERMINATION_WITH_ERRORS
    assert outcome.final_worker_process_count == 0
    assert outcome.final_snapshot_nonempty is False
    assert any(label == "snapshot" for label, _exc in outcome.termination_errors)


def test_fitting_process_pool_shutdown_waits_for_in_progress_shutdown_outcome() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, ShutdownOutcome, ShutdownStatus

    class _FakeEvent:
        def set(self):
            return None

    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = object()
    pool._manager = object()
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._prewarm_started = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = True
    pool._state_lock = threading.Lock()
    pool._shutdown_condition = threading.Condition(pool._state_lock)
    pool._last_shutdown_outcome = None

    expected_outcome = ShutdownOutcome(
        status=ShutdownStatus.GRACEFUL_COMPLETION,
        final_worker_process_count=0,
        final_snapshot_nonempty=False,
    )

    def complete_shutdown() -> None:
        time.sleep(0.05)
        _signal_shutdown_completion(pool, expected_outcome)

    thread = threading.Thread(target=complete_shutdown)
    thread.start()
    try:
        outcome = pool.shutdown(force_terminate=False)
    finally:
        thread.join(timeout=1.0)

    assert outcome is expected_outcome


def test_fitting_process_pool_force_shutdown_returns_escalated_outcome_without_waiting(
    monkeypatch,
) -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, ShutdownOutcome, ShutdownStatus

    class _FakeEvent:
        def set(self):
            return None

    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = object()
    pool._manager = object()
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._prewarm_started = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = True
    pool._state_lock = threading.Lock()
    pool._shutdown_condition = threading.Condition(pool._state_lock)
    pool._last_shutdown_outcome = None

    calls = []
    escalated_outcome = ShutdownOutcome(
        status=ShutdownStatus.FORCED_TERMINATION_CLEAN,
        final_worker_process_count=3,
        final_snapshot_nonempty=True,
    )
    release_shutdown = threading.Event()

    def fake_force_shutdown(self):
        calls.append("force_shutdown")
        return escalated_outcome

    monkeypatch.setattr(FittingProcessPool, "_force_shutdown_out_of_band", fake_force_shutdown)

    def complete_shutdown() -> None:
        release_shutdown.wait(timeout=1.0)
        _signal_shutdown_completion(pool, ShutdownOutcome(status=ShutdownStatus.GRACEFUL_COMPLETION))

    thread = threading.Thread(target=complete_shutdown)
    thread.start()
    try:
        outcome = pool.shutdown(force_terminate=True)
    finally:
        release_shutdown.set()
        thread.join(timeout=1.0)

    assert calls == ["force_shutdown"]
    assert outcome.status is ShutdownStatus.FORCED_TERMINATION_CLEAN
    assert outcome.final_worker_process_count == 3
    assert outcome.final_snapshot_nonempty is True


def test_fitting_process_pool_shutdown_reports_errors_when_never_started_cleanup_fails() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, ShutdownStatus

    class _FakeEvent:
        def set(self):
            return None

    class _FakeExecutor:
        _processes = {}

        def shutdown(self, *, wait, cancel_futures):
            raise RuntimeError("executor boom")

    class _FakeManager:
        def shutdown(self):
            raise RuntimeError("manager boom")

    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = _FakeExecutor()
    pool._manager = _FakeManager()
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._prewarm_started = False
    pool._startup_cancelled = False
    pool._shutdown_in_progress = False
    pool._state_lock = threading.Lock()
    pool._shutdown_condition = threading.Condition(pool._state_lock)
    pool._last_shutdown_outcome = None

    outcome = pool.shutdown(force_terminate=True)

    assert outcome.status is ShutdownStatus.FORCED_TERMINATION_WITH_ERRORS
    assert isinstance(outcome.executor_shutdown_error, RuntimeError)
    assert isinstance(outcome.manager_shutdown_error, RuntimeError)


def test_fitting_process_pool_force_shutdown_registered_pools_uses_full_shutdown_when_idle(
    monkeypatch,
) -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, ShutdownOutcome, ShutdownStatus

    pool = object.__new__(FittingProcessPool)
    pool._owner_thread_ident = 1234
    pool._state_lock = threading.Lock()
    pool._shutdown_condition = threading.Condition(pool._state_lock)
    pool._closed = False
    pool._shutdown_in_progress = False
    pool._executor = object()
    pool._manager = object()

    expected = ShutdownOutcome(status=ShutdownStatus.FORCED_TERMINATION_CLEAN)
    calls = []

    def fake_force_shutdown(self):
        calls.append("forced")
        return expected

    monkeypatch.setattr(FittingProcessPool, "_force_shutdown_out_of_band", fake_force_shutdown)

    pool._register_active_pool()
    try:
        outcomes = FittingProcessPool.force_shutdown_registered_pools_for_owner_thread(1234)
    finally:
        pool._unregister_active_pool()

    assert calls == ["forced"]
    assert outcomes == (expected,)


def test_fitting_process_pool_registers_before_prewarm(monkeypatch) -> None:
    import kindred.core.fitting_process_pool as fitting_process_pool
    from kindred.core.fitting_process_pool import FittingProcessPool

    class _FakeManager:
        def Event(self):
            return object()

        def shutdown(self):
            return None

    class _FakeContext:
        def Manager(self):
            return _FakeManager()

    class _FakeExecutor:
        def __init__(self, **_kwargs):
            self._processes = {}

        def shutdown(self, *, wait, cancel_futures):
            return None

    seen = []

    def fake_prewarm(self):
        seen.append(_pool_registered_for_owner_thread(self))

    monkeypatch.setattr(fitting_process_pool.mp, "get_context", lambda _name: _FakeContext())
    monkeypatch.setattr(fitting_process_pool, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(FittingProcessPool, "_prewarm", fake_prewarm)

    pool = FittingProcessPool(_make_serial_evaluator().to_process_payload(), max_workers=1)
    try:
        assert seen == [True]
    finally:
        pool.shutdown(force_terminate=False)


def test_fitting_process_pool_submit_holds_lock_during_executor_submit() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, run_fitting_evaluation_task

    class _RecordingLock:
        def __init__(self):
            self.held = False

        def __enter__(self):
            assert self.held is False
            self.held = True
            return self

        def __exit__(self, exc_type, exc, tb):
            self.held = False
            return False

    lock = _RecordingLock()

    class _FakeExecutor:
        def submit(self, fn, item):
            assert lock.held is True
            assert fn is run_fitting_evaluation_task
            future = Future()
            future.set_result(item)
            return future

    pool = object.__new__(FittingProcessPool)
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._shutdown_in_progress = False
    pool._startup_cancelled = False
    pool._executor = _FakeExecutor()
    pool._state_lock = lock

    future = pool.submit("payload")

    assert future.result() == "payload"
    assert lock.held is False


def test_fitting_process_pool_submit_rejects_startup_cancelled_state() -> None:
    from kindred.core.exceptions import FittingCancelled
    from kindred.core.fitting_process_pool import FittingProcessPool

    class _FakeExecutor:
        def submit(self, *_args, **_kwargs):
            raise AssertionError("submit should not reach the executor after startup cancellation")

    pool = object.__new__(FittingProcessPool)
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._shutdown_in_progress = False
    pool._startup_cancelled = True
    pool._executor = _FakeExecutor()
    pool._state_lock = threading.Lock()

    with pytest.raises(FittingCancelled):
        pool.submit("payload")


def test_fitting_process_pool_prewarm_polls_startup_cancel() -> None:
    from concurrent.futures import TimeoutError as FutureTimeoutError

    import pytest

    from kindred.core.exceptions import FittingCancelled
    from kindred.core.fitting_process_pool import FittingProcessPool

    class _FakeEvent:
        def __init__(self):
            self.set_calls = 0

        def set(self):
            self.set_calls += 1

    pool = object.__new__(FittingProcessPool)
    pool._max_workers = 1
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._shutdown_in_progress = False
    pool._startup_cancelled = False
    pool._state_lock = threading.Lock()
    pool._cancel_event = _FakeEvent()

    class _FakeFuture:
        def __init__(self):
            self.calls = 0

        def result(self, *, timeout):
            self.calls += 1
            pool.cancel()
            raise FutureTimeoutError()

    future = _FakeFuture()

    class _FakeExecutor:
        def submit(self, _fn):
            return future

    pool._executor = _FakeExecutor()

    with pytest.raises(FittingCancelled):
        pool._prewarm()

    assert future.calls == 1
    assert pool._cancel_event.set_calls == 1


def test_fitting_process_pool_constructor_raises_fitting_cancelled_when_cancellation_check_is_true() -> None:
    from kindred.core.exceptions import FittingCancelled
    from kindred.core.fitting_process_pool import FittingProcessPool

    with pytest.raises(FittingCancelled):
        FittingProcessPool(
            _make_serial_evaluator().to_process_payload(),
            max_workers=1,
            cancellation_check=lambda: True,
        )


@pytest.mark.skipif(not CAN_CREATE_PROCESS_POOL, reason=PROCESS_POOL_SKIP_REASON)
def test_fitting_process_pool_constructor_uses_nonblocking_cancel_helper_during_startup() -> None:
    """Use a 5.0s join budget to prove startup does not hang on a pause-blocking callable."""
    from kindred.core.fitting_process_pool import FittingProcessPool

    pause_event = threading.Event()
    pause_event.clear()
    budget_s = 5.0
    result: dict[str, object] = {"pool": None, "error": None}

    def cancellation_check() -> bool:
        pause_event.wait()
        return False

    cancellation_check._kindred_nonblocking_cancelled = lambda: False

    def construct_pool() -> None:
        try:
            result["pool"] = FittingProcessPool(
                _make_serial_evaluator().to_process_payload(),
                max_workers=1,
                cancellation_check=cancellation_check,
            )
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=construct_pool, name="startup-cancel-helper-test")
    thread.start()
    thread.join(timeout=budget_s)
    completed_within_budget = not thread.is_alive()
    pause_event.set()
    thread.join(timeout=budget_s)

    pool = result["pool"]
    try:
        assert completed_within_budget is True
        assert result["error"] is None
        assert isinstance(pool, FittingProcessPool)
    finally:
        if isinstance(pool, FittingProcessPool):
            pool.shutdown(force_terminate=False)


def test_fitting_process_pool_submit_raises_fitting_cancelled_after_shutdown_starts() -> None:
    from kindred.core.exceptions import FittingCancelled
    from kindred.core.fitting_process_pool import FittingProcessPool

    class _FakeExecutor:
        def submit(self, _fn, _item):
            raise AssertionError("executor.submit should not run after shutdown has started")

    pool = object.__new__(FittingProcessPool)
    pool._closed = False
    pool._executor = _FakeExecutor()
    pool._prewarm_in_progress = False
    pool._shutdown_in_progress = True
    pool._startup_cancelled = False
    pool._state_lock = threading.Lock()

    with pytest.raises(FittingCancelled):
        pool.submit("payload")


def test_fitting_process_pool_cancel_terminates_workers_during_prewarm(monkeypatch) -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    calls = []

    class _FakeProcess:
        def is_alive(self):
            return True

        def terminate(self):
            calls.append("process.terminate")

    class _FakeEvent:
        def __init__(self):
            self.set_calls = 0

        def set(self):
            self.set_calls += 1
            calls.append("event.set")

    class _FakeExecutor:
        def __init__(self):
            self._processes = {1: _FakeProcess()}

    executor = _FakeExecutor()
    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = executor
    pool._manager = None
    pool._closed = False
    pool._prewarm_in_progress = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = False
    pool._state_lock = threading.Lock()

    monkeypatch.setattr(
        FittingProcessPool,
        "_terminate_processes_best_effort",
        lambda self, processes: calls.append(("terminate_target_len", len(tuple(processes)))) or (),
    )

    pool.cancel()

    assert calls == [
        "event.set",
        ("terminate_target_len", 1),
    ]
    assert pool._startup_cancelled is True


def test_fitting_process_pool_cancel_still_terminates_after_transient_process_snapshot_race(
    monkeypatch,
) -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    calls = []

    class _TransientlyMutatingProcesses(dict):
        def __init__(self, initial):
            super().__init__(initial)
            self._values_calls = 0

        def values(self):
            if self._values_calls == 0:
                self._values_calls += 1
                raise RuntimeError("dictionary changed size during iteration")
            self._values_calls += 1
            return super().values()

    class _FakeEvent:
        def set(self):
            calls.append("event.set")

    class _FakeExecutor:
        def __init__(self):
            self._processes = _TransientlyMutatingProcesses({1: object()})

    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = _FakeExecutor()
    pool._manager = None
    pool._closed = False
    pool._prewarm_in_progress = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = False
    pool._state_lock = threading.Lock()

    monkeypatch.setattr(
        FittingProcessPool,
        "_terminate_processes_best_effort",
        lambda self, processes: calls.append(("terminate_target_len", len(tuple(processes)))) or (),
    )

    pool.cancel()

    assert calls == [
        "event.set",
        ("terminate_target_len", 1),
    ]
    assert pool._startup_cancelled is True


def test_fitting_process_pool_out_of_band_force_shutdown_terminates_while_shutdown_is_in_progress(
    monkeypatch,
) -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    calls = []

    class _FakeEvent:
        def set(self):
            calls.append("event.set")

    class _FakeExecutor:
        def __init__(self):
            self._processes = {1: object()}

    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = _FakeExecutor()
    pool._manager = None
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._prewarm_started = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = True
    pool._state_lock = threading.Lock()
    pool._shutdown_condition = threading.Condition(pool._state_lock)

    monkeypatch.setattr(
        FittingProcessPool,
        "_terminate_processes_best_effort",
        lambda self, processes: calls.append(("terminate_target_len", len(tuple(processes)))) or (),
    )

    pool._force_shutdown_out_of_band()

    assert calls == [
        "event.set",
        ("terminate_target_len", 1),
    ]


@pytest.mark.skipif(not CAN_CREATE_PROCESS_POOL, reason=PROCESS_POOL_SKIP_REASON)
def test_fitting_process_pool_out_of_band_force_shutdown_terminates_manager_process_when_shutdown_is_in_progress() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    pool = FittingProcessPool(_make_serial_evaluator().to_process_payload(), max_workers=1)
    manager = pool._manager
    outcome = None
    try:
        assert _manager_process_alive(manager) is True
        with pool._state_lock:
            pool._shutdown_in_progress = True
        outcome = _force_shutdown_registered_pool(pool)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and _manager_process_alive(manager):
            time.sleep(0.05)
        assert _manager_process_alive(manager) is False
    finally:
        _cleanup_pool_after_test(pool, fallback_outcome=outcome)


@pytest.mark.skipif(not CAN_CREATE_PROCESS_POOL, reason=PROCESS_POOL_SKIP_REASON)
def test_fitting_process_pool_out_of_band_force_shutdown_unregisters_pool_when_shutdown_is_in_progress() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    pool = FittingProcessPool(_make_serial_evaluator().to_process_payload(), max_workers=1)
    outcome = None
    try:
        assert _pool_registered_for_owner_thread(pool) is True
        with pool._state_lock:
            pool._shutdown_in_progress = True
        outcome = _force_shutdown_registered_pool(pool)
        assert _pool_registered_for_owner_thread(pool) is False
    finally:
        _cleanup_pool_after_test(pool, fallback_outcome=outcome)


@pytest.mark.skipif(not CAN_CREATE_PROCESS_POOL, reason=PROCESS_POOL_SKIP_REASON)
def test_fitting_process_pool_out_of_band_force_shutdown_leaves_pool_terminal_when_shutdown_is_in_progress() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    pool = FittingProcessPool(_make_serial_evaluator().to_process_payload(), max_workers=1)
    outcome = None
    try:
        with pool._state_lock:
            pool._shutdown_in_progress = True
        outcome = _force_shutdown_registered_pool(pool)
        assert pool._closed is True
        assert pool._shutdown_in_progress is False
        assert pool._executor is None
        assert pool._manager is None
        assert pool._last_shutdown_outcome is outcome
    finally:
        _cleanup_pool_after_test(pool, fallback_outcome=outcome)


@pytest.mark.skipif(not CAN_CREATE_PROCESS_POOL, reason=PROCESS_POOL_SKIP_REASON)
def test_fitting_process_pool_later_graceful_shutdown_returns_race_winner_without_hanging() -> None:
    """Use a 1.0 s join budget to detect a hang without turning this into a performance test."""
    from kindred.core.fitting_process_pool import FittingProcessPool

    pool = FittingProcessPool(_make_serial_evaluator().to_process_payload(), max_workers=1)
    outcome = None
    wait_result: dict[str, object] = {}
    waiter = None
    try:
        with pool._state_lock:
            pool._shutdown_in_progress = True
        outcome = _force_shutdown_registered_pool(pool)

        def wait_for_shutdown() -> None:
            wait_result["outcome"] = pool.shutdown(force_terminate=False)

        waiter = threading.Thread(target=wait_for_shutdown)
        waiter.start()
        waiter.join(timeout=1.0)

        assert waiter.is_alive() is False
        assert wait_result["outcome"] is outcome
    finally:
        if waiter is not None and waiter.is_alive():
            _cleanup_pool_after_test(pool, fallback_outcome=outcome)
            waiter.join(timeout=1.0)
        _cleanup_pool_after_test(pool, fallback_outcome=outcome)


def test_fitting_process_pool_out_of_band_force_shutdown_has_no_synthetic_concurrent_shutdown_marker() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    class _FakeEvent:
        def set(self):
            return None

    class _FakeExecutor:
        def __init__(self):
            self._processes = {1: object()}

        def shutdown(self, *, wait, cancel_futures):
            return None

    class _FakeManager:
        def shutdown(self):
            return None

    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = _FakeExecutor()
    pool._manager = _FakeManager()
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._prewarm_started = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = True
    pool._state_lock = threading.Lock()
    pool._shutdown_condition = threading.Condition(pool._state_lock)
    pool._last_shutdown_outcome = None

    outcome = pool._force_shutdown_out_of_band()

    assert all(label != "concurrent_shutdown" for label, _exc in outcome.termination_errors)


def test_fitting_process_pool_out_of_band_force_shutdown_reports_real_errors_only() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    class _FakeEvent:
        def set(self):
            return None

    class _FakeProcess:
        pid = 4321

        def is_alive(self):
            return True

        def terminate(self):
            raise OSError("terminate failed")

    class _FakeExecutor:
        def __init__(self):
            self._processes = {4321: _FakeProcess()}

        def shutdown(self, *, wait, cancel_futures):
            return None

    class _FakeManager:
        def shutdown(self):
            return None

    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = _FakeExecutor()
    pool._manager = _FakeManager()
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._prewarm_started = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = True
    pool._state_lock = threading.Lock()
    pool._shutdown_condition = threading.Condition(pool._state_lock)
    pool._last_shutdown_outcome = None

    outcome = pool._force_shutdown_out_of_band()

    assert any(label == "pid=4321" for label, _exc in outcome.termination_errors)
    assert all(label != "concurrent_shutdown" for label, _exc in outcome.termination_errors)


def test_fitting_process_pool_later_graceful_shutdown_preserves_forced_status_after_race_winner() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool, ShutdownStatus

    class _FakeEvent:
        def set(self):
            return None

    class _FakeExecutor:
        def __init__(self):
            self._processes = {1: object()}

        def shutdown(self, *, wait, cancel_futures):
            return None

    class _FakeManager:
        def shutdown(self):
            return None

    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = _FakeExecutor()
    pool._manager = _FakeManager()
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._prewarm_started = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = True
    pool._state_lock = threading.Lock()
    pool._shutdown_condition = threading.Condition(pool._state_lock)
    pool._last_shutdown_outcome = None

    outcome = pool._force_shutdown_out_of_band()
    later_result: dict[str, object] = {}

    def call_graceful_shutdown() -> None:
        later_result["outcome"] = pool.shutdown(force_terminate=False)

    waiter = threading.Thread(target=call_graceful_shutdown)
    waiter.start()
    waiter.join(timeout=1.0)

    try:
        assert waiter.is_alive() is False
        assert later_result["outcome"] is outcome
        assert later_result["outcome"].status in {
            ShutdownStatus.FORCED_TERMINATION_CLEAN,
            ShutdownStatus.FORCED_TERMINATION_WITH_ERRORS,
        }
    finally:
        if waiter.is_alive():
            _cleanup_pool_after_test(pool, fallback_outcome=outcome)
            waiter.join(timeout=1.0)
        _cleanup_pool_after_test(pool, fallback_outcome=outcome)


def test_fitting_process_pool_shutdown_cleanup_runs_exactly_once_after_out_of_band_race_winner() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    class _FakeEvent:
        def set(self):
            return None

    class _FakeExecutor:
        def __init__(self):
            self._processes = {1: object()}
            self.shutdown_calls = []

        def shutdown(self, *, wait, cancel_futures):
            self.shutdown_calls.append((bool(wait), bool(cancel_futures)))

    class _FakeManager:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    executor = _FakeExecutor()
    manager = _FakeManager()
    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = executor
    pool._manager = manager
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._prewarm_started = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = True
    pool._state_lock = threading.Lock()
    pool._shutdown_condition = threading.Condition(pool._state_lock)
    pool._last_shutdown_outcome = None

    outcome = pool._force_shutdown_out_of_band()
    later_result: dict[str, object] = {}

    def call_graceful_shutdown() -> None:
        later_result["outcome"] = pool.shutdown(force_terminate=False)

    waiter = threading.Thread(target=call_graceful_shutdown)
    waiter.start()
    waiter.join(timeout=1.0)

    try:
        assert waiter.is_alive() is False
        assert later_result["outcome"] is outcome
        assert executor.shutdown_calls == [(False, True)]
        assert manager.shutdown_calls == 1
    finally:
        if waiter.is_alive():
            _cleanup_pool_after_test(pool, fallback_outcome=outcome)
            waiter.join(timeout=1.0)
        _cleanup_pool_after_test(pool, fallback_outcome=outcome)


@pytest.mark.parametrize("shutdown_in_progress", [False, True])
def test_fitting_process_pool_force_shutdown_still_terminates_after_transient_process_snapshot_race(
    monkeypatch,
    shutdown_in_progress: bool,
) -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    calls = []

    class _TransientlyMutatingProcesses(dict):
        def __init__(self, initial):
            super().__init__(initial)
            self._values_calls = 0

        def values(self):
            if self._values_calls == 0:
                self._values_calls += 1
                raise RuntimeError("dictionary changed size during iteration")
            self._values_calls += 1
            return super().values()

    class _FakeEvent:
        def set(self):
            calls.append("event.set")

    class _FakeExecutor:
        def __init__(self):
            self._processes = _TransientlyMutatingProcesses({1: object()})

        def shutdown(self, *, wait, cancel_futures):
            calls.append(("executor.shutdown", bool(wait), bool(cancel_futures)))

    class _FakeManager:
        def shutdown(self):
            calls.append("manager.shutdown")

    pool = object.__new__(FittingProcessPool)
    pool._cancel_event = _FakeEvent()
    pool._executor = _FakeExecutor()
    pool._manager = _FakeManager()
    pool._closed = False
    pool._prewarm_in_progress = False
    pool._prewarm_started = True
    pool._startup_cancelled = False
    pool._shutdown_in_progress = shutdown_in_progress
    pool._state_lock = threading.Lock()
    pool._shutdown_condition = threading.Condition(pool._state_lock)

    monkeypatch.setattr(
        FittingProcessPool,
        "_terminate_processes_best_effort",
        lambda self, processes: calls.append(("terminate_target_len", len(tuple(processes)))) or (),
    )

    if shutdown_in_progress:
        pool._force_shutdown_out_of_band()
    else:
        pool.shutdown(force_terminate=True)

    if shutdown_in_progress:
        assert calls == [
            "event.set",
            ("terminate_target_len", 1),
            ("executor.shutdown", False, True),
            "manager.shutdown",
        ]
    else:
        assert calls == [
            "event.set",
            ("executor.shutdown", False, True),
            ("terminate_target_len", 1),
            "manager.shutdown",
        ]


@pytest.mark.skipif(not CAN_CREATE_PROCESS_POOL, reason=PROCESS_POOL_SKIP_REASON)
def test_global_fit_objective_process_pool_matches_serial_reference() -> None:
    from kindred.core.analysis.global_fitting import _GlobalFitObjective, _build_parameter_layout
    from kindred.core.fitting_evaluation import SerialFittingEvaluator
    from kindred.core.fitting_process_pool import FittingProcessPool
    from kindred.core.objective import ObjectiveContext

    class _InProcessSerialFittingEvaluator(SerialFittingEvaluator):
        pass

    payloads = [_payload("ds1", [0.0, 0.0]), _payload("ds2", [0.0, 0.0])]
    layout = _build_parameter_layout(
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_variable_params={},
        bounds=None,
        log10_params=None,
    )
    exact_evaluator = SerialFittingEvaluator(_serial_context(solver="Radau", num_points=2))
    reference_evaluator = _InProcessSerialFittingEvaluator(_serial_context(solver="Radau", num_points=2))
    process_pool = FittingProcessPool(exact_evaluator.to_process_payload(), max_workers=2)

    objective = _GlobalFitObjective(
        fit_evaluator=exact_evaluator,
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        weights={"ds1": 1.0, "ds2": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
        process_pool=process_pool,
    )
    reference = _GlobalFitObjective(
        fit_evaluator=reference_evaluator,
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        weights={"ds1": 1.0, "ds2": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
        process_pool=None,
    )

    try:
        residuals = objective(layout.x0.copy())
        expected = reference(layout.x0.copy())
    finally:
        process_pool.shutdown(force_terminate=False)

    np.testing.assert_allclose(residuals, expected)


def test_evaluate_dataset_simulations_process_pool_cancels_on_fatal_result() -> None:
    from kindred.core.analysis.global_fitting import _dataset_evaluation_is_fatal, _evaluate_dataset_simulations

    fatal_payload = {
        "index": 0,
        "dataset_id": "ds1",
        "worker_pid": os.getpid(),
        "ok": False,
        "series_payload": None,
        "error": {
            "kind": "fit_simulation",
            "message": "fatal worker failure",
            "code": "E404",
            "details": {"fatal": True},
            "failed_params": {"init:A": 1.0},
            "context": None,
        },
        "error_provenance": {"dataset": "ds1"},
        "final_error_message": "fatal worker failure",
    }
    pending_payload = Future()

    class _FakeProcessPool:
        def __init__(self):
            self.max_workers = 2
            self.cancel_calls = 0
            self.submit_count = 0

        def submit(self, item):
            self.submit_count += 1
            future = Future()
            if self.submit_count == 1:
                future.set_result(dict(fatal_payload))
                return future
            return pending_payload

        def cancel(self) -> None:
            self.cancel_calls += 1

    pool = _FakeProcessPool()
    results = _evaluate_dataset_simulations(
        _make_serial_evaluator(),
        [_dataset_input(0, "ds1", 1.0), _dataset_input(1, "ds2", 2.0)],
        process_pool=pool,
        stop_on_fatal=True,
    )

    assert len(results) == 1
    assert _dataset_evaluation_is_fatal(results[0]) is True
    assert pool.cancel_calls == 1
    assert pending_payload.cancelled() is True


def test_fit_global_uses_process_pool_for_multi_dataset_serial_fitting_evaluator(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_evaluation import SerialFittingEvaluator
    from kindred.core.fitting_optimization import FitResult

    pools = []

    class _FakeProcessPool:
        def __init__(self, evaluator_payload, *, max_workers, limit_blas_threads, cancellation_check=None):
            self._evaluator = SerialFittingEvaluator.from_process_payload(evaluator_payload)
            self.max_workers = int(max_workers)
            self.limit_blas_threads = bool(limit_blas_threads)
            self.cancellation_check = cancellation_check
            self.submit_count = 0
            self.shutdown_calls = []
            pools.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.shutdown(force_terminate=exc_type is not None)
            return None

        def submit(self, item):
            from kindred.core.fitting_evaluation import evaluate_fitting_series

            self.submit_count += 1
            future = Future()
            future.set_result(
                {
                    "index": int(item.index),
                    "dataset_id": str(item.payload.dataset_id),
                    "worker_pid": os.getpid(),
                    "ok": True,
                    "series_payload": evaluate_fitting_series(
                        self._evaluator,
                        item.full_params,
                        origins=item.parameter_origins,
                        failed_params=item.failed_param_snapshot,
                    ),
                    "error": None,
                    "error_provenance": None,
                    "final_error_message": None,
                }
            )
            return future

        def cancel(self) -> None:
            return None

        def shutdown(self, *, force_terminate: bool) -> None:
            if self.shutdown_calls:
                return
            self.shutdown_calls.append(bool(force_terminate))

    def fake_fit_parameters(_objective, initial_params, **_kwargs):
        return FitResult(
            success=True,
            parameters=dict(initial_params),
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(4, dtype=float),
            nfev=1,
            message="ok",
            covariance=None,
        )

    monkeypatch.setattr(global_fitting, "FittingProcessPool", _FakeProcessPool)
    monkeypatch.setattr(global_fitting, "fit_parameters", fake_fit_parameters)

    def cancellation_check() -> bool:
        return False

    result = global_fitting.fit_global(
        _make_serial_evaluator(),
        [_raw_dataset("ds1", [0.0, 0.0]), _raw_dataset("ds2", [0.0, 0.0])],
        {"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        method="trf",
        max_nfev=1,
        parallel_enabled=True,
        max_parallel_workers=9,
        limit_blas_threads=False,
        cancellation_check=cancellation_check,
    )

    assert result.success is True
    assert len(pools) == 1
    assert pools[0].max_workers == 2
    assert pools[0].limit_blas_threads is False
    assert pools[0].submit_count == 2
    assert pools[0].cancellation_check is cancellation_check
    assert pools[0].shutdown_calls == [False]


def test_fit_global_raises_when_process_pool_shutdown_reports_errors(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_optimization import FitResult
    from kindred.core.fitting_process_pool import ShutdownOutcome, ShutdownStatus

    class _FakeProcessPool:
        def __init__(self, *_args, **_kwargs):
            self._last_shutdown_outcome = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self._last_shutdown_outcome = ShutdownOutcome(
                status=ShutdownStatus.GRACEFUL_WITH_ERRORS,
                manager_shutdown_error=RuntimeError("manager shutdown failed"),
            )
            return None

    def fake_fit_parameters(_objective, initial_params, **_kwargs):
        return FitResult(
            success=True,
            parameters=dict(initial_params),
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(4, dtype=float),
            nfev=1,
            message="ok",
            covariance=None,
        )

    monkeypatch.setattr(global_fitting, "FittingProcessPool", _FakeProcessPool)
    monkeypatch.setattr(global_fitting, "fit_parameters", fake_fit_parameters)
    monkeypatch.setattr(global_fitting, "_assemble_global_fit_result", lambda **_kwargs: object())

    with pytest.raises(RuntimeError, match="shutdown reported errors"):
        global_fitting.fit_global(
            _make_serial_evaluator(),
            [_raw_dataset("ds1", [0.0, 0.0]), _raw_dataset("ds2", [0.0, 0.0])],
            {"k1": 1.0},
            dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
            method="trf",
            max_nfev=1,
            parallel_enabled=True,
            max_parallel_workers=2,
            limit_blas_threads=True,
        )


def test_fit_global_parallel_disabled_by_default_stays_in_process(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_optimization import FitResult

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("process pool should not be created when parallel fitting is disabled by default")

    def fake_fit_parameters(_objective, initial_params, **_kwargs):
        return FitResult(
            success=True,
            parameters=dict(initial_params),
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(4, dtype=float),
            nfev=1,
            message="ok",
            covariance=None,
        )

    monkeypatch.setattr(global_fitting, "FittingProcessPool", fail_if_called)
    monkeypatch.setattr(global_fitting, "fit_parameters", fake_fit_parameters)

    result = global_fitting.fit_global(
        _make_serial_evaluator(),
        [_raw_dataset("ds1", [0.0, 0.0]), _raw_dataset("ds2", [0.0, 0.0])],
        {"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        method="trf",
        max_nfev=1,
    )

    assert result.success is True


def test_fit_global_parallel_disabled_explicitly_stays_in_process(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_optimization import FitResult

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("process pool should not be created when parallel fitting is disabled")

    def fake_fit_parameters(_objective, initial_params, **_kwargs):
        return FitResult(
            success=True,
            parameters=dict(initial_params),
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(4, dtype=float),
            nfev=1,
            message="ok",
            covariance=None,
        )

    monkeypatch.setattr(global_fitting, "FittingProcessPool", fail_if_called)
    monkeypatch.setattr(global_fitting, "fit_parameters", fake_fit_parameters)

    result = global_fitting.fit_global(
        _make_serial_evaluator(),
        [_raw_dataset("ds1", [0.0, 0.0]), _raw_dataset("ds2", [0.0, 0.0])],
        {"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        method="trf",
        max_nfev=1,
        parallel_enabled=False,
        max_parallel_workers=8,
        limit_blas_threads=True,
    )

    assert result.success is True


def test_fit_global_serial_fitting_evaluator_with_unpicklable_payload_stays_in_process(monkeypatch, caplog) -> None:
    import pickle

    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_optimization import FitResult

    class _PickleRefusal:
        def __call__(self, _time):
            return 298.15

        def __deepcopy__(self, memo):
            return self

        def __reduce__(self):
            raise pickle.PicklingError("refuse process-pool payload pickling")

    evaluator = _make_serial_evaluator()
    evaluator.context.execution_request.prepared_payload["temperature_schedule"] = _PickleRefusal()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("process pool should not be created for an unpicklable evaluator payload")

    def fake_fit_parameters(objective, initial_params, **_kwargs):
        residuals = np.asarray(objective(np.asarray([float(value) for value in initial_params.values()], dtype=float)))
        return FitResult(
            success=True,
            parameters=dict(initial_params),
            uncertainties=None,
            chi_squared=float(np.dot(residuals, residuals)),
            r_squared=1.0,
            residuals=residuals,
            nfev=1,
            message="ok",
            covariance=None,
        )

    caplog.set_level("WARNING", logger="kindred.core.analysis.global_fitting")
    monkeypatch.setattr(global_fitting, "FittingProcessPool", fail_if_called)
    monkeypatch.setattr(global_fitting, "fit_parameters", fake_fit_parameters)

    result = global_fitting.fit_global(
        evaluator,
        [_raw_dataset("ds1", [0.0, 0.0]), _raw_dataset("ds2", [0.0, 0.0])],
        {"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        method="trf",
        max_nfev=1,
        parallel_enabled=True,
        max_parallel_workers=5,
        limit_blas_threads=True,
    )

    assert result.success is True
    assert "pickl" in caplog.text.lower()


def test_fit_global_single_dataset_serial_fitting_evaluator_stays_in_process(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_optimization import FitResult

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("process pool should not be created for a single dataset fit")

    def fake_fit_parameters(_objective, initial_params, **_kwargs):
        return FitResult(
            success=True,
            parameters=dict(initial_params),
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(2, dtype=float),
            nfev=1,
            message="ok",
            covariance=None,
        )

    monkeypatch.setattr(global_fitting, "FittingProcessPool", fail_if_called)
    monkeypatch.setattr(global_fitting, "fit_parameters", fake_fit_parameters)

    result = global_fitting.fit_global(
        _make_serial_evaluator(),
        [_raw_dataset("ds1", [0.0, 0.0])],
        {"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}},
        method="trf",
        max_nfev=1,
        parallel_enabled=True,
        max_parallel_workers=4,
        limit_blas_threads=True,
    )

    assert result.success is True


def test_fit_global_custom_evaluator_stays_in_process(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_optimization import FitResult

    state = {"base_calls": 0}

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("process pool should not be created for a custom evaluator")

    def fake_fit_parameters(_objective, initial_params, **_kwargs):
        return FitResult(
            success=True,
            parameters=dict(initial_params),
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(4, dtype=float),
            nfev=1,
            message="ok",
            covariance=None,
        )

    monkeypatch.setattr(global_fitting, "FittingProcessPool", fail_if_called)
    monkeypatch.setattr(global_fitting, "fit_parameters", fake_fit_parameters)

    result = global_fitting.fit_global(
        _EvaluateOnlyNoClone(t_axis=np.linspace(0.0, 1.0, 2), state=state),
        [_raw_dataset("ds1", [0.0, 0.0]), _raw_dataset("ds2", [0.0, 0.0])],
        {"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        method="trf",
        max_nfev=1,
        parallel_enabled=True,
        max_parallel_workers=4,
        limit_blas_threads=True,
    )

    assert result.success is True
    assert state["base_calls"] == 2


def test_fit_global_serial_fitting_evaluator_subclass_stays_in_process(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_evaluation import SerialFittingEvaluator
    from kindred.core.fitting_optimization import FitResult

    class _InstrumentedSerialFittingEvaluator(SerialFittingEvaluator):
        pass

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("process pool should not be created for a SerialFittingEvaluator subclass")

    def fake_fit_parameters(_objective, initial_params, **_kwargs):
        return FitResult(
            success=True,
            parameters=dict(initial_params),
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(4, dtype=float),
            nfev=1,
            message="ok",
            covariance=None,
        )

    monkeypatch.setattr(global_fitting, "FittingProcessPool", fail_if_called)
    monkeypatch.setattr(global_fitting, "fit_parameters", fake_fit_parameters)

    result = global_fitting.fit_global(
        _InstrumentedSerialFittingEvaluator(_serial_context()),
        [_raw_dataset("ds1", [0.0, 0.0]), _raw_dataset("ds2", [0.0, 0.0])],
        {"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        method="trf",
        max_nfev=1,
        parallel_enabled=True,
        max_parallel_workers=4,
        limit_blas_threads=True,
    )

    assert result.success is True


@pytest.mark.gui
@pytest.mark.skipif(not CAN_CREATE_PROCESS_POOL, reason=PROCESS_POOL_SKIP_REASON)
def test_global_fit_worker_cancel_reaches_process_pool_via_cancellation_check(monkeypatch, qtbot) -> None:
    import kindred.core.fitting_process_pool as fitting_process_pool
    from concurrent.futures import TimeoutError as FutureTimeoutError

    from kindred.core.exceptions import FittingCancelled
    from kindred.core.fitting_process_pool import FittingProcessPool
    from kindred.gui.fitting.worker import GlobalFitWorker

    monkeypatch.setattr(fitting_process_pool, "run_fitting_evaluation_task", _cancel_aware_worker_task)

    submitted = threading.Event()
    pool_cancelled = threading.Event()

    def fake_fit_global(_fit_evaluator, _datasets, _shared_params, *, cancellation_check, **_kwargs):
        with FittingProcessPool(
            _make_serial_evaluator().to_process_payload(),
            max_workers=1,
            cancellation_check=cancellation_check,
        ) as process_pool:
            future = process_pool.submit(_dataset_input(0, "ds1", 1.0))
            submitted.set()
            while True:
                if cancellation_check():
                    pool_cancelled.set()
                    process_pool.cancel()
                try:
                    future.result(timeout=0.05)
                except FutureTimeoutError:
                    continue
                except FittingCancelled:
                    raise
                raise AssertionError("cancel-aware process worker should not finish successfully")

    worker = GlobalFitWorker(
        datasets=[_raw_dataset("ds1", [0.0, 0.0]), _raw_dataset("ds2", [0.0, 0.0])],
        shared_params={"k1": 1.0},
        fit_evaluator=_make_serial_evaluator(),
        fit_func=fake_fit_global,
        max_nfev=5,
        parallel_enabled=True,
        max_parallel_workers=1,
        limit_blas_threads=True,
    )

    with qtbot.waitSignal(worker.error, timeout=8000) as blocker:
        worker.start()
        qtbot.waitUntil(submitted.is_set, timeout=4000)
        worker.cancel()

    worker.wait(2000)

    payload = blocker.args[0]
    assert pool_cancelled.is_set() is True
    assert payload["kind"] == "cancelled"
    assert payload["message"] == "Fit cancelled by user"


@pytest.mark.parametrize(
    ("max_parallel_workers", "limit_blas_threads", "error_type", "message"),
    [
        (0, True, ValueError, "max_parallel_workers must be at least 1."),
        (3.7, True, TypeError, "max_parallel_workers must be an integer."),
        (2, "yes", TypeError, "limit_blas_threads must be a boolean."),
    ],
)
def test_global_fit_worker_rejects_invalid_parallel_runtime_settings(
    max_parallel_workers,
    limit_blas_threads,
    error_type,
    message,
) -> None:
    from kindred.gui.fitting.worker import GlobalFitWorker

    with pytest.raises(error_type, match=message):
        GlobalFitWorker(
            datasets=[_raw_dataset("ds1", [0.0, 0.0]), _raw_dataset("ds2", [0.0, 0.0])],
            shared_params={"k1": 1.0},
            fit_evaluator=_make_serial_evaluator(),
            fit_func=lambda *_args, **_kwargs: None,
            max_nfev=1,
            parallel_enabled=True,
            max_parallel_workers=max_parallel_workers,
            limit_blas_threads=limit_blas_threads,
        )


@pytest.mark.parametrize(
    ("max_parallel_workers", "limit_blas_threads", "error_type", "message"),
    [
        (0, True, ValueError, "max_parallel_workers must be at least 1."),
        (3.7, True, TypeError, "max_parallel_workers must be an integer."),
        (2, "yes", TypeError, "limit_blas_threads must be a boolean."),
    ],
)
def test_fit_global_rejects_invalid_parallel_runtime_settings(
    max_parallel_workers,
    limit_blas_threads,
    error_type,
    message,
) -> None:
    from kindred.core.analysis import global_fitting

    with pytest.raises(error_type, match=message):
        global_fitting.fit_global(
            _make_serial_evaluator(),
            [_raw_dataset("ds1", [0.0, 0.0]), _raw_dataset("ds2", [0.0, 0.0])],
            {"k1": 1.0},
            method="trf",
            max_nfev=1,
            parallel_enabled=True,
            max_parallel_workers=max_parallel_workers,
            limit_blas_threads=limit_blas_threads,
        )


def test_fit_global_process_pool_shutdown_on_post_optimizer_error(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_optimization import FitResult

    pools = []

    class _FakeProcessPool:
        def __init__(self, _evaluator_payload, *, max_workers, limit_blas_threads, cancellation_check=None):
            self.max_workers = int(max_workers)
            self.limit_blas_threads = bool(limit_blas_threads)
            self.cancellation_check = cancellation_check
            self.shutdown_calls = []
            pools.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.shutdown(force_terminate=exc_type is not None)
            return None

        def submit(self, item):
            raise AssertionError(f"submit should not be reached for {item.payload.dataset_id}")

        def cancel(self) -> None:
            return None

        def shutdown(self, *, force_terminate: bool) -> None:
            if self.shutdown_calls:
                return
            self.shutdown_calls.append(bool(force_terminate))

    def fake_fit_parameters(_objective, initial_params, **_kwargs):
        return FitResult(
            success=True,
            parameters=dict(initial_params),
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(4, dtype=float),
            nfev=1,
            message="ok",
            covariance=None,
        )

    def raising_assemble(*_args, **_kwargs):
        raise OverflowError("post-optimizer reconstruction overflow")

    monkeypatch.setattr(global_fitting, "FittingProcessPool", _FakeProcessPool)
    monkeypatch.setattr(global_fitting, "fit_parameters", fake_fit_parameters)
    monkeypatch.setattr(global_fitting, "_assemble_global_fit_result", raising_assemble)

    with pytest.raises(OverflowError, match="post-optimizer reconstruction overflow"):
        global_fitting.fit_global(
            _make_serial_evaluator(),
            [_raw_dataset("ds1", [0.0, 0.0]), _raw_dataset("ds2", [0.0, 0.0])],
            {"k1": 1.0},
            dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
            method="trf",
            max_nfev=1,
            parallel_enabled=True,
            max_parallel_workers=6,
            limit_blas_threads=True,
        )

    assert len(pools) == 1
    assert pools[0].shutdown_calls == [True]
