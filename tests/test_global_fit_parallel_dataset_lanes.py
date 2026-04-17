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


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


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
    from kindred.core.fitting_process_pool import FittingProcessPool

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
    pool._shutdown_in_progress = False
    pool._startup_cancelled = False
    pool._state_lock = threading.Lock()

    pool.shutdown(force_terminate=False)

    assert calls == [
        "event.set",
        ("executor.shutdown", True, True),
        "manager.shutdown",
    ]


def test_fitting_process_pool_shutdown_cleans_partial_construction_even_when_closed() -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

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
    pool._closed = True
    pool._prewarm_in_progress = False
    pool._startup_cancelled = False
    pool._shutdown_in_progress = False
    pool._state_lock = threading.Lock()

    pool.shutdown(force_terminate=True)

    assert calls == [
        "event.set",
        ("executor.shutdown", False, True),
        "manager.shutdown",
    ]


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
        lambda self, processes: calls.append(("terminate_target_len", len(tuple(processes)))),
    )

    pool.cancel()

    assert calls == [
        "event.set",
        ("terminate_target_len", 1),
    ]
    assert pool._startup_cancelled is True


def test_fitting_process_pool_cancel_ignores_process_snapshot_race_during_prewarm(monkeypatch) -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    calls = []

    class _MutatingProcesses(dict):
        def values(self):
            raise RuntimeError("dictionary changed size during iteration")

    class _FakeEvent:
        def set(self):
            calls.append("event.set")

    class _FakeExecutor:
        def __init__(self):
            self._processes = _MutatingProcesses({1: object()})

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
        lambda self, processes: calls.append(("terminate_target_len", len(tuple(processes)))),
    )

    pool.cancel()

    assert calls == ["event.set"]
    assert pool._startup_cancelled is True


def test_fitting_process_pool_force_shutdown_escalates_while_graceful_shutdown_is_in_progress(monkeypatch) -> None:
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
    pool._startup_cancelled = False
    pool._shutdown_in_progress = True
    pool._state_lock = threading.Lock()

    monkeypatch.setattr(
        FittingProcessPool,
        "_terminate_processes_best_effort",
        lambda self, processes: calls.append(("terminate_target_len", len(tuple(processes)))),
    )

    pool.shutdown(force_terminate=True)

    assert calls == [
        "event.set",
        ("terminate_target_len", 1),
    ]


@pytest.mark.parametrize("shutdown_in_progress", [False, True])
def test_fitting_process_pool_force_shutdown_ignores_process_snapshot_race(
    monkeypatch,
    shutdown_in_progress: bool,
) -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    calls = []

    class _MutatingProcesses(dict):
        def values(self):
            raise RuntimeError("dictionary changed size during iteration")

    class _FakeEvent:
        def set(self):
            calls.append("event.set")

    class _FakeExecutor:
        def __init__(self):
            self._processes = _MutatingProcesses({1: object()})

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
    pool._startup_cancelled = False
    pool._shutdown_in_progress = shutdown_in_progress
    pool._state_lock = threading.Lock()

    monkeypatch.setattr(
        FittingProcessPool,
        "_terminate_processes_best_effort",
        lambda self, processes: calls.append(("terminate_target_len", len(tuple(processes)))),
    )

    pool.shutdown(force_terminate=True)

    if shutdown_in_progress:
        assert calls == ["event.set"]
    else:
        assert calls == [
            "event.set",
            ("executor.shutdown", False, True),
            "manager.shutdown",
        ]


@pytest.mark.skipif(not CAN_CREATE_PROCESS_POOL, reason=PROCESS_POOL_SKIP_REASON)
def test_fitting_process_pool_publishes_handle_before_prewarm(monkeypatch) -> None:
    from kindred.core.fitting_process_pool import FittingProcessPool

    published = {}
    original_prewarm = FittingProcessPool._prewarm

    def tracking_prewarm(self):
        assert published["pool"] is self
        return original_prewarm(self)

    monkeypatch.setattr(FittingProcessPool, "_prewarm", tracking_prewarm)

    pool = FittingProcessPool(
        _make_serial_evaluator().to_process_payload(),
        max_workers=1,
        publish_callback=lambda process_pool: published.setdefault("pool", process_pool),
    )

    try:
        assert published["pool"] is pool
    finally:
        pool.shutdown(force_terminate=False)


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
        def __init__(self, evaluator_payload, *, max_workers, limit_blas_threads, publish_callback=None):
            self._evaluator = SerialFittingEvaluator.from_process_payload(evaluator_payload)
            self.max_workers = int(max_workers)
            self.limit_blas_threads = bool(limit_blas_threads)
            self.submit_count = 0
            self.shutdown_calls = []
            pools.append(self)
            if publish_callback is not None:
                publish_callback(self)

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
    )

    assert result.success is True
    assert len(pools) == 1
    assert pools[0].max_workers == 2
    assert pools[0].limit_blas_threads is False
    assert pools[0].submit_count == 2
    assert pools[0].shutdown_calls == [False]


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

    callback_values = []
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
        process_pool_callback=lambda pool: callback_values.append(pool),
    )

    assert result.success is True
    assert callback_values == [None]
    assert "pickl" in caplog.text.lower()


def test_fit_global_ignores_cleanup_process_pool_callback_error(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_evaluation import SerialFittingEvaluator
    from kindred.core.fitting_optimization import FitResult

    pools = []
    callback_values = []

    class _FakeProcessPool:
        def __init__(self, evaluator_payload, *, max_workers, limit_blas_threads, publish_callback=None):
            self._evaluator = SerialFittingEvaluator.from_process_payload(evaluator_payload)
            self.max_workers = int(max_workers)
            self.limit_blas_threads = bool(limit_blas_threads)
            self.shutdown_calls = []
            pools.append(self)
            if publish_callback is not None:
                publish_callback(self)

        def submit(self, item):
            from kindred.core.fitting_evaluation import evaluate_fitting_series

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

    def process_pool_callback(pool):
        callback_values.append(pool)
        if pool is None:
            raise RuntimeError("cleanup callback failed")

    monkeypatch.setattr(global_fitting, "FittingProcessPool", _FakeProcessPool)
    monkeypatch.setattr(global_fitting, "fit_parameters", fake_fit_parameters)

    result = global_fitting.fit_global(
        _make_serial_evaluator(),
        [_raw_dataset("ds1", [0.0, 0.0]), _raw_dataset("ds2", [0.0, 0.0])],
        {"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}, "ds2": {"init:A": 10.0}},
        method="trf",
        max_nfev=1,
        parallel_enabled=True,
        max_parallel_workers=4,
        limit_blas_threads=True,
        process_pool_callback=process_pool_callback,
    )

    assert result.success is True
    assert callback_values == [pools[0], None]
    assert pools[0].shutdown_calls == [False]


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


def test_global_fit_worker_retro_cancels_process_pool_when_handle_arrives_late() -> None:
    from kindred.gui.fitting.worker import GlobalFitWorker

    worker = GlobalFitWorker(
        datasets=[_raw_dataset("ds1", [0.0, 0.0])],
        shared_params={"k1": 1.0},
        fit_evaluator=lambda _params: {"t": np.asarray([0.0, 1.0]), "species": {"A": np.asarray([0.0, 0.0])}},
        fit_func=lambda *_args, **_kwargs: None,
        max_nfev=1,
    )

    class _FakeProcessPool:
        def __init__(self):
            self.cancel_calls = 0

        def cancel(self):
            self.cancel_calls += 1

    pool = _FakeProcessPool()
    worker.cancel()
    worker._set_active_process_pool(pool)

    assert pool.cancel_calls == 1


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
        def __init__(self, _evaluator_payload, *, max_workers, limit_blas_threads, publish_callback=None):
            self.max_workers = int(max_workers)
            self.limit_blas_threads = bool(limit_blas_threads)
            self.shutdown_calls = []
            pools.append(self)
            if publish_callback is not None:
                publish_callback(self)

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
