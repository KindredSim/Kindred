from __future__ import annotations

import json
import multiprocessing
import os
import subprocess  # nosec B404 - tests invoke the local interpreter with controlled args
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.test_containment_kernel import _ACCEPT_TIMEOUT_S, _READY_TIMEOUT_S, _require_spawn_primitive_support

pytestmark = pytest.mark.unit

_EXPECTED_CONTAINED_CHILD_BLAS_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class _BatchContainmentTestHandler:
    def __init__(self, startup_payload: Mapping[str, Any]) -> None:
        payload = dict(startup_payload or {})
        startup_delay = float(payload.get("startup_delay_s") or 0.0)
        if startup_delay > 0.0:
            time.sleep(startup_delay)

    def before_accept(self, payload: Mapping[str, Any], _context: Any) -> None:
        behavior = str(dict(payload or {}).get("behavior") or "echo")
        if behavior == "hang_before_accept":
            while True:
                time.sleep(0.05)

    def handle_request(self, payload: Mapping[str, Any], context: Any) -> dict[str, Any]:
        request = dict(payload or {})
        behavior = str(request.get("behavior") or "echo")
        if behavior == "stale_then_result":
            context.output_queue.put(
                {
                    "kind": "result",
                    "owner_epoch": int(context.owner_epoch) + 1,
                    "request_id": int(context.request_id),
                    "run_id": int(request.get("run_id") or 0),
                    "set_id": str(request.get("set_id") or ""),
                    "payload": {"stale": "owner_epoch"},
                }
            )
            context.output_queue.put(
                {
                    "kind": "result",
                    "owner_epoch": int(context.owner_epoch),
                    "request_id": int(context.request_id) + 1,
                    "run_id": int(request.get("run_id") or 0),
                    "set_id": str(request.get("set_id") or ""),
                    "payload": {"stale": "request_id"},
                }
            )
            context.output_queue.put(
                {
                    "kind": "result",
                    "owner_epoch": int(context.owner_epoch),
                    "request_id": int(context.request_id),
                    "run_id": int(request.get("run_id") or 0) + 1,
                    "set_id": str(request.get("set_id") or ""),
                    "payload": {"stale": "run_id"},
                }
            )
            context.output_queue.put(
                {
                    "kind": "result",
                    "owner_epoch": int(context.owner_epoch),
                    "request_id": int(context.request_id),
                    "run_id": int(request.get("run_id") or 0),
                    "set_id": f"{request.get('set_id')}-stale",
                    "payload": {"stale": "set_id"},
                }
            )
        if behavior == "hang_after_accept":
            while True:
                time.sleep(0.05)
        if behavior == "env_probe":
            numpy_preimported = "numpy" in sys.modules
            env_before_numpy = {name: os.environ.get(name) for name in _EXPECTED_CONTAINED_CHILD_BLAS_ENV}
            import numpy  # noqa: F401

            return {
                "success": True,
                "numpy_preimported": numpy_preimported,
                "env_before_numpy": env_before_numpy,
                "run_id": int(request.get("run_id") or 0),
                "set_id": str(request.get("set_id") or ""),
            }
        return {
            "success": True,
            "echo": request,
            "run_id": int(request.get("run_id") or 0),
            "set_id": str(request.get("set_id") or ""),
            "owner_epoch": int(context.owner_epoch),
            "request_id": int(context.request_id),
        }


def make_batch_containment_test_handler(startup_payload: Mapping[str, Any]) -> _BatchContainmentTestHandler:
    return _BatchContainmentTestHandler(startup_payload)


def _mp_context() -> multiprocessing.context.BaseContext:
    return _require_spawn_primitive_support()


def test_batch_containment_import_is_stdlib_lazy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import json
import sys
import kindred.core.batch_containment
print(json.dumps({
    "numpy": "numpy" in sys.modules,
    "scipy": "scipy" in sys.modules,
    "pyside": "PySide6" in sys.modules,
    "batch_parallel": "kindred.core.batch_parallel" in sys.modules,
    "simulation_containment": "kindred.core.simulation_containment" in sys.modules,
    "fitting_containment": "kindred.core.fitting_containment" in sys.modules,
}))
"""
    result = subprocess.run(  # nosec B603 - test invokes local Python only
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout.strip()) == {
        "numpy": False,
        "scipy": False,
        "pyside": False,
        "batch_parallel": False,
        "simulation_containment": False,
        "fitting_containment": False,
    }


def test_default_batch_handler_prewarms_backend_during_startup(monkeypatch) -> None:
    from kindred.core import batch_parallel
    from kindred.core.batch_containment import make_batch_simulation_handler

    calls: list[str] = []

    def _prewarm_worker_imports() -> bool:
        calls.append("prewarm")
        return True

    def _run_batch_simulation_task(payload: Mapping[str, Any]) -> dict[str, Any]:
        calls.append("run")
        return {"success": True, "payload": dict(payload)}

    monkeypatch.setattr(batch_parallel, "prewarm_worker_imports", _prewarm_worker_imports)
    monkeypatch.setattr(batch_parallel, "run_batch_simulation_task", _run_batch_simulation_task)

    handler = make_batch_simulation_handler({})

    assert calls == ["prewarm"]
    assert handler.handle_request({"set_id": "set-a"}, None) == {
        "success": True,
        "payload": {"set_id": "set-a"},
    }
    assert calls == ["prewarm", "run"]


def test_warm_batch_lane_rejects_stale_owner_request_run_and_set_replies() -> None:
    from kindred.core.batch_containment import WarmBatchSimulationLane

    lane = WarmBatchSimulationLane(
        lane_id="lane-1",
        handler_import_path="tests.test_batch_containment:make_batch_containment_test_handler",
        mp_context=_mp_context(),
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
    )
    try:
        outcome = lane.run(
            {"behavior": "stale_then_result", "run_id": 9, "set_id": "set-9"},
            run_id=9,
            request_id=44,
            set_id="set-9",
            active_timeout_s=1.0,
        )
        assert outcome.success is True
        assert outcome.payload["echo"]["behavior"] == "stale_then_result"
        assert outcome.run_id == 9
        assert outcome.request_id == 44
        assert outcome.set_id == "set-9"
        assert outcome.owner_epoch == 1
        assert "stale_ignored" in [event.kind for event in outcome.events]
    finally:
        lane.close(kill=True)


def test_warm_batch_lane_maps_active_timeout_after_accept_to_set_failure() -> None:
    from kindred.core.batch_containment import WarmBatchSimulationLane

    lane = WarmBatchSimulationLane(
        lane_id="lane-timeout",
        handler_import_path="tests.test_batch_containment:make_batch_containment_test_handler",
        mp_context=_mp_context(),
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
    )
    try:
        outcome = lane.run(
            {"behavior": "hang_after_accept", "run_id": 12, "set_id": "slow"},
            run_id=12,
            request_id=4,
            set_id="slow",
            active_timeout_s=0.05,
        )
        assert outcome.success is False
        assert outcome.run_id == 12
        assert outcome.request_id == 4
        assert outcome.set_id == "slow"
        assert outcome.failure["kind"] == "active_timeout"
        assert outcome.failure["phase"] == "active"
        assert outcome.owner_epoch == 1
        assert lane.owner_epoch == 1
        assert lane.is_running is False
    finally:
        lane.close(kill=True)


def test_warm_batch_lane_maps_accept_timeout_before_active_phase() -> None:
    from kindred.core.batch_containment import WarmBatchSimulationLane

    lane = WarmBatchSimulationLane(
        lane_id="lane-accept-timeout",
        handler_import_path="tests.test_batch_containment:make_batch_containment_test_handler",
        mp_context=_mp_context(),
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=0.05,
    )
    try:
        outcome = lane.run(
            {"behavior": "hang_before_accept", "run_id": 13, "set_id": "accept"},
            run_id=13,
            request_id=5,
            set_id="accept",
            active_timeout_s=1.0,
        )
        assert outcome.success is False
        assert outcome.failure["kind"] == "accept_timeout"
        assert outcome.failure["phase"] == "accept"
        assert outcome.set_id == "accept"
    finally:
        lane.close(kill=True)


def test_warm_batch_lane_maps_startup_timeout_to_set_failure() -> None:
    from kindred.core.batch_containment import WarmBatchSimulationLane

    lane = WarmBatchSimulationLane(
        lane_id="lane-startup-timeout",
        handler_import_path="tests.test_batch_containment:make_batch_containment_test_handler",
        startup_payload={"startup_delay_s": 1.0},
        mp_context=_mp_context(),
        ready_timeout_s=0.05,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
    )
    try:
        outcome = lane.run(
            {"behavior": "echo", "run_id": 14, "set_id": "startup"},
            run_id=14,
            request_id=6,
            set_id="startup",
            active_timeout_s=1.0,
        )
        assert outcome.success is False
        assert outcome.failure["kind"] == "startup_timeout"
        assert outcome.failure["phase"] == "startup"
        assert outcome.set_id == "startup"
    finally:
        lane.close(kill=True)


def test_batch_lane_sets_blas_env_before_child_imports_numpy() -> None:
    from kindred.core.batch_containment import WarmBatchSimulationLane

    lane = WarmBatchSimulationLane(
        lane_id="lane-env",
        handler_import_path="tests.test_batch_containment:make_batch_containment_test_handler",
        limit_blas_threads_per_worker=True,
        mp_context=_mp_context(),
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
    )
    try:
        outcome = lane.run(
            {"behavior": "env_probe", "run_id": 15, "set_id": "env"},
            run_id=15,
            request_id=7,
            set_id="env",
            active_timeout_s=1.0,
        )
        assert outcome.success is True
        assert outcome.payload["numpy_preimported"] is False
        assert outcome.payload["env_before_numpy"] == _EXPECTED_CONTAINED_CHILD_BLAS_ENV
    finally:
        lane.close(kill=True)


def test_batch_lane_can_disable_blas_env_caps() -> None:
    from kindred.core.batch_containment import WarmBatchSimulationLane

    lane = WarmBatchSimulationLane(
        lane_id="lane-env-disabled",
        handler_import_path="tests.test_batch_containment:make_batch_containment_test_handler",
        limit_blas_threads_per_worker=False,
        mp_context=_mp_context(),
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
    )
    try:
        assert lane._owner._kernel_owner._handler_spec.env == {}
    finally:
        lane.close(kill=True)


def test_batch_lane_pool_bounds_retained_lanes_and_event_history() -> None:
    from kindred.core.batch_containment import BatchLanePool

    pool = BatchLanePool(
        max_lanes=2,
        handler_import_path="tests.test_batch_containment:make_batch_containment_test_handler",
        mp_context=_mp_context(),
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
        event_history_limit=3,
    )
    try:
        for index in range(5):
            outcome = pool.run(
                {"behavior": "echo", "run_id": 20, "set_id": f"set-{index}"},
                run_id=20,
                request_id=100 + index,
                set_id=f"set-{index}",
                active_timeout_s=1.0,
            )
            assert outcome.success is True

        assert pool.retained_lane_count <= 2
        assert len(pool.diagnostic_events()) <= 3
    finally:
        pool.close(kill=True)


def test_batch_lane_pool_warm_lanes_creates_ready_child_lanes() -> None:
    from kindred.core.batch_containment import BatchLanePool

    pool = BatchLanePool(
        max_lanes=2,
        handler_import_path="tests.test_batch_containment:make_batch_containment_test_handler",
        mp_context=_mp_context(),
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
    )
    try:
        pool.warm_lanes(2, wait=True)

        assert pool.retained_lane_count == 2
        ready_events = [event for event in pool.diagnostic_events() if event.kind == "owner_ready"]
        assert len(ready_events) == 2
    finally:
        pool.close(kill=True)


def test_batch_lane_pool_reuses_warmed_idle_lane_before_growing_pool(monkeypatch) -> None:
    from kindred.core import batch_containment
    from kindred.core.batch_containment import BatchLaneOutcome

    class _FakeLane:
        def __init__(self, *, lane_id: str, **_kwargs: object) -> None:
            self.lane_id = str(lane_id)
            self._lock = threading.Lock()

        @property
        def is_busy(self) -> bool:
            return self._lock.locked()

        def reserve(self) -> bool:
            return bool(self._lock.acquire(blocking=False))

        def release_reservation(self) -> None:
            self._lock.release()

        def warm(self, *, wait: bool = True) -> None:
            _ = wait

        def drain_events(self) -> tuple[object, ...]:
            return ()

        def close(self, *, kill: bool = False) -> None:
            _ = kill

        def run_reserved(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            try:
                return BatchLaneOutcome(
                    lane_id=self.lane_id,
                    run_id=run_id,
                    request_id=request_id,
                    set_id=set_id,
                    owner_epoch=1,
                    success=True,
                    payload={"success": True, "task": dict(task or {})},
                )
            finally:
                self.release_reservation()

        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            return self.run_reserved(
                task,
                run_id=run_id,
                request_id=request_id,
                set_id=set_id,
                active_timeout_s=active_timeout_s,
            )

    monkeypatch.setattr(batch_containment, "WarmBatchSimulationLane", _FakeLane)
    pool = batch_containment.BatchLanePool(max_lanes=2)
    try:
        pool.warm_lanes(1)
        outcome = pool.run({"set_id": "set-a"}, run_id=1, request_id=1, set_id="set-a")

        assert outcome.lane_id == "batch-lane-1"
        assert pool.retained_lane_count == 1
    finally:
        pool.close(kill=True)


def test_batch_lane_pool_concurrent_runs_grow_instead_of_waiting_on_reserved_lane(monkeypatch) -> None:
    from kindred.core import batch_containment
    from kindred.core.batch_containment import BatchLaneOutcome

    started: list[str] = []
    started_lock = threading.Lock()
    first_started = threading.Event()
    release_first = threading.Event()

    class _FakeLane:
        def __init__(self, *, lane_id: str, **_kwargs: object) -> None:
            self.lane_id = str(lane_id)
            self._lock = threading.Lock()

        @property
        def is_busy(self) -> bool:
            return self._lock.locked()

        def reserve(self) -> bool:
            return bool(self._lock.acquire(blocking=False))

        def release_reservation(self) -> None:
            self._lock.release()

        def warm(self, *, wait: bool = True) -> None:
            _ = wait

        def drain_events(self) -> tuple[object, ...]:
            return ()

        def close(self, *, kill: bool = False) -> None:
            _ = kill

        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            with self._lock:
                return self._finish(task, run_id=run_id, request_id=request_id, set_id=set_id)

        def run_reserved(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            try:
                return self._finish(task, run_id=run_id, request_id=request_id, set_id=set_id)
            finally:
                self.release_reservation()

        def _finish(self, task, *, run_id: int, request_id: int, set_id: str):
            with started_lock:
                started.append(self.lane_id)
            if self.lane_id == "batch-lane-1":
                first_started.set()
                assert release_first.wait(timeout=1.0)
            return BatchLaneOutcome(
                lane_id=self.lane_id,
                run_id=run_id,
                request_id=request_id,
                set_id=set_id,
                owner_epoch=1,
                success=True,
                payload={"success": True, "task": dict(task or {})},
            )

    monkeypatch.setattr(batch_containment, "WarmBatchSimulationLane", _FakeLane)
    pool = batch_containment.BatchLanePool(max_lanes=2)
    pool.warm_lanes(1)
    outcomes: list[BatchLaneOutcome] = []

    def _run(set_id: str) -> None:
        outcomes.append(
            pool.run(
                {"set_id": set_id},
                run_id=31,
                request_id=1,
                set_id=set_id,
                active_timeout_s=1.0,
            )
        )

    first = threading.Thread(target=_run, args=("set-a",))
    first.start()
    assert first_started.wait(timeout=1.0)
    second = threading.Thread(target=_run, args=("set-b",))
    second.start()
    release_first.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert sorted(outcome.lane_id for outcome in outcomes) == ["batch-lane-1", "batch-lane-2"]
    assert started[:2] == ["batch-lane-1", "batch-lane-2"]
    assert pool.retained_lane_count == 2


def test_warm_batch_lane_force_close_reaches_owner_during_active_run(monkeypatch) -> None:
    from kindred.core import batch_containment

    solve_started = threading.Event()
    release_solve = threading.Event()
    close_called = threading.Event()

    class _BlockingRuntimeOwner:
        owner_epoch = 1

        def __init__(self, **_kwargs: object) -> None:
            return None

        @property
        def is_running(self) -> bool:
            return not close_called.is_set()

        def drain_events(self) -> tuple[object, ...]:
            return ()

        def close(self, *, kill: bool = False) -> None:
            assert kill is True
            close_called.set()

        def solve(self, payload, *, active_timeout_s: float, reply_fields):
            _ = active_timeout_s, reply_fields
            solve_started.set()
            assert release_solve.wait(timeout=1.0)
            return {
                "success": True,
                "run_id": int(payload["run_id"]),
                "set_id": str(payload["set_id"]),
                "request_id": int(payload["request_id"]),
            }

    monkeypatch.setattr(batch_containment, "SimulationRuntimeOwner", _BlockingRuntimeOwner)
    lane = batch_containment.WarmBatchSimulationLane(lane_id="lane-force-close")
    outcomes: list[batch_containment.BatchLaneOutcome] = []

    run_thread = threading.Thread(
        target=lambda: outcomes.append(
            lane.run(
                {"run_id": 1, "request_id": 2, "set_id": "set-a"},
                run_id=1,
                request_id=2,
                set_id="set-a",
                active_timeout_s=1.0,
            )
        )
    )
    run_thread.start()
    assert solve_started.wait(timeout=1.0)

    close_thread = threading.Thread(target=lambda: lane.close(kill=True))
    close_thread.start()
    close_reached_owner_while_run_active = close_called.wait(timeout=0.2)

    release_solve.set()
    run_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert close_reached_owner_while_run_active is True
    assert not run_thread.is_alive()
    assert not close_thread.is_alive()
    assert outcomes and outcomes[0].success is True


def test_warm_batch_lane_graceful_close_kills_busy_owner_without_waiting(monkeypatch) -> None:
    from kindred.core import batch_containment

    solve_started = threading.Event()
    release_solve = threading.Event()
    close_returned = threading.Event()
    close_calls: list[bool] = []

    class _BlockingRuntimeOwner:
        owner_epoch = 1

        def __init__(self, **_kwargs: object) -> None:
            return None

        @property
        def is_running(self) -> bool:
            return True

        def drain_events(self) -> tuple[object, ...]:
            return ()

        def close(self, *, kill: bool = False) -> None:
            close_calls.append(bool(kill))

        def solve(self, payload, *, active_timeout_s: float, reply_fields):
            _ = payload, active_timeout_s, reply_fields
            solve_started.set()
            assert release_solve.wait(timeout=1.0)
            return {"success": True, "run_id": 1, "set_id": "set-a", "request_id": 2}

    monkeypatch.setattr(batch_containment, "SimulationRuntimeOwner", _BlockingRuntimeOwner)
    lane = batch_containment.WarmBatchSimulationLane(lane_id="lane-graceful-close")
    run_thread = threading.Thread(
        target=lambda: lane.run(
            {"run_id": 1, "request_id": 2, "set_id": "set-a"},
            run_id=1,
            request_id=2,
            set_id="set-a",
            active_timeout_s=1.0,
        )
    )
    run_thread.start()
    assert solve_started.wait(timeout=1.0)

    close_thread = threading.Thread(target=lambda: (lane.close(kill=False), close_returned.set()))
    close_thread.start()

    assert close_returned.wait(timeout=0.2)
    assert close_calls == [True]

    release_solve.set()
    run_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)
    lane.close(kill=True)


def test_warm_batch_lane_close_kill_restart_is_idempotent() -> None:
    from kindred.core.batch_containment import WarmBatchSimulationLane

    lane = WarmBatchSimulationLane(
        lane_id="lane-restart",
        handler_import_path="tests.test_batch_containment:make_batch_containment_test_handler",
        mp_context=_mp_context(),
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
    )
    try:
        first = lane.run(
            {"behavior": "echo", "run_id": 21, "set_id": "set-a"},
            run_id=21,
            request_id=1,
            set_id="set-a",
            active_timeout_s=1.0,
        )
        assert first.success is True
        first_epoch = lane.owner_epoch

        lane.close(kill=True)
        lane.close(kill=True)

        second = lane.run(
            {"behavior": "echo", "run_id": 22, "set_id": "set-b"},
            run_id=22,
            request_id=2,
            set_id="set-b",
            active_timeout_s=1.0,
        )
        assert second.success is True
        assert lane.owner_epoch > first_epoch
    finally:
        lane.close(kill=True)
