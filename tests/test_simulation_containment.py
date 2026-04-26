from __future__ import annotations

import multiprocessing
import queue
import time
from typing import Any, Callable, Mapping, MutableMapping, Optional

import numpy as np
import pytest

from kindred.core.simulation_failure import build_simulation_failure, simulation_failure_from_exception

pytestmark = pytest.mark.unit


def _process_context() -> multiprocessing.context.BaseContext:
    return multiprocessing.get_context("spawn")


def _require_spawn_queue_support() -> multiprocessing.context.BaseContext:
    mp_context = _process_context()
    try:
        probe_queue: multiprocessing.Queue = mp_context.Queue(maxsize=1)
    except (OSError, PermissionError) as exc:
        pytest.skip(f"multiprocessing spawn Queue unavailable in this environment: {exc}")
    else:
        probe_queue.close()
        probe_queue.join_thread()
    return mp_context


def _contained_call_child(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    output_queue: multiprocessing.Queue,
) -> None:
    try:
        output_queue.put({"success": True, "result": target(*args, **dict(kwargs))})
    except BaseException as exc:  # noqa: BLE001 - child boundary must serialize every failure
        output_queue.put({"success": False, "error": simulation_failure_from_exception(exc)})


def _terminate_process(proc: multiprocessing.Process) -> None:
    if not proc.is_alive():
        return
    proc.terminate()
    proc.join(timeout=0.5)
    if proc.is_alive() and hasattr(proc, "kill"):
        proc.kill()
        proc.join(timeout=0.5)


def _timeout_failure(*, walltime_s: float) -> dict[str, Any]:
    return build_simulation_failure(
        "timeout",
        f"Simulation timed out after {float(walltime_s):.3g} seconds.",
        code="E306",
        details={"walltime_s": float(walltime_s)},
        exc_type="SimulationTimeoutError",
    )


def _cancelled_failure() -> dict[str, Any]:
    return build_simulation_failure(
        "cancelled",
        "Simulation cancelled by user",
        code="E305",
        exc_type="SimulationCancelled",
    )


def _execute_contained_call(
    target: Callable[..., Any],
    *args: Any,
    walltime_s: float,
    cancel_check: Optional[Callable[[], bool]] = None,
    poll_interval_s: float = 0.02,
    **kwargs: Any,
) -> dict[str, Any]:
    walltime = max(0.001, float(walltime_s))
    poll_interval = max(0.001, min(float(poll_interval_s), 0.1))
    mp_context = _require_spawn_queue_support()
    output_queue: multiprocessing.Queue = mp_context.Queue(maxsize=1)
    proc = mp_context.Process(
        target=_contained_call_child,
        args=(target, tuple(args), dict(kwargs), output_queue),
    )
    proc.start()
    deadline = time.monotonic() + walltime

    try:
        while True:
            if cancel_check is not None and bool(cancel_check()):
                _terminate_process(proc)
                return {"success": False, "error": _cancelled_failure()}
            try:
                outcome = output_queue.get(timeout=poll_interval)
            except queue.Empty:
                outcome = None
            if isinstance(outcome, MutableMapping):
                proc.join(timeout=0.2)
                return dict(outcome)
            if not proc.is_alive():
                proc.join(timeout=0.2)
                try:
                    outcome = output_queue.get_nowait()
                except queue.Empty:
                    return {
                        "success": False,
                        "error": build_simulation_failure(
                            "simulation_error",
                            "Contained simulation child exited without returning a result.",
                            details={"exitcode": proc.exitcode},
                        ),
                    }
                return dict(outcome)
            if time.monotonic() >= deadline:
                _terminate_process(proc)
                return {"success": False, "error": _timeout_failure(walltime_s=walltime)}
    finally:
        if proc.is_alive():
            _terminate_process(proc)
        output_queue.close()


def _never_returning_rhs(_t, y):
    while True:
        time.sleep(0.05)
    return np.asarray(y, dtype=float)


def _run_never_returning_solver():
    from kindred.core.simulator.solvers import SimulationRequest, solve_ode

    return solve_ode(
        SimulationRequest(
            rhs=_never_returning_rhs,
            t_span=(0.0, 1.0),
            y0=np.asarray([1.0], dtype=float),
            solver="BDF",
            grid={"N": 3},
        )
    )


def _sleep_until_killed():
    while True:
        time.sleep(0.05)


def test_spawned_solver_child_times_out_and_returns_structured_failure():
    outcome = _execute_contained_call(_run_never_returning_solver, walltime_s=0.2)

    assert outcome["success"] is False
    assert outcome["error"]["kind"] == "timeout"
    assert "timed out" in outcome["error"]["message"].lower()
    assert outcome["error"]["details"]["walltime_s"] == pytest.approx(0.2)


def test_spawned_child_cancellation_terminates_before_timeout():
    started = time.monotonic()

    def _cancel_after_start() -> bool:
        return (time.monotonic() - started) >= 0.1

    outcome = _execute_contained_call(
        _sleep_until_killed,
        walltime_s=5.0,
        cancel_check=_cancel_after_start,
    )

    assert outcome["success"] is False
    assert outcome["error"]["kind"] == "cancelled"


def test_containment_helper_uses_spawn_context():
    assert _process_context().get_start_method() == "spawn"
