from __future__ import annotations

import multiprocessing
import os
import queue
import time
from typing import Any, Callable, Mapping, MutableMapping, Optional

import numpy as np
import pytest

from kindred.core.simulation_failure import build_simulation_failure, simulation_failure_from_exception

pytestmark = pytest.mark.unit

_OWNER_TEST_READY_TIMEOUT_S = 2.0
_OWNER_TEST_ACCEPT_TIMEOUT_S = 2.0
_EXPECTED_CONTAINED_CHILD_BLAS_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _process_context() -> multiprocessing.context.BaseContext:
    return multiprocessing.get_context("spawn")


def _spawn_probe_child(output_queue: multiprocessing.Queue) -> None:
    output_queue.put({"ok": True})


def _require_spawn_primitive_support() -> multiprocessing.context.BaseContext:
    mp_context = _process_context()
    probe_queue = None
    probe_sem = None
    try:
        probe_queue = mp_context.Queue(maxsize=1)
        probe_sem = mp_context.Semaphore(1)
        proc = mp_context.Process(target=_spawn_probe_child, args=(probe_queue,))
        proc.start()
        proc.join(timeout=1.0)
        if proc.is_alive():
            _terminate_process(proc)
            pytest.skip("multiprocessing spawn Process did not join in this environment")
        if proc.exitcode != 0:
            pytest.skip(f"multiprocessing spawn Process failed in this environment: exitcode={proc.exitcode}")
        try:
            probe = probe_queue.get_nowait()
        except queue.Empty:
            pytest.skip("multiprocessing spawn Process did not return probe output")
        if probe != {"ok": True}:
            pytest.skip(f"multiprocessing spawn Process returned unexpected probe output: {probe!r}")
    except (OSError, PermissionError) as exc:
        pytest.skip(f"multiprocessing spawn primitives unavailable in this environment: {exc}")
    finally:
        if probe_queue is not None:
            probe_queue.close()
            probe_queue.join_thread()
        del probe_sem
    return mp_context


def _require_spawn_queue_support() -> multiprocessing.context.BaseContext:
    return _require_spawn_primitive_support()


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


def _warm_owner_test_child(
    startup_payload: Mapping[str, Any],
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    owner_epoch: int,
) -> None:
    startup_count_path = str(startup_payload.get("startup_count_path") or "")
    if startup_count_path:
        with open(startup_count_path, "a", encoding="utf-8") as handle:
            handle.write(f"{int(owner_epoch)}\n")

    ready_delay = float(startup_payload.get("ready_delay_s") or 0.0)
    if ready_delay > 0.0:
        time.sleep(ready_delay)

    output_queue.put({"kind": "ready", "owner_epoch": int(owner_epoch)})
    while True:
        message = input_queue.get()
        if not isinstance(message, Mapping):
            continue
        kind = str(message.get("kind") or "")
        if kind == "close":
            return
        if kind != "solve":
            continue
        request_id = int(message.get("request_id", -1))
        payload = dict(message.get("payload") or {})
        behavior = str(payload.get("behavior") or "echo")
        if behavior == "env_echo":
            env_names = [str(name) for name in (payload.get("env_names") or [])]
            output_queue.put(
                {
                    "kind": "accepted",
                    "owner_epoch": int(owner_epoch),
                    "request_id": request_id,
                }
            )
            output_queue.put(
                {
                    "kind": "result",
                    "owner_epoch": int(owner_epoch),
                    "request_id": request_id,
                    "payload": {"env": {name: os.environ.get(name) for name in env_names}},
                }
            )
            continue
        if behavior == "stale_then_result":
            output_queue.put(
                {
                    "kind": "result",
                    "owner_epoch": int(owner_epoch) - 1,
                    "request_id": request_id,
                    "payload": {"stale": True},
                }
            )
        if behavior == "fatal_before_accept":
            time.sleep(float(payload.get("fatal_delay_s") or 0.0))
            output_queue.put(
                {
                    "kind": "fatal",
                    "owner_epoch": int(owner_epoch),
                    "request_id": request_id,
                    "failure": build_simulation_failure(
                        "simulation_containment_reconstruction",
                        "Reconstruction failed before acceptance.",
                    ),
                }
            )
            continue
        output_queue.put(
            {
                "kind": "accepted",
                "owner_epoch": int(owner_epoch),
                "request_id": request_id,
            }
        )
        if behavior == "hang_after_accept":
            while True:
                time.sleep(0.05)
        output_queue.put(
            {
                "kind": "result",
                "owner_epoch": int(owner_epoch),
                "request_id": request_id,
                "payload": {
                    "success": True,
                    "echo": payload,
                    "owner_epoch": int(owner_epoch),
                    "request_id": request_id,
                },
            }
        )


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


def test_spawn_capability_guard_probes_queue_semaphore_and_process_start_join():
    mp_context = _require_spawn_primitive_support()

    assert mp_context.get_start_method() == "spawn"


def test_warm_simulation_owner_sets_blas_env_for_runtime_child():
    from kindred.core.simulation_containment import WarmSimulationOwner

    owner = WarmSimulationOwner({})

    try:
        runtime_owner = owner._runtime_owner
        assert runtime_owner is not None
        assert runtime_owner._kernel_owner._handler_spec.env == _EXPECTED_CONTAINED_CHILD_BLAS_ENV
    finally:
        owner.close(kill=True)


def test_warm_simulation_owner_adapter_maps_runtime_warm_failures(monkeypatch):
    import kindred.core.simulation_containment as containment
    from kindred.core.containment_kernel import (
        ContainmentKernelCancelled,
        ContainmentKernelChildFailure,
        ContainmentKernelProtocolError,
        ContainmentKernelStartupTimeout,
    )
    from kindred.core.exceptions import SimulationCancelled

    class _FakeRuntimeOwner:
        warm_exception: BaseException | None = None

        def __init__(self, **_kwargs):
            self.owner_epoch = 7
            self.is_running = False
            self.is_ready = False
            self.close_calls: list[bool] = []

        def warm(self, **_kwargs):
            if self.warm_exception is not None:
                raise self.warm_exception

        def solve(self, *_args, **_kwargs):
            return {"success": True}

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    monkeypatch.setattr(containment, "SimulationRuntimeOwner", _FakeRuntimeOwner)
    cases = [
        (
            ContainmentKernelStartupTimeout(0.25),
            containment.SimulationContainmentStartupTimeout,
        ),
        (
            ContainmentKernelChildFailure({"kind": "boom", "message": "child failed"}),
            containment.SimulationContainmentChildFailure,
        ),
        (
            ContainmentKernelProtocolError("bad protocol"),
            containment.SimulationContainmentProtocolError,
        ),
        (ContainmentKernelCancelled(), SimulationCancelled),
    ]

    for runtime_exc, expected_exc in cases:
        _FakeRuntimeOwner.warm_exception = runtime_exc
        owner = containment.WarmSimulationOwner({})
        with pytest.raises(expected_exc):
            owner.start(wait=True)


def test_warm_simulation_owner_adapter_maps_runtime_solve_failures(monkeypatch):
    import kindred.core.simulation_containment as containment
    from kindred.core.containment_kernel import (
        ContainmentKernelAcceptTimeout,
        ContainmentKernelActiveTimeout,
        ContainmentKernelChildFailure,
        ContainmentKernelProtocolError,
        ContainmentKernelStartupTimeout,
    )

    class _FakeRuntimeOwner:
        solve_exception: BaseException | None = None

        def __init__(self, **_kwargs):
            self.owner_epoch = 7
            self.is_running = False
            self.is_ready = True

        def warm(self, **_kwargs):
            return None

        def solve(self, *_args, **_kwargs):
            if self.solve_exception is not None:
                raise self.solve_exception
            return {"success": True}

        def close(self, *, kill: bool = False) -> None:
            _ = kill

    monkeypatch.setattr(containment, "SimulationRuntimeOwner", _FakeRuntimeOwner)
    cases = [
        (
            ContainmentKernelStartupTimeout(0.25),
            containment.SimulationContainmentStartupTimeout,
        ),
        (
            ContainmentKernelAcceptTimeout(0.5),
            containment.SimulationContainmentAcceptTimeout,
        ),
        (
            ContainmentKernelActiveTimeout(0.75),
            containment.SimulationContainmentTimeout,
        ),
        (
            ContainmentKernelChildFailure({"kind": "boom", "message": "child failed"}),
            containment.SimulationContainmentChildFailure,
        ),
        (
            ContainmentKernelProtocolError("bad protocol"),
            containment.SimulationContainmentProtocolError,
        ),
    ]

    for runtime_exc, expected_exc in cases:
        _FakeRuntimeOwner.solve_exception = runtime_exc
        owner = containment.WarmSimulationOwner({})
        with pytest.raises(expected_exc):
            owner.solve({})


def test_warm_simulation_owner_request_reconstruction_failure_includes_child_traceback():
    from kindred.core.simulation_containment import (
        SimulationContainmentChildFailure,
        SimulationContainmentTimeout,
        WarmSimulationOwner,
    )

    mp_context = _require_spawn_primitive_support()
    owner = WarmSimulationOwner(
        {},
        mp_context=mp_context,
        ready_timeout_s=_OWNER_TEST_READY_TIMEOUT_S,
        accept_timeout_s=_OWNER_TEST_ACCEPT_TIMEOUT_S,
        active_timeout_s=0.001,
    )

    try:
        with pytest.raises(SimulationContainmentChildFailure) as exc:
            owner.solve({"simulation_plan_payload": {"version": 1}})
        failure = exc.value.failure
        assert failure["kind"] == "simulation_containment_reconstruction"
        context = failure.get("context")
        assert isinstance(context, dict)
        stack_trace = str(context.get("stack_trace") or "")
        assert "ValueError" in stack_trace
        assert "SimulationPlan payload missing execution_request" in stack_trace
        assert "before_accept" in stack_trace
        assert owner.is_running is False
    except SimulationContainmentTimeout as exc:  # pragma: no cover - documents the red/green distinction
        pytest.fail(f"default runtime reconstruction failure used active timeout path: {exc!r}")
    finally:
        owner.close(kill=True)


def test_warm_simulation_owner_startup_failure_includes_child_traceback():
    from kindred.core.simulation_containment import SimulationContainmentChildFailure, WarmSimulationOwner

    mp_context = _require_spawn_primitive_support()
    owner = WarmSimulationOwner(
        {
            "version": 1,
            "execution_mode": "explicit",
            "algebra_policy": "gui_best_effort",
            "execution_request": "not-a-mapping",
        },
        mp_context=mp_context,
        ready_timeout_s=_OWNER_TEST_READY_TIMEOUT_S,
        accept_timeout_s=_OWNER_TEST_ACCEPT_TIMEOUT_S,
        active_timeout_s=0.5,
    )

    try:
        with pytest.raises(SimulationContainmentChildFailure) as exc:
            owner.solve({})
        failure = exc.value.failure
        assert failure["kind"] == "simulation_containment_startup"
        context = failure.get("context")
        assert isinstance(context, dict)
        stack_trace = str(context.get("stack_trace") or "")
        assert "ValueError" in stack_trace
        assert "SimulationPlan execution_request must be a mapping" in stack_trace
        assert "create_simulation_child_handler" in stack_trace
    finally:
        owner.close(kill=True)


def test_warm_simulation_owner_preserves_gui_algebra_outputs_in_result_payload():
    from kindred.core.simulation_containment import (
        WarmSimulationOwner,
        build_contained_simulation_plan_payload,
    )
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
    from kindred.core.simulation_preparation import (
        SimulationExecutionRequest,
        prepare_bound_mechanism,
    )

    mp_context = _require_spawn_primitive_support()
    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "param scale = 2.0",
            "let total = [A] + [B]",
        ]
    )
    bound = prepare_bound_mechanism(
        mechanism_text,
        [],
        temperature_K=298.15,
        initials={"A": 1.0, "B": 0.0},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=False,
    )
    request = SimulationExecutionRequest(
        prepared_payload=bound.as_serializable_execution_payload(),
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 0.05),
        solver_config={"solver": "BDF", "grid": {"N": 4}, "use_sparse_jacobian": False},
        mechanism_text=mechanism_text,
        simulation_identity={"schema_id": "algebra", "param_fingerprint": "algebra"},
    )
    plan = SimulationPlan.from_execution_request(
        request,
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
    )
    contained_payload = build_contained_simulation_plan_payload(plan)
    owner = WarmSimulationOwner(
        contained_payload,
        mp_context=mp_context,
        ready_timeout_s=_OWNER_TEST_READY_TIMEOUT_S,
        accept_timeout_s=_OWNER_TEST_ACCEPT_TIMEOUT_S,
        active_timeout_s=5.0,
    )

    try:
        payload = owner.solve(
            {
                "include_mechanism_in_result_payload": False,
            }
        )

        assert payload["base_species_count"] == 2
        assert payload["species_names"] == ["A", "B", "total"]
        assert np.asarray(payload["Y"]).shape[0] == 3
        assert isinstance(payload["algebra_scalars"], dict)
        assert payload["algebra_errors"] == []

        second_payload = owner.solve({"include_mechanism_in_result_payload": False})

        assert second_payload["species_names"] == ["A", "B", "total"]
        assert owner.owner_epoch == 1
    finally:
        owner.close(kill=True)


def test_warm_simulation_owner_preview_sparse_dynamic_override_omits_jacobian_hint():
    from kindred.core.simulation_containment import (
        WarmSimulationOwner,
        build_contained_simulation_plan_payload,
    )
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
    from kindred.core.simulation_preparation import SimulationExecutionRequest

    mp_context = _require_spawn_primitive_support()
    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "init: A=1.0, B=0.0",
        ]
    )
    request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={},
        t_span=(0.0, 1.0),
        solver_config={
            "solver": "BDF",
            "grid": {"N": 6},
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
        mechanism_text=mechanism_text,
        parameter_overrides={"k1": 2.0},
        simulation_identity={"schema_id": "preview-sparse-dynamic", "param_fingerprint": "k1=2"},
    )
    plan = SimulationPlan.from_execution_request(
        request,
        execution_mode="preview",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
    )
    contained_payload = build_contained_simulation_plan_payload(plan)
    owner = WarmSimulationOwner(
        contained_payload,
        mp_context=mp_context,
        ready_timeout_s=_OWNER_TEST_READY_TIMEOUT_S,
        accept_timeout_s=_OWNER_TEST_ACCEPT_TIMEOUT_S,
        active_timeout_s=5.0,
    )

    try:
        payload = owner.solve({"include_mechanism_in_result_payload": False})

        assert payload["success"] is True
        assert payload["species_names"] == ["A", "B"]
        assert np.asarray(payload["Y"]).shape == (2, 6)
        assert payload["provenance"]["symbolic_jacobian"] is True
        assert payload["provenance"]["jacobian_sparsity_hint"] is False
    finally:
        owner.close(kill=True)
