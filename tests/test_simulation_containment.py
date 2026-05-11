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


def test_warm_simulation_owner_custom_child_target_applies_handler_env():
    from kindred.core.simulation_containment import WarmSimulationOwner

    owner = WarmSimulationOwner(
        {},
        child_target=_warm_owner_test_child,
        handler_env={"KINDRED_TEST_CHILD_ENV": "applied"},
        mp_context=_require_spawn_queue_support(),
        ready_timeout_s=_OWNER_TEST_READY_TIMEOUT_S,
        accept_timeout_s=_OWNER_TEST_ACCEPT_TIMEOUT_S,
        active_timeout_s=1.0,
    )
    try:
        result = owner.solve(
            {
                "behavior": "env_echo",
                "env_names": ["KINDRED_TEST_CHILD_ENV"],
            }
        )
        assert result["env"] == {"KINDRED_TEST_CHILD_ENV": "applied"}
    finally:
        owner.close(kill=True)


def test_warm_simulation_owner_delayed_ready_is_startup_timeout_not_active_timeout():
    from kindred.core.simulation_containment import (
        SimulationContainmentStartupTimeout,
        SimulationContainmentTimeout,
        WarmSimulationOwner,
    )

    mp_context = _require_spawn_primitive_support()
    owner = WarmSimulationOwner(
        {"ready_delay_s": 0.25},
        mp_context=mp_context,
        child_target=_warm_owner_test_child,
        ready_timeout_s=0.05,
        accept_timeout_s=_OWNER_TEST_ACCEPT_TIMEOUT_S,
        active_timeout_s=0.05,
    )

    try:
        with pytest.raises(SimulationContainmentStartupTimeout):
            owner.solve({"behavior": "echo"})
    except SimulationContainmentTimeout as exc:  # pragma: no cover - documents the red/green distinction
        pytest.fail(f"delayed READY used active timeout path: {exc!r}")
    finally:
        owner.close(kill=True)


def test_warm_simulation_owner_active_timeout_starts_after_accept_and_kills_child():
    from kindred.core.simulation_containment import SimulationContainmentTimeout, WarmSimulationOwner

    mp_context = _require_spawn_primitive_support()
    owner = WarmSimulationOwner(
        {},
        mp_context=mp_context,
        child_target=_warm_owner_test_child,
        ready_timeout_s=_OWNER_TEST_READY_TIMEOUT_S,
        accept_timeout_s=_OWNER_TEST_ACCEPT_TIMEOUT_S,
        active_timeout_s=0.05,
    )

    try:
        with pytest.raises(SimulationContainmentTimeout) as exc:
            owner.solve({"behavior": "hang_after_accept"})
        assert exc.value.failure["kind"] == "timeout"
        assert owner.owner_epoch == 1
        assert owner.is_running is False
    finally:
        owner.close(kill=True)


def test_warm_simulation_owner_accept_timeout_kills_child():
    from kindred.core.simulation_containment import SimulationContainmentAcceptTimeout, WarmSimulationOwner

    mp_context = _require_spawn_primitive_support()
    owner = WarmSimulationOwner(
        {},
        mp_context=mp_context,
        child_target=_warm_owner_test_child,
        ready_timeout_s=_OWNER_TEST_READY_TIMEOUT_S,
        accept_timeout_s=0.05,
        active_timeout_s=5.0,
    )

    try:
        with pytest.raises(SimulationContainmentAcceptTimeout):
            owner.solve({"behavior": "fatal_before_accept", "fatal_delay_s": 0.2})
        assert owner.is_running is False
    finally:
        owner.close(kill=True)


def test_warm_simulation_owner_reconstruction_failure_before_accept_is_not_active_timeout():
    from kindred.core.simulation_containment import (
        SimulationContainmentChildFailure,
        SimulationContainmentTimeout,
        WarmSimulationOwner,
    )

    mp_context = _require_spawn_primitive_support()
    owner = WarmSimulationOwner(
        {},
        mp_context=mp_context,
        child_target=_warm_owner_test_child,
        ready_timeout_s=_OWNER_TEST_READY_TIMEOUT_S,
        accept_timeout_s=_OWNER_TEST_ACCEPT_TIMEOUT_S,
        active_timeout_s=0.01,
    )

    try:
        with pytest.raises(SimulationContainmentChildFailure) as exc:
            owner.solve({"behavior": "fatal_before_accept", "fatal_delay_s": 0.05})
        assert exc.value.failure["kind"] == "simulation_containment_reconstruction"
    except SimulationContainmentTimeout as exc:  # pragma: no cover - documents the red/green distinction
        pytest.fail(f"pre-accept reconstruction failure used active timeout path: {exc!r}")
    finally:
        owner.close(kill=True)


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


def test_warm_simulation_owner_cancellation_kills_child_and_reply_gate_ignores_stale_replies():
    from kindred.core.exceptions import SimulationCancelled
    from kindred.core.simulation_containment import SimulationReplyGate, WarmSimulationOwner

    mp_context = _require_spawn_primitive_support()
    owner = WarmSimulationOwner(
        {},
        mp_context=mp_context,
        child_target=_warm_owner_test_child,
        ready_timeout_s=_OWNER_TEST_READY_TIMEOUT_S,
        accept_timeout_s=_OWNER_TEST_ACCEPT_TIMEOUT_S,
        active_timeout_s=5.0,
    )
    started = time.monotonic()

    def _cancel_after_accept() -> bool:
        return (time.monotonic() - started) >= 0.05

    try:
        with pytest.raises(SimulationCancelled):
            owner.solve({"behavior": "hang_after_accept"}, cancellation_check=_cancel_after_accept)
        assert owner.is_running is False

        gate = SimulationReplyGate(owner_epoch=3, request_id=9)
        assert gate.is_current({"owner_epoch": 2, "request_id": 9}) is False
        assert gate.is_current({"owner_epoch": 3, "request_id": 8}) is False
        assert gate.is_current({"owner_epoch": 3, "request_id": 9}) is True
    finally:
        owner.close(kill=True)


def test_warm_simulation_owner_restart_after_timeout_is_lazy_and_increments_epoch():
    from kindred.core.simulation_containment import SimulationContainmentTimeout, WarmSimulationOwner

    mp_context = _require_spawn_primitive_support()
    owner = WarmSimulationOwner(
        {},
        mp_context=mp_context,
        child_target=_warm_owner_test_child,
        ready_timeout_s=_OWNER_TEST_READY_TIMEOUT_S,
        accept_timeout_s=_OWNER_TEST_ACCEPT_TIMEOUT_S,
        active_timeout_s=0.05,
    )

    try:
        with pytest.raises(SimulationContainmentTimeout):
            owner.solve({"behavior": "hang_after_accept"})
        assert owner.owner_epoch == 1
        assert owner.is_running is False

        result = owner.solve({"behavior": "echo"})

        assert result["success"] is True
        assert result["owner_epoch"] == 2
        assert owner.owner_epoch == 2
    finally:
        owner.close(kill=True)


def test_warm_simulation_owner_cancel_restart_is_lazy_and_increments_epoch():
    from kindred.core.exceptions import SimulationCancelled
    from kindred.core.simulation_containment import WarmSimulationOwner

    mp_context = _require_spawn_primitive_support()
    owner = WarmSimulationOwner(
        {},
        mp_context=mp_context,
        child_target=_warm_owner_test_child,
        ready_timeout_s=_OWNER_TEST_READY_TIMEOUT_S,
        accept_timeout_s=_OWNER_TEST_ACCEPT_TIMEOUT_S,
        active_timeout_s=5.0,
    )
    started = time.monotonic()

    try:
        with pytest.raises(SimulationCancelled):
            owner.solve(
                {"behavior": "hang_after_accept"},
                cancellation_check=lambda: (time.monotonic() - started) >= 0.05,
            )
        assert owner.owner_epoch == 1
        assert owner.is_running is False

        result = owner.solve({"behavior": "echo"})

        assert result["owner_epoch"] == 2
        assert owner.owner_epoch == 2
    finally:
        owner.close(kill=True)


def test_warm_simulation_owner_reuses_ready_owner_for_two_requests(tmp_path):
    from kindred.core.simulation_containment import WarmSimulationOwner

    mp_context = _require_spawn_primitive_support()
    startup_count_path = tmp_path / "startup-count.txt"
    owner = WarmSimulationOwner(
        {"startup_count_path": str(startup_count_path)},
        mp_context=mp_context,
        child_target=_warm_owner_test_child,
        ready_timeout_s=_OWNER_TEST_READY_TIMEOUT_S,
        accept_timeout_s=_OWNER_TEST_ACCEPT_TIMEOUT_S,
        active_timeout_s=0.5,
    )

    try:
        first = owner.solve({"value": 1})
        second = owner.solve({"value": 2})

        assert first["owner_epoch"] == second["owner_epoch"] == 1
        assert startup_count_path.read_text(encoding="utf-8").splitlines() == ["1"]
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
            "total = [A] + [B]",
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
        assert payload["provenance"]["symbolic_jacobian"] is False
        assert payload["provenance"]["jacobian_sparsity_hint"] is False
    finally:
        owner.close(kill=True)


def test_warm_simulation_owner_ignores_stale_result_before_current_acceptance():
    from kindred.core.simulation_containment import WarmSimulationOwner

    mp_context = _require_spawn_primitive_support()
    owner = WarmSimulationOwner(
        {},
        mp_context=mp_context,
        child_target=_warm_owner_test_child,
        ready_timeout_s=_OWNER_TEST_READY_TIMEOUT_S,
        accept_timeout_s=_OWNER_TEST_ACCEPT_TIMEOUT_S,
        active_timeout_s=0.5,
    )

    try:
        result = owner.solve({"behavior": "stale_then_result"})

        assert result["success"] is True
        assert "stale" not in result
        assert result["owner_epoch"] == 1
    finally:
        owner.close(kill=True)
