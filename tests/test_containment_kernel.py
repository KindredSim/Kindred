from __future__ import annotations

import json
import multiprocessing
import os
import queue
import subprocess  # nosec B404 - tests invoke the local interpreter with controlled args
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_READY_TIMEOUT_S = 2.0
_ACCEPT_TIMEOUT_S = 2.0


def _process_context() -> multiprocessing.context.BaseContext:
    return multiprocessing.get_context("spawn")


def _spawn_probe_child(output_queue: multiprocessing.Queue) -> None:
    output_queue.put({"ok": True})


def _terminate_process(proc: multiprocessing.Process) -> None:
    if not proc.is_alive():
        return
    proc.terminate()
    proc.join(timeout=0.5)
    if proc.is_alive() and hasattr(proc, "kill"):
        proc.kill()
        proc.join(timeout=0.5)


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


class _KernelTestHandler:
    def __init__(self, startup_payload: Mapping[str, Any]) -> None:
        payload = dict(startup_payload or {})
        startup_count_path = str(payload.get("startup_count_path") or "")
        if startup_count_path:
            owner_epoch = int(payload.get("owner_epoch_for_file") or 0)
            with open(startup_count_path, "a", encoding="utf-8") as handle:
                handle.write(f"{owner_epoch}\n")
        startup_delay = float(payload.get("startup_delay_s") or 0.0)
        if startup_delay > 0.0:
            time.sleep(startup_delay)
        if payload.get("startup_error"):
            raise RuntimeError("startup failed")

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
                    "owner_epoch": int(context.owner_epoch) - 1,
                    "request_id": int(context.request_id),
                    "payload": {"stale": True},
                }
            )
        if behavior == "stale_batch_dimensions_then_result":
            context.output_queue.put(
                {
                    "kind": "result",
                    "owner_epoch": int(context.owner_epoch),
                    "request_id": int(context.request_id),
                    "run_id": int(request.get("run_id") or 0) + 1,
                    "set_id": f"{request.get('set_id')}-stale",
                    "payload": {"stale": True},
                }
            )
        if behavior == "hang_after_accept":
            while True:
                time.sleep(0.05)
        if behavior == "cancelled":
            from kindred.core.containment_kernel import ContainmentHandlerResponse

            return ContainmentHandlerResponse(kind="cancelled", failure={"kind": "cancelled", "message": "cancelled"})
        if behavior == "env_probe":
            numpy_preimported = "numpy" in sys.modules
            env_before_numpy = {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
            import numpy  # noqa: F401

            return {
                "numpy_preimported": numpy_preimported,
                "env_before_numpy": env_before_numpy,
            }
        return {"echo": request, "owner_epoch": int(context.owner_epoch), "request_id": int(context.request_id)}


def make_kernel_test_handler(startup_payload: Mapping[str, Any]) -> _KernelTestHandler:
    payload = dict(startup_payload or {})
    if "owner_epoch" in payload:
        payload["owner_epoch_for_file"] = payload["owner_epoch"]
    return _KernelTestHandler(payload)


def _owner(*, startup_payload: Mapping[str, Any] | None = None, **kwargs: Any):
    from kindred.core.containment_kernel import ContainmentHandlerSpec, ContainmentKernelOwner

    return ContainmentKernelOwner(
        ContainmentHandlerSpec(import_path="tests.test_containment_kernel:make_kernel_test_handler"),
        startup_payload=dict(startup_payload or {}),
        mp_context=_require_spawn_primitive_support(),
        **kwargs,
    )


def test_kernel_import_is_stdlib_lazy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import json
import sys
import kindred.core.containment_kernel
print(json.dumps({
    "numpy": "numpy" in sys.modules,
    "scipy": "scipy" in sys.modules,
    "pyside": "PySide6" in sys.modules,
    "solvers": "kindred.core.simulator.solvers" in sys.modules,
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
    payload = json.loads(result.stdout.strip())
    assert payload == {
        "numpy": False,
        "scipy": False,
        "pyside": False,
        "solvers": False,
        "batch_parallel": False,
        "simulation_containment": False,
        "fitting_containment": False,
    }


def test_kernel_ready_and_accepted_request_succeeds() -> None:
    owner = _owner(ready_timeout_s=_READY_TIMEOUT_S, accept_timeout_s=_ACCEPT_TIMEOUT_S)
    try:
        result = owner.request({"behavior": "echo", "value": 7}, active_timeout_s=1.0)
        assert result["echo"] == {"behavior": "echo", "value": 7}
        assert result["owner_epoch"] == 1
        assert result["request_id"] == 1
        event_kinds = [event.kind for event in owner.drain_events()]
        assert "owner_ready" in event_kinds
        assert "request_accepted" in event_kinds
        assert "request_result" in event_kinds
    finally:
        owner.close(kill=True)


def test_kernel_retained_owner_event_history_is_bounded() -> None:
    from kindred.core.containment_kernel import ContainmentHandlerSpec, ContainmentKernelOwner

    owner = ContainmentKernelOwner(
        ContainmentHandlerSpec(import_path="tests.test_containment_kernel:make_kernel_test_handler"),
        mp_context=_require_spawn_primitive_support(),
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
        event_history_limit=5,
    )
    try:
        for value in range(4):
            result = owner.request({"behavior": "echo", "value": value}, active_timeout_s=1.0)
            assert result["echo"]["value"] == value
            assert owner.is_running is True

        events = owner.drain_events()
        assert len(events) == 5
        assert events[-1].kind == "request_result"
        assert events[-1].request_id == 4
        assert min(event.request_id for event in events if event.request_id is not None) >= 3
        assert owner.drain_events() == []
    finally:
        owner.close(kill=True)


def test_kernel_startup_timeout_kills_child() -> None:
    from kindred.core.containment_kernel import ContainmentKernelStartupTimeout

    owner = _owner(
        startup_payload={"startup_delay_s": 0.25},
        ready_timeout_s=0.05,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
    )
    try:
        with pytest.raises(ContainmentKernelStartupTimeout):
            owner.request({"behavior": "echo"}, active_timeout_s=1.0)
        assert owner.is_running is False
    finally:
        owner.close(kill=True)


def test_kernel_accept_timeout_kills_child() -> None:
    from kindred.core.containment_kernel import ContainmentKernelAcceptTimeout

    owner = _owner(ready_timeout_s=_READY_TIMEOUT_S, accept_timeout_s=0.05)
    try:
        with pytest.raises(ContainmentKernelAcceptTimeout):
            owner.request({"behavior": "hang_before_accept"}, active_timeout_s=1.0)
        assert owner.is_running is False
    finally:
        owner.close(kill=True)


def test_kernel_active_timeout_starts_after_accepted_and_restarts_epoch() -> None:
    from kindred.core.containment_kernel import ContainmentKernelActiveTimeout

    owner = _owner(ready_timeout_s=_READY_TIMEOUT_S, accept_timeout_s=_ACCEPT_TIMEOUT_S)
    try:
        with pytest.raises(ContainmentKernelActiveTimeout):
            owner.request({"behavior": "hang_after_accept"}, active_timeout_s=0.05)
        assert owner.owner_epoch == 1
        assert owner.is_running is False

        result = owner.request({"behavior": "echo"}, active_timeout_s=1.0)
        assert result["owner_epoch"] == 2
        assert owner.owner_epoch == 2
    finally:
        owner.close(kill=True)


def test_kernel_rejects_stale_replies_before_current_result() -> None:
    owner = _owner(ready_timeout_s=_READY_TIMEOUT_S, accept_timeout_s=_ACCEPT_TIMEOUT_S)
    try:
        result = owner.request({"behavior": "stale_then_result"}, active_timeout_s=1.0)
        assert "stale" not in result
        assert result["owner_epoch"] == 1
        event_kinds = [event.kind for event in owner.drain_events()]
        assert "stale_ignored" in event_kinds
    finally:
        owner.close(kill=True)


def test_kernel_rejects_caller_owned_stale_reply_fields() -> None:
    owner = _owner(ready_timeout_s=_READY_TIMEOUT_S, accept_timeout_s=_ACCEPT_TIMEOUT_S)
    try:
        result = owner.request(
            {"behavior": "stale_batch_dimensions_then_result", "run_id": 4, "set_id": "set-4"},
            active_timeout_s=1.0,
            reply_fields={"run_id": 4, "set_id": "set-4"},
        )
        assert "stale" not in result
        event_kinds = [event.kind for event in owner.drain_events()]
        assert "stale_ignored" in event_kinds
    finally:
        owner.close(kill=True)


def test_kernel_maps_child_cancelled_reply_to_cancelled_exception() -> None:
    from kindred.core.containment_kernel import ContainmentKernelCancelled

    owner = _owner(ready_timeout_s=_READY_TIMEOUT_S, accept_timeout_s=_ACCEPT_TIMEOUT_S)
    try:
        with pytest.raises(ContainmentKernelCancelled):
            owner.request({"behavior": "cancelled"}, active_timeout_s=1.0)
        assert owner.is_running is False
        event_kinds = [event.kind for event in owner.drain_events()]
        assert "request_cancelled" in event_kinds
    finally:
        owner.close(kill=True)


def test_kernel_close_is_idempotent() -> None:
    owner = _owner(ready_timeout_s=_READY_TIMEOUT_S, accept_timeout_s=_ACCEPT_TIMEOUT_S)
    owner.start(wait=False)
    owner.close(kill=True)
    owner.close(kill=True)
    owner.close(kill=False)
    assert owner.is_running is False


def test_kernel_bootstrap_sets_blas_env_before_child_imports_numpy() -> None:
    from kindred.core.containment_kernel import ContainmentHandlerSpec, ContainmentKernelOwner

    owner = ContainmentKernelOwner(
        ContainmentHandlerSpec(
            import_path="tests.test_containment_kernel:make_kernel_test_handler",
            env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        ),
        startup_payload={},
        mp_context=_require_spawn_primitive_support(),
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
    )
    try:
        result = owner.request({"behavior": "env_probe"}, active_timeout_s=1.0)
        assert result["numpy_preimported"] is False
        assert result["env_before_numpy"] == {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    finally:
        owner.close(kill=True)
