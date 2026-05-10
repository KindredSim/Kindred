from __future__ import annotations

import json
import multiprocessing
import os
import queue
import subprocess  # nosec B404 - tests invoke the local interpreter with controlled args
import sys
import tempfile
import threading
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
        if behavior == "hard_exit":
            os._exit(int(request.get("exit_code") or 87))
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


class _EnvCapturingQueue:
    def put(self, _message: Mapping[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None

    def join_thread(self) -> None:
        return None


class _EnvCapturingProcess:
    def __init__(self, *, env_names: tuple[str, ...], captured: list[dict[str, str | None]]) -> None:
        self._env_names = tuple(env_names)
        self._captured = captured
        self.pid: int | None = None

    def start(self) -> None:
        self._captured.append({name: os.environ.get(name) for name in self._env_names})
        self.pid = 12345

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        _ = timeout


class _EnvCapturingContext:
    def __init__(self, *, env_names: tuple[str, ...]) -> None:
        self.env_names = tuple(env_names)
        self.captured: list[dict[str, str | None]] = []

    def Queue(self) -> _EnvCapturingQueue:
        return _EnvCapturingQueue()

    def Process(self, **_kwargs: Any) -> _EnvCapturingProcess:
        return _EnvCapturingProcess(env_names=self.env_names, captured=self.captured)


class _RaisingEnvCapturingProcess(_EnvCapturingProcess):
    def start(self) -> None:
        self._captured.append({name: os.environ.get(name) for name in self._env_names})
        raise RuntimeError("start boom")


class _ConcurrentEnvCapturingProcess:
    def __init__(self, *, env_names: tuple[str, ...], captured: list[dict[str, str | None]]) -> None:
        self._env_names = tuple(env_names)
        self._captured = captured

    def start(self) -> None:
        self._captured.append({name: os.environ.get(name) for name in self._env_names})


class _NestedConcurrentStartProcess:
    def __init__(self, *, env_name: str) -> None:
        self._env_name = str(env_name)
        self.inner_captured: list[dict[str, str | None]] = []
        self.inner_started = threading.Event()
        self.inner_finished = threading.Event()
        self.inner_finished_during_outer_start = False
        self.inner_thread: threading.Thread | None = None

    def start(self) -> None:
        from kindred.core.containment_kernel import _start_process_with_env

        inner = _ConcurrentEnvCapturingProcess(
            env_names=(self._env_name,),
            captured=self.inner_captured,
        )

        def _run_inner_start() -> None:
            self.inner_started.set()
            _start_process_with_env(inner, {})
            self.inner_finished.set()

        self.inner_thread = threading.Thread(target=_run_inner_start)
        self.inner_thread.start()
        assert self.inner_started.wait(timeout=1.0)
        time.sleep(0.05)
        self.inner_finished_during_outer_start = self.inner_finished.is_set()


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


def test_kernel_applies_handler_env_during_process_start_and_restores_parent(monkeypatch) -> None:
    from kindred.core.containment_kernel import ContainmentHandlerSpec, ContainmentKernelOwner

    env_names = ("KINDRED_TEST_PRESTART_NEW", "KINDRED_TEST_PRESTART_EXISTING")
    monkeypatch.delenv("KINDRED_TEST_PRESTART_NEW", raising=False)
    monkeypatch.setenv("KINDRED_TEST_PRESTART_EXISTING", "parent")
    mp_context = _EnvCapturingContext(env_names=env_names)

    owner = ContainmentKernelOwner(
        ContainmentHandlerSpec(
            import_path="tests.test_containment_kernel:make_kernel_test_handler",
            env={
                "KINDRED_TEST_PRESTART_NEW": "child",
                "KINDRED_TEST_PRESTART_EXISTING": "child",
            },
        ),
        mp_context=mp_context,  # type: ignore[arg-type]
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
    )

    owner.start(wait=False)
    owner.close(kill=True)

    assert mp_context.captured == [
        {
            "KINDRED_TEST_PRESTART_NEW": "child",
            "KINDRED_TEST_PRESTART_EXISTING": "child",
        }
    ]
    assert os.environ.get("KINDRED_TEST_PRESTART_NEW") is None
    assert os.environ.get("KINDRED_TEST_PRESTART_EXISTING") == "parent"


def test_kernel_restores_handler_env_when_process_start_raises(monkeypatch) -> None:
    from kindred.core.containment_kernel import _start_process_with_env

    env_names = ("KINDRED_TEST_PRESTART_RAISE_NEW", "KINDRED_TEST_PRESTART_RAISE_EXISTING")
    monkeypatch.delenv("KINDRED_TEST_PRESTART_RAISE_NEW", raising=False)
    monkeypatch.setenv("KINDRED_TEST_PRESTART_RAISE_EXISTING", "parent")
    captured: list[dict[str, str | None]] = []
    process = _RaisingEnvCapturingProcess(env_names=env_names, captured=captured)

    with pytest.raises(RuntimeError, match="start boom"):
        _start_process_with_env(
            process,  # type: ignore[arg-type]
            {
                "KINDRED_TEST_PRESTART_RAISE_NEW": "child",
                "KINDRED_TEST_PRESTART_RAISE_EXISTING": "child",
            },
        )

    assert captured == [
        {
            "KINDRED_TEST_PRESTART_RAISE_NEW": "child",
            "KINDRED_TEST_PRESTART_RAISE_EXISTING": "child",
        }
    ]
    assert os.environ.get("KINDRED_TEST_PRESTART_RAISE_NEW") is None
    assert os.environ.get("KINDRED_TEST_PRESTART_RAISE_EXISTING") == "parent"


def test_kernel_serializes_empty_env_process_start_against_temporary_parent_env(monkeypatch) -> None:
    from kindred.core.containment_kernel import _start_process_with_env

    env_name = "KINDRED_TEST_PRESTART_CONCURRENT"
    monkeypatch.delenv(env_name, raising=False)
    outer = _NestedConcurrentStartProcess(env_name=env_name)

    _start_process_with_env(
        outer,  # type: ignore[arg-type]
        {env_name: "child"},
    )
    assert outer.inner_thread is not None
    outer.inner_thread.join(timeout=1.0)

    assert outer.inner_finished_during_outer_start is False
    assert outer.inner_finished.is_set() is True
    assert outer.inner_captured == [{env_name: None}]
    assert os.environ.get(env_name) is None


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


def test_kernel_unexpected_child_exit_reports_exit_code_without_sidecar_diagnostic() -> None:
    from kindred.core.containment_kernel import (
        ContainmentHandlerSpec,
        ContainmentKernelOwner,
        ContainmentKernelProtocolError,
    )

    class _ExitedProcess:
        pid = 123456
        exitcode = 87

        def is_alive(self) -> bool:
            return False

        def join(self, timeout: float | None = None) -> None:
            _ = timeout

    diagnostic_path = Path(tempfile.gettempdir()) / "kindred-contained-child-123456.faulthandler.log"
    diagnostic_path.write_text("stale sidecar diagnostic should be ignored\n", encoding="utf-8")
    owner = ContainmentKernelOwner(
        ContainmentHandlerSpec(import_path="tests.test_containment_kernel:make_kernel_test_handler")
    )
    try:
        owner._process = _ExitedProcess()  # type: ignore[assignment]
        with pytest.raises(ContainmentKernelProtocolError) as exc_info:
            owner._raise_if_process_exited()

        message = str(exc_info.value)
        assert "Contained child exited unexpectedly with code 87." in message
        assert "Child diagnostic log:" not in message
        assert str(diagnostic_path) not in message
        assert "stale sidecar diagnostic" not in message
    finally:
        owner.close(kill=True)
        try:
            diagnostic_path.unlink()
        except FileNotFoundError:
            pass


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
