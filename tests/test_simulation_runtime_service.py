from __future__ import annotations

import json
import subprocess  # nosec B404 - tests invoke the local interpreter with controlled args
import sys
from pathlib import Path

import pytest

from tests.test_containment_kernel import _ACCEPT_TIMEOUT_S, _READY_TIMEOUT_S, _require_spawn_primitive_support

pytestmark = pytest.mark.unit


def test_runtime_service_import_is_lazy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import json
import sys
import kindred.core.simulation_runtime_service
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
    assert json.loads(result.stdout.strip()) == {
        "numpy": False,
        "scipy": False,
        "pyside": False,
        "solvers": False,
        "batch_parallel": False,
        "simulation_containment": False,
        "fitting_containment": False,
    }


def test_runtime_owner_records_event_order_for_request() -> None:
    from kindred.core.simulation_runtime_service import SimulationRuntimeOwner

    owner = SimulationRuntimeOwner(
        handler_import_path="tests.test_containment_kernel:make_kernel_test_handler",
        startup_payload={},
        mp_context=_require_spawn_primitive_support(),
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
    )
    try:
        result = owner.solve({"behavior": "echo"}, active_timeout_s=1.0)
        assert result["echo"] == {"behavior": "echo"}
        assert [event.kind for event in owner.drain_events()] == [
            "owner_starting",
            "owner_ready",
            "request_submitted",
            "request_accepted",
            "request_result",
        ]
    finally:
        owner.close(kill=True)


def test_runtime_owner_can_start_nonblocking_then_accept_later_request() -> None:
    from kindred.core.simulation_runtime_service import SimulationRuntimeOwner

    owner = SimulationRuntimeOwner(
        handler_import_path="tests.test_containment_kernel:make_kernel_test_handler",
        startup_payload={},
        mp_context=_require_spawn_primitive_support(),
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
    )
    try:
        owner.warm(wait=False)
        assert owner.owner_epoch == 1
        result = owner.solve({"behavior": "echo", "source": "after-warm"}, active_timeout_s=1.0)
        assert result["echo"] == {"behavior": "echo", "source": "after-warm"}
    finally:
        owner.close(kill=True)


def test_runtime_owner_forwards_caller_owned_reply_fields() -> None:
    from kindred.core.simulation_runtime_service import SimulationRuntimeOwner

    owner = SimulationRuntimeOwner(
        handler_import_path="tests.test_containment_kernel:make_kernel_test_handler",
        startup_payload={},
        mp_context=_require_spawn_primitive_support(),
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
    )
    try:
        result = owner.solve(
            {"behavior": "stale_batch_dimensions_then_result", "run_id": 9, "set_id": "set-9"},
            active_timeout_s=1.0,
            reply_fields={"run_id": 9, "set_id": "set-9"},
        )

        assert "stale" not in result
        event_kinds = [event.kind for event in owner.drain_events()]
        assert "stale_ignored" in event_kinds
    finally:
        owner.close(kill=True)


def test_runtime_owner_maps_child_cancelled_reply_to_cancelled_exception() -> None:
    from kindred.core.containment_kernel import ContainmentKernelCancelled
    from kindred.core.simulation_runtime_service import SimulationRuntimeOwner

    owner = SimulationRuntimeOwner(
        handler_import_path="tests.test_containment_kernel:make_kernel_test_handler",
        startup_payload={},
        mp_context=_require_spawn_primitive_support(),
        ready_timeout_s=_READY_TIMEOUT_S,
        accept_timeout_s=_ACCEPT_TIMEOUT_S,
    )
    try:
        with pytest.raises(ContainmentKernelCancelled):
            owner.solve({"behavior": "cancelled"}, active_timeout_s=1.0)
        assert owner.is_running is False
    finally:
        owner.close(kill=True)
