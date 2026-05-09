from __future__ import annotations

import pytest

from kindred.gui.controllers.serial_worker_launch import (
    ContainedSerialWorkerLaunchOwner,
    ContainedSerialWorkerLaunchRequest,
)
from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity


pytestmark = pytest.mark.unit


class _Worker:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.progress = object()
        self.result_ready = object()
        self.error = object()


class _RaisingWorker:
    def __init__(self, **_kwargs):
        raise RuntimeError("worker construction failed")


class _RuntimeApplication:
    def __init__(self, owner: object, *, release_error: Exception | None = None) -> None:
        self.owner = owner
        self.release_error = release_error
        self.acquire_calls: list[dict[str, object]] = []
        self.release_calls: list[dict[str, object]] = []

    def acquire_ready_owner(self, *, mode: str, payload: dict[str, object]) -> object:
        self.acquire_calls.append({"mode": str(mode), "payload": dict(payload)})
        return self.owner

    def release_owner(self, owner: object, *, kill: bool = False) -> None:
        self.release_calls.append({"owner": owner, "kill": bool(kill)})
        if self.release_error is not None:
            raise self.release_error


def test_contained_serial_worker_launch_stamps_callback_identity_on_worker() -> None:
    acquired_owner = object()
    runtime_application = _RuntimeApplication(acquired_owner)

    owner = ContainedSerialWorkerLaunchOwner(
        runtime_application=runtime_application,
        worker_factory=_Worker,
        contained_payload_builder=lambda payload: {"contained": dict(payload)},
    )
    identity = SimulationCallbackIdentity.capture(
        run_id=9,
        fast_mode=True,
        request_id=11,
        owner_epoch=13,
        batch_set="Set 1",
        batch_set_id="id1",
        cache_key="cache-1",
    )

    worker = owner.create_worker(
        ContainedSerialWorkerLaunchRequest(
            plan_payload={"plan": "payload"},
            callback_identity=identity,
            include_mechanism_in_result_payload=True,
            worker_signature="signature-1",
            parent=object(),
        )
    )

    assert isinstance(worker, _Worker)
    assert runtime_application.acquire_calls == [
        {"mode": "preview", "payload": {"contained": {"plan": "payload"}}}
    ]
    assert worker.kwargs["owner"] is acquired_owner
    assert worker.kwargs["simulation_plan_payload"] == {"contained": {"plan": "payload"}}
    assert worker._run_id == 9
    assert worker._request_id == 11
    assert worker._fast_mode is True
    assert worker._batch_set_name == "Set 1"
    assert worker._batch_set_id == "id1"
    assert worker._batch_cache_key == "cache-1"
    assert worker._batch_mechanism_signature == "signature-1"
    assert worker._simulation_plan == {"plan": "payload"}


def test_contained_serial_worker_launch_preserves_worker_failure_when_release_fails() -> None:
    release_errors: list[str] = []
    acquired_owner = object()
    runtime_application = _RuntimeApplication(acquired_owner, release_error=RuntimeError("release failed"))

    owner = ContainedSerialWorkerLaunchOwner(
        runtime_application=runtime_application,
        worker_factory=_RaisingWorker,
        contained_payload_builder=lambda payload: dict(payload),
        record_nonfatal_exception=lambda _message, exc: release_errors.append(str(exc)),
    )
    identity = SimulationCallbackIdentity.capture(
        run_id=9,
        fast_mode=False,
        request_id=11,
        owner_epoch=None,
        batch_set="Set 1",
        batch_set_id="id1",
        cache_key="cache-1",
    )

    with pytest.raises(RuntimeError, match="worker construction failed"):
        owner.create_worker(
            ContainedSerialWorkerLaunchRequest(
                plan_payload={"plan": "payload"},
                callback_identity=identity,
                include_mechanism_in_result_payload=True,
                worker_signature=None,
                parent=object(),
            )
        )

    assert release_errors == ["release failed"]
    assert runtime_application.release_calls == [{"owner": acquired_owner, "kill": False}]
