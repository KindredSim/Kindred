from __future__ import annotations

import threading

import pytest

from kindred.core.simulation_runtime_readiness import SimulationRuntimeApplication


class _Owner:
    def __init__(self, payload: dict[str, object], *, ready_event: threading.Event | None = None) -> None:
        self._payload = dict(payload)
        self.ready_event = ready_event
        self.ready = False
        self.start_calls: list[dict[str, object]] = []
        self.close_calls: list[bool] = []
        self.close_thread_ids: list[int] = []

    @property
    def simulation_plan_payload(self) -> dict[str, object]:
        return dict(self._payload)

    @property
    def is_ready(self) -> bool:
        return bool(self.ready)

    def start(self, *, wait: bool = True) -> None:
        self.start_calls.append({"wait": bool(wait)})
        if self.ready_event is not None:
            self.ready_event.wait(timeout=2.0)
        self.ready = True

    def close(self, *, kill: bool = False) -> None:
        self.close_calls.append(bool(kill))
        self.close_thread_ids.append(threading.get_ident())


class _PrepareCapableOwner(_Owner):
    def __init__(self, payload: dict[str, object], *, ready_event: threading.Event | None = None) -> None:
        super().__init__(payload, ready_event=ready_event)
        self.prepare_calls: list[dict[str, object]] = []

    def prepare_runtime_payload(self, payload: dict[str, object]) -> None:
        self.prepare_calls.append(dict(payload))
        self._payload = dict(payload)
        self.ready = True


@pytest.mark.unit
def test_readiness_service_returns_owner_only_after_exact_owner_is_ready():
    service = SimulationRuntimeApplication()
    payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=1"}}
    ready_event = threading.Event()
    owners: list[_Owner] = []

    def _factory(owner_payload):
        owner = _Owner(dict(owner_payload), ready_event=ready_event)
        owners.append(owner)
        return owner

    owner = service.request_warm(mode="ordinary", payload=payload, owner_factory=_factory, wait=False)

    assert owner is owners[0]
    assert service.ready_owner(mode="ordinary", payload=payload) is None

    ready_event.set()
    snapshot = service.snapshot(mode="ordinary")
    thread = getattr(service._slots["ordinary"], "thread")
    thread.join(timeout=2.0)

    assert owners[0].start_calls == [{"wait": True}]
    assert service.ready_owner(mode="ordinary", payload=payload) is owners[0]
    assert snapshot.mode == "ordinary"


@pytest.mark.unit
def test_readiness_service_replaces_mismatched_owner_and_reuses_exact_ready_owner():
    service = SimulationRuntimeApplication()
    first_payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=1"}}
    second_payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=2"}}
    owners: list[_Owner] = []

    def _factory(owner_payload):
        owner = _Owner(dict(owner_payload))
        owners.append(owner)
        return owner

    first = service.request_warm(mode="preview", payload=first_payload, owner_factory=_factory, wait=True)
    second = service.request_warm(mode="preview", payload=second_payload, owner_factory=_factory, wait=True)
    reused = service.request_warm(mode="preview", payload=second_payload, owner_factory=_factory, wait=True)

    assert first is owners[0]
    assert second is owners[1]
    assert reused is second
    assert owners[0].close_calls == [False]
    assert owners[1].close_calls == []
    assert service.ready_owner(mode="preview", payload=first_payload) is None
    assert service.ready_owner(mode="preview", payload=second_payload) is second


@pytest.mark.unit
def test_readiness_service_replaces_prepare_capable_owner_for_identity_change():
    service = SimulationRuntimeApplication()
    first_payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=1"}}
    second_payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=2"}}
    owners: list[_PrepareCapableOwner] = []

    def _factory(owner_payload):
        owner = _PrepareCapableOwner(dict(owner_payload))
        owners.append(owner)
        return owner

    first = service.request_warm(mode="ordinary", payload=first_payload, owner_factory=_factory, wait=True)
    second = service.request_warm(mode="ordinary", payload=second_payload, owner_factory=_factory, wait=True)

    assert second is not first
    assert owners == [first, second]
    assert first.prepare_calls == [first_payload]
    assert second.prepare_calls == [second_payload]
    assert first.close_calls == [False]
    assert service.ready_owner(mode="ordinary", payload=first_payload) is None
    assert service.ready_owner(mode="ordinary", payload=second_payload) is second


@pytest.mark.unit
def test_readiness_service_does_not_reuse_prepare_capable_owner_while_old_prepare_is_warming():
    service = SimulationRuntimeApplication()
    first_payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=1"}}
    second_payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=2"}}
    release_first_prepare = threading.Event()
    first_prepare_started = threading.Event()
    owners: list[_PrepareCapableOwner] = []

    class _BlockingFirstPrepareOwner(_PrepareCapableOwner):
        def prepare_runtime_payload(self, payload: dict[str, object]) -> None:
            self.prepare_calls.append(dict(payload))
            if dict(payload) == first_payload:
                first_prepare_started.set()
                release_first_prepare.wait(timeout=2.0)
            self._payload = dict(payload)
            self.ready = True

    def _factory(owner_payload):
        if not owners:
            owner = _BlockingFirstPrepareOwner(dict(owner_payload))
        else:
            owner = _PrepareCapableOwner(dict(owner_payload))
        owners.append(owner)
        return owner

    first = service.request_warm(mode="preview", payload=first_payload, owner_factory=_factory, wait=False)
    assert first_prepare_started.wait(timeout=1.0)

    second = service.request_warm(mode="preview", payload=second_payload, owner_factory=_factory, wait=True)
    release_first_prepare.set()
    old_thread = getattr(first, "ready_event", None)
    _ = old_thread

    assert second is not first
    assert owners == [first, second]
    assert first.prepare_calls == [first_payload]
    assert second.prepare_calls == [second_payload]
    assert service.ready_owner(mode="preview", payload=second_payload) is second


@pytest.mark.unit
def test_readiness_service_does_not_close_stale_owner_again_from_warm_thread():
    service = SimulationRuntimeApplication()
    payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=1"}}
    release_start = threading.Event()
    start_entered = threading.Event()
    owners: list[_Owner] = []

    class _BlockingOwner(_Owner):
        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})
            start_entered.set()
            release_start.wait(timeout=2.0)
            self.ready = True

    def _factory(owner_payload):
        owner = _BlockingOwner(dict(owner_payload))
        owners.append(owner)
        return owner

    service.request_warm(mode="preview", payload=payload, owner_factory=_factory, wait=False)
    assert start_entered.wait(timeout=1.0)
    thread = service._slots["preview"].thread

    service.close(mode="preview", kill=True)
    close_thread_id = threading.get_ident()
    release_start.set()
    assert thread is not None
    thread.join(timeout=2.0)

    assert owners[0].close_calls == [True]
    assert owners[0].close_thread_ids == [close_thread_id]
    assert service.snapshot(mode="preview").status == "missing"


@pytest.mark.unit
def test_readiness_service_warms_and_returns_all_exact_serial_queue_owners():
    service = SimulationRuntimeApplication()
    first_payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=1"}, "metadata": {"set_id": "id1"}}
    second_payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=1"}, "metadata": {"set_id": "id2"}}
    owners: list[_Owner] = []

    def _factory(owner_payload):
        owner = _Owner(dict(owner_payload))
        owners.append(owner)
        return owner

    service.ensure_ready_many(
        mode="ordinary",
        payloads=[first_payload, second_payload],
        owner_factory=_factory,
        wait=True,
    )

    assert len(owners) == 2
    assert service.ready_owner(mode="ordinary", payload=first_payload) is owners[0]
    assert service.ready_owner(mode="ordinary", payload=second_payload) is owners[1]
    assert owners[0].start_calls == [{"wait": True}]
    assert owners[1].start_calls == [{"wait": True}]


@pytest.mark.unit
def test_readiness_service_reconciles_single_and_serial_queue_owner_families():
    service = SimulationRuntimeApplication()
    first_payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=1"}, "metadata": {"set_id": "id1"}}
    second_payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=1"}, "metadata": {"set_id": "id2"}}
    owners: list[_Owner] = []

    def _factory(owner_payload):
        owner = _Owner(dict(owner_payload))
        owners.append(owner)
        return owner

    first = service.ensure_ready(
        mode="ordinary",
        payload=first_payload,
        owner_factory=_factory,
        wait=True,
    )
    queued = service.ensure_ready_many(
        mode="ordinary",
        payloads=[first_payload, second_payload],
        owner_factory=_factory,
        wait=True,
    )

    assert queued[0] is first
    assert len(owners) == 2
    assert service.ready_owner(mode="ordinary", payload=first_payload) is first
    assert service.ready_owner(mode="ordinary", payload=second_payload) is owners[1]

    single = service.ensure_ready(
        mode="ordinary",
        payload=second_payload,
        owner_factory=_factory,
        wait=True,
    )

    assert single is owners[1]
    assert len(owners) == 2
    assert owners[0].close_calls == [False]
    assert owners[1].close_calls == []
    assert service.ready_owner(mode="ordinary", payload=first_payload) is None
    assert service.ready_owner(mode="ordinary", payload=second_payload) is owners[1]


@pytest.mark.unit
def test_readiness_service_defers_closing_active_owner_until_released():
    service = SimulationRuntimeApplication()
    first_payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=1"}}
    second_payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=2"}}
    owners: list[_Owner] = []

    def _factory(owner_payload):
        owner = _Owner(dict(owner_payload))
        owners.append(owner)
        return owner

    first = service.ensure_ready(
        mode="ordinary",
        payload=first_payload,
        owner_factory=_factory,
        wait=True,
    )
    acquired = service.acquire_ready_owner(mode="ordinary", payload=first_payload)

    assert acquired is first

    service.ensure_ready_many(
        mode="ordinary",
        payloads=[second_payload],
        owner_factory=_factory,
        wait=True,
    )

    assert owners[0].close_calls == []
    assert service.ready_owner(mode="ordinary", payload=first_payload) is None
    assert service.ready_owner(mode="ordinary", payload=second_payload) is owners[1]

    service.release_owner(first)

    assert owners[0].close_calls == [False]
    assert owners[1].close_calls == []


@pytest.mark.unit
def test_readiness_refresh_reuses_active_matching_owner_after_release():
    service = SimulationRuntimeApplication()
    payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=1"}}
    owners: list[_Owner] = []

    def _factory(owner_payload):
        owner = _Owner(dict(owner_payload))
        owners.append(owner)
        return owner

    first = service.ensure_ready(
        mode="ordinary",
        payload=payload,
        owner_factory=_factory,
        wait=True,
    )
    acquired = service.acquire_ready_owner(mode="ordinary", payload=payload)

    assert acquired is first

    refreshed = service.ensure_ready_many(
        mode="ordinary",
        payloads=[payload],
        owner_factory=_factory,
        wait=True,
    )

    assert refreshed == [first]
    assert owners == [first]
    assert service.ready_owner(mode="ordinary", payload=payload) is None

    service.release_owner(first)

    assert owners == [first]
    assert owners[0].close_calls == []
    assert service.ready_owner(mode="ordinary", payload=payload) is first


@pytest.mark.unit
def test_single_readiness_refresh_reuses_active_matching_owner_after_release():
    service = SimulationRuntimeApplication()
    payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=1"}}
    owners: list[_Owner] = []

    def _factory(owner_payload):
        owner = _Owner(dict(owner_payload))
        owners.append(owner)
        return owner

    first = service.ensure_ready(
        mode="ordinary",
        payload=payload,
        owner_factory=_factory,
        wait=True,
    )
    acquired = service.acquire_ready_owner(mode="ordinary", payload=payload)

    assert acquired is first

    refreshed = service.ensure_ready(
        mode="ordinary",
        payload=payload,
        owner_factory=_factory,
        wait=True,
    )

    assert refreshed is first
    assert owners == [first]
    assert service.ready_owner(mode="ordinary", payload=payload) is None

    service.release_owner(first)

    assert owners == [first]
    assert owners[0].close_calls == []
    assert service.ready_owner(mode="ordinary", payload=payload) is first


@pytest.mark.unit
def test_readiness_refresh_clears_active_owner_retirement_when_payload_becomes_desired_again():
    service = SimulationRuntimeApplication()
    first_payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=1"}}
    second_payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=2"}}
    owners: list[_Owner] = []

    def _factory(owner_payload):
        owner = _Owner(dict(owner_payload))
        owners.append(owner)
        return owner

    first = service.ensure_ready(
        mode="ordinary",
        payload=first_payload,
        owner_factory=_factory,
        wait=True,
    )
    acquired = service.acquire_ready_owner(mode="ordinary", payload=first_payload)

    assert acquired is first

    service.ensure_ready_many(
        mode="ordinary",
        payloads=[second_payload],
        owner_factory=_factory,
        wait=True,
    )
    service.ensure_ready_many(
        mode="ordinary",
        payloads=[first_payload],
        owner_factory=_factory,
        wait=True,
    )

    service.release_owner(first)

    assert len(owners) == 2
    assert first.close_calls == []
    assert service.ready_owner(mode="ordinary", payload=first_payload) is first


@pytest.mark.unit
def test_readiness_service_does_not_acquire_owner_until_warm_thread_finishes():
    service = SimulationRuntimeApplication()
    payload = {"execution_request": {"mechanism_text": "reaction: A -> B; k=1"}}
    start_entered = threading.Event()
    finish_start = threading.Event()
    owners: list[_Owner] = []

    class _ReadyBeforeReturnOwner(_Owner):
        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})
            self.ready = True
            start_entered.set()
            finish_start.wait(timeout=2.0)

    def _factory(owner_payload):
        owner = _ReadyBeforeReturnOwner(dict(owner_payload))
        owners.append(owner)
        return owner

    service.request_warm(mode="ordinary", payload=payload, owner_factory=_factory, wait=False)
    assert start_entered.wait(timeout=1.0)

    assert owners[0].is_ready is True
    assert service.ready_owner(mode="ordinary", payload=payload) is None
    assert service.acquire_ready_owner(mode="ordinary", payload=payload) is None
    assert owners[0].close_calls == []

    thread = service._slots["ordinary"].thread
    finish_start.set()
    assert thread is not None
    thread.join(timeout=2.0)

    acquired = service.acquire_ready_owner(mode="ordinary", payload=payload)

    assert acquired is owners[0]
    assert owners[0].close_calls == []
