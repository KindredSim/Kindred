from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable, Mapping, Optional


@dataclass(frozen=True)
class RuntimeReadinessSnapshot:
    mode: str
    status: str
    ready: bool
    generation: int
    failure: Optional[str] = None
    message: Optional[str] = None
    required: bool = True
    controls_ready: bool = False
    polling: bool = True

    @property
    def should_poll(self) -> bool:
        return bool(self.required and (not self.ready) and self.polling)


@dataclass
class _RuntimeSlot:
    payload: dict[str, object]
    owner: object
    generation: int
    mode_key: str
    status: str = "warming"
    failure: Optional[str] = None
    thread: Optional[threading.Thread] = None
    active_count: int = 0
    close_when_released: bool = False


class SimulationRuntimeApplication:
    """Owns exact warm serial simulation runtime slots without importing GUI code."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._slots: dict[str, _RuntimeSlot] = {}
        self._queue_slots: dict[str, list[_RuntimeSlot]] = {}
        self._active_slots: dict[int, _RuntimeSlot] = {}
        self._generation = 0

    def request_warm(
        self,
        *,
        mode: str,
        payload: Mapping[str, object],
        owner_factory: Callable[[Mapping[str, object]], object],
        wait: bool = False,
    ) -> Optional[object]:
        return self.ensure_ready(mode=mode, payload=payload, owner_factory=owner_factory, wait=wait)

    def ensure_ready(
        self,
        *,
        mode: str,
        payload: Mapping[str, object],
        owner_factory: Callable[[Mapping[str, object]], object],
        wait: bool = False,
    ) -> Optional[object]:
        payload_map = dict(payload or {})
        if not payload_map:
            return None
        mode_key = self._mode_key(mode)
        old_slot: Optional[_RuntimeSlot] = None
        stale_queue_slots: list[_RuntimeSlot] = []
        ready_owner: Optional[object] = None
        thread: Optional[threading.Thread] = None
        owner: Optional[object] = None
        with self._lock:
            queue_slots = self._queue_slots.pop(mode_key, [])
            if queue_slots:
                queue_slot = self._matching_slot(queue_slots, payload_map)
                if queue_slot is not None:
                    queue_slots.remove(queue_slot)
                    current_slot = self._slots.get(mode_key)
                    if current_slot is not None and current_slot.owner is not queue_slot.owner:
                        old_slot = current_slot
                    self._slots[mode_key] = queue_slot
                stale_queue_slots.extend(queue_slots)
            slot = self._slots.get(mode_key)
            if slot is not None and self._payloads_match(slot.payload, payload_map):
                if self._slot_ready(slot):
                    slot.status = "ready"
                    slot.failure = None
                    ready_owner = slot.owner
                elif slot.status == "warming":
                    thread = slot.thread
                    owner = slot.owner
                    if thread is None or not thread.is_alive():
                        old_slot = slot
                        del self._slots[mode_key]
                        thread = None
                        owner = None
                elif slot.status == "failed":
                    old_slot = slot
                    del self._slots[mode_key]
                    thread = None
                    owner = None
                else:
                    thread = None
                    owner = slot.owner
            else:
                active_match = self._active_matching_slot_locked(mode_key, payload_map)
                if active_match is not None:
                    active_match.close_when_released = False
                    ready_owner = active_match.owner
                    thread = active_match.thread
                    owner = active_match.owner
                elif slot is not None:
                    slot_thread = slot.thread
                    slot_warming = (
                        str(slot.status) == "warming"
                        and slot_thread is not None
                        and slot_thread.is_alive()
                    )
                    if slot_warming:
                        old_slot = slot
                    else:
                        old_slot = slot
                    owner = None
                    del self._slots[mode_key]
                else:
                    owner = None
                    thread = None
        if old_slot is not None:
            self._close_or_defer_retired_slot(old_slot, kill=False)
        for stale_slot in stale_queue_slots:
            self._close_or_defer_retired_slot(stale_slot, kill=False)
        if ready_owner is not None:
            return ready_owner
        if owner is None:
            owner = owner_factory(payload_map)
        if thread is None:
            with self._lock:
                self._generation += 1
                generation = int(self._generation)
                slot = _RuntimeSlot(
                    payload=dict(payload_map),
                    owner=owner,
                    generation=generation,
                    mode_key=mode_key,
                    status="warming",
                )
                self._slots[mode_key] = slot
                thread = threading.Thread(
                    target=self._warm_owner,
                    args=(mode_key, generation, owner, dict(payload_map)),
                    name=f"kindred-{mode_key}-runtime-readiness",
                    daemon=True,
                )
                slot.thread = thread
                thread.start()
        if wait and thread is not None:
            thread.join()
        return owner

    def ensure_ready_many(
        self,
        *,
        mode: str,
        payloads: list[Mapping[str, object]] | tuple[Mapping[str, object], ...],
        owner_factory: Callable[[Mapping[str, object]], object],
        wait: bool = False,
    ) -> list[object]:
        payload_maps = [dict(payload or {}) for payload in payloads or () if dict(payload or {})]
        if not payload_maps:
            return []
        mode_key = self._mode_key(mode)
        stale_slots: list[_RuntimeSlot] = []
        owners_and_threads: list[tuple[object, Optional[threading.Thread]]] = []
        with self._lock:
            existing_slots = list(self._queue_slots.get(mode_key, []))
            primary_slot = self._slots.pop(mode_key, None)
            if primary_slot is not None:
                existing_slots.append(primary_slot)
            next_slots: list[_RuntimeSlot] = []
            for payload_map in payload_maps:
                slot = self._matching_slot(existing_slots, payload_map)
                if slot is None:
                    active_match = self._active_matching_slot_locked(mode_key, payload_map)
                    if active_match is not None:
                        active_match.close_when_released = False
                        owners_and_threads.append((active_match.owner, active_match.thread))
                        continue
                if slot is None:
                    owner = owner_factory(payload_map)
                    self._generation += 1
                    generation = int(self._generation)
                    slot = _RuntimeSlot(
                        payload=dict(payload_map),
                        owner=owner,
                        generation=generation,
                        mode_key=mode_key,
                        status="warming",
                    )
                    thread = threading.Thread(
                        target=self._warm_owner,
                        args=(mode_key, generation, owner, dict(payload_map)),
                        name=f"kindred-{mode_key}-runtime-readiness",
                        daemon=True,
                    )
                    slot.thread = thread
                    thread.start()
                else:
                    existing_slots.remove(slot)
                    if self._slot_ready(slot):
                        slot.status = "ready"
                        slot.failure = None
                    elif slot.status == "failed":
                        stale_slots.append(slot)
                        owner = owner_factory(payload_map)
                        self._generation += 1
                        generation = int(self._generation)
                        slot = _RuntimeSlot(
                            payload=dict(payload_map),
                            owner=owner,
                            generation=generation,
                            mode_key=mode_key,
                            status="warming",
                        )
                        thread = threading.Thread(
                            target=self._warm_owner,
                            args=(mode_key, generation, owner, dict(payload_map)),
                            name=f"kindred-{mode_key}-runtime-readiness",
                            daemon=True,
                        )
                        slot.thread = thread
                        thread.start()
                    elif slot.status != "warming":
                        self._generation += 1
                        generation = int(self._generation)
                        thread = threading.Thread(
                            target=self._warm_owner,
                            args=(mode_key, generation, slot.owner, dict(payload_map)),
                            name=f"kindred-{mode_key}-runtime-readiness",
                            daemon=True,
                        )
                        slot.generation = generation
                        slot.status = "warming"
                        slot.failure = None
                        slot.thread = thread
                        thread.start()
                next_slots.append(slot)
                owners_and_threads.append((slot.owner, slot.thread))
            stale_slots.extend(existing_slots)
            self._queue_slots[mode_key] = next_slots
            for active_slot in self._active_slots.values():
                if str(active_slot.mode_key) != mode_key:
                    continue
                if not any(self._payloads_match(active_slot.payload, payload_map) for payload_map in payload_maps):
                    active_slot.close_when_released = True
        for slot in stale_slots:
            self._close_or_defer_retired_slot(slot, kill=False)
        if wait:
            current_thread = threading.current_thread()
            for _owner, thread in owners_and_threads:
                if thread is not None and thread is not current_thread:
                    thread.join()
        return [owner for owner, _thread in owners_and_threads]

    def current_owner(self, *, mode: str) -> Optional[object]:
        mode_key = self._mode_key(mode)
        with self._lock:
            slot = self._slots.get(mode_key)
            if slot is not None:
                return slot.owner
            queue_slots = self._queue_slots.get(mode_key, [])
            if not queue_slots:
                return None
            return queue_slots[0].owner

    def adopt_owner(
        self,
        *,
        mode: str,
        owner: Optional[object],
        payload: Mapping[str, object] | None = None,
    ) -> None:
        mode_key = self._mode_key(mode)
        with self._lock:
            if owner is None:
                self._slots.pop(mode_key, None)
                return
            payload_map = dict(payload or {})
            if not payload_map:
                owner_payload = getattr(owner, "simulation_plan_payload", None)
                if isinstance(owner_payload, Mapping):
                    payload_map = dict(owner_payload)
            self._generation += 1
            self._slots[mode_key] = _RuntimeSlot(
                payload=payload_map,
                owner=owner,
                generation=int(self._generation),
                mode_key=mode_key,
                status="ready" if self._owner_ready(owner) else "missing",
            )

    def detach_owner(self, *, mode: str) -> Optional[object]:
        mode_key = self._mode_key(mode)
        with self._lock:
            slot = self._slots.pop(mode_key, None)
            if slot is None:
                return None
            return slot.owner

    def ready_owner(self, *, mode: str, payload: Mapping[str, object]) -> Optional[object]:
        payload_map = dict(payload or {})
        if not payload_map:
            return None
        mode_key = self._mode_key(mode)
        with self._lock:
            slot = self._slots.get(mode_key)
            if slot is None or not self._payloads_match(slot.payload, payload_map):
                slot = self._matching_slot(self._queue_slots.get(mode_key, []), payload_map)
                if slot is None:
                    return None
            if self._slot_ready(slot):
                slot.status = "ready"
                slot.failure = None
                return slot.owner
            return None

    def acquire_ready_owner(self, *, mode: str, payload: Mapping[str, object]) -> Optional[object]:
        payload_map = dict(payload or {})
        if not payload_map:
            return None
        mode_key = self._mode_key(mode)
        with self._lock:
            slot = self._slots.get(mode_key)
            if slot is not None and self._payloads_match(slot.payload, payload_map):
                if self._slot_ready(slot) and int(slot.active_count) <= 0:
                    slot.status = "ready"
                    slot.failure = None
                    del self._slots[mode_key]
                    slot.active_count += 1
                    self._active_slots[id(slot.owner)] = slot
                    return slot.owner
                return None
            queue_slots = self._queue_slots.get(mode_key, [])
            slot = self._matching_slot(queue_slots, payload_map)
            if slot is None:
                return None
            if (not self._slot_ready(slot)) or int(slot.active_count) > 0:
                return None
            slot.status = "ready"
            slot.failure = None
            slot.active_count += 1
            self._active_slots[id(slot.owner)] = slot
            queue_slots.remove(slot)
            if queue_slots:
                self._queue_slots[mode_key] = queue_slots
            else:
                self._queue_slots.pop(mode_key, None)
            return slot.owner

    def release_owner(self, owner: object, *, kill: bool = False) -> None:
        if owner is None:
            return
        close_owner = False
        with self._lock:
            slot = self._active_slots.get(id(owner))
            if slot is None:
                return
            slot.active_count = max(0, int(slot.active_count) - 1)
            if slot.active_count <= 0:
                self._active_slots.pop(id(owner), None)
                close_owner = bool(slot.close_when_released)
                if not close_owner and self._owner_ready(slot.owner):
                    slot.close_when_released = False
                    existing = self._slots.get(slot.mode_key)
                    if existing is None:
                        self._slots[slot.mode_key] = slot
                    else:
                        queue_slots = self._queue_slots.setdefault(slot.mode_key, [])
                        if all(candidate.owner is not slot.owner for candidate in queue_slots):
                            queue_slots.append(slot)
        if close_owner:
            self._close_owner(owner, kill=bool(kill))

    def snapshot(self, *, mode: str) -> RuntimeReadinessSnapshot:
        mode_key = self._mode_key(mode)
        with self._lock:
            slot = self._slots.get(mode_key)
            if slot is None:
                queue_slots = self._queue_slots.get(mode_key, [])
                if not queue_slots:
                    return RuntimeReadinessSnapshot(
                        mode=mode_key,
                        status="missing",
                        ready=False,
                        generation=0,
                        controls_ready=False,
                        polling=True,
                    )
                ready_count = 0
                latest_generation = 0
                status = "missing"
                failure = None
                for queue_slot in queue_slots:
                    latest_generation = max(latest_generation, int(queue_slot.generation))
                    if self._slot_ready(queue_slot):
                        queue_slot.status = "ready"
                        queue_slot.failure = None
                        ready_count += 1
                    elif str(queue_slot.status) == "failed":
                        status = "failed"
                        failure = queue_slot.failure
                    elif status == "missing":
                        status = str(queue_slot.status)
                        failure = queue_slot.failure
                all_ready = ready_count == len(queue_slots)
                return RuntimeReadinessSnapshot(
                    mode=mode_key,
                    status="ready" if all_ready else status,
                    ready=bool(all_ready),
                    generation=int(latest_generation),
                    failure=failure,
                    controls_ready=bool(all_ready),
                    polling=str(status) not in {"failed"},
                )
            ready = self._slot_ready(slot)
            if ready:
                slot.status = "ready"
                slot.failure = None
            return RuntimeReadinessSnapshot(
                mode=mode_key,
                status=str(slot.status),
                ready=bool(ready),
                generation=int(slot.generation),
                failure=slot.failure,
                controls_ready=bool(ready),
                polling=str(slot.status) not in {"failed"},
            )

    def invalidate(self, *, mode: Optional[str] = None, kill: bool = False) -> None:
        self.close(mode=mode, kill=kill)

    def close(self, *, mode: Optional[str] = None, kill: bool = False) -> None:
        if mode is None:
            mode_keys = ["ordinary", "preview"]
        else:
            mode_keys = [self._mode_key(mode)]
        owners: list[object] = []
        threads: list[threading.Thread] = []
        seen_owner_ids: set[int] = set()
        with self._lock:
            for mode_key in mode_keys:
                slot = self._slots.pop(mode_key, None)
                if slot is not None:
                    if id(slot.owner) not in seen_owner_ids:
                        owners.append(slot.owner)
                        seen_owner_ids.add(id(slot.owner))
                    if slot.thread is not None:
                        threads.append(slot.thread)
                queue_slots = self._queue_slots.pop(mode_key, [])
                for queue_slot in queue_slots:
                    if id(queue_slot.owner) not in seen_owner_ids:
                        owners.append(queue_slot.owner)
                        seen_owner_ids.add(id(queue_slot.owner))
                    if queue_slot.thread is not None:
                        threads.append(queue_slot.thread)
                for owner_id, active_slot in list(self._active_slots.items()):
                    if str(active_slot.mode_key) != mode_key:
                        continue
                    self._active_slots.pop(owner_id, None)
                    if id(active_slot.owner) not in seen_owner_ids:
                        owners.append(active_slot.owner)
                        seen_owner_ids.add(id(active_slot.owner))
                    if active_slot.thread is not None:
                        threads.append(active_slot.thread)
        for owner in owners:
            self._close_owner(owner, kill=bool(kill))
        current_thread = threading.current_thread()
        for thread in threads:
            if thread is current_thread or not thread.is_alive():
                continue
            thread.join(timeout=2.0)

    def _warm_owner(
        self,
        mode_key: str,
        generation: int,
        owner: object,
        payload: Mapping[str, object],
    ) -> None:
        status = "warming"
        failure: Optional[str] = None
        try:
            prepare = getattr(owner, "prepare_runtime_payload", None)
            if callable(prepare):
                prepare(dict(payload))
            else:
                start = getattr(owner, "start", None)
                if callable(start):
                    start(wait=True)
            status = "ready" if self._owner_ready(owner) else "not_ready"
        except Exception as exc:
            status = "failed"
            failure = f"{type(exc).__name__}: {exc}"
        with self._lock:
            slot = self._slots.get(mode_key)
            queue_match = None
            if slot is None or slot.owner is not owner or int(slot.generation) != int(generation):
                for candidate in self._queue_slots.get(mode_key, []):
                    if candidate.owner is owner and int(candidate.generation) == int(generation):
                        queue_match = candidate
                        break
            active_match = self._active_slots.get(id(owner))
            if (
                active_match is not None
                and active_match.owner is owner
                and str(active_match.mode_key) == str(mode_key)
                and int(active_match.generation) == int(generation)
            ):
                active_match.status = status
                active_match.failure = failure
                return
            if (
                (slot is None or slot.owner is not owner or int(slot.generation) != int(generation))
                and queue_match is None
            ):
                return
            if queue_match is not None:
                slot = queue_match
            slot.status = status
            slot.failure = failure

    def _matching_slot(
        self,
        slots: list[_RuntimeSlot],
        payload: Mapping[str, object],
    ) -> Optional[_RuntimeSlot]:
        for slot in slots:
            if self._payloads_match(slot.payload, payload):
                return slot
        return None

    def _active_matching_slot_locked(
        self,
        mode_key: str,
        payload: Mapping[str, object],
    ) -> Optional[_RuntimeSlot]:
        for active_slot in self._active_slots.values():
            if str(active_slot.mode_key) == mode_key and self._payloads_match(active_slot.payload, payload):
                return active_slot
        return None

    def _payloads_match(self, left: Mapping[str, object], right: Mapping[str, object]) -> bool:
        try:
            from kindred.core.simulation_containment import contained_owner_payloads_match

            return bool(contained_owner_payloads_match(dict(left), dict(right)))
        except Exception:
            return False

    def _slot_ready(self, slot: _RuntimeSlot) -> bool:
        return bool(str(slot.status) == "ready" and self._owner_ready(slot.owner))

    @staticmethod
    def _owner_ready(owner: object) -> bool:
        try:
            return bool(getattr(owner, "is_ready", False))
        except Exception:
            return False

    @staticmethod
    def _close_owner(owner: object, *, kill: bool) -> None:
        close = getattr(owner, "close", None)
        if callable(close):
            try:
                close(kill=bool(kill))
            except TypeError:
                close()

    def _close_or_defer_retired_slot(self, slot: _RuntimeSlot, *, kill: bool) -> None:
        if int(getattr(slot, "active_count", 0) or 0) > 0:
            slot.close_when_released = True
            with self._lock:
                self._active_slots[id(slot.owner)] = slot
            return
        self._close_owner(slot.owner, kill=bool(kill))

    @staticmethod
    def _mode_key(mode: str) -> str:
        return "preview" if str(mode) == "preview" else "ordinary"
