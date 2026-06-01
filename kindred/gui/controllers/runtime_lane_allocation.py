from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RuntimeLaunchIntent:
    intent_kind: str
    ui_action: str
    rows: tuple[int, ...] = ()
    set_ids: tuple[str, ...] = ()
    requested_show_set_ids: tuple[str, ...] = ()
    requested_show_labels_by_set_id: Mapping[str, str] = field(default_factory=dict)
    request_token: int | None = None
    preview_request_id: int | None = None
    preview_epoch: int | None = None
    runtime_input_epochs: Mapping[str, int] = field(default_factory=dict)
    created_from_current_ui: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_kind", str(self.intent_kind or "ordinary"))
        object.__setattr__(self, "ui_action", str(self.ui_action or "run"))
        object.__setattr__(self, "rows", tuple(int(row) for row in self.rows or ()))
        object.__setattr__(
            self,
            "set_ids",
            tuple(str(set_id) for set_id in self.set_ids or () if str(set_id)),
        )
        object.__setattr__(
            self,
            "requested_show_set_ids",
            tuple(str(set_id) for set_id in self.requested_show_set_ids or () if str(set_id)),
        )
        object.__setattr__(
            self,
            "requested_show_labels_by_set_id",
            {
                str(set_id): str(label)
                for set_id, label in dict(self.requested_show_labels_by_set_id or {}).items()
                if str(set_id)
            },
        )
        object.__setattr__(
            self,
            "runtime_input_epochs",
            {
                str(key): int(value)
                for key, value in dict(self.runtime_input_epochs or {}).items()
                if str(key)
            },
        )


@dataclass(frozen=True)
class RuntimeCompatibilityKey:
    structural_digest: str
    runtime_parameter_names: tuple[str, ...] = ()
    execution_profile: str = "explicit"
    environment_key: str = ""
    schema_key: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "structural_digest", str(self.structural_digest or ""))
        object.__setattr__(
            self,
            "runtime_parameter_names",
            tuple(sorted({str(name) for name in self.runtime_parameter_names or () if str(name)})),
        )
        object.__setattr__(self, "execution_profile", str(self.execution_profile or "explicit"))
        object.__setattr__(self, "environment_key", str(self.environment_key or ""))
        object.__setattr__(self, "schema_key", str(self.schema_key or ""))

    def to_payload(self) -> dict[str, Any]:
        return {
            "structural_digest": str(self.structural_digest),
            "runtime_parameter_names": tuple(self.runtime_parameter_names),
            "execution_profile": str(self.execution_profile),
            "environment_key": str(self.environment_key),
            "schema_key": str(self.schema_key),
        }


@dataclass(frozen=True)
class RuntimePreparationBlockedReason:
    source: str
    code: str
    message: str
    rows: tuple[int, ...] = ()
    set_ids: tuple[str, ...] = ()
    retryable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", str(self.source or "preparation"))
        object.__setattr__(self, "code", str(self.code or "blocked"))
        object.__setattr__(self, "message", str(self.message or "Runtime request is blocked."))
        object.__setattr__(self, "rows", tuple(int(row) for row in self.rows or ()))
        object.__setattr__(
            self,
            "set_ids",
            tuple(str(set_id) for set_id in self.set_ids or () if str(set_id)),
        )


@dataclass(frozen=True)
class RuntimeTaskDescriptor:
    task_id: str
    row: int
    set_id: str
    request_token: int | None
    compatibility_key: RuntimeCompatibilityKey
    exact_descriptor_hash: str
    plan_payload: Mapping[str, Any] | None = None
    parameter_overrides: Mapping[str, Any] = field(default_factory=dict)
    initial_condition_overrides: Mapping[str, Any] = field(default_factory=dict)
    preview_request_id: int | None = None
    preview_epoch: int | None = None
    runtime_input_epochs: Mapping[str, int] = field(default_factory=dict)
    cache_key: str = ""
    set_label: str = ""
    mechanism_text: str = ""
    mechanism_signature: str = ""
    simulation_identity: Mapping[str, Any] = field(default_factory=dict)
    owned_species: tuple[str, ...] = ()
    preview_batch_cache_token: str = ""
    run_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", str(self.task_id or ""))
        object.__setattr__(self, "row", int(self.row))
        object.__setattr__(self, "set_id", str(self.set_id or ""))
        object.__setattr__(self, "exact_descriptor_hash", str(self.exact_descriptor_hash or ""))
        object.__setattr__(self, "parameter_overrides", dict(self.parameter_overrides or {}))
        object.__setattr__(self, "initial_condition_overrides", dict(self.initial_condition_overrides or {}))
        object.__setattr__(
            self,
            "runtime_input_epochs",
            {
                str(key): int(value)
                for key, value in dict(self.runtime_input_epochs or {}).items()
                if str(key)
            },
        )
        object.__setattr__(self, "cache_key", str(self.cache_key or ""))
        object.__setattr__(self, "set_label", str(self.set_label or ""))
        object.__setattr__(self, "mechanism_text", str(self.mechanism_text or ""))
        object.__setattr__(self, "mechanism_signature", str(self.mechanism_signature or ""))
        object.__setattr__(self, "simulation_identity", dict(self.simulation_identity or {}))
        object.__setattr__(
            self,
            "owned_species",
            tuple(str(name) for name in self.owned_species or () if str(name)),
        )
        object.__setattr__(self, "preview_batch_cache_token", str(self.preview_batch_cache_token or ""))


@dataclass(frozen=True)
class PreparedRuntimeRequestSet:
    intent: RuntimeLaunchIntent
    compatibility_key: RuntimeCompatibilityKey
    task_descriptors: tuple[RuntimeTaskDescriptor, ...] = ()
    required_lane_capacity: int = 1
    preferred_lane_capacity: int | None = None
    blocked_reason: RuntimePreparationBlockedReason | None = None

    def __post_init__(self) -> None:
        descriptors = tuple(self.task_descriptors or ())
        object.__setattr__(self, "task_descriptors", descriptors)
        required = max(1, int(self.required_lane_capacity or 1))
        preferred = self.preferred_lane_capacity
        if preferred is None:
            preferred = max(required, len(descriptors) or required)
        object.__setattr__(self, "required_lane_capacity", required)
        object.__setattr__(self, "preferred_lane_capacity", max(required, int(preferred or required)))

    @property
    def prepared(self) -> bool:
        return self.blocked_reason is None and bool(self.task_descriptors)

    @property
    def descriptor_hashes(self) -> frozenset[str]:
        return frozenset(str(descriptor.exact_descriptor_hash) for descriptor in self.task_descriptors)


@dataclass(frozen=True)
class RuntimeLaneAllocationRequest:
    compatibility_key: RuntimeCompatibilityKey
    required_lane_capacity: int
    preferred_lane_capacity: int
    task_count: int
    request_token: int | None
    scope: str
    queue_policy: str = "queue_tasks"
    nonblocking: bool = True

    def __post_init__(self) -> None:
        required = max(1, int(self.required_lane_capacity or 1))
        preferred = max(required, int(self.preferred_lane_capacity or required))
        object.__setattr__(self, "required_lane_capacity", required)
        object.__setattr__(self, "preferred_lane_capacity", preferred)
        object.__setattr__(self, "task_count", max(0, int(self.task_count or 0)))
        object.__setattr__(self, "scope", str(self.scope or "ordinary"))
        object.__setattr__(self, "queue_policy", str(self.queue_policy or "queue_tasks"))


@dataclass(frozen=True)
class RuntimeLane:
    lane_id: str
    compatibility_key: RuntimeCompatibilityKey
    generation: int = 0
    state: str = "ready"
    backend_pool_token: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane_id", str(self.lane_id or ""))
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "state", str(self.state or "ready"))
        object.__setattr__(self, "backend_pool_token", str(self.backend_pool_token or ""))


@dataclass(frozen=True)
class RuntimeLaneReservation:
    allocation_id: str
    compatibility_key: RuntimeCompatibilityKey
    lanes: tuple[RuntimeLane, ...]
    release_token: str
    state: str = "reserved"


@dataclass(frozen=True)
class RuntimeLaunchAllocation:
    allocation_id: str
    status: str
    launch_intent: RuntimeLaunchIntent | None
    prepared_request_set: PreparedRuntimeRequestSet | None
    reservation: RuntimeLaneReservation
    accepted_capacity: int
    task_count: int
    descriptor_hashes: frozenset[str]
    message: str = ""
    retryable: bool = True
    retain_lanes_after_success: bool = True


@dataclass(frozen=True)
class RuntimeDispatchPlan:
    launch_allocation: RuntimeLaunchAllocation
    ordered_task_descriptors: tuple[RuntimeTaskDescriptor, ...]
    lane_assignments: tuple[tuple[str, str], ...]
    release_token: str

    def assignment_for_task(self, task_id: str) -> "RuntimeTaskLaneAssignment | None":
        task_key = str(task_id or "")
        if not task_key:
            return None
        lane_id_by_task = {
            str(assigned_task_id): str(lane_id)
            for assigned_task_id, lane_id in self.lane_assignments
            if str(assigned_task_id) and str(lane_id)
        }
        lane_id = lane_id_by_task.get(task_key)
        if not lane_id:
            return None
        for lane in self.launch_allocation.reservation.lanes:
            if str(lane.lane_id) != str(lane_id):
                continue
            return RuntimeTaskLaneAssignment(
                task_id=task_key,
                lane_id=str(lane.lane_id),
                lane_generation=int(lane.generation),
                compatibility_key=lane.compatibility_key,
            )
        return RuntimeTaskLaneAssignment(
            task_id=task_key,
            lane_id=str(lane_id),
            lane_generation=0,
            compatibility_key=self.launch_allocation.reservation.compatibility_key,
        )


@dataclass(frozen=True)
class RuntimeTaskLaneAssignment:
    task_id: str
    lane_id: str
    lane_generation: int
    compatibility_key: RuntimeCompatibilityKey

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", str(self.task_id or ""))
        object.__setattr__(self, "lane_id", str(self.lane_id or ""))
        object.__setattr__(self, "lane_generation", int(self.lane_generation))


@dataclass(frozen=True)
class RuntimeAllocationConsumeResult:
    status: str
    dispatch_plan: RuntimeDispatchPlan | None = None
    message: str = ""
    retryable: bool = True


class RuntimeLaneAllocator:
    def __init__(
        self,
        *,
        lane_warmer: Callable[..., Sequence[tuple[str, int]] | int | None] | None = None,
        backend_lane_is_live: Callable[[RuntimeLane], bool] | None = None,
    ) -> None:
        self._allocation_counter = count(1)
        self._lanes: dict[str, RuntimeLane] = {}
        self._allocations: dict[str, RuntimeLaunchAllocation] = {}
        self._released: set[str] = set()
        self._lane_warmer = lane_warmer
        self._backend_lane_is_live = backend_lane_is_live

    def register_ready_lane(
        self,
        *,
        compatibility_key: RuntimeCompatibilityKey,
        lane_id: str,
        generation: int = 0,
        backend_pool_token: str = "",
    ) -> RuntimeLane:
        lane_key = str(lane_id or "")
        current = self._lanes.get(lane_key)
        if current is not None and str(current.state) != "ready":
            return current
        lane = RuntimeLane(
            lane_id=lane_key,
            compatibility_key=compatibility_key,
            generation=int(generation),
            state="ready",
            backend_pool_token=str(backend_pool_token or ""),
        )
        self._lanes[lane.lane_id] = lane
        return lane

    def ensure_ready_lanes(
        self,
        *,
        compatibility_key: RuntimeCompatibilityKey,
        capacity: int,
        task_count: int = 0,
        nonblocking: bool = True,
    ) -> tuple[RuntimeLane, ...]:
        capacity_i = max(1, int(capacity or 1))
        ready = self._ready_lanes(compatibility_key)
        missing = max(0, capacity_i - len(ready))
        if missing > 0:
            self._warm_runtime_lanes(
                compatibility_key=compatibility_key,
                capacity=capacity_i,
                wait=not bool(nonblocking),
            )
        return self._ready_lanes(compatibility_key)

    def allocate(self, request: RuntimeLaneAllocationRequest) -> RuntimeLaunchAllocation:
        ready = self._ready_lanes(request.compatibility_key)
        accepted = min(len(ready), int(request.preferred_lane_capacity))
        allocation_id = f"alloc-{next(self._allocation_counter)}"
        selected = tuple(ready[:accepted])
        if accepted < int(request.required_lane_capacity):
            reservation = RuntimeLaneReservation(
                allocation_id=allocation_id,
                compatibility_key=request.compatibility_key,
                lanes=(),
                release_token=f"{allocation_id}:release",
                state="waiting",
            )
            allocation = RuntimeLaunchAllocation(
                allocation_id=allocation_id,
                status="waiting",
                launch_intent=None,
                prepared_request_set=None,
                reservation=reservation,
                accepted_capacity=0,
                task_count=int(request.task_count),
                descriptor_hashes=frozenset(),
                message="Compatible runtime lanes are not ready.",
            )
            self._allocations[allocation_id] = allocation
            return allocation
        for lane in selected:
            self._lanes[lane.lane_id] = RuntimeLane(
                lane_id=lane.lane_id,
                compatibility_key=lane.compatibility_key,
                generation=lane.generation,
                state="reserved",
                backend_pool_token=lane.backend_pool_token,
            )
        reservation = RuntimeLaneReservation(
            allocation_id=allocation_id,
            compatibility_key=request.compatibility_key,
            lanes=selected,
            release_token=f"{allocation_id}:release",
            state="reserved",
        )
        allocation = RuntimeLaunchAllocation(
            allocation_id=allocation_id,
            status="ready",
            launch_intent=None,
            prepared_request_set=None,
            reservation=reservation,
            accepted_capacity=accepted,
            task_count=int(request.task_count),
            descriptor_hashes=frozenset(),
        )
        self._allocations[allocation_id] = allocation
        return allocation

    def consume(
        self,
        allocation: RuntimeLaunchAllocation,
        prepared: PreparedRuntimeRequestSet,
        *,
        expected: PreparedRuntimeRequestSet | None = None,
    ) -> RuntimeAllocationConsumeResult:
        if allocation.status != "ready":
            return RuntimeAllocationConsumeResult(
                status=str(allocation.status or "not_ready"),
                message=allocation.message,
                retryable=allocation.retryable,
            )
        if allocation.reservation.compatibility_key != prepared.compatibility_key:
            return RuntimeAllocationConsumeResult(
                status="incompatible",
                message="Prepared request compatibility changed after allocation.",
            )
        if expected is not None and expected.descriptor_hashes != prepared.descriptor_hashes:
            return RuntimeAllocationConsumeResult(
                status="stale_descriptor",
                message="Prepared request descriptors changed after allocation.",
            )
        if not prepared.prepared:
            reason = prepared.blocked_reason
            return RuntimeAllocationConsumeResult(
                status="blocked",
                message=str(reason.message if reason is not None else "Prepared request is blocked."),
                retryable=bool(reason.retryable) if reason is not None else False,
            )
        selected_lane_ids = [lane.lane_id for lane in allocation.reservation.lanes]
        if not selected_lane_ids:
            return RuntimeAllocationConsumeResult(
                status="waiting",
                message="No runtime lanes are reserved for dispatch.",
            )
        assignments: list[tuple[str, str]] = []
        for index, descriptor in enumerate(prepared.task_descriptors):
            lane_id = selected_lane_ids[index % len(selected_lane_ids)]
            assignments.append((str(descriptor.task_id), lane_id))
        launch_allocation = RuntimeLaunchAllocation(
            allocation_id=allocation.allocation_id,
            status="ready",
            launch_intent=prepared.intent,
            prepared_request_set=prepared,
            reservation=allocation.reservation,
            accepted_capacity=allocation.accepted_capacity,
            task_count=len(prepared.task_descriptors),
            descriptor_hashes=prepared.descriptor_hashes,
            retain_lanes_after_success=self._retain_after_success(prepared.intent),
        )
        self._allocations[allocation.allocation_id] = launch_allocation
        return RuntimeAllocationConsumeResult(
            status="ready",
            dispatch_plan=RuntimeDispatchPlan(
                launch_allocation=launch_allocation,
                ordered_task_descriptors=prepared.task_descriptors,
                lane_assignments=tuple(assignments),
                release_token=allocation.reservation.release_token,
            ),
        )

    def release(self, allocation_id: str, *, failed: bool = False) -> bool:
        allocation_key = str(allocation_id or "")
        if not allocation_key or allocation_key in self._released:
            return False
        allocation = self._allocations.get(allocation_key)
        self._released.add(allocation_key)
        if allocation is None:
            return False
        reservation_live = self._reservation_lanes_live(allocation.reservation.lanes)
        reusable = (
            not bool(failed)
            and bool(allocation.retain_lanes_after_success)
            and reservation_live
        )
        next_state = "ready" if reusable else "failed" if bool(failed) or not reservation_live else "released"
        for lane in allocation.reservation.lanes:
            current = self._lanes.get(lane.lane_id, lane)
            self._lanes[lane.lane_id] = RuntimeLane(
                lane_id=current.lane_id,
                compatibility_key=current.compatibility_key,
                generation=current.generation,
                state=next_state,
                backend_pool_token=str(current.backend_pool_token or lane.backend_pool_token),
            )
        return True

    def release_all(self, *, failed: bool = False) -> None:
        for allocation_id in list(self._allocations):
            self.release(allocation_id, failed=failed)
        self._lanes.clear()
        self._allocations.clear()

    def _warm_runtime_lanes(
        self,
        *,
        compatibility_key: RuntimeCompatibilityKey,
        capacity: int,
        wait: bool,
    ) -> None:
        requested = max(1, int(capacity or 1))
        if self._lane_warmer is None:
            warmed = requested
        else:
            try:
                warmed = self._lane_warmer(requested, wait=bool(wait))
            except TypeError:
                warmed = self._lane_warmer(requested)
        if warmed is None:
            warmed_records: Sequence[tuple[str, int]] = ()
        elif isinstance(warmed, int):
            warmed_records = tuple((f"runtime-lane-{index + 1}", 1, "") for index in range(max(0, warmed)))
        else:
            warmed_records = tuple(warmed)
        for index, record in enumerate(warmed_records):
            backend_pool_token = ""
            if isinstance(record, RuntimeLane):
                lane_id = record.lane_id
                generation = record.generation
                backend_pool_token = record.backend_pool_token
            else:
                try:
                    lane_id, generation, backend_pool_token = record
                except Exception:
                    try:
                        lane_id, generation = record
                    except Exception:
                        lane_id, generation = f"runtime-lane-{index + 1}", 1
            self.register_ready_lane(
                compatibility_key=compatibility_key,
                lane_id=str(lane_id),
                generation=int(generation),
                backend_pool_token=str(backend_pool_token or ""),
            )

    def _ready_lanes(
        self,
        compatibility_key: RuntimeCompatibilityKey,
    ) -> tuple[RuntimeLane, ...]:
        ready: list[RuntimeLane] = []
        for lane in tuple(self._lanes.values()):
            if lane.compatibility_key != compatibility_key or lane.state != "ready":
                continue
            if not self._lane_is_live(lane):
                self._lanes[lane.lane_id] = RuntimeLane(
                    lane_id=lane.lane_id,
                    compatibility_key=lane.compatibility_key,
                    generation=lane.generation,
                    state="failed",
                    backend_pool_token=lane.backend_pool_token,
                )
                continue
            ready.append(lane)
        return tuple(ready)

    def _reservation_lanes_live(self, lanes: Sequence[RuntimeLane]) -> bool:
        return bool(lanes) and all(self._lane_is_live(lane) for lane in lanes)

    def _lane_is_live(self, lane: RuntimeLane) -> bool:
        provider = self._backend_lane_is_live
        if provider is None:
            return True
        try:
            return bool(provider(lane))
        except Exception:
            return False

    @staticmethod
    def _retain_after_success(intent: RuntimeLaunchIntent | None) -> bool:
        if intent is None:
            return True
        return str(intent.intent_kind or "ordinary") in {"ordinary", "run", "run_selected", "preview"}
