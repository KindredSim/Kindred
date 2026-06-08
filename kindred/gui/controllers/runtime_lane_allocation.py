from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Any, Mapping, Protocol, Sequence

from kindred.gui.controllers.preview_target_identity import normalize_preview_target_set_ids


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
            normalize_preview_target_set_ids(self.set_ids),
        )
        object.__setattr__(
            self,
            "requested_show_set_ids",
            normalize_preview_target_set_ids(self.requested_show_set_ids),
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


class RuntimeReleaseReason(Enum):
    SUCCESS_RETAIN = "success_retain"
    NEUTRAL_RETAIN = "neutral_retain"
    SUPERSEDED = "superseded"
    SHUTDOWN = "shutdown"
    FAILURE = "failure"


@dataclass(frozen=True)
class RuntimeReleaseResult:
    status: str
    allocation_id: str = ""
    reason: RuntimeReleaseReason | None = None

    @property
    def released(self) -> bool:
        return self.status in {"released", "already_released"}


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
            normalize_preview_target_set_ids(self.set_ids),
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
    require_backend_lease: bool = False

    def __post_init__(self) -> None:
        required = max(1, int(self.required_lane_capacity or 1))
        preferred = max(required, int(self.preferred_lane_capacity or required))
        object.__setattr__(self, "required_lane_capacity", required)
        object.__setattr__(self, "preferred_lane_capacity", preferred)
        object.__setattr__(self, "task_count", max(0, int(self.task_count or 0)))
        object.__setattr__(self, "scope", str(self.scope or "ordinary"))
        object.__setattr__(self, "queue_policy", str(self.queue_policy or "queue_tasks"))
        object.__setattr__(self, "require_backend_lease", bool(self.require_backend_lease))


@dataclass(frozen=True)
class RuntimeLaneReadinessProbeResult:
    status: str
    ready_capacity: int = 0
    required_capacity: int = 1
    message: str = ""
    retryable: bool = True

    @property
    def ready(self) -> bool:
        return str(self.status or "") == "ready"


@dataclass(frozen=True)
class RuntimeLane:
    lane_id: str
    compatibility_key: RuntimeCompatibilityKey
    generation: int = 0
    state: str = "ready"
    backend_pool_token: str = ""
    backend_lease_id: str = ""
    backend_generation: int = 0
    backend_lease_capacity: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane_id", str(self.lane_id or ""))
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "state", str(self.state or "ready"))
        object.__setattr__(self, "backend_pool_token", str(self.backend_pool_token or ""))
        object.__setattr__(self, "backend_lease_id", str(self.backend_lease_id or ""))
        object.__setattr__(
            self,
            "backend_generation",
            int(self.backend_generation or 0),
        )
        object.__setattr__(
            self,
            "backend_lease_capacity",
            max(0, int(self.backend_lease_capacity or 0)),
        )


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
        return None


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
class RuntimeBackendLease:
    lease_id: str
    pool_token: str
    generation: int
    compatibility_key: RuntimeCompatibilityKey
    capacity: int
    state: str = "live"

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", str(self.lease_id or ""))
        object.__setattr__(self, "pool_token", str(self.pool_token or ""))
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "capacity", max(1, int(self.capacity or 1)))
        object.__setattr__(self, "state", str(self.state or "live"))


@dataclass(frozen=True)
class RuntimeBackendTask:
    descriptor: RuntimeTaskDescriptor
    dispatch_plan_id: str
    allocation_id: str
    release_token: str
    lane_assignment: RuntimeTaskLaneAssignment | None
    backend_lease: RuntimeBackendLease

    def __post_init__(self) -> None:
        object.__setattr__(self, "dispatch_plan_id", str(self.dispatch_plan_id or ""))
        object.__setattr__(self, "allocation_id", str(self.allocation_id or ""))
        object.__setattr__(self, "release_token", str(self.release_token or ""))

    def request_metadata(self) -> dict[str, object]:
        intent_kind = "preview" if self.descriptor.preview_request_id is not None else "ordinary"
        metadata: dict[str, object] = {
            "runtime_task_id": self.descriptor.task_id,
            "runtime_descriptor_hash": self.descriptor.exact_descriptor_hash,
            "runtime_row": int(self.descriptor.row),
            "runtime_set_id": self.descriptor.set_id,
            "runtime_intent_kind": intent_kind,
            "runtime_request_token": int(self.descriptor.request_token or 0),
            "runtime_allocation_id": self.allocation_id,
            "runtime_release_token": self.release_token,
            "runtime_lease_id": self.backend_lease.lease_id,
            "runtime_pool_token": self.backend_lease.pool_token,
            "runtime_pool_generation": int(self.backend_lease.generation),
            "runtime_lease_capacity": int(self.backend_lease.capacity),
        }
        if self.lane_assignment is not None:
            metadata["runtime_lane_id"] = self.lane_assignment.lane_id
            metadata["runtime_lane_generation"] = int(self.lane_assignment.lane_generation)
        if self.descriptor.preview_request_id is not None:
            metadata["runtime_preview_request_id"] = int(self.descriptor.preview_request_id)
        if self.descriptor.preview_epoch is not None:
            metadata["runtime_preview_epoch"] = int(self.descriptor.preview_epoch)
        return metadata

    def executable_payload(self, *, run_id: int, set_name: str) -> dict[str, object]:
        return {
            "run_id": int(run_id),
            "request_id": int(self.descriptor.request_token or 0),
            "set_id": str(self.descriptor.set_id),
            "set_name": str(set_name),
            "include_mechanism_in_result_payload": True,
            "simulation_plan": dict(self.descriptor.plan_payload or {}),
        }


class RuntimeBackendLeaseProvider(Protocol):
    def ensure_backend_lease(
        self,
        compatibility_key: RuntimeCompatibilityKey,
        capacity: int,
        *,
        wait: bool,
    ) -> RuntimeBackendLease | None: ...

    def invalidate_backend_lease(
        self,
        lease: RuntimeBackendLease | None,
        *,
        reason: RuntimeReleaseReason,
    ) -> None: ...


@dataclass(frozen=True)
class RuntimeAllocationConsumeResult:
    status: str
    dispatch_plan: RuntimeDispatchPlan | None = None
    message: str = ""
    retryable: bool = True


@dataclass(frozen=True)
class _RuntimeLaneIdentity:
    compatibility_key: RuntimeCompatibilityKey
    backend_lease_id: str
    backend_pool_token: str
    backend_generation: int
    slot_index: int


class RuntimeLaneAllocator:
    def __init__(
        self,
        *,
        backend_lease_provider: RuntimeBackendLeaseProvider | None = None,
    ) -> None:
        self._allocation_counter = count(1)
        self._lane_id_counter = count(1)
        self._lanes: dict[str, RuntimeLane] = {}
        self._lane_ids_by_identity: dict[_RuntimeLaneIdentity, str] = {}
        self._allocations: dict[str, RuntimeLaunchAllocation] = {}
        self._released: set[str] = set()
        self._backend_lease_provider = backend_lease_provider

    def register_ready_lane(
        self,
        *,
        compatibility_key: RuntimeCompatibilityKey,
        backend_slot: int = 0,
        generation: int = 0,
        backend_pool_token: str = "",
        backend_lease_id: str = "",
        backend_generation: int = 0,
        backend_lease_capacity: int = 0,
    ) -> RuntimeLane:
        lane_key = self._lane_id_for_identity(
            _RuntimeLaneIdentity(
                compatibility_key=compatibility_key,
                backend_lease_id=str(backend_lease_id or ""),
                backend_pool_token=str(backend_pool_token or ""),
                backend_generation=int(backend_generation or generation or 0),
                slot_index=max(0, int(backend_slot or 0)),
            )
        )
        current = self._lanes.get(lane_key)
        if current is not None and current.state == "reserved":
            return current
        lane = RuntimeLane(
            lane_id=lane_key,
            compatibility_key=compatibility_key,
            generation=int(generation),
            state="ready",
            backend_pool_token=str(backend_pool_token or ""),
            backend_lease_id=str(backend_lease_id or ""),
            backend_generation=int(backend_generation or 0),
            backend_lease_capacity=max(0, int(backend_lease_capacity or 0)),
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

    def ready_lanes(
        self,
        compatibility_key: RuntimeCompatibilityKey,
        *,
        require_backend_lease: bool = False,
    ) -> tuple[RuntimeLane, ...]:
        return self._ready_lanes(
            compatibility_key,
            require_backend_lease=bool(require_backend_lease),
        )

    def probe_readiness(
        self,
        request: RuntimeLaneAllocationRequest,
    ) -> RuntimeLaneReadinessProbeResult:
        ready = self._ready_lanes(
            request.compatibility_key,
            require_backend_lease=bool(request.require_backend_lease),
        )
        ready_capacity = min(len(ready), int(request.preferred_lane_capacity))
        required = max(1, int(request.required_lane_capacity or 1))
        if ready_capacity < required:
            return RuntimeLaneReadinessProbeResult(
                status="waiting",
                ready_capacity=ready_capacity,
                required_capacity=required,
                message="Compatible runtime lanes are not ready.",
            )
        return RuntimeLaneReadinessProbeResult(
            status="ready",
            ready_capacity=ready_capacity,
            required_capacity=required,
        )

    def allocate(self, request: RuntimeLaneAllocationRequest) -> RuntimeLaunchAllocation:
        ready = self._ready_lanes(
            request.compatibility_key,
            require_backend_lease=bool(request.require_backend_lease),
        )
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
                backend_lease_id=lane.backend_lease_id,
                backend_generation=lane.backend_generation,
                backend_lease_capacity=lane.backend_lease_capacity,
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
            self.release(allocation.allocation_id, reason=RuntimeReleaseReason.NEUTRAL_RETAIN)
            return RuntimeAllocationConsumeResult(
                status="incompatible",
                message="Prepared request compatibility changed after allocation.",
            )
        if expected is not None and expected.descriptor_hashes != prepared.descriptor_hashes:
            self.release(allocation.allocation_id, reason=RuntimeReleaseReason.NEUTRAL_RETAIN)
            return RuntimeAllocationConsumeResult(
                status="stale_descriptor",
                message="Prepared request descriptors changed after allocation.",
            )
        if not prepared.prepared:
            reason = prepared.blocked_reason
            self.release(allocation.allocation_id, reason=RuntimeReleaseReason.NEUTRAL_RETAIN)
            return RuntimeAllocationConsumeResult(
                status="blocked",
                message=str(reason.message if reason is not None else "Prepared request is blocked."),
                retryable=bool(reason.retryable) if reason is not None else False,
            )
        selected_lane_ids = [lane.lane_id for lane in allocation.reservation.lanes]
        if not selected_lane_ids:
            self.release(allocation.allocation_id, reason=RuntimeReleaseReason.NEUTRAL_RETAIN)
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

    def release(
        self,
        allocation_id: str,
        *,
        reason: RuntimeReleaseReason,
        backend_failure: bool = False,
    ) -> RuntimeReleaseResult:
        return self._release_with_reason(
            allocation_id,
            reason=reason,
            backend_failure=bool(backend_failure),
        )

    def clear_backend_pool(
        self,
        *,
        pool_token: str,
        generation: int = 0,
        reason: RuntimeReleaseReason = RuntimeReleaseReason.SHUTDOWN,
    ) -> int:
        token = str(pool_token or "")
        if not token:
            return 0
        generation_i = int(generation or 0)
        next_state = "failed" if reason is RuntimeReleaseReason.FAILURE else "released"
        cleared = 0
        for lane in tuple(self._lanes.values()):
            if str(lane.backend_pool_token or "") != token:
                continue
            if generation_i > 0 and int(lane.backend_generation or 0) != generation_i:
                continue
            self._lanes[lane.lane_id] = RuntimeLane(
                lane_id=lane.lane_id,
                compatibility_key=lane.compatibility_key,
                generation=lane.generation,
                state=next_state,
            )
            cleared += 1
        return cleared

    def _release_with_reason(
        self,
        allocation_id: str,
        *,
        reason: RuntimeReleaseReason,
        backend_failure: bool = False,
    ) -> RuntimeReleaseResult:
        if not isinstance(reason, RuntimeReleaseReason):
            raise TypeError("Runtime release requires RuntimeReleaseReason.")
        allocation_key = str(allocation_id or "")
        if not allocation_key or allocation_key in self._released:
            return RuntimeReleaseResult(
                status="already_released" if allocation_key in self._released else "missing",
                allocation_id=allocation_key,
                reason=reason,
            )
        allocation = self._allocations.get(allocation_key)
        if allocation is None:
            return RuntimeReleaseResult(
                status="missing",
                allocation_id=allocation_key,
                reason=reason,
            )
        reservation_live = self._reservation_lanes_live(allocation.reservation.lanes)
        reusable = (
            reason in {
                RuntimeReleaseReason.SUCCESS_RETAIN,
                RuntimeReleaseReason.NEUTRAL_RETAIN,
                RuntimeReleaseReason.SUPERSEDED,
            }
            and bool(allocation.retain_lanes_after_success)
            and reservation_live
        )
        next_state = (
            "ready"
            if reusable
            else "failed"
            if reason is RuntimeReleaseReason.FAILURE or not reservation_live
            else "released"
        )
        invalidated_leases: set[tuple[str, str, int]] = set()
        if bool(backend_failure):
            for lane in allocation.reservation.lanes:
                current = self._lanes.get(lane.lane_id, lane)
                lease = self._backend_lease_from_lane(current)
                if lease is None:
                    lease = self._backend_lease_from_lane(lane)
                if lease is None:
                    continue
                lease_key = (lease.lease_id, lease.pool_token, int(lease.generation))
                if lease_key in invalidated_leases:
                    continue
                invalidated_leases.add(lease_key)
                provider = self._backend_lease_provider
                if provider is not None:
                    provider.invalidate_backend_lease(
                        lease,
                        reason=RuntimeReleaseReason.FAILURE,
                    )
        self._released.add(allocation_key)
        for lane in allocation.reservation.lanes:
            current = self._lanes.get(lane.lane_id, lane)
            self._lanes[lane.lane_id] = RuntimeLane(
                lane_id=current.lane_id,
                compatibility_key=current.compatibility_key,
                generation=current.generation,
                state=next_state,
                backend_pool_token=str(current.backend_pool_token or lane.backend_pool_token),
                backend_lease_id=str(current.backend_lease_id or lane.backend_lease_id),
                backend_generation=int(current.backend_generation or 0),
                backend_lease_capacity=max(0, int(current.backend_lease_capacity or lane.backend_lease_capacity or 0)),
            )
        return RuntimeReleaseResult(
            status="released",
            allocation_id=allocation_key,
            reason=reason,
        )


    def _warm_runtime_lanes(
        self,
        *,
        compatibility_key: RuntimeCompatibilityKey,
        capacity: int,
        wait: bool,
    ) -> None:
        requested = max(1, int(capacity or 1))
        provider = self._backend_lease_provider
        if provider is None:
            return
        lease = provider.ensure_backend_lease(
            compatibility_key,
            requested,
            wait=bool(wait),
        )
        if lease is None:
            return
        for index in range(max(1, int(lease.capacity or requested))):
            self.register_ready_lane(
                compatibility_key=compatibility_key,
                backend_slot=index,
                generation=int(lease.generation),
                backend_pool_token=str(lease.pool_token),
                backend_lease_id=str(lease.lease_id),
                backend_generation=int(lease.generation),
                backend_lease_capacity=int(lease.capacity),
            )

    def _ready_lanes(
        self,
        compatibility_key: RuntimeCompatibilityKey,
        *,
        require_backend_lease: bool = False,
    ) -> tuple[RuntimeLane, ...]:
        ready: list[RuntimeLane] = []
        for lane in tuple(self._lanes.values()):
            if lane.compatibility_key != compatibility_key or lane.state != "ready":
                continue
            if bool(require_backend_lease) and (
                not str(lane.backend_lease_id or "")
                or int(lane.backend_generation or 0) <= 0
                or int(lane.backend_lease_capacity or 0) <= 0
            ):
                continue
            ready.append(lane)
        return tuple(ready)

    def _reservation_lanes_live(self, lanes: Sequence[RuntimeLane]) -> bool:
        if not lanes:
            return False
        for lane in lanes:
            current = self._lanes.get(lane.lane_id, lane)
            if str(current.state or "") not in {"reserved", "ready"}:
                return False
            if str(current.backend_lease_id or "") and self._backend_lease_from_lane(current) is None:
                return False
        return True

    @staticmethod
    def _backend_lease_from_lane(
        lane: RuntimeLane | None,
    ) -> RuntimeBackendLease | None:
        if lane is None or not str(lane.backend_lease_id or ""):
            return None
        generation = int(lane.backend_generation or 0)
        if generation <= 0:
            return None
        capacity = int(lane.backend_lease_capacity or 0)
        if capacity <= 0:
            return None
        return RuntimeBackendLease(
            lease_id=str(lane.backend_lease_id),
            pool_token=str(lane.backend_pool_token),
            generation=generation,
            compatibility_key=lane.compatibility_key,
            capacity=capacity,
            state="live",
        )

    @staticmethod
    def _retain_after_success(intent: RuntimeLaunchIntent | None) -> bool:
        if intent is None:
            return True
        return str(intent.intent_kind or "ordinary") in {"ordinary", "run", "run_selected", "preview"}

    def _lane_id_for_identity(self, identity: _RuntimeLaneIdentity) -> str:
        current = self._lane_ids_by_identity.get(identity)
        if current:
            return current
        lane_id = f"lane-{next(self._lane_id_counter)}"
        self._lane_ids_by_identity[identity] = lane_id
        return lane_id
