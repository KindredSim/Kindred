from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from kindred.gui.controllers.runtime_lane_allocation import (
    RuntimeBackendLease,
    RuntimeBackendTask,
    RuntimeDispatchPlan,
    RuntimeLane,
    RuntimeReleaseResult,
    RuntimeTaskDescriptor,
)
from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity
from kindred.gui.controllers.simulation_runtime_orchestrator import RuntimeUiEffect


@dataclass(frozen=True)
class SimulationRuntimeDispatchResult:
    started: bool
    effects: tuple[RuntimeUiEffect, ...] = ()
    release_result: RuntimeReleaseResult | None = None


@dataclass(frozen=True)
class SimulationRuntimeDispatchDependencies:
    next_run_id: Callable[[], int]
    load_context: Callable[..., Mapping[str, Any]]
    callback_identity_for_descriptor: Callable[..., SimulationCallbackIdentity]
    runtime_lifecycle: Any
    record_nonfatal_exception: Callable[[str, BaseException], None]
    deactivate_dispatch_context: Callable[[Mapping[str, Any]], None] | None = None


class SimulationRuntimeDispatchOwner:
    """Consumes prepared runtime dispatch plans without owning readiness."""

    def __init__(
        self,
        *,
        ui: Any,
        batch_executor: Any,
        dependencies: SimulationRuntimeDispatchDependencies,
        parent: Any,
    ) -> None:
        self._ui = ui
        self._batch_executor = batch_executor
        self._deps = dependencies
        self._parent = parent

    def dispatch(self, dispatch_plan: RuntimeDispatchPlan) -> SimulationRuntimeDispatchResult:
        descriptors = tuple(dispatch_plan.ordered_task_descriptors or ())
        if not descriptors:
            consequence = self._deps.runtime_lifecycle.dispatch_rejected(
                dispatch_plan,
                message="Runtime dispatch plan has no tasks.",
                retryable=True,
            )
            return SimulationRuntimeDispatchResult(
                started=False,
                effects=tuple(consequence.effects),
                release_result=consequence.release_result,
            )
        try:
            self._validate_dispatch_plan_ready(dispatch_plan)
            backend_tasks = tuple(
                self._backend_task_for_descriptor(
                    descriptor,
                    dispatch_plan=dispatch_plan,
                )
                for descriptor in descriptors
            )
            for descriptor in descriptors:
                if not dict(descriptor.plan_payload or {}):
                    raise RuntimeError("Runtime task descriptor is missing a simulation plan.")
        except Exception as exc:
            consequence = self._deps.runtime_lifecycle.dispatch_rejected(
                dispatch_plan,
                message=str(exc),
                retryable=True,
            )
            return SimulationRuntimeDispatchResult(
                started=False,
                effects=tuple(consequence.effects),
                release_result=consequence.release_result,
            )
        return self._dispatch_task_queue(dispatch_plan, descriptors, backend_tasks)

    def _dispatch_task_queue(
        self,
        dispatch_plan: RuntimeDispatchPlan,
        descriptors: tuple[RuntimeTaskDescriptor, ...],
        backend_tasks: tuple[RuntimeBackendTask, ...],
    ) -> SimulationRuntimeDispatchResult:
        context: Mapping[str, Any] | None = None
        began = False
        submitted = 0
        abort_release_result: RuntimeReleaseResult | None = None
        abort_effects: tuple[RuntimeUiEffect, ...] = ()
        release_result: RuntimeReleaseResult | None = None
        try:
            run_id = self._deps.next_run_id()
            context = self._deps.load_context(
                dispatch_plan=dispatch_plan,
                run_id=run_id,
                active=True,
            )
            queue_ids = [descriptor.set_id for descriptor in descriptors]
            queue_names = [
                self._ui.batch.batch_set_name_for_id(descriptor.set_id) or descriptor.set_id
                for descriptor in descriptors
            ]
            request_id = int(dispatch_plan.launch_allocation.launch_intent.request_token or 0)
            fast_mode = dispatch_plan.launch_allocation.launch_intent.intent_kind == "preview"
            accepted_capacity = max(1, int(dispatch_plan.launch_allocation.accepted_capacity or 1))
            self._batch_executor.begin_run(
                run_id=int(run_id),
                request_id=int(request_id),
                fast_mode=bool(fast_mode),
                queue_ids=queue_ids,
                queue_names=queue_names,
                preview_owner_epoch=dispatch_plan.launch_allocation.launch_intent.preview_epoch,
            )
            began = True
            started_consequence = self._deps.runtime_lifecycle.dispatch_started(dispatch_plan)
            runtime_effects = tuple(started_consequence.effects)
            for descriptor, backend_task in zip(descriptors, backend_tasks):
                self._submit_task_descriptor(
                    descriptor,
                    backend_task,
                    dispatch_plan=dispatch_plan,
                    run_id=run_id,
                    context=context,
                )
                submitted += 1
            _ = accepted_capacity
            return SimulationRuntimeDispatchResult(started=True, effects=runtime_effects)
        except Exception as exc:
            if began or submitted:
                try:
                    abort_consequence = self._deps.runtime_lifecycle.dispatch_aborted(
                        dispatch_plan,
                        message=str(exc),
                        retryable=True,
                    )
                    abort_release_result = abort_consequence.release_result
                    abort_effects = tuple(abort_consequence.effects)
                except Exception as shutdown_exc:
                    self._deps.record_nonfatal_exception(
                        "Failed to shut down partially submitted runtime dispatch",
                        shutdown_exc,
                    )
            if context is not None and callable(self._deps.deactivate_dispatch_context):
                self._deps.deactivate_dispatch_context(context)
            if not (began or submitted):
                rejection_consequence = self._deps.runtime_lifecycle.dispatch_rejected(
                    dispatch_plan,
                    message=str(exc),
                    retryable=True,
                )
                release_result = rejection_consequence.release_result
                abort_effects = tuple(rejection_consequence.effects)
            return SimulationRuntimeDispatchResult(
                started=False,
                effects=abort_effects,
                release_result=abort_release_result or release_result,
            )

    def _submit_task_descriptor(
        self,
        descriptor: RuntimeTaskDescriptor,
        backend_task: RuntimeBackendTask,
        *,
        dispatch_plan: RuntimeDispatchPlan,
        run_id: int,
        context: Mapping[str, Any],
    ) -> None:
        set_name = self._ui.batch.batch_set_name_for_id(descriptor.set_id) or descriptor.set_id
        callback_identity = self._deps.callback_identity_for_descriptor(
            descriptor,
            dispatch_plan=dispatch_plan,
            run_id=run_id,
            context=context,
        )
        self._batch_executor.submit_task(
            backend_task,
            set_id=str(descriptor.set_id),
            set_name=str(set_name),
            callback_identity=callback_identity,
        )

    def _backend_task_for_descriptor(
        self,
        descriptor: RuntimeTaskDescriptor,
        *,
        dispatch_plan: RuntimeDispatchPlan,
    ) -> RuntimeBackendTask:
        lane_assignment = dispatch_plan.assignment_for_task(descriptor.task_id)
        runtime_lane = self._runtime_lane_for_assignment(dispatch_plan, lane_assignment)
        if lane_assignment is None or runtime_lane is None:
            raise RuntimeError("Runtime task descriptor has no runtime lane assignment.")
        backend_pool_token = str(runtime_lane.backend_pool_token or "")
        backend_generation = int(runtime_lane.backend_generation or 0)
        backend_lease_id = str(runtime_lane.backend_lease_id or "")
        if not backend_lease_id:
            raise RuntimeError("Runtime task descriptor is missing a provider backend lease.")
        if not backend_pool_token:
            raise RuntimeError("Runtime task descriptor is missing a provider backend pool token.")
        if backend_generation <= 0:
            raise RuntimeError("Runtime task descriptor is missing a provider backend generation.")
        backend_capacity = int(runtime_lane.backend_lease_capacity or 0)
        if backend_capacity <= 0:
            raise RuntimeError("Runtime task descriptor is missing provider backend lease capacity.")
        return RuntimeBackendTask(
            descriptor=descriptor,
            dispatch_plan_id=str(dispatch_plan.launch_allocation.allocation_id),
            allocation_id=str(dispatch_plan.launch_allocation.allocation_id),
            release_token=str(dispatch_plan.release_token),
            lane_assignment=lane_assignment,
            backend_lease=RuntimeBackendLease(
                lease_id=backend_lease_id,
                pool_token=backend_pool_token,
                generation=backend_generation,
                compatibility_key=runtime_lane.compatibility_key,
                capacity=backend_capacity,
            ),
        )

    @staticmethod
    def _validate_dispatch_plan_ready(dispatch_plan: RuntimeDispatchPlan) -> None:
        allocation = dispatch_plan.launch_allocation
        if str(allocation.status or "") != "ready":
            raise RuntimeError("Runtime dispatch plan is not ready with a reserved allocation.")
        reservation = allocation.reservation
        if str(reservation.state or "") != "reserved":
            raise RuntimeError("Runtime dispatch plan is not ready with a reserved allocation.")
        if not tuple(reservation.lanes or ()):
            raise RuntimeError("Runtime dispatch plan has no reserved runtime lanes.")

    @staticmethod
    def _runtime_lane_for_assignment(
        dispatch_plan: RuntimeDispatchPlan,
        lane_assignment: Any,
    ) -> RuntimeLane | None:
        if lane_assignment is None:
            return None
        lane_id = str(getattr(lane_assignment, "lane_id", "") or "")
        if not lane_id:
            return None
        for lane in dispatch_plan.launch_allocation.reservation.lanes:
            if str(lane.lane_id) == lane_id:
                return lane
        return None
