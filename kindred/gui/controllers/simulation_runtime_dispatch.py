from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from kindred.gui.controllers.runtime_lane_allocation import (
    RuntimeBackendLease,
    RuntimeBackendTask,
    RuntimeDispatchPlan,
    RuntimeLane,
    RuntimeTaskDescriptor,
)
from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity


@dataclass(frozen=True)
class SimulationRuntimeDispatchDependencies:
    next_run_id: Callable[[], int]
    load_context: Callable[..., Mapping[str, Any]]
    callback_identity_for_descriptor: Callable[..., SimulationCallbackIdentity]
    set_simulation_running: Callable[[bool], None]
    set_slider_simulation_active: Callable[[bool], None]
    release_dispatch_plan: Callable[..., bool]
    render_failure: Callable[..., None]
    set_active_dispatch_plan: Callable[[RuntimeDispatchPlan | None], None]
    record_nonfatal_exception: Callable[[str, BaseException], None]
    start_completion_poll_timer: Callable[[], None]
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

    def dispatch(self, dispatch_plan: RuntimeDispatchPlan) -> bool:
        descriptors = tuple(dispatch_plan.ordered_task_descriptors or ())
        if not descriptors:
            self._deps.release_dispatch_plan(dispatch_plan, failed=True)
            return False
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
            self._deps.release_dispatch_plan(dispatch_plan, failed=True)
            self._deps.render_failure(str(exc), retryable=True)
            return False
        return self._dispatch_task_queue(dispatch_plan, descriptors, backend_tasks)

    def _dispatch_task_queue(
        self,
        dispatch_plan: RuntimeDispatchPlan,
        descriptors: tuple[RuntimeTaskDescriptor, ...],
        backend_tasks: tuple[RuntimeBackendTask, ...],
    ) -> bool:
        context: Mapping[str, Any] | None = None
        began = False
        submitted = 0
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
                keep_lane_pool_alive=bool(dispatch_plan.launch_allocation.retain_lanes_after_success),
                preview_owner_epoch=dispatch_plan.launch_allocation.launch_intent.preview_epoch,
                cache_key=str(descriptors[0].cache_key if descriptors else ""),
            )
            began = True
            self._deps.set_active_dispatch_plan(dispatch_plan)
            for descriptor, backend_task in zip(descriptors, backend_tasks):
                self._submit_task_descriptor(
                    descriptor,
                    backend_task,
                    dispatch_plan=dispatch_plan,
                    run_id=run_id,
                    context=context,
                )
                submitted += 1
            self._deps.set_simulation_running(True)
            self._deps.set_slider_simulation_active(bool(fast_mode))
            self._ui.run_ui.set_run_button_enabled(False)
            self._ui.run_ui.set_stop_button_enabled(True)
            self._ui.run_ui.set_sim_progress_value(0)
            self._ui.run_ui.set_status_text(
                f"Running {len(descriptors)} sets on {accepted_capacity} runtime lanes..."
            )
            self._deps.start_completion_poll_timer()
            return True
        except Exception as exc:
            if began or submitted:
                try:
                    self._batch_executor.shutdown(
                        force_terminate=True,
                        record_nonfatal_exception=self._deps.record_nonfatal_exception,
                    )
                except Exception as shutdown_exc:
                    self._deps.record_nonfatal_exception(
                        "Failed to shut down partially submitted runtime dispatch",
                        shutdown_exc,
                    )
            if context is not None and callable(self._deps.deactivate_dispatch_context):
                self._deps.deactivate_dispatch_context(context)
            self._deps.set_active_dispatch_plan(None)
            self._deps.set_simulation_running(False)
            self._deps.set_slider_simulation_active(False)
            self._ui.run_ui.set_run_button_enabled(True)
            self._ui.run_ui.set_stop_button_enabled(False)
            self._deps.release_dispatch_plan(dispatch_plan, failed=True)
            self._deps.render_failure(str(exc), retryable=True)
            return False

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
                capacity=1,
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
