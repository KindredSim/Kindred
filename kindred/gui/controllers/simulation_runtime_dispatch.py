from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from kindred.gui.controllers.runtime_lane_allocation import (
    RuntimeDispatchPlan,
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
        return self._dispatch_task_queue(dispatch_plan, descriptors)

    def _dispatch_task_queue(
        self,
        dispatch_plan: RuntimeDispatchPlan,
        descriptors: tuple[RuntimeTaskDescriptor, ...],
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
            for descriptor in descriptors:
                if not dict(descriptor.plan_payload or {}):
                    raise RuntimeError("Runtime task descriptor is missing a simulation plan.")
                self._submit_task_descriptor(
                    descriptor,
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
            {
                "run_id": int(run_id),
                "request_id": int(descriptor.request_token or 0),
                "set_id": str(descriptor.set_id),
                "set_name": str(set_name),
                "include_mechanism_in_result_payload": True,
                "simulation_plan": dict(descriptor.plan_payload or {}),
            },
            set_id=str(descriptor.set_id),
            set_name=str(set_name),
            callback_identity=callback_identity,
        )
