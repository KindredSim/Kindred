from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity


@dataclass(frozen=True)
class ContainedSerialWorkerLaunchRequest:
    plan_payload: Mapping[str, Any] | None
    callback_identity: SimulationCallbackIdentity
    include_mechanism_in_result_payload: bool
    worker_signature: str | None
    parent: Any


class ContainedSerialWorkerLaunchOwner:
    def __init__(
        self,
        *,
        acquire_ready_owner_for_plan: Callable[..., Any],
        release_owner: Callable[..., None],
        worker_factory: Callable[..., Any] | None = None,
        contained_payload_builder: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
        record_nonfatal_exception: Callable[[str, BaseException], None] | None = None,
    ) -> None:
        self._acquire_ready_owner_for_plan = acquire_ready_owner_for_plan
        self._release_owner = release_owner
        self._worker_factory = worker_factory
        self._contained_payload_builder = contained_payload_builder
        self._record_nonfatal_exception = record_nonfatal_exception

    def create_worker(self, request: ContainedSerialWorkerLaunchRequest) -> Any | None:
        if not isinstance(request.plan_payload, Mapping):
            raise ValueError("Missing simulation plan payload for contained batch dispatch")
        contained_payload = self._build_contained_payload(request.plan_payload)
        identity = request.callback_identity
        contained_owner = self._acquire_ready_owner_for_plan(
            fast_mode=bool(identity.fast_mode),
            simulation_plan_payload=contained_payload,
        )
        if contained_owner is None:
            return None
        try:
            worker = self._worker_class()(
                owner=contained_owner,
                simulation_plan_payload=contained_payload,
                include_mechanism_in_result_payload=bool(request.include_mechanism_in_result_payload),
                parent=request.parent,
            )
        except Exception:
            try:
                self._release_owner(contained_owner, kill=False)
            except Exception as release_exc:
                if self._record_nonfatal_exception is not None:
                    self._record_nonfatal_exception("Failed to release simulation runtime owner", release_exc)
            raise
        self._stamp_worker_identity(
            worker,
            identity=identity,
            plan_payload=dict(request.plan_payload),
            worker_signature=request.worker_signature,
        )
        return worker

    def _build_contained_payload(self, plan_payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._contained_payload_builder is not None:
            return dict(self._contained_payload_builder(dict(plan_payload)))
        from kindred.core.simulation_containment import build_contained_simulation_plan_payload

        return build_contained_simulation_plan_payload(dict(plan_payload))

    def _worker_class(self) -> Callable[..., Any]:
        if self._worker_factory is not None:
            return self._worker_factory
        from kindred.gui.simulation_worker import ContainedSimulationWorker

        return ContainedSimulationWorker

    def _stamp_worker_identity(
        self,
        worker: Any,
        *,
        identity: SimulationCallbackIdentity,
        plan_payload: Mapping[str, Any],
        worker_signature: str | None,
    ) -> None:
        worker._run_id = int(identity.run_id or 0)
        worker._request_id = int(identity.request_id or 0)
        worker._fast_mode = bool(identity.fast_mode)
        worker._batch_set_name = str(identity.batch_set or "")
        worker._batch_set_id = str(identity.batch_set_id or "")
        worker._batch_cache_key = str(identity.cache_key or "")
        if worker_signature:
            worker._batch_mechanism_signature = str(worker_signature)
        worker._simulation_plan = dict(plan_payload)
