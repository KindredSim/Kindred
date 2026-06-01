from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity


@dataclass(frozen=True)
class SimulationCallbackFreshnessDependencies:
    run_state: Any
    batch_context_owner: Any
    preview_ownership: Callable[[], Any]
    shutdown_requested: Callable[[], bool]
    current_global_epoch: Callable[[], int]
    current_epoch: Callable[[], int]
    current_set_epoch: Callable[[str], int]
    finalize_batch_queue_done_without_result: Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class SimulationCallbackFreshnessDecision:
    active_run_id: int
    latest_request_id: int
    shutdown_requested: bool
    current_global_epoch: int
    callback_preview_owner_epoch: Optional[int]
    stale_run: bool
    runtime_input_stale: bool
    missing_preview_owner_epoch: bool
    preview_owner_matches: bool
    superseded_fast_request: bool
    dispatch_identity_stale: bool = False


class SimulationCallbackFreshnessOwner:
    """Owns the immutable freshness decision for one captured callback."""

    def __init__(self, dependencies: SimulationCallbackFreshnessDependencies) -> None:
        self._deps = dependencies

    def assess_callback(
        self,
        callback_identity: SimulationCallbackIdentity,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> SimulationCallbackFreshnessDecision:
        batch_set_id = str(callback_identity.batch_set_id or "").strip()
        current_global_epoch = int(self._deps.current_global_epoch())
        current_epoch = int(self._deps.current_epoch())
        current_set_epoch = int(self._deps.current_set_epoch(batch_set_id))
        callback_preview_owner_epoch = (
            None
            if callback_identity.preview_owner_epoch is None
            else int(callback_identity.preview_owner_epoch)
        )
        active_run_id = int(getattr(self._deps.run_state, "active_run_id", 0))
        latest_request_id = int(getattr(self._deps.run_state, "latest_sim_request_id", 0))
        preview_owner_matches = self._preview_request_matches_current_preview_owner_epoch(
            callback_identity.request_id,
            callback_preview_owner_epoch,
        )
        missing_preview_owner_epoch = self._missing_preview_owner_epoch_is_stale_for_fast_callback(
            fast_mode=callback_identity.fast_mode,
            request_id=callback_identity.request_id,
            preview_owner_epoch=callback_preview_owner_epoch,
            latest_request_id=latest_request_id,
        )
        runtime_input_stale = False
        dispatch_identity_stale = False
        if isinstance(context, Mapping):
            runtime_input_stale = self._deps.batch_context_owner.runtime_input_stale_for_set(
                context,
                batch_set_id=batch_set_id,
                current_global_epoch=current_global_epoch,
                current_set_epoch=current_set_epoch,
                current_epoch=current_epoch,
            )
            dispatch_identity_stale = self._dispatch_identity_stale(
                callback_identity,
                context=context,
                batch_set_id=batch_set_id,
            )
        return SimulationCallbackFreshnessDecision(
            active_run_id=active_run_id,
            latest_request_id=latest_request_id,
            shutdown_requested=bool(self._deps.shutdown_requested()),
            current_global_epoch=current_global_epoch,
            callback_preview_owner_epoch=callback_preview_owner_epoch,
            stale_run=int(callback_identity.run_id) != active_run_id,
            runtime_input_stale=bool(runtime_input_stale),
            missing_preview_owner_epoch=bool(missing_preview_owner_epoch),
            preview_owner_matches=bool(preview_owner_matches),
            superseded_fast_request=bool(
                callback_identity.fast_mode
                and (bool(missing_preview_owner_epoch) or not bool(preview_owner_matches))
            ),
            dispatch_identity_stale=bool(dispatch_identity_stale),
        )

    def mark_stale_runtime_input_callback_consumed(
        self,
        *,
        batch_set_id: Optional[str],
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        set_id = str(batch_set_id or "").strip()
        if not set_id:
            return
        if isinstance(context, Mapping) and not self._deps.batch_context_owner.context_matches_current_run_identity(context):
            return
        self._mark_runtime_task_callback_consumed(set_id=set_id)

    def mark_stale_dispatch_identity_callback_consumed(
        self,
        *,
        batch_set_id: Optional[str],
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        set_id = str(batch_set_id or "").strip()
        if not set_id:
            return
        if isinstance(context, Mapping) and not self._deps.batch_context_owner.context_matches_current_run_identity(context):
            return
        self._mark_runtime_task_callback_consumed(set_id=set_id)

    def _mark_runtime_task_callback_consumed(self, *, set_id: str) -> None:
        transition = self._deps.batch_context_owner.record_runtime_task_stale_callback_consumed_if_active(set_id=set_id)
        if transition is None:
            return
        if transition.batch_done:
            self._deps.finalize_batch_queue_done_without_result(transition.context)

    def _preview_request_matches_current_preview_owner_epoch(
        self,
        request_id: Optional[int],
        preview_owner_epoch: Optional[int],
    ) -> bool:
        if request_id is None:
            return True
        ownership = self._deps.preview_ownership()
        owner_request_id = getattr(ownership, "request_id", None)
        if owner_request_id is None:
            return False
        if int(owner_request_id) != int(request_id):
            return False
        if preview_owner_epoch is None:
            return True
        return int(getattr(ownership, "epoch", 0)) == int(preview_owner_epoch)

    def _missing_preview_owner_epoch_is_stale_for_fast_callback(
        self,
        *,
        fast_mode: Optional[bool],
        request_id: Optional[int],
        preview_owner_epoch: Optional[int],
        latest_request_id: int,
    ) -> bool:
        if (not bool(fast_mode)) or request_id is None or preview_owner_epoch is not None:
            return False
        ownership = self._deps.preview_ownership()
        owner_request_id = getattr(ownership, "request_id", None)
        if owner_request_id is None:
            return False
        return int(owner_request_id) == int(request_id) and int(request_id) != int(latest_request_id)

    @staticmethod
    def _dispatch_identity_stale(
        callback_identity: SimulationCallbackIdentity,
        *,
        context: Mapping[str, Any],
        batch_set_id: str,
    ) -> bool:
        identity_by_set_id = context.get("runtime_task_identity_by_set_id")
        if not isinstance(identity_by_set_id, Mapping):
            return False
        expected = identity_by_set_id.get(str(batch_set_id or ""))
        if not isinstance(expected, Mapping):
            return False
        comparisons = (
            ("allocation_id", str(getattr(callback_identity, "allocation_id", "") or ""), str(expected.get("allocation_id") or "")),
            ("lane_id", str(getattr(callback_identity, "lane_id", "") or ""), str(expected.get("lane_id") or "")),
            (
                "lane_generation",
                int(getattr(callback_identity, "lane_generation", 0) or 0),
                int(expected.get("lane_generation", 0) or 0),
            ),
            (
                "row",
                None if getattr(callback_identity, "row", None) is None else int(callback_identity.row),
                None if expected.get("row") is None else int(expected.get("row")),
            ),
            (
                "exact_descriptor_hash",
                str(getattr(callback_identity, "exact_descriptor_hash", "") or ""),
                str(expected.get("exact_descriptor_hash") or ""),
            ),
            ("cache_key", str(callback_identity.cache_key or ""), str(expected.get("cache_key") or "")),
        )
        for _name, actual, expected_value in comparisons:
            if actual != expected_value:
                return True
        actual_compatibility = getattr(callback_identity, "compatibility_key", None)
        if isinstance(actual_compatibility, Mapping) or isinstance(expected.get("compatibility_key"), Mapping):
            return dict(actual_compatibility or {}) != dict(expected.get("compatibility_key") or {})
        return False
