"""Owner-authorized result storage and bounded exact-request hydration cache."""

from __future__ import annotations

from kindred.core.quantity_units import Unit
from kindred.core.volume import VolumeTrajectory

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from kindred.core.intervention_schedule_compiler import InterventionExecutionContract
from kindred.core.simulator.solver_types import SimulationPresentationBoundary, SimulationTermination
from kindred.core.simulation_failure import FactualCause
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
)

__all__ = [
    "ResultRepository",
    "SimulationResultValue",
]


@dataclass(frozen=True, slots=True, eq=False)
class SimulationResultValue:
    """Immutable numerical output and its complete execution truth."""

    t: np.ndarray
    temperature_K: np.ndarray
    series: Mapping[str, np.ndarray]
    segment_ids: np.ndarray
    segment_keys: tuple[str, ...]
    discontinuity_ids: np.ndarray
    intervention_execution: InterventionExecutionContract
    solver_provenance: Mapping[str, Any]
    termination: SimulationTermination
    requested_algebra_outputs: tuple[str, ...] = ()
    succeeded_algebra_outputs: tuple[str, ...] = ()
    failed_algebra_outputs: tuple[str, ...] = ()
    algebra_errors: tuple[FactualCause, ...] = ()
    scalars: Mapping[str, float] = MappingProxyType({})
    parameters: tuple[tuple[str, float, str], ...] = ()
    parameter_classifications: tuple[tuple[str, str, str], ...] = ()
    presentation_boundaries: tuple[SimulationPresentationBoundary, ...] = ()
    quantity_units: tuple[tuple[str, Unit | None], ...] = ()
    presentation_units: tuple[tuple[str, Unit | None], ...] = ()
    volume: VolumeTrajectory | None = None

    def __post_init__(self) -> None:
        # Canonical finalization is the sole producer: it selects succeeded and
        # failed outputs disjointly from the requested roster and classifies
        # exactly the parameters it records.  The carrier checks only that the
        # Algebra evaluation gave every requested output a terminal outcome.
        requested = set(self.requested_algebra_outputs)
        if requested.difference(self.succeeded_algebra_outputs, self.failed_algebra_outputs):
            raise ValueError("requested algebra outputs lack a terminal outcome.")


class ResultRepository:
    """Store completed result values and a bounded exact-request Run cache."""

    def __init__(
        self,
        tab_id: str,
        *,
        cache_capacity: int,
    ) -> None:
        self._tab_id = tab_id
        # The configuration owner admits the positive capacity.
        self._cache_capacity = cache_capacity
        self._current_by_set_id: dict[
            str,
            tuple[str, SimulationExecutionRequest, SimulationResultValue],
        ] = {}
        self._reference_by_set_id: dict[
            str,
            tuple[str, SimulationExecutionRequest, SimulationResultValue],
        ] = {}
        self._current_run_set_ids: set[str] = set()
        self._explicit_cache: list[
            tuple[str, SimulationExecutionRequest, SimulationResultValue]
        ] = []

    @property
    def tab_id(self) -> str:
        return self._tab_id

    def copy_accepted_runs_from(
        self, source: ResultRepository, set_ids: Mapping[str, str]
    ) -> None:
        """Populate a fresh repository for an identical accepted setup.

        Reference retains the canonical Run while Slider Preview is active;
        scientific invalidation clears that authority on accepted input changes.
        Neither Preview nor historical cache entries establish Run authority.
        """

        self._cache_capacity = source._cache_capacity
        for source_id, destination_id in set_ids.items():
            entry = (
                source._current_by_set_id[source_id]
                if source_id in source._current_run_set_ids
                else source._reference_by_set_id.get(source_id)
            )
            if entry is not None:
                _, request, result = entry
                self._current_by_set_id[destination_id] = (
                    destination_id, request, result
                )
                self._current_run_set_ids.add(destination_id)

    def resize_cache_capacity(self, cache_capacity: int) -> None:
        """Apply a positive capacity immediately and trim this repository's LRU."""

        self._cache_capacity = cache_capacity
        self._trim_explicit_cache()

    def current_result(self, set_id: str) -> SimulationResultValue | None:
        entry = self._current_by_set_id.get(set_id)
        return None if entry is None else entry[2]

    def reference_result(self, set_id: str) -> SimulationResultValue | None:
        entry = self._reference_by_set_id.get(set_id)
        return None if entry is None else entry[2]

    def current_request(self, set_id: str) -> SimulationExecutionRequest | None:
        entry = self._current_by_set_id.get(set_id)
        return None if entry is None else entry[1]

    def reference_request(self, set_id: str) -> SimulationExecutionRequest | None:
        entry = self._reference_by_set_id.get(set_id)
        return None if entry is None else entry[1]

    def retain_explicit_as_reference(self, set_id: str) -> None:
        """Retain the current Run unless its Reference is already retained."""

        sid = set_id
        if (
            sid not in self._current_run_set_ids
            or sid in self._reference_by_set_id
        ):
            return
        explicit = self._current_by_set_id.pop(sid, None)
        self._current_run_set_ids.discard(sid)
        if explicit is not None:
            self._reference_by_set_id[sid] = explicit

    def clear_preview(self, set_id: str, *, clear_reference: bool = True) -> None:
        sid = set_id
        if sid not in self._current_run_set_ids:
            self._current_by_set_id.pop(sid, None)
        if clear_reference:
            self._reference_by_set_id.pop(sid, None)

    def restore_reference_or_current(self, set_id: str) -> bool:
        """Expose the retained canonical Run without consulting the cache.

        A Preview current value is expected to have been cleared first.  An
        existing current Run remains current; otherwise an exact retained
        Reference is promoted directly to current Run authority.  Cache
        membership and LRU order are deliberately untouched.
        """

        sid = set_id
        if sid in self._current_run_set_ids:
            return True
        reference = self._reference_by_set_id.pop(sid, None)
        if reference is None:
            return False
        self._current_by_set_id[sid] = reference
        self._current_run_set_ids.add(sid)
        return True

    def preview_set_ids(self) -> tuple[str, ...]:
        """Return semantic current values published as exact Preview."""

        return tuple(
            sorted(
                set(self._current_by_set_id).difference(
                    self._current_run_set_ids
                )
            )
        )

    def publish_run(
        self,
        set_id: str,
        execution_request: SimulationExecutionRequest,
        result: SimulationResultValue,
    ) -> SimulationResultValue:
        """Publish a completed Run and admit it to the bounded exact cache."""

        sid = set_id
        entry = (sid, execution_request, result)
        self._current_by_set_id[sid] = entry
        self._current_run_set_ids.add(sid)
        self._reference_by_set_id.pop(sid, None)
        self._explicit_cache[:] = [
            cached
            for cached in self._explicit_cache
            if cached[0] != sid
            or not (cached[1] == entry[1])
        ]
        self._explicit_cache.append(entry)
        self._trim_explicit_cache()
        return result

    def publish_preview(
        self,
        set_id: str,
        execution_request: SimulationExecutionRequest,
        result: SimulationResultValue,
    ) -> SimulationResultValue:
        """Publish a Preview current value without cache admission or Reference changes."""

        sid = set_id
        self._current_by_set_id[sid] = (
            sid,
            execution_request,
            result,
        )
        self._current_run_set_ids.discard(sid)
        return result

    def hydrate_cached(self, set_id: str, execution_request: SimulationExecutionRequest) -> SimulationResultValue | None:
        """Hydrate a structurally equal full request; a miss changes nothing."""
        sid = set_id
        entry = next(
            (
                cached
                for cached in reversed(self._explicit_cache)
                if cached[0] == sid
                and (cached[1] == execution_request)
            ),
            None,
        )
        if entry is None:
            return None
        self._explicit_cache.remove(entry)
        self._explicit_cache.append(entry)
        self._current_by_set_id[sid] = entry
        self._current_run_set_ids.add(sid)
        self._reference_by_set_id.pop(sid, None)
        return entry[2]

    def deauthorize_sets(self, set_ids: Iterable[str]) -> None:
        """Clear foreground values and the Reference; keep the exact Run cache.

        Deauthorization never manufactures a Reference value.
        """

        for set_id in set_ids:
            self._current_by_set_id.pop(set_id, None)
            self._current_run_set_ids.discard(set_id)
            self._reference_by_set_id.pop(set_id, None)

    def purge_sets(self, set_ids: Iterable[str]) -> None:
        """Delete/history removal purges visible entries and cached storage."""

        removed = set(set_ids)
        self.deauthorize_sets(removed)
        self._explicit_cache[:] = [
            entry for entry in self._explicit_cache if entry[0] not in removed
        ]

    def _trim_explicit_cache(self) -> None:
        del self._explicit_cache[:-self._cache_capacity]
