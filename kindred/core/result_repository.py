"""Owner-authorized result storage."""

from __future__ import annotations

from kindred.core.quantity_units import Unit
from kindred.core.volume import VolumeTrajectory

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np

from kindred.core.intervention_schedule_compiler import InterventionExecutionContract
from kindred.core.simulator.solver_types import SimulationPresentationBoundary, SimulationTermination
from kindred.core.simulation_failure import FactualCause
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
)

if TYPE_CHECKING:
    from kindred.core.simulation_parameters import DerivedParameterValue

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
    derived_parameters: tuple[DerivedParameterValue, ...] = ()
    presentation_boundaries: tuple[SimulationPresentationBoundary, ...] = ()
    quantity_units: tuple[tuple[str, Unit | None], ...] = ()
    presentation_units: tuple[tuple[str, Unit | None], ...] = ()
    volume: VolumeTrajectory | None = None
    # Each species' cumulative temporal concentration in M·s, the integral of
    # its concentration minus its final concentration over the executed span on
    # the solver's steps; only Run and Preview results without interventions
    # other than temperature changes carry it.
    species_ctc: Mapping[str, float] | None = None
    # Each species' largest concentration in M over the executed span and its
    # time in s, located on the solver's steps; Run and Preview results carry it.
    species_maximum: Mapping[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        # Canonical finalization is the sole producer: it selects succeeded and
        # failed outputs disjointly from the requested roster and classifies
        # exactly the parameters it records.  The carrier checks only that the
        # Algebra evaluation gave every requested output a terminal outcome.
        requested = set(self.requested_algebra_outputs)
        if requested.difference(self.succeeded_algebra_outputs, self.failed_algebra_outputs):
            raise ValueError("requested algebra outputs lack a terminal outcome.")


class ResultRepository:
    """Store each set's current result value and its Reference Run."""

    def __init__(self) -> None:
        # Each set's (request, result) pairs.
        self._current_by_set_id: dict[
            str, tuple[SimulationExecutionRequest, SimulationResultValue]
        ] = {}
        self._reference_by_set_id: dict[
            str, tuple[SimulationExecutionRequest, SimulationResultValue]
        ] = {}
        self._current_run_set_ids: set[str] = set()

    def copy_accepted_runs_from(
        self, source: ResultRepository, set_ids: Mapping[str, str]
    ) -> None:
        """Populate a fresh repository for an identical accepted setup.

        Reference retains the canonical Run while Slider Preview is active;
        scientific invalidation clears that authority on accepted input changes.
        Preview does not establish Run authority.
        """

        for source_id, destination_id in set_ids.items():
            entry = (
                source._current_by_set_id[source_id]
                if source_id in source._current_run_set_ids
                else source._reference_by_set_id.get(source_id)
            )
            if entry is not None:
                self._current_by_set_id[destination_id] = entry
                self._current_run_set_ids.add(destination_id)

    def current_result(self, set_id: str) -> SimulationResultValue | None:
        entry = self._current_by_set_id.get(set_id)
        return None if entry is None else entry[1]

    def reference_result(self, set_id: str) -> SimulationResultValue | None:
        entry = self._reference_by_set_id.get(set_id)
        return None if entry is None else entry[1]

    def has_result(self, set_id: str) -> bool:
        """Whether the set shows a current, Preview or Reference result."""

        return set_id in self._current_by_set_id or set_id in self._reference_by_set_id

    def current_request(self, set_id: str) -> SimulationExecutionRequest | None:
        entry = self._current_by_set_id.get(set_id)
        return None if entry is None else entry[0]

    def reference_request(self, set_id: str) -> SimulationExecutionRequest | None:
        entry = self._reference_by_set_id.get(set_id)
        return None if entry is None else entry[0]

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

    def clear_preview(self, set_id: str) -> None:
        """Remove the set's Preview value; its Reference stays retained."""

        sid = set_id
        if sid not in self._current_run_set_ids:
            self._current_by_set_id.pop(sid, None)

    def restore_reference_or_current(self, set_id: str) -> bool:
        """Expose the retained canonical Run.

        A Preview current value is expected to have been cleared first.  An
        existing current Run remains current; otherwise an exact retained
        Reference is promoted directly to current Run authority.
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
        *,
        keep_preview: bool = False,
    ) -> SimulationResultValue:
        """Publish a completed Run.

        With ``keep_preview``, a Run published while the set shows a Preview
        becomes that Preview's Reference instead of replacing it.
        """

        sid = set_id
        entry = (execution_request, result)
        if keep_preview and sid in self.preview_set_ids():
            self._reference_by_set_id[sid] = entry
        else:
            self._current_by_set_id[sid] = entry
            self._current_run_set_ids.add(sid)
            self._reference_by_set_id.pop(sid, None)
        return result

    def publish_preview(
        self,
        set_id: str,
        execution_request: SimulationExecutionRequest,
        result: SimulationResultValue,
    ) -> SimulationResultValue:
        """Publish a Preview current value without Reference changes."""

        sid = set_id
        self._current_by_set_id[sid] = (execution_request, result)
        self._current_run_set_ids.discard(sid)
        return result

    def deauthorize_sets(self, set_ids: Iterable[str]) -> None:
        """Clear the sets' current values and References.

        Deauthorization never manufactures a Reference value.
        """

        for set_id in set_ids:
            self._current_by_set_id.pop(set_id, None)
            self._current_run_set_ids.discard(set_id)
            self._reference_by_set_id.pop(set_id, None)
