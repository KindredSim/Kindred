"""Owner-authorized result storage and bounded exact-request hydration cache."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from kindred.core.intervention_schedule_compiler import InterventionExecutionContract
from kindred.core.simulator.solver_types import SimulationPresentationBoundary
from kindred.core.simulation_failure import FactualCause
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
)

__all__ = [
    "ResultRepository",
    "SimulationResultValue",
]


def _required_id(value: object, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} must be nonblank.")
    return normalized


def _freeze_provenance_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(name): _freeze_provenance_value(item)
                for name, item in value.items()
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_provenance_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(
            _freeze_provenance_value(item)
            for item in sorted(value, key=repr)
        )
    if isinstance(value, np.ndarray):
        array = np.asarray(value).copy()
        array.setflags(write=False)
        return array
    return value


def _frozen_provenance_mapping(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(name): _freeze_provenance_value(item)
            for name, item in value.items()
        }
    )


@dataclass(frozen=True, slots=True, eq=False)
class SimulationResultValue:
    """Immutable numerical output and its complete execution truth."""

    t: np.ndarray
    series: Mapping[str, np.ndarray]
    segment_ids: np.ndarray
    segment_keys: tuple[str, ...]
    discontinuity_ids: np.ndarray
    intervention_execution: InterventionExecutionContract
    solver_provenance: Mapping[str, Any]
    preparation_warnings: tuple[str, ...] = ()
    requested_algebra_outputs: tuple[str, ...] = ()
    succeeded_algebra_outputs: tuple[str, ...] = ()
    failed_algebra_outputs: tuple[str, ...] = ()
    omitted_algebra_outputs: tuple[str, ...] = ()
    algebra_errors: tuple[FactualCause, ...] = ()
    scalars: Mapping[str, float] = MappingProxyType({})
    parameters: tuple[tuple[str, float, str], ...] = ()
    presentation_boundaries: tuple[SimulationPresentationBoundary, ...] = ()

    def __post_init__(self) -> None:
        # Canonical finalization is the sole producer; solver-owned arrays and
        # boundaries already crossed their typed admission boundary.
        presentation_boundaries = tuple(self.presentation_boundaries)
        series = dict(self.series)

        scalars: dict[str, float] = {}
        for raw_name, raw_value in dict(self.scalars or {}).items():
            name = _required_id(raw_name, "scalar name")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(f"scalar {name!r} must be finite.")
            scalars[name] = value

        parameters: list[tuple[str, float, str]] = []
        parameter_names: set[str] = set()
        for raw_name, raw_value, raw_unit in self.parameters:
            name = _required_id(raw_name, "parameter name")
            if name in parameter_names:
                raise ValueError(f"parameters contain duplicate name {name!r}.")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(f"parameter {name!r} must be finite.")
            parameter_names.add(name)
            parameters.append((name, value, str(raw_unit or "")))

        def normalized_names(values: Iterable[str], field: str) -> tuple[str, ...]:
            result = tuple(_required_id(value, field) for value in values)
            if len(result) != len(set(result)):
                raise ValueError(f"{field} contains duplicate output names.")
            return result

        preparation_warnings = tuple(str(value) for value in self.preparation_warnings)
        requested = normalized_names(self.requested_algebra_outputs, "requested_algebra_outputs")
        succeeded = normalized_names(self.succeeded_algebra_outputs, "succeeded_algebra_outputs")
        failed = normalized_names(self.failed_algebra_outputs, "failed_algebra_outputs")
        omitted = normalized_names(self.omitted_algebra_outputs, "omitted_algebra_outputs")
        requested_set = set(requested)
        partitions = (set(succeeded), set(failed), set(omitted))
        if any(not values <= requested_set for values in partitions):
            raise ValueError("algebra outcome contains an unrequested output.")
        if set.intersection(*partitions):
            raise ValueError("algebra output outcome partitions overlap.")
        if requested_set.difference(set.union(*partitions)):
            raise ValueError("requested algebra outputs lack a terminal outcome.")
        errors = tuple(self.algebra_errors)
        if any(not isinstance(value, FactualCause) for value in errors):
            raise TypeError("algebra_errors must contain only FactualCause values.")

        object.__setattr__(self, "series", MappingProxyType(series))
        object.__setattr__(self, "scalars", MappingProxyType(scalars))
        object.__setattr__(self, "parameters", tuple(parameters))
        object.__setattr__(self, "solver_provenance", _frozen_provenance_mapping(self.solver_provenance))
        object.__setattr__(self, "preparation_warnings", preparation_warnings)
        object.__setattr__(self, "requested_algebra_outputs", requested)
        object.__setattr__(self, "succeeded_algebra_outputs", succeeded)
        object.__setattr__(self, "failed_algebra_outputs", failed)
        object.__setattr__(self, "omitted_algebra_outputs", omitted)
        object.__setattr__(self, "algebra_errors", errors)
        object.__setattr__(
            self,
            "presentation_boundaries",
            presentation_boundaries,
        )


    @property
    def completed_with_warnings(self) -> bool:
        return bool(self.preparation_warnings or self.failed_algebra_outputs or self.omitted_algebra_outputs or self.algebra_errors)
class ResultRepository:
    """Store completed result values and a bounded exact-request Run cache."""

    def __init__(
        self,
        tab_id: str,
        *,
        cache_capacity: int,
    ) -> None:
        self._tab_id = _required_id(tab_id, "tab_id")
        if type(cache_capacity) is not int:
            raise TypeError("cache_capacity must be an int.")
        self._cache_capacity = cache_capacity
        if cache_capacity < 1:
            raise ValueError("cache_capacity must be at least one.")
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

    def resize_cache_capacity(self, cache_capacity: int) -> None:
        """Apply a positive capacity immediately and trim this repository's LRU."""

        if type(cache_capacity) is not int:
            raise TypeError("cache_capacity must be an int.")
        if cache_capacity < 1:
            raise ValueError("cache_capacity must be at least one.")
        self._cache_capacity = cache_capacity
        self._trim_explicit_cache()

    def current_result(self, set_id: str) -> SimulationResultValue | None:
        entry = self._current_by_set_id.get(str(set_id or ""))
        return None if entry is None else entry[2]

    def reference_result(self, set_id: str) -> SimulationResultValue | None:
        entry = self._reference_by_set_id.get(str(set_id or ""))
        return None if entry is None else entry[2]

    def current_request(self, set_id: str) -> SimulationExecutionRequest | None:
        entry = self._current_by_set_id.get(str(set_id or ""))
        return None if entry is None else entry[1]

    def reference_request(self, set_id: str) -> SimulationExecutionRequest | None:
        entry = self._reference_by_set_id.get(str(set_id or ""))
        return None if entry is None else entry[1]

    def retain_explicit_as_reference(self, set_id: str) -> None:
        """Retain the current Run unless its Reference is already retained."""

        sid = _required_id(set_id, "set_id")
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
        sid = _required_id(set_id, "set_id")
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

        sid = _required_id(set_id, "set_id")
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

        sid = _required_id(set_id, "set_id")
        if not isinstance(execution_request, SimulationExecutionRequest):
            raise TypeError("execution_request must be a SimulationExecutionRequest.")
        if not isinstance(result, SimulationResultValue):
            raise TypeError("result must be a SimulationResultValue.")
        entry = (sid, execution_request, result)
        self._current_by_set_id[sid] = entry
        self._current_run_set_ids.add(sid)
        self._reference_by_set_id.pop(sid, None)
        from kindred.core.simulation_preparation import simulation_execution_requests_equal
        self._explicit_cache[:] = [
            cached
            for cached in self._explicit_cache
            if cached[0] != sid
            or not simulation_execution_requests_equal(
                cached[1],
                entry[1],
            )
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

        sid = _required_id(set_id, "set_id")
        if not isinstance(execution_request, SimulationExecutionRequest):
            raise TypeError("execution_request must be a SimulationExecutionRequest.")
        if not isinstance(result, SimulationResultValue):
            raise TypeError("result must be a SimulationResultValue.")
        self._current_by_set_id[sid] = (
            sid,
            execution_request,
            result,
        )
        self._current_run_set_ids.discard(sid)
        return result

    def hydrate_cached(self, set_id: str, execution_request: SimulationExecutionRequest) -> SimulationResultValue | None:
        """Hydrate a structurally equal full request; a miss changes nothing."""
        from kindred.core.simulation_preparation import simulation_execution_requests_equal
        sid = _required_id(set_id, "set_id")
        if not isinstance(execution_request, SimulationExecutionRequest):
            raise TypeError("execution_request must be a SimulationExecutionRequest.")
        entry = next(
            (
                cached
                for cached in reversed(self._explicit_cache)
                if cached[0] == sid
                and simulation_execution_requests_equal(cached[1], execution_request)
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

    def clear_semantic_set(self, set_id: str) -> None:
        """Clear visible semantic entries while retaining exact cache storage."""
        sid = _required_id(set_id, "set_id")
        self._current_by_set_id.pop(sid, None)
        self._current_run_set_ids.discard(sid)
        self._reference_by_set_id.pop(sid, None)

    def deauthorize_sets(
        self,
        set_ids: Iterable[str],
        *,
        preserve_reference: bool = False,
    ) -> None:
        """Clear foreground values while retaining exact reusable Run cache.

        Slider Preview reaches this canonical deauthorization boundary only
        after constructing its exact request candidates. Its established
        Run-to-Reference behavior moves the prior explicit entry separately,
        before this boundary. Deauthorization never manufactures a Reference
        value. An existing Reference may be retained for Preview admission but
        remains absent for ordinary invalidation.
        """

        targets = {
            _required_id(set_id, "set_id")
            for set_id in set_ids
        }
        if not targets:
            return
        for set_id in targets:
            self._current_by_set_id.pop(set_id, None)
            self._current_run_set_ids.discard(set_id)
            if not preserve_reference:
                self._reference_by_set_id.pop(set_id, None)

    def clear_all_semantic_entries(self) -> None:
        self._current_by_set_id.clear()
        self._current_run_set_ids.clear()
        self._reference_by_set_id.clear()

    def purge_set(self, set_id: str) -> None:
        """Delete/history removal purges visible entries and cached storage."""

        sid = _required_id(set_id, "set_id")
        self.clear_semantic_set(sid)
        self._explicit_cache[:] = [
            entry for entry in self._explicit_cache if entry[0] != sid
        ]

    def purge_sets(self, set_ids: Iterable[str]) -> None:
        for set_id in tuple(set_ids):
            self.purge_set(set_id)

    def clear(self) -> None:
        self.clear_all_semantic_entries()
        self._explicit_cache.clear()

    def _trim_explicit_cache(self) -> None:
        del self._explicit_cache[:-self._cache_capacity]
