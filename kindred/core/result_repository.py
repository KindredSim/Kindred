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


def _frozen_vector(value: object, *, field: str) -> np.ndarray:
    candidate = np.asarray(value, dtype=float)
    if candidate.ndim != 1:
        raise ValueError(f"{field} must be one-dimensional.")
    array = (
        value
        if (
            isinstance(value, np.ndarray)
            and value.dtype == np.dtype(float)
            and value.ndim == 1
            and not value.flags.writeable
        )
        else candidate.copy()
    )
    if array.size == 0:
        raise ValueError(f"{field} must not be empty.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{field} must contain only finite values.")
    array.setflags(write=False)
    return array


def _frozen_identity_vector(
    value: object,
    *,
    field: str,
    size: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{field} must be a one-dimensional integer array.")
    candidate = raw.astype(np.int64, copy=False)
    array = (
        value
        if (
            isinstance(value, np.ndarray)
            and value.dtype == np.dtype(np.int64)
            and value.ndim == 1
            and not value.flags.writeable
        )
        else candidate.copy()
    )
    if array.size != int(size):
        raise ValueError(f"{field} must align with result time.")
    if np.any(array < 0) or np.any(np.diff(array) < 0):
        raise ValueError(f"{field} must be ordered and nonnegative.")
    array.setflags(write=False)
    return array


def _validate_discontinuity_regions(
    time: np.ndarray,
    discontinuity_ids: np.ndarray,
    *,
    execution: InterventionExecutionContract,
    execution_start: float,
    execution_end: float,
) -> None:
    """Validate executor region identity against exact executed boundaries."""

    if int(discontinuity_ids[0]) != 0:
        raise ValueError("result discontinuity identity must start at zero.")
    expected_advances = {
        float(value)
        for value in execution.discontinuities
        if execution_start < float(value) <= execution_end
    }
    expected_advances.update(
        float(firing.root_time)
        for firing in execution.trigger_firings
        if execution_start < float(firing.root_time) <= execution_end
    )
    observed_advances: set[float] = set()
    for index in range(1, int(time.size)):
        delta = int(discontinuity_ids[index]) - int(
            discontinuity_ids[index - 1]
        )
        if delta not in {0, 1}:
            raise ValueError(
                "result discontinuity identity must advance one region at a time."
            )
        if delta == 1:
            observed_advances.add(float(time[index]))
    if observed_advances != expected_advances:
        raise ValueError(
            "result discontinuity regions conflict with executed intervention boundaries."
        )


def _executed_segment_catalog(
    provenance: Mapping[str, Any],
) -> tuple[tuple[int, str, float, float, bool, bool], ...]:
    solver_provenance = provenance.get("solver", provenance)
    if not isinstance(solver_provenance, Mapping):
        raise TypeError("solver provenance must be a mapping.")
    raw_catalog = solver_provenance.get("integration_segments")
    if not isinstance(raw_catalog, (tuple, list)) or not raw_catalog:
        raise ValueError(
            "solver provenance must contain the actual executed segment catalog."
        )
    catalog: list[tuple[int, str, float, float, bool, bool]] = []
    seen_ids: set[int] = set()
    seen_keys: set[str] = set()
    for position, raw_entry in enumerate(raw_catalog):
        if not isinstance(raw_entry, Mapping):
            raise TypeError("executed segment catalog entries must be mappings.")
        raw_id = raw_entry.get("segment_id")
        if type(raw_id) is not int or raw_id < 0:
            raise ValueError(
                "executed segment catalog IDs must be nonnegative integers."
            )
        segment_id = int(raw_id)
        if segment_id != position or segment_id in seen_ids:
            raise ValueError(
                "executed segment catalog IDs must be unique and contiguous."
            )
        segment_key = _required_id(
            raw_entry.get("segment_key"),
            "executed segment key",
        )
        if segment_key in seen_keys:
            raise ValueError("executed segment catalog keys must be unique.")
        try:
            start = float(raw_entry["t_start"])
            end = float(raw_entry["t_end"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "executed segment catalog spans must be explicit finite values."
            ) from exc
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError(
                "executed segment catalog spans must be finite and increasing."
            )
        identity = raw_entry.get("segment_identity")
        if not isinstance(identity, Mapping):
            raise TypeError(
                "executed segment catalog entries require direct segment identity."
            )
        if (
            identity.get("key") != segment_key
            or identity.get("kind") != "integration"
            or identity.get("ordinal") != segment_id
            or identity.get("t_start") != start
            or identity.get("t_end") != end
        ):
            raise ValueError(
                "executed segment catalog identity conflicts with its direct span."
            )
        temperature_boundary = raw_entry.get("temperature_step_boundary")
        starts_after_discontinuity = raw_entry.get(
            "starts_after_discontinuity"
        )
        if type(temperature_boundary) is not bool or type(
            starts_after_discontinuity
        ) is not bool:
            raise TypeError(
                "executed segment catalog boundary facts must be explicit booleans."
            )
        seen_ids.add(segment_id)
        seen_keys.add(segment_key)
        catalog.append(
            (
                segment_id,
                segment_key,
                start,
                end,
                temperature_boundary,
                starts_after_discontinuity,
            )
        )
    for previous, current in zip(catalog, catalog[1:]):
        if current[2] != previous[3]:
            raise ValueError(
                "executed segment catalog must describe one contiguous execution."
            )
    return tuple(catalog)


def _validate_rows_against_executed_segments(
    *,
    time: np.ndarray,
    segment_ids: np.ndarray,
    segment_keys: tuple[str, ...],
    catalog: tuple[tuple[int, str, float, float, bool, bool], ...],
) -> None:
    by_id = {entry[0]: entry for entry in catalog}
    for row, (raw_id, key) in enumerate(
        zip(segment_ids, segment_keys, strict=True)
    ):
        segment = by_id.get(int(raw_id))
        if segment is None or key != segment[1]:
            raise ValueError(
                "every result row must name one actual executed segment."
            )
        row_time = float(time[row])
        if row_time < segment[2] or row_time > segment[3]:
            raise ValueError(
                "result row time lies outside its executed segment span."
            )

    if (
        float(time[0]) != catalog[0][2]
        or int(segment_ids[0]) != catalog[0][0]
        or segment_keys[0] != catalog[0][1]
    ):
        raise ValueError(
            "the first public result row must identify execution start."
        )
    if float(time[-1]) != catalog[-1][3]:
        raise ValueError(
            "the final public result row must identify execution end."
        )

    for (
        segment_id,
        segment_key,
        start,
        _end,
        temperature_boundary,
        starts_after_discontinuity,
    ) in catalog:
        if segment_id == 0 or not (
            temperature_boundary or starts_after_discontinuity
        ):
            continue
        matching_rows = [
            row
            for row in range(int(time.size))
            if int(segment_ids[row]) == segment_id
            and segment_keys[row] == segment_key
            and float(time[row]) == start
        ]
        if len(matching_rows) != 1:
            raise ValueError(
                "every public execution boundary must publish one row in its new segment."
            )


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
        if not isinstance(self.solver_provenance, Mapping):
            raise TypeError("solver_provenance must be a mapping.")
        if not isinstance(
            self.intervention_execution, InterventionExecutionContract
        ):
            raise TypeError(
                "intervention_execution must be an InterventionExecutionContract."
            )
        presentation_boundaries = tuple(self.presentation_boundaries)
        if any(
            not isinstance(boundary, SimulationPresentationBoundary)
            for boundary in presentation_boundaries
        ):
            raise TypeError(
                "presentation_boundaries must contain only "
                "SimulationPresentationBoundary values."
            )
        if any(
            current.time <= previous.time
            for previous, current in zip(
                presentation_boundaries,
                presentation_boundaries[1:],
            )
        ):
            raise ValueError(
                "presentation boundaries must be strictly ordered by time."
            )
        time = _frozen_vector(self.t, field="result time")
        if time.size > 1 and np.any(np.diff(time) <= 0.0):
            raise ValueError("result time must be strictly increasing.")
        if presentation_boundaries and (
            presentation_boundaries[0].time < float(time[0])
            or presentation_boundaries[-1].time > float(time[-1])
        ):
            raise ValueError(
                "presentation boundaries must lie within the result time span."
            )
        if any(
            not bool(np.any(time == boundary.time))
            for boundary in presentation_boundaries
        ):
            raise ValueError(
                "presentation boundaries must align with canonical result times."
            )
        segment_ids = _frozen_identity_vector(
            self.segment_ids,
            field="result segment identity",
            size=time.size,
        )
        discontinuity_ids = _frozen_identity_vector(
            self.discontinuity_ids,
            field="result discontinuity identity",
            size=time.size,
        )
        segment_keys = tuple(
            _required_id(value, "result segment key")
            for value in self.segment_keys
        )
        if len(segment_keys) != time.size:
            raise ValueError("result segment keys must align with result time.")
        catalog = _executed_segment_catalog(self.solver_provenance)
        _validate_rows_against_executed_segments(
            time=time,
            segment_ids=segment_ids,
            segment_keys=segment_keys,
            catalog=catalog,
        )
        _validate_discontinuity_regions(
            time,
            discontinuity_ids,
            execution=self.intervention_execution,
            execution_start=catalog[0][2],
            execution_end=catalog[-1][3],
        )

        series: dict[str, np.ndarray] = {}
        for raw_name, raw_values in dict(self.series or {}).items():
            name = _required_id(raw_name, "series name")
            values = _frozen_vector(raw_values, field=f"series {name!r}")
            if values.size != time.size:
                raise ValueError(
                    f"series {name!r} has {values.size} values for "
                    f"{time.size} time points."
                )
            series[name] = values
        if not series:
            raise ValueError("result must contain at least one series.")

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

        object.__setattr__(self, "t", time)
        object.__setattr__(self, "series", MappingProxyType(series))
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "segment_keys", segment_keys)
        object.__setattr__(self, "discontinuity_ids", discontinuity_ids)
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
