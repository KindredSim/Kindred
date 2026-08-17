"""Owner-authorized result storage and bounded exact-key hydration cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, TypeAlias

import numpy as np

from kindred.core.intervention_schedule_compiler import InterventionExecutionContract
from kindred.core.simulator.solver_types import SimulationPresentationBoundary
from kindred.core.simulation_failure import FactualCause
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    simulation_execution_request_fingerprint,
)

__all__ = [
    "ExplicitResultEntry",
    "ExplicitResultKey",
    "explicit_result_key_for_request",
    "explicit_result_key_is_current",
    "PreviewResultIdentity",
    "preview_result_identity_for_request",
    "PreviewResultEntry",
    "PreviewResultKey",
    "ResultEntry",
    "ResultRepository",
    "SimulationResultProvenance",
    "SimulationResultValue",
]


def _required_id(value: object, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} must be nonblank.")
    return normalized


def explicit_result_key_for_request(
    *,
    tab_id: str,
    set_id: str,
    execution_request: SimulationExecutionRequest,
) -> "ExplicitResultKey":
    """Project one admitted request plus Run target coordinates into its key."""

    return ExplicitResultKey(
        tab_id=tab_id,
        set_id=set_id,
        request_fingerprint=simulation_execution_request_fingerprint(
            execution_request
        ),
    )


def preview_result_identity_for_request(
    *,
    tab_id: str,
    set_id: str,
    execution_request: SimulationExecutionRequest,
) -> "PreviewResultIdentity":
    """Project one admitted request plus Preview target coordinates."""

    return PreviewResultIdentity(
        tab_id=tab_id,
        set_id=set_id,
        request_fingerprint=simulation_execution_request_fingerprint(
            execution_request
        ),
    )


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
    provenance: SimulationResultProvenance,
) -> tuple[tuple[int, str, float, float, bool, bool], ...]:
    raw_catalog = provenance.solver.get("integration_segments")
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


@dataclass(frozen=True, slots=True)
class ExplicitResultKey:
    """Lightweight exact projection of one admitted Run request."""

    tab_id: str
    set_id: str
    request_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tab_id", _required_id(self.tab_id, "tab_id"))
        object.__setattr__(self, "set_id", _required_id(self.set_id, "set_id"))
        object.__setattr__(
            self,
            "request_fingerprint",
            _required_id(self.request_fingerprint, "request_fingerprint"),
        )
        if not self.request_fingerprint.startswith("simulation-request:"):
            raise ValueError(
                "Explicit result key requires a simulation-request fingerprint."
            )


def explicit_result_key_is_current(
    key: ExplicitResultKey,
    current_key: ExplicitResultKey | None,
) -> bool:
    """Return whether ``key`` is exactly the captured current result key."""

    return (
        isinstance(key, ExplicitResultKey)
        and isinstance(current_key, ExplicitResultKey)
        and key == current_key
    )


@dataclass(frozen=True, slots=True)
class PreviewResultIdentity:
    """Generation-free lightweight projection of one admitted Preview request."""

    tab_id: str
    set_id: str
    request_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tab_id", _required_id(self.tab_id, "tab_id"))
        object.__setattr__(self, "set_id", _required_id(self.set_id, "set_id"))
        object.__setattr__(
            self,
            "request_fingerprint",
            _required_id(self.request_fingerprint, "request_fingerprint"),
        )
        if not self.request_fingerprint.startswith("simulation-request:"):
            raise ValueError(
                "Preview identity requires a simulation-request fingerprint."
            )


@dataclass(frozen=True, slots=True)
class PreviewResultKey:
    """Exact latest-only identity minted by ``SimulationJobOwner``."""

    tab_id: str
    set_id: str
    request_fingerprint: str
    generation: int
    kind: str = "preview"

    def __post_init__(self) -> None:
        identity = PreviewResultIdentity(
            tab_id=self.tab_id,
            set_id=self.set_id,
            request_fingerprint=self.request_fingerprint,
        )
        object.__setattr__(self, "tab_id", identity.tab_id)
        object.__setattr__(self, "set_id", identity.set_id)
        object.__setattr__(self, "request_fingerprint", identity.request_fingerprint)
        if self.kind != "preview":
            raise ValueError("PreviewResultKey kind must be exactly 'preview'.")
        if type(self.generation) is not int:
            raise TypeError("Preview generation must be an int.")
        generation = self.generation
        if generation < 0:
            raise ValueError("Preview generation must be nonnegative.")
        object.__setattr__(self, "generation", generation)

    @classmethod
    def from_identity(
        cls,
        identity: PreviewResultIdentity,
        *,
        generation: int,
    ) -> PreviewResultKey:
        if not isinstance(identity, PreviewResultIdentity):
            raise TypeError("identity must be a PreviewResultIdentity.")
        return cls(
            tab_id=identity.tab_id,
            set_id=identity.set_id,
            request_fingerprint=identity.request_fingerprint,
            generation=generation,
        )

    @property
    def identity(self) -> PreviewResultIdentity:
        return PreviewResultIdentity(
            tab_id=self.tab_id,
            set_id=self.set_id,
            request_fingerprint=self.request_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class SimulationResultProvenance:
    """Typed immutable provenance retained with one successful result."""

    solver: Mapping[str, Any]
    preparation_warnings: tuple[str, ...] = ()
    requested_algebra_outputs: tuple[str, ...] = ()
    succeeded_algebra_outputs: tuple[str, ...] = ()
    failed_algebra_outputs: tuple[str, ...] = ()
    omitted_algebra_outputs: tuple[str, ...] = ()
    algebra_errors: tuple[FactualCause, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.solver, Mapping):
            raise TypeError("solver provenance must be a mapping.")
        preparation_warnings = tuple(
            str(value) for value in self.preparation_warnings
        )
        requested = self._normalized_algebra_names(
            self.requested_algebra_outputs,
            field="requested_algebra_outputs",
        )
        succeeded = self._normalized_algebra_names(
            self.succeeded_algebra_outputs,
            field="succeeded_algebra_outputs",
        )
        failed = self._normalized_algebra_names(
            self.failed_algebra_outputs,
            field="failed_algebra_outputs",
        )
        omitted = self._normalized_algebra_names(
            self.omitted_algebra_outputs,
            field="omitted_algebra_outputs",
        )
        requested_set = set(requested)
        partitions = {
            "succeeded_algebra_outputs": set(succeeded),
            "failed_algebra_outputs": set(failed),
            "omitted_algebra_outputs": set(omitted),
        }
        for field, values in partitions.items():
            unexpected = values.difference(requested_set)
            if unexpected:
                raise ValueError(
                    f"{field} contains unrequested output(s): "
                    + ", ".join(sorted(unexpected))
                )
        overlap = (
            partitions["succeeded_algebra_outputs"]
            & partitions["failed_algebra_outputs"]
        ) | (
            partitions["succeeded_algebra_outputs"]
            & partitions["omitted_algebra_outputs"]
        ) | (
            partitions["failed_algebra_outputs"]
            & partitions["omitted_algebra_outputs"]
        )
        if overlap:
            raise ValueError(
                "algebra output outcome partitions overlap: "
                + ", ".join(sorted(overlap))
            )
        classified = set().union(*partitions.values())
        missing = requested_set.difference(classified)
        if missing:
            raise ValueError(
                "requested algebra outputs lack a terminal outcome: "
                + ", ".join(sorted(missing))
            )
        if isinstance(self.algebra_errors, Mapping):
            raise TypeError(
                "algebra_errors must be a tuple of FactualCause values, "
                "not a mapping."
            )
        errors = tuple(self.algebra_errors)
        if any(not isinstance(value, FactualCause) for value in errors):
            raise TypeError("algebra_errors must contain only FactualCause values.")
        object.__setattr__(
            self,
            "solver",
            _frozen_provenance_mapping(self.solver),
        )
        object.__setattr__(
            self,
            "preparation_warnings",
            preparation_warnings,
        )
        object.__setattr__(self, "requested_algebra_outputs", requested)
        object.__setattr__(self, "succeeded_algebra_outputs", succeeded)
        object.__setattr__(self, "failed_algebra_outputs", failed)
        object.__setattr__(self, "omitted_algebra_outputs", omitted)
        object.__setattr__(self, "algebra_errors", errors)

    @staticmethod
    def _normalized_algebra_names(
        values: Iterable[str],
        *,
        field: str,
    ) -> tuple[str, ...]:
        normalized = tuple(_required_id(value, field) for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{field} contains duplicate output names.")
        return normalized

    @property
    def completed_with_warnings(self) -> bool:
        """Whether prominent completion warning presentation is required."""

        return bool(
            self.preparation_warnings
            or self.failed_algebra_outputs
            or self.omitted_algebra_outputs
            or self.algebra_errors
        )


@dataclass(frozen=True, slots=True, eq=False)
class SimulationResultValue:
    """Immutable numerical output with executor-issued execution identity."""

    t: np.ndarray
    series: Mapping[str, np.ndarray]
    segment_ids: np.ndarray
    segment_keys: tuple[str, ...]
    discontinuity_ids: np.ndarray
    intervention_execution: InterventionExecutionContract
    provenance: SimulationResultProvenance
    scalars: Mapping[str, float] = MappingProxyType({})
    parameters: tuple[tuple[str, float, str], ...] = ()
    presentation_boundaries: tuple[SimulationPresentationBoundary, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, SimulationResultProvenance):
            raise TypeError(
                "provenance must be a SimulationResultProvenance."
            )
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
        catalog = _executed_segment_catalog(self.provenance)
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

        object.__setattr__(self, "t", time)
        object.__setattr__(self, "series", MappingProxyType(series))
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "segment_keys", segment_keys)
        object.__setattr__(self, "discontinuity_ids", discontinuity_ids)
        object.__setattr__(self, "scalars", MappingProxyType(scalars))
        object.__setattr__(self, "parameters", tuple(parameters))
        object.__setattr__(
            self,
            "presentation_boundaries",
            presentation_boundaries,
        )


@dataclass(frozen=True, slots=True)
class ExplicitResultEntry:
    key: ExplicitResultKey
    result: SimulationResultValue


@dataclass(frozen=True, slots=True)
class PreviewResultEntry:
    key: PreviewResultKey
    result: SimulationResultValue


ResultEntry: TypeAlias = ExplicitResultEntry | PreviewResultEntry


class ResultRepository:
    """Store completed result values and a bounded exact-key Run cache."""

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
        self._explicit_by_set_id: dict[str, ExplicitResultEntry] = {}
        self._preview_by_set_id: dict[str, PreviewResultEntry] = {}
        self._reference_by_set_id: dict[str, ExplicitResultEntry] = {}
        self._explicit_cache: OrderedDict[
            ExplicitResultKey,
            ExplicitResultEntry,
        ] = OrderedDict()

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

    def explicit_entry(self, set_id: str) -> ExplicitResultEntry | None:
        return self._explicit_by_set_id.get(str(set_id or ""))

    def preview_entry(self, set_id: str) -> PreviewResultEntry | None:
        return self._preview_by_set_id.get(str(set_id or ""))

    def reference_entry(self, set_id: str) -> ExplicitResultEntry | None:
        return self._reference_by_set_id.get(str(set_id or ""))

    def retain_explicit_as_reference(self, set_id: str) -> None:
        """Move one completed foreground Run to comparison-only storage."""

        sid = _required_id(set_id, "set_id")
        explicit = self._explicit_by_set_id.pop(sid, None)
        if explicit is not None:
            self._reference_by_set_id[sid] = explicit

    def store_preview(
        self,
        key: PreviewResultKey,
        result: SimulationResultValue,
    ) -> PreviewResultEntry:
        """Store one completed Preview value already accepted by its owner."""

        self._require_preview_key(key)
        if not isinstance(result, SimulationResultValue):
            raise TypeError("result must be a SimulationResultValue.")
        entry = PreviewResultEntry(key=key, result=result)
        self._preview_by_set_id[key.set_id] = entry
        self._explicit_by_set_id.pop(key.set_id, None)
        return entry

    def clear_preview(self, set_id: str, *, clear_reference: bool = True) -> None:
        sid = _required_id(set_id, "set_id")
        self._preview_by_set_id.pop(sid, None)
        if clear_reference:
            self._reference_by_set_id.pop(sid, None)

    def put_explicit(
        self,
        key: ExplicitResultKey,
        result: SimulationResultValue,
    ) -> ExplicitResultEntry:
        self._require_explicit_key(key)
        if not isinstance(result, SimulationResultValue):
            raise TypeError("result must be a SimulationResultValue.")
        entry = ExplicitResultEntry(key=key, result=result)
        self._explicit_by_set_id[key.set_id] = entry
        self._preview_by_set_id.pop(key.set_id, None)
        self._reference_by_set_id.pop(key.set_id, None)
        self._explicit_cache[key] = entry
        self._explicit_cache.move_to_end(key)
        self._trim_explicit_cache()
        return entry

    def hydrate_cached(self, key: ExplicitResultKey) -> ExplicitResultEntry | None:
        """Hydrate one exact-key cache hit; a miss changes nothing."""

        self._require_explicit_key(key)
        entry = self._explicit_cache.get(key)
        if entry is not None:
            self._explicit_cache.move_to_end(key)
        else:
            entry = self._reference_by_set_id.get(key.set_id)
            if entry is None:
                entry = self._explicit_by_set_id.get(key.set_id)
            if entry is not None and entry.key != key:
                entry = None
        if entry is None:
            return None
        self._explicit_by_set_id[key.set_id] = entry
        self._preview_by_set_id.pop(key.set_id, None)
        self._reference_by_set_id.pop(key.set_id, None)
        return entry

    def clear_semantic_set(self, set_id: str) -> None:
        """Clear visible semantic entries while retaining exact cache storage."""
        sid = _required_id(set_id, "set_id")
        self._explicit_by_set_id.pop(sid, None)
        self._preview_by_set_id.pop(sid, None)
        self._reference_by_set_id.pop(sid, None)

    def deauthorize_sets(
        self,
        set_ids: Iterable[str],
        *,
        preserve_reference: bool = False,
    ) -> None:
        """Clear foreground values while retaining exact reusable Run cache.

        Slider Preview reaches this canonical deauthorization boundary only
        after constructing its exact request/key candidates. Its established
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
            self._explicit_by_set_id.pop(set_id, None)
            if not preserve_reference:
                self._reference_by_set_id.pop(set_id, None)
            self._preview_by_set_id.pop(set_id, None)

    def clear_all_semantic_entries(self) -> None:
        self._explicit_by_set_id.clear()
        self._preview_by_set_id.clear()
        self._reference_by_set_id.clear()

    def purge_set(self, set_id: str) -> None:
        """Delete/history removal purges visible entries and cached storage."""

        sid = _required_id(set_id, "set_id")
        self.clear_semantic_set(sid)
        for key in tuple(self._explicit_cache):
            if key.set_id == sid:
                self._explicit_cache.pop(key, None)

    def purge_sets(self, set_ids: Iterable[str]) -> None:
        for set_id in tuple(set_ids):
            self.purge_set(set_id)

    def clear(self) -> None:
        self.clear_all_semantic_entries()
        self._explicit_cache.clear()

    def _trim_explicit_cache(self) -> None:
        while len(self._explicit_cache) > self._cache_capacity:
            self._explicit_cache.popitem(last=False)

    def _require_explicit_key(self, key: ExplicitResultKey) -> None:
        if not isinstance(key, ExplicitResultKey):
            raise TypeError("key must be an ExplicitResultKey.")
        if key.tab_id != self._tab_id:
            raise ValueError(
                f"ExplicitResultKey belongs to tab {key.tab_id!r}, "
                f"not {self._tab_id!r}."
            )

    def _require_preview_key(self, key: PreviewResultKey) -> None:
        if not isinstance(key, PreviewResultKey):
            raise TypeError("key must be a PreviewResultKey.")
        if key.tab_id != self._tab_id:
            raise ValueError(
                f"PreviewResultKey belongs to tab {key.tab_id!r}, "
                f"not {self._tab_id!r}."
            )
