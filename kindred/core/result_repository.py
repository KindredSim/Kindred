"""Owner-authorized result storage and bounded exact-key hydration cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from kindred.core.intervention_schedule_compiler import InterventionExecutionContract

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    SimulationExecutionSettings,
    TabScientificSettings,
    validate_simulation_execution_settings,
    validate_tab_scientific_settings,
)

__all__ = [
    "ExplicitResultEntry",
    "ExplicitResultKey",
    "PreviewResultIdentity",
    "PreviewResultEntry",
    "PreviewResultKey",
    "ResultRepository",
    "SimulationResultProvenance",
    "SimulationResultValue",
]


def _required_id(value: object, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} must be nonblank.")
    return normalized


def _finite_pairs(
    values: Iterable[tuple[str, object]],
    *,
    field: str,
    nonnegative: bool = False,
) -> tuple[tuple[str, float], ...]:
    normalized: list[tuple[str, float]] = []
    names: set[str] = set()
    for raw_name, raw_value in values:
        name = _required_id(raw_name, f"{field} name")
        if name in names:
            raise ValueError(f"{field} contains duplicate name {name!r}.")
        if type(raw_value) is bool or not isinstance(raw_value, Real):
            raise TypeError(f"{field} value for {name!r} must be numeric.")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"{field} value for {name!r} must be finite.")
        if nonnegative and value < 0.0:
            raise ValueError(
                f"{field} value for {name!r} must be nonnegative."
            )
        names.add(name)
        normalized.append((name, value))
    return tuple(normalized)


def _frozen_vector(value: object, *, field: str) -> np.ndarray:
    candidate = np.asarray(value, dtype=float).reshape(-1)
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
    candidate = np.asarray(value, dtype=np.int64).reshape(-1)
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
    """Exact reusable scientific identity for one explicit Run target."""

    tab_id: str
    set_id: str
    mechanism_source: MechanismAuthoringSource
    analysis_identity: str
    initial_values: tuple[tuple[str, float], ...]
    execution_settings: SimulationExecutionSettings
    scientific_settings: TabScientificSettings

    def __post_init__(self) -> None:
        object.__setattr__(self, "tab_id", _required_id(self.tab_id, "tab_id"))
        object.__setattr__(self, "set_id", _required_id(self.set_id, "set_id"))
        if not isinstance(self.mechanism_source, MechanismAuthoringSource):
            raise TypeError("mechanism_source must be a MechanismAuthoringSource.")
        object.__setattr__(
            self,
            "analysis_identity",
            _required_id(self.analysis_identity, "analysis_identity"),
        )
        object.__setattr__(
            self,
            "initial_values",
            _finite_pairs(
                self.initial_values,
                field="initial_values",
                nonnegative=True,
            ),
        )
        if not isinstance(self.execution_settings, SimulationExecutionSettings):
            raise TypeError(
                "execution_settings must be a SimulationExecutionSettings value."
            )
        validate_simulation_execution_settings(self.execution_settings)
        validate_tab_scientific_settings(self.scientific_settings)


@dataclass(frozen=True, slots=True)
class PreviewResultIdentity:
    """Generation-free scientific identity captured by the Slider owner."""

    tab_id: str
    set_id: str
    mechanism_source: MechanismAuthoringSource
    analysis_identity: str
    parameter_values: tuple[tuple[str, float], ...]
    initial_values: tuple[tuple[str, float], ...]
    execution_settings: SimulationExecutionSettings
    scientific_settings: TabScientificSettings

    def __post_init__(self) -> None:
        object.__setattr__(self, "tab_id", _required_id(self.tab_id, "tab_id"))
        object.__setattr__(self, "set_id", _required_id(self.set_id, "set_id"))
        if not isinstance(self.mechanism_source, MechanismAuthoringSource):
            raise TypeError("mechanism_source must be a MechanismAuthoringSource.")
        object.__setattr__(
            self,
            "analysis_identity",
            _required_id(self.analysis_identity, "analysis_identity"),
        )
        object.__setattr__(
            self,
            "parameter_values",
            _finite_pairs(self.parameter_values, field="parameter_values"),
        )
        object.__setattr__(
            self,
            "initial_values",
            _finite_pairs(
                self.initial_values,
                field="initial_values",
                nonnegative=True,
            ),
        )
        if not isinstance(self.execution_settings, SimulationExecutionSettings):
            raise TypeError(
                "execution_settings must be a SimulationExecutionSettings value."
            )
        validate_simulation_execution_settings(self.execution_settings)
        validate_tab_scientific_settings(self.scientific_settings)


@dataclass(frozen=True, slots=True)
class PreviewResultKey:
    """Exact latest-only identity minted by ``SimulationJobOwner``."""

    tab_id: str
    set_id: str
    mechanism_source: MechanismAuthoringSource
    analysis_identity: str
    parameter_values: tuple[tuple[str, float], ...]
    initial_values: tuple[tuple[str, float], ...]
    execution_settings: SimulationExecutionSettings
    scientific_settings: TabScientificSettings
    generation: int
    kind: str = "preview"

    def __post_init__(self) -> None:
        identity = PreviewResultIdentity(
            tab_id=self.tab_id,
            set_id=self.set_id,
            mechanism_source=self.mechanism_source,
            analysis_identity=self.analysis_identity,
            parameter_values=self.parameter_values,
            initial_values=self.initial_values,
            execution_settings=self.execution_settings,
            scientific_settings=self.scientific_settings,
        )
        object.__setattr__(self, "tab_id", identity.tab_id)
        object.__setattr__(self, "set_id", identity.set_id)
        object.__setattr__(self, "mechanism_source", identity.mechanism_source)
        object.__setattr__(self, "analysis_identity", identity.analysis_identity)
        object.__setattr__(self, "parameter_values", identity.parameter_values)
        object.__setattr__(self, "initial_values", identity.initial_values)
        object.__setattr__(
            self,
            "execution_settings",
            identity.execution_settings,
        )
        object.__setattr__(
            self,
            "scientific_settings",
            identity.scientific_settings,
        )
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
            mechanism_source=identity.mechanism_source,
            analysis_identity=identity.analysis_identity,
            parameter_values=identity.parameter_values,
            initial_values=identity.initial_values,
            execution_settings=identity.execution_settings,
            scientific_settings=identity.scientific_settings,
            generation=generation,
        )

    @property
    def identity(self) -> PreviewResultIdentity:
        return PreviewResultIdentity(
            tab_id=self.tab_id,
            set_id=self.set_id,
            mechanism_source=self.mechanism_source,
            analysis_identity=self.analysis_identity,
            parameter_values=self.parameter_values,
            initial_values=self.initial_values,
            execution_settings=self.execution_settings,
            scientific_settings=self.scientific_settings,
        )


@dataclass(frozen=True, slots=True)
class SimulationResultProvenance:
    """Typed immutable provenance retained with one successful result."""

    solver: Mapping[str, Any]
    preparation_warnings: tuple[str, ...] = ()
    algebra_errors: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.solver, Mapping):
            raise TypeError("solver provenance must be a mapping.")
        warnings = tuple(str(value) for value in self.preparation_warnings)
        errors: list[Mapping[str, Any]] = []
        for value in self.algebra_errors:
            if not isinstance(value, Mapping):
                raise TypeError(
                    "algebra_errors must contain only mapping values."
                )
            errors.append(_frozen_provenance_mapping(value))
        object.__setattr__(
            self,
            "solver",
            _frozen_provenance_mapping(self.solver),
        )
        object.__setattr__(self, "preparation_warnings", warnings)
        object.__setattr__(self, "algebra_errors", tuple(errors))


@dataclass(frozen=True, slots=True, eq=False)
class SimulationResultValue:
    """Immutable numerical output, including full-resolution time and provenance."""

    t: np.ndarray
    series: Mapping[str, np.ndarray]
    segment_ids: np.ndarray
    segment_keys: tuple[str, ...]
    discontinuity_ids: np.ndarray
    intervention_execution: InterventionExecutionContract
    provenance: SimulationResultProvenance
    scalars: Mapping[str, float] = MappingProxyType({})
    parameters: tuple[tuple[str, float, str], ...] = ()

    def __post_init__(self) -> None:
        time = _frozen_vector(self.t, field="result time")
        if time.size > 1 and np.any(np.diff(time) <= 0.0):
            raise ValueError("result time must be strictly increasing.")
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
        key_by_id: dict[int, str] = {}
        for segment_id, segment_key in zip(
            segment_ids,
            segment_keys,
            strict=True,
        ):
            existing = key_by_id.setdefault(int(segment_id), segment_key)
            if existing != segment_key:
                raise ValueError(
                    "one result segment identity cannot name multiple segments."
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
        if time.size > 1:
            expected_discontinuities = np.searchsorted(
                np.asarray(
                    self.intervention_execution.runtime_discontinuities_for_span(
                        float(time[0]),
                        float(time[-1]),
                    ),
                    dtype=float,
                ),
                time,
                side="right",
            )
            if not np.array_equal(discontinuity_ids, expected_discontinuities):
                raise ValueError(
                    "result discontinuity identity conflicts with execution truth."
                )
        if self.provenance.solver.get("state_at_boundary") != "post-stage-7":
            raise ValueError(
                "result provenance must declare post-stage-7 boundary state."
            )


@dataclass(frozen=True, slots=True)
class ExplicitResultEntry:
    key: ExplicitResultKey
    result: SimulationResultValue


@dataclass(frozen=True, slots=True)
class PreviewResultEntry:
    key: PreviewResultKey
    result: SimulationResultValue


class ResultRepository:
    """Store results while enforcing owner-issued Preview display eligibility."""

    DEFAULT_CACHE_CAPACITY = 32

    def __init__(
        self,
        tab_id: str,
        *,
        cache_capacity: int = DEFAULT_CACHE_CAPACITY,
    ) -> None:
        self._tab_id = _required_id(tab_id, "tab_id")
        self._cache_capacity = int(cache_capacity)
        if self._cache_capacity < 1:
            raise ValueError("cache_capacity must be at least one.")
        self._explicit_by_set_id: dict[str, ExplicitResultEntry] = {}
        self._preview_by_set_id: dict[str, PreviewResultEntry] = {}
        self._preview_display_key_by_set_id: dict[
            str,
            PreviewResultKey,
        ] = {}
        self._reference_by_set_id: dict[str, ExplicitResultEntry] = {}
        self._explicit_cache: OrderedDict[
            ExplicitResultKey,
            SimulationResultValue,
        ] = OrderedDict()

    @property
    def tab_id(self) -> str:
        return self._tab_id

    @property
    def cache_capacity(self) -> int:
        return self._cache_capacity

    def explicit_entry(self, set_id: str) -> ExplicitResultEntry | None:
        return self._explicit_by_set_id.get(str(set_id or ""))

    def explicit_entries(self) -> tuple[ExplicitResultEntry, ...]:
        return tuple(self._explicit_by_set_id.values())

    def preview_entry(self, set_id: str) -> PreviewResultEntry | None:
        sid = str(set_id or "")
        entry = self._preview_by_set_id.get(sid)
        if (
            entry is None
            or self._preview_display_key_by_set_id.get(sid) != entry.key
        ):
            return None
        return entry

    def preview_entries(self) -> tuple[PreviewResultEntry, ...]:
        return tuple(
            entry
            for set_id, entry in self._preview_by_set_id.items()
            if self._preview_display_key_by_set_id.get(set_id) == entry.key
        )

    def reference_entry(self, set_id: str) -> ExplicitResultEntry | None:
        return self._reference_by_set_id.get(str(set_id or ""))

    def capture_preview_reference(self, key: PreviewResultKey) -> None:
        """Move the exact foreground Run to Reference before Preview debounce."""
        self._require_preview_key(key)
        explicit = self._explicit_by_set_id.pop(key.set_id, None)
        if explicit is not None:
            self._reference_by_set_id[key.set_id] = explicit

    def authorize_preview_display(self, key: PreviewResultKey) -> None:
        """Authorize exactly one owner-current key for display/export."""

        self._require_preview_key(key)
        self._preview_display_key_by_set_id[key.set_id] = key

    def store_preview(
        self,
        key: PreviewResultKey,
        result: SimulationResultValue,
    ) -> PreviewResultEntry:
        """Store one Preview result after owner authorization."""
        self._require_preview_key(key)
        if not isinstance(result, SimulationResultValue):
            raise TypeError("result must be a SimulationResultValue.")
        if self._preview_display_key_by_set_id.get(key.set_id) != key:
            raise ValueError(
                "Preview result key is not owner-authorized for display."
            )
        entry = PreviewResultEntry(key=key, result=result)
        self._preview_by_set_id[key.set_id] = entry
        self._explicit_by_set_id.pop(key.set_id, None)
        return entry

    def clear_preview(self, set_id: str, *, clear_reference: bool = True) -> None:
        sid = _required_id(set_id, "set_id")
        self._preview_by_set_id.pop(sid, None)
        self._preview_display_key_by_set_id.pop(sid, None)
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
        self._preview_display_key_by_set_id.pop(key.set_id, None)
        self._reference_by_set_id.pop(key.set_id, None)
        self._explicit_cache[key] = result
        self._explicit_cache.move_to_end(key)
        self._trim_explicit_cache()
        return entry

    def cached_result(self, key: ExplicitResultKey) -> SimulationResultValue | None:
        self._require_explicit_key(key)
        result = self._explicit_cache.get(key)
        if result is not None:
            self._explicit_cache.move_to_end(key)
        return result

    def hydrate_cached(self, key: ExplicitResultKey) -> ExplicitResultEntry | None:
        """Hydrate one owner-authorized exact cache hit; a miss changes nothing."""
        result = self.cached_result(key)
        if result is None:
            retained = self._reference_by_set_id.get(key.set_id)
            if retained is None:
                retained = self._explicit_by_set_id.get(key.set_id)
            if retained is not None and retained.key == key:
                result = retained.result
        if result is None:
            return None
        entry = ExplicitResultEntry(key=key, result=result)
        self._explicit_by_set_id[key.set_id] = entry
        self._preview_by_set_id.pop(key.set_id, None)
        self._preview_display_key_by_set_id.pop(key.set_id, None)
        self._reference_by_set_id.pop(key.set_id, None)
        return entry

    def clear_semantic_set(self, set_id: str) -> None:
        """Clear visible semantic entries while retaining exact cache storage."""
        sid = _required_id(set_id, "set_id")
        self._explicit_by_set_id.pop(sid, None)
        self._preview_by_set_id.pop(sid, None)
        self._preview_display_key_by_set_id.pop(sid, None)
        self._reference_by_set_id.pop(sid, None)

    def clear_semantic_sets(self, set_ids: Iterable[str]) -> None:
        seen: set[str] = set()
        for set_id in set_ids:
            sid = _required_id(set_id, "set_id")
            if sid not in seen:
                seen.add(sid)
                self.clear_semantic_set(sid)

    def deauthorize_sets(
        self,
        set_ids: Iterable[str],
        *,
        explicit_key_is_current: Callable[[ExplicitResultKey], bool],
    ) -> None:
        """Clear display truth and purge only owner-declared stale cache keys."""

        if not callable(explicit_key_is_current):
            raise TypeError("explicit_key_is_current must be callable.")
        targets = {
            _required_id(set_id, "set_id")
            for set_id in set_ids
        }
        self.clear_semantic_sets(targets)
        for key in tuple(self._explicit_cache):
            if key.set_id not in targets:
                continue
            try:
                current = bool(explicit_key_is_current(key))
            except Exception:
                current = False
            if not current:
                self._explicit_cache.pop(key, None)

    def clear_all_semantic_entries(self) -> None:
        self._explicit_by_set_id.clear()
        self._preview_by_set_id.clear()
        self._preview_display_key_by_set_id.clear()
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
