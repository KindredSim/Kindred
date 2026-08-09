"""Immutable raw-dataset values and pure dataset admission validators.

This module deliberately has no dependency on project-document or mapping
policy.  Project admission and dataset mapping import these values downward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence, TypeAlias

from kindred.core.datasets.source_rules import (
    identify_physical_columns,
    validate_tabular_source_bytes,
)

__all__ = [
    "DatasetMappedTarget",
    "FitInclusionValidation",
    "FrozenJsonArray",
    "FrozenJsonObject",
    "FrozenJsonValue",
    "ImportedSource",
    "ObservationSeries",
    "RawDatasetRecord",
    "RawDatasetValidationError",
    "freeze_json_value",
    "thaw_json_value",
    "validate_fit_inclusion",
    "validate_imported_source",
    "validate_raw_dataset_record",
    "validate_raw_observation_id",
]

JsonScalar: TypeAlias = str | int | float | bool | None

class RawDatasetValidationError(ValueError):
    """Raised when immutable raw-dataset values fail foundation admission."""


@dataclass(frozen=True, slots=True)
class FrozenJsonArray:
    items: tuple["FrozenJsonValue", ...]


@dataclass(frozen=True, slots=True)
class FrozenJsonObject:
    items: tuple[tuple[str, "FrozenJsonValue"], ...]


FrozenJsonValue: TypeAlias = JsonScalar | FrozenJsonArray | FrozenJsonObject


def _require_finite(value: object, label: str) -> float:
    if type(value) is bool or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    return parsed


def freeze_json_value(value: object) -> FrozenJsonValue:
    """Normalize a JSON-compatible value into an immutable representation."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return int(value)
    if type(value) is float:
        return _require_finite(value, "JSON number")
    if isinstance(value, Mapping):
        items: list[tuple[str, FrozenJsonValue]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings.")
            items.append((key, freeze_json_value(item)))
        items.sort(key=lambda item: item[0])
        return FrozenJsonObject(tuple(items))
    if isinstance(value, (list, tuple)):
        return FrozenJsonArray(tuple(freeze_json_value(item) for item in value))
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}.")


def thaw_json_value(value: FrozenJsonValue) -> object:
    if isinstance(value, FrozenJsonObject):
        return {str(key): thaw_json_value(item) for key, item in value.items}
    if isinstance(value, FrozenJsonArray):
        return [thaw_json_value(item) for item in value.items]
    return value


def validate_raw_observation_id(
    value: object,
    label: str = "raw observation ID",
) -> str:
    """Require one already-allocated opaque raw observation identity."""
    if type(value) is not str:
        raise TypeError(f"{label} must be a str.")
    if not value or value != value.strip():
        raise ValueError(f"{label} must be nonblank and edge-normalized.")
    return value


@dataclass(frozen=True, slots=True)
class ObservationSeries:
    raw_observation_id: str
    column_name: str
    t: tuple[float, ...]
    y: tuple[float, ...]
    source_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_raw_observation_id(self.raw_observation_id)


@dataclass(frozen=True, slots=True)
class ImportedSource:
    """One immutable source capture owned by the canonical project."""

    source_id: str
    original_filename: str
    original_format: str
    exact_source_bytes: bytes


@dataclass(frozen=True, slots=True)
class RawDatasetRecord:
    dataset_id: str
    display_name: str
    source_id: str
    source_metadata: FrozenJsonValue
    observations: tuple[ObservationSeries, ...]


@dataclass(frozen=True, slots=True)
class DatasetMappedTarget:
    dataset_id: str
    raw_observation_id: str
    model_target: str
    t: tuple[float, ...]
    y: tuple[float, ...]
    source_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id).strip()
        if (
            not isinstance(self.raw_observation_id, str)
            or not self.raw_observation_id
            or self.raw_observation_id != self.raw_observation_id.strip()
        ):
            raise ValueError(
                "Mapped dataset target raw observation identity must be a "
                "nonblank edge-normalized string."
            )
        raw_observation_id = self.raw_observation_id
        model_target = str(self.model_target).strip()
        if not dataset_id or not raw_observation_id or not model_target:
            raise ValueError("Mapped dataset target identities must be nonblank.")
        if (
            type(self.t) is not tuple
            or type(self.y) is not tuple
            or type(self.source_rows) is not tuple
            or not self.t
            or len(self.t) != len(self.y)
            or len(self.t) != len(self.source_rows)
        ):
            raise ValueError(
                "Mapped dataset target time, value, and source-row tuples "
                "must be immutable and have equal nonzero length."
            )
        if (
            any(
                type(value) is not float or not math.isfinite(value)
                for value in (*self.t, *self.y)
            )
            or any(type(row) is not int or row <= 0 for row in self.source_rows)
            or len(set(self.source_rows)) != len(self.source_rows)
        ):
            raise ValueError(
                "Mapped dataset targets require finite values and unique "
                "positive physical source rows."
            )
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "raw_observation_id", raw_observation_id)
        object.__setattr__(self, "model_target", model_target)


@dataclass(frozen=True, slots=True)
class FitInclusionValidation:
    dataset_id: str
    set_id: str
    resolved_t_start: float
    targets: tuple[DatasetMappedTarget, ...]

    def __post_init__(self) -> None:
        if (
            type(self.resolved_t_start) is not float
            or not math.isfinite(self.resolved_t_start)
            or self.resolved_t_start < 0.0
        ):
            raise ValueError(
                "Fit inclusion resolved_t_start must be a finite "
                "nonnegative float."
            )
        targets = tuple(self.targets)
        if not targets or any(target.dataset_id != self.dataset_id for target in targets):
            raise ValueError(
                "Fit inclusion targets must be nonempty and share the stable "
                "dataset ID."
            )
        object.__setattr__(self, "targets", targets)


def validate_raw_dataset_record(record: RawDatasetRecord) -> RawDatasetRecord:
    """Fail closed unless one immutable raw record satisfies dataset law."""
    if not isinstance(record, RawDatasetRecord):
        raise TypeError("record must be a RawDatasetRecord.")
    def fail(message: str) -> None:
        raise RawDatasetValidationError(message)

    if (
        not isinstance(record.dataset_id, str)
        or not record.dataset_id
        or record.dataset_id != record.dataset_id.strip()
    ):
        fail("Raw dataset ID must be a nonblank edge-normalized string.")
    if (
        not isinstance(record.display_name, str)
        or not record.display_name
        or record.display_name != record.display_name.strip()
    ):
        fail(
            f"Raw dataset {record.dataset_id!r} display name must be a nonblank edge-normalized string."
        )
    if (
        not isinstance(record.source_id, str)
        or not record.source_id
        or record.source_id != record.source_id.strip()
    ):
        fail(f"Raw dataset {record.dataset_id!r} source ID must be nonblank and edge-normalized.")
    if type(record.observations) is not tuple:
        fail(f"Raw dataset {record.dataset_id!r} observations must be an immutable tuple.")
    if not record.observations:
        fail(f"Raw dataset {record.dataset_id!r} requires at least one observation series.")

    thawed_metadata = thaw_json_value(record.source_metadata)
    if not isinstance(thawed_metadata, dict):
        fail(f"Raw dataset {record.dataset_id!r} source metadata must be an immutable JSON object.")
    physical_values = thawed_metadata.get("physical_columns")
    if not isinstance(physical_values, list) or not physical_values:
        fail(f"Raw dataset {record.dataset_id!r} must retain its physical source-column catalog.")
    physical_rows: list[tuple[int, str, str]] = []
    for index, raw_physical in enumerate(physical_values):
        if not isinstance(raw_physical, dict) or set(raw_physical) != {"source_index", "authored_name", "assigned_name"}:
            fail(f"Raw dataset {record.dataset_id!r} physical column {index} has an invalid schema.")
        source_index = raw_physical["source_index"]
        authored_name = raw_physical["authored_name"]
        assigned_name = raw_physical["assigned_name"]
        if (
            type(source_index) is not int
            or source_index != index
            or not isinstance(authored_name, str)
            or not authored_name
            or authored_name != authored_name.strip()
            or not isinstance(assigned_name, str)
            or not assigned_name
            or assigned_name != assigned_name.strip()
        ):
            fail(f"Raw dataset {record.dataset_id!r} physical column {index} does not retain exact ordered source identity.")
        physical_rows.append((source_index, authored_name, assigned_name))
    expected_columns = identify_physical_columns(
        [authored_name for _index, authored_name, _assigned in physical_rows]
    )
    expected_rows = tuple(
        (column.source_index, column.authored_name, column.assigned_name)
        for column in expected_columns
    )
    if tuple(physical_rows) != expected_rows:
        fail(f"Raw dataset {record.dataset_id!r} physical column names do not match the canonical case-sensitive duplicate allocation.")

    observation_names: list[str] = []
    raw_observation_ids: set[str] = set()
    for series in record.observations:
        if not isinstance(series, ObservationSeries):
            fail(f"Raw dataset {record.dataset_id!r} observations must contain only ObservationSeries values.")
        name = series.column_name
        if not isinstance(name, str) or not name or name != name.strip():
            fail(f"Raw dataset {record.dataset_id!r} observation names must be nonblank edge-normalized strings.")
        if name in observation_names:
            fail(f"Raw dataset {record.dataset_id!r} contains duplicate assigned observation name {name!r}.")
        try:
            raw_observation_id = validate_raw_observation_id(
                series.raw_observation_id,
                f"Raw dataset {record.dataset_id!r} observation {name!r} raw observation ID",
            )
        except (TypeError, ValueError) as exc:
            fail(str(exc))
        if raw_observation_id in raw_observation_ids:
            fail(f"Raw dataset {record.dataset_id!r} contains duplicate raw observation ID {raw_observation_id!r}.")
        raw_observation_ids.add(raw_observation_id)
        if type(series.t) is not tuple or type(series.y) is not tuple or type(series.source_rows) is not tuple:
            fail(f"Raw dataset {record.dataset_id!r} observation {name!r} must own immutable t/y/source_rows tuples.")
        if (
            not series.t
            or len(series.t) != len(series.y)
            or len(series.t) != len(series.source_rows)
            or any(type(value) is not float or not math.isfinite(value) for value in (*series.t, *series.y))
        ):
            fail(f"Raw dataset {record.dataset_id!r} observation {name!r} must retain equal nonempty finite float time/value tuples aligned with source rows.")
        if any(type(row) is not int or row <= 0 for row in series.source_rows):
            fail(f"Raw dataset {record.dataset_id!r} observation {name!r} source rows must be positive integers.")
        if len(set(series.source_rows)) != len(series.source_rows):
            fail(f"Raw dataset {record.dataset_id!r} observation {name!r} contains duplicate physical source-row identities.")
        observation_names.append(name)
    if thawed_metadata.get("observation_columns") != observation_names:
        fail(f"Raw dataset {record.dataset_id!r} observation-column metadata must exactly match its immutable ordered series labels.")
    physical_assigned = {assigned_name for _source_index, _authored_name, assigned_name in physical_rows}
    time_column = thawed_metadata.get("time_column")
    if (
        not isinstance(time_column, str)
        or time_column not in physical_assigned
        or time_column in observation_names
        or any(name not in physical_assigned for name in observation_names)
    ):
        fail(f"Raw dataset {record.dataset_id!r} time/observation columns do not reference retained physical source columns.")
    if thawed_metadata.get("mapping_source") != "explicit":
        fail(f"Raw dataset {record.dataset_id!r} mapping source must be exactly 'explicit'.")
    sheet_name = thawed_metadata.get("sheet_name")
    if sheet_name is not None and (not isinstance(sheet_name, str) or not sheet_name.strip()):
        fail(f"Raw dataset {record.dataset_id!r} worksheet identity must be nonblank when present.")
    return record


def validate_imported_source(source: ImportedSource) -> ImportedSource:
    """Validate one canonical source capture, including exact bytes."""

    if not isinstance(source, ImportedSource):
        raise TypeError("source must be an ImportedSource.")
    if (
        not isinstance(source.source_id, str)
        or not source.source_id
        or source.source_id != source.source_id.strip()
    ):
        raise RawDatasetValidationError("Imported source ID must be nonblank and edge-normalized.")
    if source.original_format not in {"csv", "xlsx"}:
        raise RawDatasetValidationError("Imported source format must be exactly 'csv' or 'xlsx'.")
    if (
        not isinstance(source.original_filename, str)
        or not source.original_filename
        or source.original_filename != source.original_filename.strip()
        or os.path.basename(source.original_filename) != source.original_filename
    ):
        raise RawDatasetValidationError("Imported source filename must be a nonblank basename.")
    expected_suffix = f".{source.original_format}"
    if Path(source.original_filename).suffix.lower() != expected_suffix:
        raise RawDatasetValidationError(
            f"Imported source filename must end in {expected_suffix!r}."
        )
    try:
        validate_tabular_source_bytes(
            source.exact_source_bytes,
            source_format=source.original_format,
            source_label=source.original_filename,
        )
    except (TypeError, ValueError) as exc:
        raise RawDatasetValidationError(f"Imported source admission failed: {exc}") from exc
    return source


def validate_fit_inclusion(
    *,
    dataset: RawDatasetRecord,
    set_id: str | None,
    observation_mapping: Sequence[tuple[str, str]],
    initial_values: Mapping[str, object],
    mechanism_species: Sequence[str],
    resolved_t_start: object,
) -> FitInclusionValidation:
    """Validate one Fit inclusion from already-resolved immutable values."""
    if not isinstance(dataset, RawDatasetRecord):
        raise TypeError("dataset must be a RawDatasetRecord.")
    if not isinstance(set_id, str) or not set_id or set_id != set_id.strip():
        raise RawDatasetValidationError("Fit inclusion requires an initial-condition set ID.")
    roster = tuple(str(species).strip() for species in mechanism_species)
    if not roster or any(not species for species in roster):
        raise RawDatasetValidationError("Fit inclusion requires a nonempty valid mechanism species roster.")
    if len(set(roster)) != len(roster):
        raise RawDatasetValidationError("Mechanism species identities must be unique.")
    if not isinstance(initial_values, Mapping):
        raise TypeError("initial_values must be a mapping.")
    values = dict(initial_values)
    if set(values) != set(roster):
        missing = tuple(species for species in roster if species not in values)
        extra = tuple(species for species in values if species not in set(roster))
        raise RawDatasetValidationError(
            f"Mapped initial-condition state is incomplete for the current mechanism (missing={missing!r}, extra={extra!r})."
        )
    if any(type(value) is bool or not math.isfinite(float(value)) or float(value) < 0.0 for value in values.values()):
        raise RawDatasetValidationError("Mapped initial-condition state must be finite and nonnegative.")
    raw_ids = tuple(series.raw_observation_id for series in dataset.observations)
    mapping = tuple(observation_mapping)
    if not mapping:
        raise RawDatasetValidationError(
            "Fit inclusion requires at least one valid mapped observation target."
        )
    seen_ids: set[str] = set()
    seen_species: set[str] = set()
    for item in mapping:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            or not item[0]
            or item[0] != item[0].strip()
            or not item[1]
            or item[1] != item[1].strip()
        ):
            raise RawDatasetValidationError(
                "Fit inclusion observation mapping must contain canonical ID/species pairs."
            )
        raw_id, species = item
        if raw_id not in raw_ids:
            raise RawDatasetValidationError(
                f"Fit inclusion mapping references unknown raw observation ID {raw_id!r}."
            )
        if species not in roster:
            raise RawDatasetValidationError(
                f"Fit inclusion mapping references unknown model species {species!r}."
            )
        if raw_id in seen_ids or species in seen_species:
            raise RawDatasetValidationError(
                "Fit inclusion observation mapping must be a partial bijection."
            )
        seen_ids.add(raw_id)
        seen_species.add(species)
    if tuple(raw_id for raw_id, _species in mapping) != tuple(
        raw_id for raw_id in raw_ids if raw_id in seen_ids
    ):
        raise RawDatasetValidationError(
            "Fit inclusion observation mapping must already be canonical and ordered by raw observation identity."
        )
    series_by_raw_id = {series.raw_observation_id: series for series in dataset.observations}
    targets: list[DatasetMappedTarget] = []
    for raw_observation_id, model_target in mapping:
        series = series_by_raw_id[raw_observation_id]
        if (
            not series.source_rows
            or len(series.source_rows) != len(series.t)
            or len(series.t) != len(series.y)
            or any(type(row) is not int or row <= 0 for row in series.source_rows)
        ):
            raise RawDatasetValidationError(
                f"Observation {series.column_name!r} (raw ID {raw_observation_id!r}) does not retain complete positive physical source-row identity."
            )
        targets.append(
            DatasetMappedTarget(
                dataset_id=dataset.dataset_id,
                raw_observation_id=raw_observation_id,
                model_target=model_target,
                t=series.t,
                y=series.y,
                source_rows=series.source_rows,
            )
        )
    if type(resolved_t_start) is bool:
        raise RawDatasetValidationError("Fit inclusion requires a numeric resolved start time.")
    try:
        start = float(resolved_t_start)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RawDatasetValidationError("Fit inclusion requires a numeric resolved start time.") from exc
    if not math.isfinite(start) or start < 0.0:
        raise RawDatasetValidationError("Fit inclusion requires a finite nonnegative resolved start time.")
    return FitInclusionValidation(
        dataset_id=dataset.dataset_id,
        set_id=set_id,
        resolved_t_start=start,
        targets=tuple(targets),
    )
