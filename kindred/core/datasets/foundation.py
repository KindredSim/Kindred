"""Immutable raw-dataset values and pure dataset admission validators.

This module deliberately has no dependency on project-document or mapping
policy.  Project admission and dataset mapping import these values downward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from typing import Mapping, TypeAlias
from kindred.core.quantity_units import Unit, scientific_unit

from kindred.core.datasets.source_rules import (
    identify_physical_columns,
    validate_tabular_source_bytes,
)
from kindred.core.species import (
    validate_imported_time_seconds,
    validate_observation,
)

__all__ = [
    "FrozenJsonArray",
    "FrozenJsonObject",
    "FrozenJsonValue",
    "ImportedSource",
    "ObservationSeries",
    "RawDatasetRecord",
    "RawDatasetValidationError",
    "freeze_json_value",
    "thaw_json_value",
    "validate_original_basename",
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
    unit: Unit
    # Empty means no recorded experimental time; rows remain the observation identity.
    t: tuple[float, ...]
    y: tuple[float, ...]
    # Physical rows for imports; logical worksheet rows for created datasets.
    source_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_raw_observation_id(self.raw_observation_id)
        if self.unit != scientific_unit(self.unit.dimensions):
            raise ValueError("Observation values require scientific-coordinate units.")
        if type(self.t) is not tuple or type(self.y) is not tuple:
            raise TypeError("ObservationSeries t/y must be immutable tuples.")
        if not self.y or (self.t and len(self.t) != len(self.y)):
            raise ValueError("ObservationSeries requires values and either aligned recorded times or no times.")
        for index, time in enumerate(self.t):
            validate_imported_time_seconds(time, field=f"observation time[{index}]")
        for index, value in enumerate(self.y):
            validate_observation(value, field=f"observation value[{index}]")


@dataclass(frozen=True, slots=True)
class ImportedSource:
    """One immutable source capture owned by the canonical project."""

    source_id: str
    original_filename: str
    original_filepath: str
    original_format: str
    exact_source_bytes: bytes


def validate_original_basename(value: object, *, label: str = "original filename") -> str:
    """Validate one exact filename with no POSIX or Windows path semantics."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{label} must be a nonblank edge-normalized basename.")
    if (
        value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or ":" in value
        or value.endswith((".", " "))
    ):
        raise ValueError(f"{label} must be an exact basename without path semantics.")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or windows.root
        or any(part in {".", ".."} for part in (*posix.parts, *windows.parts))
    ):
        raise ValueError(f"{label} must be an exact basename without path semantics.")
    windows_reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
    windows_stem = value.rstrip(" .").split(".", 1)[0].casefold()
    if windows_stem in windows_reserved:
        raise ValueError(f"{label} uses a reserved Windows device name.")
    return value


@dataclass(frozen=True, slots=True)
class RawDatasetRecord:
    dataset_id: str
    display_name: str
    source_id: str | None
    source_metadata: FrozenJsonValue
    observations: tuple[ObservationSeries, ...]


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
    if type(record.observations) is not tuple:
        fail(f"Raw dataset {record.dataset_id!r} observations must be an immutable tuple.")
    if not record.observations:
        fail(f"Raw dataset {record.dataset_id!r} requires at least one observation series.")

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
            not series.y
            or (series.t and len(series.t) != len(series.y))
            or len(series.y) != len(series.source_rows)
        ):
            fail(f"Raw dataset {record.dataset_id!r} observation {name!r} must retain equal nonempty time/value tuples aligned with source rows.")
        if any(type(row) is not int or row <= 0 for row in series.source_rows):
            fail(f"Raw dataset {record.dataset_id!r} observation {name!r} source rows must be positive integers.")
        if len(set(series.source_rows)) != len(series.source_rows):
            fail(f"Raw dataset {record.dataset_id!r} observation {name!r} contains duplicate observation row identities.")
        observation_names.append(name)
    if record.source_id is None:
        if record.source_metadata is not None:
            fail("Created datasets cannot retain imported source metadata.")
        timed = bool(record.observations[0].t)
        times_by_row: dict[int, float] = {}
        for series in record.observations:
            if bool(series.t) != timed:
                fail("Created dataset columns must agree on whether time was recorded.")
            if any(left >= right for left, right in zip(series.source_rows, series.source_rows[1:])):
                fail("Created observation row identities must increase in worksheet order.")
            for row, time in zip(series.source_rows, series.t):
                if row in times_by_row and times_by_row[row] != time:
                    fail("Created observations sharing a row must have the same recorded time.")
                times_by_row[row] = time
        return record

    if (
        not isinstance(record.source_id, str)
        or not record.source_id
        or record.source_id != record.source_id.strip()
    ):
        fail(f"Raw dataset {record.dataset_id!r} source ID must be nonblank and edge-normalized.")
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
            or source_index < 0
            or not isinstance(authored_name, str)
            or not authored_name
            or authored_name != authored_name.strip()
            or not isinstance(assigned_name, str)
            or not assigned_name
            or assigned_name != assigned_name.strip()
        ):
            fail(f"Raw dataset {record.dataset_id!r} physical column {index} does not retain exact ordered source identity.")
        physical_rows.append((source_index, authored_name, assigned_name))
    source_indexes = tuple(source_index for source_index, _authored, _assigned in physical_rows)
    if source_indexes != tuple(sorted(source_indexes)) or len(set(source_indexes)) != len(source_indexes):
        fail(f"Raw dataset {record.dataset_id!r} physical source indexes must be unique and strictly increasing.")
    expected_columns = identify_physical_columns(
        [authored_name for _index, authored_name, _assigned in physical_rows]
    )
    expected_rows = tuple(
        (source_index, column.authored_name, column.assigned_name)
        for source_index, column in zip(source_indexes, expected_columns, strict=True)
    )
    if tuple(physical_rows) != expected_rows:
        fail(f"Raw dataset {record.dataset_id!r} physical column names do not match the canonical case-sensitive duplicate allocation.")

    if thawed_metadata.get("observation_columns") != observation_names:
        fail(f"Raw dataset {record.dataset_id!r} observation-column metadata must exactly match its immutable ordered series labels.")
    physical_assigned = {assigned_name for _source_index, _authored_name, assigned_name in physical_rows}
    time_column = thawed_metadata.get("time_column")
    if (
        (time_column is not None and (
            not isinstance(time_column, str)
            or time_column not in physical_assigned
            or time_column in observation_names
        ))
        or any(bool(series.t) != (time_column is not None) for series in record.observations)
        or any(name not in physical_assigned for name in observation_names)
    ):
        fail(f"Raw dataset {record.dataset_id!r} time/observation columns do not reference retained physical source columns.")
    if thawed_metadata.get("mapping_source") != "explicit":
        fail(f"Raw dataset {record.dataset_id!r} mapping source must be exactly 'explicit'.")
    sheet_name = thawed_metadata.get("sheet_name")
    if sheet_name is not None and (not isinstance(sheet_name, str) or not sheet_name):
        fail(f"Raw dataset {record.dataset_id!r} worksheet identity must be nonempty when present.")
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
    try:
        validate_original_basename(source.original_filename)
    except ValueError as exc:
        raise RawDatasetValidationError(str(exc)) from exc
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
