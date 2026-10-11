"""Physical-column-safe parsing for CSV and worksheet rows."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import itertools
import io
import csv
from typing import Callable, Iterable, Mapping, Sequence

from kindred.core.datasets.foundation import ObservationSeries, freeze_json_value
from kindred.core.datasets.observation_payload import (
    DatasetObservationPayload,
    new_raw_observation_id,
)
from kindred.core.datasets.source_rules import (
    PhysicalColumn,
    identify_physical_columns,
)
from kindred.core.species import validate_imported_time_seconds, validate_observation
from kindred.core.quantity_units import Unit, SECOND, conversion_scale, scientific_unit

__all__ = [
    "CsvImportInterrupted",
    "PhysicalRow",
    "TemporalCell",
    "UncachedFormulaCell",
    "read_csv_rows_from_bytes",
    "parse_physical_rows",
]

class CsvImportInterrupted(Exception):
    """Raised internally to unwind when an import is interrupted."""


@dataclass(frozen=True, slots=True)
class UncachedFormulaCell:
    """Selected XLSX formula text for which no cached value exists."""

    formula: str

    def __str__(self) -> str:
        return self.formula


@dataclass(frozen=True, slots=True)
class TemporalCell:
    """XLSX date, time or duration cell retained with its typed meaning."""

    value: dt.datetime | dt.date | dt.time | dt.timedelta

    @property
    def elapsed_seconds(self) -> float | None:
        """Durations and clock times are elapsed seconds; calendar cells are not."""
        value = self.value
        if isinstance(value, dt.timedelta):
            return value.total_seconds()
        if isinstance(value, dt.time):
            return (value.hour * 3600 + value.minute * 60 + value.second
                    + value.microsecond / 1_000_000)
        return None

    def __str__(self) -> str:
        value = self.value
        if isinstance(value, dt.timedelta):
            sign, seconds = ("-", -value) if value < dt.timedelta(0) else ("", value)
            hours, rest = divmod(seconds, dt.timedelta(hours=1))
            return f"{sign}{hours}:{str(rest)[2:]}"
        if isinstance(value, dt.datetime):
            return value.isoformat(sep=" ")
        return value.isoformat()


@dataclass(frozen=True, slots=True)
class PhysicalRow:
    """One parsed logical row with its exact physical source-row start."""

    source_row: int
    values: Mapping[PhysicalColumn, object]


def read_csv_rows_from_bytes(
    source_bytes: bytes,
    *,
    source_label: str,
) -> tuple[tuple[PhysicalColumn, ...], Iterable[PhysicalRow]]:
    """Read one immutable CSV snapshot into canonical physical rows.

    The returned iterator owns its UTF-8 reader and closes it when exhausted
    or explicitly closed.  Header positions remain physical, including blank
    and sparse positions, for the shared ``parse_physical_rows`` boundary.
    """
    owned_bytes = bytes(source_bytes)
    label = str(source_label).strip() or "<csv>"
    handle = io.TextIOWrapper(
        io.BytesIO(owned_bytes),
        encoding="utf-8-sig",
        newline="",
    )
    reader = csv.reader(handle)
    # The source bytes were admitted as UTF-8 text at capture or on Open.
    try:
        header = next(reader, None)
    except csv.Error as exc:
        handle.close()
        raise ValueError(f"CSV file {label!r}: {exc}") from exc
    except BaseException:
        handle.close()
        raise
    if header is None:
        handle.close()
        raise ValueError(f"CSV file {label!r} is empty.")
    columns = tuple(identify_physical_columns(header))
    if not columns:
        handle.close()
        raise ValueError(f"CSV file {label!r} has no named columns.")

    def _iter_rows() -> Iterable[PhysicalRow]:
        try:
            previous_end_row = reader.line_num
            for row in reader:
                row_end_row = reader.line_num
                yield PhysicalRow(
                    source_row=previous_end_row + 1,
                    values={
                        column: (
                            row[column.source_index]
                            if column.source_index < len(row)
                            else ""
                        )
                        for column in columns
                    },
                )
                previous_end_row = row_end_row
        except csv.Error as exc:
            raise ValueError(f"CSV file {label!r}: {exc}") from exc
        finally:
            handle.close()

    return columns, _iter_rows()


def parse_physical_rows(
    rows: Iterable[Mapping[PhysicalColumn, object] | PhysicalRow],
    *,
    columns: Sequence[PhysicalColumn],
    time_column: PhysicalColumn | None,
    species_columns: Sequence[PhysicalColumn],
    time_unit: Unit,
    observation_units: Mapping[str, Unit],
    interruption_checker: Callable[[], bool] | None = None,
    source_label: str | None = None,
    sheet_name: str | None = None,
    source_row_start: int = 2,
    metadata_extras: Mapping[str, object] | None = None,
) -> DatasetObservationPayload:
    """Convert physical source rows without ever keying by authored headers."""
    # The sheet plan comes from resolve_sheet_plan or admitted replay metadata.
    physical_columns = tuple(columns)

    def _raise_if_interrupted() -> None:
        if interruption_checker and interruption_checker():
            raise CsvImportInterrupted()

    def _source_prefix() -> str:
        parts: list[str] = []
        if str(source_label or "").strip():
            parts.append(f"File {str(source_label).strip()!r}")
        if sheet_name is not None:
            parts.append(f"sheet {sheet_name!r}")
        return "" if not parts else ", ".join(parts) + ", "

    row_iter = iter(rows)
    first_item = next(row_iter, None)
    if first_item is None:
        raise ValueError(f"{_source_prefix()}the source contains no data rows.")
    first_source_row = source_row_start

    def _row_values_and_source_row(
        item: Mapping[PhysicalColumn, object] | PhysicalRow,
        logical_index: int,
    ) -> tuple[Mapping[PhysicalColumn, object], int]:
        if isinstance(item, PhysicalRow):
            return item.values, item.source_row
        return item, first_source_row + logical_index

    first, first_row_source = _row_values_and_source_row(first_item, 0)

    resolved_time = time_column
    resolved_species = tuple(species_columns)

    time_scale = conversion_scale(time_unit, SECOND)
    scientific_units = {
        name: scientific_unit(unit.dimensions) for name, unit in observation_units.items()
    }
    observation_scales = {
        name: conversion_scale(unit, scientific_units[name])
        for name, unit in observation_units.items()
    }

    observations_by_column = {
        column: {"t": [], "y": [], "source_rows": []}
        for column in resolved_species
    }
    selected_columns = tuple(
        column for column in physical_columns
        if column == resolved_time or column in resolved_species
    )
    saw_nonempty_row = False
    def _cell(row: Mapping[PhysicalColumn, object], column: PhysicalColumn) -> str:
        value = row.get(column, "")
        if isinstance(value, UncachedFormulaCell):
            raise ValueError(
                f"{_source_prefix()}row {row_index}, column {column.assigned_name!r}: selected "
                f"XLSX formula {value.formula!r} has no cached value."
            )
        if isinstance(value, TemporalCell):
            raise ValueError(
                f"{_source_prefix()}row {row_index}, column "
                f"{column.assigned_name!r}: date or time cell {str(value)!r} "
                "is not a numeric observation. Format measurements as plain numbers."
            )
        return "" if value is None else str(value).strip()

    row_items = itertools.chain((first,), row_iter)
    for logical_index, row_item in enumerate(row_items):
        if logical_index == 0:
            row, row_index = first, first_row_source
        else:
            row, row_index = _row_values_and_source_row(row_item, logical_index)
        _raise_if_interrupted()
        if not any(
            value is not None and str(value).strip()
            for value in (row.get(column, "") for column in selected_columns)
        ):
            continue
        saw_nonempty_row = True
        if resolved_time is not None:
            temporal = row.get(resolved_time)
            if isinstance(temporal, TemporalCell):
                parsed_time = temporal.elapsed_seconds
                if parsed_time is None:
                    raise ValueError(
                        f"{_source_prefix()}row {row_index}, time column "
                        f"{resolved_time.assigned_name!r}: calendar date cell "
                        f"{str(temporal)!r} is not an elapsed time. Format "
                        "times as [h]:mm:ss durations or plain numbers."
                    )
            else:
                raw_time = _cell(row, resolved_time)
                if not raw_time:
                    raise ValueError(
                        f"{_source_prefix()}row {row_index}, time column "
                        f"{resolved_time.assigned_name!r}: missing value."
                    )
                try:
                    parsed_time = time_scale.apply(float(raw_time))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{_source_prefix()}row {row_index}, time column "
                        f"{resolved_time.assigned_name!r}: invalid numeric value "
                        f"{raw_time!r}."
                    ) from exc
            parsed_time = validate_imported_time_seconds(
                parsed_time,
                field=(
                    f"{_source_prefix()}row {row_index}, time column "
                    f"{resolved_time.assigned_name!r}"
                ),
            )

        for column in resolved_species:
            _raise_if_interrupted()
            raw_value = _cell(row, column)
            if not raw_value:
                continue
            try:
                parsed_value = observation_scales[column.assigned_name].apply(float(raw_value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{_source_prefix()}row {row_index}, column "
                    f"{column.assigned_name!r}: invalid numeric observation "
                    f"{raw_value!r}."
                ) from exc
            parsed_value = validate_observation(
                parsed_value,
                field=(
                    f"{_source_prefix()}row {row_index}, column "
                    f"{column.assigned_name!r} observation"
                ),
            )
            spec = observations_by_column[column]
            if resolved_time is not None:
                spec["t"].append(parsed_time)
            spec["y"].append(parsed_value)
            spec["source_rows"].append(row_index)

    if not saw_nonempty_row:
        raise ValueError(
            f"{_source_prefix()}the selected columns contain only blank data rows."
        )

    observations: list[ObservationSeries] = []
    retained_columns: list[PhysicalColumn] = []
    for column in resolved_species:
        _raise_if_interrupted()
        spec = observations_by_column[column]
        if not spec["y"]:
            continue
        retained_columns.append(column)
        observations.append(ObservationSeries(
            raw_observation_id=new_raw_observation_id(),
            column_name=column.assigned_name,
            unit=scientific_units[column.assigned_name],
            t=tuple(spec["t"]),
            y=tuple(spec["y"]),
            source_rows=tuple(spec["source_rows"]),
        ))
    if not observations:
        raise ValueError(f"{_source_prefix()}no numeric observation columns were found.")

    metadata: dict[str, object] = {
        "original_time_unit": time_unit.symbol,
        "original_concentration_units": {
            name: unit.symbol
            for name, unit in observation_units.items()
        },
        "time_column": resolved_time.assigned_name if resolved_time is not None else None,
        "observation_columns": [
            column.assigned_name for column in retained_columns
        ],
        "physical_columns": [
            {
                "source_index": column.source_index,
                "authored_name": column.authored_name,
                "assigned_name": column.assigned_name,
            }
            for column in physical_columns
        ],
        "mapping_source": "explicit",
    }
    if sheet_name is not None:
        metadata["sheet_name"] = str(sheet_name)
    metadata.update(metadata_extras or {})
    return DatasetObservationPayload(
        observations=tuple(observations),
        source_metadata=freeze_json_value(metadata),
    )
