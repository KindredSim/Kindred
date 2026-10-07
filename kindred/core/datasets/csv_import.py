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

    def __post_init__(self) -> None:
        if (
            not isinstance(self.formula, str)
            or not self.formula
            or not self.formula.startswith("=")
        ):
            raise ValueError("Uncached XLSX formulas require authored formula text.")

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

    def __post_init__(self) -> None:
        if type(self.source_row) is not int or self.source_row <= 0:
            raise ValueError("Physical row source_row must be a positive integer.")
        if not isinstance(self.values, Mapping):
            raise TypeError("Physical row values must be a column mapping.")


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
    try:
        header = next(reader, None)
    except (UnicodeDecodeError, csv.Error) as exc:
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
        except (UnicodeDecodeError, csv.Error) as exc:
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
    physical_columns = tuple(columns)
    if not physical_columns:
        raise ValueError("The source has no columns.")
    if len({column.source_index for column in physical_columns}) != len(
        physical_columns
    ):
        raise ValueError("Physical column indexes must be unique.")
    if len({column.assigned_name for column in physical_columns}) != len(
        physical_columns
    ):
        raise ValueError("Assigned source column names must be unique.")

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
    first_source_row = int(source_row_start)
    if first_source_row <= 0:
        raise ValueError("source_row_start must be a positive physical row number.")

    def _row_values_and_source_row(
        item: Mapping[PhysicalColumn, object] | PhysicalRow,
        logical_index: int,
    ) -> tuple[Mapping[PhysicalColumn, object], int]:
        if isinstance(item, PhysicalRow):
            return item.values, item.source_row
        return item, first_source_row + logical_index

    first, first_row_source = _row_values_and_source_row(first_item, 0)

    if time_column is not None and not isinstance(time_column, PhysicalColumn):
        raise TypeError("An exact admitted PhysicalColumn time column is required.")
    if time_column is not None and time_column not in physical_columns:
        raise ValueError("The admitted time column is absent from the source.")
    resolved_time = time_column
    resolved_species = tuple(species_columns)
    if not resolved_species:
        raise ValueError("At least one exact admitted observation column is required.")
    if len(set(resolved_species)) != len(resolved_species):
        raise ValueError("Admitted observation columns must be unique.")
    if any(column not in physical_columns for column in resolved_species):
        raise ValueError("An admitted observation column is absent from the source.")
    if resolved_time in resolved_species:
        raise ValueError("The admitted time column cannot also be an observation column.")

    time_scale = conversion_scale(time_unit, SECOND)
    if set(observation_units) != {column.assigned_name for column in resolved_species}:
        raise ValueError("Observation units must exactly match the selected columns.")
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
        if row_index <= 0:
            raise ValueError("Physical source rows must be positive.")
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
                        f"{raw_time!r}: {exc}"
                    ) from exc
            try:
                parsed_time = validate_imported_time_seconds(
                    parsed_time,
                    field=(
                        f"{_source_prefix()}row {row_index}, time column "
                        f"{resolved_time.assigned_name!r}"
                    ),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(str(exc)) from exc

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
                    f"{raw_value!r}: {exc}"
                ) from exc
            try:
                parsed_value = validate_observation(
                    parsed_value,
                    field=(
                        f"{_source_prefix()}row {row_index}, column "
                        f"{column.assigned_name!r} observation"
                    ),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(str(exc)) from exc
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
    extras = dict(metadata_extras or {})
    collisions = set(extras) & set(metadata)
    if collisions:
        raise ValueError(
            "Import metadata extras cannot replace canonical fields: "
            + ", ".join(sorted(collisions))
        )
    metadata.update(extras)
    return DatasetObservationPayload(
        observations=tuple(observations),
        source_metadata=freeze_json_value(metadata),
    )
