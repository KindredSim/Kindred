"""Physical-column-safe parsing for CSV and worksheet rows."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import logging
import math
from typing import Callable, Iterable, Mapping, Sequence

from kindred.core.datasets.observation_payload import (
    DatasetObservationPayload,
    immutable_dataset_payload,
    new_raw_observation_id,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CsvImportInterrupted",
    "PhysicalColumn",
    "UncachedFormulaCell",
    "identify_physical_columns",
    "parse_physical_rows",
    "validate_tabular_source_bytes",
]

_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_MAX_SOURCE_BYTES = 32 * 1024 * 1024
_MAX_PHYSICAL_ROWS = 100_000
_MAX_PHYSICAL_COLUMNS = 256
_MAX_RETAINED_OBSERVATION_CELLS = 1_000_000


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


def validate_tabular_source_bytes(
    source_bytes: bytes,
    *,
    source_format: str,
    source_label: str,
) -> None:
    """Admit exact nonempty bytes for one current CSV or XLSX source."""

    if type(source_bytes) is not bytes:
        raise TypeError("Dataset source content must be exact bytes.")
    label = str(source_label).strip() or "<dataset>"
    if not source_bytes:
        raise ValueError(f"Dataset source {label!r} is empty.")
    if len(source_bytes) > _MAX_SOURCE_BYTES:
        raise ValueError(
            f"File {label!r} exceeds the 32 MiB supported-size limit."
        )
    signature = source_bytes[:8]
    if source_format == "xlsx":
        if not signature.startswith(_ZIP_SIGNATURES):
            raise ValueError(
                f"File {label!r} has an .xlsx extension but is not an XLSX "
                "workbook."
            )
        return
    if source_format != "csv":
        raise ValueError(
            f"Unsupported exact dataset source format {source_format!r}."
        )
    if signature.startswith(_ZIP_SIGNATURES) or signature.startswith(
        _OLE_SIGNATURE
    ):
        raise ValueError(
            f"File {label!r} has a .csv extension but contains a workbook or "
            "other binary container."
        )
    if b"\x00" in source_bytes:
        raise ValueError(
            f"File {label!r} is binary data, not a UTF-8 CSV source."
        )
    try:
        source_bytes.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"File {label!r} is not a valid UTF-8 CSV source."
        ) from exc


@dataclass(frozen=True, slots=True)
class PhysicalColumn:
    """One source-position identity with its stable assigned observation name."""

    source_index: int
    authored_name: str
    assigned_name: str

    def __post_init__(self) -> None:
        if type(self.source_index) is not int or self.source_index < 0:
            raise ValueError("Physical column source_index must be nonnegative.")
        authored = str(self.authored_name).strip()
        assigned = str(self.assigned_name).strip()
        if not authored or not assigned:
            raise ValueError("Physical column names must be nonblank.")
        object.__setattr__(self, "authored_name", authored)
        object.__setattr__(self, "assigned_name", assigned)


def identify_physical_columns(
    headers: Sequence[object],
) -> tuple[PhysicalColumn, ...]:
    """Assign exact-case collision-safe names before any row mapping exists."""
    authored = tuple(str(value).strip() for value in headers)
    if not authored:
        raise ValueError("The source has no columns.")
    if any(not name for name in authored):
        raise ValueError("Source column names must be nonblank after trimming.")
    reserved = set(authored)
    assigned: set[str] = set()
    occurrences: dict[str, int] = {}
    columns: list[PhysicalColumn] = []
    for source_index, name in enumerate(authored):
        occurrence = occurrences.get(name, 0)
        if occurrence == 0 and name not in assigned:
            candidate = name
        else:
            suffix = occurrence
            while True:
                candidate = f"{name}_copy_{suffix}"
                if candidate not in reserved and candidate not in assigned:
                    break
                suffix += 1
            occurrences[name] = suffix
        occurrences[name] = occurrences.get(name, 0) + 1
        assigned.add(candidate)
        columns.append(
            PhysicalColumn(
                source_index=source_index,
                authored_name=name,
                assigned_name=candidate,
            )
        )
    return tuple(columns)


def parse_physical_rows(
    rows: Iterable[Mapping[PhysicalColumn, object]],
    *,
    columns: Sequence[PhysicalColumn],
    time_column: PhysicalColumn | None = None,
    species_columns: Sequence[PhysicalColumn] | None = None,
    interruption_checker: Callable[[], bool] | None = None,
    source_label: str | None = None,
    sheet_name: str | None = None,
    source_row_start: int = 2,
    time_factor: float = 1.0,
    observation_factors: Mapping[str, float] | None = None,
    metadata_extras: Mapping[str, object] | None = None,
) -> tuple[PhysicalColumn, DatasetObservationPayload]:
    """Convert physical source rows without ever keying by authored headers."""
    physical_columns = tuple(columns)
    if not physical_columns:
        raise ValueError("The source has no columns.")
    if len(physical_columns) > _MAX_PHYSICAL_COLUMNS:
        raise ValueError(
            "The source exceeds the 256 physical-column supported-size limit."
        )
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

    row_iter = iter(rows)
    first = next(row_iter, None)
    if first is None:
        raise ValueError("The source contains no data rows.")

    explicit_mapping = time_column is not None or species_columns is not None
    explicit_species_mapping = species_columns is not None
    if time_column is not None:
        if time_column not in physical_columns:
            raise ValueError("The selected time column is absent from the source.")
        resolved_time = time_column
    else:
        resolved_time = next(
            (
                column
                for candidate in ("time", "time_s", "t", "Time", "T", "x")
                for column in physical_columns
                if column.assigned_name == candidate
            ),
            None,
        )
        if resolved_time is None:
            raise ValueError(
                "Time column not found. Select one of time, time_s, t, Time, T, or x."
            )

    if species_columns is None:
        resolved_species = tuple(
            column for column in physical_columns if column != resolved_time
        )
    else:
        resolved_species = tuple(species_columns)
        if not resolved_species:
            raise ValueError("Select at least one observation column.")
        if len(set(resolved_species)) != len(resolved_species):
            raise ValueError("Selected observation columns must be unique.")
        if any(column not in physical_columns for column in resolved_species):
            raise ValueError(
                "A selected observation column is absent from the source."
            )
        if resolved_time in resolved_species:
            raise ValueError(
                "The time column cannot also be an observation column."
            )

    first_source_row = int(source_row_start)
    if first_source_row <= 0:
        raise ValueError("source_row_start must be a positive physical row number.")
    resolved_time_factor = float(time_factor)
    if (
        isinstance(time_factor, bool)
        or not math.isfinite(resolved_time_factor)
        or resolved_time_factor <= 0.0
    ):
        raise ValueError("time_factor must be finite and positive.")
    raw_observation_factors = dict(observation_factors or {})
    expected_factor_names = {
        column.assigned_name for column in resolved_species
    }
    if raw_observation_factors and set(raw_observation_factors) != expected_factor_names:
        raise ValueError(
            "Observation factors must exactly match the selected observation "
            "columns."
        )
    resolved_observation_factors: dict[str, float] = {}
    for name in expected_factor_names:
        raw_factor = raw_observation_factors.get(name, 1.0)
        factor = float(raw_factor)
        if (
            isinstance(raw_factor, bool)
            or not math.isfinite(factor)
            or factor <= 0.0
        ):
            raise ValueError(
                f"Observation factor for {name!r} must be finite and positive."
            )
        resolved_observation_factors[name] = factor

    observations_by_column = {
        column: {"t": [], "y": [], "source_rows": []}
        for column in resolved_species
    }
    saw_nonempty_row = False
    retained_cells = 0

    def _cell(row: Mapping[PhysicalColumn, object], column: PhysicalColumn) -> str:
        value = row.get(column, "")
        if isinstance(value, UncachedFormulaCell):
            raise ValueError(
                f"{_source_prefix()}column {column.assigned_name!r}: selected "
                f"XLSX formula {value.formula!r} has no cached value."
            )
        return "" if value is None else str(value).strip()

    def _source_prefix() -> str:
        parts: list[str] = []
        if str(source_label or "").strip():
            parts.append(f"file {str(source_label).strip()!r}")
        if str(sheet_name or "").strip():
            parts.append(f"sheet {str(sheet_name).strip()!r}")
        return "" if not parts else ", ".join(parts) + ", "

    for row_index, row in enumerate(
        itertools.chain((first,), row_iter),
        start=first_source_row,
    ):
        if row_index - first_source_row + 1 > _MAX_PHYSICAL_ROWS:
            raise ValueError(
                "The source exceeds the 100000 physical-row supported-size "
                "limit."
            )
        _raise_if_interrupted()
        raw_values = tuple(row.get(column, "") for column in physical_columns)
        if not any(
            value is not None and str(value).strip()
            for value in raw_values
        ):
            continue
        saw_nonempty_row = True
        raw_time = _cell(row, resolved_time)
        if not raw_time:
            raise ValueError(
                f"{_source_prefix()}row {row_index}, time column "
                f"{resolved_time.assigned_name!r}: missing value."
            )
        try:
            parsed_time = float(raw_time) * resolved_time_factor
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{_source_prefix()}row {row_index}, time column "
                f"{resolved_time.assigned_name!r}: invalid numeric value "
                f"{raw_time!r}."
            ) from exc
        if not math.isfinite(parsed_time):
            raise ValueError(
                f"{_source_prefix()}row {row_index}, time column "
                f"{resolved_time.assigned_name!r}: value must be finite."
            )

        for column in resolved_species:
            _raise_if_interrupted()
            raw_value = _cell(row, column)
            if not raw_value:
                continue
            try:
                parsed_value = (
                    float(raw_value)
                    * resolved_observation_factors[column.assigned_name]
                )
            except (TypeError, ValueError) as exc:
                if explicit_species_mapping:
                    raise ValueError(
                        f"{_source_prefix()}row {row_index}, column "
                        f"{column.assigned_name!r}: invalid numeric observation "
                        f"{raw_value!r}."
                    ) from exc
                logger.debug(
                    "Skipping nonnumeric observation in physical column %d.",
                    column.source_index,
                )
                continue
            if not math.isfinite(parsed_value):
                if explicit_species_mapping:
                    raise ValueError(
                        f"{_source_prefix()}row {row_index}, column "
                        f"{column.assigned_name!r}: observation must be finite."
                    )
                continue
            retained_cells += 1
            if retained_cells > _MAX_RETAINED_OBSERVATION_CELLS:
                raise ValueError(
                    "The import exceeds the 1000000 retained-observation-cell "
                    "limit."
                )
            spec = observations_by_column[column]
            spec["t"].append(parsed_time)
            spec["y"].append(parsed_value)
            spec["source_rows"].append(row_index)

    if not saw_nonempty_row:
        raise ValueError("The source contains only blank data rows.")

    observations: dict[str, dict[str, object]] = {}
    retained_columns: list[PhysicalColumn] = []
    for column in resolved_species:
        _raise_if_interrupted()
        spec = observations_by_column[column]
        if not spec["t"]:
            continue
        retained_columns.append(column)
        observations[column.assigned_name] = {
            "raw_observation_id": new_raw_observation_id(),
            "t": spec["t"],
            "y": spec["y"],
            "source_rows": spec["source_rows"],
        }
    if not observations:
        raise ValueError("No numeric observation columns were found.")

    metadata: dict[str, object] = {
        "time_column": resolved_time.assigned_name,
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
        "mapping_source": "explicit" if explicit_mapping else "auto",
    }
    if sheet_name is not None and str(sheet_name).strip():
        metadata["sheet_name"] = str(sheet_name)
    extras = dict(metadata_extras or {})
    collisions = set(extras) & set(metadata)
    if collisions:
        raise ValueError(
            "Import metadata extras cannot replace canonical fields: "
            + ", ".join(sorted(collisions))
        )
    metadata.update(extras)
    return (
        resolved_time,
        immutable_dataset_payload(observations, metadata=metadata),
    )
