"""Editable import settings and pure physical-column unit planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple
import uuid

from kindred.core.datasets.foundation import ImportedSource, validate_imported_source
from kindred.core.datasets.source_rules import PhysicalColumn
from kindred.core.datasets.units import (
    looks_like_unit_row,
    parse_measurement_unit,
    parse_time_unit,
)
from kindred.core.quantity_units import Unit, SECOND


def capture_import_source(
    filepath: str | os.PathLike[str],
) -> ImportedSource:
    """Read one source exactly once before any semantic preview or planning."""

    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix == ".xls":
        raise ValueError(
            "Legacy .xls format is not supported. Save the source as .xlsx."
        )
    if suffix not in {".csv", ".xlsx"}:
        shown = suffix or "(none)"
        raise ValueError(
            f"Unsupported dataset extension {shown!r}. Choose a .csv or .xlsx file."
        )
    if not path.is_file():
        raise ValueError(f"Dataset source does not exist: {path}.")
    try:
        with path.open("rb") as source:
            source_bytes = source.read()
    except OSError as exc:
        raise ValueError(f"Cannot read dataset source {path!s}: {exc}") from exc
    source = ImportedSource(
        source_id=f"source-{uuid.uuid4().hex}",
        original_format=suffix[1:],
        original_filename=path.name,
        original_filepath=str(path.absolute()),
        exact_source_bytes=source_bytes,
    )
    return validate_imported_source(source)


class UnitSource(Enum):
    FILE = "file"
    DEFAULT = "default"


@dataclass
class ImportSettings:
    """One sheet's editable choices; hidden time-column species choice survives."""

    time_column: PhysicalColumn | None
    species_columns: set[PhysicalColumn]
    time_choice: str | UnitSource = UnitSource.FILE
    default_unit: str = "M"
    # Missing choices follow the file when present, otherwise the sheet default.
    column_units: dict[PhysicalColumn, str | UnitSource] = field(default_factory=dict)
    # None retains detection per sheet; a boolean is an explicit row decision.
    first_row_contains_units: bool | None = None


def classify_unit_row(
    row: Mapping[PhysicalColumn, object] | None,
    *,
    contains_units: bool | None = None,
) -> Mapping[PhysicalColumn, object] | None:
    """Resolve an explicit row choice or detect textual units for this sheet."""
    if row is None:
        return None
    selected = looks_like_unit_row(list(row.values())) if contains_units is None else contains_units
    return row if selected else None


def resolve_column_unit(
    column: PhysicalColumn,
    unit_row: Mapping[PhysicalColumn, object] | None,
    settings: ImportSettings,
    *,
    time: bool = False,
) -> Unit:
    """Admit selected declarations and resolve incoming units in one place."""
    raw = unit_row.get(column) if unit_row is not None else None
    declared = "" if raw is None else str(raw).strip()
    choice = settings.time_choice if time else settings.column_units.get(column, UnitSource.FILE)
    default = "s" if time else settings.default_unit
    text = (declared or default) if choice is UnitSource.FILE else default if choice is UnitSource.DEFAULT else choice
    try:
        return parse_time_unit(text) if time else parse_measurement_unit(text)
    except ValueError as exc:
        if time and ":" in text:
            # A units-row clock format such as h:mm:ss is not a unit.
            raise ValueError(
                f"Column {column.assigned_name!r}: {text!r} is a time format, not a "
                "unit. Excel duration and clock cells need no unit (set Time unit "
                "to Use default), and CSV times must be plain numbers."
            ) from exc
        raise ValueError(f"Column {column.assigned_name!r}: {exc}") from exc


@dataclass(frozen=True)
class ResolvedSheetPlan:
    """Fully resolved import plan for a single sheet (or CSV file)."""

    sheet_name: Optional[str]
    time_column: PhysicalColumn | None
    species_columns: Tuple[PhysicalColumn, ...]
    skip_unit_row: bool
    time_unit: Unit
    observation_units: Mapping[str, Unit]

    def __post_init__(self) -> None:
        # The worker retains this plan after GUI admission.  Copy caller
        # mappings once, then expose read-only views so later caller edits
        # cannot change worker normalization.
        object.__setattr__(
            self,
            "observation_units",
            MappingProxyType(dict(self.observation_units)),
        )


@dataclass(frozen=True)
class ImportConfig:
    source: ImportedSource
    plans: tuple[ResolvedSheetPlan, ...]
    remaining_file_template: ImportSettings | None = None


def resolve_sheet_plan(
    sheet_name: str | None,
    columns: Sequence[PhysicalColumn],
    first_row: Mapping[PhysicalColumn, object] | None,
    settings: ImportSettings,
) -> ResolvedSheetPlan:
    try:
        time_column = settings.time_column
        unit_row = classify_unit_row(first_row, contains_units=settings.first_row_contains_units)
        species = settings.species_columns - {time_column}
        if not species:
            raise ValueError("Select at least one species column.")
        for column in ({time_column, *species} - {None}):
            if column not in columns:
                raise ValueError(f"Column {column.assigned_name!r} not found.")
        ordered_species = tuple(column for column in columns if column in species)
        time_unit = (
            resolve_column_unit(time_column, unit_row, settings, time=True)
            if time_column is not None else SECOND
        )
        observations = {
            column.assigned_name: resolve_column_unit(
                column, unit_row, settings
            )
            for column in ordered_species
        }
        return ResolvedSheetPlan(
            sheet_name=sheet_name,
            time_column=time_column,
            species_columns=ordered_species,
            skip_unit_row=unit_row is not None,
            time_unit=time_unit,
            observation_units=observations,
        )
    except ValueError as exc:
        if sheet_name is None:
            raise
        raise ValueError(f"Sheet {sheet_name!r}: {exc}") from exc
