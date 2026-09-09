"""Editable import settings and pure physical-column unit planning."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple
import uuid

from kindred.core.datasets.foundation import ImportedSource, validate_imported_source
from kindred.core.datasets.source_rules import PhysicalColumn
from kindred.core.datasets.units import (
    looks_like_unit_row,
    parse_concentration_unit,
    parse_time_unit,
    parse_unit,
)


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
            f"Unsupported dataset extension {shown!r}; choose a .csv or .xlsx file."
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


@dataclass
class ImportSettings:
    """One sheet's editable choices; hidden time-column species choice survives."""

    time_column: PhysicalColumn | None
    species_columns: set[PhysicalColumn]
    time_choice: str | None = None
    time_fallback: str = "s"
    concentration_default: str = "M"
    concentration_fallbacks: dict[PhysicalColumn, str] = field(default_factory=dict)
    override_no_unit_row: bool = False


def classify_unit_row(
    row: Mapping[PhysicalColumn, object],
) -> Mapping[PhysicalColumn, object] | None:
    """Retain the candidate row itself when the whole-row heuristic accepts it."""
    return row if looks_like_unit_row(list(row.values())) else None


def resolve_column_unit(
    column: PhysicalColumn,
    category: str,
    unit_row: Mapping[PhysicalColumn, object] | None,
    settings: ImportSettings,
) -> tuple[str, float]:
    """Admit selected declarations and resolve incoming units in one place."""
    if settings.override_no_unit_row:
        return ("s" if category == "time" else "M"), 1.0
    declared = str(unit_row.get(column, "")).strip() if unit_row is not None else ""
    if declared:
        try:
            actual_category, _ = parse_unit(declared)
        except ValueError as exc:
            raise ValueError(
                f"Classified unit row cell for selected column {column.assigned_name!r} "
                f"contains unsupported unit {declared!r}."
            ) from exc
        if actual_category != category:
            raise ValueError(
                f"Classified unit row cell for selected column {column.assigned_name!r} "
                f"must contain a supported {category} unit, not {declared!r}."
            )
    if category == "time":
        unit = (
            settings.time_choice
            if settings.time_choice is not None
            else declared or settings.time_fallback
        )
        return unit, parse_time_unit(unit)
    unit = (
        declared
        or settings.concentration_fallbacks.get(column)
        or settings.concentration_default
    )
    return unit, parse_concentration_unit(unit)


def sharing_settings(
    settings: ImportSettings,
    unit_row: Mapping[PhysicalColumn, object] | None,
) -> ImportSettings:
    """Detach choices and retain source-effective fallbacks without editing drafts."""
    time_unit = (
        resolve_column_unit(settings.time_column, "time", unit_row, settings)[0]
        if settings.time_column is not None else "s"
    )
    fallbacks = {
        column: resolve_column_unit(column, "concentration", unit_row, settings)[0]
        for column in settings.species_columns
        if column != settings.time_column
    }
    return replace(
        settings,
        species_columns=set(settings.species_columns),
        time_fallback=time_unit,
        concentration_fallbacks=fallbacks,
    )


@dataclass(frozen=True)
class ResolvedSheetPlan:
    """Fully resolved import plan for a single sheet (or CSV file)."""

    sheet_name: Optional[str]
    time_column: PhysicalColumn | None
    species_columns: Tuple[PhysicalColumn, ...]
    skip_unit_row: bool
    time_factor: float
    conc_factors: Mapping[str, float]
    original_time_unit: str
    original_conc_units: Mapping[str, str]

    def __post_init__(self) -> None:
        # The worker retains this plan after GUI admission.  Copy caller
        # mappings once, then expose read-only views so later caller edits
        # cannot change worker normalization.
        object.__setattr__(
            self, "conc_factors", MappingProxyType(dict(self.conc_factors))
        )
        object.__setattr__(
            self,
            "original_conc_units",
            MappingProxyType(dict(self.original_conc_units)),
        )


@dataclass(frozen=True)
class ImportConfig:
    source: ImportedSource
    plans: tuple[ResolvedSheetPlan, ...]
    remaining_file_template: ImportSettings | None = None


def resolve_sheet_plan(
    sheet_name: str | None,
    columns: Sequence[PhysicalColumn],
    unit_row: Mapping[PhysicalColumn, object] | None,
    settings: ImportSettings,
) -> ResolvedSheetPlan:
    time_column = settings.time_column
    species = settings.species_columns - {time_column}
    if not species:
        raise ValueError("Select at least one species column.")
    for column in ({time_column, *species} - {None}):
        if column not in columns:
            raise ValueError(
                f"Column {column.assigned_name!r} not found in sheet {sheet_name!r}."
            )
    ordered_species = tuple(column for column in columns if column in species)
    time_unit, time_factor = (
        resolve_column_unit(time_column, "time", unit_row, settings)
        if time_column is not None else ("s", 1.0)
    )
    concentrations = {
        column.assigned_name: resolve_column_unit(
            column, "concentration", unit_row, settings
        )
        for column in ordered_species
    }
    return ResolvedSheetPlan(
        sheet_name=sheet_name,
        time_column=time_column,
        species_columns=ordered_species,
        skip_unit_row=unit_row is not None and not settings.override_no_unit_row,
        time_factor=time_factor,
        conc_factors={name: value[1] for name, value in concentrations.items()},
        original_time_unit=time_unit,
        original_conc_units={name: value[0] for name, value in concentrations.items()},
    )
