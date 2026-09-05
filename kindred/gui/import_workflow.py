"""Composed import configuration types and pure helpers for dataset import.

Defines the structured types that replace the flat ImportConfig dataclass:
UnitDetection, UserImportIntent, ResolvedSheetPlan, and the top-level
ImportConfig. Also provides ``detect_units_from_physical_row``,
``rebuild_intent_for_target``, and ``resolve_import_plans`` for building and
adapting these objects from raw preview data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import uuid

from kindred.core.datasets.foundation import ImportedSource, validate_imported_source
from kindred.core.datasets.source_rules import PhysicalColumn
from kindred.core.datasets.units import (
    looks_like_unit_row,
    parse_concentration_unit,
    parse_time_unit,
    parse_unit,
)

__all__ = [
    "ImportConfig",
    "ResolvedSheetPlan",
    "SheetImportIntent",
    "UnitDetection",
    "UserImportIntent",
    "detect_units_from_physical_row",
    "capture_import_source",
    "rebuild_intent_for_target",
    "resolve_import_plans",
]

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
        exact_source_bytes=source_bytes,
    )
    return validate_imported_source(source)


@dataclass(frozen=True)
class UnitDetection:
    """Result of heuristic unit-row detection on a single sheet/file."""

    has_unit_row: bool
    detected_time_unit: Optional[str]
    detected_conc_unit_by_column: Dict[PhysicalColumn, Optional[str]]
    selected_unit_text_by_column: Dict[PhysicalColumn, str] = field(
        default_factory=dict
    )
    selected_unit_category_by_column: Dict[PhysicalColumn, str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        detected = dict(self.detected_conc_unit_by_column)
        selected_text = dict(self.selected_unit_text_by_column)
        selected_category = dict(self.selected_unit_category_by_column)
        if type(self.has_unit_row) is not bool:
            raise TypeError("Unit detection has_unit_row must be a bool.")
        if self.detected_time_unit is not None and not isinstance(
            self.detected_time_unit,
            str,
        ):
            raise TypeError("Detected time unit must be a string or None.")
        if any(
            not isinstance(column, PhysicalColumn)
            or (unit is not None and not isinstance(unit, str))
            for column, unit in detected.items()
        ):
            raise TypeError(
                "Detected concentration units require physical-column keys "
                "and string-or-None values."
            )
        if any(
            not isinstance(column, PhysicalColumn)
            or not isinstance(text, str)
            or text != text.strip()
            for column, text in selected_text.items()
        ):
            raise TypeError(
                "Selected unit cells require physical-column keys and "
                "edge-normalized text values."
            )
        if (
            any(
                not isinstance(column, PhysicalColumn)
                or category not in {"time", "concentration"}
                for column, category in selected_category.items()
            )
            or not set(selected_category).issubset(selected_text)
        ):
            raise TypeError(
                "Selected unit categories require selected physical-column "
                "unit cells and exact supported categories."
            )
        object.__setattr__(self, "detected_conc_unit_by_column", detected)
        object.__setattr__(self, "selected_unit_text_by_column", selected_text)
        object.__setattr__(
            self,
            "selected_unit_category_by_column",
            selected_category,
        )

    @staticmethod
    def empty() -> UnitDetection:
        return UnitDetection(
            has_unit_row=False,
            detected_time_unit=None,
            detected_conc_unit_by_column={},
        )


@dataclass(frozen=True)
class UserImportIntent:
    """Captures the user's choices from the import configuration dialog."""

    sheet_names: Tuple[str, ...]
    apply_to_remaining: bool

    def __post_init__(self) -> None:
        sheet_names = tuple(self.sheet_names)
        if (
            any(
                not isinstance(name, str)
                or not name
                for name in sheet_names
            )
            or len(sheet_names) != len(set(sheet_names))
        ):
            raise ValueError(
                "Selected worksheet names must be unique nonempty strings."
            )
        if type(self.apply_to_remaining) is not bool:
            raise TypeError("apply_to_remaining must be a bool.")
        object.__setattr__(self, "sheet_names", sheet_names)


@dataclass(frozen=True)
class SheetImportIntent:
    """Captures the user's choices for one sheet (or one CSV file)."""

    time_column: PhysicalColumn
    species_columns: Tuple[PhysicalColumn, ...]
    time_unit: str
    concentration_units: Dict[PhysicalColumn, str]
    override_no_unit_row: bool

    def __post_init__(self) -> None:
        species_columns = tuple(self.species_columns)
        concentration_units = dict(self.concentration_units)
        if not isinstance(self.time_column, PhysicalColumn):
            raise TypeError("Import time_column must be a PhysicalColumn.")
        if (
            not species_columns
            or any(
                not isinstance(column, PhysicalColumn)
                for column in species_columns
            )
            or len(species_columns) != len(set(species_columns))
            or self.time_column in species_columns
        ):
            raise ValueError(
                "Import species_columns must be nonempty, unique physical "
                "columns distinct from time_column."
            )
        if set(concentration_units) != set(species_columns):
            raise ValueError("concentration_units keys must match species_columns")
        if (
            not isinstance(self.time_unit, str)
            or not self.time_unit.strip()
            or any(
                not isinstance(value, str) or not value.strip()
                for value in concentration_units.values()
            )
        ):
            raise ValueError("Import units must be nonblank strings.")
        if type(self.override_no_unit_row) is not bool:
            raise TypeError("override_no_unit_row must be a bool.")
        object.__setattr__(self, "species_columns", species_columns)
        object.__setattr__(self, "concentration_units", concentration_units)


@dataclass(frozen=True)
class ResolvedSheetPlan:
    """Fully resolved import plan for a single sheet (or CSV file)."""

    sheet_name: Optional[str]
    time_column: PhysicalColumn
    species_columns: Tuple[PhysicalColumn, ...]
    skip_unit_row: bool
    time_factor: float
    conc_factors: Mapping[str, float]
    original_time_unit: str
    original_conc_units: Mapping[str, str]

    def __post_init__(self) -> None:
        species_columns = tuple(self.species_columns)
        conc_factors = dict(self.conc_factors)
        original_conc_units = dict(self.original_conc_units)
        if self.sheet_name is not None and (
            not isinstance(self.sheet_name, str)
            or not self.sheet_name
        ):
            raise ValueError(
                "Resolved worksheet name must be None or a nonempty string."
            )
        if not isinstance(self.time_column, PhysicalColumn):
            raise TypeError(
                "Resolved import time_column must be a PhysicalColumn."
            )
        if (
            not species_columns
            or any(
                not isinstance(column, PhysicalColumn)
                for column in species_columns
            )
            or len(species_columns) != len(set(species_columns))
            or self.time_column in species_columns
        ):
            raise ValueError(
                "Resolved import species_columns must be nonempty, unique, "
                "and distinct from time_column."
            )
        assigned_names = {
            column.assigned_name for column in species_columns
        }
        if (
            set(conc_factors) != assigned_names
            or set(original_conc_units) != assigned_names
        ):
            raise ValueError(
                "Resolved concentration values must key every assigned "
                "observation name exactly once."
            )
        if (
            type(self.skip_unit_row) is not bool
            or isinstance(self.time_factor, bool)
            or not math.isfinite(float(self.time_factor))
            or float(self.time_factor) <= 0.0
            or any(
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in conc_factors.values()
            )
            or not isinstance(self.original_time_unit, str)
            or not self.original_time_unit.strip()
            or any(
                not isinstance(value, str) or not value.strip()
                for value in original_conc_units.values()
            )
        ):
            raise ValueError(
                "Resolved import units and scale factors must be valid and "
                "strictly positive."
            )
        object.__setattr__(self, "species_columns", species_columns)
        object.__setattr__(self, "time_factor", float(self.time_factor))
        # The worker retains this plan after GUI admission.  Copy caller
        # mappings once, then expose read-only views so later caller edits
        # cannot change worker normalization.
        object.__setattr__(self, "conc_factors", MappingProxyType(conc_factors))
        object.__setattr__(
            self,
            "original_conc_units",
            MappingProxyType(original_conc_units),
        )


@dataclass(frozen=True)
class ImportConfig:
    """Top-level import configuration grouping source intent and plans."""

    source: ImportedSource
    file_intent: UserImportIntent
    plans: Tuple[ResolvedSheetPlan, ...]
    remaining_file_template: Optional[SheetImportIntent] = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, ImportedSource):
            raise TypeError(
                "Import configuration requires one immutable source snapshot."
            )
        if not isinstance(self.file_intent, UserImportIntent):
            raise TypeError("Import configuration requires UserImportIntent.")
        admitted_type = self.source.original_format
        plans = tuple(self.plans)
        if not plans:
            raise ValueError("Import configuration requires at least one plan.")
        if any(not isinstance(plan, ResolvedSheetPlan) for plan in plans):
            raise TypeError("Import configuration plans must be resolved plans.")
        if admitted_type == "csv" and (
            len(plans) != 1 or plans[0].sheet_name is not None
        ):
            raise ValueError("A CSV import requires exactly one non-sheet plan.")
        if admitted_type == "xlsx" and any(
            plan.sheet_name is None for plan in plans
        ):
            raise ValueError("Every XLSX import plan requires a sheet name.")
        plan_keys = tuple(plan.sheet_name for plan in plans)
        if admitted_type == "xlsx" and tuple(
            self.file_intent.sheet_names
        ) != tuple(key for key in plan_keys if key is not None):
            raise ValueError(
                "Selected worksheet identity must exactly match resolved plans."
            )
        if self.remaining_file_template is not None and not isinstance(
            self.remaining_file_template,
            SheetImportIntent,
        ):
            raise TypeError(
                "remaining_file_template must be SheetImportIntent or None."
            )
        object.__setattr__(self, "plans", plans)

def detect_units_from_physical_row(
    row_mapping: Mapping[PhysicalColumn, object],
    relevant_columns: Optional[Sequence[PhysicalColumn]] = None,
) -> UnitDetection:
    """Build unit detection without reducing physical columns to text keys.

    Parameters
    ----------
    row_mapping:
        Mapping of column name to the cell value in the candidate unit row.
    relevant_columns:
        If provided, unit extraction is restricted to these columns while
        ``has_unit_row`` still inspects the full row so unit-like text in
        unselected columns can still identify a physical unit row.
    """
    full_values = [str(value).strip() for value in row_mapping.values()]
    if not looks_like_unit_row(full_values):
        return UnitDetection.empty()

    if relevant_columns is not None:
        scoped_columns = [
            column for column in relevant_columns if column in row_mapping
        ]
    else:
        scoped_columns = list(row_mapping.keys())

    detected_time_unit: Optional[str] = None
    conc_unit_by_column: Dict[PhysicalColumn, Optional[str]] = {}
    selected_text_by_column: Dict[PhysicalColumn, str] = {}
    selected_category_by_column: Dict[PhysicalColumn, str] = {}

    for col in scoped_columns:
        val = str(row_mapping[col]).strip()
        selected_text_by_column[col] = val
        if not val:
            continue
        try:
            category, _factor = parse_unit(val)
        except ValueError:
            continue
        selected_category_by_column[col] = category
        if category == "time" and detected_time_unit is None:
            detected_time_unit = val
        elif category == "concentration":
            conc_unit_by_column[col] = val

    return UnitDetection(
        has_unit_row=True,
        detected_time_unit=detected_time_unit,
        detected_conc_unit_by_column=conc_unit_by_column,
        selected_unit_text_by_column=selected_text_by_column,
        selected_unit_category_by_column=selected_category_by_column,
    )


def rebuild_intent_for_target(
    template_intent: SheetImportIntent,
    target_detection: UnitDetection,
) -> SheetImportIntent:
    """Build a ``SheetImportIntent`` for a target sheet or file."""
    conc_units: Dict[PhysicalColumn, str] = {}
    for col in template_intent.species_columns:
        if not template_intent.override_no_unit_row:
            target_detected = target_detection.detected_conc_unit_by_column.get(col)
        else:
            target_detected = None
        if target_detected is not None:
            conc_units[col] = target_detected
        else:
            conc_units[col] = template_intent.concentration_units[col]
    if not template_intent.override_no_unit_row and target_detection.detected_time_unit:
        time_unit = str(target_detection.detected_time_unit)
    else:
        time_unit = template_intent.time_unit
    return SheetImportIntent(
        time_column=template_intent.time_column,
        species_columns=template_intent.species_columns,
        time_unit=time_unit,
        concentration_units=conc_units,
        override_no_unit_row=template_intent.override_no_unit_row,
    )


def resolve_import_plans(
    source: ImportedSource,
    per_sheet_intents: Dict[Optional[str], SheetImportIntent],
    per_sheet_detections: Dict[Optional[str], UnitDetection],
    per_sheet_columns: Dict[Optional[str], Sequence[PhysicalColumn]],
) -> List[ResolvedSheetPlan]:
    """Resolve the user's intent into concrete per-sheet import plans.

    Raises ``ValueError`` when the intent is inconsistent with the detected
    file structure (missing sheets, missing columns, or ambiguous
    concentration factors).
    """
    if not isinstance(source, ImportedSource):
        raise TypeError("Import planning requires one immutable source snapshot.")
    admitted_type = source.original_format
    target_sheets = list(per_sheet_intents.keys())
    if not target_sheets:
        raise ValueError("Select at least one CSV source or XLSX sheet.")
    if admitted_type == "csv" and target_sheets != [None]:
        raise ValueError("A CSV import requires exactly one non-sheet intent.")
    if admitted_type == "xlsx" and any(
        sheet is None for sheet in target_sheets
    ):
        raise ValueError("Every XLSX import intent requires a sheet name.")

    # (a) Every target sheet must have a detection and columns entry.
    for sheet in target_sheets:
        if sheet not in per_sheet_intents:
            raise ValueError(
                f"No import intent available for sheet {sheet!r}."
            )
        if sheet not in per_sheet_detections:
            raise ValueError(
                f"No unit detection available for sheet {sheet!r}."
            )
        if sheet not in per_sheet_columns:
            raise ValueError(
                f"No column information available for sheet {sheet!r}."
            )

    # (b) Each sheet's configured time column must exist.
    for sheet in target_sheets:
        columns = per_sheet_columns[sheet]
        sheet_intent = per_sheet_intents[sheet]
        if sheet_intent.time_column not in columns:
            raise ValueError(
                f"Time column {sheet_intent.time_column.assigned_name!r} "
                f"not found in sheet {sheet!r}."
            )

    # (c) Each sheet's configured species columns must exist.
    for sheet in target_sheets:
        columns = per_sheet_columns[sheet]
        sheet_intent = per_sheet_intents[sheet]
        for species_col in sheet_intent.species_columns:
            if species_col not in columns:
                raise ValueError(
                    f"Observation column {species_col.assigned_name!r} "
                    f"not found in sheet {sheet!r}."
                )

    # Build plans.
    plans: List[ResolvedSheetPlan] = []
    for sheet in target_sheets:
        detection = per_sheet_detections[sheet]
        sheet_intent = per_sheet_intents[sheet]
        species_columns = sheet_intent.species_columns

        skip_unit_row = bool(detection.has_unit_row and not sheet_intent.override_no_unit_row)

        if skip_unit_row:
            selected_text = detection.selected_unit_text_by_column
            selected_categories = detection.selected_unit_category_by_column
            selected_roles = (
                (sheet_intent.time_column, "time"),
                *((column, "concentration") for column in species_columns),
            )
            for column, expected_category in selected_roles:
                cell_text = selected_text.get(column, "")
                if not cell_text:
                    continue
                actual_category = selected_categories.get(column)
                if actual_category is None:
                    raise ValueError(
                        f"Classified unit row cell for selected column "
                        f"{column.assigned_name!r} contains unsupported unit "
                        f"{cell_text!r}."
                    )
                if actual_category != expected_category:
                    raise ValueError(
                        f"Classified unit row cell for selected column "
                        f"{column.assigned_name!r} must contain a supported "
                        f"{expected_category} unit, not {cell_text!r}."
                    )

        if sheet_intent.override_no_unit_row:
            time_unit = "s"
            time_factor = 1.0
            conc_factors = {
                column.assigned_name: 1.0 for column in species_columns
            }
            original_conc_units = {
                column.assigned_name: "M" for column in species_columns
            }
        else:
            time_unit = sheet_intent.time_unit
            time_factor = parse_time_unit(time_unit)
            conc_factors = {}
            original_conc_units = {}
            for col in species_columns:
                col_unit = sheet_intent.concentration_units[col]
                conc_factors[col.assigned_name] = parse_concentration_unit(
                    col_unit
                )
                original_conc_units[col.assigned_name] = col_unit

        plans.append(
            ResolvedSheetPlan(
                sheet_name=sheet,
                time_column=sheet_intent.time_column,
                species_columns=species_columns,
                skip_unit_row=skip_unit_row,
                time_factor=time_factor,
                conc_factors=conc_factors,
                original_time_unit=time_unit,
                original_conc_units=original_conc_units,
            )
        )

    return plans
