"""Composed import configuration types and pure helpers for dataset import.

Defines the structured types that replace the flat ImportConfig dataclass:
UnitDetection, UserImportIntent, ResolvedSheetPlan, and the top-level
ImportConfig. Also provides ``detect_units_from_physical_row``,
``rebuild_intent_for_target``, and ``resolve_import_plans`` for building and
adapting these objects from raw preview data.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from kindred.core.datasets.csv_import import PhysicalColumn
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
    "classify_import_source",
    "rebuild_intent_for_target",
    "resolve_import_plans",
    "validate_import_source_bytes",
]

_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def validate_import_source_bytes(
    source_bytes: bytes,
    *,
    file_type: str,
    source_label: str,
) -> None:
    """Reject extension-disguised or unsupported binary source content."""
    owned = bytes(source_bytes)
    label = str(source_label).strip() or "<dataset>"
    if not owned:
        raise ValueError(f"Dataset source {label!r} is empty.")
    signature = owned[:8]
    if file_type == "excel":
        if not signature.startswith(_ZIP_SIGNATURES):
            raise ValueError(
                f"File {label!r} has an .xlsx extension but is not an XLSX "
                "workbook."
            )
        return
    if file_type != "csv":
        raise ValueError(f"Unsupported import source type {file_type!r}.")
    if signature.startswith(_ZIP_SIGNATURES) or signature.startswith(
        _OLE_SIGNATURE
    ):
        raise ValueError(
            f"File {label!r} has a .csv extension but contains a workbook or "
            "other binary container."
        )
    if b"\x00" in signature:
        raise ValueError(
            f"File {label!r} is binary data, not a UTF-8 CSV source."
        )


def classify_import_source(filepath: str | os.PathLike[str]) -> str:
    """Admit only extension/content-consistent CSV or XLSX sources."""
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
            source_prefix = source.read(8)
    except OSError as exc:
        raise ValueError(f"Cannot read dataset source {path!s}: {exc}") from exc
    file_type = "excel" if suffix == ".xlsx" else "csv"
    validate_import_source_bytes(
        source_prefix,
        file_type=file_type,
        source_label=path.name,
    )
    return file_type


@dataclass(frozen=True)
class UnitDetection:
    """Result of heuristic unit-row detection on a single sheet/file."""

    has_unit_row: bool
    detected_time_unit: Optional[str]
    detected_conc_unit_by_column: Dict[PhysicalColumn, Optional[str]]

    def __post_init__(self) -> None:
        detected = dict(self.detected_conc_unit_by_column)
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
        object.__setattr__(self, "detected_conc_unit_by_column", detected)

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
                or name != name.strip()
                for name in sheet_names
            )
            or len(sheet_names) != len(set(sheet_names))
        ):
            raise ValueError(
                "Selected worksheet names must be unique nonblank strings."
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

    filepath: str
    sheet_name: Optional[str]
    time_column: PhysicalColumn
    species_columns: Tuple[PhysicalColumn, ...]
    skip_unit_row: bool
    time_factor: float
    conc_factors: Dict[str, float]
    original_time_unit: str
    original_conc_units: Dict[str, str]

    def __post_init__(self) -> None:
        species_columns = tuple(self.species_columns)
        conc_factors = dict(self.conc_factors)
        original_conc_units = dict(self.original_conc_units)
        if not isinstance(self.filepath, str) or not self.filepath:
            raise ValueError("Resolved import filepath must be nonblank.")
        if self.sheet_name is not None and (
            not isinstance(self.sheet_name, str)
            or not self.sheet_name
            or not self.sheet_name.strip()
        ):
            raise ValueError(
                "Resolved worksheet name must be None or a nonblank string."
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
        object.__setattr__(self, "conc_factors", conc_factors)
        object.__setattr__(self, "original_conc_units", original_conc_units)


@dataclass(frozen=True)
class ImportConfig:
    """Top-level import configuration grouping file intent, sheet intents, and plans."""

    filepath: str
    file_type: str
    file_intent: UserImportIntent
    per_sheet_intents: Tuple[Tuple[Optional[str], SheetImportIntent], ...]
    plans: Tuple[ResolvedSheetPlan, ...]
    remaining_file_template: Optional[SheetImportIntent] = None

    def __post_init__(self) -> None:
        if not isinstance(self.file_intent, UserImportIntent):
            raise TypeError("Import configuration requires UserImportIntent.")
        admitted_type = classify_import_source(self.filepath)
        if str(self.file_type) != admitted_type:
            raise ValueError(
                f"Import type {self.file_type!r} does not match admitted "
                f"source type {admitted_type!r}."
            )
        plans = tuple(self.plans)
        if not plans:
            raise ValueError("Import configuration requires at least one plan.")
        if any(Path(plan.filepath) != Path(self.filepath) for plan in plans):
            raise ValueError(
                "Every import plan must reference the admitted source path."
            )
        if admitted_type == "csv" and (
            len(plans) != 1 or plans[0].sheet_name is not None
        ):
            raise ValueError("A CSV import requires exactly one non-sheet plan.")
        if admitted_type == "excel" and any(
            plan.sheet_name is None for plan in plans
        ):
            raise ValueError("Every XLSX import plan requires a sheet name.")
        intent_items = tuple(self.per_sheet_intents)
        if (
            any(
                not isinstance(intent, SheetImportIntent)
                for _key, intent in intent_items
            )
            or len({key for key, _intent in intent_items})
            != len(intent_items)
        ):
            raise ValueError(
                "Per-sheet import intents must be unique typed intents."
            )
        intent_keys = tuple(key for key, _intent in intent_items)
        plan_keys = tuple(plan.sheet_name for plan in plans)
        if intent_keys != plan_keys:
            raise ValueError(
                "Per-sheet import intent order must exactly match resolved "
                "plan order."
            )
        if admitted_type == "excel" and tuple(
            self.file_intent.sheet_names
        ) != tuple(key for key in intent_keys if key is not None):
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
        object.__setattr__(self, "per_sheet_intents", intent_items)
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

    for col in scoped_columns:
        val = str(row_mapping[col]).strip()
        if not val:
            continue
        try:
            category, _factor = parse_unit(val)
        except ValueError:
            continue
        if category == "time" and detected_time_unit is None:
            detected_time_unit = val
        elif category == "concentration":
            conc_unit_by_column[col] = val

    return UnitDetection(
        has_unit_row=True,
        detected_time_unit=detected_time_unit,
        detected_conc_unit_by_column=conc_unit_by_column,
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
    filepath: str,
    file_type: str,
    per_sheet_intents: Dict[Optional[str], SheetImportIntent],
    per_sheet_detections: Dict[Optional[str], UnitDetection],
    per_sheet_columns: Dict[Optional[str], Sequence[PhysicalColumn]],
) -> List[ResolvedSheetPlan]:
    """Resolve the user's intent into concrete per-sheet import plans.

    Raises ``ValueError`` when the intent is inconsistent with the detected
    file structure (missing sheets, missing columns, or ambiguous
    concentration factors).
    """
    admitted_type = classify_import_source(filepath)
    if str(file_type) != admitted_type:
        raise ValueError(
            f"Import type {file_type!r} does not match admitted source type "
            f"{admitted_type!r}."
        )
    target_sheets = list(per_sheet_intents.keys())
    if not target_sheets:
        raise ValueError("Select at least one CSV source or XLSX sheet.")
    if admitted_type == "csv" and target_sheets != [None]:
        raise ValueError("A CSV import requires exactly one non-sheet intent.")
    if admitted_type == "excel" and any(
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
                filepath=filepath,
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
