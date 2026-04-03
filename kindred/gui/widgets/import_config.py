"""Composed import configuration types and pure helpers for dataset import.

Defines the structured types that replace the flat ImportConfig dataclass:
UnitDetection, UserImportIntent, ResolvedSheetPlan, and the top-level
ImportConfig. Also provides ``detect_units_from_row_mapping``,
``rebuild_intent_for_target``, and ``resolve_import_plans`` for building and
adapting these objects from raw preview data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    "detect_units_from_row_mapping",
    "rebuild_intent_for_target",
    "resolve_import_plans",
]


@dataclass(frozen=True)
class UnitDetection:
    """Result of heuristic unit-row detection on a single sheet/file."""

    has_unit_row: bool
    detected_time_unit: Optional[str]
    detected_conc_unit_by_column: Dict[str, Optional[str]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "detected_conc_unit_by_column", dict(self.detected_conc_unit_by_column))

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


@dataclass(frozen=True)
class SheetImportIntent:
    """Captures the user's choices for one sheet (or one CSV file)."""

    time_column: str
    species_columns: Tuple[str, ...]
    time_unit: str
    concentration_units: Dict[str, str]  # col_name → unit string
    override_no_unit_row: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "concentration_units", dict(self.concentration_units))
        if set(self.concentration_units) != set(self.species_columns):
            raise ValueError("concentration_units keys must match species_columns")


@dataclass(frozen=True)
class ResolvedSheetPlan:
    """Fully resolved import plan for a single sheet (or CSV file)."""

    filepath: str
    sheet_name: Optional[str]
    time_column: str
    species_columns: Tuple[str, ...]
    skip_unit_row: bool
    time_factor: float
    conc_factors: Dict[str, float]  # col_name → factor
    original_time_unit: str
    original_conc_units: Dict[str, str]  # col_name → unit string

    def __post_init__(self) -> None:
        object.__setattr__(self, "conc_factors", dict(self.conc_factors))
        object.__setattr__(self, "original_conc_units", dict(self.original_conc_units))


@dataclass(frozen=True)
class ImportConfig:
    """Top-level import configuration grouping file intent, sheet intents, and plans."""

    filepath: str
    file_type: str
    file_intent: UserImportIntent
    per_sheet_intents: Tuple[Tuple[Optional[str], SheetImportIntent], ...]
    plans: Tuple[ResolvedSheetPlan, ...]
    remaining_file_template: Optional[SheetImportIntent] = None


def detect_units_from_row_mapping(
    row_mapping: dict,
    relevant_column_names: Optional[Sequence[str]] = None,
) -> UnitDetection:
    """Build a ``UnitDetection`` from a column-name-to-cell-value mapping.

    Parameters
    ----------
    row_mapping:
        Mapping of column name to the cell value in the candidate unit row.
    relevant_column_names:
        If provided, unit extraction is restricted to these columns while
        ``has_unit_row`` still inspects the full row so unit-like text in
        unselected columns can still identify a physical unit row.
    """
    full_values = [str(value).strip() for value in row_mapping.values()]
    if not looks_like_unit_row(full_values):
        return UnitDetection.empty()

    if relevant_column_names is not None:
        scoped_columns = [col for col in relevant_column_names if col in row_mapping]
    else:
        scoped_columns = list(row_mapping.keys())

    detected_time_unit: Optional[str] = None
    conc_unit_by_column: Dict[str, Optional[str]] = {}

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
    source_intent: SheetImportIntent,
    target_detection: UnitDetection,
) -> SheetImportIntent:
    """Build a ``SheetImportIntent`` for a target sheet or file."""
    conc_units: Dict[str, str] = {}
    for col in source_intent.species_columns:
        if not source_intent.override_no_unit_row:
            target_detected = target_detection.detected_conc_unit_by_column.get(col)
        else:
            target_detected = None
        if target_detected is not None:
            conc_units[col] = target_detected
        else:
            conc_units[col] = source_intent.concentration_units[col]
    return SheetImportIntent(
        time_column=source_intent.time_column,
        species_columns=source_intent.species_columns,
        time_unit=source_intent.time_unit,
        concentration_units=conc_units,
        override_no_unit_row=source_intent.override_no_unit_row,
    )


def resolve_import_plans(
    filepath: str,
    file_type: str,
    per_sheet_intents: Dict[Optional[str], SheetImportIntent],
    per_sheet_detections: Dict[Optional[str], UnitDetection],
    per_sheet_columns: Dict[Optional[str], Sequence[str]],
) -> List[ResolvedSheetPlan]:
    """Resolve the user's intent into concrete per-sheet import plans.

    Raises ``ValueError`` when the intent is inconsistent with the detected
    file structure (missing sheets, missing columns, or ambiguous
    concentration factors).
    """
    target_sheets = list(per_sheet_intents.keys())

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
                f"Time column {sheet_intent.time_column!r} not found in sheet {sheet!r}. "
                f"Available columns: {list(columns)}"
            )

    # (c) Each sheet's configured species columns must exist.
    for sheet in target_sheets:
        columns = per_sheet_columns[sheet]
        sheet_intent = per_sheet_intents[sheet]
        for species_col in sheet_intent.species_columns:
            if species_col not in columns:
                raise ValueError(
                    f"Species column {species_col!r} not found in sheet {sheet!r}. "
                    f"Available columns: {list(columns)}"
                )

    # Build plans.
    plans: List[ResolvedSheetPlan] = []
    for sheet in target_sheets:
        detection = per_sheet_detections[sheet]
        sheet_intent = per_sheet_intents[sheet]
        species_columns = sheet_intent.species_columns

        skip_unit_row = detection.has_unit_row

        if sheet_intent.override_no_unit_row:
            time_unit = "s"
            time_factor = 1.0
            conc_factors = {col: 1.0 for col in species_columns}
            original_conc_units = {col: "M" for col in species_columns}
        else:
            time_unit = sheet_intent.time_unit
            time_factor = parse_time_unit(time_unit)
            conc_factors = {}
            original_conc_units = {}
            for col in species_columns:
                col_unit = sheet_intent.concentration_units[col]
                conc_factors[col] = parse_concentration_unit(col_unit)
                original_conc_units[col] = col_unit

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
