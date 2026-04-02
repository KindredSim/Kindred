"""Composed import configuration types and resolver for the dataset import pipeline.

Defines the structured types that replace the flat ImportConfig dataclass:
UnitDetection, UserImportIntent, ResolvedSheetPlan, and the top-level
ImportConfig.  Also provides ``detect_units_from_row_mapping`` and
``resolve_import_plans`` for building these objects from raw preview data.
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
    "UnitDetection",
    "UserImportIntent",
    "detect_units_from_row_mapping",
    "resolve_import_plans",
]


@dataclass(frozen=True)
class UnitDetection:
    """Result of heuristic unit-row detection on a single sheet/file."""

    has_unit_row: bool
    detected_time_unit: Optional[str]
    detected_conc_unit: Optional[str]
    detected_conc_units: Tuple[str, ...]

    @staticmethod
    def empty() -> UnitDetection:
        return UnitDetection(
            has_unit_row=False,
            detected_time_unit=None,
            detected_conc_unit=None,
            detected_conc_units=(),
        )


@dataclass(frozen=True)
class UserImportIntent:
    """Captures the user's choices from the import configuration dialog."""

    time_column: str
    species_columns: Tuple[str, ...]
    time_unit: str
    concentration_unit: str
    override_no_unit_row: bool
    sheet_names: Tuple[str, ...]
    apply_to_remaining: bool


@dataclass(frozen=True)
class ResolvedSheetPlan:
    """Fully resolved import plan for a single sheet (or CSV file)."""

    filepath: str
    sheet_name: Optional[str]
    time_column: str
    species_columns: Tuple[str, ...]
    skip_unit_row: bool
    time_factor: float
    conc_factor: float
    original_time_unit: str
    original_conc_unit: str


@dataclass(frozen=True)
class ImportConfig:
    """Top-level import configuration grouping detection, intent, and plans."""

    filepath: str
    file_type: str
    detection: UnitDetection
    intent: UserImportIntent
    plans: Tuple[ResolvedSheetPlan, ...]


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
        If provided, both the ``has_unit_row`` heuristic and the unit
        extraction are restricted to these columns so that unselected
        columns (which may contain unit-like text) do not influence the
        result.
    """
    if relevant_column_names is not None:
        scoped_values = [
            str(row_mapping[col]).strip()
            for col in relevant_column_names
            if col in row_mapping
        ]
    else:
        scoped_values = [str(v).strip() for v in row_mapping.values()]

    if not looks_like_unit_row(scoped_values):
        return UnitDetection.empty()

    extraction_values = scoped_values

    detected_time_unit: Optional[str] = None
    detected_conc_unit: Optional[str] = None
    conc_units_seen: list[str] = []

    for val in extraction_values:
        if not val:
            continue
        try:
            category, _factor = parse_unit(val)
        except ValueError:
            continue
        if category == "time" and detected_time_unit is None:
            detected_time_unit = val
        elif category == "concentration":
            if detected_conc_unit is None:
                detected_conc_unit = val
            if val not in conc_units_seen:
                conc_units_seen.append(val)

    return UnitDetection(
        has_unit_row=True,
        detected_time_unit=detected_time_unit,
        detected_conc_unit=detected_conc_unit,
        detected_conc_units=tuple(conc_units_seen),
    )


def resolve_import_plans(
    filepath: str,
    file_type: str,
    intent: UserImportIntent,
    per_sheet_detections: Dict[Optional[str], UnitDetection],
    per_sheet_columns: Dict[Optional[str], Sequence[str]],
) -> List[ResolvedSheetPlan]:
    """Resolve the user's intent into concrete per-sheet import plans.

    Raises ``ValueError`` when the intent is inconsistent with the detected
    file structure (missing sheets, missing columns, mixed unit rows, or
    ambiguous concentration factors).
    """
    target_sheets: List[Optional[str]]
    if file_type == "csv":
        target_sheets = [None]
    else:
        target_sheets = list(intent.sheet_names)  # type: ignore[arg-type]

    # (a) Every target sheet must have a detection and columns entry.
    for sheet in target_sheets:
        if sheet not in per_sheet_detections:
            raise ValueError(
                f"No unit detection available for sheet {sheet!r}."
            )
        if sheet not in per_sheet_columns:
            raise ValueError(
                f"No column information available for sheet {sheet!r}."
            )

    # (b) intent.time_column exists in every target sheet's columns.
    for sheet in target_sheets:
        columns = per_sheet_columns[sheet]
        if intent.time_column not in columns:
            raise ValueError(
                f"Time column {intent.time_column!r} not found in sheet {sheet!r}. "
                f"Available columns: {list(columns)}"
            )

    # (c) All intent.species_columns exist in every target sheet.
    for sheet in target_sheets:
        columns = per_sheet_columns[sheet]
        for species_col in intent.species_columns:
            if species_col not in columns:
                raise ValueError(
                    f"Species column {species_col!r} not found in sheet {sheet!r}. "
                    f"Available columns: {list(columns)}"
                )

    # (d) Cross-sheet: all sheets agree on has_unit_row (only when >1 sheet).
    if len(target_sheets) > 1:
        unit_row_flags = {per_sheet_detections[s].has_unit_row for s in target_sheets}
        if len(unit_row_flags) > 1:
            raise ValueError(
                "Sheets disagree on unit-row presence. "
                "Some sheets have a unit row and some do not."
            )

    # Build plans.
    plans: List[ResolvedSheetPlan] = []
    for sheet in target_sheets:
        detection = per_sheet_detections[sheet]

        # (e) Per-sheet: reject if >1 distinct concentration FACTOR among
        #     detected_conc_units when has_unit_row=True and NOT override.
        if detection.has_unit_row and not intent.override_no_unit_row:
            if len(detection.detected_conc_units) > 1:
                factors_seen: set[float] = set()
                for unit_str in detection.detected_conc_units:
                    factors_seen.add(parse_concentration_unit(unit_str))
                if len(factors_seen) > 1:
                    raise ValueError(
                        f"Sheet {sheet!r} contains multiple distinct concentration "
                        f"factors among detected units {detection.detected_conc_units}. "
                        f"Cannot resolve a single concentration factor automatically."
                    )

        skip_unit_row = detection.has_unit_row

        if intent.override_no_unit_row:
            time_unit = "s"
            conc_unit = "M"
            time_factor = 1.0
            conc_factor = 1.0
        elif detection.has_unit_row:
            time_unit = intent.time_unit
            conc_unit = intent.concentration_unit
            time_factor = parse_time_unit(time_unit)
            conc_factor = parse_concentration_unit(conc_unit)
        else:
            time_unit = intent.time_unit
            conc_unit = intent.concentration_unit
            time_factor = parse_time_unit(time_unit)
            conc_factor = parse_concentration_unit(conc_unit)

        plans.append(
            ResolvedSheetPlan(
                filepath=filepath,
                sheet_name=sheet,
                time_column=intent.time_column,
                species_columns=intent.species_columns,
                skip_unit_row=skip_unit_row,
                time_factor=time_factor,
                conc_factor=conc_factor,
                original_time_unit=time_unit,
                original_conc_unit=conc_unit,
            )
        )

    return plans
