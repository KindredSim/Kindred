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
    "SheetImportIntent",
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

    sheet_names: Tuple[str, ...]
    apply_to_remaining: bool


@dataclass(frozen=True)
class SheetImportIntent:
    """Captures the user's choices for one sheet (or one CSV file)."""

    time_column: str
    species_columns: Tuple[str, ...]
    time_unit: str
    concentration_unit: str
    override_no_unit_row: bool


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
    """Top-level import configuration grouping file intent, sheet intents, and plans."""

    filepath: str
    file_type: str
    file_intent: UserImportIntent
    per_sheet_intents: Tuple[Tuple[Optional[str], SheetImportIntent], ...]
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
        If provided, unit extraction is restricted to these columns while
        ``has_unit_row`` still inspects the full row so unit-like text in
        unselected columns can still identify a physical unit row.
    """
    full_values = [str(value).strip() for value in row_mapping.values()]
    if not looks_like_unit_row(full_values):
        return UnitDetection.empty()

    if relevant_column_names is not None:
        scoped_values = [
            str(row_mapping[col]).strip()
            for col in relevant_column_names
            if col in row_mapping
        ]
    else:
        scoped_values = full_values

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

        # (e) Per-sheet: reject if >1 distinct concentration FACTOR among
        #     detected_conc_units when has_unit_row=True and NOT override.
        if detection.has_unit_row and not sheet_intent.override_no_unit_row:
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

        if sheet_intent.override_no_unit_row:
            time_unit = "s"
            conc_unit = "M"
            time_factor = 1.0
            conc_factor = 1.0
        else:
            time_unit = sheet_intent.time_unit
            conc_unit = sheet_intent.concentration_unit
            time_factor = parse_time_unit(time_unit)
            conc_factor = parse_concentration_unit(conc_unit)

        plans.append(
            ResolvedSheetPlan(
                filepath=filepath,
                sheet_name=sheet,
                time_column=sheet_intent.time_column,
                species_columns=sheet_intent.species_columns,
                skip_unit_row=skip_unit_row,
                time_factor=time_factor,
                conc_factor=conc_factor,
                original_time_unit=time_unit,
                original_conc_unit=conc_unit,
            )
        )

    return plans
