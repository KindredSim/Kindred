from __future__ import annotations

import csv
import itertools
import logging
import os
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from kindred.core.datasets.observation_payload import dataset_payload_from_observations

logger = logging.getLogger(__name__)

__all__ = ["CsvImportInterrupted", "load_csv_dataset", "parse_csv_rows"]


class CsvImportInterrupted(Exception):
    """Raised internally to unwind when an import is interrupted."""


def load_csv_dataset(
    filepath: str,
    time_column: Optional[str] = None,
    species_columns: Optional[Sequence[str]] = None,
    *,
    interruption_checker: Optional[Callable[[], bool]] = None,
) -> Tuple[str, Dict[str, object]]:
    """
    Load a CSV dataset using explicit or automatic column mapping.

    Returns (dataset_name, canonical dataset payload).
    """
    with open(filepath, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _time_source, data = parse_csv_rows(
            reader,
            time_column=time_column,
            species_columns=species_columns,
            interruption_checker=interruption_checker,
            source_label=os.path.basename(filepath),
        )
    dataset_name = os.path.basename(filepath)
    return dataset_name, data


def parse_csv_rows(
    rows: Iterable[Mapping[str, str]],
    time_column: Optional[str] = None,
    species_columns: Optional[Sequence[str]] = None,
    interruption_checker: Optional[Callable[[], bool]] = None,
    *,
    source_label: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> Tuple[str, Dict[str, object]]:
    """Convert CSV rows to a dataset payload."""

    def _raise_if_interrupted() -> None:
        if interruption_checker and interruption_checker():
            raise CsvImportInterrupted()

    row_iter = iter(rows)
    first = next(row_iter, None)
    if first is None:
        raise ValueError("CSV file is empty")

    columns = list(first.keys())
    explicit_mapping = bool(time_column or species_columns)
    explicit_species_mapping = bool(species_columns)

    # Resolve time column
    if time_column:
        if time_column not in columns:
            raise ValueError(
                f"Time column '{time_column}' not found. Available columns: {', '.join(columns)}"
            )
        time_col = time_column
    else:
        time_col = None
        time_candidates = ["time", "time_s", "t", "Time", "T", "x"]
        for candidate in time_candidates:
            if candidate in columns:
                time_col = candidate
                break
        if time_col is None:
            raise ValueError(
                "Time column not found. Provide a column name or ensure one of "
                f"{', '.join(repr(c) for c in time_candidates)} exists."
            )

    _raise_if_interrupted()

    def _normalize_cell(value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _row_snippet(row: Dict[str, str], max_len: int = 120) -> str:
        parts = [f"{col}={row.get(col, '')!r}" for col in columns]
        snippet = ", ".join(parts)
        if len(snippet) > max_len:
            return snippet[: max_len - 3] + "..."
        return snippet

    def _source_prefix() -> str:
        parts: list[str] = []
        if str(source_label or "").strip():
            parts.append(f"file '{str(source_label).strip()}'")
        if str(sheet_name or "").strip():
            parts.append(f"sheet '{str(sheet_name).strip()}'")
        if not parts:
            return ""
        return ", ".join(parts) + ", "

    def _species_error(row_index: int, column: str, raw_value: str, exc: BaseException) -> ValueError:
        return ValueError(
            f"{_source_prefix()}row {row_index}, column '{column}': invalid numeric observation "
            f"{raw_value!r} ({exc})."
        )

    # Resolve species columns
    if species_columns:
        requested = [name.strip() for name in species_columns if name and name.strip()]
        if not requested:
            raise ValueError("Provide at least one species column name.")
        species_candidates: List[str] = []
        for name in requested:
            if name == time_col:
                raise ValueError(f"Species column '{name}' cannot be the same as the time column.")
            if name not in columns:
                raise ValueError(
                    f"Species column '{name}' not found. Available columns: {', '.join(columns)}"
                )
            species_candidates.append(name)
    else:
        species_candidates = [col for col in columns if col != time_col]

    active_species_columns = list(species_candidates)
    species_observations: Dict[str, Dict[str, List[float]]] = {
        col: {"t": [], "y": []} for col in active_species_columns
    }
    saw_nonempty_row = False

    for row_index, row in enumerate(itertools.chain((first,), row_iter), start=1):
        _raise_if_interrupted()
        normalized_row = {col: _normalize_cell(row.get(col, "")) for col in columns}

        if not any(value for value in normalized_row.values()):
            continue
        saw_nonempty_row = True

        time_value = normalized_row.get(time_col, "")
        if not time_value:
            raise ValueError(
                f"Row {row_index}: missing value in time column '{time_col}' "
                f"for non-empty row ({_row_snippet(normalized_row)})."
            )

        try:
            parsed_time = float(time_value)
        except Exception as exc:
            raise ValueError(
                f"Row {row_index}: invalid numeric value {time_value!r} in time column "
                f"'{time_col}' ({_row_snippet(normalized_row)})."
            ) from exc

        for col in list(active_species_columns):
            _raise_if_interrupted()
            raw_value = normalized_row[col]
            if not raw_value:
                continue
            try:
                parsed_value = float(raw_value)
            except Exception as exc:
                error = _species_error(row_index, col, raw_value, exc)
                logger.debug("Invalid non-numeric observation in column '%s': %s", col, exc)
                if explicit_species_mapping:
                    raise error from exc
                # Auto mode salvages usable numeric observations row-by-row and
                # treats nonnumeric cells as absent observations for that row.
                continue
            species_observations[col]["t"].append(parsed_time)
            species_observations[col]["y"].append(parsed_value)

    if not saw_nonempty_row:
        raise ValueError("CSV file contains only blank lines in the data section.")

    observations: Dict[str, Dict[str, np.ndarray]] = {}
    for col in species_candidates:
        _raise_if_interrupted()
        spec = species_observations.get(col)
        if spec is None or not spec["t"]:
            continue
        observations[col] = {
            "t": np.asarray(spec["t"], dtype=float),
            "y": np.asarray(spec["y"], dtype=float),
        }

    if not observations:
        raise ValueError("No numeric species columns found. Provide explicit column names.")

    mapping_source = "explicit" if explicit_mapping else "auto"
    metadata = {
        "time_column": time_col,
        "species_columns": list(observations.keys()),
        "mapping_source": mapping_source,
    }

    data = dataset_payload_from_observations(observations, metadata=metadata)
    return str(time_col), data
