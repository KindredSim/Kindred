from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from openpyxl import load_workbook

from kindred.core.datasets.csv_import import parse_csv_rows

__all__ = [
    "ExcelWorkbookLoadResult",
    "list_sheets",
    "load_excel_dataset",
    "load_excel_workbook",
    "read_excel_sheet_rows",
]


@dataclass(frozen=True)
class ExcelWorkbookLoadResult:
    successes: List[Tuple[str, Dict[str, object]]]
    failures: List[Tuple[str, str]]


def list_sheets(filepath: str) -> List[str]:
    """Return workbook sheet names in order."""
    workbook = _open_workbook(filepath)
    try:
        sheet_names = list(workbook.sheetnames)
    finally:
        workbook.close()
    if not sheet_names:
        raise ValueError(f"Excel workbook {filepath!r} has no sheets.")
    return sheet_names


def read_excel_sheet_rows(filepath: str, sheet_name: str) -> Iterable[Mapping[str, str]]:
    """Yield sheet rows as string-valued mappings compatible with parse_csv_rows."""
    def _iter_rows() -> Iterator[Mapping[str, str]]:
        workbook = _open_workbook(filepath)
        try:
            if sheet_name not in workbook.sheetnames:
                raise ValueError(f"Sheet {sheet_name!r} not found in workbook {filepath!r}.")
            sheet = workbook[sheet_name]
            row_iter = sheet.iter_rows(values_only=True)
            header_row = next(row_iter, None)
            if header_row is None:
                raise ValueError(f"Sheet {sheet_name!r} in workbook {filepath!r} has no rows.")
            headers = [_stringify_cell_value(value).strip() for value in header_row]
            for row in row_iter:
                values = [_stringify_cell_value(value) for value in row]
                yield dict(zip(headers, values))
        finally:
            workbook.close()

    return _iter_rows()


def load_excel_dataset(
    filepath: str,
    sheet_name: str,
    time_column: Optional[str] = None,
    species_columns: Optional[Sequence[str]] = None,
    *,
    interruption_checker: Optional[Callable[[], bool]] = None,
) -> Tuple[str, Dict[str, object]]:
    """Load one Excel worksheet into the same payload shape as load_csv_dataset."""
    rows = read_excel_sheet_rows(filepath, sheet_name)
    _time_source, data = parse_csv_rows(
        rows,
        time_column=time_column,
        species_columns=species_columns,
        interruption_checker=interruption_checker,
    )
    dataset_name = f"{os.path.basename(filepath)}::{sheet_name}"
    return dataset_name, data


def load_excel_workbook(
    filepath: str,
    sheet_names: Optional[List[str]] = None,
    time_column: Optional[str] = None,
    species_columns: Optional[Sequence[str]] = None,
    *,
    interruption_checker: Optional[Callable[[], bool]] = None,
) -> ExcelWorkbookLoadResult:
    """Load one or more workbook sheets, collecting successes and failures."""
    selected_sheet_names = list_sheets(filepath) if sheet_names is None else list(sheet_names)
    successes: List[Tuple[str, Dict[str, object]]] = []
    failures: List[Tuple[str, str]] = []
    for sheet_name in selected_sheet_names:
        try:
            successes.append(
                load_excel_dataset(
                    filepath,
                    sheet_name,
                    time_column=time_column,
                    species_columns=species_columns,
                    interruption_checker=interruption_checker,
                )
            )
        except Exception as exc:
            failures.append((sheet_name, str(exc)))
    return ExcelWorkbookLoadResult(successes=successes, failures=failures)


def _open_workbook(filepath: str):  # noqa: ANN202
    try:
        return load_workbook(filepath, read_only=True, data_only=True)
    except Exception as exc:
        raise ValueError(f"Failed to open Excel workbook {filepath!r}: {exc}") from exc


def _stringify_cell_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)
