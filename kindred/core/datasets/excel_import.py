from __future__ import annotations

import datetime as dt
import os
from contextlib import closing
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from openpyxl import load_workbook

from kindred.core.datasets.csv_import import CsvImportInterrupted, parse_csv_rows
from kindred.core.datasets.units import looks_like_unit_row

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
            yield from _read_worksheet_rows(workbook[sheet_name], filepath=filepath, sheet_name=sheet_name)
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
    with closing(_strip_unit_row(read_excel_sheet_rows(filepath, sheet_name))) as rows:
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
    workbook = _open_workbook(filepath)
    try:
        available_sheet_names = list(workbook.sheetnames)
        if not available_sheet_names:
            raise ValueError(f"Excel workbook {filepath!r} has no sheets.")
        selected_sheet_names = available_sheet_names if sheet_names is None else list(sheet_names)
        successes: List[Tuple[str, Dict[str, object]]] = []
        failures: List[Tuple[str, str]] = []
        for sheet_name in selected_sheet_names:
            try:
                if sheet_name not in workbook.sheetnames:
                    raise ValueError(f"Sheet {sheet_name!r} not found in workbook {filepath!r}.")
                with closing(
                    _strip_unit_row(
                        _read_worksheet_rows(
                            workbook[sheet_name],
                            filepath=filepath,
                            sheet_name=sheet_name,
                        )
                    )
                ) as rows:
                    _time_source, data = parse_csv_rows(
                        rows,
                        time_column=time_column,
                        species_columns=species_columns,
                        interruption_checker=interruption_checker,
                    )
                dataset_name = f"{os.path.basename(filepath)}::{sheet_name}"
                successes.append((dataset_name, data))
            except CsvImportInterrupted:
                raise
            except Exception as exc:
                failures.append((sheet_name, str(exc)))
        return ExcelWorkbookLoadResult(successes=successes, failures=failures)
    finally:
        workbook.close()


def _read_worksheet_rows(
    worksheet, *, filepath: str, sheet_name: str
) -> Iterator[Mapping[str, str]]:  # noqa: ANN001
    row_iter = worksheet.iter_rows(values_only=True)
    header_row = next(row_iter, None)
    if header_row is None:
        raise ValueError(f"Sheet {sheet_name!r} in workbook {filepath!r} has no rows.")
    headers = [_stringify_cell_value(value).strip() for value in header_row]
    for row in row_iter:
        values = [_stringify_cell_value(value) for value in row]
        yield dict(zip(headers, values))


def _open_workbook(filepath: str):  # noqa: ANN202
    try:
        return load_workbook(filepath, read_only=True, data_only=True)
    except Exception as exc:
        raise ValueError(f"Failed to open Excel workbook {filepath!r}: {exc}") from exc


def _strip_unit_row(rows: Iterable[Mapping[str, str]]) -> Iterable[Mapping[str, str]]:
    row_iter = iter(rows)
    try:
        first_row = next(row_iter)
    except StopIteration:
        def _empty_rows() -> Iterator[Mapping[str, str]]:
            _close_rows(rows)
            yield from ()

        return _empty_rows()
    except Exception:
        _close_rows(rows)
        raise

    def _iter_rows() -> Iterator[Mapping[str, str]]:
        try:
            if not looks_like_unit_row(list(first_row.values())):
                yield first_row
            yield from row_iter
        finally:
            _close_rows(rows)

    return _iter_rows()


def _close_rows(rows: Iterable[Mapping[str, str]]) -> None:
    close = getattr(rows, "close", None)
    if callable(close):
        close()


def _stringify_cell_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return repr(value.timestamp())
    if isinstance(value, dt.date):
        return repr(dt.datetime.combine(value, dt.time(), tzinfo=dt.timezone.utc).timestamp())
    if isinstance(value, dt.time):
        return repr(
            (
                value.hour * 3600
                + value.minute * 60
                + value.second
                + value.microsecond / 1_000_000
            )
        )
    if isinstance(value, float):
        return repr(value)
    return str(value)
