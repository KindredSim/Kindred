from __future__ import annotations

import datetime as dt
from io import BytesIO
from typing import Iterator, List, Mapping
from zipfile import BadZipFile, ZipFile

from openpyxl import load_workbook

from kindred.core.datasets.csv_import import (
    UncachedFormulaCell,
)
from kindred.core.datasets.source_rules import (
    PhysicalColumn,
    identify_physical_columns,
    validate_tabular_source_bytes,
)

__all__ = [
    "ExcelWorkbookPair",
]


class ExcelWorkbookPair:
    """One explicitly owned read-only data/formula workbook pair.

    A pair is deliberately scoped to one preview, worker, or source-plan
    operation.  It is not a cache: callers must enter and close it, and sheet
    row iterators must be consumed before the pair is closed.
    """

    def __init__(self, source_bytes: bytes, *, source_label: str) -> None:
        self._source_bytes = bytes(source_bytes)
        self._source_label = str(source_label).strip() or "<workbook>"
        self._workbook = None
        self._authored_workbook = None

    def __enter__(self) -> "ExcelWorkbookPair":
        if self._workbook is not None:
            raise RuntimeError("Excel workbook pair is already open.")
        try:
            _validate_xlsx_zip(self._source_bytes, source_label=self._source_label)
            self._workbook = _open_workbook_bytes(
                self._source_bytes,
                source_label=self._source_label,
                data_only=True,
            )
            self._authored_workbook = _open_workbook_bytes(
                self._source_bytes,
                source_label=self._source_label,
                data_only=False,
            )
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        workbook, authored = self._workbook, self._authored_workbook
        self._workbook = None
        self._authored_workbook = None
        for item in (authored, workbook):
            if item is not None:
                item.close()

    def _require_open(self):  # noqa: ANN202
        if self._workbook is None or self._authored_workbook is None:
            raise RuntimeError("Excel workbook pair is closed.")
        return self._workbook, self._authored_workbook

    @property
    def sheet_names(self) -> List[str]:
        workbook, _authored = self._require_open()
        return list(workbook.sheetnames)

    def read_sheet_rows(
        self,
        sheet_name: str,
    ) -> Iterator[Mapping[PhysicalColumn, object]]:
        workbook, authored_workbook = self._require_open()
        if sheet_name not in workbook.sheetnames:
            raise ValueError(
                f"Sheet {sheet_name!r} not found in workbook "
                f"{self._source_label!r}."
            )
        if sheet_name not in authored_workbook.sheetnames:
            raise ValueError(
                f"Sheet {sheet_name!r} not found in workbook "
                f"{self._source_label!r}."
            )
        yield from _read_worksheet_rows(
            workbook[sheet_name],
            authored_worksheet=authored_workbook[sheet_name],
            filepath=self._source_label,
            sheet_name=sheet_name,
        )

def _exact_worksheet_width(worksheet) -> int:  # noqa: ANN001
    """Return the declared width or exhaustively discover it from real rows."""
    declared = worksheet.max_column
    if declared is not None:
        width = int(declared)
        if width <= 0:
            raise ValueError("Worksheet column extent must be positive.")
        return width
    return max(
        (len(row) for row in worksheet.iter_rows(values_only=False)),
        default=0,
    )


def _read_worksheet_rows(
    worksheet,
    *,
    authored_worksheet,
    filepath: str,
    sheet_name: str,
) -> Iterator[Mapping[PhysicalColumn, object]]:  # noqa: ANN001
    physical_width = max(
        _exact_worksheet_width(worksheet),
        _exact_worksheet_width(authored_worksheet),
    )
    if physical_width <= 0:
        raise ValueError(f"Sheet {sheet_name!r} in workbook {filepath!r} has no rows.")
    cell_rows = worksheet.iter_rows(
        max_col=physical_width,
        values_only=False,
    )
    authored_rows = authored_worksheet.iter_rows(
        max_col=physical_width,
        values_only=False,
    )
    header_cells = next(cell_rows, None)
    authored_header_cells = next(authored_rows, None)
    if header_cells is None or authored_header_cells is None:
        raise ValueError(f"Sheet {sheet_name!r} in workbook {filepath!r} has no rows.")
    header_values = tuple(
        _stringify_cell_value(cell.value) for cell in header_cells
    )
    columns = identify_physical_columns(header_values)
    # A worksheet with no named physical headers is a valid, non-importable
    # sheet.  Returning an empty iterator lets the dialog cache that state
    # without making the entire workbook fatal.
    if not columns:
        return
    retained_width = max(column.source_index for column in columns) + 1
    row_iter = worksheet.iter_rows(
        min_row=2,
        max_col=retained_width,
        values_only=False,
    )
    authored_row_iter = authored_worksheet.iter_rows(
        min_row=2,
        max_col=retained_width,
        values_only=False,
    )
    for row, authored_row in zip(row_iter, authored_row_iter, strict=True):
        values: list[object] = []
        for cell, authored_cell in zip(row, authored_row, strict=True):
            if (
                authored_cell.data_type == "f"
                and cell.value is None
            ):
                formula = str(authored_cell.value or "")
                if not formula.startswith("="):
                    formula = f"={formula}"
                values.append(UncachedFormulaCell(formula))
            else:
                values.append(_stringify_cell_value(cell.value))
        yield {
            column: values[column.source_index]
            if column.source_index < len(values)
            else ""
            for column in columns
        }


def _open_workbook_bytes(
    source_bytes: bytes,
    *,
    source_label: str,
    data_only: bool,
):  # noqa: ANN202
    try:
        return load_workbook(
            BytesIO(source_bytes),
            read_only=True,
            data_only=data_only,
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to open Excel workbook {source_label!r}: {exc}"
        ) from exc


def _validate_xlsx_zip(source_bytes: bytes, *, source_label: str) -> None:
    validate_tabular_source_bytes(
        source_bytes,
        source_format="xlsx",
        source_label=source_label,
    )
    try:
        with ZipFile(BytesIO(source_bytes)) as archive:
            archive.infolist()
    except (BadZipFile, OSError, ValueError) as exc:
        raise ValueError(
            f"Failed to inspect Excel workbook {source_label!r}: {exc}"
        ) from exc


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
