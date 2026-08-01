from __future__ import annotations

import datetime as dt
from io import BytesIO
from typing import Iterable, Iterator, List, Mapping

from openpyxl import load_workbook

from kindred.core.datasets.csv_import import (
    PhysicalColumn,
    identify_physical_columns,
)

__all__ = [
    "list_sheets",
    "read_excel_sheet_rows",
    "read_excel_sheet_rows_from_bytes",
]


def list_sheets(filepath: str) -> List[str]:
    """Return workbook sheet names in order."""
    workbook = _open_workbook(filepath, data_only=True)
    try:
        sheet_names = list(workbook.sheetnames)
    finally:
        workbook.close()
    if not sheet_names:
        raise ValueError(f"Excel workbook {filepath!r} has no sheets.")
    return sheet_names


def read_excel_sheet_rows(
    filepath: str,
    sheet_name: str,
) -> Iterable[Mapping[PhysicalColumn, str]]:
    """Yield worksheet rows keyed by unique physical-column identities."""

    def _iter_rows() -> Iterator[Mapping[PhysicalColumn, str]]:
        workbook = _open_workbook(filepath, data_only=True)
        authored_workbook = None
        try:
            if sheet_name not in workbook.sheetnames:
                raise ValueError(f"Sheet {sheet_name!r} not found in workbook {filepath!r}.")
            worksheet = workbook[sheet_name]
            authored_worksheet = worksheet
            if _requires_authored_tail_scan(worksheet):
                authored_workbook = _open_workbook(filepath, data_only=False)
                if sheet_name not in authored_workbook.sheetnames:
                    raise ValueError(
                        f"Sheet {sheet_name!r} not found in workbook "
                        f"{filepath!r}."
                    )
                authored_worksheet = authored_workbook[sheet_name]
            yield from _read_worksheet_rows(
                worksheet,
                authored_worksheet=authored_worksheet,
                filepath=filepath,
                sheet_name=sheet_name,
            )
        finally:
            workbook.close()
            if authored_workbook is not None:
                authored_workbook.close()

    return _iter_rows()


def read_excel_sheet_rows_from_bytes(
    source_bytes: bytes,
    sheet_name: str,
    *,
    source_label: str,
) -> Iterable[Mapping[PhysicalColumn, str]]:
    """Yield rows from the exact workbook bytes retained by an import worker."""
    owned_bytes = bytes(source_bytes)
    label = str(source_label).strip() or "<workbook>"

    def _iter_rows() -> Iterator[Mapping[PhysicalColumn, str]]:
        workbook = _open_workbook_bytes(
            owned_bytes,
            source_label=label,
            data_only=True,
        )
        authored_workbook = None
        try:
            if sheet_name not in workbook.sheetnames:
                raise ValueError(
                    f"Sheet {sheet_name!r} not found in workbook {label!r}."
                )
            worksheet = workbook[sheet_name]
            authored_worksheet = worksheet
            if _requires_authored_tail_scan(worksheet):
                authored_workbook = _open_workbook_bytes(
                    owned_bytes,
                    source_label=label,
                    data_only=False,
                )
                if sheet_name not in authored_workbook.sheetnames:
                    raise ValueError(
                        f"Sheet {sheet_name!r} not found in workbook "
                        f"{label!r}."
                    )
                authored_worksheet = authored_workbook[sheet_name]
            yield from _read_worksheet_rows(
                worksheet,
                authored_worksheet=authored_worksheet,
                filepath=label,
                sheet_name=sheet_name,
            )
        finally:
            workbook.close()
            if authored_workbook is not None:
                authored_workbook.close()

    return _iter_rows()


def _requires_authored_tail_scan(worksheet) -> bool:  # noqa: ANN001
    header = next(
        worksheet.iter_rows(
            min_row=1,
            max_row=1,
            values_only=True,
        ),
        None,
    )
    if header is None:
        return False
    values = tuple(_stringify_cell_value(value) for value in header)
    named_indexes = tuple(
        index for index, value in enumerate(values) if value.strip()
    )
    return bool(named_indexes and named_indexes[-1] + 1 < len(values))


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
) -> Iterator[Mapping[PhysicalColumn, str]]:  # noqa: ANN001
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
    named_indexes = tuple(
        index for index, value in enumerate(header_values) if value.strip()
    )
    retained_width = len(header_values)
    if named_indexes:
        first_trailing_index = named_indexes[-1] + 1
        if first_trailing_index < retained_width:
            trailing_indexes = tuple(
                range(first_trailing_index, retained_width)
            )
            style_seen = {
                index: bool(
                    getattr(header_cells[index], "has_style", False)
                    or getattr(
                        authored_header_cells[index],
                        "has_style",
                        False,
                    )
                )
                for index in trailing_indexes
            }
            value_seen = {index: False for index in trailing_indexes}
            for cells, authored_cells in zip(
                cell_rows,
                authored_rows,
                strict=True,
            ):
                for index in trailing_indexes:
                    if index >= len(cells) or index >= len(authored_cells):
                        continue
                    cell = cells[index]
                    authored_cell = authored_cells[index]
                    if (
                        cell.value is not None
                        or authored_cell.value is not None
                    ):
                        value_seen[index] = True
                    if (
                        getattr(cell, "has_style", False)
                        or getattr(authored_cell, "has_style", False)
                    ):
                        style_seen[index] = True
            if not any(value_seen.values()) and all(style_seen.values()):
                retained_width = first_trailing_index

    header_row = header_values[:retained_width]
    columns = identify_physical_columns(
        header_row
    )
    row_iter = worksheet.iter_rows(
        min_row=2,
        max_col=retained_width,
        values_only=True,
    )
    for row in row_iter:
        values = [_stringify_cell_value(value) for value in row]
        yield {
            column: values[column.source_index]
            if column.source_index < len(values)
            else ""
            for column in columns
        }


def _open_workbook(filepath: str, *, data_only: bool):  # noqa: ANN202
    try:
        return load_workbook(
            filepath,
            read_only=True,
            data_only=data_only,
        )
    except Exception as exc:
        raise ValueError(f"Failed to open Excel workbook {filepath!r}: {exc}") from exc


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
