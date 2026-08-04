from __future__ import annotations

import datetime as dt
from io import BytesIO
from typing import Iterable, Iterator, List, Mapping
from zipfile import BadZipFile, ZipFile

from openpyxl import load_workbook

from kindred.core.datasets.csv_import import (
    PhysicalColumn,
    UncachedFormulaCell,
    identify_physical_columns,
    validate_tabular_source_bytes,
)

__all__ = [
    "list_sheets_from_bytes",
    "read_excel_sheet_rows_from_bytes",
]

_MAX_XLSX_ZIP_MEMBERS = 512
_MAX_XLSX_MEMBER_BYTES = 128 * 1024 * 1024
_MAX_XLSX_TOTAL_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_PHYSICAL_ROWS = 100_000
_MAX_PHYSICAL_COLUMNS = 256


def list_sheets_from_bytes(
    source_bytes: bytes,
    *,
    source_label: str,
) -> List[str]:
    """Return worksheet names from the one immutable import snapshot."""

    workbook = _open_workbook_bytes(
        source_bytes,
        source_label=source_label,
        data_only=True,
    )
    try:
        sheet_names = list(workbook.sheetnames)
    finally:
        workbook.close()
    if not sheet_names:
        raise ValueError(f"Excel workbook {source_label!r} has no sheets.")
    return sheet_names


def read_excel_sheet_rows_from_bytes(
    source_bytes: bytes,
    sheet_name: str,
    *,
    source_label: str,
) -> Iterable[Mapping[PhysicalColumn, object]]:
    """Yield rows from the exact workbook bytes retained by an import worker."""
    owned_bytes = bytes(source_bytes)
    label = str(source_label).strip() or "<workbook>"

    def _iter_rows() -> Iterator[Mapping[PhysicalColumn, str]]:
        workbook = _open_workbook_bytes(
            owned_bytes,
            source_label=label,
            data_only=True,
        )
        authored_workbook = _open_workbook_bytes(
            owned_bytes,
            source_label=label,
            data_only=False,
        )
        try:
            if sheet_name not in workbook.sheetnames:
                raise ValueError(
                    f"Sheet {sheet_name!r} not found in workbook {label!r}."
                )
            worksheet = workbook[sheet_name]
            if sheet_name not in authored_workbook.sheetnames:
                raise ValueError(
                    f"Sheet {sheet_name!r} not found in workbook {label!r}."
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
            authored_workbook.close()

    return _iter_rows()


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
    declared_rows = max(
        int(getattr(worksheet, "max_row", 0) or 0),
        int(getattr(authored_worksheet, "max_row", 0) or 0),
    )
    if declared_rows <= 0:
        raise ValueError(f"Sheet {sheet_name!r} in workbook {filepath!r} has no rows.")
    if declared_rows - 1 > _MAX_PHYSICAL_ROWS:
        raise ValueError(
            f"Sheet {sheet_name!r} in workbook {filepath!r} exceeds the "
            "100000 physical-row supported-size limit."
        )
    physical_width = max(
        _exact_worksheet_width(worksheet),
        _exact_worksheet_width(authored_worksheet),
    )
    if physical_width <= 0:
        raise ValueError(f"Sheet {sheet_name!r} in workbook {filepath!r} has no rows.")
    if physical_width > _MAX_PHYSICAL_COLUMNS:
        raise ValueError(
            f"Sheet {sheet_name!r} in workbook {filepath!r} exceeds the "
            "256 physical-column supported-size limit."
        )
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
                getattr(authored_cell, "data_type", None) == "f"
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
    _validate_xlsx_zip(source_bytes, source_label=source_label)
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
            members = archive.infolist()
    except (BadZipFile, OSError, ValueError) as exc:
        raise ValueError(
            f"Failed to inspect Excel workbook {source_label!r}: {exc}"
        ) from exc
    if len(members) > _MAX_XLSX_ZIP_MEMBERS:
        raise ValueError(
            f"Excel workbook {source_label!r} exceeds the 512 ZIP-member "
            "supported-size limit."
        )
    total_uncompressed = 0
    for member in members:
        member_size = int(member.file_size)
        if member_size > _MAX_XLSX_MEMBER_BYTES:
            raise ValueError(
                f"Excel workbook {source_label!r} contains a ZIP member "
                "larger than the 128 MiB supported-size limit."
            )
        total_uncompressed += member_size
        if total_uncompressed > _MAX_XLSX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"Excel workbook {source_label!r} exceeds the 256 MiB "
                "aggregate uncompressed-size limit."
            )


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
