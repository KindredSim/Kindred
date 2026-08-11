"""Canonical source-byte admission and physical-column allocation rules.

This is the lowest dataset import policy module.  Parsers consume its exact
source admission and column identities, while the immutable dataset
foundation reuses the same rules when it validates a retained record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "PhysicalColumn",
    "identify_physical_columns",
    "validate_tabular_source_bytes",
]

_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def validate_tabular_source_bytes(
    source_bytes: bytes,
    *,
    source_format: str,
    source_label: str,
) -> None:
    """Admit exact nonempty bytes for one current CSV or XLSX source."""

    if type(source_bytes) is not bytes:
        raise TypeError("Dataset source content must be exact bytes.")
    label = str(source_label).strip() or "<dataset>"
    if not source_bytes:
        raise ValueError(f"Dataset source {label!r} is empty.")
    signature = source_bytes[:8]
    if source_format == "xlsx":
        if not signature.startswith(_ZIP_SIGNATURES):
            raise ValueError(
                f"File {label!r} has an .xlsx extension but is not an XLSX "
                "workbook."
            )
        return
    if source_format != "csv":
        raise ValueError(f"Unsupported exact dataset source format {source_format!r}.")
    if signature.startswith(_ZIP_SIGNATURES) or signature.startswith(
        _OLE_SIGNATURE
    ):
        raise ValueError(
            f"File {label!r} has a .csv extension but contains a workbook or "
            "other binary container."
        )
    if b"\x00" in source_bytes:
        raise ValueError(
            f"File {label!r} is binary data, not a UTF-8 CSV source."
        )
    try:
        source_bytes.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"File {label!r} is not a valid UTF-8 CSV source."
        ) from exc


@dataclass(frozen=True, slots=True)
class PhysicalColumn:
    """One source-position identity with its stable assigned observation name."""

    source_index: int
    authored_name: str
    assigned_name: str

    def __post_init__(self) -> None:
        if type(self.source_index) is not int or self.source_index < 0:
            raise ValueError("Physical column source_index must be nonnegative.")
        authored = str(self.authored_name).strip()
        assigned = str(self.assigned_name).strip()
        if not authored or not assigned:
            raise ValueError("Physical column names must be nonblank.")
        object.__setattr__(self, "authored_name", authored)
        object.__setattr__(self, "assigned_name", assigned)


def identify_physical_columns(
    headers: Sequence[object],
) -> tuple[PhysicalColumn, ...]:
    """Allocate exact-case collision-safe names by physical source position."""

    authored = tuple(
        "" if value is None else str(value).strip()
        for value in headers
    )
    # Blank physical positions have no import identity.  Keep their original
    # positions out of the retained catalog; callers still use each retained
    # column's source_index to read the physical row.
    named = tuple(
        (source_index, name)
        for source_index, name in enumerate(authored)
        if name
    )
    if not named:
        return ()
    reserved = {name for _source_index, name in named}
    assigned: set[str] = set()
    occurrences: dict[str, int] = {}
    columns: list[PhysicalColumn] = []
    for source_index, name in named:
        occurrence = occurrences.get(name, 0)
        if occurrence == 0 and name not in assigned:
            candidate = name
        else:
            suffix = occurrence
            while True:
                candidate = f"{name}_copy_{suffix}"
                if candidate not in reserved and candidate not in assigned:
                    break
                suffix += 1
            occurrences[name] = suffix
        occurrences[name] = occurrences.get(name, 0) + 1
        assigned.add(candidate)
        columns.append(
            PhysicalColumn(
                source_index=source_index,
                authored_name=name,
                assigned_name=candidate,
            )
        )
    return tuple(columns)
