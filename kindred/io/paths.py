"""Export-dialog start folders and local destination identity."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "resolve_export_start_dir",
    "same_local_path",
]


def resolve_export_start_dir(last_folder: str | None) -> str:
    """Start an export dialog in the remembered folder, else in Documents."""

    if last_folder:
        remembered = Path(last_folder).expanduser()
        if remembered.is_dir():
            return str(remembered.resolve())
    # Imported lazily so kindred.io stays importable without Qt.
    from PySide6.QtCore import QStandardPaths

    documents = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.DocumentsLocation
    )
    if documents and Path(documents).is_dir():
        return str(documents)
    return str(Path.home())


def _canonical_local_path(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(
        os.path.realpath(os.path.abspath(os.fsdecode(os.fspath(path))))
    )


def same_local_path(
    first: str | os.PathLike[str],
    second: str | os.PathLike[str] | None,
) -> bool:
    """Whether two local paths name the same file after resolution."""

    return second is not None and (
        _canonical_local_path(first) == _canonical_local_path(second)
    )
