"""Export-dialog start folders, local destination identity, destination-atomic
file writing, and file error text."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import uuid

__all__ = [
    "create_sibling_temporary",
    "file_error_text",
    "replace_destination",
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


def create_sibling_temporary(
    destination: str | os.PathLike[str],
    suffix: str = ".tmp",
) -> tuple[int, str]:
    """Create a new hidden file beside ``destination``; return its descriptor
    and path.

    The umask decides its permissions. The caller owns the descriptor and the
    file.
    """

    directory, name = os.path.split(os.fspath(destination))
    path = os.path.join(directory, f".{name}.{uuid.uuid4().hex}{suffix}")
    # Callers wrap the descriptor with os.fdopen, whose file object sets
    # binary mode on Windows, so no O_BINARY flag is needed here.
    return os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o666), path


def replace_destination(
    temporary: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> None:
    """Replace ``destination`` with the written ``temporary`` file.

    An existing destination keeps its permission bits. On Windows they would
    only carry the read-only flag, which would leave the temporary file
    undeletable when the replacement fails.
    """

    if sys.platform != "win32":
        try:
            mode = os.stat(destination).st_mode
        except FileNotFoundError:
            pass
        else:
            os.chmod(temporary, stat.S_IMODE(mode))
    os.replace(temporary, destination)


def file_error_text(
    exc: BaseException,
    path: str | os.PathLike[str] | None,
) -> str:
    """Display text for a caught file error.

    An operating-system refusal reads as its reason, never its errno, class or
    internal temporary file name, and names ``path``. A caller passes ``None``
    only when its surrounding text already names the file, or the file is an
    internal one. Kindred's own errors already carry their sentence.
    """

    if not isinstance(exc, OSError) or not exc.strerror:
        return str(exc)
    if path is None:
        return f"{exc.strerror}."
    return f"{exc.strerror}: {os.fspath(path)}."
