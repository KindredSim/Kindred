# kindred/io/paths.py

"""
Outputs directory management (no placeholders).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, TypeGuard

logger = logging.getLogger(__name__)

__all__ = ["find_outputs_dir", "resolve_start_dir"]

_APP_NAME = "Kindred"


def _is_frozen() -> bool:
    """Return True when running in a frozen (PyInstaller/Nuitka) build."""
    return bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()


def _dir_from_env(env_var: str) -> Optional[Path]:
    raw = os.environ.get(env_var)
    if not raw:
        return None
    try:
        return Path(str(raw)).expanduser().resolve()
    except Exception:
        return None


def _qt_app_data_dir() -> Optional[Path]:
    """
    Best-effort resolve of a user-writable application data directory via Qt.

    Kept optional so tests and other non-GUI code paths do not have to import Qt.
    """
    try:
        from PySide6.QtCore import QStandardPaths  # type: ignore

        location = getattr(QStandardPaths, "AppDataLocation", None)
        if location is None:
            location = QStandardPaths.StandardLocation.AppDataLocation  # type: ignore[attr-defined]

        base = QStandardPaths.writableLocation(location)
        if base:
            return Path(str(base)).expanduser().resolve()
    except Exception:
        return None
    return None


def _default_user_data_dir() -> Path:
    """
    Return a sensible, user-writable base directory for Kindred data.

    - Windows: %LOCALAPPDATA%\\Kindred (fallback: %APPDATA%\\Kindred)
    - macOS: ~/Library/Application Support/Kindred
    - Linux: $XDG_DATA_HOME/kindred (fallback: ~/.local/share/kindred)
    """
    qt_dir = _qt_app_data_dir()
    if qt_dir is not None:
        return qt_dir

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return (Path(base) / _APP_NAME).expanduser()
        return Path.home() / "AppData" / "Local" / _APP_NAME

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP_NAME

    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / _APP_NAME.lower()
    return Path.home() / ".local" / "share" / _APP_NAME.lower()


def _app_base_dir() -> Path:
    """
    Resolve the app/repo root deterministically without relying on cwd.

    Priority:
      1) KINDRED_BASE_DIR for frozen/Windows builds (user-writable default)
      2) repository root inferred from this file location (two levels up from 'kindred')
    """
    if _is_frozen() or sys.platform == "win32":
        # In frozen builds (and on Windows generally), avoid writing beside the executable.
        # Default to a user-writable location unless explicitly overridden.
        override = _dir_from_env("KINDRED_BASE_DIR")
        return override or _default_user_data_dir()

    # Repo root: <repo_root> / kindred / io / paths.py  => parents[2] = <repo_root>
    here = Path(__file__).resolve()
    # parents[0]=.../io, [1]=.../kindred, [2]=<repo_root>
    try:
        return here.parents[2]
    except Exception as exc:
        # Fallback to the package root if unusual layout
        logger.warning(f"Failed to access parents[2] for base directory, using fallback: {exc}")
        return here.parents[1]


def find_outputs_dir() -> str:
    """Return default outputs directory path (user-writable when frozen/Windows)."""
    override = _dir_from_env("KINDRED_OUTPUT_DIR")
    if override is not None:
        return str(override)
    base = _app_base_dir()
    return str((base / "outputs").resolve())

def _is_dirlike(path: Optional[str]) -> TypeGuard[str]:
    return isinstance(path, str) and bool(path.strip())


def resolve_start_dir(last_folder: Optional[str]) -> str:
    """Pick a start directory for file dialogs.

    Priority:
      1) explicit `last_folder` arg if provided
      2) outputs/ at app/repo root
    """
    if _is_dirlike(last_folder):
        return str(Path(last_folder).expanduser().resolve())

    # Do not create directories during startup/browse; provide a deterministic default.
    return find_outputs_dir()
