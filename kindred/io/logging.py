"""Canonical logging setup for Kindred."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

if sys.platform != "win32":
    import fcntl

__all__ = ["diagnostic_log_path", "setup_logging"]

_DEFAULT_LOG_LEVEL = "INFO"
# Each Kindred writes its own log, so several open at once never share a file.
_LOG_PATTERN = "kindred-*.log"
_KEPT_LOGS = 10
_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _get_normal_log_dir() -> Path:
    """Return the platform log directory, honoring the supported override."""

    configured_dir = os.environ.get("KINDRED_LOG_DIR")
    if configured_dir:
        return Path(configured_dir).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "kindred"
    if sys.platform == "win32":
        local_app_data = os.getenv(
            "LOCALAPPDATA",
            str(Path.home() / "AppData" / "Local"),
        )
        return Path(local_app_data) / "kindred" / "logs"
    xdg_cache = os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(xdg_cache) / "kindred" / "logs"


def _get_fallback_log_dir() -> Path:
    """Return the system temporary-directory log location."""

    return Path(tempfile.gettempdir()) / "kindred" / "logs"


def _usable_stderr() -> object | None:
    """Return a usable stderr stream, if this process has one."""

    for stream in (getattr(sys, "stderr", None), getattr(sys, "__stderr__", None)):
        if stream is None or getattr(stream, "closed", False):
            continue
        if callable(getattr(stream, "write", None)) and callable(
            getattr(stream, "flush", None)
        ):
            return stream
    return None


def _get_log_level_from_env() -> str:
    """Resolve a valid configured log level or use the default."""

    for variable in ("LOG_LEVEL", "KINDRED_LOG_LEVEL"):
        configured_level = os.environ.get(variable, "").upper()
        if configured_level in _VALID_LOG_LEVELS:
            return configured_level
    return _DEFAULT_LOG_LEVEL


def diagnostic_log_path() -> Path:
    """Return the active diagnostic log path."""

    # setup_logging installs the file handler or startup stops.
    handler = next(item for item in logging.getLogger().handlers if isinstance(item, logging.FileHandler))
    handler.flush()
    return Path(handler.baseFilename)


def _in_use(path: Path) -> bool:
    """Whether another open Kindred still writes this log. Windows cannot delete an open log at all, so
    only Linux and macOS check the lock its Kindred holds."""
    if sys.platform == "win32":
        return False
    with path.open("rb") as stream:
        try:
            # A shared lock needs only read access, also on NFS, and still meets the exclusive lock of
            # a running Kindred.
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def _remove_old_logs(directory: Path) -> None:
    """Delete the oldest logs beyond the newest _KEPT_LOGS, except those another open Kindred still writes."""
    # Each name starts with its start time, so the newest sort last.
    for path in sorted(directory.glob(_LOG_PATTERN))[:-_KEPT_LOGS]:
        try:
            if not _in_use(path):
                path.unlink()
        except OSError:
            # Windows refuses to delete a log another Kindred has open.
            continue


def setup_logging() -> None:
    """Configure a log file of this Kindred and usable terminal diagnostics.

    Calling this function replaces and closes all existing root handlers. The
    platform log directory is always attempted first, followed by a system
    temporary directory when the normal location cannot be opened. Each start
    writes a new file and deletes the oldest beyond the newest _KEPT_LOGS.
    """

    resolved_level = _get_log_level_from_env()
    numeric_level = getattr(logging, resolved_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)
        existing_handler.close()

    formatter = logging.Formatter(
        "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_stream = _usable_stderr()
    if console_stream is not None:
        console_handler = logging.StreamHandler(console_stream)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    log_name = f"kindred-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"
    normal_log = _get_normal_log_dir() / log_name
    fallback_log = _get_fallback_log_dir() / log_name
    log_paths = tuple(dict.fromkeys((normal_log, fallback_log)))
    file_error: OSError | None = None
    log_path: Path | None = None
    for candidate in log_paths:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(candidate, encoding="utf-8")
            if sys.platform != "win32":
                # Held while the file is open, so another Kindred leaves it in place.
                fcntl.flock(file_handler.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            log_path = candidate
            _remove_old_logs(candidate.parent)
            root_logger.info(
                "Logging configured: level=%s, file=%s",
                resolved_level,
                candidate,
            )
            if candidate != normal_log:
                root_logger.warning(
                    "Normal log location %s is unavailable; using temporary log %s",
                    normal_log,
                    candidate,
                )
            break
        except OSError as error:
            file_error = error

    if log_path is None:
        message = f"Failed to set up normal and temporary log files: {file_error}"
        raise RuntimeError(message) from file_error

    # Errors that escape a Qt slot or a thread would otherwise vanish when
    # Kindred runs without a console; record them in the log.
    sys.excepthook = _log_unhandled
    threading.excepthook = lambda args: _log_unhandled(args.exc_type, args.exc_value, args.exc_traceback)

    root_logger.debug("Log level: %s", resolved_level)
    root_logger.debug("Console handler: %s", console_stream is not None)
    root_logger.debug("Log file: %s", log_path)


def _log_unhandled(exc_type, exc, traceback) -> None:
    logging.getLogger("kindred").critical("Unhandled error", exc_info=(exc_type, exc, traceback))
