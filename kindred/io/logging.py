"""Canonical logging setup for Kindred."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import tempfile
import threading
from pathlib import Path

__all__ = ["diagnostic_log_path", "setup_logging"]

_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_LOG_FILENAME = "kindred.log"
_MAX_LOG_SIZE = 10 * 1024 * 1024
_BACKUP_COUNT = 5
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

    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            handler.flush()
            return Path(handler.baseFilename)
    raise RuntimeError("The diagnostic log is not available.")


def setup_logging() -> None:
    """Configure bounded rotating-file logging and usable terminal diagnostics.

    Calling this function replaces and closes all existing root handlers. The
    platform log file is always attempted first, followed by a system
    temporary-directory file when the normal location cannot be opened.
    """

    resolved_level = _get_log_level_from_env()
    numeric_level = getattr(logging, resolved_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)
        existing_handler.close()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_stream = _usable_stderr()
    if console_stream is not None:
        console_handler = logging.StreamHandler(console_stream)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    normal_log = _get_normal_log_dir() / _DEFAULT_LOG_FILENAME
    fallback_log = _get_fallback_log_dir() / _DEFAULT_LOG_FILENAME
    log_paths = tuple(dict.fromkeys((normal_log, fallback_log)))
    file_error: OSError | None = None
    log_path: Path | None = None
    for candidate in log_paths:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            rotating_handler = logging.handlers.RotatingFileHandler(
                candidate,
                maxBytes=_MAX_LOG_SIZE,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            rotating_handler.setLevel(numeric_level)
            rotating_handler.setFormatter(formatter)
            root_logger.addHandler(rotating_handler)
            log_path = candidate
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
    root_logger.debug("Max log size: %.1f MB", _MAX_LOG_SIZE / (1024 * 1024))
    root_logger.debug("Backup count: %s", _BACKUP_COUNT)


def _log_unhandled(exc_type, exc, traceback) -> None:
    logging.getLogger("kindred").critical("Unhandled error", exc_info=(exc_type, exc, traceback))
