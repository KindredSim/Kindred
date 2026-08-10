"""Canonical logging setup for Kindred."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import tempfile
from pathlib import Path

__all__ = ["setup_logging"]

_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_LOG_FILENAME = "kindred.log"
_MAX_LOG_SIZE = 10 * 1024 * 1024
_BACKUP_COUNT = 5
_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _get_log_dir() -> Path:
    """Create and return the first available platform-appropriate log directory."""

    configured_dir = os.environ.get("KINDRED_LOG_DIR")
    if configured_dir:
        log_dir = Path(configured_dir).expanduser()
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            return log_dir
        except OSError:
            pass

    if sys.platform == "darwin":
        log_dir = Path.home() / "Library" / "Logs" / "kindred"
    elif sys.platform == "win32":
        local_app_data = os.getenv(
            "LOCALAPPDATA",
            str(Path.home() / "AppData" / "Local"),
        )
        log_dir = Path(local_app_data) / "kindred" / "logs"
    else:
        xdg_cache = os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        log_dir = Path(xdg_cache) / "kindred" / "logs"

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "kindred" / "logs"
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
        except OSError:
            return log_dir


def _get_log_level_from_env() -> str:
    """Resolve a valid configured log level or use the default."""

    for variable in ("LOG_LEVEL", "KINDRED_LOG_LEVEL"):
        configured_level = os.environ.get(variable, "").upper()
        if configured_level in _VALID_LOG_LEVELS:
            return configured_level
    return _DEFAULT_LOG_LEVEL


def setup_logging(
    level: str | None = None,
    log_file: str | None = None,
    console: bool = True,
    file_handler: bool = True,
    format_string: str | None = None,
) -> None:
    """Configure console and bounded rotating-file logging for Kindred.

    Calling this function replaces and closes all existing root handlers. If
    the requested file cannot be opened, startup continues with the console
    handler when one was requested; otherwise setup fails explicitly.
    """

    resolved_level = (level or _get_log_level_from_env()).upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)
        existing_handler.close()

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    if console:
        console_stream = sys.__stderr__ if sys.__stderr__ is not None else sys.stderr
        console_handler = logging.StreamHandler(console_stream)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if file_handler:
        if log_file is None:
            log_file = str(_get_log_dir() / _DEFAULT_LOG_FILENAME)

        try:
            rotating_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=_MAX_LOG_SIZE,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            rotating_handler.setLevel(numeric_level)
            rotating_handler.setFormatter(formatter)
            root_logger.addHandler(rotating_handler)
            root_logger.info(
                "Logging configured: level=%s, file=%s",
                resolved_level,
                log_file,
            )
        except OSError as error:
            if not console:
                raise RuntimeError(f"Failed to set up logging: {error}") from error
            root_logger.warning("Failed to create log file %s: %s", log_file, error)
            root_logger.warning("Continuing with console logging only")

    logging.getLogger("PIL").setLevel(logging.WARNING)

    root_logger.debug("Log level: %s", resolved_level)
    root_logger.debug("Console handler: %s", console)
    root_logger.debug("File handler: %s", file_handler)
    if file_handler and log_file:
        root_logger.debug("Log file: %s", log_file)
        root_logger.debug("Max log size: %.1f MB", _MAX_LOG_SIZE / (1024 * 1024))
        root_logger.debug("Backup count: %s", _BACKUP_COUNT)
