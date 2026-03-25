"""
Basic logging configuration helpers for Kindred.

INTERNAL MODULE
---------------
This module provides low-level logging setup utilities and is intended for
internal use by kindred.io.logging. External code should NOT import from this
module directly.

**Public API**: Use `kindred.io.logging` or `kindred.io` instead.

Examples
--------
❌ Do NOT use::

    from kindred.io.logging_config import setup_logging  # INTERNAL

✅ Use instead::

    from kindred.io.logging import setup_logging  # PUBLIC API
    from kindred.io import setup_logging          # Also OK

This module provides:
- Rotating file handler with size-based rotation
- Console handler with stderr output
- Environment variable override (LOG_LEVEL, KINDRED_LOG_LEVEL)
- Thread-safe configuration
- Platform-specific log directory resolution
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

__all__ = ["setup_logging", "get_log_dir", "DEFAULT_LOG_LEVEL"]

# Default log level
DEFAULT_LOG_LEVEL = "INFO"

# Log file configuration
DEFAULT_LOG_FILENAME = "kindred.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5  # Keep 5 backup files


def get_log_dir() -> Path:
    """
    Get the directory for log files.

    Returns
    -------
    Path
        Log directory path (creates if doesn't exist)

    Notes
    -----
    Log directory priority:
    1. KINDRED_LOG_DIR environment variable
    2. XDG_CACHE_HOME/kindred/logs (Linux)
    3. ~/Library/Logs/kindred (macOS)
    4. %LOCALAPPDATA%/kindred/logs (Windows)
    5. ~/.cache/kindred/logs (fallback)
    """
    # Check environment variable first
    if "KINDRED_LOG_DIR" in os.environ:
        log_dir = Path(os.environ["KINDRED_LOG_DIR"]).expanduser()
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            return log_dir
        except (OSError, PermissionError):
            # Fall back to platform defaults below
            pass

    # Platform-specific defaults
    if sys.platform == "darwin":
        # macOS
        log_dir = Path.home() / "Library" / "Logs" / "kindred"
    elif sys.platform == "win32":
        # Windows
        local_app_data = os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        log_dir = Path(local_app_data) / "kindred" / "logs"
    else:
        # Linux/Unix
        xdg_cache = os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        log_dir = Path(xdg_cache) / "kindred" / "logs"

    # Create directory if it doesn't exist (best-effort; do not crash if unwritable)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    except (OSError, PermissionError):
        pass

    # Last-resort fallback: temp directory (often writable even when home/appdata is not)
    try:
        fallback = Path(tempfile.gettempdir()) / "kindred" / "logs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    except Exception:
        return log_dir


def get_log_level_from_env() -> str:
    """
    Get log level from environment variables.

    Returns
    -------
    str
        Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Notes
    -----
    Checks (in priority order):
    1. LOG_LEVEL environment variable
    2. KINDRED_LOG_LEVEL environment variable
    3. Returns DEFAULT_LOG_LEVEL if not set
    """
    # Check generic LOG_LEVEL first
    if "LOG_LEVEL" in os.environ:
        level = os.environ["LOG_LEVEL"].upper()
        if level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            return level

    # Check app-specific override
    if "KINDRED_LOG_LEVEL" in os.environ:
        level = os.environ["KINDRED_LOG_LEVEL"].upper()
        if level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            return level

    return DEFAULT_LOG_LEVEL


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    file_handler: bool = True,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for Kindred application.

    Parameters
    ----------
    level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        If None, uses environment variable or DEFAULT_LOG_LEVEL.
    log_file : str, optional
        Log file path. If None, uses default location.
    console : bool, default=True
        Enable console (stderr) logging
    file_handler : bool, default=True
        Enable rotating file handler
    format_string : str, optional
        Custom log format string. If None, uses default.

    Notes
    -----
    - File handler uses RotatingFileHandler with 10MB max size, 5 backups
    - Console handler outputs to stderr for better stream separation
    - Thread-safe: Can be called multiple times (will reconfigure)
    - Environment variables override defaults:
        - LOG_LEVEL or KINDRED_LOG_LEVEL: Sets log level
        - KINDRED_LOG_DIR: Sets log directory

    Examples
    --------
    >>> setup_logging()  # Use defaults
    >>> setup_logging(level="DEBUG")  # Debug mode
    >>> setup_logging(level="WARNING", file_handler=False)  # Console only
    >>> setup_logging(log_file="/tmp/custom.log")  # Custom log file
    """
    # Determine log level
    if level is None:
        level = get_log_level_from_env()

    level = level.upper()
    numeric_level = getattr(logging, level, logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Define format
    if format_string is None:
        # Detailed format with timestamp, level, module, and message
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler (stderr)
    if console:
        # Bind to the original stderr object so later sys.stderr wrappers
        # (including wrappers that log) cannot recurse through logging.
        console_stream = sys.__stderr__ if sys.__stderr__ is not None else sys.stderr
        console_handler = logging.StreamHandler(console_stream)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Rotating file handler
    if file_handler:
        if log_file is None:
            log_dir = get_log_dir()
            log_file = str(log_dir / DEFAULT_LOG_FILENAME)

        try:
            # Create rotating file handler
            file_handler_obj = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=MAX_LOG_SIZE,
                backupCount=BACKUP_COUNT,
                encoding="utf-8"
            )
            file_handler_obj.setLevel(numeric_level)
            file_handler_obj.setFormatter(formatter)
            root_logger.addHandler(file_handler_obj)

            # Log initialization message
            root_logger.info(f"Logging configured: level={level}, file={log_file}")

        except (OSError, PermissionError) as e:
            # Fallback to console only if file handler fails
            if console:
                root_logger.warning(f"Failed to create log file {log_file}: {e}")
                root_logger.warning("Continuing with console logging only")
            else:
                # No handlers available
                raise RuntimeError(f"Failed to set up logging: {e}") from e

    # Reduce noise from verbose libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Log startup information
    root_logger.debug(f"Log level: {level}")
    root_logger.debug(f"Console handler: {console}")
    root_logger.debug(f"File handler: {file_handler}")
    if file_handler and log_file:
        root_logger.debug(f"Log file: {log_file}")
        root_logger.debug(f"Max log size: {MAX_LOG_SIZE / (1024*1024):.1f} MB")
        root_logger.debug(f"Backup count: {BACKUP_COUNT}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)

    Returns
    -------
    logging.Logger
        Logger instance

    Notes
    -----
    Use this instead of logging.getLogger() to ensure consistent configuration.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Message")
    """
    return logging.getLogger(name)
