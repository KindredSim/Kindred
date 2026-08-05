"""
Canonical logging helpers for Kindred internals.

This module is the logging owner used by the GUI entrypoint and internal
modules. `kindred.io.logging_config` remains a lower-level implementation
detail.

Features
--------
- **Basic logging**: Simple setup with rotating file handlers
- **Structured logging**: JSON-formatted logs for production/analysis

Usage Examples
--------------
Basic logging::

    from kindred.io.logging import setup_logging, get_logger

    setup_logging(level="INFO")
    logger = get_logger(__name__)
    logger.info("Starting simulation")
    logger.debug("Parameter values", extra={"params": {"k": 1.5}})

Structured logging (JSON)::

    from kindred.io.logging import setup_structured_logging, get_logger

    setup_structured_logging(level="INFO")
    logger = get_logger(__name__)
    logger.info("Simulation complete", extra={
        "duration": 2.5,
        "n_species": 10,
        "solver": "BDF"
    })

"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional

from kindred.io.logging_config import (
    BACKUP_COUNT,
    MAX_LOG_SIZE,
    setup_logging as _setup_basic_logging,
    get_log_dir,
    get_logger as _get_basic_logger,
    DEFAULT_LOG_LEVEL,
)

__all__ = [
    # Basic logging
    "get_logger",
    "setup_logging",
    "get_log_dir",
    # Structured logging
    "setup_structured_logging",
    "JSONFormatter",
    "StructuredMessage",
]


# ----------------------------- Structured Logging ------------------------------


@dataclass
class StructuredMessage:
    """
    Structured log message with typed fields.

    Attributes
    ----------
    message : str
        Human-readable message
    timestamp : str
        ISO 8601 timestamp
    level : str
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger_name : str
        Logger name
    module : str
        Module name
    function : str
        Function name
    line : int
        Line number
    extra : dict, optional
        Additional structured data
    exception : dict, optional
        Exception information (type, message, traceback)
    """
    message: str
    timestamp: str
    level: str
    logger_name: str
    module: str
    function: str
    line: int
    extra: Optional[Dict[str, Any]] = None
    exception: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON with structured fields for:
    - Standard log fields (timestamp, level, message)
    - Module/function context
    - Custom extra data
    - Exception information

    Compatible with log aggregation tools (ELK, Splunk, CloudWatch).
    """

    def __init__(self, indent: Optional[int] = None):
        """
        Initialize JSON formatter.

        Parameters
        ----------
        indent : int, optional
            JSON indentation (None for compact, 2/4 for pretty)
        """
        super().__init__()
        self.indent = indent

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Build structured message
        message = StructuredMessage(
            message=record.getMessage(),
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=record.levelname,
            logger_name=record.name,
            module=record.module,
            function=record.funcName,
            line=record.lineno,
        )

        # Add extra fields (exclude standard record attributes)
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs", "message",
            "pathname", "process", "processName", "relativeCreated",
            "thread", "threadName", "exc_info", "exc_text", "stack_info",
        }
        extra = {
            k: v for k, v in record.__dict__.items()
            if k not in standard_attrs and not k.startswith("_")
        }
        if extra:
            message.extra = extra

        # Add exception info if present
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            message.exception = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Serialize to JSON
        return json.dumps(message.to_dict(), indent=self.indent)


def setup_structured_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    file_handler: bool = True,
    indent: Optional[int] = None,
) -> None:
    """
    Configure structured (JSON) logging.

    Parameters
    ----------
    level : str, optional
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        JSON log file path. If None, uses default location.
    console : bool
        Enable console output
    file_handler : bool
        Enable the bounded rotating JSON-lines file handler
    indent : int, optional
        JSON indentation (None=compact, 2/4=pretty)

    Examples
    --------
    >>> setup_structured_logging(level="INFO")
    >>> logger = get_logger(__name__)
    >>> logger.info("Event occurred", extra={"user_id": 123})
    """
    # Determine log level
    if level is None:
        level = DEFAULT_LOG_LEVEL
    level = level.upper()
    numeric_level = getattr(logging, level, logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Create JSON formatter
    formatter = JSONFormatter(indent=indent)

    # Console handler
    if console:
        # Use the original stderr stream to avoid recursion when sys.stderr
        # is replaced by wrappers that feed messages back into logging.
        console_stream = sys.__stderr__ if sys.__stderr__ is not None else sys.stderr
        console_handler = logging.StreamHandler(console_stream)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if file_handler:
        if log_file is None:
            log_file = str(get_log_dir() / "kindred_structured.jsonl")
        try:
            file_handler_obj = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=MAX_LOG_SIZE,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler_obj.setLevel(numeric_level)
            file_handler_obj.setFormatter(formatter)
            root_logger.addHandler(file_handler_obj)
        except (OSError, PermissionError) as exc:
            if console:
                root_logger.warning("Failed to create structured log file: %s", exc)
            else:
                raise RuntimeError(
                    f"Failed to set up structured logging: {exc}"
                ) from exc


# ------------------------------- helpers ---------------------------------------


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

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Message")
    >>> logger.debug("Debug info", extra={"data": value})
    """
    return _get_basic_logger(name)


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    file_handler: bool = True,
    structured: bool = False,
    **kwargs,
) -> None:
    """
    Configure logging for Kindred.

    Parameters
    ----------
    level : str, optional
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        Log file path
    console : bool
        Enable console output
    file_handler : bool
        Enable file handler
    structured : bool
        Use JSON structured logging (default: False)
    **kwargs : dict
        Additional arguments passed to underlying setup functions

    Examples
    --------
    >>> setup_logging()  # Basic logging
    >>> setup_logging(level="DEBUG")  # Debug mode
    >>> setup_logging(structured=True)  # JSON logging
    >>> setup_logging(level="INFO", structured=True, indent=2)  # Pretty JSON
    """
    if structured:
        setup_structured_logging(
            level=level,
            log_file=log_file,
            console=console,
            file_handler=file_handler,
            **kwargs
        )
    else:
        _setup_basic_logging(
            level=level,
            log_file=log_file,
            console=console,
            file_handler=file_handler,
            **kwargs
        )
