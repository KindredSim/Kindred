"""
Canonical logging API for Kindred.

PUBLIC API
----------
This is the official logging module for Kindred. All external code should import
logging functionality from this module (or from `kindred.io` which re-exports it).

**DO NOT** import from `kindred.io.logging_config` directly - that module is for
internal use only.

Features
--------
- **Basic logging**: Simple setup with rotating file handlers
- **Structured logging**: JSON-formatted logs for production/analysis
- **Operation tracking**: Automatic timing and success/failure logging
- **Per-module levels**: Fine-grained log level control
- **Lazy evaluation**: Performance-optimized debug logging

Usage Examples
--------------
Basic logging::

    from kindred.io.logging import setup_logging, get_logger
    # or: from kindred.io import setup_logging, get_logger

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
        "solver": "LSODA"
    })

Operation tracking::

    from kindred.io.logging import log_operation, get_logger

    logger = get_logger(__name__)
    with log_operation("Parameter fitting", logger, level="INFO"):
        # Fitting code here
        pass
    # Automatically logs start, duration, and success/failure

Per-module levels::

    from kindred.io.logging import configure_module_levels

    configure_module_levels({
        "kindred.core": "DEBUG",
        "kindred.gui": "WARNING",
    })
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional, Iterator

from kindred.io.logging_config import (
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
    # Configuration
    "configure_module_levels",
    "set_log_level",
    # Operation tracking
    "log_operation",
    "OperationTimer",
    # Performance
    "LazyMessage",
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

    # File handler
    if log_file or console:
        if log_file is None and not console:
            log_dir = get_log_dir()
            log_file = str(log_dir / "kindred_structured.jsonl")

        if log_file:
            try:
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setLevel(numeric_level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            except (OSError, PermissionError) as e:
                if console:
                    root_logger.warning(f"Failed to create log file: {e}")
                else:
                    raise


# ------------------------------- Configuration ---------------------------------


def configure_module_levels(module_levels: Dict[str, str]) -> None:
    """
    Configure per-module log levels.

    Parameters
    ----------
    module_levels : dict
        Mapping of module names to log levels
        Example: {"kindred.core": "DEBUG", "kindred.gui": "WARNING"}

    Examples
    --------
    >>> configure_module_levels({
    ...     "kindred.core": "DEBUG",
    ...     "kindred.core.simulator": "INFO",
    ...     "kindred.gui": "WARNING",
    ... })
    """
    for module_name, level_str in module_levels.items():
        level = getattr(logging, level_str.upper(), logging.INFO)
        logger = logging.getLogger(module_name)
        logger.setLevel(level)


def set_log_level(level: str, module: Optional[str] = None) -> None:
    """
    Set log level for a specific module or root logger.

    Parameters
    ----------
    level : str
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    module : str, optional
        Module name. If None, sets root logger level.

    Examples
    --------
    >>> set_log_level("DEBUG")  # All modules
    >>> set_log_level("WARNING", "kindred.gui")  # Specific module
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if module:
        logger = logging.getLogger(module)
    else:
        logger = logging.getLogger()
    logger.setLevel(numeric_level)


# ----------------------------- Operation Tracking ------------------------------


@dataclass
class OperationTimer:
    """
    Timer for tracking operation duration.

    Attributes
    ----------
    name : str
        Operation name
    start_time : float
        Start timestamp
    end_time : float, optional
        End timestamp
    success : bool
        Whether operation succeeded
    error : Exception, optional
        Exception if operation failed
    metadata : dict
        Additional operation metadata
    """
    name: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = True
    error: Optional[Exception] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def stop(self, success: bool = True, error: Optional[Exception] = None) -> None:
        """
        Stop the timer.

        Parameters
        ----------
        success : bool
            Whether operation succeeded
        error : Exception, optional
            Exception if operation failed
        """
        self.end_time = time.time()
        self.success = success
        self.error = error

    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = {
            "operation": self.name,
            "duration_ms": int(self.duration * 1000) if self.duration else None,
            "success": self.success,
        }
        if self.error:
            data["error"] = str(self.error)
        if self.metadata:
            data.update(self.metadata)
        return data


@contextmanager
def log_operation(
    name: str,
    logger: logging.Logger,
    level: str = "INFO",
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[OperationTimer]:
    """
    Context manager for logging operation duration and success/failure.

    Parameters
    ----------
    name : str
        Operation name
    logger : logging.Logger
        Logger instance
    level : str
        Log level (INFO, DEBUG, etc.)
    metadata : dict, optional
        Additional metadata to include in logs

    Yields
    ------
    OperationTimer
        Timer object (can be used to add metadata during operation)

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> with log_operation("Parameter fitting", logger) as timer:
    ...     # Do fitting
    ...     timer.metadata["n_params"] = 5
    # Logs: "Parameter fitting started"
    # Logs: "Parameter fitting completed in 1.23s" (with metadata)

    >>> with log_operation("Simulation", logger):
    ...     raise ValueError("Solver failed")
    # Logs: "Simulation started"
    # Logs: "Simulation failed after 0.01s: Solver failed"
    """
    timer = OperationTimer(name=name, start_time=time.time(), metadata=metadata or {})

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, f"{name} started")

    try:
        yield timer
        timer.stop(success=True)

        # Log completion
        logger.log(
            log_level,
            f"{name} completed in {timer.duration:.3f}s",
            extra=timer.to_log_dict()
        )

    except Exception as e:
        timer.stop(success=False, error=e)

        # Log failure
        logger.log(
            logging.ERROR,
            f"{name} failed after {timer.duration:.3f}s: {e}",
            extra=timer.to_log_dict(),
            exc_info=True
        )
        raise


# ---------------------------- Performance Utilities ----------------------------


class LazyMessage:
    """
    Lazy log message evaluation for performance.

    Delays expensive string formatting until message is actually logged.
    Useful for DEBUG-level logs with complex formatting that won't be shown
    in production.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.debug(LazyMessage(lambda: f"Complex data: {expensive_operation()}"))
    # expensive_operation() only called if DEBUG level is enabled
    """

    def __init__(self, func):
        """
        Initialize lazy message.

        Parameters
        ----------
        func : callable
            Function returning message string
        """
        self.func = func

    def __str__(self):
        """Evaluate message when needed."""
        return str(self.func())


# ------------------------------- Public API ------------------------------------


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
