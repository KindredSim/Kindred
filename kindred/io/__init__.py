"""
I/O utilities for logging, resources, and paths.

This module re-exports the public logging API from kindred.io.logging for
convenient access. You can import logging functions from either:

    from kindred.io.logging import setup_logging, get_logger
    from kindred.io import setup_logging, get_logger  # Equivalent

Both forms are supported and equivalent.
"""

from kindred.io.logging import (
    # Basic logging
    get_logger,
    setup_logging,
    get_log_dir,
    # Structured logging
    setup_structured_logging,
    JSONFormatter,
    StructuredMessage,
    # Configuration
    configure_module_levels,
    set_log_level,
    # Operation tracking
    log_operation,
    OperationTimer,
    # Performance
    LazyMessage,
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
