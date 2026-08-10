"""Canonical basic logging helpers for Kindred internals."""

from __future__ import annotations

import logging
from typing import Optional

from kindred.io.logging_config import (
    get_log_dir,
    get_logger as _get_basic_logger,
    setup_logging as _setup_basic_logging,
)

__all__ = ["get_logger", "setup_logging", "get_log_dir"]


def get_logger(name: str) -> logging.Logger:
    """Return a standard-library logger owned by the basic logging setup."""

    return _get_basic_logger(name)


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    file_handler: bool = True,
    format_string: Optional[str] = None,
) -> None:
    """Configure Kindred's console and bounded rotating-file logging."""

    _setup_basic_logging(
        level=level,
        log_file=log_file,
        console=console,
        file_handler=file_handler,
        format_string=format_string,
    )
