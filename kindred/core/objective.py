from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Optional

__all__ = ["ObjectiveContext"]


class ObjectiveContext:
    """Per-call objective context stored in ContextVars for thread safety."""

    def __init__(self) -> None:
        self._error: ContextVar[Optional[Any]] = ContextVar("objective_error", default=None)

    def clear(self) -> None:
        self._error.set(None)

    def set_error(self, err: Any) -> None:
        self._error.set(err)

    @property
    def last_error(self) -> Any:
        return self._error.get()
