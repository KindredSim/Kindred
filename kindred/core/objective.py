from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Optional

__all__ = ["ObjectiveContext"]


class ObjectiveContext:
    """Per-call objective context stored in ContextVars for thread safety."""

    def __init__(self) -> None:
        self._error: ContextVar[Optional[Any]] = ContextVar("objective_error", default=None)
        self._prov: ContextVar[Optional[Any]] = ContextVar("objective_error_provenance", default=None)

    def clear(self) -> None:
        self._error.set(None)
        self._prov.set(None)

    def set_error(self, err: Any, provenance: Any = None) -> None:
        self._error.set(err)
        self._prov.set(provenance)

    @property
    def last_error(self) -> Any:
        return self._error.get()

    @property
    def last_error_provenance(self) -> Any:
        return self._prov.get()
