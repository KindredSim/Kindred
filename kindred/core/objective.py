from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable, Optional

import numpy as np

__all__ = ["ObjectiveContext", "ObjectiveWrapper"]


class ObjectiveContext:
    """Per-call objective context stored in ContextVars for thread safety."""

    def __init__(self) -> None:
        self._error: ContextVar[Optional[Any]] = ContextVar("objective_error", default=None)
        self._prov: ContextVar[Optional[Any]] = ContextVar("objective_error_provenance", default=None)
        self._model: ContextVar[Optional[Any]] = ContextVar("objective_model", default=None)

    def clear(self) -> None:
        self._error.set(None)
        self._prov.set(None)
        self._model.set(None)

    def set_error(self, err: Any, provenance: Any = None) -> None:
        self._error.set(err)
        self._prov.set(provenance)

    def set_model(self, model: Any) -> None:
        self._model.set(model)

    @property
    def last_error(self) -> Any:
        return self._error.get()

    @property
    def last_error_provenance(self) -> Any:
        return self._prov.get()

    @property
    def last_model(self) -> Any:
        return self._model.get()


class ObjectiveWrapper:
    """Thread-safe wrapper exposing ContextVar-backed metadata."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], ctx: ObjectiveContext) -> None:
        self._fn = fn
        self._ctx = ctx
        self._last_error = None
        self._last_error_provenance = None
        self._last_model = None

    def __call__(self, params: np.ndarray) -> np.ndarray:
        self._ctx.clear()
        try:
            return self._fn(params)
        finally:
            self._last_error = self._ctx.last_error
            self._last_error_provenance = self._ctx.last_error_provenance
            self._last_model = self._ctx.last_model
            self._ctx.clear()

    @property
    def last_error(self):
        return self._last_error

    @property
    def last_error_provenance(self):
        return self._last_error_provenance

    @property
    def last_model(self):
        return self._last_model

    @property
    def context(self) -> ObjectiveContext:
        return self._ctx

    def __getattr__(self, name: str):
        try:
            return self.__dict__[name]
        except KeyError:
            return getattr(self._fn, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_fn", "_ctx"):
            super().__setattr__(name, value)
        else:
            self.__dict__[name] = value

