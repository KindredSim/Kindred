from __future__ import annotations

from typing import Literal

from PySide6 import QtCore

__all__ = ["ComputationGate", "ComputationKind"]

ComputationKind = Literal["run", "preview", "fit", "scan", "calculator"]


class ComputationGate(QtCore.QObject):
    """GUI-thread identity guard for scientific computation exclusion."""

    changed = QtCore.Signal()

    def __init__(self, *, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._active_owner: object | None = None
        self._kind: ComputationKind | None = None

    @property
    def is_idle(self) -> bool:
        return self._active_owner is None

    @property
    def active_kind(self) -> ComputationKind | None:
        return self._kind

    def try_acquire(self, owner: object, kind: ComputationKind) -> bool:
        if self._active_owner is not None:
            return False
        self._active_owner = owner
        self._kind = kind
        self.changed.emit()
        return True

    def release(self, owner: object) -> bool:
        if self._active_owner is not owner:
            return False
        self._active_owner = None
        self._kind = None
        self.changed.emit()
        return True
