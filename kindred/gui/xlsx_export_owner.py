"""Application-owned asynchronous XLSX export."""

from __future__ import annotations

import logging
import os
import stat
import tempfile

from PySide6 import QtCore

from kindred.gui.workspace_result_presenter import (
    ResultDataBlock,
    write_xlsx,
)
from kindred.io.paths import file_error_text, same_local_path

logger = logging.getLogger(__name__)

__all__ = ["XlsxExportOwner"]


class _XlsxExportCancelled(Exception):
    """Internal cooperative cancellation marker."""


class _XlsxExportWorker(QtCore.QThread):
    """Write one workbook to a temporary destination, then atomically replace."""

    succeeded = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, str)
    cancelled = QtCore.Signal(str)

    def __init__(
        self,
        *,
        blocks: tuple[ResultDataBlock, ...],
        path: str,
        parent: QtCore.QObject,
    ) -> None:
        super().__init__(parent)
        self._blocks = blocks
        self._path = os.path.abspath(os.fspath(path))
        self._commit_started = False

    @property
    def path(self) -> str:
        return self._path

    def cancel(self) -> None:
        """Request cooperative cancellation before the atomic commit boundary."""

        if not self._commit_started and not self.isFinished():
            self.requestInterruption()

    def _check_cancelled(self) -> None:
        if self.isInterruptionRequested():
            raise _XlsxExportCancelled()

    def run(self) -> None:
        temporary_path = ""
        descriptor = -1
        try:
            self._check_cancelled()
            parent = os.path.dirname(self._path) or os.curdir
            if not os.path.isdir(parent):
                raise FileNotFoundError(
                    f"Export directory does not exist: {parent}."
                )
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{os.path.basename(self._path)}.",
                suffix=".partial",
                dir=parent,
            )
            self._check_cancelled()
            with os.fdopen(descriptor, "w+b") as destination:
                descriptor = -1
                self._check_cancelled()
                completion = write_xlsx(
                    self._blocks,
                    destination,
                    cancellation_check=self._check_cancelled,
                )
                self._check_cancelled()
                destination.flush()
                self._check_cancelled()
            try:
                existing_mode = stat.S_IMODE(os.stat(self._path).st_mode)
            except FileNotFoundError:
                existing_mode = None
            if existing_mode is not None:
                os.chmod(temporary_path, existing_mode)
            self._check_cancelled()
            # Cancellation no longer interrupts the single atomic replacement.
            self._commit_started = True
            os.replace(temporary_path, self._path)
            temporary_path = ""
        except _XlsxExportCancelled:
            self.cancelled.emit(self._path)
        except Exception as exc:
            if not self._commit_started and self.isInterruptionRequested():
                self.cancelled.emit(self._path)
            else:
                # The failure is shown with its path.
                self.failed.emit(self._path, file_error_text(exc, None))
        else:
            self.succeeded.emit(self._path, completion)
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning(
                        "Could not remove partial XLSX export %s",
                        temporary_path,
                        exc_info=True,
                    )


class XlsxExportOwner(QtCore.QObject):
    """Own one exact XLSX worker and its application-close fence."""

    succeeded = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, str)
    cancelled = QtCore.Signal(str)
    application_close_settlement_changed = QtCore.Signal()

    def __init__(self, *, parent: QtCore.QObject) -> None:
        super().__init__(parent)
        self._active_worker: _XlsxExportWorker | None = None
        self._application_close_fenced = False

    @property
    def is_exporting(self) -> bool:
        return self._active_worker is not None

    @property
    def application_close_settled(self) -> bool:
        """Close is settled precisely when no worker remains in custody."""

        return self._active_worker is None

    def owns_destination(self, path: str | os.PathLike[str]) -> bool:
        worker = self._active_worker
        return bool(
            worker is not None
            and same_local_path(path, worker.path)
        )

    def start(
        self,
        blocks: tuple[ResultDataBlock, ...],
        path: str,
    ) -> None:
        """Admit and start exactly one worker; synchronous failures raise."""

        absolute_path = os.path.abspath(os.fspath(path))
        if self._application_close_fenced:
            raise RuntimeError("XLSX export is unavailable while Kindred closes.")
        if self._active_worker is not None:
            raise RuntimeError("Another XLSX export is already active.")
        worker = _XlsxExportWorker(
            blocks=blocks,
            path=absolute_path,
            parent=self,
        )
        self._active_worker = worker
        # Direct signal-to-signal publication carries the worker's admitted
        # path and terminal result without a forwarding/revalidation layer.
        worker.succeeded.connect(self.succeeded)
        worker.failed.connect(self.failed)
        worker.cancelled.connect(self.cancelled)
        worker.finished.connect(self._worker_finished)
        worker.start()

    @QtCore.Slot()
    def _worker_finished(self) -> None:
        worker = self.sender()
        if worker is not self._active_worker:
            return
        self._active_worker = None
        worker.deleteLater()
        self.application_close_settlement_changed.emit()

    def cancel(self) -> None:
        worker = self._active_worker
        if worker is not None:
            worker.cancel()

    def cancel_for_application_close(self) -> None:
        self._application_close_fenced = True
        self.cancel()
