"""Application lifetime for Chemical Space Scan; trajectories never enter Qt."""
from __future__ import annotations

from dataclasses import replace
import os
import logging
from threading import Event

from PySide6 import QtCore, QtWidgets

from kindred.core.exceptions import SimulationCancelled
from kindred.scan.runtime import run_scan
from kindred.scan.sampling import PreparedScanDesign
from kindred.io.paths import same_local_path


logger = logging.getLogger(__name__)


class _ScanThread(QtCore.QThread):
    progress = QtCore.Signal(object)

    def __init__(self, config, prepared, check_only, parent):
        super().__init__(parent)
        self.config, self.prepared, self.check_only = config, prepared, check_only
        self.cancel_event = Event()
        self.result = None

    def run(self):
        if not self.check_only:
            self.result = run_scan(self.config, self.prepared, self.cancel_event, self.progress.emit)
            return

        def check():
            if self.cancel_event.is_set():
                raise SimulationCancelled()

        try:
            with PreparedScanDesign(self.config, self.prepared, check) as design:
                lines = ["Setup is feasible. No simulations were attempted."]
                for key, intervals in design.effective_ranges.items():
                    if key in self.config.ranges:
                        lines.append(f"{key}: " + "; ".join(
                            f"{'[' if i.include_lower else '('}{i.lower:g}, {i.upper:g}{']' if i.include_upper else ')'}"
                            for i in intervals))
                for key, expression in design.derived.items():
                    terms = " ".join(f"{float(value):+g} × {name}" for name, value in expression.terms.items())
                    lines.append(f"Calculate {key} = {float(expression.constant):g} {terms}")
                self.result = "\n".join(lines)
        except SimulationCancelled:
            self.result = "Setup check cancelled. No simulations were attempted."
        except Exception as exc:
            self.result = f"Setup cannot start: {exc}"


class ChemicalSpaceScanOwner(QtCore.QObject):
    progress = QtCore.Signal(object)
    completed = QtCore.Signal(object)
    status = QtCore.Signal(str)
    application_close_settlement_changed = QtCore.Signal()

    def __init__(self, gate, exports, session, parent):
        super().__init__(parent)
        self.gate = gate
        self.exports = exports
        self._session = session
        self.parent_widget = parent
        self.dialog = None
        self.capture = None
        self.worker = None
        self.close_fenced = False
        self.session_closing = False

    def open(self, initial_capture):
        if self.close_fenced or self.session_closing:
            return
        if self.dialog is None:
            if self.exports.is_exporting or not self.gate.is_idle:
                raise ValueError("Another computation or export is active.")
            capture = initial_capture()
            if capture.prepared_dsl is None:
                raise ValueError("Enter a valid mechanism before opening Chemical Space Scan.")
            from kindred.gui.widgets.chemical_space_scan import ChemicalSpaceScanDialog

            self.capture = capture
            self.dialog = ChemicalSpaceScanDialog(capture, self, self.parent_widget)
            self.dialog.destroyed.connect(self._dialog_destroyed)
            self.application_close_settlement_changed.emit()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def request_close(self):
        if self.session_closing:
            return
        self.session_closing = True
        self.cancel()
        self.application_close_settlement_changed.emit()
        if not self.busy:
            # Leave closeEvent/reject before destroying their receiver.
            QtCore.QTimer.singleShot(0, self.dialog, self._destroy_dialog)

    def _destroy_dialog(self):
        self.session_closing = True
        if self.dialog is not None:
            dialog = self.dialog
            dialog.hide()
            dialog.deleteLater()
            QtCore.QCoreApplication.sendPostedEvents(dialog, QtCore.QEvent.Type.DeferredDelete)
        else:
            self._dialog_destroyed()

    @QtCore.Slot()
    def _dialog_destroyed(self):
        self.dialog = None
        self.capture = None
        self.session_closing = False
        self.application_close_settlement_changed.emit()

    @QtCore.Slot(bool)
    def project_state_changed(self, project_replaced):
        if project_replaced:
            self._destroy_dialog()

    def close_idle_for_application_close(self):
        if self.busy:
            return False
        self.close_fenced = True
        self._destroy_dialog()
        return self.dialog is None

    @property
    def busy(self):
        return self.worker is not None

    @property
    def application_close_settled(self):
        return self.worker is None and (not self.session_closing or self.dialog is None)

    def owns_destination(self, path):
        worker = self.worker
        return bool(worker is not None and not worker.check_only
                    and same_local_path(path, worker.config.destination))

    def start(self, config, *, check_only=False):
        if self.close_fenced or self.session_closing or self.busy or self.exports.is_exporting or not self.gate.is_idle:
            return False
        if not check_only:
            destination = os.path.abspath(config.destination)
            if same_local_path(destination, self._session.current_path):
                raise ValueError("Cannot overwrite the currently open project archive.")
            if os.path.lexists(destination):
                answer = QtWidgets.QMessageBox.question(
                    self.dialog, "Overwrite Chemical Space Scan",
                    f"The destination already exists:\n{destination}\n\nOverwrite it?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No)
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    raise ValueError("Export was not started because overwrite was declined.")
            config = replace(config, destination=destination)
        # A modal overwrite question may have serviced other application events.
        if self.close_fenced or self.session_closing or self.dialog is None or self.exports.is_exporting:
            return False
        if not check_only and same_local_path(config.destination, self._session.current_path):
            raise ValueError("Cannot overwrite the currently open project archive.")
        if not self.gate.try_acquire(self, "scan"):
            return False
        self.worker = _ScanThread(config, self.capture.prepared_dsl, check_only, self)
        self.worker.progress.connect(self.progress)
        self.worker.finished.connect(self._settled)
        self.worker.start()
        self.application_close_settlement_changed.emit()
        return True

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel_event.set()

    def cancel_for_application_close(self):
        self.close_fenced = True
        self.session_closing = True
        self.cancel()
        if not self.busy:
            self._destroy_dialog()

    @QtCore.Slot()
    def _settled(self):
        worker = self.worker
        self.worker = None
        self.gate.release(self)
        if worker.check_only:
            message = worker.result
        else:
            result = worker.result
            message = (f"Scan {result.state}: {result.attempted}/{result.requested} attempted; "
                       f"{result.saved} records saved. {result.path} {result.error}")
        logger.info(message)
        self.status.emit(message)
        self.completed.emit(worker.result)
        worker.deleteLater()
        if self.close_fenced or self.session_closing:
            self._destroy_dialog()
        self.application_close_settlement_changed.emit()
