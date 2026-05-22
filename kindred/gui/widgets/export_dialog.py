# kindred/gui/widgets/export_dialog.py

"""
CSV Export dialog (no placeholders).

What this dialog does
---------------------
- Lets the user choose a target .csv file.
- Offers a simple scope choice:
    * "Use current axis selections"
    * "All available series"
- Validates filename and appends `.csv` if missing.
- Emits `exportAccepted(config: dict)` when the user confirms.

Returned config schema
----------------------
{
  "path": str,                         # absolute or OS path chosen by the user
  "scope": "axis" | "all",
  "overwrite": bool                    # whether user allowed overwrite
}

Notes
-----
- Actual CSV writing is handled by the caller (see `kindred.gui.controllers.project_controller.ProjectController`).
- The caller may pass an initial directory via `set_initial_directory(path)`
  or by `open_with_suggested_path(path)`.
- No filesystem changes are performed here beyond asking the user for a path.
- No cwd or network usage.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6 import QtCore, QtWidgets

from kindred.io import paths as kindred_io_paths

__all__ = ["ExportDialog"]


class ExportDialog(QtWidgets.QDialog):
    exportAccepted = QtCore.Signal(dict)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export CSV")
        self.setModal(True)
        self.resize(600, 400)

        # ---------------- Path row ----------------
        self._edit_path = QtWidgets.QLineEdit(self)
        self._btn_browse = QtWidgets.QPushButton("Browse…", self)
        self._btn_browse.clicked.connect(self._on_browse)

        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self._edit_path, 1)
        path_row.addWidget(self._btn_browse)

        # ---------------- Scope group ----------------
        self._radio_scope_axis = QtWidgets.QRadioButton("Use current axis selections", self)
        self._radio_scope_all = QtWidgets.QRadioButton("All available series", self)
        self._radio_scope_axis.setChecked(True)

        grp_scope_lay = QtWidgets.QVBoxLayout()
        grp_scope_lay.addWidget(self._radio_scope_axis)
        grp_scope_lay.addWidget(self._radio_scope_all)
        grp_scope = QtWidgets.QGroupBox("Scope", self)
        grp_scope.setLayout(grp_scope_lay)
        grp_scope.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )

        # ---------------- Buttons ----------------
        self._btn_ok = QtWidgets.QPushButton("Save")
        self._btn_cancel = QtWidgets.QPushButton("Cancel")
        self._btn_ok.setDefault(True)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self._btn_ok)
        btns.addWidget(self._btn_cancel)

        # ---------------- Main layout ----------------
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)
        lay.addLayout(path_row)
        lay.addWidget(grp_scope)
        lay.addStretch(1)
        lay.addLayout(btns)

        # ---------------- Wiring ----------------
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_ok.clicked.connect(self._on_accept)

        # State
        self._initial_dir: Optional[str] = None

    # ---------------- Public helpers ----------------

    def set_initial_directory(self, path: Optional[str]) -> None:
        """Set a directory to start the file dialog in (e.g., ./outputs)."""
        if path and isinstance(path, str):
            self._initial_dir = path

    def open_with_suggested_path(self, path: Optional[str]) -> None:
        """
        Open the dialog with an initial suggested file path in the line edit.
        Does not create any file.
        """
        if path:
            self._edit_path.setText(path)
        self.open()

    # ---------------- Internal logic ----------------

    def _on_browse(self) -> None:
        candidate = self._initial_dir or os.path.dirname(self._edit_path.text() or "")
        start_dir = kindred_io_paths.resolve_start_dir(candidate)
        title = "Save CSV"
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            title,
            start_dir,
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            if not path.lower().endswith(".csv"):
                path = path + ".csv"
            self._edit_path.setText(path)

    def _on_accept(self) -> None:
        # Validate path
        raw_path = self._edit_path.text().strip()
        if not raw_path:
            QtWidgets.QMessageBox.warning(self, "Missing path", "Please choose a file path for the CSV.")
            return

        # Normalize extension
        path = raw_path if raw_path.lower().endswith(".csv") else (raw_path + ".csv")

        # Scope
        scope = "axis" if self._radio_scope_axis.isChecked() else "all"

        # Confirm overwrite
        overwrite = True
        if os.path.isfile(path):
            resp = QtWidgets.QMessageBox.question(
                self,
                "Overwrite file?",
                f"The file already exists:\n{path}\n\nDo you want to overwrite it?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if resp != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            overwrite = True

        payload = {
            "path": path,
            "scope": scope,
            "overwrite": overwrite,
        }
        self.exportAccepted.emit(payload)
        self.accept()

    # ---------------- Sizing ----------------


    # ---------------- Programmatic control ----------------

    def set_scope(self, scope: str) -> None:
        """Set export scope programmatically.

        Parameters
        ----------
        scope : str
            "axis"/"visible" or "all"
        """
        s = str(scope).strip().lower()
        if s in {"axis", "visible"}:
            self._radio_scope_axis.setChecked(True)
        elif s == "all":
            self._radio_scope_all.setChecked(True)
        else:
            raise ValueError(f"Unknown export scope: {scope!r}")

    # (No hardcoded sizeHint override: allow Qt layouts to compute native hints.)
