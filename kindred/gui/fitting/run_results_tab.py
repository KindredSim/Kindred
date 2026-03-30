"""
Results tab widget for the fitting window.

Extracted from FittingWindow — owns the run-stamp data and statistics
(displayed via a non-modal popup dialog triggered from the footer).
Read-only display with one-way data flow from fit-completion results.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal


def _get_clipboard():
    """
    Clipboard accessor seam (monkeypatchable in tests).

    Returns a Qt clipboard-like object with `setText(str)` / `text()` when available,
    otherwise returns None.
    """
    try:
        app = QtWidgets.QApplication.instance()
        return app.clipboard() if app is not None else None
    except Exception:
        return None


class ResultsSummaryDialog(QtWidgets.QDialog):
    """Non-modal popup displaying fit run stamp and statistics."""

    statusMessage = Signal(str)

    def __init__(
        self,
        stamp: dict,
        stamp_hash: str,
        stamp_short: str,
        stats: Optional[Dict[str, Any]] = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("Results Summary")
        self.resize(480, 360)

        self._stamp = dict(stamp)
        self._stamp_hash = str(stamp_hash)
        self._stamp_short = str(stamp_short)

        layout = QtWidgets.QVBoxLayout(self)

        info_label = QtWidgets.QLabel("Review and copy the most recent fit run stamp.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(info_label)

        row = QtWidgets.QHBoxLayout()
        self._run_stamp_label = QtWidgets.QLabel(f"Stamp: {stamp_short}")
        self._run_stamp_label.setObjectName("global_fit_run_stamp_label")
        self._run_stamp_label.setStyleSheet("font-size: 11px;")

        self._copy_stamp_button = QtWidgets.QPushButton("Copy")
        self._copy_stamp_button.setObjectName("global_fit_copy_stamp_button")
        self._copy_stamp_button.clicked.connect(self._on_copy_run_stamp_short)

        self._copy_stamp_json_button = QtWidgets.QPushButton("Copy JSON")
        self._copy_stamp_json_button.setObjectName("global_fit_copy_stamp_json_button")
        self._copy_stamp_json_button.clicked.connect(self._on_copy_run_stamp_json)

        row.addWidget(self._run_stamp_label, stretch=1)
        row.addWidget(self._copy_stamp_button)
        row.addWidget(self._copy_stamp_json_button)
        layout.addLayout(row)

        # Statistics form
        stats_group = QtWidgets.QGroupBox("Fit Statistics")
        form = QtWidgets.QFormLayout(stats_group)
        self._stats_labels: Dict[str, QtWidgets.QLabel] = {}
        for label_text in ["Datasets", "Series", "Points", "Parameters", "DF", "SSQ", "Weighted SSQ", "-logL"]:
            value_label = QtWidgets.QLabel("\u2014")
            form.addRow(f"{label_text}:", value_label)
            self._stats_labels[label_text] = value_label
        layout.addWidget(stats_group)

        if stats:
            self._update_stats(stats)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _update_stats(self, stats: Dict[str, Any]) -> None:
        for key, label in self._stats_labels.items():
            value = stats.get(key)
            if value is None:
                label.setText("\u2014")
            elif isinstance(value, float):
                label.setText(f"{value:.6g}")
            else:
                label.setText(str(value))

    # ------------------------------------------------------------------
    # Copy handlers
    # ------------------------------------------------------------------

    def _on_copy_run_stamp_short(self) -> None:
        stamp_short = self._stamp_short.strip()
        if not stamp_short:
            return
        clipboard = _get_clipboard()
        if clipboard is None:
            self.statusMessage.emit("Clipboard unavailable")
            return
        try:
            clipboard.setText(stamp_short)
        except Exception:
            self.statusMessage.emit("Failed to copy stamp")
            return
        self.statusMessage.emit("Copied stamp hash")

    def refresh(
        self,
        stamp: dict,
        stamp_hash: str,
        stamp_short: str,
        stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update all stamp and stats content from the owning tab."""
        self._stamp = dict(stamp)
        self._stamp_hash = str(stamp_hash)
        self._stamp_short = str(stamp_short)
        self._run_stamp_label.setText(f"Stamp: {stamp_short}")
        self._update_stats(stats or {})

    def _on_copy_run_stamp_json(self) -> None:
        stamp = self._stamp
        if not isinstance(stamp, dict) or not stamp:
            return
        clipboard = _get_clipboard()
        if clipboard is None:
            self.statusMessage.emit("Clipboard unavailable")
            return
        try:
            text = json.dumps(stamp, sort_keys=True, indent=2, ensure_ascii=True)
        except Exception:
            self.statusMessage.emit("Failed to serialize stamp")
            return
        try:
            clipboard.setText(text)
        except Exception:
            self.statusMessage.emit("Failed to copy stamp")
            return
        self.statusMessage.emit("Copied stamp JSON")


# Keep old name importable for tests that reference it directly.
RunStampDialog = ResultsSummaryDialog


class RunResultsTab(QtWidgets.QWidget):
    """Drop-in replacement for FittingWindow._create_run_results_tab()."""

    statusMessage = Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_run_stamp: dict = {}
        self._last_run_stamp_hash: str = ""
        self._last_run_stamp_short: str = ""
        self._last_stats: Dict[str, Any] = {}
        self._stamp_dialog: Optional[ResultsSummaryDialog] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        placeholder = QtWidgets.QLabel("Run a fit to see results here.")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(placeholder, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_run_stamp(self, stamp: dict, stamp_hash: str, stamp_short: str) -> None:
        """Store stamp data for later display via the popup dialog."""
        self._last_run_stamp = dict(stamp)
        self._last_run_stamp_hash = str(stamp_hash)
        self._last_run_stamp_short = str(stamp_short)
        self._last_stats = {}
        if self._stamp_dialog is not None and self._stamp_dialog.isVisible():
            self._stamp_dialog.refresh(
                self._last_run_stamp,
                self._last_run_stamp_hash,
                self._last_run_stamp_short,
                None,
            )

    def open_results_summary_dialog(self) -> None:
        """Show the results summary popup dialog."""
        if not self._last_run_stamp:
            return
        if self._stamp_dialog is not None and self._stamp_dialog.isVisible():
            self._stamp_dialog.raise_()
            self._stamp_dialog.activateWindow()
            return
        dialog = ResultsSummaryDialog(
            self._last_run_stamp,
            self._last_run_stamp_hash,
            self._last_run_stamp_short,
            stats=self._last_stats or None,
            parent=self,
        )
        dialog.statusMessage.connect(self.statusMessage)
        dialog.destroyed.connect(lambda: setattr(self, '_stamp_dialog', None))
        self._stamp_dialog = dialog
        dialog.show()

    def update_statistics(self, stats: Dict[str, Any]) -> None:
        """Store stat values; update dialog if open."""
        self._last_stats = dict(stats)
        if self._stamp_dialog is not None and self._stamp_dialog.isVisible():
            self._stamp_dialog.refresh(
                self._last_run_stamp,
                self._last_run_stamp_hash,
                self._last_run_stamp_short,
                self._last_stats,
            )
