"""
Run & Results tab widget for the fitting window.

Extracted from FittingWindow — owns the run-stamp display panel and
the statistics summary panel.  Read-only display with one-way data flow
from fit-completion results.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from PySide6 import QtWidgets
from PySide6.QtCore import Signal


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


class RunResultsTab(QtWidgets.QWidget):
    """Drop-in replacement for FittingWindow._create_run_results_tab()."""

    statusMessage = Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_run_stamp: dict = {}
        self._last_run_stamp_hash: str = ""
        self._last_run_stamp_short: str = ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)
        layout.addWidget(self._create_run_stamp_panel(), stretch=0)
        layout.addWidget(self._create_stats_panel(), stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_run_stamp(self, stamp: dict, stamp_hash: str, stamp_short: str) -> None:
        """Store stamp data, update label text, and enable copy buttons."""
        self._last_run_stamp = dict(stamp)
        self._last_run_stamp_hash = str(stamp_hash)
        self._last_run_stamp_short = str(stamp_short)
        self._run_stamp_label.setText(f"Stamp: {stamp_short}")
        self._run_stamp_label.setVisible(True)
        self._copy_stamp_button.setEnabled(True)
        self._copy_stamp_json_button.setEnabled(True)

    def update_statistics(self, stats: Dict[str, Any]) -> None:
        """Write stat values into the stats labels."""
        for key, label in self._stats_labels.items():
            value = stats.get(key)
            if value is None:
                label.setText("\u2014")
            elif isinstance(value, float):
                label.setText(f"{value:.6g}")
            else:
                label.setText(str(value))

    # ------------------------------------------------------------------
    # Layout construction (moved from FittingWindow factories)
    # ------------------------------------------------------------------

    def _create_run_stamp_panel(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox("Run Stamp")
        layout = QtWidgets.QVBoxLayout(group)

        info_label = QtWidgets.QLabel("Review and copy the most recent fit run stamp.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(info_label)

        row = QtWidgets.QHBoxLayout()
        self._run_stamp_label = QtWidgets.QLabel("")
        self._run_stamp_label.setObjectName("global_fit_run_stamp_label")
        self._run_stamp_label.setStyleSheet("font-size: 11px;")
        self._run_stamp_label.setVisible(False)

        self._copy_stamp_button = QtWidgets.QPushButton("Copy")
        self._copy_stamp_button.setObjectName("global_fit_copy_stamp_button")
        self._copy_stamp_button.setEnabled(False)
        self._copy_stamp_button.clicked.connect(self._on_copy_run_stamp_short)

        self._copy_stamp_json_button = QtWidgets.QPushButton("Copy JSON")
        self._copy_stamp_json_button.setObjectName("global_fit_copy_stamp_json_button")
        self._copy_stamp_json_button.setEnabled(False)
        self._copy_stamp_json_button.clicked.connect(self._on_copy_run_stamp_json)

        row.addWidget(self._run_stamp_label, stretch=1)
        row.addWidget(self._copy_stamp_button)
        row.addWidget(self._copy_stamp_json_button)
        layout.addLayout(row)
        return group

    def _create_stats_panel(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)
        self._stats_labels: Dict[str, QtWidgets.QLabel] = {}
        for label in ["Datasets", "Series", "Points", "Parameters", "DF", "SSQ", "Weighted SSQ", "-logL"]:
            value_label = QtWidgets.QLabel("\u2014")
            form.addRow(f"{label}:", value_label)
            self._stats_labels[label] = value_label
        return widget

    # ------------------------------------------------------------------
    # Copy handlers (moved from FittingWindow)
    # ------------------------------------------------------------------

    def _on_copy_run_stamp_short(self) -> None:
        stamp_short = self._last_run_stamp_short.strip()
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

    def _on_copy_run_stamp_json(self) -> None:
        stamp = self._last_run_stamp
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
