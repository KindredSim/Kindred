from __future__ import annotations

import pytest
from PySide6 import QtWidgets


pytestmark = pytest.mark.gui


def test_message_box_critical_without_details_uses_static_dialog(main_window, monkeypatch):
    calls: list[tuple[object, str, str]] = []

    def _critical(parent, title, message):
        calls.append((parent, str(title), str(message)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    def _unexpected_exec(self):
        raise AssertionError("details-free critical dialog should use the static helper")

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", _critical)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _unexpected_exec)

    main_window.message_box_critical("Simulation Error", "Simulation failed")

    assert calls == [(main_window, "Simulation Error", "Simulation failed")]


def test_message_box_critical_with_details_uses_detail_pane(main_window, monkeypatch):
    dialogs: list[dict[str, object]] = []

    def _unexpected_critical(*_args, **_kwargs):
        raise AssertionError("details must be rendered by an instance dialog")

    def _capture_exec(self):
        dialogs.append(
            {
                "title": self.windowTitle(),
                "text": self.text(),
                "details": self.detailedText(),
                "icon": self.icon(),
            }
        )
        return int(QtWidgets.QMessageBox.StandardButton.Ok)

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", _unexpected_critical)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

    main_window.message_box_critical(
        "Simulation Error",
        "Simulation failed:\n\nsolver blew up",
        details="Traceback line 1\nTraceback line 2",
    )

    assert dialogs == [
        {
            "title": "Simulation Error",
            "text": "Simulation failed:\n\nsolver blew up",
            "details": "Traceback line 1\nTraceback line 2",
            "icon": QtWidgets.QMessageBox.Icon.Critical,
        }
    ]
