from __future__ import annotations

import pytest
from PySide6 import QtWidgets

from kindred.gui.widgets.collapsible_section import CollapsibleSection


pytestmark = pytest.mark.gui


def test_collapsible_section_state_does_not_depend_on_rendered_header_text(qt_app) -> None:
    _ = qt_app
    section = CollapsibleSection("Solver Settings")
    section.set_content_widget(QtWidgets.QLabel("content"))
    section._header.setText("broken header text")

    section.set_collapsed(True)
    assert section.is_collapsed is True
    assert section._header.text() == "▶ Solver Settings"

    section.set_collapsed(False)
    assert section.is_collapsed is False
    assert section._header.text() == "▼ Solver Settings"


def test_collapsible_section_toggle_preserves_declared_title(qt_app) -> None:
    _ = qt_app
    section = CollapsibleSection("Advanced Integration Settings")

    section.toggle()
    section.toggle()

    assert section._header.text() == "▼ Advanced Integration Settings"
