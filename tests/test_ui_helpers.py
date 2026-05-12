import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QDoubleValidator

from kindred.gui.ui_helpers import (
    make_bounded_label,
    make_placeholder_label,
    make_pyqtgraph_fallback_widget,
    make_scroll_area,
    set_bounded_label_text,
    setup_scientific_validator,
)

pytestmark = [pytest.mark.gui]


def test_make_scroll_area_configures_resizable_and_frameless(qt_app, qtbot):
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)

    scroll = make_scroll_area(parent)
    qtbot.addWidget(scroll)

    assert scroll.widgetResizable() is True
    assert scroll.frameShape() == QtWidgets.QFrame.NoFrame
    assert scroll.parent() is parent

def test_make_placeholder_label_is_centered_and_muted(qt_app, qtbot):
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)

    label = make_placeholder_label("Hello", parent)
    qtbot.addWidget(label)

    assert isinstance(label, QtWidgets.QLabel)
    assert label.text() == "Hello"
    assert label.alignment() == QtCore.Qt.AlignCenter
    style = label.styleSheet()
    assert "font-style: italic" in style
    assert "background: transparent" in style
    assert "padding: 0" in style
    assert label.parent() is parent


def test_set_bounded_label_text_elides_display_and_preserves_tooltip(qt_app, qtbot):
    label = QtWidgets.QLabel()
    qtbot.addWidget(label)
    full = "Selected dataset: " + ("very-long-name-" * 20)

    set_bounded_label_text(label, full, max_width=120)

    assert label.maximumWidth() == 120
    assert label.toolTip() == full
    assert label.text() != full
    assert len(label.text()) < len(full)


def test_make_bounded_label_applies_initial_text(qt_app, qtbot):
    label = make_bounded_label("hello", max_width=96)
    qtbot.addWidget(label)

    assert label.text() == "hello"
    assert label.toolTip() == ""
    assert label.maximumWidth() == 96

def test_make_pyqtgraph_fallback_widget_returns_centered_red_warning(qt_app, qtbot):
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)

    widget = make_pyqtgraph_fallback_widget(parent)
    qtbot.addWidget(widget)

    assert isinstance(widget, QtWidgets.QLabel)
    assert "PyQtGraph" in widget.text()
    assert widget.alignment() == QtCore.Qt.AlignCenter
    assert widget.styleSheet() == "QLabel { font-weight: bold; font-size: 14px; }"
    assert widget.parent() is parent

def test_make_pyqtgraph_fallback_widget_allows_custom_text(qt_app, qtbot):
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)

    widget = make_pyqtgraph_fallback_widget(parent, text="Custom warning")
    qtbot.addWidget(widget)

    assert widget.text() == "Custom warning"

def test_setup_scientific_validator_attaches_scientific_validator(qt_app, qtbot):
    edit = QtWidgets.QLineEdit()
    qtbot.addWidget(edit)

    validator = setup_scientific_validator(edit)

    assert validator is edit.validator()
    assert isinstance(validator, QDoubleValidator)
    assert validator.notation() == QDoubleValidator.Notation.ScientificNotation
