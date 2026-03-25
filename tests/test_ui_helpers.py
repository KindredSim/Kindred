import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QDoubleValidator

from kindred.gui.ui_helpers import (
    make_placeholder_label,
    make_pyqtgraph_fallback_widget,
    make_scroll_area,
    safe_float_parse,
    setup_scientific_validator,
)


@pytest.mark.gui
def test_make_scroll_area_configures_resizable_and_frameless(qt_app, qtbot):
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)

    scroll = make_scroll_area(parent)
    qtbot.addWidget(scroll)

    assert scroll.widgetResizable() is True
    assert scroll.frameShape() == QtWidgets.QFrame.NoFrame
    assert scroll.parent() is parent


@pytest.mark.gui
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


@pytest.mark.gui
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


@pytest.mark.gui
def test_make_pyqtgraph_fallback_widget_allows_custom_text(qt_app, qtbot):
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)

    widget = make_pyqtgraph_fallback_widget(parent, text="Custom warning")
    qtbot.addWidget(widget)

    assert widget.text() == "Custom warning"


@pytest.mark.unit
def test_safe_float_parse_rejects_nan_and_inf():
    assert safe_float_parse("nan", 1.25) == 1.25
    assert safe_float_parse("inf", 1.25) == 1.25
    assert safe_float_parse("-inf", 1.25) == 1.25


@pytest.mark.unit
def test_safe_float_parse_parses_finite_and_rejects_bad_values():
    assert safe_float_parse("1.5", 9.0) == 1.5
    assert safe_float_parse("1e-12", 9.0) == 1e-12
    assert safe_float_parse("not-a-number", 9.0) == 9.0
    assert safe_float_parse(None, 9.0) == 9.0  # type: ignore[arg-type]


@pytest.mark.gui
def test_setup_scientific_validator_attaches_scientific_validator(qt_app, qtbot):
    edit = QtWidgets.QLineEdit()
    qtbot.addWidget(edit)

    validator = setup_scientific_validator(edit)

    assert validator is edit.validator()
    assert isinstance(validator, QDoubleValidator)
    assert validator.notation() == QDoubleValidator.Notation.ScientificNotation
