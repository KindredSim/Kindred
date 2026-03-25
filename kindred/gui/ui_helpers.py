from __future__ import annotations

import math
from typing import Optional

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QDoubleValidator

__all__ = [
    "make_placeholder_label",
    "make_pyqtgraph_fallback_widget",
    "make_scroll_area",
    "safe_float_parse",
    "setup_scientific_validator",
]


_PLACEHOLDER_LABEL_STYLE = (
    "QLabel { font-style: italic; background: transparent; padding: 0px; }"
)
_PYQTGRAPH_FALLBACK_LABEL_STYLE = "QLabel { font-weight: bold; font-size: 14px; }"
_DEFAULT_PYQTGRAPH_FALLBACK_TEXT = (
    "⚠️ PyQtGraph not available\n\n"
    "Install with: pip install pyqtgraph\n\n"
    "Plotting features require PyQtGraph."
)


def make_scroll_area(parent: Optional[QtWidgets.QWidget] = None) -> QtWidgets.QScrollArea:
    scroll = QtWidgets.QScrollArea(parent)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    return scroll


def make_placeholder_label(text: str, parent: Optional[QtWidgets.QWidget] = None) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(str(text), parent)
    label.setAlignment(QtCore.Qt.AlignCenter)
    label.setStyleSheet(_PLACEHOLDER_LABEL_STYLE)
    return label


def make_pyqtgraph_fallback_widget(
    parent: Optional[QtWidgets.QWidget] = None,
    *,
    text: Optional[str] = None,
) -> QtWidgets.QWidget:
    warning = QtWidgets.QLabel(_DEFAULT_PYQTGRAPH_FALLBACK_TEXT if text is None else str(text), parent)
    warning.setAlignment(QtCore.Qt.AlignCenter)
    warning.setStyleSheet(_PYQTGRAPH_FALLBACK_LABEL_STYLE)
    return warning


def safe_float_parse(val_str: str, default: float) -> float:
    try:
        parsed = float(val_str)
    except (ValueError, TypeError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def setup_scientific_validator(line_edit: QtWidgets.QLineEdit) -> QDoubleValidator:
    validator = QDoubleValidator(0.0, 1e300, 323, line_edit)
    validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
    line_edit.setValidator(validator)
    return validator
