from __future__ import annotations

import math
from typing import Optional

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QDoubleValidator

__all__ = [
    "make_bounded_label",
    "make_placeholder_label",
    "make_pyqtgraph_fallback_widget",
    "make_scroll_area",
    "safe_float_parse",
    "set_bounded_label_text",
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
DEFAULT_DYNAMIC_LABEL_MAX_WIDTH = 320


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


def set_bounded_label_text(
    label: QtWidgets.QLabel,
    text: str,
    *,
    max_width: int = DEFAULT_DYNAMIC_LABEL_MAX_WIDTH,
    empty_text: str = "",
    tooltip: bool = True,
) -> None:
    """Set single-line dynamic label text without letting content define layout width."""
    full_text = str(text if text is not None else empty_text)
    if not full_text:
        full_text = str(empty_text)
    width = max(24, int(max_width))
    label.setMaximumWidth(width)
    label.setSizePolicy(QtWidgets.QSizePolicy.Maximum, label.sizePolicy().verticalPolicy())
    label.setWordWrap(False)
    elided = label.fontMetrics().elidedText(full_text, QtCore.Qt.ElideRight, width)
    label.setText(elided)
    label.setToolTip(full_text if bool(tooltip) and elided != full_text else "")


def make_bounded_label(
    text: str = "",
    parent: Optional[QtWidgets.QWidget] = None,
    *,
    max_width: int = DEFAULT_DYNAMIC_LABEL_MAX_WIDTH,
    tooltip: bool = True,
) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel("", parent)
    set_bounded_label_text(label, str(text), max_width=max_width, tooltip=tooltip)
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
