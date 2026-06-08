from __future__ import annotations

import math
from typing import Optional

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QDoubleValidator

from kindred.gui.display_name_policy import (
    DATASET_LIST_LABEL_MAX_CHARS,
    CompactText,
    compact_diagnostic_text,
    compact_text,
)

__all__ = [
    "make_bounded_label",
    "make_compact_label",
    "make_placeholder_label",
    "make_pyqtgraph_fallback_widget",
    "make_scroll_area",
    "safe_float_parse",
    "set_bounded_label_text",
    "set_compact_label_text",
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
DEFAULT_DYNAMIC_LABEL_MAX_WIDTH = 220
DEFAULT_COMPACT_LABEL_MAX_CHARS = DATASET_LIST_LABEL_MAX_CHARS
DEFAULT_COMPACT_LABEL_MAX_WIDTH = 220


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


def _apply_non_layout_bearing_label_policy(label: QtWidgets.QLabel, *, max_width: int) -> None:
    width = max(24, int(max_width))
    label.setMinimumWidth(0)
    label.setMaximumWidth(width)
    label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, label.sizePolicy().verticalPolicy())
    label.setWordWrap(False)


def set_bounded_label_text(
    label: QtWidgets.QLabel,
    text: str,
    *,
    max_width: int = DEFAULT_DYNAMIC_LABEL_MAX_WIDTH,
    empty_text: str = "",
    tooltip: bool = True,
    tooltip_text: str | None = None,
) -> None:
    """Set single-line dynamic label text without letting content define layout width."""
    full_text = str(text if text is not None else empty_text)
    if not full_text:
        full_text = str(empty_text)
    explicit_tooltip = str(tooltip_text or "")
    _apply_non_layout_bearing_label_policy(label, max_width=max_width)
    width = max(24, int(max_width))
    elided = label.fontMetrics().elidedText(full_text, QtCore.Qt.ElideRight, width)
    label.setText(elided)
    if bool(tooltip):
        label.setToolTip(explicit_tooltip or (full_text if elided != full_text else ""))
    else:
        label.setToolTip("")


def set_compact_label_text(
    label: QtWidgets.QLabel,
    text: str,
    *,
    max_chars: int = DEFAULT_COMPACT_LABEL_MAX_CHARS,
    max_width: int = DEFAULT_COMPACT_LABEL_MAX_WIDTH,
    empty_text: str = "",
    tooltip: bool = True,
    diagnostic: bool = False,
    tooltip_text: str | None = None,
) -> CompactText:
    """Apply the central compact-display policy to a QLabel.

    Returns the compact policy result so callers can reuse the full/tooltip
    value when needed.
    """
    compact = (
        compact_diagnostic_text(text, max_chars=max_chars)
        if bool(diagnostic)
        else compact_text(text, max_chars=max_chars, empty_text=empty_text, mode="middle")
    )
    _apply_non_layout_bearing_label_policy(label, max_width=max_width)
    width = max(24, int(max_width))
    visible = label.fontMetrics().elidedText(compact.display, QtCore.Qt.ElideRight, width)
    width_elided = visible != compact.display
    label.setText(visible)
    explicit_tooltip = str(tooltip_text or "")
    if bool(tooltip):
        label.setToolTip(
            explicit_tooltip
            or (compact.full if (compact.was_elided or width_elided) else "")
        )
    else:
        label.setToolTip("")
    return compact


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


def make_compact_label(
    text: str = "",
    parent: Optional[QtWidgets.QWidget] = None,
    *,
    max_chars: int = DEFAULT_COMPACT_LABEL_MAX_CHARS,
    max_width: int = DEFAULT_COMPACT_LABEL_MAX_WIDTH,
    tooltip: bool = True,
) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel("", parent)
    set_compact_label_text(label, str(text), max_chars=max_chars, max_width=max_width, tooltip=tooltip)
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
