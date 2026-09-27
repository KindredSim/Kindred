from __future__ import annotations

import sys
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QDoubleValidator

from kindred.core.quantity_units import format_unit_for_display
from kindred.gui.display_name_policy import (
    DATASET_LIST_LABEL_MAX_CHARS,
    CompactText,
    literal_tooltip,
    compact_diagnostic_text,
    compact_text,
)

__all__ = [
    "SCIENTIFIC_NAME_HTML_ROLE",
    "ControlRowsLayout",
    "configure_compact_numeric_edit",
    "ScientificNameComboBox",
    "ScientificNameDelegate",
    "ScientificNameHeaderView",
    "clamp_initial_size",
    "make_bounded_label",
    "make_placeholder_label",
    "make_scroll_area",
    "set_bounded_label_text",
    "set_compact_label_text",
    "setup_scientific_validator",
]


_PLACEHOLDER_LABEL_STYLE = (
    "QLabel { font-style: italic; background: transparent; padding: 0px; }"
)
DEFAULT_DYNAMIC_LABEL_MAX_WIDTH = 220
DEFAULT_COMPACT_LABEL_MAX_CHARS = DATASET_LIST_LABEL_MAX_CHARS
DEFAULT_COMPACT_LABEL_MAX_WIDTH = 220


class ControlRowsLayout(QtWidgets.QLayout):
    """Measure compact control strips at their allocated width before paint."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(8)

    def addItem(self, item) -> None:
        self._items.append(item)
        self.invalidate()

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            item = self._items.pop(index)
            self.invalidate()
            return item
        return None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._place(QtCore.QRect(0, 0, width, 0), apply=False)

    def minimumHeightForWidth(self, width: int) -> int:
        return self.heightForWidth(width)

    def minimumSize(self) -> QtCore.QSize:
        size = QtCore.QSize(0, 0)
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QtCore.QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    def sizeHint(self) -> QtCore.QSize:
        sizes = [item.sizeHint() for item in self._items if not item.isEmpty()]
        margins = self.contentsMargins()
        return QtCore.QSize(
            sum(size.width() for size in sizes) + max(0, len(sizes) - 1) * self.spacing()
            + margins.left() + margins.right(),
            max((size.height() for size in sizes), default=0)
            + margins.top() + margins.bottom(),
        )

    def setGeometry(self, rect: QtCore.QRect) -> None:
        super().setGeometry(rect)
        self._place(rect, apply=True)

    def _place(self, rect: QtCore.QRect, *, apply: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, row_height = area.x(), area.y(), 0
        for item in self._items:
            if item.isEmpty():
                continue
            size = item.sizeHint().boundedTo(item.maximumSize())
            width = min(size.width(), max(0, area.width()))
            if x > area.x() and x + width > area.right() + 1:
                x = area.x()
                y += row_height + self.spacing()
                row_height = 0
            height = item.heightForWidth(width) if item.hasHeightForWidth() else size.height()
            if apply:
                item_x = area.right() + 1 - width if item.alignment() & QtCore.Qt.AlignRight else x
                item.setGeometry(QtCore.QRect(item_x, y, width, height))
            x += width + self.spacing()
            row_height = max(row_height, height)
        return y + row_height - rect.y() + margins.bottom()


class _CompactNumericSizing(QtCore.QObject):
    def __init__(self, edit: QtWidgets.QLineEdit, characters: int | None) -> None:
        super().__init__(edit)
        self.edit = edit
        self.characters = characters
        edit.installEventFilter(self)
        self.update()

    def update(self) -> None:
        edit = self.edit
        if self.characters is None:
            edit.setMinimumWidth(0)
            edit.setMaximumWidth(16777215)
            edit.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
            return
        edit.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        option = QtWidgets.QStyleOptionFrame()
        edit.initStyleOption(option)
        margins = edit.textMargins()
        contents = QtCore.QSize(
            edit.fontMetrics().horizontalAdvance("0" * self.characters)
            + margins.left() + margins.right() + 4,
            edit.fontMetrics().height(),
        )
        size = edit.style().sizeFromContents(
            QtWidgets.QStyle.CT_LineEdit, option, contents, edit
        )
        edit.setFixedWidth(size.width())

    def eventFilter(self, watched, event):
        if event.type() in {QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange}:
            self.update()
        return False


def configure_compact_numeric_edit(
    edit: QtWidgets.QLineEdit, *, characters: int | None,
) -> None:
    """Reserve a numeric viewport; None restores the native preferred width."""
    sizing = getattr(edit, "_compact_numeric_sizing", None)
    if sizing is None:
        edit._compact_numeric_sizing = _CompactNumericSizing(edit, characters)
    else:
        sizing.characters = characters
        sizing.update()


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


class _ResponsiveLabelText(QtCore.QObject):
    """Keep the original literal text while projecting to the current label width."""

    def __init__(self, label: QtWidgets.QLabel) -> None:
        super().__init__(label)
        self.label = label
        self.visible = ""
        self.last_tooltip = ""
        self.compact = CompactText("", "", False)
        self.tooltip = True
        self.explicit_tooltip = ""
        label.installEventFilter(self)

    def update(self) -> None:
        label = self.label
        width = max(0, label.contentsRect().width() - 2 * label.margin())
        self.visible = label.fontMetrics().elidedText(
            self.compact.display, QtCore.Qt.ElideRight, width
        )
        label.setText(self.visible)
        label.setAccessibleName(self.compact.full)
        self.last_tooltip = (
            self.explicit_tooltip
            or (self.compact.full if self.compact.tooltip or self.visible != self.compact.full else "")
        ) if self.tooltip else ""
        self.last_tooltip = literal_tooltip(self.last_tooltip)
        label.setToolTip(self.last_tooltip)

    def eventFilter(self, watched, event):
        if event.type() in {
            QtCore.QEvent.Resize, QtCore.QEvent.FontChange,
            QtCore.QEvent.StyleChange, QtCore.QEvent.Show, QtCore.QEvent.Paint,
        }:
            if self.label.text() != self.visible:
                full = self.label.text()
                self.compact = CompactText(full, full, False)
            if self.label.toolTip() != self.last_tooltip:
                self.explicit_tooltip = self.label.toolTip()
            self.update()
        return False


def _set_label_projection(label, compact, *, max_width, tooltip, tooltip_text):
    _apply_non_layout_bearing_label_policy(label, max_width=max_width)
    label.setTextFormat(QtCore.Qt.PlainText)
    projection = getattr(label, "_responsive_text", None)
    if projection is None:
        projection = _ResponsiveLabelText(label)
        label._responsive_text = projection
    projection.compact = compact
    projection.tooltip = tooltip
    projection.explicit_tooltip = str(tooltip_text or "")
    projection.update()


def set_bounded_label_text(
    label: QtWidgets.QLabel, text: str, *,
    max_width: int = DEFAULT_DYNAMIC_LABEL_MAX_WIDTH,
    empty_text: str = "", tooltip: bool = True,
    tooltip_text: str | None = None,
) -> None:
    full = str(text if text is not None else empty_text) or str(empty_text)
    _set_label_projection(label, CompactText(full, full, False),
                          max_width=max_width, tooltip=tooltip, tooltip_text=tooltip_text)


def set_compact_label_text(
    label: QtWidgets.QLabel, text: str, *,
    max_chars: int = DEFAULT_COMPACT_LABEL_MAX_CHARS,
    max_width: int = DEFAULT_COMPACT_LABEL_MAX_WIDTH,
    empty_text: str = "", tooltip: bool = True, diagnostic: bool = False,
    tooltip_text: str | None = None,
) -> CompactText:
    compact = (
        compact_diagnostic_text(text, max_chars=max_chars) if diagnostic
        else compact_text(text, max_chars=max_chars, empty_text=empty_text, mode="middle")
    )
    _set_label_projection(label, compact, max_width=max_width,
                          tooltip=tooltip, tooltip_text=tooltip_text)
    return compact


def clamp_initial_size(widget: QtWidgets.QWidget, width: int, height: int) -> None:
    """Choose a preferred initial size within the current screen's work area."""
    available = widget.screen().availableGeometry()
    widget.resize(min(width, available.width()), min(height, available.height()))


SCIENTIFIC_NAME_HTML_ROLE = int(QtCore.Qt.UserRole) + 173


def paint_markup(painter, rect, markup, *, font, alignment, palette) -> None:
    """Paint one unwrapped line of scientific markup, vertically centred in rect."""
    document = QtGui.QTextDocument()
    document.setDocumentMargin(0)
    document.setDefaultFont(font)
    text_option = QtGui.QTextOption(alignment)
    text_option.setWrapMode(QtGui.QTextOption.WrapMode.NoWrap)
    document.setDefaultTextOption(text_option)
    document.setTextWidth(max(0, rect.width()))
    document.setHtml(markup)
    painter.save()
    painter.setClipRect(rect)
    painter.translate(rect.left(), rect.top() + max(0, (rect.height() - document.size().height()) / 2))
    context = QtGui.QAbstractTextDocumentLayout.PaintContext()
    context.palette = palette
    try:
        document.documentLayout().draw(painter, context)
    finally:
        painter.restore()


def data_unit_labels(unit) -> tuple[str, str]:
    """Plain text and scientific markup for a measurement unit on data surfaces."""
    if unit is None:
        return "Unspecified", "Unspecified"
    if unit.symbol == "1":
        return "Dimensionless", "Dimensionless"
    return unit.symbol, format_unit_for_display(unit)


class ScientificNameDelegate(QtWidgets.QStyledItemDelegate):
    """Paint explicit scientific markup while retaining the model's plain text."""

    def paint(self, painter, option, index):
        markup = index.data(SCIENTIFIC_NAME_HTML_ROLE)
        if not markup:
            super().paint(painter, option, index)
            return
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget else QtWidgets.QApplication.style()
        opt.text = ""
        style.drawControl(QtWidgets.QStyle.CE_ItemViewItem, opt, painter, opt.widget)
        rect = style.subElementRect(QtWidgets.QStyle.SE_ItemViewItemText, opt, opt.widget)
        inset = max(opt.fontMetrics.horizontalAdvance(" "), style.pixelMetric(QtWidgets.QStyle.PM_FocusFrameHMargin))
        rect.adjust(inset, 0, -inset, 0)
        palette = QtGui.QPalette(opt.palette)
        if opt.state & QtWidgets.QStyle.State_Selected:
            palette.setColor(QtGui.QPalette.Text, opt.palette.color(QtGui.QPalette.HighlightedText))
        paint_markup(painter, rect, markup, font=opt.font, alignment=opt.displayAlignment, palette=palette)


class ScientificNameComboBox(QtWidgets.QComboBox):
    """Show a scientific markup role in both popup rows and the closed label."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setItemDelegate(ScientificNameDelegate(self))

    def paintEvent(self, event) -> None:
        if self.currentIndex() < 0:
            super().paintEvent(event)
            return
        index = self.model().index(
            self.currentIndex(), self.modelColumn(), self.rootModelIndex(),
        )
        if not index.isValid() or not index.data(SCIENTIFIC_NAME_HTML_ROLE):
            super().paintEvent(event)
            return

        painter = QtWidgets.QStylePainter(self)
        try:
            option = QtWidgets.QStyleOptionComboBox()
            self.initStyleOption(option)
            option.currentText = ""
            painter.drawComplexControl(QtWidgets.QStyle.ComplexControl.CC_ComboBox, option)
            rect = self.style().subControlRect(
                QtWidgets.QStyle.ComplexControl.CC_ComboBox,
                option,
                QtWidgets.QStyle.SubControl.SC_ComboBoxEditField,
                self,
            )
            item_option = QtWidgets.QStyleOptionViewItem()
            item_option.initFrom(self)
            item_option.rect = rect
            item_option.widget = self
            item_option.font = self.font()
            item_option.palette = option.palette
            item_option.displayAlignment = (
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self.itemDelegate().paint(painter, item_option, index)
        finally:
            painter.end()


class ScientificNameHeaderView(QtWidgets.QHeaderView):
    """Paint explicitly marked header text as rich text without changing its data."""

    def paintSection(self, painter, rect, logical_index) -> None:
        markup = self.model().headerData(
            logical_index, self.orientation(), SCIENTIFIC_NAME_HTML_ROLE,
        )
        if not markup:
            super().paintSection(painter, rect, logical_index)
            return

        option = QtWidgets.QStyleOptionHeader()
        self.initStyleOption(option)
        self.initStyleOptionForIndex(option, logical_index)
        option.rect = rect
        option.text = ""
        style = self.style()
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_Header, option, painter, self)
        text_rect = style.subElementRect(
            QtWidgets.QStyle.SubElement.SE_HeaderLabel, option, self,
        )
        paint_markup(painter, text_rect, markup, font=self.font(),
                     alignment=option.textAlignment, palette=option.palette)


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


def setup_scientific_validator(
    line_edit: QtWidgets.QLineEdit,
    *,
    minimum: float = 0.0,
    maximum: float = sys.float_info.max,
) -> QDoubleValidator:
    """Install the full binary64 scientific-text admission range.

    The validator only governs text admission. Callers still perform the
    semantic positive/finite check when constructing their typed value.
    """

    validator = QDoubleValidator(float(minimum), float(maximum), 323, line_edit)
    validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
    line_edit.setValidator(validator)
    return validator
