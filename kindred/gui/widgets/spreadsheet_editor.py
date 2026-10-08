"""Native editing of detached experimental worksheets."""
from __future__ import annotations

import ironcalc
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.datasets.units import MEASUREMENT_UNIT_DISPLAY, TIME_UNIT_DISPLAY, parse_measurement_unit, parse_time_unit
from kindred.core.quantity_units import parse_unit
from kindred.gui.display_name_policy import literal_tooltip
from kindred.gui.spreadsheet_formulas import formula_at_row
from kindred.gui.spreadsheet_worker import SpreadsheetWorker
from kindred.gui.ui_helpers import (
    SCIENTIFIC_NAME_HTML_ROLE, ScientificNameComboBox, data_unit_labels, ignore_wheel, paint_markup,
    set_bounded_label_text,
)

# Requests that change nothing the user could lose.
_READS = frozenset({"initializing", "view", "copy", "outputs", "select"})


class SpreadsheetSelectionDialog(QtWidgets.QDialog):
    def __init__(self, records, selected_id, *, parent):
        super().__init__(parent)
        self.setWindowTitle("Create from Data")
        self.resize(430, 360)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Select datasets to open as worksheets:", self))
        self._list = QtWidgets.QListWidget(self)
        self._list.setAccessibleName("Datasets to open")
        for record in records:
            item = QtWidgets.QListWidgetItem(record.display_name, self._list)
            item.setData(QtCore.Qt.UserRole, record.dataset_id)
            item.setCheckState(QtCore.Qt.Checked if record.dataset_id == selected_id else QtCore.Qt.Unchecked)
        layout.addWidget(self._list)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        self._open = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self._open.setText("Open Editor")
        self._list.itemChanged.connect(lambda: self._open.setEnabled(bool(self.selected_ids())))
        self._open.setEnabled(bool(self.selected_ids()))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_ids(self):
        return tuple(self._list.item(index).data(QtCore.Qt.UserRole)
                     for index in range(self._list.count())
                     if self._list.item(index).checkState() == QtCore.Qt.Checked)


class _SheetModel(QtCore.QAbstractTableModel):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.info = {"rows": 0, "columns": []}
        self.cells = {}

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else self.info["rows"] + 1

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.info["columns"])

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        display, content, error = self.cells.get((index.row() + 1, index.column() + 1), ("", "", False))
        if role == QtCore.Qt.DisplayRole:
            return display
        if role == QtCore.Qt.EditRole:
            return self.entry(index.row() + 1, index.column() + 1)
        if role == QtCore.Qt.ToolTipRole:
            return content
        if role == QtCore.Qt.ForegroundRole and error:
            return QtGui.QBrush(QtGui.QColor("#c53b32"))
        if role == QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignCenter)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Vertical:
            return str(section + 1) if role == QtCore.Qt.DisplayRole else None
        if role == QtCore.Qt.DisplayRole:
            return ironcalc.column_name_from_number(section + 1)
        if role == QtCore.Qt.ToolTipRole:
            column = self.info["columns"][section]
            tip = f"{column["name"]}\n{_unit_labels(column)[0]}\nDouble-click the name to edit."
            if column["formula"] is not None and column["unit"] is None:
                tip += ("\n\nThe unit of the formula could not be inferred. Its numbers are calculated from the "
                        "scientific values (M, s, K, J/mol, m³) of the columns it uses and are used as they are. "
                        "To give it a unit, take plain numbers from a column that has a unit, or delete the "
                        "formula including = and use Correct Unit Declaration… on this column.")
            return literal_tooltip(tip)
        return None

    def entry(self, row, column):
        """The text a cell edit starts from: the row's formula or the loaded number."""
        formula = self.info["columns"][column - 1]["formula"]
        if formula is not None:
            return formula_at_row(formula, row)
        cell = self.cells.get((row, column))
        return None if cell is None else cell[1]

    def flags(self, index):
        flags = super().flags(index)
        loaded = (index.row() + 1, index.column() + 1) in self.cells
        editable = loaded and self.editor._editing_available
        return flags | QtCore.Qt.ItemIsEditable if editable else flags

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if role != QtCore.Qt.EditRole or not index.isValid():
            return False
        cell = index.row() + 1, index.column() + 1
        if cell not in self.cells:
            return False
        return self.editor._commit_entry(*cell, str(value))


class _CellDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, dialog):
        super().__init__(dialog._table)
        self.active_editor = None
        self._initialized = False

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        editor.setAlignment(QtCore.Qt.AlignCenter)
        self.active_editor = editor
        self._initialized = False
        return editor

    def setEditorData(self, editor, index):
        # Replies update the display cache, not unfinished input.
        if not self._initialized:
            super().setEditorData(editor, index)
            self._initialized = True

    def destroyEditor(self, editor, index):
        if self.active_editor is editor:
            self.active_editor = None
        super().destroyEditor(editor, index)

    def finish_edit(self):
        editor = self.active_editor
        if editor is not None:
            self.commitData.emit(editor)
            self.closeEditor.emit(editor, QtWidgets.QAbstractItemDelegate.NoHint)

    def discard_edit(self):
        if self.active_editor is not None:
            self.closeEditor.emit(self.active_editor, QtWidgets.QAbstractItemDelegate.RevertModelCache)


def _unit_labels(column):
    """Plain text and scientific markup for a column's unit."""
    if column["unit"] is None:
        text = "Unspecified" if column["declared"] else "Choose Unit"
        return text, text
    return data_unit_labels(parse_unit(column["unit"]))


class _SheetHeader(QtWidgets.QHeaderView):
    """Worksheet addresses and, horizontally, editable measurement metadata."""

    def __init__(self, orientation, dialog):
        super().__init__(orientation, dialog._table)
        self.dialog = dialog
        self._press = None
        self._edit = None
        self._input = None
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setSectionsClickable(True)
        self.setHighlightSections(True)
        self.setSectionsMovable(False)
        self.sectionResized.connect(self._position_editor)
        self.geometriesChanged.connect(self._position_editor)

    def _letter_height(self):
        return self.fontMetrics().height() + 8

    def _metadata_height(self):
        return self.fontMetrics().height() + 14

    def sizeHint(self):
        size = super().sizeHint()
        if self.orientation() == QtCore.Qt.Horizontal:
            size.setHeight(self._letter_height() + 2 * self._metadata_height())
        return size

    def sectionSizeFromContents(self, section):
        size = super().sectionSizeFromContents(section)
        if self.orientation() == QtCore.Qt.Horizontal:
            column = self.dialog._model.info["columns"][section]
            metrics = self.fontMetrics()
            size.setWidth(max(metrics.horizontalAdvance(column["name"]) + 34,
                              metrics.horizontalAdvance(_unit_labels(column)[0]) + 36,
                              metrics.horizontalAdvance("−9.99999e−308") + 22))
            size.setHeight(self._letter_height() + 2 * self._metadata_height())
        return size

    def _metadata_rects(self, section):
        left, width = self.sectionViewportPosition(section), self.sectionSize(section)
        height = self._metadata_height()
        return (QtCore.QRect(left + 5, self._letter_height(), width - 10, height),
                QtCore.QRect(left + 4, self._letter_height() + height + 2, width - 8, height - 4))

    def paintSection(self, painter, rect, section):
        if self.orientation() == QtCore.Qt.Vertical:
            super().paintSection(painter, rect, section)
            return
        painter.save()
        painter.setClipRect(rect)
        column = self.dialog._model.info["columns"][section]
        option = QtWidgets.QStyleOptionHeader()
        self.initStyleOption(option)
        option.rect, option.section, option.text = rect, section, ""
        self.style().drawControl(QtWidgets.QStyle.CE_Header, option, painter, self)
        option.rect = QtCore.QRect(rect.left(), rect.top(), rect.width(), self._letter_height())
        role = " · Time" if self.dialog._model.info["timed"] and section == 0 else " · fx" if column["formula"] is not None else ""
        option.text = ironcalc.column_name_from_number(section + 1) + role
        option.textAlignment = QtCore.Qt.AlignCenter
        if self.dialog._table.selectionModel().isColumnSelected(section):
            option.state |= QtWidgets.QStyle.State_On
        self.style().drawControl(QtWidgets.QStyle.CE_Header, option, painter, self)
        painter.setFont(self.font())
        name_rect, unit_rect = self._metadata_rects(section)
        if name_rect.contains(self.viewport().mapFromGlobal(QtGui.QCursor.pos())):
            painter.fillRect(name_rect, self.palette().brush(QtGui.QPalette.AlternateBase))
        painter.setPen(self.palette().color(QtGui.QPalette.Link))
        painter.drawText(name_rect.adjusted(1, 0, -22, 0), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft,
                         self.fontMetrics().elidedText(column["name"], QtCore.Qt.ElideRight, name_rect.width() - 23))
        painter.drawText(name_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, "✎")
        painter.setPen(self.palette().color(QtGui.QPalette.Text))
        markup = _unit_labels(column)[1]
        if column["formula"] is not None:
            paint_markup(painter, unit_rect, markup, font=self.font(), alignment=QtCore.Qt.AlignCenter,
                         palette=self.palette())
        else:
            combo = QtWidgets.QStyleOptionComboBox()
            combo.initFrom(self)
            combo.rect = unit_rect
            self.style().drawComplexControl(QtWidgets.QStyle.CC_ComboBox, combo, painter, self)
            field = self.style().subControlRect(QtWidgets.QStyle.CC_ComboBox, combo,
                                                QtWidgets.QStyle.SC_ComboBoxEditField, self)
            paint_markup(painter, field, markup, font=self.font(),
                         alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, palette=self.palette())
        painter.restore()

    def paintEvent(self, event):
        super().paintEvent(event)
        coordinate = self.dialog._insertion_coordinate(self.orientation())
        if coordinate is not None:
            painter = QtGui.QPainter(self.viewport())
            painter.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.Highlight), 3))
            if self.orientation() == QtCore.Qt.Horizontal:
                painter.drawLine(coordinate, 0, coordinate, self.height())
            else:
                painter.drawLine(0, coordinate, self.width(), coordinate)

    def begin_edit(self, sheet, column, field, correction=False):
        spec = self.dialog._sheets[sheet]["columns"][column]
        if field == "unit" and spec["formula"] is not None:
            self.dialog._feedback.setText("This unit follows from the column formula.")
            return
        self._edit = sheet, column, field, correction
        timed = self.dialog._sheets[sheet]["timed"] and column == 0
        if field == "name":
            control = QtWidgets.QLineEdit(spec["name"], self.viewport())
            control.setAccessibleName("Column name")
            control.setToolTip(literal_tooltip(spec["name"]))
            control.selectAll()
        else:
            control = ScientificNameComboBox(self.viewport())
            ignore_wheel(control)
            control.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
            control.setAccessibleName("Column unit")
            control.setToolTip("Correct declaration: numbers stay unchanged." if correction else
                               "Changing an established unit converts its values. The first unit declares their meaning.")
            control.addItem(_unit_labels(spec)[0] if spec["unit"] is None else "Choose Unit", None)
            choices = list(TIME_UNIT_DISPLAY if timed else MEASUREMENT_UNIT_DISPLAY)
            if spec["unit"] is not None:
                # The column's unit takes the place of a preset displayed the same way (uM as µM).
                own = data_unit_labels(parse_unit(spec["unit"]))[1]
                choices = [spec["unit"] if data_unit_labels(parse_unit(choice))[1] == own else choice for choice in choices]
                if spec["unit"] not in choices:
                    choices.append(spec["unit"])
            for choice in choices:
                label, markup = data_unit_labels(parse_unit(choice))
                control.addItem(label, choice)
                control.setItemData(control.count() - 1, markup, SCIENTIFIC_NAME_HTML_ROLE)
            control.addItem("Custom Expression…", "custom")
            control.setCurrentIndex(max(0, control.findData(spec["unit"])))
            control.activated.connect(lambda: self._unit_selected(control))
        self._input = control
        control.installEventFilter(self)
        self._position_editor()
        control.show()
        control.setFocus()
        if field == "unit":
            control.showPopup()

    def _unit_selected(self, control):
        if control.currentData() == "custom":
            control.setEditable(True)
            control.setCompleter(None)
            control.setEditText("")
            control.lineEdit().installEventFilter(self)
            control.setFocus()
        else:
            self.finish_edit()

    def _position_editor(self, *_args):
        if self._edit is not None:
            _sheet, column, field, _correction = self._edit
            name, unit = self._metadata_rects(column)
            self._input.setGeometry(name if field == "name" else unit)

    def finish_edit(self, commit=True):
        if self._edit is None:
            return True
        sheet, column, field, correction = self._edit
        control = self._input
        if commit:
            spec = self.dialog._sheets[sheet]["columns"][column]
            if field == "name":
                value = control.text()
            else:
                # A listed unit submits its unit, not its label, even after Custom
                # expression made the box editable.
                listed = control.itemData(control.findText(control.currentText()))
                value = listed if listed not in (None, "custom") else (
                    control.currentText().strip() if control.isEditable() else control.currentData())
            if value is not None and value != spec[field]:
                if field == "unit":
                    try:
                        (parse_time_unit if self.dialog._sheets[sheet]["timed"] and column == 0 else parse_measurement_unit)(value)
                    except ValueError as exc:
                        self.dialog._feedback.setText(str(exc))
                        control.setFocus()
                        return False
                if not self.dialog._submit_edit("column", (column, field, value, correction), sheet=sheet):
                    return False
        control.blockSignals(True)
        self._input = self._edit = None
        control.hide()
        control.deleteLater()
        return True

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Escape):
                if self.finish_edit(event.key() != QtCore.Qt.Key_Escape):
                    self.dialog._table.setFocus()
                return True
        return super().eventFilter(watched, event)

    def mouseDoubleClickEvent(self, event):
        section = self.logicalIndexAt(event.position().toPoint())
        if self.orientation() == QtCore.Qt.Horizontal and section >= 0:
            name, _unit = self._metadata_rects(section)
            if name.contains(event.position().toPoint()):
                self.dialog._begin_header_edit(section, "name")
                return
        super().mouseDoubleClickEvent(event)

    def _handle_at(self, point):
        """The section whose resize handle is at point, as QHeaderView decides, or -1."""
        position = point.x() if self.orientation() == QtCore.Qt.Horizontal else point.y()
        section = self.logicalIndexAt(position)
        if section < 0:
            return -1
        grip = self.style().pixelMetric(QtWidgets.QStyle.PM_HeaderGripMargin, None, self)
        start = self.sectionViewportPosition(section)
        if position < start + grip:
            section -= 1
        elif position <= start + self.sectionSize(section) - grip:
            return -1
        return section if section >= 0 and self.sectionResizeMode(section) == QtWidgets.QHeaderView.Interactive else -1

    def mousePressEvent(self, event):
        # One press means, in order: resize, unit choice, moving a selection, else Qt's selection.
        point = event.position().toPoint()
        section = self.logicalIndexAt(point)
        if event.button() != QtCore.Qt.LeftButton or section < 0 or self._handle_at(point) >= 0:
            return super().mousePressEvent(event)
        horizontal = self.orientation() == QtCore.Qt.Horizontal
        if horizontal and self._metadata_rects(section)[1].contains(point):
            self.dialog._begin_header_edit(section, "unit")
            return
        if not self.dialog._finish_edits():
            return
        selection = self.dialog._table.selectionModel()
        selected = selection.isColumnSelected(section) if horizontal else selection.isRowSelected(section)
        source = self.dialog._area()
        if selected and not event.modifiers() and self.dialog._can_move(self.orientation(), source):
            self._press = point, source, section
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.viewport().update()
        if self._press is not None:
            point, source, _section = self._press
            if self.dialog._gesture is None and (event.position().toPoint() - point).manhattanLength() >= QtWidgets.QApplication.startDragDistance():
                self.dialog._begin_move(self.orientation(), source)
            if self.dialog._gesture is not None:
                self.dialog._update_gesture(self.viewport().mapToGlobal(event.position().toPoint()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._press is not None:
            _point, _source, section = self._press
            self._press = None
            if self.dialog._gesture is None:
                # A click without a drag selects just that column or row.
                table = self.dialog._table
                (table.selectColumn if self.orientation() == QtCore.Qt.Horizontal else table.selectRow)(section)
                return
            self.dialog._update_gesture(self.viewport().mapToGlobal(event.position().toPoint()))
            self.dialog._finish_gesture(True)
            return
        super().mouseReleaseEvent(event)

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self._position_editor()


class _SheetTable(QtWidgets.QTableView):
    def __init__(self, dialog):
        super().__init__(dialog)
        self.dialog = dialog

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self.viewport())
        color = self.palette().color(QtGui.QPalette.Highlight)
        handle = self.dialog._fill_handle()
        if not handle.isNull():
            painter.fillRect(handle, color)
        gesture = self.dialog._gesture
        if gesture is not None and gesture["kind"] == "fill":
            top, left, _bottom, right = gesture["source"]
            first = self.visualRect(self.model().index(top - 1, left - 1))
            last = self.visualRect(self.model().index(gesture["destination"] - 1, right - 1))
            painter.setPen(QtGui.QPen(color, 2, QtCore.Qt.DashLine))
            painter.drawRect(first.united(last).adjusted(1, 1, -1, -1))
        painter.setPen(QtGui.QPen(color, 3))
        for orientation in (QtCore.Qt.Horizontal, QtCore.Qt.Vertical):
            coordinate = self.dialog._insertion_coordinate(orientation)
            if coordinate is not None:
                if orientation == QtCore.Qt.Horizontal:
                    painter.drawLine(coordinate, 0, coordinate, self.viewport().height())
                else:
                    painter.drawLine(0, coordinate, self.viewport().width(), coordinate)


class SpreadsheetEditorDialog(QtWidgets.QDialog):
    outputsReady = QtCore.Signal(object)
    settled = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, records, names, *, parent):
        super().__init__(parent)
        self.setWindowTitle("Create from Data")
        self.resize(1040, 690)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self._sheets = []
        self._sheet = 0
        self._command = "initializing"
        self._closing = False
        self._settled = False
        # Requests made while a reply is outstanding, run in order.
        self._queue = []
        self._result_area = None
        self._formula_edit = None
        self._gesture = None
        self._gesture_point = None
        self._projecting = False
        self._can_undo = self._can_redo = False
        self._view_area = None
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self._name = QtWidgets.QLineEdit(self)
        self._name.setAccessibleName("New dataset name")
        form.addRow("New dataset", self._name)
        layout.addLayout(form)
        self._toolbar = QtWidgets.QToolBar(self)
        layout.addWidget(self._toolbar)
        self._undo = self._action("Undo", lambda: self._request("undo"), QtGui.QKeySequence.Undo)
        self._redo = self._action("Redo", lambda: self._request("redo"), QtGui.QKeySequence.Redo)
        self._toolbar.addSeparator()
        self._action("Copy", self._copy, QtGui.QKeySequence.Copy)
        self._action("Paste", self._paste, QtGui.QKeySequence.Paste)
        self._action("Fill Down", self._fill, "Ctrl+D")
        self._clear = self._action("Clear Cells", self._clear_cells, "Delete", toolbar=False)
        self._toolbar.addSeparator()
        self._rows_menu = QtWidgets.QMenu(self)
        self._rows_menu.addAction("Insert Rows", lambda: self._rows(False))
        self._rows_menu.addAction("Delete Rows", lambda: self._rows(True))
        self._columns_menu = QtWidgets.QMenu(self)
        self._columns_menu.addAction("Insert Column", lambda: self._columns("insert"))
        self._columns_menu.addAction("Add Column", lambda: self._columns("append"))
        self._columns_menu.addAction("Correct Unit Declaration…", self._correct_declaration)
        self._columns_menu.addAction("Delete Columns", lambda: self._columns("delete"))
        for text, menu in (("Rows", self._rows_menu), ("Columns", self._columns_menu)):
            button = QtWidgets.QToolButton(self)
            button.setText(text)
            button.setMenu(menu)
            button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
            self._toolbar.addWidget(button)
        formula_row = QtWidgets.QHBoxLayout()
        self._address = QtWidgets.QLabel("", self)
        self._address.setMinimumWidth(65)
        formula_row.addWidget(self._address)
        formula_row.addWidget(QtWidgets.QLabel("fx", self))
        self._formula = QtWidgets.QLineEdit(self)
        self._formula.setAccessibleName("Cell contents")
        self._formula.setToolTip("A number, or a formula such as =B5*D5 that computes its whole column from cells of the same row.\n"
                                 "Plain numbers in formulas are scientific values (M, s, K, J/mol, m³).\n"
                                 "Use ln(x) for natural logarithms and log10(x) for base 10.")
        formula_row.addWidget(self._formula)
        layout.addLayout(formula_row)
        self._table = _SheetTable(self)
        self._table.setAccessibleName("Temporary experimental worksheet")
        self._model = _SheetModel(self)
        self._table.setModel(self._model)
        self._header = _SheetHeader(QtCore.Qt.Horizontal, self)
        self._table.setHorizontalHeader(self._header)
        self._table.setVerticalHeader(_SheetHeader(QtCore.Qt.Vertical, self))
        self._header.setMinimumSectionSize(80)
        self._delegate = _CellDelegate(self)
        self._table.setItemDelegate(self._delegate)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ContiguousSelection)
        self._table.verticalHeader().setDefaultSectionSize(25)
        self._table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self._table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self._table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self._table, 1)
        self._tabs = QtWidgets.QTabBar(self)
        layout.addWidget(self._tabs)
        # One fixed line: longer messages elide and show in full on hover.
        self._feedback = QtWidgets.QLabel(self)
        set_bounded_label_text(self._feedback, "Opening worksheets…", max_width=16777215)
        layout.addWidget(self._feedback)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self._cancel = QtWidgets.QPushButton("Cancel", self)
        self._review = QtWidgets.QPushButton("Review and Create…", self)
        self._cancel.setAutoDefault(False)
        self._review.setAutoDefault(False)
        footer.addWidget(self._cancel)
        footer.addWidget(self._review)
        layout.addLayout(footer)
        self._view_timer = QtCore.QTimer(self)
        self._view_timer.setSingleShot(True)
        self._view_timer.timeout.connect(self._read_view)
        self._drag_timer = QtCore.QTimer(self)
        self._drag_timer.setInterval(50)
        self._drag_timer.timeout.connect(self._scroll_gesture)
        self._cancel_drag = self._action("Cancel drag", lambda: self._finish_gesture(False), "Escape", toolbar=False)
        self._cancel_drag.setShortcutContext(QtCore.Qt.WindowShortcut)
        self._table.verticalScrollBar().valueChanged.connect(self._schedule_view)
        self._table.horizontalScrollBar().valueChanged.connect(self._schedule_view)
        self._table.verticalHeader().sectionResized.connect(self._schedule_view)
        self._table.horizontalHeader().sectionResized.connect(self._schedule_view)
        self._table.selectionModel().currentChanged.connect(self._selection_changed)
        self._table.selectionModel().selectionChanged.connect(lambda *_args: self._table.viewport().update())
        self._table.installEventFilter(self)
        self._table.viewport().installEventFilter(self)
        self._formula.installEventFilter(self)
        self._name.installEventFilter(self)
        self._tabs.currentChanged.connect(self._select_sheet)
        self._formula.textEdited.connect(self._start_formula_edit)
        self._formula.editingFinished.connect(lambda: self._finish_formula(True))
        self._name.editingFinished.connect(self._commit_name)
        self._cancel.clicked.connect(self.reject)
        self._review.clicked.connect(lambda: self._request("outputs"))
        self._worker = SpreadsheetWorker(records, names, parent=self)
        self._worker.result.connect(self._receive)
        self._worker.failed.connect(self._failure)
        self._worker.finished.connect(self._worker_finished)
        self._availability()
        self._worker.start()

    @property
    def busy(self):
        return self._command is not None or self._closing

    @property
    def _editing_available(self):
        return self._command in (None, "view") and not self._queue and not self._closing

    @property
    def _pending(self):
        return tuple(operation for operation in (self._command, *(entry[0] for entry in self._queue)) if operation)

    @property
    def has_changes(self):
        """Whether closing now would discard something the user did."""
        draft = self._formula_edit is not None and self._formula.text() != self._formula_edit[3]
        cell = self._delegate.active_editor is not None and self._delegate.active_editor.isModified()
        header = self._header._input
        header = header is not None and (header.isModified() if isinstance(header, QtWidgets.QLineEdit)
                                         else header.isEditable() and header.lineEdit().isModified())
        return (self._can_undo or any(operation not in _READS for operation in self._pending)
                or draft or cell or header or self._name.isModified())

    @property
    def application_close_settled(self):
        return self._settled

    def cancel_for_application_close(self):
        self._close_session()

    def _action(self, text, callback, shortcut, *, toolbar=True):
        action = QtGui.QAction(text, self)
        action.setShortcut(QtGui.QKeySequence(shortcut))
        action.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        action.triggered.connect(callback)
        self.addAction(action)
        if toolbar:
            self._toolbar.addAction(action)
        return action

    def _availability(self):
        # Viewport reads never block commands; commands that capture worksheet
        # coordinates wait for pending changes, while Undo and Redo queue in order.
        ready = self._editing_available and self._gesture is None
        for action in (*self._toolbar.actions(), *self._rows_menu.actions(), *self._columns_menu.actions()):
            action.setEnabled(ready)
        self._name.setEnabled(ready)
        self._tabs.setEnabled(bool(self._sheets) and not self._closing and self._command != "outputs" and self._gesture is None)
        self._formula.setEnabled(ready)
        # Read after disabling fields: a focused field's focus loss can queue an edit.
        pending = self._pending
        # Focus loss can submit an edit before a mouse click reaches Review.
        # Keep the button available to record that click while the edit finishes.
        self._review.setEnabled(bool(self._sheets) and not self._closing and "outputs" not in pending)
        history = not self._closing and self._gesture is None
        self._undo.setEnabled(history and (self._can_undo or any(operation not in _READS for operation in pending)))
        self._redo.setEnabled(history and (self._can_redo or "undo" in pending))
        self._clear.setEnabled(ready)
        self._cancel_drag.setEnabled(self._gesture is not None)
        self._table.viewport().update()

    def _area(self):
        selection = self._table.selectionModel().selection()
        if selection:
            area = selection[0]
            return area.top() + 1, area.left() + 1, area.bottom() + 1, area.right() + 1
        index = self._table.currentIndex()
        row, column = max(0, index.row()) + 1, max(0, index.column()) + 1
        return row, column, row, column

    def _request(self, operation, args=(), *, sheet=None):
        """Run a worksheet command after active edits, behind any outstanding work."""
        if self._closing or not self._finish_edits():
            return False
        sheet = self._sheet if sheet is None else sheet
        if self.busy:
            self._queue.append((operation, args, sheet))
            self._availability()
        else:
            self._send(operation, args, sheet)
        return True

    def _submit_edit(self, operation, args, *, sheet=None):
        """Submit a captured edit; only a viewport read may be outstanding."""
        if self._closing:
            return False
        # Capture the destination now: selection may move before the reply.
        sheet = self._sheet if sheet is None else sheet
        if self._command == "view":
            self._queue.append((operation, args, sheet))
            self._availability()
            return True
        if self.busy:
            return False
        self._send(operation, args, sheet)
        return True

    def _send(self, operation, args, sheet):
        self._command = operation
        self._availability()
        if operation != "view":
            self._feedback.setText("Calculating…" if operation != "outputs" else "Preparing numeric measurements…")
        self._worker.submit(operation, sheet, args)

    def _receive(self, operation, data, error):
        if self._closing:
            return
        self._command = None
        if operation == "failed":
            self._failure(error)
            return
        if operation != "view":
            self._feedback.setText(error)
        if operation == "view":
            self._model.cells.update(data)
            top, left, bottom, right = self._view_area
            self._model.dataChanged.emit(self._model.index(top - 1, left - 1), self._model.index(bottom - 1, right - 1))
            self._selection_changed()
        elif operation == "copy":
            QtWidgets.QApplication.clipboard().setText(data)
        elif operation == "outputs":
            self._availability()
            if not error:
                self.outputsReady.emit(data)
            if self._closing:
                return
        else:
            # Unchanged columns keep their widths wherever structural changes move them.
            previous = {spec["name"]: (spec, self._table.columnWidth(column))
                        for column, spec in enumerate(self._model.info["columns"])}
            self._sheets, self._can_undo, self._can_redo, shown = data
            if operation == "ready":
                blocker = QtCore.QSignalBlocker(self._tabs)
                for info in self._sheets:
                    tab = self._tabs.addTab(info["sheet"])
                    self._tabs.setTabToolTip(tab, literal_tooltip(f"Worksheet: {info['sheet']}\nSource dataset: {info['source']}"))
                del blocker
            if shown is not None and shown != self._sheet:
                # Undo and Redo show the worksheet they change.
                self._sheet = shown
                with QtCore.QSignalBlocker(self._tabs):
                    self._tabs.setCurrentIndex(shown)
                previous = None
            if operation == "formula" and not error:
                sheet, _row, column, _entry = self._formula_edit
                self._formula_edit = None
                spec = self._sheets[sheet]["columns"][column - 1]
                if spec["formula"] is not None and spec["unit"] is None:
                    self._feedback.setText(f"Column {ironcalc.column_name_from_number(column)} has no unit. It is "
                                           "calculated from scientific values (M, s, …). Hover its unit for how to set one.")
            area = None
            if self._result_area is not None and operation == self._result_area[0]:
                if not error:
                    area = self._result_area[1]
                self._result_area = None
            self._project_sheet(area)
            self._fit_columns(previous)
        if error:
            self._queue.clear()
            with QtCore.QSignalBlocker(self._tabs):
                self._tabs.setCurrentIndex(self._sheet)
        self._availability()
        if error and operation == "formula":
            self._feedback.setText(f"{error}{'' if error.endswith('.') else '.'} Press Escape to discard the entry.")
            self._formula.setReadOnly(False)
            self._formula.setFocus()
        while self._queue and not self.busy:
            operation, args, sheet = self._queue.pop(0)
            if operation == "select":
                self._select_sheet(sheet)
            else:
                self._send(operation, args, sheet)
        if not self.busy:
            self._availability()
            self._schedule_view()

    def _project_sheet(self, area=None):
        area = self._area() if area is None else area
        self._projecting = True
        self._model.beginResetModel()
        self._model.info = self._sheets[self._sheet]
        self._model.cells = {}
        self._model.endResetModel()
        self._view_area = None
        self._name.setText(self._model.info["name"])
        if self._model.columnCount():
            top, left, bottom, right = area
            top, bottom = min(top, self._model.rowCount()) - 1, min(bottom, self._model.rowCount()) - 1
            left, right = min(left, self._model.columnCount()) - 1, min(right, self._model.columnCount()) - 1
            self._table.setCurrentIndex(self._model.index(top, left))
            self._table.selectionModel().select(QtCore.QItemSelection(self._model.index(top, left), self._model.index(bottom, right)), QtCore.QItemSelectionModel.ClearAndSelect)
        self._projecting = False
        self._selection_changed()

    def _select_sheet(self, sheet):
        if sheet < 0 or self._closing or not self._sheets:
            return
        # Tab clicks can keep a line edit focused. Commit against the old
        # worksheet before projecting or queueing the requested worksheet.
        with QtCore.QSignalBlocker(self._tabs):
            self._tabs.setCurrentIndex(self._sheet)
        if not self._finish_edits():
            return
        with QtCore.QSignalBlocker(self._tabs):
            self._tabs.setCurrentIndex(sheet)
        if self.busy:
            # Only the latest tab choice waits; the tab bar shows it meanwhile.
            self._queue = [entry for entry in self._queue if entry[0] != "select"]
            self._queue.append(("select", None, sheet))
            self._availability()
            return
        self._sheet = sheet
        self._project_sheet()
        self._fit_columns()
        self._schedule_view()

    def _selection_changed(self, *_args):
        if self._projecting or not self._sheets or self._formula_edit is not None:
            return
        index = self._table.currentIndex()
        if not index.isValid():
            self._address.clear()
            self._formula.clear()
            self._formula.setReadOnly(True)
            return
        row, column = index.row() + 1, index.column() + 1
        entry = self._model.entry(row, column)
        self._address.setText(f"{ironcalc.column_name_from_number(column)}{row}")
        # A number is editable here only once its current value has loaded.
        self._formula.setReadOnly(entry is None)
        self._formula.setText(entry or "")

    def _fit_columns(self, previous=None):
        """Fit columns to their headers; given earlier definitions and widths by name, only those that changed."""
        for column, spec in enumerate(self._model.info["columns"]):
            kept = None if previous is None else previous.get(spec["name"])
            if kept is not None and kept[0] == spec:
                self._table.setColumnWidth(column, kept[1])
            else:
                self._table.setColumnWidth(column, min(260, self._header.sectionSizeFromContents(column).width()))
        self._header._position_editor()

    def _begin_header_edit(self, column, field, correction=False):
        if not self._editing_available or self._gesture is not None:
            return
        if not self._finish_edits() or not self._editing_available:
            return
        self._header.begin_edit(self._sheet, column, field, correction)

    def _schedule_view(self, *_args):
        self._view_timer.start(0)

    def _read_view(self):
        if self.busy or not self._sheets or not self._model.columnCount():
            return
        viewport = self._table.viewport()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return
        top, left = self._table.rowAt(0), self._table.columnAt(0)
        if top < 0 or left < 0:
            return
        bottom = self._table.rowAt(viewport.height() - 1)
        right = self._table.columnAt(viewport.width() - 1)
        if bottom < 0:
            bottom = self._model.rowCount() - 1
        if right < 0:
            right = self._model.columnCount() - 1
        area = top + 1, left + 1, bottom + 1, right + 1
        if area != self._view_area:
            self._view_area = area
            self._send("view", area, self._sheet)

    def _commit_entry(self, row, column, text):
        """Commit a cell entry through the formula bar, which keeps it if refused."""
        self._formula_edit = self._sheet, row, column, self._model.entry(row, column)
        self._address.setText(f"{ironcalc.column_name_from_number(column)}{row}")
        self._formula.setText(text)
        return self._finish_formula(True)

    def _start_formula_edit(self, *_args):
        if self._formula_edit is None:
            index = self._table.currentIndex()
            row, column = index.row() + 1, index.column() + 1
            self._formula_edit = self._sheet, row, column, self._model.entry(row, column)

    def _correct_declaration(self):
        index = self._table.currentIndex()
        if index.isValid():
            self._begin_header_edit(index.column(), "unit", correction=True)

    def _finish_formula(self, commit):
        if self._formula_edit is None or "formula" in self._pending:
            return True
        if commit:
            sheet, row, column, entry = self._formula_edit
            text = self._formula.text()
            if text == entry:
                self._formula_edit = None
                return True
            if not self._submit_edit("formula", (row, column, text), sheet=sheet):
                return False
        else:
            self._formula_edit = None
            self._selection_changed()
            self._table.setFocus()
        return True

    def eventFilter(self, watched, event):
        if watched in (self._formula, self._name) and event.type() == QtCore.QEvent.FocusIn:
            if not self._header.finish_edit():
                return True
            if watched is self._formula and not self._formula.isReadOnly():
                self._start_formula_edit()
        if watched is self._formula and event.type() == QtCore.QEvent.KeyPress:
            if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Escape):
                self._finish_formula(event.key() != QtCore.Qt.Key_Escape)
                return True
        if watched is self._table.viewport() and event.type() == QtCore.QEvent.Resize:
            self._schedule_view()
        if watched is self._table.viewport():
            if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                if not self._finish_edits():
                    return True
                if self._fill_handle().contains(event.position().toPoint()):
                    self._gesture = {"kind": "fill", "sheet": self._sheet, "source": self._area(), "destination": self._area()[2]}
                    self._table.viewport().grabMouse()
                    self._update_gesture(self._table.viewport().mapToGlobal(event.position().toPoint()))
                    self._drag_timer.start()
                    self._availability()
                    return True
            elif self._gesture is not None and self._gesture["kind"] == "fill":
                if event.type() == QtCore.QEvent.MouseMove:
                    self._update_gesture(self._table.viewport().mapToGlobal(event.position().toPoint()))
                    return True
                if event.type() == QtCore.QEvent.MouseButtonRelease:
                    self._update_gesture(self._table.viewport().mapToGlobal(event.position().toPoint()))
                    self._finish_gesture(True)
                    return True
        return super().eventFilter(watched, event)

    def event(self, event):
        if event.type() == QtCore.QEvent.WindowDeactivate and self._gesture is not None:
            self._finish_gesture(False)
        if event.type() == QtCore.QEvent.FontChange:
            self._fit_columns()
        return super().event(event)

    def _commit_name(self):
        # Clear first: submitting disables the field, and its focus loss commits again.
        if self._name.isModified():
            self._name.setModified(False)
            if not self._submit_edit("name", (self._name.text(),)):
                self._name.setModified(True)

    def _finish_edits(self):
        if not self._header.finish_edit():
            return False
        self._delegate.finish_edit()
        if not self._finish_formula(True):
            return False
        self._commit_name()
        return True

    def _copy(self):
        self._request("copy", self._area())

    def _paste(self):
        self._request("paste", (self._area(), QtWidgets.QApplication.clipboard().text()))

    def _fill(self):
        top, left, bottom, right = self._area()
        # A whole-column selection includes the entry row below the data.
        self._request("fill", ((top, left, top, right), min(bottom, self._model.info["rows"])))

    def _fill_handle(self):
        if not self._sheets or self._formula_edit is not None or not self._editing_available:
            return QtCore.QRect()
        _top, _left, bottom, right = self._area()
        if bottom > self._model.info["rows"]:
            return QtCore.QRect()
        corner = self._table.visualRect(self._model.index(bottom - 1, right - 1)).bottomRight()
        return QtCore.QRect(corner.x() - 4, corner.y() - 4, 8, 8)

    def _can_move(self, orientation, source):
        """Whether a selected row or column range can be dragged to a new position."""
        if not self._editing_available or self._formula_edit is not None:
            return False
        _top, left, bottom, _right = source
        if orientation == QtCore.Qt.Horizontal:
            return not (self._model.info["timed"] and left == 1)
        return bottom <= self._model.info["rows"]

    def _begin_move(self, orientation, source):
        if not self._can_move(orientation, source):
            return
        top, left, bottom, right = source
        horizontal = orientation == QtCore.Qt.Horizontal
        start, end = (left, right) if horizontal else (top, bottom)
        self._gesture = {"kind": "columns" if horizontal else "rows", "sheet": self._sheet,
                         "source": source, "start": start, "count": end - start + 1, "destination": start}
        header = self._header if horizontal else self._table.verticalHeader()
        header.viewport().grabMouse()
        self._drag_timer.start()
        self._availability()

    def _update_gesture(self, point):
        gesture = self._gesture
        if gesture is None:
            return
        self._gesture_point = point
        local = self._table.viewport().mapFromGlobal(point)
        if gesture["kind"] == "fill":
            y = max(0, min(local.y(), self._table.viewport().height() - 1))
            row = self._table.rowAt(y)
            row = self._model.info["rows"] if row < 0 else row + 1
            gesture["destination"] = max(gesture["source"][2], min(row, self._model.info["rows"]))
        else:
            horizontal = gesture["kind"] == "columns"
            header = self._header if horizontal else self._table.verticalHeader()
            coordinate = local.x() if horizontal else local.y()
            section = header.logicalIndexAt(max(0, coordinate))
            extent = self._model.columnCount() if horizontal else self._model.info["rows"]
            if section < 0:
                destination = extent + 1
            else:
                destination = section + 1 + (coordinate >= header.sectionViewportPosition(section) + header.sectionSize(section) / 2)
            first = 2 if horizontal and self._model.info["timed"] else 1
            gesture["destination"] = max(first, min(destination, extent + 1))
        self._paint_gesture()

    def _paint_gesture(self):
        self._table.viewport().update()
        self._header.viewport().update()
        self._table.verticalHeader().viewport().update()

    def _insertion_coordinate(self, orientation):
        if self._gesture is None or self._gesture["kind"] != ("columns" if orientation == QtCore.Qt.Horizontal else "rows"):
            return None
        header = self._header if orientation == QtCore.Qt.Horizontal else self._table.verticalHeader()
        section = self._gesture["destination"] - 1
        if section == header.count():
            return header.sectionViewportPosition(section - 1) + header.sectionSize(section - 1)
        return header.sectionViewportPosition(section)

    def _scroll_gesture(self):
        if self._gesture is None or self._gesture_point is None:
            return
        viewport = self._table.viewport()
        point = viewport.mapFromGlobal(self._gesture_point)
        horizontal = self._gesture["kind"] == "columns"
        coordinate, extent = (point.x(), viewport.width()) if horizontal else (point.y(), viewport.height())
        direction = 1 if coordinate >= extent - 24 else -1 if coordinate < 24 and self._gesture["kind"] != "fill" else 0
        if direction:
            bar = self._table.horizontalScrollBar() if horizontal else self._table.verticalScrollBar()
            bar.setValue(bar.value() + direction * (30 if horizontal else self._table.verticalHeader().defaultSectionSize()))
            self._update_gesture(self._gesture_point)

    def _finish_gesture(self, commit):
        gesture, self._gesture = self._gesture, None
        self._gesture_point = None
        self._header._press = None
        self._table.verticalHeader()._press = None
        self._drag_timer.stop()
        grabber = QtWidgets.QWidget.mouseGrabber()
        if grabber in (self._table.viewport(), self._header.viewport(), self._table.verticalHeader().viewport()):
            grabber.releaseMouse()
        self._paint_gesture()
        self._availability()
        if not commit or gesture is None:
            return
        top, left, bottom, right = gesture["source"]
        destination = gesture["destination"]
        if gesture["kind"] == "fill":
            if destination <= bottom:
                return
            operation, args = "fill", (gesture["source"], destination)
            area = top, left, destination, right
        else:
            start, count = gesture["start"], gesture["count"]
            if start <= destination <= start + count:
                return
            delta = destination - start - (count if destination > start else 0)
            operation, args = "move_" + gesture["kind"], (start, count, delta)
            area = (top, left + delta, bottom, right + delta) if gesture["kind"] == "columns" else (top + delta, left, bottom + delta, right)
        self._result_area = operation, area
        if not self._request(operation, args, sheet=gesture["sheet"]):
            self._result_area = None

    def _clear_cells(self):
        self._request("clear", self._area())

    def _rows(self, delete):
        top, _left, bottom, _right = self._area()
        count = bottom - top + 1
        if delete:
            count = min(count, self._model.info["rows"] - top + 1)
        if count > 0:
            self._request("delete_rows" if delete else "insert_rows", (top, count))

    def _columns(self, operation):
        _top, left, _bottom, right = self._area()
        timed = self._model.info["timed"]
        if operation == "append":
            left = self._model.columnCount() + 1
        elif timed and operation == "insert":
            left = max(2, left)
        # The session admits deletions, refusing ones that include Recorded Time.
        count = max(0, right - left + 1) if operation == "delete" else 1
        if count:
            self._request("delete_columns" if operation == "delete" else "insert_columns", (left, count))

    def _context_menu(self, point):
        menu = QtWidgets.QMenu(self)
        for action in self._toolbar.actions():
            if not isinstance(action, QtWidgets.QWidgetAction):
                menu.addAction(action)
        menu.addAction(self._clear)
        menu.addMenu(self._rows_menu).setText("Rows")
        menu.addMenu(self._columns_menu).setText("Columns")
        menu.exec(self._table.viewport().mapToGlobal(point))
        menu.deleteLater()

    def finish_creation(self):
        self._close_session()

    def reject(self):
        """Cancel, Escape and the window close ask before discarding edits."""
        if self._closing:
            return
        if QtWidgets.QApplication.mouseButtons() != QtCore.Qt.NoButton:
            # Escape during a native resize or drag leaves the editor open.
            return
        if self.has_changes:
            dialog = QtWidgets.QMessageBox(self)
            dialog.setWindowTitle("Create from Data")
            dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            dialog.setText("Discard the temporary worksheets and close the editor?")
            discard = dialog.addButton("Discard Worksheets", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
            dialog.setDefaultButton(dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel))
            dialog.exec()
            if dialog.clickedButton() is not discard or self._closing:
                return
        self._close_session()

    def _close_session(self):
        if self._closing:
            return
        self._finish_gesture(False)
        self._header.finish_edit(False)
        self._closing = True
        self._delegate.discard_edit()
        self._view_timer.stop()
        self.setEnabled(False)
        self._feedback.setText("Closing editor…")
        self._worker.request_shutdown()
        if self._worker.isFinished():
            self._worker_finished()

    def closeEvent(self, event):
        if self._worker.isFinished():
            event.accept()
        else:
            event.ignore()
            self.reject()

    def _failure(self, message):
        if not self._closing:
            self.failed.emit(message)
            self._close_session()

    def _worker_finished(self):
        if self._settled:
            return
        self.done(QtWidgets.QDialog.Rejected)
        self._settled = True
        self.settled.emit()
