"""Native editing of detached experimental worksheets."""
from __future__ import annotations

import ironcalc
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.datasets.units import MEASUREMENT_UNIT_DISPLAY, TIME_UNIT_DISPLAY, parse_measurement_unit, parse_time_unit
from kindred.gui.spreadsheet_worker import SpreadsheetWorker


_SPREADSHEET_MIME = "application/x-kindred-spreadsheet-copy"


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
        self._open.setText("Open editor")
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
        if role in (QtCore.Qt.EditRole, QtCore.Qt.ToolTipRole):
            return content
        if role == QtCore.Qt.ForegroundRole and error:
            return QtGui.QBrush(QtGui.QColor("#c53b32"))
        if role == QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Vertical:
            return str(section + 1)
        name, unit = self.info["columns"][section]
        label = "Choose unit" if unit is None else "Dimensionless" if unit == "1" else unit
        return f"{ironcalc.column_name_from_number(section + 1)}\n{name}\n{label}"

    def flags(self, index):
        flags = super().flags(index)
        loaded = (index.row() + 1, index.column() + 1) in self.cells
        return flags | QtCore.Qt.ItemIsEditable if loaded and self.editor._editing_available and self.editor._draft_cell is None else flags

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if role != QtCore.Qt.EditRole or not index.isValid():
            return False
        if self.editor._draft_cell is not None:
            return True
        cell = index.row() + 1, index.column() + 1
        if cell not in self.cells:
            return False
        return self.editor._submit_edit("set", (*cell, str(value)))


class _CellDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, dialog):
        super().__init__(dialog._table)
        self.dialog = dialog
        self.active_editor = None
        self._initialized = False

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        self.active_editor = editor
        self._initialized = False
        cell = self.dialog._sheet, index.row() + 1, index.column() + 1
        editor.textEdited.connect(lambda *_args: self.dialog._edit_draft(editor, cell))
        for signal in (editor.cursorPositionChanged, editor.selectionChanged):
            signal.connect(lambda *_args: self.dialog._observe_draft_cursor(editor))
        return editor

    def setEditorData(self, editor, index):
        # A viewport reply updates the display cache, not an open editor's text.
        if not self._initialized:
            super().setEditorData(editor, index)
            self._initialized = True

    def eventFilter(self, editor, event):
        if editor is self.active_editor and event.type() == QtCore.QEvent.FocusIn:
            index = self.dialog._table.currentIndex()
            self.dialog._focus_draft(editor, (self.dialog._sheet, index.row() + 1, index.column() + 1))
        if editor is self.active_editor and self.dialog._draft_cell is not None:
            if event.type() == QtCore.QEvent.FocusOut:
                self.dialog._remember_draft_cursor(editor)
                return False
            if event.type() == QtCore.QEvent.KeyPress:
                if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Escape):
                    self.dialog._finish_formula(event.key() != QtCore.Qt.Key_Escape)
                    return True
        return super().eventFilter(editor, event)

    def destroyEditor(self, editor, index):
        if self.active_editor is editor:
            self.active_editor = None
        super().destroyEditor(editor, index)

    def finish_edit(self):
        if self.dialog._draft_cell is not None:
            self.dialog._finish_formula(True)
            return
        editor = self.active_editor
        if editor is not None:
            self.commitData.emit(editor)
            self.closeEditor.emit(editor, QtWidgets.QAbstractItemDelegate.NoHint)

    def discard_edit(self):
        if self.active_editor is not None:
            self.closeEditor.emit(self.active_editor, QtWidgets.QAbstractItemDelegate.RevertModelCache)


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
        self._created = False
        self._pending_action = None
        self._pending_sheet = None
        self._pending_edit = None
        self._control_cell = None
        self._clipboard_id = None
        self._draft_cell = None
        self._draft_cursor = (0, -1, 0)
        self._draft_commit_pending = False
        self._picking_reference = False
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
        self._action("Fill down", self._fill, "Ctrl+D")
        self._clear = self._action("Clear cells", self._clear_cells, "Delete", toolbar=False)
        self._toolbar.addSeparator()
        self._rows_menu = QtWidgets.QMenu(self)
        self._rows_menu.addAction("Insert rows", lambda: self._rows(False))
        self._rows_menu.addAction("Delete rows", lambda: self._rows(True))
        self._columns_menu = QtWidgets.QMenu(self)
        self._columns_menu.addAction("Insert column", lambda: self._columns("insert"))
        self._columns_menu.addAction("Add column", lambda: self._columns("append"))
        self._columns_menu.addAction("Delete columns", lambda: self._columns("delete"))
        for text, menu in (("Rows", self._rows_menu), ("Columns", self._columns_menu)):
            button = QtWidgets.QToolButton(self)
            button.setText(text)
            button.setMenu(menu)
            button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
            self._toolbar.addWidget(button)
        formula_row = QtWidgets.QHBoxLayout()
        self._address = QtWidgets.QLabel("A1", self)
        self._address.setMinimumWidth(65)
        formula_row.addWidget(self._address)
        formula_row.addWidget(QtWidgets.QLabel("fx", self))
        self._formula = QtWidgets.QLineEdit(self)
        self._formula.setAccessibleName("Cell value or formula")
        self._formula.setToolTip("Spreadsheet formulas: LOG(x) is base 10; LN(x) is the natural logarithm.\nCross-sheet example: ='Data 2'!B1")
        formula_row.addWidget(self._formula)
        layout.addLayout(formula_row)
        column_row = QtWidgets.QHBoxLayout()
        column_row.addWidget(QtWidgets.QLabel("Column name", self))
        self._column_name = QtWidgets.QLineEdit(self)
        self._column_name.setAccessibleName("Measurement column name")
        column_row.addWidget(self._column_name)
        column_row.addWidget(QtWidgets.QLabel("Unit", self))
        self._unit = QtWidgets.QComboBox(self)
        self._unit.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self._unit.setAccessibleName("Column unit")
        self._unit.setMinimumWidth(180)
        self._unit.setToolTip("Changing the unit leaves the numbers unchanged and changes their interpretation.")
        column_row.addWidget(self._unit)
        layout.addLayout(column_row)
        self._table = QtWidgets.QTableView(self)
        self._table.setAccessibleName("Temporary experimental worksheet")
        self._model = _SheetModel(self)
        self._table.setModel(self._model)
        self._delegate = _CellDelegate(self)
        self._table.setItemDelegate(self._delegate)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ContiguousSelection)
        self._table.horizontalHeader().setDefaultSectionSize(150)
        self._table.verticalHeader().setDefaultSectionSize(25)
        self._table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self._table, 1)
        self._tabs = QtWidgets.QTabBar(self)
        layout.addWidget(self._tabs)
        self._feedback = QtWidgets.QLabel("Opening worksheets...", self)
        self._feedback.setWordWrap(True)
        self._feedback.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(self._feedback)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self._cancel = QtWidgets.QPushButton("Cancel", self)
        self._review = QtWidgets.QPushButton("Review and create...", self)
        self._cancel.setAutoDefault(False)
        self._review.setAutoDefault(False)
        footer.addWidget(self._cancel)
        footer.addWidget(self._review)
        layout.addLayout(footer)
        self._view_timer = QtCore.QTimer(self)
        self._view_timer.setSingleShot(True)
        self._view_timer.timeout.connect(self._read_view)
        self._table.verticalScrollBar().valueChanged.connect(self._schedule_view)
        self._table.horizontalScrollBar().valueChanged.connect(self._schedule_view)
        self._table.verticalHeader().sectionResized.connect(self._schedule_view)
        self._table.horizontalHeader().sectionResized.connect(self._schedule_view)
        self._table.selectionModel().currentChanged.connect(self._selection_changed)
        self._table.installEventFilter(self)
        self._table.viewport().installEventFilter(self)
        self._formula.installEventFilter(self)
        self._tabs.currentChanged.connect(self._select_sheet)
        self._formula.textEdited.connect(lambda *_args: self._edit_draft(self._formula))
        for signal in (self._formula.cursorPositionChanged, self._formula.selectionChanged):
            signal.connect(lambda *_args: self._observe_draft_cursor(self._formula))
        self._formula.editingFinished.connect(self._commit_formula)
        self._name.editingFinished.connect(self._commit_name)
        self._column_name.editingFinished.connect(self._commit_column)
        self._unit.activated.connect(self._unit_selected)
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
        return self._command in (None, "view") and self._pending_edit is None and not self._closing

    @property
    def application_close_settled(self):
        return self._settled

    def cancel_for_application_close(self):
        self.reject()

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
        ready = not self.busy
        drafting = self._draft_cell is not None
        self._toolbar.setEnabled(ready and not drafting)
        for action in (*self._toolbar.actions(), *self._rows_menu.actions(), *self._columns_menu.actions()):
            action.setEnabled(ready and not drafting)
        for widget in (self._name, self._column_name, self._unit):
            widget.setEnabled(self._editing_available and not drafting)
        self._tabs.setEnabled(ready)
        self._formula.setEnabled(self._editing_available and not self._draft_commit_pending)
        # Focus loss can submit an edit before a mouse click reaches Review.
        # Keep the button available to record that click while the edit finishes.
        self._review.setEnabled(bool(self._sheets) and not self._closing and self._command != "outputs" and self._pending_action is None)
        self._undo.setEnabled(ready and not drafting and self._can_undo)
        self._redo.setEnabled(ready and not drafting and self._can_redo)
        self._clear.setEnabled(ready and not drafting)

    def _area(self):
        selection = self._table.selectionModel().selection()
        if selection:
            area = selection[0]
            return area.top() + 1, area.left() + 1, area.bottom() + 1, area.right() + 1
        index = self._table.currentIndex()
        row, column = max(0, index.row()) + 1, max(0, index.column()) + 1
        return row, column, row, column

    def _request(self, operation, args=()):
        if self._closing:
            return False
        if operation not in ("view", "set", "column", "name"):
            if self._pending_action is not None or not self._finish_edits():
                return False
            if self.busy:
                self._pending_action = operation, args
                self._availability()
                return True
        if self.busy:
            return False
        self._command = operation
        self._availability()
        if operation != "view":
            self._feedback.setText("Calculating..." if operation != "outputs" else "Preparing numeric measurements...")
        self._worker.submit(operation, self._sheet, args)
        return True

    def _submit_edit(self, operation, args):
        if self._closing or self._pending_edit is not None:
            return False
        if self._command == "view":
            # Capture the destination now: selection may move before the reply.
            self._pending_edit = operation, args
            self._availability()
            return True
        return self._request(operation, args)

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
            self._clipboard_id = QtCore.QUuid.createUuid().toByteArray()
            mime = QtCore.QMimeData()
            mime.setText(data)
            mime.setData(_SPREADSHEET_MIME, self._clipboard_id)
            QtWidgets.QApplication.clipboard().setMimeData(mime)
        elif operation == "outputs":
            self._availability()
            if not error:
                self.outputsReady.emit(data)
            if self._closing:
                return
        else:
            self._sheets, self._can_undo, self._can_redo = data
            if operation == "ready":
                blocker = QtCore.QSignalBlocker(self._tabs)
                for info in self._sheets:
                    tab = self._tabs.addTab(info["sheet"])
                    self._tabs.setTabToolTip(tab, info["source"])
                del blocker
            self._project_sheet()
        self._availability()
        if self._pending_edit is not None:
            pending, self._pending_edit = self._pending_edit, None
            self._request(*pending)
        elif self._draft_commit_pending:
            self._finish_formula(True)
        elif self._pending_sheet is not None:
            sheet, self._pending_sheet = self._pending_sheet, None
            self._select_sheet(sheet)
        elif self._pending_action is not None:
            pending, self._pending_action = self._pending_action, None
            self._request(*pending)
        else:
            self._schedule_view()

    def _project_sheet(self):
        area = self._area()
        previous = self._control_cell
        previous_column = self._model.info["columns"][previous[2] - 1] if previous is not None else None
        self._projecting = True
        self._model.beginResetModel()
        self._model.info = self._sheets[self._sheet]
        self._model.cells = {}
        self._model.endResetModel()
        self._view_area = None
        self._control_cell = None
        self._name.setText(self._model.info["name"])
        if self._model.columnCount():
            top, left, bottom, right = area
            top, bottom = min(top, self._model.rowCount()) - 1, min(bottom, self._model.rowCount()) - 1
            left, right = min(left, self._model.columnCount()) - 1, min(right, self._model.columnCount()) - 1
            self._table.setCurrentIndex(self._model.index(top, left))
            self._table.selectionModel().select(QtCore.QItemSelection(self._model.index(top, left), self._model.index(bottom, right)), QtCore.QItemSelectionModel.ClearAndSelect)
        self._projecting = False
        column = self._table.currentIndex().column()
        preserve_column = (
            previous is not None and (previous[0], previous[2]) == (self._sheet, column + 1)
            and previous_column == self._model.info["columns"][column]
        )
        self._selection_changed(preserve_column=preserve_column)

    def _select_sheet(self, sheet):
        if sheet < 0 or self.busy or not self._sheets:
            return
        # Tab clicks can keep a line edit focused. Commit against the old
        # worksheet before projecting the requested worksheet's controls.
        with QtCore.QSignalBlocker(self._tabs):
            self._tabs.setCurrentIndex(self._sheet)
        if self._draft_cell is not None:
            self._delegate.discard_edit()
        else:
            if not self._finish_edits():
                return
        if self.busy:
            self._pending_sheet = sheet
            return
        self._sheet = sheet
        with QtCore.QSignalBlocker(self._tabs):
            self._tabs.setCurrentIndex(sheet)
        self._project_sheet()
        self._schedule_view()

    def _selection_changed(self, *_args, preserve_column=False):
        if self._projecting or not self._sheets:
            return
        index = self._table.currentIndex()
        if not index.isValid():
            return
        row, column = index.row() + 1, index.column() + 1
        cell = self._sheet, row, column
        previous = self._control_cell
        column_changed = previous is None or (previous[0], previous[2]) != (self._sheet, column)
        if previous is not None and previous != cell:
            self._commit_formula()
            if column_changed and not self._commit_column():
                with QtCore.QSignalBlocker(self._table.selectionModel()):
                    self._table.setCurrentIndex(self._model.index(previous[1] - 1, previous[2] - 1))
                return
        self._control_cell = cell
        self._projecting = True
        if self._draft_cell is None:
            self._address.setText(f"{ironcalc.column_name_from_number(column)}{row}")
            loaded = (row, column) in self._model.cells
            was_loading = self._formula.isReadOnly()
            self._formula.setReadOnly(not loaded)
            if previous != cell or was_loading or not (self._formula.hasFocus() or self._formula.isModified()):
                self._formula.setText(self._model.cells.get((row, column), ("", "", False))[1])
        if column_changed and not preserve_column:
            name, unit = self._model.info["columns"][column - 1]
            timed = self._model.info["timed"] and column == 1
            self._column_name.setText(name)
            self._column_name.setReadOnly(timed)
            self._unit.setEditable(False)
            self._unit.clear()
            self._unit.addItem("Choose unit...", None)
            for choice in TIME_UNIT_DISPLAY if timed else MEASUREMENT_UNIT_DISPLAY:
                self._unit.addItem("Dimensionless" if choice == "1" else choice, choice)
            if unit is not None and self._unit.findData(unit) < 0:
                self._unit.addItem(unit, unit)
            self._unit.addItem("Custom expression...", "custom")
            self._unit.setCurrentIndex(max(0, self._unit.findData(unit)))
        self._projecting = False

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
            self._request("view", area)

    def _commit_formula(self):
        if self._projecting or self._closing or self._draft_cell is not None or self._control_cell is None:
            return
        _sheet, row, column = self._control_cell
        if (row, column) not in self._model.cells:
            return
        text = self._formula.text()
        if text != self._model.cells[row, column][1]:
            if self._submit_edit("set", (row, column, text)):
                self._formula.setModified(False)

    def _edit_draft(self, source, cell=None):
        if self._projecting or self._closing:
            return
        if self._draft_cell is None:
            if not source.text().startswith("="):
                return
            if cell is None:
                index = self._table.currentIndex()
                if not index.isValid():
                    return
                cell = self._sheet, index.row() + 1, index.column() + 1
            self._draft_cell = cell
            self._availability()
        if source.hasFocus():
            self._remember_draft_cursor(source)
        if source is not self._formula:
            # The formula bar retains the draft when a sheet switch destroys
            # the inline editor. Preserve its cursor and selected text too.
            with QtCore.QSignalBlocker(self._formula):
                self._formula.setText(source.text())
                start, length = source.selectionStart(), len(source.selectedText())
                if start < 0:
                    self._formula.setCursorPosition(source.cursorPosition())
                elif source.cursorPosition() == start:
                    self._formula.setSelection(start + length, -length)
                else:
                    self._formula.setSelection(start, length)

    def _remember_draft_cursor(self, source):
        self._draft_cursor = source.cursorPosition(), source.selectionStart(), len(source.selectedText())

    def _observe_draft_cursor(self, source):
        if self._draft_cell is not None and not self._projecting and not self._closing and source.hasFocus():
            self._remember_draft_cursor(source)

    def _focus_draft(self, source, cell=None):
        if self._projecting or self._closing or self._draft_commit_pending:
            return
        if self._draft_cell is None:
            self._edit_draft(source, cell)
        if self._draft_cell is None:
            return
        # Both editing locations resume the one retained formula and cursor.
        cursor, start, length = self._draft_cursor
        with QtCore.QSignalBlocker(source):
            if source is not self._formula:
                source.setText(self._formula.text())
            if start < 0:
                source.setCursorPosition(cursor)
            elif cursor == start:
                source.setSelection(start + length, -length)
            else:
                source.setSelection(start, length)

    def _finish_formula(self, commit):
        if self._draft_cell is None:
            return
        if commit and self.busy:
            self._draft_commit_pending = True
            self._delegate.discard_edit()
            self._availability()
            return
        sheet, row, column = self._draft_cell
        text = self._formula.text()
        self._projecting = True
        self._draft_cell = None
        self._draft_commit_pending = False
        self._picking_reference = False
        self._control_cell = None
        self._delegate.discard_edit()
        self._projecting = False
        if commit:
            if sheet != self._sheet:
                self._sheet = sheet
                with QtCore.QSignalBlocker(self._tabs):
                    self._tabs.setCurrentIndex(sheet)
                self._project_sheet()
            self._table.setCurrentIndex(self._model.index(row - 1, column - 1))
            self._request("set", (row, column, text))
        else:
            self._selection_changed()
            self._availability()
        self._table.setFocus()

    def _insert_reference(self):
        if self._draft_cell is None or self._draft_commit_pending or self._closing:
            return
        top, left, bottom, right = self._area()
        reference = f"{ironcalc.column_name_from_number(left)}{top}"
        if (top, left) != (bottom, right):
            reference += f":{ironcalc.column_name_from_number(right)}{bottom}"
        if self._sheet != self._draft_cell[0]:
            reference = f"{ironcalc.quote_name(self._sheets[self._sheet]['sheet'])}!{reference}"
        cursor, start, length = self._draft_cursor
        self._formula.setFocus()
        with QtCore.QSignalBlocker(self._formula):
            if start < 0:
                self._formula.setCursorPosition(cursor)
            elif cursor == start:
                self._formula.setSelection(start + length, -length)
            else:
                self._formula.setSelection(start, length)
        self._formula.insert(reference)
        self._remember_draft_cursor(self._formula)

    def eventFilter(self, watched, event):
        if watched is self._formula and event.type() == QtCore.QEvent.FocusIn:
            self._focus_draft(watched)
        if watched is self._table.viewport() and event.type() == QtCore.QEvent.Resize:
            self._schedule_view()
        if self._draft_cell is not None:
            if watched is self._formula and event.type() == QtCore.QEvent.FocusOut:
                # QLineEdit deselects text before the table's mouse press.
                self._remember_draft_cursor(self._formula)
            elif watched is self._table.viewport() and event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                self._delegate.discard_edit()
                self._picking_reference = True
            elif watched is self._table.viewport() and event.type() == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton and self._picking_reference:
                self._picking_reference = False
                QtCore.QTimer.singleShot(0, self._insert_reference)
            elif event.type() == QtCore.QEvent.KeyPress:
                if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Escape):
                    self._finish_formula(event.key() != QtCore.Qt.Key_Escape)
                    return True
                if watched is self._table:
                    self._formula.setFocus()
                    QtWidgets.QApplication.sendEvent(self._formula, event)
                    return True
        return super().eventFilter(watched, event)

    def _commit_name(self):
        if not self._projecting and self._name.text() != self._model.info.get("name"):
            self._submit_edit("name", (self._name.text(),))

    def _unit_selected(self):
        if self._unit.currentData() == "custom":
            self._unit.setEditable(True)
            self._unit.setEditText("")
            self._unit.lineEdit().editingFinished.connect(self._commit_column)
            self._unit.setFocus()
            return
        self._unit.setEditable(False)
        self._commit_column()

    def _commit_column(self):
        if self._projecting or self._closing or self._control_cell is None:
            return True
        index = self._control_cell[2] - 1
        unit = self._unit.currentText().strip() if self._unit.isEditable() else self._unit.currentData()
        if unit is not None:
            try:
                parser = parse_time_unit if self._model.info["timed"] and index == 0 else parse_measurement_unit
                parser(unit)
            except ValueError as exc:
                self._feedback.setText(str(exc))
                return False
        value = self._column_name.text(), unit
        if value != self._model.info["columns"][index]:
            self._submit_edit("column", (index, *value))
        return True

    def _finish_edits(self):
        if not self._commit_column():
            return False
        self._delegate.finish_edit()
        focused = self.focusWidget()
        if focused is not None:
            focused.clearFocus()
        self._commit_formula()
        self._commit_name()
        return True

    def _copy(self):
        self._request("copy", self._area())

    def _paste(self):
        clipboard = QtWidgets.QApplication.clipboard()
        mime = clipboard.mimeData()
        native = self._clipboard_id is not None and mime is not None and mime.data(_SPREADSHEET_MIME) == self._clipboard_id
        self._request("paste", (self._area(), clipboard.text(), native))

    def _fill(self):
        self._request("fill", self._area())

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
        elif timed:
            left = max(2, left)
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

    def finish_creation(self):
        self._created = True
        self.reject()

    def reject(self):
        if self._closing:
            return
        self._closing = True
        self._draft_cell = None
        self._draft_commit_pending = False
        self._pending_edit = None
        self._pending_action = None
        self._delegate.discard_edit()
        self._view_timer.stop()
        self.setEnabled(False)
        self._feedback.setText("Closing editor...")
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
            self.reject()

    def _worker_finished(self):
        if self._settled:
            return
        if not self._closing:
            self._failure("Spreadsheet calculation stopped.")
            return
        self.done(QtWidgets.QDialog.Accepted if self._created else QtWidgets.QDialog.Rejected)
        self._settled = True
        self.settled.emit()
