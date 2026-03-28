from __future__ import annotations

from typing import Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.batch_initial_conditions import BatchInitialConditionsStore

__all__ = [
    "BatchInitialConditionsTableModel",
    "BatchInitialConditionsTableView",
]


_INVALID_BRUSH = QtGui.QBrush(QtGui.QColor(255, 210, 210))


class BatchInitialConditionsTableModel(QtCore.QAbstractTableModel):
    showMembershipChanged = QtCore.Signal()
    sliderEditTargetsChanged = QtCore.Signal()

    def __init__(self, store: BatchInitialConditionsStore, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._store = store
        self._invalid: set[Tuple[int, str]] = set()
        self._slider_edit_target_set_ids: list[str] = []
        self._focused_effective_edit_target_set_id = ""

    def store(self) -> BatchInitialConditionsStore:
        return self._store

    def _base_column_count(self) -> int:
        return self._store.column_count()

    def edit_target_column(self) -> int:
        return int(self._base_column_count())

    def show_column(self) -> int:
        return int(self._base_column_count()) + 1

    def is_control_column(self, column: int) -> bool:
        return int(column) in (self.edit_target_column(), self.show_column())

    def slider_edit_target_set_ids(self) -> list[str]:
        valid_ids = set(self._store.set_ids())
        return [str(set_id) for set_id in self._slider_edit_target_set_ids if str(set_id) in valid_ids]

    def set_slider_edit_target_set_ids(self, set_ids: Sequence[str]) -> bool:
        valid_ids = set(self._store.set_ids())
        normalized: list[str] = []
        for set_id in set_ids or ():
            set_id_s = str(set_id or "").strip()
            if not set_id_s or set_id_s in normalized or set_id_s not in valid_ids:
                continue
            normalized.append(set_id_s)
        before_ids = self.slider_edit_target_set_ids()
        if normalized == before_ids:
            return False
        self._slider_edit_target_set_ids = list(normalized)
        changed_ids = set(before_ids) | set(normalized)
        changed_rows = [
            int(row)
            for row, set_id in enumerate(self._store.set_ids())
            if str(set_id) in changed_ids
        ]
        self._emit_checkbox_column_change(self.edit_target_column(), changed_rows)
        self.sliderEditTargetsChanged.emit()
        return True

    def focused_effective_edit_target_set_id(self) -> str:
        valid_ids = set(self._store.set_ids())
        focused_set_id = str(self._focused_effective_edit_target_set_id or "").strip()
        return focused_set_id if focused_set_id in valid_ids else ""

    def set_focused_effective_edit_target_set_id(self, set_id: str | None) -> bool:
        valid_ids = set(self._store.set_ids())
        normalized = str(set_id or "").strip()
        if normalized not in valid_ids:
            normalized = ""
        before = self.focused_effective_edit_target_set_id()
        if normalized == before:
            return False
        self._focused_effective_edit_target_set_id = normalized
        changed_ids = {sid for sid in (before, normalized) if sid}
        changed_rows = [
            int(row)
            for row, current_set_id in enumerate(self._store.set_ids())
            if str(current_set_id) in changed_ids
        ]
        self._emit_checkbox_column_change(self.edit_target_column(), changed_rows)
        return True

    def shown_set_ids(self) -> list[str]:
        return [str(set_id) for set_id in self._store.shown_set_ids()]

    def set_row_shown(self, row: int, shown: bool) -> bool:
        row_i = int(row)
        shown_b = bool(shown)
        if self._store.is_shown(row_i) == shown_b:
            return False
        self._store.set_shown(row_i, shown_b)
        self._emit_checkbox_column_change(self.show_column(), [row_i])
        self.showMembershipChanged.emit()
        return True

    def _emit_checkbox_column_change(self, column: int, rows: Sequence[int]) -> None:
        valid_rows = [int(row) for row in rows if 0 <= int(row) < self.rowCount()]
        if not valid_rows:
            return
        top_left = self.index(min(valid_rows), int(column))
        bottom_right = self.index(max(valid_rows), int(column))
        self.dataChanged.emit(
            top_left,
            bottom_right,
            [
                QtCore.Qt.CheckStateRole,
                QtCore.Qt.DisplayRole,
                QtCore.Qt.ToolTipRole,
                QtCore.Qt.ForegroundRole,
                QtCore.Qt.FontRole,
            ],
        )

    def _is_temporarily_focus_target_row(self, row: int) -> bool:
        set_id = str(self._store.set_id_for_row(int(row)) or "").strip()
        if not set_id:
            return False
        return (
            set_id == self.focused_effective_edit_target_set_id()
            and set_id not in set(self.slider_edit_target_set_ids())
        )

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return self._store.row_count()

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return self._base_column_count() + 2

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.DisplayRole):  # noqa: N802
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return "Set Name"
            species = self._store.visible_species()
            if 0 <= section - 1 < len(species):
                return f"{species[section - 1]} (M)"
            if section == self.edit_target_column():
                return "Slider"
            if section == self.show_column():
                return "Show"
            return ""
        return str(section + 1)

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:  # noqa: N802
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        if self.is_control_column(int(index.column())):
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        row = int(index.row())
        col = int(index.column())

        if role == QtCore.Qt.CheckStateRole:
            if col == self.edit_target_column():
                set_id = self._store.set_id_for_row(row)
                return QtCore.Qt.Checked if str(set_id) in self.slider_edit_target_set_ids() else QtCore.Qt.Unchecked
            if col == self.show_column():
                return QtCore.Qt.Checked if self._store.is_shown(row) else QtCore.Qt.Unchecked

        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            if col == self.edit_target_column():
                return "focus" if self._is_temporarily_focus_target_row(row) else ""
            if col == self.show_column():
                return ""
            if col == 0:
                names = self._store.set_names()
                return names[row] if 0 <= row < len(names) else ""
            species = self._store.visible_species()
            if 0 <= col - 1 < len(species):
                return self._store.get_value(row, species[col - 1])
            return ""

        if role == QtCore.Qt.TextAlignmentRole and self.is_control_column(col):
            return int(QtCore.Qt.AlignCenter)

        if role == QtCore.Qt.ToolTipRole:
            if col == 0:
                names = self._store.set_names()
                return names[row] if 0 <= row < len(names) else None
            if 1 <= col < self.edit_target_column():
                return self.data(index, QtCore.Qt.DisplayRole)
            if col == self.edit_target_column():
                if self._is_temporarily_focus_target_row(row):
                    return "Focused row is temporarily included in slider edit scope."
                return None
            return None

        if role == QtCore.Qt.ForegroundRole and col == self.edit_target_column():
            if self._is_temporarily_focus_target_row(row):
                return QtGui.QBrush(QtGui.QColor(56, 82, 132))
            return None

        if role == QtCore.Qt.FontRole and col == self.edit_target_column():
            if self._is_temporarily_focus_target_row(row):
                font = QtGui.QFont()
                font.setItalic(True)
                return font
            return None

        if role == QtCore.Qt.BackgroundRole and col > 0:
            species = self._store.visible_species()
            if 0 <= col - 1 < len(species) and (row, species[col - 1]) in self._invalid:
                return _INVALID_BRUSH
        return None

    @staticmethod
    def _coerce_checked_state(value) -> bool:
        checked_value = getattr(QtCore.Qt.Checked, "value", QtCore.Qt.Checked)
        try:
            return int(getattr(value, "value", value)) == int(checked_value)
        except Exception:
            return bool(value)

    def _set_control_check_state(self, *, row: int, column: int, checked: bool) -> bool:
        if int(column) == self.edit_target_column():
            set_id = self._store.set_id_for_row(int(row))
            next_ids = [
                sid for sid in self.slider_edit_target_set_ids()
                if str(sid) != str(set_id)
            ]
            if bool(checked):
                next_ids.append(str(set_id))
            return self.set_slider_edit_target_set_ids(next_ids)
        if int(column) == self.show_column():
            return self.set_row_shown(int(row), bool(checked))
        return False

    def setData(self, index: QtCore.QModelIndex, value, role: int = QtCore.Qt.EditRole) -> bool:  # noqa: N802
        if not index.isValid():
            return False
        row = int(index.row())
        col = int(index.column())

        if role == QtCore.Qt.CheckStateRole:
            return self._set_control_check_state(
                row=row,
                column=col,
                checked=self._coerce_checked_state(value),
            )

        if role != QtCore.Qt.EditRole:
            return False
        if col == 0:
            self._store.set_set_name(row, str(value))
            self.dataChanged.emit(index, index, [QtCore.Qt.DisplayRole])
            return True

        species = self._store.visible_species()
        if not (0 <= col - 1 < len(species)):
            return False
        sp = species[col - 1]
        self._store.set_value(row, sp, str(value))

        invalid_now = self._store.validate_numeric_cells(rows=[row])
        if (row, sp) in invalid_now:
            self._invalid.add((row, sp))
        else:
            self._invalid.discard((row, sp))
        self.dataChanged.emit(index, index, [QtCore.Qt.DisplayRole, QtCore.Qt.BackgroundRole])
        return True

    def reset_invalid(self) -> None:
        if not self._invalid:
            return
        self._invalid.clear()
        top_left = self.index(0, 1)
        bottom_right = self.index(max(0, self.rowCount() - 1), max(1, self._base_column_count() - 1))
        self.dataChanged.emit(top_left, bottom_right, [QtCore.Qt.BackgroundRole])

    def set_species(self, species_names: Sequence[str]) -> None:
        self.beginResetModel()
        try:
            self._store.set_species(species_names)
            self._invalid.clear()
        finally:
            self.endResetModel()

    def validate_rows(self, rows: Sequence[int]) -> set[Tuple[int, str]]:
        rows_int = [int(r) for r in (rows or [])]
        rows_int = [r for r in rows_int if 0 <= r < int(self.rowCount())]
        if not rows_int:
            return set()
        target_rows = sorted(set(rows_int))
        invalid_for_rows = self._store.validate_numeric_cells(rows=target_rows)

        before = set(self._invalid)
        target_set = set(target_rows)
        self._invalid = {pair for pair in self._invalid if pair[0] not in target_set}
        self._invalid.update(invalid_for_rows)
        if self._invalid == before:
            return set(invalid_for_rows)

        if self._base_column_count() > 1:
            top_left = self.index(min(target_rows), 1)
            bottom_right = self.index(max(target_rows), max(1, self._base_column_count() - 1))
            self.dataChanged.emit(top_left, bottom_right, [QtCore.Qt.BackgroundRole])
        return set(invalid_for_rows)


class BatchInitialConditionsTableView(QtWidgets.QTableView):
    pasteError = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._pressed_checkbox_index = QtCore.QModelIndex()
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.horizontalHeader().setStretchLastSection(False)
        self.setAlternatingRowColors(True)

    def setModel(self, model: Optional[QtCore.QAbstractItemModel]) -> None:  # noqa: N802
        old_model = self.model()
        if isinstance(old_model, BatchInitialConditionsTableModel):
            try:
                old_model.modelReset.disconnect(self._apply_column_presentation)
            except (TypeError, RuntimeError):
                pass
        super().setModel(model)
        if isinstance(model, BatchInitialConditionsTableModel):
            model.modelReset.connect(self._apply_column_presentation)
        self._apply_column_presentation()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        control_index = self._control_index_at_event(event)
        if control_index is not None:
            self._pressed_checkbox_index = control_index
            event.accept()
            return
        self._pressed_checkbox_index = QtCore.QModelIndex()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        control_index = self._control_index_at_event(event)
        if (
            self._pressed_checkbox_index.isValid()
            and control_index is not None
            and control_index == self._pressed_checkbox_index
            and event.button() == QtCore.Qt.LeftButton
        ):
            self._toggle_checkbox_index(control_index)
            self._pressed_checkbox_index = QtCore.QModelIndex()
            event.accept()
            return
        self._pressed_checkbox_index = QtCore.QModelIndex()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._control_index_at_event(event) is not None:
            self._pressed_checkbox_index = QtCore.QModelIndex()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if event.matches(QtGui.QKeySequence.Paste):
            self._handle_paste()
            return
        super().keyPressEvent(event)

    def _apply_column_presentation(self) -> None:
        model = self.model()
        if not isinstance(model, BatchInitialConditionsTableModel):
            return
        header = self.horizontalHeader()
        desired_order = [model.edit_target_column(), model.show_column(), 0]
        desired_order.extend(range(1, model.edit_target_column()))
        movable_before = header.sectionsMovable()
        header.setSectionsMovable(True)
        for visual_index, logical_index in enumerate(desired_order):
            current_visual = header.visualIndex(int(logical_index))
            if current_visual != visual_index:
                header.moveSection(current_visual, visual_index)
        header.setSectionsMovable(bool(movable_before))
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Interactive)
        for column in range(1, model.edit_target_column()):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Interactive)
        self._auto_fit_columns()
        for column in (model.edit_target_column(), model.show_column()):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.setColumnWidth(column, 72 if column == model.edit_target_column() else 56)

    def _auto_fit_columns(self) -> None:
        model = self.model()
        if not isinstance(model, BatchInitialConditionsTableModel):
            return
        if model.rowCount() == 0:
            return
        fm = self.horizontalHeader().fontMetrics()
        set_name_min = fm.horizontalAdvance(" Set Name ") + 20
        for column in range(0, model.edit_target_column()):
            self.resizeColumnToContents(column)
            if column == 0 and self.columnWidth(column) < set_name_min:
                self.setColumnWidth(column, set_name_min)

    def _control_index_at_event(self, event: QtGui.QMouseEvent) -> QtCore.QModelIndex | None:
        if event.button() != QtCore.Qt.LeftButton:
            return None
        model = self.model()
        if not isinstance(model, BatchInitialConditionsTableModel):
            return None
        index = self.indexAt(event.position().toPoint())
        if not index.isValid() or not model.is_control_column(int(index.column())):
            return None
        return index

    def _toggle_checkbox_index(self, index: QtCore.QModelIndex) -> bool:
        model = self.model()
        if not isinstance(model, BatchInitialConditionsTableModel) or not index.isValid():
            return False
        current_state = model.data(index, QtCore.Qt.CheckStateRole)
        next_state = (
            QtCore.Qt.Unchecked
            if current_state == QtCore.Qt.Checked
            else QtCore.Qt.Checked
        )
        return bool(model.setData(index, next_state, QtCore.Qt.CheckStateRole))

    def _handle_paste(self) -> None:
        model = self.model()
        if not isinstance(model, BatchInitialConditionsTableModel):
            return
        index = self.currentIndex()
        if not index.isValid():
            return
        clipboard = QtWidgets.QApplication.clipboard()
        text = clipboard.text() if clipboard is not None else ""
        if not text:
            return
        store = model.store()
        try:
            changed = store.apply_paste_block(
                start_row=int(index.row()),
                start_col=int(index.column()),
                text=text,
            )
        except ValueError as exc:
            self.pasteError.emit(str(exc))
            return

        if not changed:
            return

        impacted_rows = sorted({int(r) for r, _c in changed})
        model.validate_rows(impacted_rows)

        top_left = model.index(min(r for r, _c in changed), min(c for _r, c in changed))
        bottom_right = model.index(max(r for r, _c in changed), max(c for _r, c in changed))
        model.dataChanged.emit(top_left, bottom_right, [QtCore.Qt.DisplayRole, QtCore.Qt.BackgroundRole])
