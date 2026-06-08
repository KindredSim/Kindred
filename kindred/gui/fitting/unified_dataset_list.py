"""Unified dataset list for the fitting window master/detail layout."""

from __future__ import annotations

from typing import Optional

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from kindred.gui.display_name_policy import DATASET_LIST_LABEL_MAX_CHARS, compact_dataset_label


class _ValidationForegroundDelegate(QtWidgets.QStyledItemDelegate):
    """Paints item text using ForegroundRole color, bypassing stylesheet overrides."""

    def paint(self, painter, option, index):
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        fg_variant = index.data(Qt.ForegroundRole)
        if isinstance(fg_variant, QtGui.QBrush) and fg_variant.style() != Qt.BrushStyle.NoBrush:
            text = opt.text
            opt.text = ""
            style = option.widget.style() if option.widget else QtWidgets.QApplication.style()
            style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, option.widget)
            painter.save()
            if opt.state & QtWidgets.QStyle.StateFlag.State_Selected:
                painter.setPen(opt.palette.color(QtGui.QPalette.ColorRole.HighlightedText))
            else:
                painter.setPen(fg_variant.color())
            text_rect = style.subElementRect(QtWidgets.QStyle.SubElement.SE_ItemViewItemText, opt, option.widget)
            alignment = opt.displayAlignment
            metrics = QtGui.QFontMetrics(opt.font)
            elided = metrics.elidedText(text, Qt.ElideRight, text_rect.width())
            painter.drawText(text_rect, alignment, elided)
            painter.restore()
        else:
            super().paint(painter, option, index)


class UnifiedDatasetList(QtWidgets.QWidget):
    """Checkable dataset list with Add/Remove buttons for unified navigation."""

    currentDatasetChanged = Signal(str)
    datasetIncludeChanged = Signal(int, str, bool)
    addRequested = Signal()
    removeRequested = Signal(list)

    def __init__(self, *, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QtWidgets.QLabel("Datasets")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        self._list = QtWidgets.QListWidget(self)
        self._list.setObjectName("global_fit_unified_dataset_list")
        self._list.setUniformItemSizes(True)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._list.setItemDelegate(_ValidationForegroundDelegate(self._list))
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._list.currentItemChanged.connect(self._on_current_item_changed)
        self._list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list, stretch=1)

        button_row = QtWidgets.QHBoxLayout()
        self._add_button = QtWidgets.QPushButton("Add\u2026")
        self._add_button.setObjectName("global_fit_unified_datasets_add")
        self._add_button.clicked.connect(self.addRequested)
        self._remove_button = QtWidgets.QPushButton("Remove")
        self._remove_button.setObjectName("global_fit_unified_datasets_remove")
        self._remove_button.setEnabled(False)
        self._remove_button.clicked.connect(self._on_remove_clicked)
        button_row.addWidget(self._add_button)
        button_row.addWidget(self._remove_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        # Track previous check states to detect toggle vs. other itemChanged triggers.
        self._check_states: dict[int, Qt.CheckState] = {}
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(self, dataset_entries: list) -> None:
        old_id = self.selected_dataset_id()
        self._list.blockSignals(True)
        self._list.clear()
        self._check_states.clear()
        for entry in dataset_entries:
            ds_id = str(entry.get("id") or "").strip()
            label = str(entry.get("label", "") or "").strip() or str(ds_id)
            compact = compact_dataset_label(label, max_chars=DATASET_LIST_LABEL_MAX_CHARS)
            include = entry.get("include", True)
            item = QtWidgets.QListWidgetItem(compact.display)
            item.setToolTip(compact.full)
            item.setData(Qt.UserRole, ds_id)
            item.setData(Qt.UserRole + 1, compact.full)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Checked if include else Qt.Unchecked)
            self._list.addItem(item)
            self._check_states[self._list.count() - 1] = item.checkState()
        # Re-select previous active dataset if still present, otherwise first.
        if old_id and self._select_by_id(old_id):
            pass
        elif self._list.count() > 0:
            self._list.setCurrentRow(0)
        self._list.blockSignals(False)

        self._remove_button.setEnabled(self._list.count() > 0 and not self._running)

        new_id = self.selected_dataset_id()
        if new_id != old_id:
            self.currentDatasetChanged.emit(new_id or "")

    def selected_dataset_id(self) -> Optional[str]:
        item = self._list.currentItem()
        if item is None:
            return None
        ds_id = str(item.data(Qt.UserRole) or "").strip()
        return ds_id if ds_id else None

    def select_dataset(self, dataset_id: str) -> None:
        if dataset_id == self.selected_dataset_id():
            return
        self._list.blockSignals(True)
        found = self._select_by_id(dataset_id)
        self._list.blockSignals(False)
        if found:
            self.currentDatasetChanged.emit(dataset_id)

    def set_validation_state(self, dataset_id: str, state: str) -> None:
        ds_id = str(dataset_id or "").strip()
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is None:
                continue
            if str(item.data(Qt.UserRole) or "").strip() == ds_id:
                if state == "invalid_applied":
                    item.setBackground(QtGui.QBrush(QtGui.QColor(255, 225, 225)))
                    item.setForeground(QtGui.QBrush(QtGui.QColor(80, 0, 0)))
                elif state == "invalid_pending":
                    item.setBackground(QtGui.QBrush(QtGui.QColor(255, 245, 210)))
                    item.setForeground(QtGui.QBrush(QtGui.QColor(80, 60, 0)))
                else:
                    item.setBackground(QtGui.QBrush())
                    item.setForeground(QtGui.QBrush())
                break

    def set_remove_button_enabled(self, enabled: bool) -> None:
        self._remove_button.setEnabled(enabled)

    def set_running_state(self, running: bool) -> None:
        self._running = running
        self._add_button.setEnabled(not running)
        if running:
            self._remove_button.setEnabled(False)
        else:
            self._remove_button.setEnabled(self._list.currentItem() is not None)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_current_item_changed(
        self,
        current: QtWidgets.QListWidgetItem | None,
        _previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        self._remove_button.setEnabled(current is not None and not self._running)
        if current is None:
            return
        ds_id = str(current.data(Qt.UserRole) or "").strip()
        if ds_id:
            self.currentDatasetChanged.emit(ds_id)

    def _on_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        row = self._list.row(item)
        new_state = item.checkState()
        old_state = self._check_states.get(row)
        if old_state is not None and new_state == old_state:
            return
        self._check_states[row] = new_state
        ds_id = str(item.data(Qt.UserRole) or "").strip()
        self.datasetIncludeChanged.emit(row, ds_id, new_state == Qt.Checked)

    def _on_remove_clicked(self) -> None:
        ds_id = self.selected_dataset_id()
        if ds_id:
            self.removeRequested.emit([ds_id])

    def _select_by_id(self, dataset_id: str | None) -> bool:
        if not dataset_id:
            return False
        ds_id = str(dataset_id).strip()
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is not None and str(item.data(Qt.UserRole) or "").strip() == ds_id:
                self._list.setCurrentRow(i)
                return True
        return False
