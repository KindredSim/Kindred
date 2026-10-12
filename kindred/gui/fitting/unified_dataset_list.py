"""Unified dataset list for the fitting window master/detail layout."""

from __future__ import annotations

from typing import Optional

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal

from kindred.gui.display_name_policy import DATASET_LIST_LABEL_MAX_CHARS, compact_dataset_label, literal_tooltip


class UnifiedDatasetList(QtWidgets.QWidget):
    """Checkable dataset list for unified dataset navigation."""

    currentDatasetChanged = Signal(str)
    datasetIncludeChanged = Signal(int, str, bool)

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
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._list.currentItemChanged.connect(self._on_current_item_changed)
        self._list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list, stretch=1)

        # Track previous check states to detect toggle vs. other itemChanged triggers.
        self._check_states: dict[int, Qt.CheckState] = {}
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(
        self,
        dataset_entries: list,
        *,
        emit_selection: bool = True,
    ) -> None:
        old_id = self.selected_dataset_id()
        normalized: list[tuple[str, str, str, bool]] = []
        for entry in dataset_entries:
            ds_id = entry["id"]
            compact = compact_dataset_label(entry["label"], max_chars=DATASET_LIST_LABEL_MAX_CHARS)
            normalized.append((ds_id, compact.display, compact.full, entry["include"]))

        existing_ids = tuple(
            str(self._list.item(index).data(Qt.UserRole) or "").strip()
            for index in range(self._list.count())
            if self._list.item(index) is not None
        )
        incoming_ids = tuple(item[0] for item in normalized)
        self._list.blockSignals(True)
        try:
            if existing_ids != incoming_ids:
                self._list.clear()
                self._check_states.clear()
                for ds_id, display, full, include in normalized:
                    item = QtWidgets.QListWidgetItem(display)
                    item.setToolTip(f"{full}\nDataset ID: {ds_id}")
                    item.setData(Qt.UserRole, ds_id)
                    item.setData(Qt.UserRole + 1, full)
                    item.setFlags(
                        Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
                    )
                    item.setCheckState(Qt.Checked if include else Qt.Unchecked)
                    self._list.addItem(item)
                    self._check_states[self._list.count() - 1] = item.checkState()
            else:
                # Keep the QListWidgetItems (and their selection/current index)
                # when the roster is stable.  Accepted include/label changes
                # only update the affected item in place.
                for row, (_ds_id, display, full, include) in enumerate(normalized):
                    item = self._list.item(row)
                    if item is None:
                        continue
                    item.setText(display)
                    item.setToolTip(f"{full}\nDataset ID: {_ds_id}")
                    item.setData(Qt.UserRole + 1, full)
                    state = Qt.Checked if include else Qt.Unchecked
                    if item.checkState() != state:
                        item.setCheckState(state)
                    self._check_states[row] = state
            # Re-select previous active dataset if still present, otherwise first.
            if old_id and self._select_by_id(old_id):
                pass
            elif self._list.count() > 0:
                self._list.setCurrentRow(0)
        finally:
            self._list.blockSignals(False)

        new_id = self.selected_dataset_id()
        if emit_selection and new_id != old_id:
            self.currentDatasetChanged.emit(new_id or "")

    def selected_dataset_id(self) -> Optional[str]:
        item = self._list.currentItem()
        if item is None:
            return None
        ds_id = str(item.data(Qt.UserRole) or "").strip()
        return ds_id if ds_id else None

    def set_data_edit_issues(self, issues: dict[str, str | None]) -> None:
        """Annotate existing rows without changing identity or inclusion."""

        for row in range(self._list.count()):
            item = self._list.item(row)
            dataset_id = item.data(Qt.UserRole)
            full = item.data(Qt.UserRole + 1)
            display = compact_dataset_label(full, max_chars=DATASET_LIST_LABEL_MAX_CHARS).display
            issue = (issues[dataset_id] or "Pending data changes") if dataset_id in issues else ""
            prefix = ("Data error" if issues[dataset_id] else "Data pending") + " — " if dataset_id in issues else ""
            item.setText(prefix + display)
            detail = f"{full}\n{issue}" if issue else full
            item.setData(Qt.AccessibleTextRole, detail)
            item.setToolTip(literal_tooltip(f"{detail}\nDataset ID: {dataset_id}"))

    def dataset_display_name(self, dataset_id: str | None = None) -> str | None:
        """Return the full bounded-row source label, never the stable ID role."""

        requested = str(dataset_id or self.selected_dataset_id() or "").strip()
        if not requested:
            return None
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item is None:
                continue
            if str(item.data(Qt.UserRole) or "").strip() == requested:
                return str(item.data(Qt.UserRole + 1))
        return None

    def set_running_state(self, running: bool) -> None:
        """Preserve the historical disabled/locked running-state visual."""

        self._running = bool(running)
        self._list.setEnabled(not self._running)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_current_item_changed(
        self,
        current: QtWidgets.QListWidgetItem | None,
        _previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
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
