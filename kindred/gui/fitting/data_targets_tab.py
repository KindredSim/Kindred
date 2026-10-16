"""Data and Targets composition tab for the fitting window."""

from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from kindred.gui.fitting.data_tab import DataTab
from kindred.gui.fitting.unified_dataset_list import UnifiedDatasetList
from kindred.gui.fitting.unified_species_table import UnifiedSpeciesTable


class DataTargetsTab(QtWidgets.QWidget):
    """Datasets and the selected dataset's settings above its species table."""

    def __init__(
        self,
        *,
        data_settings_panel: DataTab,
        species_table: UnifiedSpeciesTable,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.data_settings_panel = data_settings_panel
        self.species_table = species_table
        self._current_detail_dataset_id: Optional[str] = None
        self._split_initialized = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QtWidgets.QSplitter(Qt.Vertical, self)
        top = QtWidgets.QSplitter(Qt.Horizontal, splitter)
        top.setChildrenCollapsible(False)
        self.unified_list = UnifiedDatasetList(parent=top)
        top.addWidget(self.unified_list)
        # Dataset weighting sits with the dataset's other settings; the
        # species table still owns its edits.
        for label, control in self.species_table.weighting_settings:
            self.data_settings_panel.add_group(label, control)
        top.addWidget(self.data_settings_panel)
        # As in the import dialog, the list keeps its width and the
        # settings take the rest.
        top.setStretchFactor(0, 0)
        top.setStretchFactor(1, 1)
        splitter.addWidget(top)
        splitter.addWidget(self.species_table)
        splitter.setChildrenCollapsible(False)
        # The species table takes the height the settings do not need.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self._splitter = splitter
        self._top_splitter = top
        layout.addWidget(splitter)

        # Wire unified list selection to both panels.
        self.unified_list.currentDatasetChanged.connect(self._on_unified_dataset_selected)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if not self._split_initialized:
            self._split_initialized = True
            # After the first layout, when the settings know their wrapped height.
            QtCore.QTimer.singleShot(0, self._initialize_split)

    def _initialize_split(self) -> None:
        width = self._top_splitter.width()
        listed = min(max(220, self.unified_list.content_width()), width // 2)
        self._top_splitter.setSizes([listed, width - listed])
        # The settings' own minimum wins; the list gets at least a few rows.
        top = max(self.data_settings_panel.minimumHeight(), 180)
        self._splitter.setSizes([top, max(1, self._splitter.height() - top)])

    def _on_unified_dataset_selected(self, dataset_id: str) -> None:
        ds_id = str(dataset_id or "").strip()
        if not ds_id:
            return
        if ds_id == self._current_detail_dataset_id:
            return
        self._current_detail_dataset_id = ds_id
        self.data_settings_panel.select_dataset(
            ds_id,
            self.unified_list.dataset_display_name(ds_id),
        )
        self.species_table.load_for_dataset(ds_id)
