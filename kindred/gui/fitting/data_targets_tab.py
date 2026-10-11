"""Data and Targets composition tab for the fitting window (unified master/detail layout)."""

from __future__ import annotations

from typing import Optional

from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from kindred.gui.fitting.unified_dataset_list import UnifiedDatasetList


class DataTargetsTab(QtWidgets.QWidget):
    """Keep dataset navigation beside the selected dataset's editors."""

    def __init__(
        self,
        *,
        data_settings_panel: QtWidgets.QWidget,
        species_table: QtWidgets.QWidget,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # The caller supplies the panel that is meant to be visible here.  In
        # the restored composition it is a panel-only DataTab, so no complete
        # DataTab is constructed and hidden behind this master/detail view.
        self.data_settings_panel = data_settings_panel
        self.species_table = species_table
        self._current_detail_dataset_id: Optional[str] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QtWidgets.QSplitter(Qt.Horizontal, self)

        left = QtWidgets.QWidget(splitter)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        self.unified_list = UnifiedDatasetList(parent=left)
        left_layout.addWidget(self.unified_list, stretch=1)
        data_scroll = QtWidgets.QScrollArea(left)
        data_scroll.setWidgetResizable(True)
        data_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        data_scroll.setWidget(self.data_settings_panel)
        data_scroll.setMinimumWidth(
            self.data_settings_panel.minimumSizeHint().width()
            + data_scroll.verticalScrollBar().sizeHint().width()
        )
        left_layout.addWidget(data_scroll, stretch=1)
        splitter.addWidget(left)
        splitter.addWidget(self.species_table)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([240, 620])
        layout.addWidget(splitter)

        # Wire unified list selection to both panels.
        self.unified_list.currentDatasetChanged.connect(self._on_unified_dataset_selected)

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
