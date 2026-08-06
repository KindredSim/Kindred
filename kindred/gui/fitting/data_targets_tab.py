"""Data and Targets composition tab for the fitting window (unified master/detail layout)."""

from __future__ import annotations

from typing import Optional

from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from kindred.gui.fitting.unified_dataset_list import UnifiedDatasetList


class DataTargetsTab(QtWidgets.QWidget):
    """Unified master/detail container: dataset list left, stacked panels right."""

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

        # LEFT: unified dataset list + Fit data panel, scrollable
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.unified_list = UnifiedDatasetList(parent=left_panel)
        left_layout.addWidget(self.unified_list, stretch=1)

        left_layout.addWidget(self.data_settings_panel)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_scroll.setWidget(left_panel)
        splitter.addWidget(left_scroll)

        # RIGHT: scroll area with species table only
        scroll = QtWidgets.QScrollArea(splitter)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        container = QtWidgets.QWidget()
        container.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        detail_layout = QtWidgets.QVBoxLayout(container)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(10)

        detail_layout.addWidget(self.species_table)

        scroll.setWidget(container)
        splitter.addWidget(scroll)

        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([240, 620])

        layout.addWidget(splitter)

        # Override MinimumExpanding policies so the scroll area can scroll.
        sp = self.species_table.sizePolicy()
        if sp.verticalPolicy() == QtWidgets.QSizePolicy.MinimumExpanding:
            sp.setVerticalPolicy(QtWidgets.QSizePolicy.Preferred)
            self.species_table.setSizePolicy(sp)

        # Ensure species table doesn't collapse inside scroll area.
        self.species_table.setMinimumHeight(300)

        # Wire unified list selection to both panels.
        self.unified_list.currentDatasetChanged.connect(self._on_unified_dataset_selected)

    def _on_unified_dataset_selected(self, dataset_id: str) -> None:
        ds_id = str(dataset_id or "").strip()
        if not ds_id:
            return
        if ds_id == self._current_detail_dataset_id:
            return
        self._current_detail_dataset_id = ds_id
        select_dataset = getattr(self.data_settings_panel, "select_dataset", None)
        if callable(select_dataset):
            select_dataset(
                ds_id,
                self.unified_list.dataset_display_name(ds_id),
            )
        self.species_table.load_for_dataset(ds_id)
