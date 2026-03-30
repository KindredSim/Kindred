"""Data & Targets composition tab for the fitting window."""

from __future__ import annotations

from PySide6 import QtWidgets
from PySide6.QtCore import Signal


class DataTargetsTab(QtWidgets.QWidget):
    """Thin composition container hosting Data, Targets & Weights, and Initial Conditions as subtabs."""

    subtabChanged = Signal(int)

    def __init__(
        self,
        *,
        data_tab: QtWidgets.QWidget,
        targets_weights_tab: QtWidgets.QWidget,
        ic_panel: QtWidgets.QWidget,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.data_tab = data_tab
        self.targets_weights_tab = targets_weights_tab
        self.ic_panel = ic_panel
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._subtabs = QtWidgets.QTabWidget(self)
        self._subtabs.setObjectName("global_fit_data_targets_subtabs")
        self._subtabs.addTab(self.data_tab, "Data")
        self._subtabs.addTab(self.targets_weights_tab, "Targets & Weights")
        self._subtabs.addTab(self.ic_panel, "Initial Conditions")
        layout.addWidget(self._subtabs)
        self._subtabs.currentChanged.connect(self.subtabChanged)

    @property
    def subtabs(self) -> QtWidgets.QTabWidget:
        return self._subtabs
