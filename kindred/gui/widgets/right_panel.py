# kindred/gui/widgets/right_panel.py
"""Right panel with Data tab."""

from __future__ import annotations

from PySide6 import QtWidgets

from kindred.gui.widgets.data_manager import DataManagerPanel
from kindred.gui.widgets.dataset_overlay_panel import DatasetOverlayPanel

__all__ = ["RightPanelTabbed"]


class RightPanelTabbed(QtWidgets.QWidget):
    """
    Right panel with the fixed Data and Plot Overlays tabs.

    Contains exactly two tabs:
    - Data: DataManagerPanel for loading CSV files
    - Plot Overlays: DatasetOverlayPanel for experimental plot membership
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        data_panel: DataManagerPanel,
        overlay_panel: DatasetOverlayPanel,
    ):
        """
        Initialize right panel with tabs.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs)

        if not isinstance(data_panel, DataManagerPanel):
            raise TypeError("RightPanelTabbed requires the retained DataManagerPanel.")
        if not isinstance(overlay_panel, DatasetOverlayPanel):
            raise TypeError(
                "RightPanelTabbed requires the retained DatasetOverlayPanel."
            )
        self._data_manager = data_panel
        self._data_manager.setObjectName("dataPanel")
        self._tabs.addTab(self._data_manager, "Data")
        self._overlay_panel = overlay_panel
        self._overlay_panel.setObjectName("datasetOverlayPanel")
        self._tabs.addTab(self._overlay_panel, "Plot Overlays")

    def select_plot_overlays_page(self) -> None:
        """Show the existing Plot Overlays page without changing its state."""

        self._tabs.setCurrentWidget(self._overlay_panel)
