# kindred/gui/widgets/right_panel.py
"""Right panel with Data tab."""

from __future__ import annotations

from typing import Optional

from PySide6 import QtWidgets

# Direct imports required to avoid circular dependency with widgets/__init__.py
from kindred.gui.widgets.data_manager import DataManagerPanel
__all__ = ["RightPanelTabbed"]


class RightPanelTabbed(QtWidgets.QWidget):
    """
    Right panel with the Data tab.

    Contains one tab:
    - Data: DataManagerPanel for loading CSV files

    This is a simple container widget that provides access to the
    data manager.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
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

        # Data tab
        self._data_manager = DataManagerPanel()
        self._data_manager.setObjectName("dataPanel")
        self._tabs.addTab(self._data_manager, "Data")

    def get_dataset(self, name: str):
        """Resolve a dataset through the public data-manager boundary."""
        return self._data_manager.get_dataset(str(name))

