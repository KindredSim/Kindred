"""Mechanism tab strip above one shared main plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import ProjectDocument
from kindred.gui.plot_config import get_plot_panel_class

if TYPE_CHECKING:
    from kindred.gui.widgets.pyqtgraph_plot_panel_impl import PyQtGraphPlotPanel

__all__ = ["MechanismWorkspaceView"]


class MechanismWorkspaceView(QtWidgets.QWidget):
    """Physical mechanism selector and shared plot host; owns no project truth."""

    tabActivated = QtCore.Signal(str)
    createTabRequested = QtCore.Signal()
    renameTabRequested = QtCore.Signal(str, str)
    closeTabRequested = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        shared_plot: PyQtGraphPlotPanel | None = None,
        main_plot_embed_analysis_tabs: bool = True,
    ) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tab_bar = QtWidgets.QTabBar(self)
        self._tab_bar.setObjectName("mechanismTabBar")
        self._tab_bar.setDocumentMode(True)
        self._tab_bar.setExpanding(False)
        self._tab_bar.setMovable(False)
        self._tab_bar.setTabsClosable(True)
        self._tab_bar.currentChanged.connect(self._on_current_changed)
        self._tab_bar.tabBarClicked.connect(self._on_tab_clicked)
        self._tab_bar.tabBarDoubleClicked.connect(self._on_tab_double_clicked)
        self._tab_bar.tabCloseRequested.connect(self._on_close_requested)
        layout.addWidget(self._tab_bar)

        if shared_plot is None:
            plot_class = get_plot_panel_class()
            shared_plot = plot_class(
                embed_analysis_tabs=bool(main_plot_embed_analysis_tabs),
                workspace_splitter_object_name="mainPlotWorkspaceSplitter",
                enable_axis_inversion_actions=True,
                enable_reference_layer_toggle_action=True,
                enable_copy_visible_data_action=True,
            )
        self._main_plot = shared_plot
        self._main_plot.setObjectName("plotPanel")
        self._main_plot_analysis_widget = (
            self._main_plot.analysis_tabs_for_workspace(
                detached=not bool(main_plot_embed_analysis_tabs),
            )
        )
        layout.addWidget(self._main_plot, 1)

    def tab_bar(self) -> QtWidgets.QTabBar:
        return self._tab_bar

    def main_plot(self) -> PyQtGraphPlotPanel:
        return self._main_plot

    def main_plot_analysis_widget(self) -> QtWidgets.QTabWidget:
        return self._main_plot_analysis_widget

    def main_plot_workspace_splitter(self) -> QtWidgets.QSplitter:
        return self._main_plot.workspace_splitter()

    def available_analysis_surfaces(self) -> list[str]:
        widget = self.main_plot_analysis_widget()
        return [str(widget.tabText(index)) for index in range(widget.count())]

    def focus_analysis_surface(self, surface_name: str) -> bool:
        widget = self.main_plot_analysis_widget()
        target = str(surface_name).strip()
        for index in range(widget.count()):
            if widget.tabText(index) != target:
                continue
            widget.setCurrentIndex(index)
            current = widget.currentWidget()
            (current or widget).setFocus()
            return True
        return False

    def render(self, document: ProjectDocument, *, active_tab_id: str) -> None:
        document.tab(active_tab_id)
        previous = self._tab_bar.blockSignals(True)
        try:
            while self._tab_bar.count():
                self._tab_bar.removeTab(self._tab_bar.count() - 1)
            active_index = 0
            for index, tab in enumerate(document.tabs):
                tab_index = self._tab_bar.addTab(tab.name)
                self._tab_bar.setTabData(tab_index, tab.tab_id)
                self._tab_bar.setTabToolTip(tab_index, tab.name)
                if tab.tab_id == active_tab_id:
                    active_index = index
            plus_index = self._tab_bar.addTab("+")
            self._tab_bar.setTabData(plus_index, None)
            self._tab_bar.setTabToolTip(plus_index, "Create a fresh mechanism tab")
            self._tab_bar.setTabButton(
                plus_index,
                QtWidgets.QTabBar.ButtonPosition.RightSide,
                None,
            )
            self._tab_bar.setCurrentIndex(active_index)
        finally:
            self._tab_bar.blockSignals(previous)

    def set_tab_close_enabled(self, tab_id: str, enabled: bool) -> None:
        index = self._index_for_tab_id(tab_id)
        button = self._tab_bar.tabButton(
            index,
            QtWidgets.QTabBar.ButtonPosition.RightSide,
        )
        if button is not None:
            button.setEnabled(bool(enabled))

    def set_tab_dirty(self, tab_id: str, dirty: bool) -> None:
        index = self._index_for_tab_id(tab_id)
        text = self._tab_bar.tabText(index).removesuffix(" *")
        self._tab_bar.setTabText(index, f"{text} *" if dirty else text)

    def show_rename_error(self, tab_id: str, message: str) -> None:
        index = self._index_for_tab_id(tab_id)
        rect = self._tab_bar.tabRect(index)
        global_position = self._tab_bar.mapToGlobal(rect.bottomLeft())
        QtWidgets.QToolTip.showText(
            global_position,
            str(message),
            self._tab_bar,
            rect,
        )

    def _index_for_tab_id(self, tab_id: str) -> int:
        target = str(tab_id)
        for index in range(self._tab_bar.count()):
            if self._tab_bar.tabData(index) == target:
                return index
        raise KeyError(f"Unknown rendered mechanism tab ID: {target!r}.")

    def _mechanism_tab_count(self) -> int:
        return max(0, self._tab_bar.count() - 1)

    def _on_tab_clicked(self, index: int) -> None:
        if index == self._tab_bar.count() - 1:
            self.createTabRequested.emit()

    def _on_current_changed(self, index: int) -> None:
        if index < 0 or index >= self._mechanism_tab_count():
            return
        tab_id = self._tab_bar.tabData(index)
        if isinstance(tab_id, str) and tab_id:
            self.tabActivated.emit(tab_id)

    def _on_tab_double_clicked(self, index: int) -> None:
        if index < 0 or index >= self._mechanism_tab_count():
            return
        tab_id = self._tab_bar.tabData(index)
        if not isinstance(tab_id, str) or not tab_id:
            return
        current_name = self._tab_bar.tabText(index).removesuffix(" *")
        name, accepted = QtWidgets.QInputDialog.getText(
            self,
            "Rename Mechanism",
            "Mechanism name:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            current_name,
        )
        if accepted:
            self.renameTabRequested.emit(tab_id, str(name))

    def _on_close_requested(self, index: int) -> None:
        if index < 0 or index >= self._mechanism_tab_count():
            return
        if self._mechanism_tab_count() <= 1:
            return
        tab_id = self._tab_bar.tabData(index)
        if isinstance(tab_id, str) and tab_id:
            self.closeTabRequested.emit(tab_id)
