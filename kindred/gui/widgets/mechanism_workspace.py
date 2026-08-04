"""Mechanism tab strip above retained tab-local plot surfaces."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import ProjectDocument
from kindred.gui.plot_config import require_pyqtgraph
from kindred.gui.widgets.pyqtgraph_plot_panel_impl import PyQtGraphPlotPanel

__all__ = ["MechanismWorkspaceView"]


class MechanismWorkspaceView(QtWidgets.QWidget):
    """Physical mechanism selector and tab-local plot host; owns no project truth."""

    tabActivated = QtCore.Signal(str)
    createTabRequested = QtCore.Signal()
    renameTabRequested = QtCore.Signal(str, str)
    closeTabRequested = QtCore.Signal(str)
    plotCreated = QtCore.Signal(str, object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
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

        self._embed_analysis_tabs = bool(main_plot_embed_analysis_tabs)
        self._dark_mode = False
        self._active_tab_id: str | None = None
        self._plot_by_tab_id: dict[str, PyQtGraphPlotPanel] = {}
        self._initial_plot = self._new_plot_panel()
        self._initial_plot.setObjectName("plotPanel")

        self._plot_stack = QtWidgets.QStackedWidget(self)
        self._plot_stack.setObjectName("mechanismPlotStack")
        self._analysis_stack = QtWidgets.QStackedWidget(self)
        self._analysis_stack.setObjectName("mechanismAnalysisStack")
        self._install_analysis_widget(self._initial_plot)
        layout.addWidget(self._plot_stack, 1)

    def _new_plot_panel(self) -> PyQtGraphPlotPanel:
        require_pyqtgraph(context="Kindred GUI")
        plot = PyQtGraphPlotPanel(
            embed_analysis_tabs=self._embed_analysis_tabs,
            workspace_splitter_object_name="mainPlotWorkspaceSplitter",
            enable_axis_inversion_actions=True,
            enable_reference_layer_toggle_action=True,
            enable_copy_visible_data_action=True,
        )
        plot.set_theme(dark_mode=self._dark_mode)
        return plot

    def _install_analysis_widget(self, plot: PyQtGraphPlotPanel) -> None:
        analysis = plot.analysis_tabs_for_workspace(
            detached=not self._embed_analysis_tabs,
        )
        if self._analysis_stack.indexOf(analysis) < 0:
            self._analysis_stack.addWidget(analysis)

    def _ensure_plot(self, tab_id: str) -> PyQtGraphPlotPanel:
        target = str(tab_id)
        plot = self._plot_by_tab_id.get(target)
        if plot is not None:
            return plot
        if self._initial_plot is not None:
            plot = self._initial_plot
            self._initial_plot = None
        else:
            plot = self._new_plot_panel()
        plot.setObjectName("plotPanel")
        self._plot_stack.addWidget(plot)
        self._install_analysis_widget(plot)
        self._plot_by_tab_id[target] = plot
        self.plotCreated.emit(target, plot)
        return plot

    def tab_bar(self) -> QtWidgets.QTabBar:
        return self._tab_bar

    def active_plot(self) -> PyQtGraphPlotPanel:
        if self._active_tab_id is not None:
            plot = self._plot_by_tab_id.get(self._active_tab_id)
            if plot is not None:
                return plot
        if self._initial_plot is not None:
            return self._initial_plot
        raise RuntimeError("Mechanism workspace has no active plot panel.")

    def plot_for_tab(self, tab_id: str) -> PyQtGraphPlotPanel:
        target = str(tab_id)
        try:
            return self._plot_by_tab_id[target]
        except KeyError as exc:
            raise KeyError(f"Unknown mechanism plot tab ID: {target!r}.") from exc

    def main_plot_analysis_widget(self) -> QtWidgets.QStackedWidget:
        return self._analysis_stack

    def main_plot_workspace_splitter(self) -> QtWidgets.QSplitter:
        return self.active_plot().workspace_splitter()

    def available_analysis_surfaces(self) -> list[str]:
        widget = self.active_plot().analysis_tabs_for_workspace(
            detached=False,
        )
        return [str(widget.tabText(index)) for index in range(widget.count())]

    def focus_analysis_surface(self, surface_name: str) -> bool:
        widget = self.active_plot().analysis_tabs_for_workspace(
            detached=False,
        )
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
        current_tab_ids = {tab.tab_id for tab in document.tabs}
        for tab in document.tabs:
            self._ensure_plot(tab.tab_id)
        for tab_id in tuple(self._plot_by_tab_id):
            if tab_id not in current_tab_ids:
                self.drop_tab_session_state(tab_id)

        previous = self._tab_bar.blockSignals(True)
        try:
            for index in range(self._tab_bar.count() - 1, -1, -1):
                rendered_tab_id = self._tab_bar.tabData(index)
                if (
                    rendered_tab_id is not None
                    and rendered_tab_id not in current_tab_ids
                ):
                    self._tab_bar.removeTab(index)

            plus_indexes = [
                index
                for index in range(self._tab_bar.count())
                if self._tab_bar.tabData(index) is None
            ]
            if plus_indexes:
                plus_index = plus_indexes[0]
                for index in reversed(plus_indexes[1:]):
                    self._tab_bar.removeTab(index)
            else:
                plus_index = self._tab_bar.addTab("+")

            for desired_index, tab in enumerate(document.tabs):
                try:
                    current_index = self._index_for_tab_id(tab.tab_id)
                except KeyError:
                    current_index = self._tab_bar.insertTab(
                        desired_index,
                        tab.name,
                    )
                    self._tab_bar.setTabData(current_index, tab.tab_id)
                if current_index != desired_index:
                    self._tab_bar.moveTab(current_index, desired_index)
                    current_index = desired_index
                dirty = self._tab_bar.tabText(current_index).endswith(" *")
                self._tab_bar.setTabText(
                    current_index,
                    f"{tab.name} *" if dirty else tab.name,
                )
                self._tab_bar.setTabToolTip(current_index, tab.name)

            plus_index = next(
                index
                for index in range(self._tab_bar.count())
                if self._tab_bar.tabData(index) is None
            )
            if plus_index != self._tab_bar.count() - 1:
                self._tab_bar.moveTab(plus_index, self._tab_bar.count() - 1)
            plus_index = self._tab_bar.count() - 1
            self._tab_bar.setTabText(plus_index, "+")
            self._tab_bar.setTabData(plus_index, None)
            self._tab_bar.setTabToolTip(plus_index, "Create a fresh mechanism tab")
            self._tab_bar.setTabButton(
                plus_index,
                QtWidgets.QTabBar.ButtonPosition.RightSide,
                None,
            )
            active_index = self._index_for_tab_id(active_tab_id)
            self._tab_bar.setCurrentIndex(active_index)
        finally:
            self._tab_bar.blockSignals(previous)

        active_plot = self._ensure_plot(active_tab_id)
        self._active_tab_id = str(active_tab_id)
        self._plot_stack.setCurrentWidget(active_plot)
        analysis = active_plot.analysis_tabs_for_workspace(detached=False)
        self._analysis_stack.setCurrentWidget(analysis)

    def select_tab_surface(self, tab_id: str) -> None:
        """Switch to an already-rendered tab surface without rebuilding it."""

        target = str(tab_id or "").strip()
        if not target:
            raise ValueError("tab_id must be nonblank.")
        plot = self._plot_by_tab_id.get(target)
        if plot is None:
            raise KeyError(f"Unknown mechanism plot tab ID: {target!r}.")
        index = self._index_for_tab_id(target)
        blocked = self._tab_bar.blockSignals(True)
        try:
            self._tab_bar.setCurrentIndex(index)
        finally:
            self._tab_bar.blockSignals(blocked)
        self._active_tab_id = target
        self._plot_stack.setCurrentWidget(plot)
        self._analysis_stack.setCurrentWidget(
            plot.analysis_tabs_for_workspace(detached=False)
        )

    def drop_tab_session_state(self, tab_id: str) -> None:
        target = str(tab_id)
        plot = self._plot_by_tab_id.pop(target, None)
        if plot is None:
            return
        plot.drop_tab_session_state(target)
        analysis = plot.analysis_tabs_for_workspace(detached=False)
        self._plot_stack.removeWidget(plot)
        self._analysis_stack.removeWidget(analysis)
        analysis.setParent(None)
        plot.setParent(None)
        analysis.deleteLater()
        plot.deleteLater()
        if self._active_tab_id == target:
            self._active_tab_id = None

    def reset_plot_session_state(self) -> None:
        """Discard every old-document physical panel and its view state."""

        for tab_id in tuple(self._plot_by_tab_id):
            self.drop_tab_session_state(tab_id)
        initial = self._initial_plot
        if initial is not None:
            initial.reset_plot_session_state()
            analysis = initial.analysis_tabs_for_workspace(detached=False)
            self._analysis_stack.removeWidget(analysis)
            analysis.setParent(None)
            initial.setParent(None)
            analysis.deleteLater()
            initial.deleteLater()
            self._initial_plot = None
        self._active_tab_id = None

    def set_theme(self, *, dark_mode: bool) -> None:
        self._dark_mode = bool(dark_mode)
        if self._initial_plot is not None:
            self._initial_plot.set_theme(dark_mode=self._dark_mode)
        for plot in self._plot_by_tab_id.values():
            plot.set_theme(dark_mode=self._dark_mode)

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
