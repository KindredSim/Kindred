"""The permanent Qt custody root for one project workspace."""

from __future__ import annotations

from collections.abc import Mapping

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import ProjectDocument
from kindred.core.project_presentation_state import WorkspacePresentationState
from kindred.gui.widgets.pyqtgraph_plot_panel_impl import PyQtGraphPlotPanel

__all__ = ["ProjectWorkspaceWidget"]

_CREATE_TAB_TOOLTIP = "Create a fresh mechanism tab"


class ProjectWorkspaceWidget(QtWidgets.QMainWindow):
    """One permanent physical project root.

    The root deliberately owns only Qt custody.  Project semantics, result
    repositories, persistence and replacement policy remain in their existing
    owners.  The tab strip and plot stack are kept in the central child while
    the six module panels are installed as ordinary ``QDockWidget`` shells
    beneath this nested ``QMainWindow``.
    """

    tabActivated = QtCore.Signal(str)
    createTabRequested = QtCore.Signal()
    renameTabRequested = QtCore.Signal(str, str)
    closeTabRequested = QtCore.Signal(str)
    plotCreated = QtCore.Signal(str, object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.setObjectName("projectWorkspaceRoot")
        self.setDockNestingEnabled(True)
        self._dock_by_identity: dict[str, QtWidgets.QDockWidget] = {}
        central = QtWidgets.QWidget(self)
        central.setObjectName("projectWorkspaceCentral")
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
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

    def install_dock_panels(
        self,
        panels: Mapping[str, QtWidgets.QWidget],
    ) -> tuple[QtWidgets.QDockWidget, ...]:
        """Install the six root-local physical module shells."""

        from kindred.gui.app_wiring import (
            ANALYSIS_DOCK_SPEC,
            BATCH_DOCK_SPEC,
            MECHANISM_DOCK_SPEC,
            RIGHT_DOCK_SPEC,
            SLIDERS_DOCK_SPEC,
            SYMBOLIC_CALCULATOR_DOCK_SPEC,
            build_module_dock,
        )

        specs = (
            (MECHANISM_DOCK_SPEC, "mechanism"),
            (SLIDERS_DOCK_SPEC, "interactive_sliders"),
            (BATCH_DOCK_SPEC, "batch_initial_conditions"),
            (RIGHT_DOCK_SPEC, "data"),
            (ANALYSIS_DOCK_SPEC, "analysis"),
            (SYMBOLIC_CALCULATOR_DOCK_SPEC, "symbolic_calculator"),
        )
        expected = {identity for _spec, identity in specs}
        supplied = {str(key) for key in panels}
        if supplied != expected:
            raise ValueError(
                "dock_panels must contain exactly the six project module keys."
            )
        if self._dock_by_identity:
            raise RuntimeError("Project workspace docks are already installed.")
        docks: list[QtWidgets.QDockWidget] = []
        for spec, identity in specs:
            dock = build_module_dock(
                self,
                spec=spec,
                panel=panels[identity],
            )
            self._dock_by_identity[identity] = dock
            docks.append(dock)
        return tuple(docks)

    def dock_for_identity(self, identity: str) -> QtWidgets.QDockWidget:
        try:
            return self._dock_by_identity[str(identity)]
        except KeyError as exc:
            raise KeyError(f"Unknown project dock identity: {identity!r}.") from exc

    def apply_default_dock_arrangement(self) -> None:
        """Restore the root-owned default six-dock arrangement."""

        docks = tuple(self._dock_by_identity.values())
        for dock in docks:
            dock.setFloating(False)
            area = dock.property("kindred/defaultDockArea")
            if area is not None:
                self.addDockWidget(QtCore.Qt.DockWidgetArea(int(area)), dock)
            dock.show()
        if len(docks) != 6:
            return
        mechanism = self.dock_for_identity("mechanism")
        sliders = self.dock_for_identity("interactive_sliders")
        batch = self.dock_for_identity("batch_initial_conditions")
        data = self.dock_for_identity("data")
        analysis = self.dock_for_identity("analysis")
        calculator = self.dock_for_identity("symbolic_calculator")
        self.splitDockWidget(mechanism, sliders, QtCore.Qt.Orientation.Vertical)
        self.splitDockWidget(batch, data, QtCore.Qt.Orientation.Vertical)
        self.tabifyDockWidget(data, analysis)
        self.tabifyDockWidget(data, calculator)
        data.raise_()

    def redock_dock(self, dock: QtWidgets.QDockWidget) -> None:
        if dock not in self._dock_by_identity.values():
            raise ValueError("dock must belong to this project workspace root.")
        dock.setFloating(False)
        area = dock.property("kindred/defaultDockArea")
        if area is not None:
            self.addDockWidget(QtCore.Qt.DockWidgetArea(int(area)), dock)
        dock.show()

    def recover_floating_docks(self, available: QtCore.QRect) -> None:
        for dock in self._dock_by_identity.values():
            if not dock.isFloating():
                continue
            if available.isValid() and available.intersects(dock.frameGeometry()):
                continue
            self.redock_dock(dock)

    def restore_dock_layout(self, state: object) -> bool:
        """Restore one persisted Qt dock state, falling back when invalid."""

        if isinstance(state, QtCore.QByteArray):
            payload = state
        elif isinstance(state, (bytes, bytearray)):
            payload = QtCore.QByteArray(bytes(state))
        else:
            self.apply_default_dock_arrangement()
            return False
        if payload.isEmpty():
            self.apply_default_dock_arrangement()
            return False
        try:
            restored = bool(self.restoreState(payload))
        except (RuntimeError, TypeError, ValueError):
            restored = False
        if not restored:
            self.apply_default_dock_arrangement()
        return restored

    def reveal_dock(
        self,
        identity: str,
        *,
        available_geometry: QtCore.QRect | None = None,
    ) -> QtWidgets.QDockWidget:
        """Show and raise a named dock, repinning it when floating off-screen."""

        dock = self.dock_for_identity(identity)
        if dock.isFloating():
            available = available_geometry
            if available is None or not available.isValid():
                screen = dock.screen() or self.screen()
                if screen is None:
                    application = QtWidgets.QApplication.instance()
                    screen = (
                        application.primaryScreen()
                        if isinstance(application, QtWidgets.QApplication)
                        else None
                    )
                available = (
                    screen.availableGeometry() if screen is not None else None
                )
            if (
                available is not None
                and available.isValid()
                and not available.intersects(dock.frameGeometry())
            ):
                self.redock_dock(dock)
        dock.show()
        dock.raise_()
        if dock.isFloating():
            dock.activateWindow()
        return dock

    def _new_plot_panel(self) -> PyQtGraphPlotPanel:
        plot = PyQtGraphPlotPanel()
        plot.set_theme(dark_mode=self._dark_mode)
        return plot

    def _install_analysis_widget(self, plot: PyQtGraphPlotPanel) -> None:
        analysis = plot.analysis_tabs()
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
        """Return the physical analysis stack used by the Analysis dock."""

        return self._analysis_stack

    def render(
        self,
        document: ProjectDocument,
        *,
        active_tab_id: str | None = None,
        presentation: WorkspacePresentationState | None = None,
    ) -> None:
        """Render one document directly into this workspace root.

        Rendering is intentionally limited to tab/plot custody.  It reads the
        document and optional presentation snapshot, and never looks up a
        registry or connects a live callback.
        """

        try:
            if not isinstance(document, ProjectDocument):
                raise TypeError("document must be a ProjectDocument.")
            if presentation is not None:
                if not isinstance(presentation, WorkspacePresentationState):
                    raise TypeError(
                        "presentation must be WorkspacePresentationState."
                    )
                incoming_active = presentation.active_tab_id
                if (
                    active_tab_id is not None
                    and str(active_tab_id) != incoming_active
                ):
                    raise ValueError("active_tab_id disagrees with presentation.")
                active_tab_id = incoming_active
            if active_tab_id is None:
                raise ValueError("active_tab_id or presentation is required.")
            target_active = str(active_tab_id)
            document.tab(target_active)
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
                            desired_index, tab.name
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
                self._tab_bar.setTabToolTip(plus_index, _CREATE_TAB_TOOLTIP)
                self._tab_bar.setTabButton(
                    plus_index,
                    QtWidgets.QTabBar.ButtonPosition.RightSide,
                    None,
                )
                active_index = self._index_for_tab_id(target_active)
                self._tab_bar.setCurrentIndex(active_index)
            finally:
                self._tab_bar.blockSignals(previous)

            active_plot = self._ensure_plot(target_active)
            self._active_tab_id = target_active
            self._plot_stack.setCurrentWidget(active_plot)
            self._analysis_stack.setCurrentWidget(active_plot.analysis_tabs())
        except Exception:
            self.dispose()
            raise

    def select_tab_surface(self, tab_id: str) -> None:
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
        self._analysis_stack.setCurrentWidget(plot.analysis_tabs())

    def drop_tab_session_state(self, tab_id: str) -> None:
        target = str(tab_id)
        plot = self._plot_by_tab_id.pop(target, None)
        if plot is None:
            return
        plot.drop_tab_session_state()
        analysis = plot.analysis_tabs()
        self._plot_stack.removeWidget(plot)
        self._analysis_stack.removeWidget(analysis)
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
            analysis = initial.analysis_tabs()
            self._analysis_stack.removeWidget(analysis)
            analysis.deleteLater()
            initial.deleteLater()
            self._initial_plot = None
        self._active_tab_id = None

    def dispose(self) -> None:
        """Best-effort child cleanup with unconditional root teardown."""

        try:
            self.reset_plot_session_state()
        except Exception:
            # The root remains the final Qt custody owner; even a damaged plot
            # child must not prevent the root from being retired.
            pass
        finally:
            self.deleteLater()

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
