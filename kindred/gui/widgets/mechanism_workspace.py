"""The permanent Qt custody root for one project workspace."""

from __future__ import annotations

from collections.abc import Mapping

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import ProjectDocument
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
    tabMenuOpening = QtCore.Signal()
    copyTabRequested = QtCore.Signal(str)
    reorderTabsRequested = QtCore.Signal(object)
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
        # Mouse gestures are resolved before activation can settle editor drafts.
        # Keyboard focus/navigation remains native; dragging never takes focus.
        self._tab_bar.setFocusPolicy(QtCore.Qt.FocusPolicy.TabFocus)
        self._tab_bar.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._tab_bar.customContextMenuRequested.connect(self._on_tab_context_menu)
        self._tab_press_position: QtCore.QPoint | None = None
        self._tab_press_id: str | None = None
        self._tab_dragging = False
        self._tab_drag_position = QtCore.QPoint()
        self._tab_scroll_timer = QtCore.QTimer(self)
        self._tab_scroll_timer.setInterval(80)
        self._tab_scroll_timer.timeout.connect(self._scroll_dragged_tab)
        self._tab_bar.installEventFilter(self)
        self._tab_bar.currentChanged.connect(self._on_current_changed)
        self._tab_bar.tabBarClicked.connect(self._on_tab_clicked)
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
            DOCK_SPECS,
            build_module_dock,
        )

        specs = DOCK_SPECS
        docks: list[QtWidgets.QDockWidget] = []
        for spec in specs:
            dock = build_module_dock(
                self,
                spec=spec,
                panel=panels[spec.identity_key],
            )
            self._dock_by_identity[spec.identity_key] = dock
            docks.append(dock)
        return tuple(docks)

    def dock_for_identity(self, identity: str) -> QtWidgets.QDockWidget:
        return self._dock_by_identity[identity]

    def _docks_are_stacked(
        self,
        upper: QtWidgets.QDockWidget,
        lower: QtWidgets.QDockWidget,
    ) -> bool:
        peers = self.tabifiedDockWidgets(lower)
        if upper in peers:
            return False
        # Inactive tabs have offscreen geometry; use the displayed member.
        displayed = next(
            (dock for dock in (lower, *peers) if not dock.visibleRegion().isEmpty()),
            lower,
        )
        upper_rect, lower_rect = upper.geometry(), displayed.geometry()
        return (
            upper_rect.bottom() < lower_rect.top()
            and upper_rect.left() == lower_rect.left()
            and upper_rect.right() == lower_rect.right()
        )

    def reset_visible_dock_layout(self) -> None:
        """Restore open panels' positions while retaining hidden Qt dock slots."""

        from kindred.gui.app_wiring import (
            ANALYSIS_DOCK_SPEC,
            BATCH_DOCK_SPEC,
            DOCK_SPECS,
            MECHANISM_DOCK_SPEC,
            RIGHT_DOCK_SPEC,
            SLIDERS_DOCK_SPEC,
            SYMBOLIC_CALCULATOR_DOCK_SPEC,
        )

        docks = {
            identity: dock
            for identity, dock in self._dock_by_identity.items()
            if not dock.isHidden()
        }
        for spec in DOCK_SPECS:
            dock = docks.get(spec.identity_key)
            if dock is None:
                continue
            if dock.isFloating():
                dock.setFloating(False)
            if self.dockWidgetArea(dock) != spec.default_area:
                self.addDockWidget(spec.default_area, dock)
        self.layout().activate()

        mechanism = docks.get(MECHANISM_DOCK_SPEC.identity_key)
        sliders = docks.get(SLIDERS_DOCK_SPEC.identity_key)
        if mechanism is not None and sliders is not None:
            if not self._docks_are_stacked(mechanism, sliders):
                self.addDockWidget(MECHANISM_DOCK_SPEC.default_area, mechanism)
                self.addDockWidget(SLIDERS_DOCK_SPEC.default_area, sliders)
                self.splitDockWidget(mechanism, sliders, QtCore.Qt.Orientation.Vertical)
        self.layout().activate()

        lower = [
            docks[spec.identity_key]
            for spec in (
                RIGHT_DOCK_SPEC,
                ANALYSIS_DOCK_SPEC,
                SYMBOLIC_CALCULATOR_DOCK_SPEC,
            )
            if spec.identity_key in docks
        ]
        if not lower:
            return
        anchor = lower[0]
        batch = docks.get(BATCH_DOCK_SPEC.identity_key)
        if batch is not None and not self._docks_are_stacked(batch, anchor):
            self.addDockWidget(BATCH_DOCK_SPEC.default_area, batch)
            self.addDockWidget(RIGHT_DOCK_SPEC.default_area, anchor)
            self.splitDockWidget(batch, anchor, QtCore.Qt.Orientation.Vertical)
        for dock in lower[1:]:
            if dock not in self.tabifiedDockWidgets(anchor):
                self.tabifyDockWidget(anchor, dock)
        anchor.raise_()

    def restore_default_panel_layout(self) -> None:
        """Restore the factory arrangement and panel visibility."""

        from kindred.gui.app_wiring import (
            BATCH_DOCK_SPEC,
            MECHANISM_DOCK_SPEC,
            RIGHT_DOCK_SPEC,
        )

        for dock in self._dock_by_identity.values():
            dock.show()
        self.reset_visible_dock_layout()
        visible = {
            MECHANISM_DOCK_SPEC.identity_key,
            BATCH_DOCK_SPEC.identity_key,
            RIGHT_DOCK_SPEC.identity_key,
        }
        for identity, dock in self._dock_by_identity.items():
            dock.setVisible(identity in visible)

    def redock_dock(self, dock: QtWidgets.QDockWidget) -> None:
        dock.setFloating(False)
        area = dock.property("kindred/defaultDockArea")
        if area is not None:
            self.addDockWidget(QtCore.Qt.DockWidgetArea(int(area)), dock)
        dock.show()

    def recover_floating_docks(self) -> None:
        available = tuple(screen.availableGeometry() for screen in QtWidgets.QApplication.screens())
        for dock in self._dock_by_identity.values():
            if not dock.isFloating():
                continue
            if any(rect.intersects(dock.frameGeometry()) for rect in available):
                continue
            self.redock_dock(dock)

    def restore_dock_layout(self, state: object) -> bool:
        """Restore one persisted Qt dock state, falling back when invalid."""

        if isinstance(state, QtCore.QByteArray):
            payload = state
        elif isinstance(state, (bytes, bytearray)):
            payload = QtCore.QByteArray(bytes(state))
        else:
            self.restore_default_panel_layout()
            return False
        if payload.isEmpty():
            self.restore_default_panel_layout()
            return False
        restored = self.restoreState(payload)
        if not restored:
            self.restore_default_panel_layout()
        return restored

    def reveal_dock(self, identity: str) -> QtWidgets.QDockWidget:
        """Show a named dock, preserving floating placement on any screen."""
        dock = self.dock_for_identity(identity)
        if dock.isFloating() and not any(
            screen.availableGeometry().intersects(dock.frameGeometry())
            for screen in QtWidgets.QApplication.screens()
        ):
            self.redock_dock(dock)
        dock.show()
        dock.raise_()
        if dock.isFloating():
            dock.activateWindow()
        return dock

    def is_dock_frontmost(self, identity: str) -> bool:
        """Return whether a dock is shown and not behind another dock tab."""

        dock = self.dock_for_identity(identity)
        if not dock.isVisible():
            return False
        if dock.isFloating():
            return True
        return not dock.visibleRegion().isEmpty()

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

    def plot_for_tab(self, tab_id: str) -> PyQtGraphPlotPanel:
        return self._plot_by_tab_id[str(tab_id)]

    def main_plot_analysis_widget(self) -> QtWidgets.QStackedWidget:
        """Return the physical analysis stack used by the Analysis dock."""

        return self._analysis_stack

    def render(
        self,
        document: ProjectDocument,
        *,
        active_tab_id: str,
    ) -> None:
        """Render one document directly into this workspace root.

        Rendering is intentionally limited to tab/plot custody.  It reads the
        document and never looks up a registry or connects a live callback.
        A failure propagates; the permanent root stays usable.
        """

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
            active_index = self._index_for_tab_id(target_active)
            self._tab_bar.setCurrentIndex(active_index)
        finally:
            self._tab_bar.blockSignals(previous)

        active_plot = self._ensure_plot(target_active)
        self._active_tab_id = target_active
        self._plot_stack.setCurrentWidget(active_plot)
        self._analysis_stack.setCurrentWidget(active_plot.analysis_tabs())

    def select_tab_surface(self, tab_id: str) -> None:
        target = tab_id
        plot = self._plot_by_tab_id[target]
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
        # The panel is deleted below; only its project content needs releasing.
        plot.dispose_project_content()
        analysis = plot.analysis_tabs()
        self._plot_stack.removeWidget(plot)
        self._analysis_stack.removeWidget(analysis)
        analysis.deleteLater()
        plot.deleteLater()
        if self._active_tab_id == target:
            self._active_tab_id = None

    def set_theme(self, *, dark_mode: bool) -> None:
        self._dark_mode = bool(dark_mode)
        if self._initial_plot is not None:
            self._initial_plot.set_theme(dark_mode=self._dark_mode)
        for plot in self._plot_by_tab_id.values():
            plot.set_theme(dark_mode=self._dark_mode)

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

    def eventFilter(self, watched, event):  # noqa: N802 - Qt override
        kind = event.type()
        if self._tab_press_position is not None and (
            kind in (QtCore.QEvent.Type.Hide, QtCore.QEvent.Type.WindowDeactivate,
                     QtCore.QEvent.Type.UngrabMouse)
        ):
            self._end_tab_gesture()
        if watched is not self._tab_bar:
            return super().eventFilter(watched, event)
        bar = self._tab_bar
        if kind == QtCore.QEvent.Type.MouseButtonDblClick:
            self._end_tab_gesture()
            return True
        if kind == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() not in (
                QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton,
            ):
                return False
            index = bar.tabAt(event.position().toPoint())
            if index < 0:
                return False
            if event.button() == QtCore.Qt.MouseButton.RightButton:
                self._end_tab_gesture()
                return True
            self._tab_press_position = event.position().toPoint()
            self._tab_press_id = bar.tabData(index)
            self._tab_dragging = False
            self.window().installEventFilter(self)
            return True
        if (
            kind == QtCore.QEvent.Type.MouseMove
            and self._tab_press_position is not None
        ):
            position = event.position().toPoint()
            distance = (position - self._tab_press_position).manhattanLength()
            if distance >= QtWidgets.QApplication.startDragDistance():
                self._tab_dragging = True
            if self._tab_dragging and self._tab_press_id is not None:
                self._tab_drag_position = position
                self._move_dragged_tab()
                if self._edge_scroll_button() is not None:
                    if not self._tab_scroll_timer.isActive():
                        self._tab_scroll_timer.start()
                else:
                    self._tab_scroll_timer.stop()
            return True
        if (
            kind == QtCore.QEvent.Type.MouseButtonRelease
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            if self._tab_press_position is None:
                return True
            dragging, tab_id = self._tab_dragging, self._tab_press_id
            self._end_tab_gesture()
            if not dragging:
                index = bar.tabAt(event.position().toPoint())
                if index >= 0 and bar.tabData(index) == tab_id:
                    bar.tabBarClicked.emit(index)
            return True
        return super().eventFilter(watched, event)

    def _end_tab_gesture(self) -> None:
        self._tab_scroll_timer.stop()
        moved = self._tab_dragging and self._tab_press_id is not None
        self._tab_press_position = None
        self._tab_press_id = None
        self._tab_dragging = False
        self.window().removeEventFilter(self)
        if moved:
            self.reorderTabsRequested.emit(
                tuple(self._tab_bar.tabData(index)
                      for index in range(self._mechanism_tab_count()))
            )

    def _move_dragged_tab(self) -> None:
        bar = self._tab_bar
        source = self._index_for_tab_id(self._tab_press_id)
        target = bar.tabAt(self._tab_drag_position)
        if target < 0:
            return
        target = min(target, self._mechanism_tab_count() - 1)
        # Cross the neighbour's midpoint before moving. Hit-testing alone
        # reverses the move when unequal tab widths rearrange under the pointer.
        midpoint = bar.tabRect(target).center().x()
        if target > source and self._tab_drag_position.x() < midpoint:
            target -= 1
        elif target < source and self._tab_drag_position.x() > midpoint:
            target += 1
        with QtCore.QSignalBlocker(bar):
            bar.moveTab(source, target)

    def _edge_scroll_button(self) -> QtWidgets.QToolButton | None:
        bar = self._tab_bar
        position = self._tab_drag_position
        if not bar.rect().contains(position):
            return None
        buttons = [
            button for button in bar.findChildren(QtWidgets.QToolButton)
            if button.isVisible()
        ]
        margin = max(24, sum(button.width() for button in buttons))
        arrow = (
            QtCore.Qt.ArrowType.LeftArrow if position.x() < margin
            else QtCore.Qt.ArrowType.RightArrow
            if position.x() >= bar.width() - margin else None
        )
        return next(
            (button for button in buttons
             if button.arrowType() == arrow and button.isEnabled()), None
        )

    def _scroll_dragged_tab(self) -> None:
        button = self._edge_scroll_button()
        if button is None:
            self._tab_scroll_timer.stop()
            return
        with QtCore.QSignalBlocker(self._tab_bar):
            button.click()
        self._move_dragged_tab()
        if self._edge_scroll_button() is None:
            self._tab_scroll_timer.stop()

    def _on_tab_context_menu(self, position: QtCore.QPoint) -> None:
        index = self._tab_bar.tabAt(position)
        if index < 0:
            return
        tab_id = self._tab_bar.tabData(index)
        if tab_id is None:
            return
        self.tabMenuOpening.emit()
        menu = QtWidgets.QMenu(self)
        menu.setToolTipsVisible(True)
        rename = menu.addAction("Rename…")
        rename.setToolTip("Change the name of this mechanism tab.")
        copy = menu.addAction("Create Copy")
        copy.setToolTip(
            "Copy this mechanism and its settings into a new tab, including "
            "simulation results that still match them. Unapplied edits and "
            "slider previews are not copied."
        )
        delete = menu.addAction("Delete")
        delete.setToolTip(
            "Remove this mechanism tab from the project. The final tab cannot be deleted."
        )
        delete.setEnabled(self._mechanism_tab_count() > 1)
        chosen = menu.exec(self._tab_bar.mapToGlobal(position))
        if chosen is rename:
            self._rename_tab(tab_id)
        elif chosen is copy:
            self.copyTabRequested.emit(tab_id)
        elif chosen is delete:
            self.closeTabRequested.emit(tab_id)
        menu.deleteLater()

    def _on_tab_clicked(self, index: int) -> None:
        if index < 0 or index >= self._tab_bar.count():
            return
        tab_id = self._tab_bar.tabData(index)
        if tab_id is None:
            self.createTabRequested.emit()
        else:
            self.tabActivated.emit(tab_id)

    def _on_current_changed(self, index: int) -> None:
        if index < 0 or index >= self._mechanism_tab_count():
            return
        self.tabActivated.emit(self._tab_bar.tabData(index))

    def _rename_tab(self, tab_id: str) -> None:
        index = self._index_for_tab_id(tab_id)
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
