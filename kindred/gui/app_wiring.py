"""Small construction helpers for the one live Kindred application graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6 import QtCore

if TYPE_CHECKING:
    from PySide6.QtWidgets import QDockWidget, QWidget

    from kindred.gui.main_window import MainWindow


@dataclass(frozen=True)
class PanelDockComponents:
    dock: QDockWidget
    container: QWidget
    panel: QWidget


@dataclass(frozen=True)
class DockShellSpec:
    identity_key: str
    title: str
    object_name: str
    default_area: QtCore.Qt.DockWidgetArea


_DEFAULT_DOCK_AREA_PROPERTY = "kindred/defaultDockArea"
MECHANISM_DOCK_SPEC = DockShellSpec(
    "mechanism",
    "Mechanism",
    "mechanismDock",
    QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
)
SLIDERS_DOCK_SPEC = DockShellSpec(
    "interactive_sliders",
    "Interactive Sliders",
    "slidersDock",
    QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
)
BATCH_DOCK_SPEC = DockShellSpec(
    "batch_initial_conditions",
    "Initial Conditions",
    "batchDock",
    QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
)
RIGHT_DOCK_SPEC = DockShellSpec(
    "data",
    "Data",
    "rightDock",
    QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
)
ANALYSIS_DOCK_SPEC = DockShellSpec(
    "analysis",
    "Analysis",
    "analysisDock",
    QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
)
SYMBOLIC_CALCULATOR_DOCK_SPEC = DockShellSpec(
    "symbolic_calculator",
    "Symbolic Calculator",
    "symbolicCalculatorDock",
    QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
)
_DOCK_SHELL_SPECS = (
    MECHANISM_DOCK_SPEC,
    SLIDERS_DOCK_SPEC,
    BATCH_DOCK_SPEC,
    RIGHT_DOCK_SPEC,
    ANALYSIS_DOCK_SPEC,
    SYMBOLIC_CALCULATOR_DOCK_SPEC,
)


def dock_shell_specs() -> tuple[DockShellSpec, ...]:
    return _DOCK_SHELL_SPECS


def dock_default_area(dock: QDockWidget):
    stored = dock.property(_DEFAULT_DOCK_AREA_PROPERTY)
    if stored is None:
        return QtCore.Qt.DockWidgetArea.NoDockWidgetArea
    return QtCore.Qt.DockWidgetArea(int(stored))


def build_shell_dock(main_window: MainWindow, spec: DockShellSpec):
    from PySide6 import QtWidgets

    from kindred.gui.widgets.dock_title_bar import DockTitleBar

    dock = QtWidgets.QDockWidget(spec.title, main_window)
    dock.setObjectName(spec.object_name)
    dock.setAllowedAreas(
        QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
        | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
    )
    dock.setProperty(_DEFAULT_DOCK_AREA_PROPERTY, int(spec.default_area.value))
    dock.setFeatures(
        QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
    )
    title_bar = DockTitleBar(title=spec.title, dock=dock, parent=dock)
    dock.setTitleBarWidget(title_bar)
    dock.windowTitleChanged.connect(title_bar.set_title)
    return dock


def build_module_dock(
    main_window: MainWindow,
    *,
    spec: DockShellSpec,
    panel: QWidget,
) -> PanelDockComponents:
    from kindred.gui.widgets.floating_dock_container import FloatingDockContainer

    dock = build_shell_dock(main_window, spec)
    container = FloatingDockContainer(
        content=panel,
        dock=dock,
        on_dock_back=lambda: main_window.redock_shell_dock(dock),
        on_reset_layout=main_window.reset_layout,
        parent=main_window,
    )
    dock.setWidget(container)
    main_window.addDockWidget(spec.default_area, dock)
    return PanelDockComponents(dock=dock, container=container, panel=panel)
