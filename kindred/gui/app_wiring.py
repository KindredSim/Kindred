"""Small, local Qt construction helpers for a project workspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

if TYPE_CHECKING:
    from kindred.gui.widgets.mechanism_workspace import ProjectWorkspaceWidget

__all__ = [
    "ANALYSIS_DOCK_SPEC",
    "BATCH_DOCK_SPEC",
    "DOCK_SPECS",
    "DockShellSpec",
    "MECHANISM_DOCK_SPEC",
    "RIGHT_DOCK_SPEC",
    "SLIDERS_DOCK_SPEC",
    "SYMBOLIC_CALCULATOR_DOCK_SPEC",
    "build_module_dock",
    "build_physical_shell",
    "build_shell_dock",
]


@dataclass(frozen=True, slots=True)
class DockShellSpec:
    identity_key: str
    title: str
    object_name: str
    default_area: QtCore.Qt.DockWidgetArea
    surface_object_name: str | None = None


_DEFAULT_DOCK_AREA_PROPERTY = "kindred/defaultDockArea"
MECHANISM_DOCK_SPEC = DockShellSpec(
    "mechanism",
    "Mechanism",
    "mechanismDock",
    QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
    "mechanismSurfaceStack",
)
SLIDERS_DOCK_SPEC = DockShellSpec(
    "interactive_sliders",
    "Interactive Sliders",
    "slidersDock",
    QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
    "sliderSurfaceStack",
)
BATCH_DOCK_SPEC = DockShellSpec(
    "batch_initial_conditions",
    "Initial Conditions",
    "batchDock",
    QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
    "initialConditionsSurfaceStack",
)
RIGHT_DOCK_SPEC = DockShellSpec(
    "data",
    "Data",
    "rightDock",
    QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
    "dataSurfaceStack",
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
    "symbolicCalculatorSurfaceStack",
)

DOCK_SPECS = (
    MECHANISM_DOCK_SPEC,
    SLIDERS_DOCK_SPEC,
    BATCH_DOCK_SPEC,
    RIGHT_DOCK_SPEC,
    ANALYSIS_DOCK_SPEC,
    SYMBOLIC_CALCULATOR_DOCK_SPEC,
)


def build_shell_dock(
    parent: QtWidgets.QWidget,
    spec: DockShellSpec,
) -> QtWidgets.QDockWidget:
    """Construct one unconnected dock shell beneath ``parent``."""

    from kindred.gui.widgets.dock_title_bar import DockTitleBar

    dock = QtWidgets.QDockWidget(spec.title, parent)
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
    workspace_root: ProjectWorkspaceWidget,
    *,
    spec: DockShellSpec,
    panel: QtWidgets.QWidget,
) -> QtWidgets.QDockWidget:
    """Build one panel shell with only root-local custody callbacks.

    ``FloatingDockContainer`` receives methods on the workspace root.  No
    policy owner, registry, simulation owner, or result owner is captured by
    this construction path.
    """

    from kindred.gui.widgets.floating_dock_container import FloatingDockContainer

    dock = build_shell_dock(workspace_root, spec)
    container = FloatingDockContainer(
        content=panel,
        dock=dock,
        on_dock_back=lambda: workspace_root.redock_dock(dock),
        on_reset_layout=workspace_root.apply_default_dock_arrangement,
        parent=workspace_root,
    )
    dock.setWidget(container)
    workspace_root.addDockWidget(spec.default_area, dock)
    return dock


def build_physical_shell(
    workspace_root: ProjectWorkspaceWidget,
) -> dict[str, QtWidgets.QStackedWidget]:
    """Create root-local stacks and the six physical dock shells.

    The returned stacks are empty until the caller renders tab surfaces. The
    physical shell is separate from semantic publication; live bindings are
    installed only after the authoritative document is selected.
    """

    try:
        stacks = {
            spec.identity_key: QtWidgets.QStackedWidget(workspace_root)
            for spec in DOCK_SPECS
            if spec.surface_object_name is not None
        }
        for spec in DOCK_SPECS:
            if spec.surface_object_name is not None:
                stacks[spec.identity_key].setObjectName(spec.surface_object_name)
        workspace_root.install_dock_panels(
            {
                **stacks,
                ANALYSIS_DOCK_SPEC.identity_key: (
                    workspace_root.main_plot_analysis_widget()
                ),
            }
        )
    except Exception:
        workspace_root.dispose()
        raise
    return stacks
