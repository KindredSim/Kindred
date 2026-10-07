"""PyQtGraph workspace surface for one immutable result input."""

from __future__ import annotations

import html

from kindred.core.quantity_units import (
    CUBIC_METRE,
    MOLAR,
    SECOND,
    Unit,
    conversion_scale,
    format_unit_for_display,
    presentation_unit,
    reconcile_unit_choices,
    scientific_unit,
)

import io
import math
import time
from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence
from typing import Any, Iterator

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.project_presentation_state import (
    ProjectPresentationState,
    SimulationSeriesAxisSelection,
    VesselVolumeAxisSelection,
)
from kindred.core.result_repository import SimulationResultValue
from kindred.gui.curve_copy_headers import copy_column_header, copy_series_label
from kindred.gui.display_name_policy import literal_tooltip
from kindred.gui.theme_manager import plot_background_color, plot_foreground_color
from kindred.gui.ui_helpers import (
    SCIENTIFIC_NAME_HTML_ROLE,
    ScientificNameComboBox,
    clamp_initial_size,
)
from kindred.gui.interactive_preview import InteractivePreviewRequest
from kindred.gui.plot_config import (
    TIME_DISPLAY_UNITS,
    require_pyqtgraph,
)
from kindred.gui.widgets.axis_toolbar import AxisToolbar
from kindred.gui.widgets.parameter_statistics_table import ParameterStatisticsTable
from kindred.gui.widgets.species_statistics_table import SpeciesStatisticsTable
from kindred.gui.workspace_result_presenter import (
    InteractiveDisplayCurve,
    InteractiveDisplayProjection,
    ResultDataBlock,
    ResultRenderInput,
    ResultLayerKind,
    encode_clipboard_text,
    encode_tsv_row,
    compose_interactive_display,
    intervention_label,
)

pg = require_pyqtgraph(context="Kindred GUI")

__all__ = ["PyQtGraphPlotPanel"]


class _PlotDataItem(pg.PlotDataItem):
    """Treat scatter axes with no finite mapped samples as having no bounds."""

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if not self.curve.isVisible() and self.scatter.isVisible():
            # 0.13.7 checks for empty scatter data before, but not after,
            # filtering nonfinite values for percentiles. Log mapping can
            # remove every finite value, including within a visible-X window.
            data = self.scatter.data
            values = data["x" if ax == 0 else "y"]
            if orthoRange is not None:
                other = data["y" if ax == 0 else "x"]
                values = values[(other >= orthoRange[0]) & (other <= orthoRange[1])]
            if not np.isfinite(values).any():
                return (None, None)
        return super().dataBounds(ax, frac=frac, orthoRange=orthoRange)


class _PlotViewBox(pg.ViewBox):
    """Keep native gestures, with effective axis masks for axis drags and rectangles."""

    def childrenBounds(self, frac=None, orthoRange=(None, None), items=None):
        items = self.addedItems if items is None else items
        # 0.13.7 ScatterPlotItem caches restricted bounds as unrestricted
        # bounds. Clear that cache before asking native code for the bounds.
        for item in items:
            if isinstance(item, pg.PlotDataItem):
                item.scatter.bounds = [None, None]
        return super().childrenBounds(frac=frac, orthoRange=orthoRange, items=items)

    def mouseDragEvent(self, event: object, axis: int | None = None) -> None:
        enabled = self.state["mouseEnabled"]
        # 0.13.7 emits mouseEnabled instead of its axis-local drag mask.
        # Narrow the native gesture itself so right drags also leave the other
        # axis automatic; restore the user's locks when this event returns.
        self.state["mouseEnabled"] = [
            value and (axis is None or index == axis)
            for index, value in enumerate(enabled)
        ]
        try:
            super().mouseDragEvent(event, axis=axis)
        finally:
            self.state["mouseEnabled"] = enabled

    def showAxRect(self, rect: QtCore.QRectF, **kwargs: object) -> None:
        rect = rect.normalized()
        ranges = {}
        if self.state["mouseEnabled"][0]:
            ranges["xRange"] = (rect.left(), rect.right())
        if self.state["mouseEnabled"][1]:
            ranges["yRange"] = (rect.top(), rect.bottom())
        if ranges:
            self.setRange(**ranges, **kwargs)
            self.sigRangeChangedManually.emit(self.state["mouseEnabled"])


@dataclass(frozen=True, slots=True)
class _AxisRangeSession:
    auto_x: bool = True
    auto_y: bool = True
    mouse_x: bool = True
    mouse_y: bool = True
    fraction_x: float = 1.0
    fraction_y: float = 1.0
    pan_x: bool = False
    pan_y: bool = False
    visible_x: bool = False
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None


@dataclass(slots=True)
class _AnnotationLabel:
    item: Any
    anchor: QtCore.QPointF
    event_x: float
    lane: int


_EPSILON = 1e-9
_DisplayAnnotation = tuple[str, str, Mapping[str, object], float]
_AnnotationKey = tuple[str, str, tuple[tuple[str, str], ...]]
_ANNOTATION_ACCENTS = ("#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2")


def _transform_samples(x: np.ndarray, y: np.ndarray, transform: str) -> tuple[np.ndarray, np.ndarray]:
    """Native adjacent-difference mapping, with undefined pairs kept as breaks."""
    if transform == "original":
        return x, y
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dx, dy = np.diff(x), np.diff(y)
        valid = np.isfinite(x[:-1]) & np.isfinite(x[1:]) & np.isfinite(y[:-1]) & np.isfinite(y[1:]) & np.isfinite(dx) & np.isfinite(dy) & (dx != 0)
        derivative = np.full(dx.shape, np.nan, dtype=float)
        np.divide(dy, dx, out=derivative, where=valid)
        derivative[~np.isfinite(derivative)] = np.nan
    return (y[:-1] if transform == "phase" else x[:-1]), derivative


@dataclass(frozen=True, slots=True)
class _RenderedPointChunk:
    """One exact point chunk handed to a static or interactive plot item."""

    x: np.ndarray
    series: tuple[tuple[str, np.ndarray], ...]
    sample_time: np.ndarray | None
    instantaneous: bool = False

    def __post_init__(self) -> None:
        x = np.asarray(self.x)
        x.setflags(write=False)
        series = tuple(
            (str(name), np.asarray(values)) for name, values in self.series
        )
        for _name, values in series:
            values.setflags(write=False)
            if values.size != x.size:
                raise ValueError("rendered point series must align with X.")
        if self.sample_time is not None:
            sample_time = np.asarray(self.sample_time)
            sample_time.setflags(write=False)
            if sample_time.size != x.size:
                raise ValueError("rendered point time must align with X.")
            object.__setattr__(self, "sample_time", sample_time)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "series", series)


@dataclass(frozen=True, slots=True)
class _RenderedPointBlock:
    """Private sampled roster entry; it is not result or export authority."""

    block_id: str
    group_id: str
    label: str
    kind: str
    x_name: str
    x_is_time: bool
    x_unit: str | None
    time_column_name: str | None
    y_names: tuple[tuple[str, str], ...]
    y_units: tuple[tuple[str, str | None], ...]
    chunks: tuple[_RenderedPointChunk, ...]
    left_endpoint: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", str(self.block_id))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "x_name", str(self.x_name))
        object.__setattr__(
            self,
            "x_unit",
            None if self.x_unit is None else str(self.x_unit),
        )
        object.__setattr__(self, "chunks", tuple(self.chunks))


class PyQtGraphPlotPanel(QtWidgets.QWidget):
    """Physical plot/analysis widget; semantic visibility stays in presentation."""

    referenceLayerVisibilityRequested = QtCore.Signal(bool)
    interventionAnnotationsVisibilityRequested = QtCore.Signal(bool)
    interventionLabelsVisibilityRequested = QtCore.Signal(bool)
    copyAllRequested = QtCore.Signal()
    xSeriesRequested = QtCore.Signal(object)
    ySeriesRequested = QtCore.Signal(object)
    plotPresentationIntentRequested = QtCore.Signal(str, object)
    resultDiagnostic = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._interactive_gesture_active = False
        self._sampling_limit: int | None = None
        self._plot_items_by_key: dict[tuple[object, ...], object] = {}
        self._annotation_items: dict[_AnnotationKey, object] = {}
        self._annotation_labels: dict[tuple[object, ...], _AnnotationLabel] = {}
        self._zoom_enabled = False
        self._presentation = ProjectPresentationState()
        self._axis_range = _AxisRangeSession()
        self._refit_axes = (False, False)
        self._axis_display_units: dict[tuple[str, tuple], Unit] = {}
        self._volume_display_unit: Unit | None = None
        self._render_input: ResultRenderInput | None = None
        self._copy_all_available = False
        self._rendered_point_roster: tuple[_RenderedPointBlock, ...] = ()
        self._settling_static_input = False
        self._interactive_timer = QtCore.QTimer(self)
        self._interactive_timer.setSingleShot(True)
        self._interactive_timer.timeout.connect(self._advance_interactive_display)
        self._interactive_items: dict[tuple[object, ...], Any] = {}
        self._interactive_visible: InteractiveDisplayProjection | None = None
        self._interactive_pending: InteractiveDisplayProjection | None = None
        self._interactive_results: dict[str, SimulationResultValue] = {}
        self._interactive_request: InteractivePreviewRequest | None = None
        self._interactive_target_set_ids: tuple[str, ...] = ()
        self._interactive_last_presented_at: float | None = None
        self._interactive_tick_requested = False
        self._interactive_frozen = False
        self._interactive_rendered_frame = False
        self._interactive_projection_sampling_limit: int | None = None
        self._interactive_projection_sampling_known = False

        self._plot_widget = pg.PlotWidget(self, viewBox=_PlotViewBox())
        self._plot_widget.setObjectName("plotViewport")
        self._plot_widget.plotItem.setContentsMargins(0, 0, 0, 10)
        self._plot_widget.setBackground(plot_background_color(False))
        self._plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self._legend = self._plot_widget.addLegend(
            offset=(-10, 10),
            labelTextColor=plot_foreground_color(self._plot_widget.backgroundBrush().color().lightness() < 128),
        )
        self._plot_widget.getPlotItem().setMenuEnabled(False)
        self._plot_widget.getPlotItem().hideButtons()
        self._plot_widget.scene().installEventFilter(self)
        self._plot_widget.scene().sigMouseClicked.connect(self._on_plot_click)
        self._axis_toolbar = AxisToolbar(self)
        self._axis_toolbar.xChanged.connect(self.xSeriesRequested.emit)
        self._axis_toolbar.ySelectionChanged.connect(self.ySeriesRequested.emit)
        self._axis_toolbar.axisRangeChanged.connect(self._apply_axis_range)
        self._axis_toolbar.unitChanged.connect(self._apply_display_unit_change)
        self._plot_widget.getViewBox().sigRangeChangedManually.connect(
            self._on_manual_navigation
        )
        self._plot_widget.getViewBox().sigRangeChanged.connect(
            self._position_annotation_labels
        )

        self._completion_warning_badge = QtWidgets.QLabel(
            "Completed with warnings",
            self,
        )
        self._completion_warning_badge.setObjectName("completedWithWarningsBadge")
        self._completion_warning_badge.setStyleSheet(
            "QLabel { border: 1px solid #b7791f; border-radius: 6px; "
            "padding: 2px 7px; color: #8a5a12; background: #fff7df; }"
        )
        self._create_plot_context_menu()
        self._apply_zoom_mode()
        self._clear_completion_warning()

        plot_host = QtWidgets.QWidget(self)
        plot_layout = QtWidgets.QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(4)
        plot_header = QtWidgets.QWidget(plot_host)
        plot_header_layout = QtWidgets.QHBoxLayout(plot_header)
        plot_header_layout.setContentsMargins(0, 0, 0, 0)
        plot_header_layout.setSpacing(6)
        plot_header_layout.addWidget(self._axis_toolbar, 1)
        plot_header_layout.addWidget(
            self._completion_warning_badge,
            0,
            QtCore.Qt.AlignmentFlag.AlignTop,
        )
        plot_layout.addWidget(plot_header)
        plot_layout.addWidget(self._plot_widget, 1)

        self._parameter_table = ParameterStatisticsTable(self)
        self._species_table = SpeciesStatisticsTable(self)
        self._analysis_tabs = QtWidgets.QTabWidget(self)
        self._analysis_tabs.addTab(self._parameter_table, "Parameters")
        self._analysis_tabs.addTab(self._species_table, "Statistics")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(plot_host)

    def analysis_tabs(self) -> QtWidgets.QTabWidget:
        """Return the panel's analysis surface for the owning workspace stack."""

        return self._analysis_tabs

    def _add_plot_data(self, x, y, **options) -> _PlotDataItem:
        item = _PlotDataItem(x, y, **options)
        self._plot_widget.addItem(item)
        return item

    def apply_result_input(self, render_input: ResultRenderInput) -> None:
        """Retain and draw one immutable result input and its live aggregate."""

        previous_input = self._render_input
        self._rendered_point_roster = ()
        self._settling_static_input = True
        try:
            # Plot membership, not a retained block, keeps an interactive target:
            # a never-run set still shows its first drag Preview.
            plotted_set_ids = {
                set_id for set_id, _label in render_input.plot_set_labels
            }
            removed = tuple(
                set_id
                for set_id in self._interactive_target_set_ids
                if set_id not in plotted_set_ids
            )
            if removed:
                self._retire_interactive_targets(removed)
            self._render_input = render_input
            self._render_result_scene(render_input)
            if self._interactive_results:
                self._recompose_interactive_display()
                if self._interactive_pending is not None:
                    self._present_interactive_projection_now()
            elif (
                self._interactive_visible
                or self._interactive_pending is not None
            ):
                self._reconcile_cached_interactive_display(
                    previous_input,
                    render_input,
                )
            if not (
                self._interactive_results
                or self._interactive_pending
                or self._interactive_visible
                or self._interactive_target_set_ids
            ):
                self._end_interactive_gesture_session()
            if any(self._refit_axes):
                self._fit_axes_once(self._refit_axes)
                self._refit_axes = (False, False)
                self._apply_axis_range_state(self._axis_range, self._presentation)
            if render_input.unavailable_x_diagnostic is not None:
                self.resultDiagnostic.emit(render_input.unavailable_x_diagnostic)
        finally:
            self._settling_static_input = False

    def _reconcile_cached_interactive_display(
        self,
        previous_input: ResultRenderInput | None,
        render_input: ResultRenderInput,
    ) -> None:
        """Reuse only a source-less Preview projection that remains exact.

        Interactive curves are projections of a result, not a second result
        source.  Once the source aggregate has been retired, a cached frame
        can still service display-only changes and narrowing/sample filtering,
        but it cannot answer a new X/Y projection or request a finer sample.
        The existing empty-state transition releases those pixels while the
        request identity and target roster remain admitted for later results.
        """

        projections = (
            (self._interactive_visible,)
            if self._interactive_visible is not None else ()
        )
        if self._interactive_pending is not None:
            projections = (*projections, self._interactive_pending)
        if not projections:
            return
        if not self._cached_interactive_display_compatible(
            previous_input,
            render_input,
            projections,
        ):
            self._clear_interactive_projection()
            return

        visible = (
            self._filter_cached_interactive_projection(
                self._interactive_visible, render_input,
            )
            if self._interactive_visible is not None else None
        )
        if not visible and self._interactive_pending is not None:
            # A pending source-less target has already been projected.  It is
            # safe to expose it as the retained frame after compatibility is
            # established.
            visible = self._filter_cached_interactive_projection(
                self._interactive_pending,
                render_input,
            )
        self._interactive_pending = None
        self._cancel_interactive_tick()
        if not visible:
            self._clear_interactive_projection()
            return
        self._render_interactive_display(visible)
        self._interactive_visible = visible
        self._interactive_projection_sampling_limit = self._sampling_limit
        self._interactive_projection_sampling_known = True

    def _cached_interactive_display_compatible(
        self,
        previous_input: ResultRenderInput | None,
        render_input: ResultRenderInput,
        projections: Sequence[InteractiveDisplayProjection],
    ) -> bool:
        previous_x = (
            None if previous_input is None else previous_input.selected_x_series
        )
        if previous_x is not None and previous_x != render_input.selected_x_series:
            return False

        if (
            previous_input is not None
            and not previous_input.selected_y_uses_default_all
            and render_input.selected_y_uses_default_all
        ):
            return False

        if self._interactive_projection_sampling_known:
            old_limit = self._interactive_projection_sampling_limit
            new_limit = self._sampling_limit
            if old_limit is not None and new_limit is None:
                return False
            if (
                old_limit is not None
                and new_limit is not None
                and new_limit > old_limit
            ):
                return False

        wanted = self._cached_interactive_y_names(render_input)
        for projection in projections:
            names_by_target: dict[str, set[str]] = {}
            for curve in projection.curves:
                key = tuple(curve.key)
                if len(key) < 4 or key[0] != "simulation":
                    continue
                names_by_target.setdefault(str(key[2]), set()).add(key[3])
            for available in names_by_target.values():
                if wanted is not None and not wanted.issubset(available):
                    return False
        return True

    @staticmethod
    def _cached_interactive_y_names(
        render_input: ResultRenderInput,
    ) -> set[str] | None:
        if render_input.selected_y_uses_default_all:
            return None
        return {
            selection
            for selection in render_input.selected_y_series
            if isinstance(selection, (SimulationSeriesAxisSelection, VesselVolumeAxisSelection))
        }

    def _filter_cached_interactive_projection(
        self,
        projection: InteractiveDisplayProjection,
        render_input: ResultRenderInput,
    ) -> InteractiveDisplayProjection | None:
        wanted = self._cached_interactive_y_names(render_input)
        curves: list[InteractiveDisplayCurve] = []
        for curve in projection.curves:
            key = tuple(curve.key)
            if len(key) < 4 or key[0] != "simulation":
                curves.append(curve)
                continue
            if wanted is not None and key[3] not in wanted:
                continue
            if render_input.selected_y_uses_default_all and isinstance(key[3], VesselVolumeAxisSelection):
                continue
            if key[1] == "ghost" and not render_input.reference_lines_visible:
                continue
            curves.append(curve)
        if not curves:
            return None
        target_ids = {
            str(curve.key[2])
            for curve in curves
            if len(curve.key) >= 3 and curve.key[0] == "simulation"
        }
        annotations = tuple(
            annotation
            for annotation in projection.intervention_annotations
            if annotation[1] in target_ids
            and (annotation[0] != "ghost" or render_input.reference_lines_visible)
        )
        filtered = replace(projection, curves=tuple(curves), intervention_annotations=annotations)
        if self._sampling_limit is not None:
            filtered = self._sample_interactive_projection(filtered)
        return filtered

    def apply_interactive_preview_result(
        self,
        tab_id: str,
        set_id: str,
        request: InteractivePreviewRequest,
        result: SimulationResultValue,
    ) -> bool:
        """Admit one transient result against the retained immutable input."""

        render_input = self._render_input
        if render_input is None or render_input.tab_id != str(tab_id):
            return False
        return self._offer_interactive_result(
            result,
            set_id=str(set_id),
            request=request,
            render_input=render_input,
        )

    def set_copy_all_available(self, available: bool) -> None:
        """Project semantic Copy All eligibility into the physical menu."""

        self._copy_all_available = bool(available)
        self._sync_plot_controls()

    def _create_plot_context_menu(self) -> None:
        self._plot_menu = QtWidgets.QMenu(self)
        self._plot_menu.setToolTipsVisible(True)
        self._plot_actions: dict[str, QtGui.QAction] = {}
        self._plot_percentages: dict[str, QtWidgets.QDoubleSpinBox] = {}
        appearance = self._plot_menu.addMenu("Appearance")
        appearance.menuAction().setToolTip(
            "Control grid lines, curve visibility, legends, and intervention annotations."
        )
        transforms = self._plot_menu.addMenu("Transform")
        transforms.menuAction().setToolTip(
            "Show original values, derivatives, or derivatives plotted against Y."
        )
        axes = self._plot_menu.addMenu("Axes")
        axes.menuAction().setToolTip(
            "Adjust axis scaling, navigation, and automatic range settings."
        )
        axis_menus = {axis: axes.addMenu(f"{axis.upper()} Axis") for axis in ("x", "y")}
        axis_menus["x"].menuAction().setToolTip(
            "Control horizontal scaling, direction, and mouse navigation."
        )
        axis_menus["y"].menuAction().setToolTip(
            "Control vertical scaling, direction, and mouse navigation."
        )
        # PySide transfers menu ownership to wrappers returned by menuAction;
        # retain both so inspecting actions cannot destroy a nested menu.
        self._plot_submenus = {
            menu.menuAction(): menu for menu in (appearance, transforms, axes, *axis_menus.values())
        }
        for menu in self._plot_submenus.values():
            menu.setToolTipsVisible(True)
        groups = (
            (self._plot_menu, (("copy_all", "Copy All"), ("copy_plotted", "Copy Plotted Data"),
                              ("copy_visible", "Copy Visible Data"),
                              ("fit_once", "Fit Once"), ("zoom", "Zoom"), ("auto_range", "Auto Range"))),
            (appearance, (("show_grid", "Show Grid"), ("grid_x", "X Grid"), ("grid_y", "Y Grid"),
                          ("simulation_markers", "Simulation Sample Markers"),
                          ("legend_visible", "Show Legend"), ("reference_lines_visible", "Show Reference"),
                          ("intervention_annotations_visible", "Show Intervention Indicators"),
                          ("intervention_labels_visible", "Show Intervention Labels"))),
            (transforms, (("original", "Original"), ("derivative", "Derivative (dy/dx)"),
                          ("phase", "Derivative versus Y"))),
            (axes, (("visible_x", "Fit Y to Visible X"),)),
            (self._plot_menu, (("sampling", "Downsampling: 1k points/curve"), ("warnings", "Warning details…"))),
        )
        for menu, group in groups:
            for key, text in group:
                action = menu.addAction(text)
                action.setCheckable(key not in {"copy_all", "copy_plotted", "copy_visible", "fit_once", "warnings"})
                self._plot_actions[key] = action
        volume_host = QtWidgets.QWidget(appearance)
        volume_layout = QtWidgets.QHBoxLayout(volume_host)
        volume_layout.addWidget(QtWidgets.QLabel("Volume Unit", volume_host))
        self._volume_unit_combo = ScientificNameComboBox(volume_host)
        self._volume_unit_combo.setAccessibleName("Volume Unit")
        self._volume_unit_combo.setToolTip(
            "Use one volume unit for plot axes and intervention labels. Authored uses the unit of the "
            "initial-volume declaration, or L when no compatible unit is declared. "
            "Inflow labels use this volume unit divided by the selected time unit."
        )
        volume_layout.addWidget(self._volume_unit_combo)
        self._volume_unit_action = QtWidgets.QWidgetAction(appearance)
        self._volume_unit_action.setDefaultWidget(volume_host)
        appearance.addAction(self._volume_unit_action)
        self._volume_unit_combo.currentIndexChanged.connect(
            lambda _index: self._apply_volume_unit_change(
                self._volume_unit_combo.currentData()
            )
        )
        for axis, menu in axis_menus.items():
            for key, label in ((f"auto_{axis}", "Auto Range"), (f"mouse_{axis}", "Mouse Navigation"),
                               (f"pan_{axis}", "Auto Pan"), (f"log_{axis}", "Log Scale"),
                               (f"{axis}_inverted", "Invert Axis")):
                action = menu.addAction(label)
                action.setCheckable(True)
                self._plot_actions[key] = action
            self._add_plot_percentage(
                menu, f"fraction_{axis}", "Auto-range coverage", minimum=1,
                tooltip="Choose the central percentage of each curve used to adjust the axis automatically. "
                "Lower percentages can leave extreme values outside the view. "
                "Copy Plotted Data keeps those points, and Copy Visible Data uses the current view.",
            )
        self._add_plot_percentage(
            appearance, "grid_opacity", "Grid opacity",
            tooltip="Adjust how clearly the grid lines appear. At 0%, the grid is invisible.",
        )
        self._add_plot_percentage(
            appearance, "simulation_opacity", "Simulation opacity",
            tooltip="Adjust the visibility of simulation curves and their points. At 100%, references and previews "
            "keep their usual transparency. At 0%, the curves and points are invisible.",
        )
        transform_group = QtGui.QActionGroup(self)
        transform_group.setExclusive(True)
        for transform in ("original", "derivative", "phase"):
            action = self._plot_actions[transform]
            transform_group.addAction(action)
            action.triggered.connect(lambda _checked, value=transform: self._request_plot_presentation("plot_transform", value))
        actions = self._plot_actions
        actions["copy_all"].setToolTip(
            "Copy all available results, references, and linked experimental data for this mechanism tab, "
            "including hidden sets and quantities. Use original values, standard units, and every recorded point."
        )
        actions["copy_plotted"].setToolTip(
            "Copy the selected curves using the current transform, units, and sampling. "
            "Include points outside the visible area."
        )
        actions["copy_visible"].setToolTip(
            "Copy existing plotted samples inside both current axis limits, using the current "
            "transform, units, and sampling. Do not interpolate points at the view boundaries."
        )
        actions["copy_visible"].setEnabled(False)
        actions["zoom"].setToolTip("Drag with the left mouse button to select a zoom area. Turn this off to drag the view instead.")
        actions["auto_range"].setToolTip(
            "Automatically adjust both axes using their coverage and Auto Pan settings. "
            "Turn this off to keep the current limits for manual adjustment."
        )
        actions["show_grid"].setToolTip("Show or hide both grid directions. This is checked when both directions are enabled.")
        actions["grid_x"].setToolTip("Show vertical grid lines at the X axis ticks.")
        actions["grid_y"].setToolTip("Show horizontal grid lines at the Y axis ticks.")
        actions["simulation_markers"].setToolTip(
            "Show points along simulation and reference curves. Downsampling changes which points are shown. "
            "Experimental markers have separate settings."
        )
        actions["original"].setToolTip("Show the selected quantities without calculating derivatives. Log scale settings still apply.")
        actions["derivative"].setToolTip(
            "Show the change in Y divided by the change in X between neighboring plotted points. "
            "Downsampling affects this calculation."
        )
        actions["phase"].setToolTip(
            "Show the numerical derivative against the original Y values for each curve. "
            "Intervention lines, shaded regions, and labels are hidden in this view."
        )
        for axis in ("x", "y"):
            actions[f"auto_{axis}"].setToolTip(
                "Automatically adjust this axis using its coverage and Auto Pan settings. "
                "Turn this off to keep its current limits for manual adjustment."
            )
            actions[f"mouse_{axis}"].setToolTip(
                "Allow mouse dragging and scrolling to move or scale this axis. Turn this off to lock mouse movement "
                "on this axis. Automatic adjustments can still move it."
            )
            actions[f"pan_{axis}"].setToolTip(
                "Keep the current range width and move its center to follow the data. Requires Auto Range. "
                "Coverage controls how much of the data determines the center."
            )
            actions[f"log_{axis}"].setToolTip(
                "Use a logarithmic scale. Only positive values can appear, and manual limits must be positive. "
                "Invalid manual limits are fitted again where data is available. The axis stays manual."
            )
        actions["legend_visible"].setToolTip("Show curve names and colors.")
        actions["reference_lines_visible"].setToolTip(
            "Show curves from the accepted model alongside Slider previews."
        )
        actions["intervention_annotations_visible"].setToolTip(
            "Show intervention lines and shaded regions. Changing this setting also turns intervention labels on or off."
        )
        actions["intervention_labels_visible"].setToolTip(
            "Show intervention labels that you can drag to a new position. "
            "Turning this on also shows the intervention lines and shaded regions."
        )
        actions["x_inverted"].setToolTip("Reverse the direction of this axis.")
        actions["y_inverted"].setToolTip("Reverse the direction of this axis.")
        actions["sampling"].setToolTip(
            "Display about 1,000 points per curve while preserving intervention boundaries. "
            "This affects displayed derivatives, Copy Plotted Data, and Copy Visible Data. Copy All keeps every point."
        )
        actions["warnings"].setToolTip("Read the warnings for the current plot.")
        actions["copy_all"].triggered.connect(self.copyAllRequested.emit)
        actions["copy_plotted"].triggered.connect(self._copy_rendered_points_to_clipboard)
        actions["copy_visible"].triggered.connect(self._copy_visible_points_to_clipboard)
        actions["zoom"].triggered.connect(self._set_zoom_enabled)
        actions["auto_range"].triggered.connect(self._set_auto_range)
        actions["fit_once"].triggered.connect(self._fit_once)
        for key in ("auto_x", "auto_y", "mouse_x", "mouse_y", "pan_x", "pan_y", "visible_x"):
            actions[key].triggered.connect(lambda enabled, option=key: self._set_navigation(option, enabled))
        actions["visible_x"].setToolTip(
            "Automatically adjust Y to the data within the visible X range. X stays manual and Y Auto Pan turns off. "
            "Moving Y manually, enabling X Auto Range, or enabling Y Auto Pan ends this mode."
        )
        actions["fit_once"].setToolTip("Fit all displayed data once, then leave both axes under manual control.")
        actions["reference_lines_visible"].triggered.connect(self.referenceLayerVisibilityRequested)
        actions["intervention_annotations_visible"].triggered.connect(self.interventionAnnotationsVisibilityRequested)
        actions["intervention_labels_visible"].triggered.connect(self.interventionLabelsVisibilityRequested)
        for option in ("legend_visible", "log_x", "log_y", "x_inverted", "y_inverted", "show_grid", "grid_x", "grid_y", "simulation_markers"):
            actions[option].triggered.connect(
                lambda enabled, key=option: self._request_plot_presentation(key, enabled)
            )
        actions["sampling"].triggered.connect(self._set_coarse_sampling)
        actions["warnings"].triggered.connect(self._show_completion_warning_details)
        self._plot_menu.aboutToShow.connect(self._sync_plot_controls)
        self._plot_menu.aboutToShow.connect(self._refresh_visible_copy_availability)

    def _add_plot_percentage(self, menu: QtWidgets.QMenu, key: str, label: str, *, tooltip: str, minimum: int = 0) -> None:
        host = QtWidgets.QWidget(menu)
        layout = QtWidgets.QHBoxLayout(host)
        caption = QtWidgets.QLabel(label, host)
        layout.addWidget(caption)
        spin = QtWidgets.QDoubleSpinBox(host)
        spin.setRange(minimum, 100)
        spin.setDecimals(0)
        spin.setSuffix(" %")
        spin.setKeyboardTracking(False)
        spin.setAccessibleName(label)
        layout.addWidget(spin)
        action = QtWidgets.QWidgetAction(menu)
        action.setDefaultWidget(host)
        for target in (action, host, caption, spin):
            target.setToolTip(tooltip)
        menu.addAction(action)
        self._plot_percentages[key] = spin
        if key.startswith("fraction_"):
            spin.valueChanged.connect(lambda value: self._set_navigation(key, value / 100))
        else:
            spin.valueChanged.connect(lambda value: self._request_plot_presentation(key, value / 100))

    def _sync_plot_controls(self) -> None:
        """Project accepted presentation and plot-session navigation."""
        state = self._axis_range
        values = {
            "zoom": self._zoom_enabled,
            "auto_range": state.auto_x and state.auto_y,
            "sampling": self._sampling_limit is not None,
            "show_grid": self._presentation.grid_x and self._presentation.grid_y,
        }
        for key in ("legend_visible", "log_x", "log_y", "x_inverted", "y_inverted", "grid_x", "grid_y", "simulation_markers"):
            values[key] = getattr(self._presentation, key)
        for key in ("auto_x", "auto_y", "mouse_x", "mouse_y", "pan_x", "pan_y", "visible_x"):
            values[key] = getattr(state, key)
        for key in ("original", "derivative", "phase"):
            values[key] = self._presentation.plot_transform == key
        for key in ("reference_lines_visible", "intervention_annotations_visible", "intervention_labels_visible"):
            values[key] = bool(self._render_input and getattr(self._render_input, key))
        for key, checked in values.items():
            self._plot_actions[key].setChecked(checked)
        for key, spin in self._plot_percentages.items():
            with QtCore.QSignalBlocker(spin):
                spin.setValue(100 * getattr(state if key.startswith("fraction_") else self._presentation, key))
            if key.startswith("fraction_"):
                spin.setEnabled(not self._interactive_gesture_active)
        self._plot_actions["copy_all"].setEnabled(self._copy_all_available)
        self._plot_actions["copy_plotted"].setEnabled(
            any(chunk.x.size for block in self._rendered_point_roster for chunk in block.chunks)
        )
        self._plot_actions["warnings"].setEnabled(bool(self._completion_warning_badge.text()))
        for key in ("auto_range", "auto_x", "auto_y", "fit_once", "visible_x", "pan_x", "pan_y"):
            self._plot_actions[key].setEnabled(not self._interactive_gesture_active)
        self._axis_toolbar.set_manual_range_controls(
            auto_x=state.auto_x, auto_y=state.auto_y,
            enabled=not self._interactive_gesture_active,
        )
        units = self._volume_unit_choices()
        with QtCore.QSignalBlocker(self._volume_unit_combo):
            self._volume_unit_combo.clear()
            self._volume_unit_combo.addItem("Authored", "Authored")
            for unit in units:
                self._volume_unit_combo.addItem(unit.symbol, unit.symbol)
                self._volume_unit_combo.setItemData(
                    self._volume_unit_combo.count() - 1,
                    format_unit_for_display(unit),
                    SCIENTIFIC_NAME_HTML_ROLE,
                )
            self._volume_unit_combo.setCurrentIndex(
                self._volume_unit_combo.findData(
                    self._volume_display_unit.symbol if self._volume_display_unit is not None else "Authored"
                )
            )
        available = self._render_input is not None and self._render_input.authored_volume_unit is not None
        self._volume_unit_action.setEnabled(available)

    def _on_plot_click(self, event: object) -> None:
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            event.accept()
            self._plot_menu.popup(event.screenPos().toPoint())

    def eventFilter(self, watched: object, event: QtCore.QEvent) -> bool:
        if watched is self._plot_widget.scene() and event.type() == QtCore.QEvent.Type.GraphicsSceneContextMenu:
            # Mouse menus belong to the scene's click/drag recognizer.
            if event.reason() != QtWidgets.QGraphicsSceneContextMenuEvent.Reason.Mouse:
                self._plot_menu.popup(event.screenPos())
            event.accept()
            return True
        return super().eventFilter(watched, event)

    @QtCore.Slot()
    def _refresh_visible_copy_availability(self) -> None:
        self._plot_actions["copy_visible"].setEnabled(_has_visible_samples(
            self._rendered_point_roster,
            self._plot_widget.getViewBox().viewRange(),
            log_x=self._presentation.log_x, log_y=self._presentation.log_y,
        ))

    @QtCore.Slot()
    def _copy_visible_points_to_clipboard(self) -> None:
        roster = self._rendered_point_roster
        view_range = self._plot_widget.getViewBox().viewRange()
        log_x, log_y = self._presentation.log_x, self._presentation.log_y
        try:
            visible = _visible_point_roster(roster, view_range, log_x=log_x, log_y=log_y)
            if not visible:
                self.resultDiagnostic.emit("No plotted samples are inside the current view.")
                return
            output = io.StringIO()
            output.writelines(_iter_rendered_point_tsv_chunks(visible, action_name="Copy Visible Data"))
            rendered_text = output.getvalue()
        except (TypeError, ValueError, OSError) as exc:
            QtWidgets.QMessageBox.critical(self, "Copy Visible Data", str(exc))
            return
        QtWidgets.QApplication.clipboard().setText(rendered_text)

    @QtCore.Slot()
    def _copy_rendered_points_to_clipboard(self) -> None:
        roster = self._rendered_point_roster
        try:
            if not any(chunk.x.size for block in roster for chunk in block.chunks):
                raise ValueError("No plotted data is available to copy.")
            output = io.StringIO()
            output.writelines(_iter_rendered_point_tsv_chunks(roster))
            rendered_text = output.getvalue()
        except (TypeError, ValueError, OSError) as exc:
            QtWidgets.QMessageBox.critical(self, "Copy Plotted Data", str(exc))
            return
        QtWidgets.QApplication.clipboard().setText(rendered_text)

    def retire_interactive_preview_targets(
        self,
        target_set_ids: tuple[str, ...],
    ) -> bool:
        """Retire selected transient targets from the physical display."""

        retired = self._retire_interactive_targets(
            tuple(str(value) for value in target_set_ids),
        )
        if retired:
            retired = self._retire_static_preview_targets(target_set_ids) or retired
        else:
            retired = self._retire_static_preview_targets(target_set_ids)
        return retired

    def retire_interactive_preview_target(
        self,
        tab_id: str,
        set_id: str,
        request: InteractivePreviewRequest,
    ) -> bool:
        """Retire one current failed Preview target and restore its static data."""

        if self._render_input is None or self._render_input.tab_id != tab_id:
            return False
        if self._interactive_request is not request:
            self._supersede_interactive_request(
                request,
                tuple(request.target_set_ids),
            )
        return self._retire_interactive_targets((set_id,))

    def _retire_static_preview_targets(
        self,
        target_set_ids: tuple[str, ...],
    ) -> bool:
        targets = {str(value) for value in target_set_ids}
        if not targets:
            return False
        retired = False
        for item_key, item in tuple(self._plot_items_by_key.items()):
            role = item_key[0] if item_key else None
            if (
                not isinstance(role, tuple)
                or len(role) != 2
                or role[0]
                not in {ResultLayerKind.PREVIEW.value, ResultLayerKind.REFERENCE.value}
                or str(role[1]) not in targets
            ):
                continue
            self._remove_plot_item(item)
            del self._plot_items_by_key[item_key]
            retired = True
        prior_roster = self._rendered_point_roster
        self._rendered_point_roster = tuple(
            block
            for block in prior_roster
            if not (
                block.kind
                in {
                    ResultLayerKind.PREVIEW.value,
                    ResultLayerKind.REFERENCE.value,
                }
                and block.block_id in targets
            )
        )
        retired = retired or len(prior_roster) != len(self._rendered_point_roster)
        for annotation_key, item in tuple(self._annotation_items.items()):
            if annotation_key[1] not in targets:
                continue
            self._remove_annotation_item(item)
            del self._annotation_items[annotation_key]
            retired = True
        return retired

    def stop_interactive_display(self) -> bool:
        """Cancel visual work and release transient physical custody."""

        if (
            not self._interactive_has_visible_frame
            and not self._interactive_has_pending_batch
        ):
            return False
        self._clear_interactive_display(reset_currentness=True)
        return True

    def set_persisted_plot_state(self, state: ProjectPresentationState) -> None:
        """Project the registry's accepted choices; the next render refits changed coordinates."""
        old = self._presentation.plot_transform
        new = state.plot_transform
        if old != new:
            self._refit_axes = (
                self._refit_axes[0] or "phase" in (old, new),
                self._refit_axes[1] or "original" in (old, new),
            )
        for index, axis in enumerate(("x", "y")):
            if (
                getattr(state, f"log_{axis}")
                and not getattr(self._presentation, f"log_{axis}")
                and not getattr(self._axis_range, f"auto_{axis}")
            ):
                low = getattr(self._axis_range, f"{axis}_min")
                high = getattr(self._axis_range, f"{axis}_max")
                if low is None or high is None or not 0 < low < high:
                    self._refit_axes = tuple(
                        pending or current == index
                        for current, pending in enumerate(self._refit_axes)
                    )
        self._presentation = state
        self._apply_plot_presentation(state)

    def reset_plot_session_state(self) -> None:
        """Project replacement/reopen resets non-archived Plot session truth."""

        self.dispose_project_content()
        self._end_interactive_gesture_session()
        self._zoom_enabled = False
        self._presentation = ProjectPresentationState()
        self._sampling_limit = None
        self._axis_range = _AxisRangeSession()
        self._refit_axes = (False, False)
        self._apply_axis_range_state(self._axis_range, self._presentation)
        self._axis_display_units: dict[tuple[str, tuple], Unit] = {}
        self._volume_display_unit = None
        self._axis_toolbar.set_axis_unit_choices(
            x_units=tuple(
                (unit, format_unit_for_display(unit))
                for unit in TIME_DISPLAY_UNITS
            ),
            selected_x="s",
            y_units=(),
            selected_y=None,
        )
        self._axis_toolbar.set_manual_ranges(None, None, None, None)
        self._sync_plot_controls()
        self._apply_zoom_mode()

    def dispose_project_content(self) -> None:
        """Stop plot-owned asynchronous work before project content teardown."""

        self._render_input = None
        self._copy_all_available = False
        self._rendered_point_roster = ()
        self._clear_interactive_display(reset_currentness=True)
        self._clear_annotation_items()

    def clear_result_input(self) -> None:
        """Clear the current result scene and its transient display."""

        self._render_input = None
        self._copy_all_available = False
        self._rendered_point_roster = ()
        self.discard_interactive_display()
        self._end_interactive_gesture_session()
        self._clear_plot_items()
        self._parameter_table.setRowCount(0)
        self._species_table.setRowCount(0)
        self._clear_completion_warning()

    @property
    def _interactive_has_visible_frame(self) -> bool:
        return bool(
            self._interactive_visible
            and self._interactive_visible.curves
        )

    @property
    def _interactive_has_pending_batch(self) -> bool:
        return bool(
            self._interactive_results
            or self._interactive_pending
        )

    def _offer_interactive_result(
        self,
        result: SimulationResultValue,
        *,
        set_id: str,
        request: InteractivePreviewRequest,
        render_input: ResultRenderInput,
    ) -> bool:
        """Aggregate out-of-order results for one request identity."""

        self._render_input = render_input
        set_id = str(set_id)
        roster = tuple(dict.fromkeys(str(value) for value in request.target_set_ids))
        if set_id not in roster:
            return False
        if self._interactive_request is None:
            self._interactive_request = request
            self._interactive_target_set_ids = roster
            self._interactive_frozen = False
        elif request is not self._interactive_request:
            self._supersede_interactive_request(request, roster)
        elif set_id not in self._interactive_target_set_ids:
            return False
        if self._interactive_frozen:
            return False
        self._interactive_results[set_id] = result
        self._recompose_interactive_display()
        return True

    def _supersede_interactive_request(
        self,
        request: InteractivePreviewRequest,
        target_set_ids: tuple[str, ...],
    ) -> None:
        """Replace aggregate authority while retaining the last rendered frame."""

        # Keep the last drawn frame until a current result replaces it. An
        # already scheduled tick must keep its deadline: restarting it for
        # every new request can starve drawing during a fast drag.
        self._interactive_results.clear()
        self._interactive_pending = None
        self._interactive_target_set_ids = target_set_ids
        self._interactive_request = request
        self._interactive_frozen = False

    def _offer_interactive_projection(
        self,
        projection: InteractiveDisplayProjection,
    ) -> None:
        self._interactive_pending = projection
        self._schedule_interactive_tick()

    def _tick_interactive_display(self) -> None:
        self._interactive_tick_requested = False
        now = float(time.monotonic())
        if self._interactive_pending is not None and self._interactive_cadence_due(now):
            self._present_interactive_projection_now()
        if self._interactive_pending is not None:
            self._schedule_interactive_tick()

    def _clear_interactive_display(self, *, reset_currentness: bool = False) -> None:
        had_physical_state = bool(
            self._interactive_visible
            or self._interactive_rendered_frame
            or self._interactive_pending
        )
        if had_physical_state:
            self._render_interactive_display(None)
            self._interactive_visible = None
            self._interactive_rendered_frame = False
            self._interactive_last_presented_at = None
            self._interactive_projection_sampling_limit = None
            self._interactive_projection_sampling_known = False
        self._interactive_pending = None
        self._interactive_results.clear()
        self._interactive_target_set_ids = ()
        self._interactive_frozen = True
        self._interactive_projection_sampling_limit = None
        self._interactive_projection_sampling_known = False
        if reset_currentness:
            self._interactive_request = None
        self._cancel_interactive_tick()

    def _retire_interactive_targets(self, target_set_ids: tuple[str, ...]) -> bool:
        targets = {str(value) for value in target_set_ids}
        if not targets:
            return False
        known = set(self._interactive_target_set_ids)
        known.update(self._interactive_results)
        known.update(self._interactive_projection_target_ids(self._interactive_visible))
        known.update(self._interactive_projection_target_ids(self._interactive_pending))
        affected = targets.intersection(known)
        if not affected:
            return True
        visible = (
            self._filter_interactive_projection(self._interactive_visible, affected)
            if self._interactive_visible is not None else None
        )
        if visible and self._render_input is not None:
            # Restore every retained item from semantic input first; rendering
            # the surviving transient targets will suppress only those targets.
            self._render_result_scene(self._render_input)
        self._render_interactive_display(visible)
        self._interactive_visible = visible or None
        self._interactive_pending = None
        self._cancel_interactive_tick()
        self._interactive_results = {
            set_id: result
            for set_id, result in self._interactive_results.items()
            if set_id not in affected
        }
        self._interactive_target_set_ids = tuple(
            set_id
            for set_id in self._interactive_target_set_ids
            if set_id not in affected
        )
        self._interactive_frozen = not bool(
            self._interactive_results
            or self._interactive_visible
            or self._interactive_target_set_ids
        )
        if not self._interactive_visible:
            self._interactive_projection_sampling_limit = None
            self._interactive_projection_sampling_known = False
        if self._interactive_results:
            self._recompose_interactive_display()
        return True

    def _recompose_interactive_display(self) -> None:
        if not self._interactive_results:
            self._clear_interactive_projection()
            return
        projection = self._compose_interactive_batch()
        if projection is None:
            self._clear_interactive_projection()
            return
        self._offer_interactive_projection(
            self._sample_interactive_projection(projection)
        )

    def _clear_interactive_projection(self) -> None:
        """Remove transient pixels while retaining the current request facts."""

        retained_static_roster = (
            self._rendered_point_roster if self._settling_static_input else None
        )
        self._interactive_pending = None
        self._cancel_interactive_tick()
        if self._interactive_visible or self._interactive_rendered_frame:
            self._render_interactive_display(None)
        self._interactive_visible = None
        self._interactive_rendered_frame = False
        self._interactive_projection_sampling_limit = None
        self._interactive_projection_sampling_known = False
        if retained_static_roster is not None:
            self._rendered_point_roster = retained_static_roster

    def _present_interactive_projection_now(self) -> None:
        """Present the latest semantic projection without waiting for cadence."""

        projection = self._interactive_pending
        if projection is None:
            return
        self._interactive_pending = None
        self._cancel_interactive_tick()
        # Draw a completed result directly; interpolation between successive
        # results adds lag and can restart indefinitely during dragging.
        self._render_interactive_display(projection)
        self._interactive_visible = projection
        self._interactive_last_presented_at = float(time.monotonic())
        self._interactive_projection_sampling_limit = self._sampling_limit
        self._interactive_projection_sampling_known = True

    def _compose_interactive_batch(self) -> InteractiveDisplayProjection | None:
        if self._render_input is None or not self._interactive_results:
            return None
        curves: list[InteractiveDisplayCurve] = []
        annotations: list[_DisplayAnnotation] = []
        latest: InteractiveDisplayProjection | None = None
        for set_id, result in self._interactive_results.items():
            if set_id not in self._interactive_target_set_ids:
                continue
            projection = self._compose_interactive_result(result, set_id=set_id)
            if projection is None:
                continue
            latest = projection
            curves.extend(projection.curves)
            annotations.extend(projection.intervention_annotations)
        if latest is None or not curves:
            return None
        return replace(latest, curves=tuple(curves), intervention_annotations=tuple(annotations))

    def _compose_interactive_result(
        self,
        result: SimulationResultValue,
        *,
        set_id: str,
    ) -> InteractiveDisplayProjection | None:
        request = self._interactive_request
        if self._render_input is None or request is None:
            return None
        target = next(
            (item for item in request.targets if item.set_id == set_id),
            None,
        )
        if target is None:
            return None
        return compose_interactive_display(
            result,
            tab_id=self._render_input.tab_id,
            set_id=set_id,
            render_input=self._render_input,
            prepared_dsl=target.execution_request.prepared_dsl,
        )

    def _sample_interactive_projection(
        self,
        projection: InteractiveDisplayProjection,
    ) -> InteractiveDisplayProjection:
        return replace(
            projection,
            curves=tuple(
                self._sample_interactive_curve(curve)
                for curve in projection.curves
            ),
        )

    def _sample_interactive_curve(
        self,
        curve: InteractiveDisplayCurve,
    ) -> InteractiveDisplayCurve:
        count = int(curve.x.size)
        sample = self._sample_indices(count, self._sampling_limit)
        if count:
            endpoints = np.asarray(
                [index for start, end in self._interactive_curve_chunks(curve)
                 for index in (start, end - 1)],
                dtype=int,
            )
            sample = np.unique(np.concatenate((sample, endpoints)))
        physical_chunks = []
        for chunk in curve.physical_chunks:
            if chunk.instantaneous:
                physical_chunks.append(chunk)
                continue
            keep = np.isin(chunk.time, curve.sample_time[sample])
            if len(keep):
                keep[0] = keep[-1] = True
            physical_chunks.append(replace(chunk, time=chunk.time[keep], x=chunk.x[keep],
                series=tuple((name, values[keep]) for name, values in chunk.series)))
        return replace(
            curve,
            physical_chunks=tuple(physical_chunks),
            x=curve.x[sample],
            y=curve.y[sample],
            sample_time=curve.sample_time[sample],
            segment_ids=(
                None
                if curve.segment_ids is None
                else curve.segment_ids[sample]
            ),
            segment_keys=(
                None
                if curve.segment_keys is None
                else tuple(curve.segment_keys[int(index)] for index in sample)
            ),
            discontinuity_ids=(
                None
                if curve.discontinuity_ids is None
                else curve.discontinuity_ids[sample]
            ),
        )

    @staticmethod
    def _interactive_projection_target_ids(
        projection: InteractiveDisplayProjection | None,
    ) -> set[str]:
        if projection is None:
            return set()
        return {
            str(curve.key[2])
            for curve in projection.curves
            if len(curve.key) >= 3
        } | {annotation[1] for annotation in projection.intervention_annotations}

    @staticmethod
    def _filter_interactive_projection(
        projection: InteractiveDisplayProjection,
        removed: set[str],
    ) -> InteractiveDisplayProjection | None:
        curves = tuple(
            curve
            for curve in projection.curves
            if len(curve.key) < 3 or str(curve.key[2]) not in removed
        )
        annotations = tuple(
            annotation
            for annotation in projection.intervention_annotations
            if annotation[1] not in removed
        )
        if not curves:
            return None
        return replace(projection, curves=curves, intervention_annotations=annotations)

    def _schedule_interactive_tick(self) -> None:
        if self._interactive_tick_requested:
            return
        self._interactive_tick_requested = True
        self._request_interactive_tick(1.0 / 30.0)

    def _cancel_interactive_tick(self) -> None:
        self._interactive_tick_requested = False
        self._interactive_timer.stop()

    def _interactive_cadence_due(self, now: float) -> bool:
        return (
            self._interactive_last_presented_at is None
            or now - self._interactive_last_presented_at >= (1.0 / 30.0) - _EPSILON
        )

    @staticmethod
    def _annotation_signature(
        annotation: _DisplayAnnotation,
    ) -> _AnnotationKey:
        role, set_id, payload, _opacity = annotation
        return (
            role,
            set_id,
            tuple(sorted((str(name), repr(value)) for name, value in payload.items())),
        )

    @staticmethod
    def _annotation_accent(annotation: Mapping[str, object]) -> QtGui.QColor:
        authored = annotation.get("authored_ordinal")
        index = int(annotation["declaration_ordinal"]) + (
            0 if authored is None else int(authored)
        )
        return QtGui.QColor(_ANNOTATION_ACCENTS[index % len(_ANNOTATION_ACCENTS)])

    def _request_interactive_tick(self, delay_seconds: float) -> None:
        milliseconds = max(0, int(np.ceil(float(delay_seconds) * 1000.0)))
        self._interactive_timer.start(milliseconds)

    @QtCore.Slot()
    def _advance_interactive_display(self) -> None:
        self._tick_interactive_display()

    def _begin_interactive_gesture_session(self) -> None:
        self._interactive_gesture_active = True
        self._sync_plot_controls()

    def _end_interactive_gesture_session(self) -> None:
        self._interactive_gesture_active = False
        self._sync_plot_controls()

    def _render_interactive_display(
        self,
        projection: InteractiveDisplayProjection | None,
    ) -> None:
        """Apply the projection, including a valid frame with no drawable samples."""
        self._rendered_point_roster = ()
        if projection is None:
            self._release_interactive_display()
            self._interactive_rendered_frame = False
            return

        existing = dict(self._interactive_items)
        rendered: dict[tuple[object, ...], Any] = {}
        rendered_points: list[_RenderedPointBlock] = []
        render_input = self._render_input
        transient_set_ids = {
            str(curve.key[2])
            for curve in projection.curves
            if len(curve.key) >= 3
            and curve.key[0:2] == ("simulation", "foreground")
        }
        transient_reference_set_ids = {
            str(curve.key[2])
            for curve in projection.curves
            if len(curve.key) >= 3
            and curve.key[0:2] == ("simulation", "ghost")
        }
        replaced_annotation_owners = (
            {("foreground", set_id) for set_id in transient_set_ids}
            | {("ghost", set_id) for set_id in transient_reference_set_ids}
            | {(role, set_id)
            for role, set_id, _payload, _opacity
            in projection.intervention_annotations}
        )
        if render_input is not None:
            for block in render_input.blocks:
                if not block.visible:
                    continue
                if block.kind is ResultLayerKind.EXPLICIT and str(block.block_id) in transient_set_ids:
                    continue
                if block.kind is ResultLayerKind.PREVIEW and str(block.block_id) in transient_set_ids:
                    continue
                if block.kind is ResultLayerKind.REFERENCE and str(block.block_id) in transient_reference_set_ids:
                    continue
                rendered_block, _chunks = self._rendered_block_for_static(
                    block,
                    self._sampling_limit,
                )
                if any(chunk.x.size for chunk in rendered_block.chunks):
                    rendered_points.extend(self._transformed_blocks(rendered_block))
        for curve in projection.curves:
            if render_input is None:
                continue
            point_block = self._rendered_block_for_interactive_curve(curve)
            if point_block is None:
                continue
            for transformed in self._transformed_blocks(point_block):
                for index, chunk in enumerate(transformed.chunks):
                    y = chunk.series[0][1]
                    if not np.any(np.isfinite(chunk.x) & np.isfinite(y)):
                        continue
                    item_key = (*curve.key, "physical", index)
                    pen = pg.mkPen(color=curve.color, width=2, style=(QtCore.Qt.PenStyle.DashLine if curve.dashed else QtCore.Qt.PenStyle.SolidLine))
                    item = existing.get(item_key)
                    if item is None:
                        item = self._add_plot_data(chunk.x, y, pen=pen, name=None, connect="finite")
                    else:
                        item.setData(chunk.x, y, connect="finite")
                        item.setPen(pen)
                    self._style_simulation_item(item, True, curve.color, curve.opacity, singleton=chunk.x.size == 1)
                    rendered[item_key] = item
        for item_key, item in existing.items():
            if item_key not in rendered:
                self._remove_interactive_plot_item(item)
        self._suppress_static_simulation_items(
            transient_set_ids,
            transient_reference_set_ids,
        )
        self._interactive_items = rendered
        if render_input is not None:
            static_annotations = tuple(
                annotation
                for annotation in render_input.intervention_annotations
                if annotation[:2] not in replaced_annotation_owners
            )
            interactive_annotations = tuple(
                (role, set_id, payload, opacity)
                for role, set_id, payload, opacity
                in projection.intervention_annotations
                if opacity > _EPSILON
            )
            self._render_annotations(
                render_input,
                self._presentation,
                annotations=(*static_annotations, *interactive_annotations),
            )
        # The displayed projection contains completed results, so Copy uses
        # those same sampled curves and their corresponding time.
        if render_input is not None:
            for curve in projection.curves:
                rendered_block = self._rendered_block_for_interactive_curve(
                    curve,
                )
                if rendered_block is not None and any(
                    chunk.x.size for chunk in rendered_block.chunks
                ):
                    rendered_points.extend(self._transformed_blocks(rendered_block))
        self._rendered_point_roster = tuple(rendered_points)
        if rendered and not self._interactive_rendered_frame:
            self._begin_interactive_gesture_session()
        self._interactive_rendered_frame = bool(rendered)

    def _remove_interactive_plot_item(self, item: object) -> None:
        """Remove one stale transient item from the physical plot."""

        if self._legend is not None:
            try:
                self._legend.removeItem(item)
            except (KeyError, RuntimeError, TypeError, ValueError):
                pass
        self._plot_widget.getPlotItem().removeItem(item)

    @staticmethod
    def _interactive_curve_chunks(
        curve: object,
    ) -> tuple[tuple[int, int], ...]:
        """Split a transient curve at every executor-issued topology change."""

        x = curve.x
        count = int(x.size)
        if count == 0:
            return ()
        segment_ids = curve.segment_ids
        segment_keys = curve.segment_keys
        discontinuities = curve.discontinuity_ids
        if segment_ids is None or segment_keys is None or discontinuities is None:
            raise ValueError("Interactive curve is missing executor topology.")
        changes = tuple(
            index for index in range(1, count)
            if (
                int(segment_ids[index]) != int(segment_ids[index - 1])
                or segment_keys[index] != segment_keys[index - 1]
                or int(discontinuities[index]) != int(discontinuities[index - 1])
            )
        )
        starts = [0, *changes]
        return tuple(
            (
                start,
                starts[position + 1] if position + 1 < len(starts) else count,
            )
            for position, start in enumerate(starts)
        )

    def _clear_interactive_items(self) -> None:
        for item in tuple(self._interactive_items.values()):
            self._remove_plot_item(item)
        self._interactive_items.clear()

    def _release_interactive_display(self) -> None:
        """Release all physical transient custody at one terminal boundary.

        The panel's transient state invokes this operation by presenting an
        empty state. External scene/lifecycle boundaries call it after
        clearing or superseding that state. Retained simulation items and
        their named legend rows stay attached throughout; the canonical static
        render below restores their visible presentation directly.
        """

        self._cancel_interactive_tick()
        self._clear_interactive_items()
        self._end_interactive_gesture_session()
        self._apply_axis_range_state(self._axis_range, self._presentation)
        if self._render_input is not None and not self._settling_static_input:
            self._render_result_scene(self._render_input)

    def _suppress_static_simulation_items(
        self,
        transient_set_ids: object,
        transient_reference_set_ids: object = (),
    ) -> None:
        """Clear retained target pixels without changing physical membership."""

        transient = {str(set_id) for set_id in transient_set_ids}
        transient_reference = {
            str(set_id) for set_id in transient_reference_set_ids
        }
        static_kinds = {
            ResultLayerKind.EXPLICIT.value,
            ResultLayerKind.PREVIEW.value,
            ResultLayerKind.REFERENCE.value,
        }
        for item_key, item in self._plot_items_by_key.items():
            if not item_key or not isinstance(item_key[0], tuple):
                continue
            role = item_key[0]
            if len(role) != 2 or role[0] not in static_kinds:
                continue
            target_set_id = str(role[1])
            if role[0] == ResultLayerKind.REFERENCE.value:
                if target_set_id not in transient_reference:
                    continue
            elif target_set_id not in transient:
                continue
            item.setData([], [])
            if role[0] == ResultLayerKind.REFERENCE.value:
                self._update_plot_item_legend(item, None)

    def discard_interactive_display(self) -> None:
        self._clear_interactive_display(reset_currentness=True)

    def _show_completion_warning_details(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Completion Warning Details")
        layout = QtWidgets.QVBoxLayout(dialog)
        details = QtWidgets.QPlainTextEdit(dialog)
        details.setReadOnly(True)
        details.setAccessibleName("Completion warning details")
        details.setPlainText(self._completion_warning_badge.accessibleDescription())
        layout.addWidget(details)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        clamp_initial_size(dialog, 560, 420)
        dialog.exec()

    def _clear_completion_warning(self) -> None:
        self._completion_warning_badge.setText("")
        self._completion_warning_badge.setToolTip("")
        self._completion_warning_badge.setAccessibleName("")
        self._completion_warning_badge.setAccessibleDescription("")
        self._completion_warning_badge.setVisible(False)
        self._sync_plot_controls()

    @staticmethod
    def _has_time_x(render_input: ResultRenderInput) -> bool:
        selection = render_input.selected_x_series
        return (
            isinstance(selection, SimulationSeriesAxisSelection)
            and selection.series_name == "t"
        )

    def _selected_display_unit(self, axis: str, source: Unit | None) -> Unit | None:
        if source is None:
            return None
        if source.compatible(CUBIC_METRE) and self._render_input is not None and self._render_input.authored_volume_unit is not None:
            return self._selected_volume_unit()
        return self._axis_display_units.get(
            (axis, source.dimensions), presentation_unit(scientific_unit(source.dimensions)),
        )

    def _selected_volume_unit(self) -> Unit:
        authored = self._render_input.authored_volume_unit if self._render_input is not None else None
        return self._volume_display_unit or authored or presentation_unit(CUBIC_METRE)

    def _volume_unit_choices(self) -> tuple[Unit, ...]:
        units = self._render_input.volume_unit_options if self._render_input is not None else ()
        return reconcile_unit_choices(units, self._volume_display_unit)

    def _display_unit_choices(self, render_input: ResultRenderInput, axis: str) -> tuple[Unit, ...]:
        source = render_input.x_unit if axis == "x" else render_input.y_unit
        selected = self._selected_display_unit(axis, source)
        if selected is None:
            return ()
        if source.compatible(CUBIC_METRE) and render_input.authored_volume_unit is not None:
            return self._volume_unit_choices() or (selected,)
        options = render_input.x_unit_options if axis == "x" else render_input.y_unit_options
        return reconcile_unit_choices(options, selected)

    def _display_unit_state(
        self,
        render_input: ResultRenderInput,
    ) -> tuple[str | None, str | None, float, float]:
        def selected(axis, unit):
            shown = self._selected_display_unit(axis, unit)
            if shown is None:
                return None, 1.0
            return shown.symbol, conversion_scale(unit, shown).number()
        x_unit, x_scale = selected("x", render_input.x_unit)
        y_unit, y_scale = selected("y", render_input.y_unit)
        return x_unit, y_unit, x_scale, y_scale

    def _curve_unit_state(self, axis: str, source: Unit | None) -> tuple[str | None, float]:
        """Keep mixed quantities scientific except for the shared volume choice."""
        if source is None:
            return None, 1.0
        render_input = self._render_input
        common = None if render_input is None else render_input.x_unit if axis == "x" else render_input.y_unit
        if common is not None:
            shown = self._selected_display_unit(axis, common)
        elif source.compatible(CUBIC_METRE) and render_input is not None and render_input.authored_volume_unit is not None:
            shown = self._selected_volume_unit()
        else:
            shown = source
        return shown.symbol, conversion_scale(source, shown).number()

    def _configure_axis_unit_controls(self, render_input: ResultRenderInput) -> None:
        x_unit, y_unit, _x_scale, _y_scale = self._display_unit_state(render_input)
        def options(axis, source, selected):
            units = tuple(
                (unit.symbol, format_unit_for_display(unit))
                for unit in self._display_unit_choices(render_input, axis)
            )
            if source is not None and source.compatible(CUBIC_METRE) and render_input.authored_volume_unit is not None:
                return (("Authored", ""), *units), selected if self._volume_display_unit is not None else "Authored"
            return units, selected
        x_units, x_unit = options("x", render_input.x_unit, x_unit)
        y_units, y_unit = options("y", render_input.y_unit, y_unit)
        self._axis_toolbar.set_axis_unit_choices(
            x_units=x_units,
            selected_x=x_unit,
            y_units=y_units,
            selected_y=y_unit,
            transformed=self._presentation.plot_transform != "original",
        )

    def _display_axis_labels(
        self,
        render_input: ResultRenderInput,
    ) -> tuple[str, str]:
        x_unit, y_unit, _x_scale, _y_scale = self._display_unit_state(
            render_input
        )
        if self._has_time_x(render_input):
            x_label = f"Time ({format_unit_for_display(x_unit)})"
        elif x_unit is not None:
            x_label = f"{html.escape(render_input.x_label)} ({format_unit_for_display(x_unit)})"
        else:
            x_label = html.escape(render_input.x_label)
        y_label = (
            f"{html.escape(render_input.y_label)} ({format_unit_for_display(y_unit)})"
            if y_unit is not None
            else html.escape(render_input.y_label)
        )
        if self._presentation.plot_transform != "original":
            derivative_label = f"d({y_label}) / d({x_label})"
            if self._presentation.plot_transform == "phase":
                x_label = y_label
            y_label = derivative_label
        return x_label, y_label

    @staticmethod
    def _derivative_unit(x_unit: str | None, y_unit: str | None) -> str | None:
        if x_unit is None or y_unit is None:
            return None
        numerator = f"({y_unit})" if "*" in y_unit or "/" in y_unit else y_unit
        denominator = f"({x_unit})" if "*" in x_unit or "/" in x_unit else x_unit
        return f"{numerator}/{denominator}"

    def _coordinate_scales(self, render_input: ResultRenderInput) -> tuple[float, float]:
        _x_unit, _y_unit, x_scale, y_scale = self._display_unit_state(render_input)
        transform = self._presentation.plot_transform
        return (
            y_scale if transform == "phase" else x_scale,
            y_scale if transform == "original" else y_scale / x_scale,
        )

    @staticmethod
    def _scaled_display_values(values: np.ndarray, scale: float) -> np.ndarray:
        array = np.asarray(values)
        if scale == 1.0:
            return array
        return np.asarray(array * scale)

    def _render_result_scene(
        self,
        render_input: ResultRenderInput,
    ) -> None:
        self._rendered_point_roster = ()
        existing_items = dict(self._plot_items_by_key)
        self._configure_axis_unit_controls(render_input)
        x_label, y_label = self._display_axis_labels(render_input)
        self._plot_widget.setLabel("bottom", x_label)
        self._plot_widget.setLabel("left", y_label)
        rendered_items: dict[tuple[object, ...], object] = {}
        rendered_points: list[_RenderedPointBlock] = []
        for block in render_input.blocks:
            if not block.visible:
                continue
            self._render_layer(
                block,
                sampling_limit=self._sampling_limit,
                existing_items=existing_items,
                rendered_items=rendered_items,
                rendered_points=rendered_points,
            )
        for item_key, item in existing_items.items():
            if item_key not in rendered_items:
                self._remove_plot_item(item)
        self._plot_items_by_key = rendered_items
        self._axis_toolbar.set_x_candidates(
            render_input.axis_candidates,
            selected=render_input.selected_x_series,
        )
        y_candidates = tuple(
            candidate for candidate in render_input.axis_candidates
            if not (
                isinstance(candidate.selection, SimulationSeriesAxisSelection)
                and candidate.selection.series_name == "t"
            )
        )
        locked_checked = dict(render_input.mapped_overlay_locks)
        self._axis_toolbar.set_y_candidates(
            [
                (
                    candidate,
                    (render_input.selected_y_uses_default_all and not isinstance(candidate.selection, VesselVolumeAxisSelection))
                    or candidate.selection in render_input.selected_y_series
                    or candidate.selection in locked_checked,
                )
                for candidate in y_candidates
            ],
            locked_checked=locked_checked,
        )
        self._apply_zoom_mode()
        presentation = self._presentation
        self._render_analysis(render_input)
        axis_range = self._axis_range
        self._axis_toolbar.set_manual_ranges(axis_range.x_min, axis_range.x_max, axis_range.y_min, axis_range.y_max)
        self._apply_plot_presentation(presentation, axis_range=axis_range)
        self._render_annotations(render_input, presentation)
        self._project_completion_warning(render_input)
        self._rendered_point_roster = tuple(rendered_points)
        self._sync_plot_controls()

    def _update_plot_item_legend(self, item: object, name: str | None) -> None:
        # PyQtGraph 0.13.7 exposes PlotDataItem.name() as an opts-backed
        # getter without a corresponding name mutator.
        item.opts["name"] = name
        if self._legend is None:
            return
        if name is None:
            # A reused item may have been the first chunk in an earlier
            # topology. Remove that stale row when it becomes a later chunk,
            # a connector, or a ghost curve.
            if self._legend.getLabel(item) is not None:
                self._legend.removeItem(item)
            return
        label = self._legend.getLabel(item)
        if label is None:
            self._legend.addItem(item, name=name)
        else:
            label.setText(name)

    def _remove_plot_item(self, item: object) -> None:
        if self._legend is not None:
            try:
                self._legend.removeItem(item)
            except (KeyError, RuntimeError, TypeError, ValueError):
                pass
        try:
            self._plot_widget.getPlotItem().removeItem(item)
        except (RuntimeError, TypeError, ValueError):
            pass

    def _rendered_block_for_static(
        self,
        block: ResultDataBlock,
        sampling_limit: int | None,
    ) -> tuple[_RenderedPointBlock, tuple[_RenderedPointChunk, ...]]:
        sample = self._sample_static_block(block, sampling_limit)
        definitions = dict(block.copy_definitions)
        x_is_time = (
            isinstance(block.x_axis, SimulationSeriesAxisSelection)
            and block.x_axis.series_name == "t"
        )
        x_name = (
            block.time_column_name
            if block.kind is ResultLayerKind.EXPERIMENTAL and x_is_time
            else block.x_name
        )
        x_unit, x_scale = self._curve_unit_state("x", block.unit_for(block.x_name, x=True))
        y_units = {name: self._curve_unit_state("y", block.unit_for(name)) for name, _ in block.series}
        if block.result is not None:
            from kindred.core.result_curve_projection import physical_curve_chunks
            projected = physical_curve_chunks(block.result, block.x_axis, block.series_axes, sample + block.first_index)
            rendered_chunks = tuple(_RenderedPointChunk(
                x=self._scaled_display_values(chunk.x, x_scale),
                series=tuple((name, self._scaled_display_values(values, y_units[name][1])) for name, values in chunk.series),
                sample_time=chunk.time, instantaneous=chunk.instantaneous,
            ) for chunk in projected)
        else:
            rendered_chunks = (_RenderedPointChunk(
                x=self._scaled_display_values(block.t[sample], x_scale),
                series=tuple((name, self._scaled_display_values(values[sample], y_units[name][1])) for name, values in block.series),
                sample_time=None if block.sample_time is None else block.sample_time[sample],
            ),)
        rendered_block = _RenderedPointBlock(
            block_id=block.block_id,
            group_id=(
                block.block_id.dataset_id
                if block.kind is ResultLayerKind.EXPERIMENTAL else block.block_id
            ),
            label=block.label,
            kind=block.kind.value,
            x_name=copy_series_label(
                x_name,
                definition=definitions.get(block.x_name),
            ),
            x_is_time=x_is_time,
            x_unit=x_unit,
            time_column_name=block.time_column_name,
            y_names=tuple(
                (name, copy_series_label(name, definition=definitions.get(name)))
                for name, _ in block.series
            ),
            y_units=tuple(
                (name, y_units[name][0])
                for name, _ in block.series
            ),
            chunks=rendered_chunks,
        )
        return rendered_block, rendered_chunks

    def _rendered_block_for_interactive_curve(
        self,
        curve: object,
    ) -> _RenderedPointBlock | None:
        key = tuple(curve.key)
        if len(key) < 4 or key[0] != "simulation":
            return None
        set_id, selection = str(key[2]), key[3]
        source = curve.source
        series_name = next(name for name, axis in source.series_axes if axis == selection)
        definitions = dict(source.copy_definitions)
        x_unit, x_scale = self._curve_unit_state("x", source.unit_for(source.x_name, x=True))
        y_unit, y_scale = self._curve_unit_state("y", source.unit_for(series_name))
        curve_chunks = tuple(_RenderedPointChunk(
            x=self._scaled_display_values(chunk.x, x_scale),
            series=tuple((name, self._scaled_display_values(values, y_scale)) for name, values in chunk.series),
            sample_time=chunk.time, instantaneous=chunk.instantaneous,
        ) for chunk in curve.physical_chunks)
        return _RenderedPointBlock(
            block_id=set_id,
            group_id=set_id,
            label=source.label,
            kind=source.kind.value,
            x_name=copy_series_label(source.x_name, definition=definitions.get(source.x_name)),
            x_is_time=(
                isinstance(source.x_axis, SimulationSeriesAxisSelection)
                and source.x_axis.series_name == "t"
            ),
            x_unit=x_unit,
            time_column_name=source.time_column_name,
            y_names=((series_name, copy_series_label(
                series_name, definition=definitions.get(series_name),
            )),),
            y_units=((series_name, y_unit),),
            chunks=curve_chunks,
        )

    def _transformed_blocks(self, block: _RenderedPointBlock) -> tuple[_RenderedPointBlock, ...]:
        transform = self._presentation.plot_transform
        if transform == "original":
            return (block,)
        names = tuple(name for name, _ in block.chunks[0].series) if block.chunks else ()
        groups = tuple((name,) for name in names) if transform == "phase" else (names,)
        denominator = copy_series_label(block.x_name, block.x_unit)
        y_names, y_units = dict(block.y_names), dict(block.y_units)
        result = []
        for group in groups:
            chunks = []
            for chunk in block.chunks:
                if chunk.instantaneous:
                    continue
                series = dict(chunk.series)
                mapped = tuple((name, _transform_samples(chunk.x, series[name], transform)) for name in group)
                if not mapped:
                    continue
                chunks.append(_RenderedPointChunk(
                    x=mapped[0][1][0],
                    series=tuple((name, coordinates[1]) for name, coordinates in mapped),
                    sample_time=None if chunk.sample_time is None else chunk.sample_time[:-1],
                ))
            result.append(replace(
                block, chunks=tuple(chunks), left_endpoint=True,
                x_name=f"{y_names[group[0]]} [phase, left endpoint]" if transform == "phase" else f"{block.x_name} [left endpoint]",
                x_is_time=block.x_is_time and transform != "phase",
                x_unit=y_units[group[0]] if transform == "phase" else block.x_unit,
                y_names=tuple(
                    (name, f"d({copy_series_label(y_names[name], y_units[name])})/d({denominator}) [adjacent samples]")
                    for name in group
                ),
                y_units=tuple((name, self._derivative_unit(block.x_unit, y_units[name])) for name in group),
            ))
        return tuple(result)

    def _render_layer(
        self,
        block: ResultDataBlock,
        *,
        sampling_limit: int | None,
        existing_items: Mapping[tuple[object, ...], object] | None = None,
        rendered_items: dict[tuple[object, ...], object] | None = None,
        rendered_points: list[_RenderedPointBlock] | None = None,
    ) -> None:
        rendered_block, rendered_chunks = self._rendered_block_for_static(
            block,
            sampling_limit,
        )
        transformed_blocks = self._transformed_blocks(rendered_block)
        transform = self._presentation.plot_transform
        colors = dict(block.color_by_series)
        item_store = rendered_items if rendered_items is not None else {}
        existing_store = existing_items if existing_items is not None else {}
        role = (block.kind.value, block.block_id)
        for series_name, _values in block.series:
            color = colors[series_name]
            transformed = next((item for item in transformed_blocks if item.chunks and series_name in dict(item.chunks[0].series)), None)
            rendered_chunks = () if transformed is None else transformed.chunks
            physical_chunks: list[tuple[np.ndarray, np.ndarray]] = []
            for chunk_index, chunk in enumerate(rendered_chunks):
                x = chunk.x
                y = dict(chunk.series)[series_name]
                if transform == "original" and chunk_index and block.result is None:
                    prior_x, prior_y = physical_chunks[-1]
                    physical_chunks[-1] = (
                        np.concatenate((prior_x, x)),
                        np.concatenate((prior_y, y)),
                    )
                else:
                    physical_chunks.append((x, y))
            legend_pending = True
            for chunk_index, (x, y) in enumerate(physical_chunks):
                if not np.any(np.isfinite(x) & np.isfinite(y)):
                    continue
                singleton = x.size == 1
                marker = bool(block.markers or singleton)
                item_key = (role, dict(block.series_axes)[series_name], chunk_index, "marker" if marker else "line")
                item = existing_store.get(item_key)
                name = f"{block.label}: {series_name}"
                unit = block.unit_for(series_name)
                if self._render_input.y_unit is None and unit is not None:
                    name += f" [{format_unit_for_display(dict(transformed.y_units)[series_name])}]"
                legend_name = name if legend_pending else None
                legend_pending = False
                if item is None:
                    if marker:
                        item = self._add_plot_data(
                            x, y, pen=None, symbol=(block.point_shape if block.markers else "o"),
                            symbolSize=(block.point_size if block.markers else 5.0),
                            symbolBrush=pg.mkBrush(color), symbolPen=pg.mkPen(color),
                            name=legend_name, connect="finite",
                        )
                    else:
                        item = self._add_plot_data(
                            x, y,
                            pen=pg.mkPen(color=color, width=2, style=(QtCore.Qt.PenStyle.DashLine if block.dashed else QtCore.Qt.PenStyle.SolidLine)),
                            name=legend_name, connect="finite",
                        )
                else:
                    item.setData(x, y, connect="finite")
                    if marker:
                        item.setSymbolBrush(pg.mkBrush(color))
                        item.setSymbolPen(pg.mkPen(color))
                        item.setSymbol(block.point_shape if block.markers else "o")
                        item.setSymbolSize(block.point_size if block.markers else 5.0)
                    else:
                        item.setPen(pg.mkPen(color=color, width=2, style=(QtCore.Qt.PenStyle.DashLine if block.dashed else QtCore.Qt.PenStyle.SolidLine)))
                self._update_plot_item_legend(item, legend_name)
                item.setVisible(True)
                self._style_simulation_item(item, block.kind is not ResultLayerKind.EXPERIMENTAL, color, block.opacity, singleton=singleton)
                item_store[item_key] = item
        if rendered_points is not None and any(
            chunk.x.size for chunk in rendered_block.chunks
        ):
            rendered_points.extend(transformed_blocks)
    def _style_simulation_item(self, item: object, simulation: bool, color: object, opacity: float, *, singleton: bool) -> None:
        if simulation:
            item.setSymbol("o" if self._presentation.simulation_markers or singleton else None)
            if self._presentation.simulation_markers or singleton:
                item.setSymbolSize(5)
                item.setSymbolBrush(pg.mkBrush(color))
                item.setSymbolPen(pg.mkPen(color))
            opacity *= self._presentation.simulation_opacity
        item.setOpacity(opacity)

    def _sample_static_block(
        self,
        block: ResultDataBlock,
        sampling_limit: int | None,
    ) -> np.ndarray:
        sample = self._sample_indices(block.t.size, sampling_limit)
        if block.kind is not ResultLayerKind.EXPERIMENTAL:
            region_endpoints = self._execution_region_endpoints(block)
            sample = np.unique(np.concatenate((sample, region_endpoints)))
        return sample

    def _render_analysis(self, render_input: ResultRenderInput) -> None:
        concentration_unit = (
            self._selected_display_unit("y", render_input.y_unit)
            if render_input.y_unit is not None and render_input.y_unit.compatible(MOLAR)
            else MOLAR
        )
        time_unit = (
            self._selected_display_unit("x", SECOND)
            if self._has_time_x(render_input)
            else SECOND
        )
        volume_unit = (
            self._selected_volume_unit()
            if render_input.authored_volume_unit is not None
            else CUBIC_METRE
        )
        self._parameter_table.update_parameters(
            render_input.parameters_for_display(
                concentration_unit=concentration_unit, time_unit=time_unit, volume_unit=volume_unit,
            )
        )
        result = render_input.statistics_result
        if result is None or not render_input.statistics_series_names:
            self._species_table.setRowCount(0)
            return
        self._species_table.update_results(
            canonical_time=result.t,
            species_data={name: result.series[name] for name in render_input.statistics_series_names},
            intervention_execution=result.intervention_execution,
            presentation_boundaries=result.presentation_boundaries,
            concentration_display_unit=concentration_unit,
            time_display_unit=time_unit,
        )
    def _project_completion_warning(
        self,
        render_input: ResultRenderInput,
    ) -> None:
        diagnostic = render_input.unavailable_x_diagnostic
        has_warnings = bool(diagnostic or render_input.completion_warning_details)
        warning_details = render_input.completion_warning_tooltip
        details = "\n\n".join(
            detail
            for detail in (diagnostic, warning_details)
            if detail
        )
        if not has_warnings:
            self._clear_completion_warning()
            return
        caption = "Selected X unavailable" if diagnostic is not None else "Completed with warnings"
        self._completion_warning_badge.setText(caption)
        self._completion_warning_badge.setAccessibleName(caption)
        self._completion_warning_badge.setToolTip(literal_tooltip(details))
        self._completion_warning_badge.setAccessibleDescription(details)
        self._completion_warning_badge.setVisible(True)
        self._sync_plot_controls()

    @staticmethod
    def _raw_to_view_coordinate(
        value: object,
        *,
        log_enabled: bool,
    ) -> float | None:
        """Map one raw semantic coordinate into the active plot view."""

        try:
            raw = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(raw):
            return None
        if not log_enabled:
            return raw
        if raw <= 0.0:
            return None
        mapped = float(np.log10(raw))
        return mapped if np.isfinite(mapped) else None

    @staticmethod
    def _view_to_raw_coordinate(
        value: object,
        *,
        log_enabled: bool,
    ) -> float | None:
        """Map an axis view coordinate back to its raw range value."""

        try:
            view = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(view):
            return None
        if not log_enabled:
            return view
        try:
            raw = float(np.power(10.0, view))
        except (FloatingPointError, OverflowError, ValueError):
            return None
        return raw if np.isfinite(raw) and raw > 0.0 else None

    def _render_annotations(
        self,
        render_input: ResultRenderInput,
        presentation: ProjectPresentationState,
        *,
        annotations: Sequence[_DisplayAnnotation] | None = None,
    ) -> None:
        if annotations is None:
            annotations = self._merge_annotation_values(render_input.intervention_annotations)
        self._render_owned_annotations(render_input, annotations, presentation)

    @staticmethod
    def _merge_annotation_values(
        annotations: Sequence[_DisplayAnnotation],
    ) -> tuple[_DisplayAnnotation, ...]:
        merged: dict[_AnnotationKey, _DisplayAnnotation] = {}
        for annotation in annotations:
            signature = PyQtGraphPlotPanel._annotation_signature(annotation)
            prior = merged.get(signature)
            opacity = annotation[3] + (0.0 if prior is None else prior[3])
            merged[signature] = (*annotation[:3], min(1.0, opacity))
        return tuple(merged.values())

    def _render_owned_annotations(
        self,
        render_input: ResultRenderInput,
        annotations: Sequence[_DisplayAnnotation],
        presentation: ProjectPresentationState,
    ) -> None:
        if (
            not render_input.intervention_annotations_visible
            or self._presentation.plot_transform == "phase"
            or not self._has_time_x(render_input)
        ):
            self._clear_annotation_items()
            return
        _x_unit, _y_unit, x_scale, _y_scale = self._display_unit_state(
            render_input
        )
        entries: list[
            tuple[_AnnotationKey, _DisplayAnnotation, float, float | None]
        ] = []
        for owned in annotations:
            role, set_id, annotation, opacity = owned
            if role == "ghost" and not render_input.reference_lines_visible:
                continue
            if "start" in annotation and "end" in annotation:
                start = self._raw_to_view_coordinate(
                    float(annotation["start"]) * x_scale,
                    log_enabled=presentation.log_x,
                )
                end = self._raw_to_view_coordinate(
                    float(annotation["end"]) * x_scale,
                    log_enabled=presentation.log_x,
                )
                if start is None or end is None:
                    continue
                entries.append((self._annotation_signature(owned), owned, start, end))
                continue
            if "time" not in annotation:
                continue
            time = self._raw_to_view_coordinate(
                float(annotation["time"]) * x_scale,
                log_enabled=presentation.log_x,
            )
            if time is None:
                continue
            entries.append((self._annotation_signature(owned), owned, time, None))

        foreground_opacity: dict[tuple[object, ...], float] = {}
        for _key, (role, set_id, _payload, opacity), start, end in entries:
            if role == "foreground":
                geometry = (set_id, start, end)
                foreground_opacity[geometry] = min(1.0, foreground_opacity.get(geometry, 0.0) + opacity)
        existing = self._annotation_items
        rendered: dict[_AnnotationKey, object] = {}
        label_candidates: dict[
            tuple[object, ...], tuple[Mapping[str, object], float, float]
        ] = {}
        for key, owned, start, end in entries:
            role, set_id, annotation, opacity = owned
            if role == "ghost":
                opacity *= 1.0 - foreground_opacity.get((set_id, start, end), 0.0)
                if opacity <= _EPSILON:
                    continue
            if role == "foreground" and render_input.intervention_labels_visible:
                occurrence = (
                    set_id, str(annotation.get("kind") or "intervention"),
                    annotation["declaration_ordinal"],
                    annotation.get("authored_ordinal"), annotation.get("window_ordinal"),
                    annotation.get("firing_ordinal"),
                )
                prior = label_candidates.get(occurrence)
                label_candidates[occurrence] = (
                    annotation,
                    start * opacity + (0.0 if prior is None else prior[1]),
                    opacity + (0.0 if prior is None else prior[2]),
                )
            primary = rendered.get(key) or existing.get(key)
            if primary is None:
                accent = self._annotation_accent(annotation)
                if end is None:
                    primary = pg.InfiniteLine(
                        pos=start,
                        angle=90,
                        movable=False,
                        pen=pg.mkPen(accent, width=1, style=QtCore.Qt.PenStyle.DashLine),
                    )
                else:
                    fill = QtGui.QColor(accent)
                    fill.setAlpha(35)
                    primary = pg.LinearRegionItem(
                        values=(start, end),
                        movable=False,
                        brush=pg.mkBrush(fill),
                        pen=pg.mkPen(accent),
                    )
                primary.setZValue(10)
                self._plot_widget.addItem(primary)
            elif end is None:
                primary.setValue(start)
            else:
                primary.setRegion((start, end))
            primary.setOpacity(opacity)
            rendered[key] = primary
        for key, item in existing.items():
            if key not in rendered:
                self._remove_annotation_item(item)
        self._annotation_items = rendered

        if not render_input.intervention_labels_visible:
            self._clear_annotation_labels()
            return

        existing_labels = self._annotation_labels
        rendered_labels: dict[tuple[object, ...], _AnnotationLabel] = {}
        for lane, (occurrence, candidate) in enumerate(label_candidates.items()):
            annotation, weighted_start, weight = candidate
            text = intervention_label(annotation, volume_unit=self._selected_volume_unit(),
                                      time_unit=self._selected_display_unit("x", SECOND))
            prior = existing_labels.get(occurrence)
            accent = self._annotation_accent(annotation)
            if prior is None:
                label = pg.TextItem(
                    "",
                    color=(238, 238, 238), anchor=(0.0, 0.5),
                    border=pg.mkPen(accent), fill=pg.mkBrush(30, 30, 30, 210),
                )
                label.setHtml(text)
                label.setFlag(
                    QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True
                )
                # Keep right clicks/drags with the scene's plot navigation.
                label.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
                label.setZValue(11)
                self._plot_widget.addItem(label, ignoreBounds=True)
                prior = _AnnotationLabel(label, QtCore.QPointF(label.pos()), weighted_start / weight, lane)
            else:
                label = prior.item
                label.setHtml(text)
                label.border = pg.mkPen(accent)
                label.update()
            prior.event_x = weighted_start / weight
            prior.lane = lane
            label.setOpacity(min(1.0, weight))
            rendered_labels[occurrence] = prior
        self._annotation_labels = rendered_labels
        for occurrence, record in existing_labels.items():
            if occurrence not in rendered_labels:
                self._remove_annotation_item(record.item)
        self._position_annotation_labels()

    def _position_annotation_labels(self) -> None:
        view_range = self._plot_widget.getViewBox().viewRange()
        x_span = float(view_range[0][1] - view_range[0][0])
        y_min, y_max = (float(value) for value in view_range[1])
        y_span = y_max - y_min
        if not np.isfinite(x_span) or x_span <= 0.0:
            x_span = 1.0
        if not np.isfinite(y_span) or y_span <= 0.0:
            y_min, y_max, y_span = 0.0, 1.0, 1.0
        for record in self._annotation_labels.values():
            lane_offset = 0.10 + 0.08 * (record.lane % 4)
            anchor = QtCore.QPointF(
                record.event_x + x_span * (-0.012 if self._presentation.x_inverted else 0.012),
                y_min + y_span * lane_offset
                if self._presentation.y_inverted else y_max - y_span * lane_offset,
            )
            offset = record.item.pos() - record.anchor
            record.anchor = anchor
            record.item.setPos(anchor + offset)

    @staticmethod
    def _sample_indices(
        count: int,
        sampling_limit: int | None,
    ) -> np.ndarray:
        if sampling_limit is None or count <= sampling_limit:
            return np.arange(count, dtype=int)
        return np.linspace(0, count - 1, sampling_limit, dtype=int)

    @staticmethod
    def _execution_region_endpoints(block: ResultDataBlock) -> np.ndarray:
        if (
            block.segment_ids is None
            or block.segment_keys is None
            or block.discontinuity_ids is None
        ):
            raise ValueError("simulation plot block lacks executor identity.")
        count = int(block.t.size)
        if count == 1:
            return np.asarray((0,), dtype=int)
        changes = np.fromiter(
            (
                index
                for index in range(1, count)
                if (
                    int(block.segment_ids[index]) != int(block.segment_ids[index - 1])
                    or block.segment_keys[index] != block.segment_keys[index - 1]
                    or int(block.discontinuity_ids[index])
                    != int(block.discontinuity_ids[index - 1])
                )
            ),
            dtype=int,
        )
        starts = np.concatenate((np.asarray((0,), dtype=int), changes))
        ends = np.concatenate((changes - 1, np.asarray((count - 1,), dtype=int)))
        return np.unique(np.concatenate((starts, ends)))

    def _clear_plot_items(self) -> None:
        self._annotation_labels.clear()
        self._plot_widget.clear()
        self._plot_items_by_key.clear()
        self._annotation_items.clear()
        self._rendered_point_roster = ()
        self._legend = self._plot_widget.addLegend(
            offset=(-10, 10),
            labelTextColor=plot_foreground_color(self._plot_widget.backgroundBrush().color().lightness() < 128),
        )

    def _remove_annotation_item(self, item: object) -> None:
        try:
            self._plot_widget.removeItem(item)
        except (RuntimeError, TypeError, ValueError):
            pass

    def _clear_annotation_items(self) -> None:
        for item in self._annotation_items.values():
            self._remove_annotation_item(item)
        self._annotation_items.clear()
        self._clear_annotation_labels()

    def _clear_annotation_labels(self) -> None:
        labels, self._annotation_labels = self._annotation_labels, {}
        for record in labels.values():
            self._remove_annotation_item(record.item)

    @staticmethod
    def _scaled_optional(value: float | None, ratio: float) -> float | None:
        return None if value is None else float(value) * ratio

    def _apply_display_unit_change(self, axis: str, unit: str) -> None:
        render_input = self._render_input
        if render_input is None:
            return
        source = render_input.x_unit if axis == "x" else render_input.y_unit
        if source is not None and source.compatible(CUBIC_METRE) and render_input.authored_volume_unit is not None:
            self._apply_volume_unit_change(unit)
            return
        options = self._display_unit_choices(render_input, axis)
        chosen = next((item for item in options if item.symbol == unit), None)
        if chosen is None:
            return
        if self._selected_display_unit(axis, source).symbol == chosen.symbol:
            return
        self._axis_range = self._capture_view_range()
        old_x_scale, old_y_scale = self._coordinate_scales(render_input)
        self._axis_display_units[axis, chosen.dimensions] = chosen
        self._refresh_display_units(old_x_scale, old_y_scale)

    def _apply_volume_unit_change(self, symbol: str) -> None:
        render_input = self._render_input
        if render_input is None or render_input.authored_volume_unit is None:
            return
        chosen = None if symbol == "Authored" else next(unit for unit in self._volume_unit_choices() if unit.symbol == symbol)
        current = self._volume_display_unit.symbol if self._volume_display_unit is not None else "Authored"
        if symbol == current:
            return
        self._axis_range = self._capture_view_range()
        old_x_scale, old_y_scale = self._coordinate_scales(render_input)
        self._volume_display_unit = chosen
        self._refresh_display_units(old_x_scale, old_y_scale)

    def _refresh_display_units(self, old_x_scale: float, old_y_scale: float) -> None:
        render_input = self._render_input
        new_x_scale, new_y_scale = self._coordinate_scales(render_input)
        x_ratio = new_x_scale / old_x_scale
        y_ratio = new_y_scale / old_y_scale
        self._axis_range = replace(
            self._axis_range,
            x_min=self._scaled_optional(self._axis_range.x_min, x_ratio),
            x_max=self._scaled_optional(self._axis_range.x_max, x_ratio),
            y_min=self._scaled_optional(self._axis_range.y_min, y_ratio),
            y_max=self._scaled_optional(self._axis_range.y_max, y_ratio),
        )
        self._axis_toolbar.set_manual_ranges(
            self._axis_range.x_min,
            self._axis_range.x_max,
            self._axis_range.y_min,
            self._axis_range.y_max,
        )
        self.apply_result_input(render_input)

    @QtCore.Slot(bool)
    def _set_coarse_sampling(self, enabled: bool) -> None:
        self._sampling_limit = 1000 if enabled else None
        if self._render_input is not None:
            self.apply_result_input(self._render_input)
        self._sync_plot_controls()

    def _capture_view_range(self) -> _AxisRangeSession:
        current_x, current_y = self._plot_widget.getViewBox().viewRange()
        return replace(
            self._axis_range,
            x_min=self._view_to_raw_coordinate(current_x[0], log_enabled=self._presentation.log_x),
            x_max=self._view_to_raw_coordinate(current_x[1], log_enabled=self._presentation.log_x),
            y_min=self._view_to_raw_coordinate(current_y[0], log_enabled=self._presentation.log_y),
            y_max=self._view_to_raw_coordinate(current_y[1], log_enabled=self._presentation.log_y),
        )

    @QtCore.Slot(bool)
    def _set_auto_range(self, enabled: bool) -> None:
        self._axis_range = replace(self._capture_view_range(), auto_x=enabled, auto_y=enabled, visible_x=False)
        self._apply_axis_range_state(self._axis_range, self._presentation)
        self._project_manual_ranges()

    def _set_navigation(self, option: str, value: bool | float) -> None:
        state = self._capture_view_range()
        changes = {option: value}
        if option == "visible_x" and value:
            changes.update(auto_x=False, auto_y=True, pan_y=False)
        elif (option == "auto_y" and not value) or (option in ("auto_x", "pan_y") and value):
            changes["visible_x"] = False
        self._axis_range = replace(state, **changes)
        self._apply_axis_range_state(self._axis_range, self._presentation)
        self._project_manual_ranges()

    def _fit_axes_once(self, axes: tuple[bool, bool]) -> None:
        view = self._plot_widget.getViewBox()
        bounds = view.childrenBounds()
        ranges = {
            f"{axis}Range": bounds[index]
            for index, axis in enumerate(("x", "y"))
            if axes[index] and bounds[index] is not None
            and all(value is not None and np.isfinite(value) for value in bounds[index])
        }
        if ranges:
            view.setRange(**ranges, disableAutoRange=False)
        self._axis_range = self._capture_view_range()
        self._project_manual_ranges()

    @QtCore.Slot()
    def _fit_once(self) -> None:
        view = self._plot_widget.getViewBox()
        view.disableAutoRange()
        view.setAutoVisible(x=False, y=False)
        self._axis_range = replace(self._axis_range, auto_x=False, auto_y=False, visible_x=False)
        self._fit_axes_once((True, True))

    def _on_manual_navigation(self, axes: Sequence[bool]) -> None:
        view = self._plot_widget.getViewBox()
        state = self._capture_view_range()
        self._axis_range = replace(
            state, auto_x=state.auto_x and not axes[0], auto_y=state.auto_y and not axes[1],
            visible_x=state.visible_x and not axes[1],
        )
        view.enableAutoRange(x=self._axis_range.fraction_x if self._axis_range.auto_x else False,
                             y=self._axis_range.fraction_y if self._axis_range.auto_y else False)
        if view.state["autoVisibleOnly"][1] != self._axis_range.visible_x:
            view.setAutoVisible(x=False, y=self._axis_range.visible_x)
        elif self._axis_range.visible_x:
            view.updateAutoRange()
        self._axis_range = self._capture_view_range()
        self._project_manual_ranges()

    def _project_manual_ranges(self) -> None:
        state = self._axis_range
        self._axis_toolbar.set_manual_ranges(state.x_min, state.x_max, state.y_min, state.y_max)
        self._sync_plot_controls()

    @QtCore.Slot(str)
    def _apply_axis_range(self, axis: str) -> None:
        values = self._axis_toolbar.get_manual_ranges()
        low, high = values[:2] if axis == "x" else values[2:]
        state = self._capture_view_range()
        if getattr(self._presentation, f"log_{axis}") and (
            (low is not None and low <= 0) or (high is not None and high <= 0)
        ):
            self._axis_range = state
            self._project_manual_ranges()
            self.resultDiagnostic.emit(f"{axis.upper()} limits must be positive for logarithmic display.")
            return
        if low is not None and high is not None and high > low:
            state = replace(state, **{f"{axis}_min": low, f"{axis}_max": high})
        self._axis_range = state
        self._apply_axis_range_state(state, self._presentation)
        self._project_manual_ranges()

    def _apply_axis_range_state(self, state: _AxisRangeSession, presentation: ProjectPresentationState) -> None:
        view = self._plot_widget.getViewBox()
        view.setMouseEnabled(x=state.mouse_x, y=state.mouse_y)
        view.enableAutoRange(x=state.fraction_x if state.auto_x else False,
                             y=state.fraction_y if state.auto_y else False)
        for axis in ("x", "y"):
            if getattr(state, f"auto_{axis}"):
                continue
            low = self._raw_to_view_coordinate(getattr(state, f"{axis}_min"), log_enabled=getattr(presentation, f"log_{axis}"))
            high = self._raw_to_view_coordinate(getattr(state, f"{axis}_max"), log_enabled=getattr(presentation, f"log_{axis}"))
            if low is not None and high is not None and high > low:
                view.setRange(**{f"{axis}Range": (low, high)}, padding=0, disableAutoRange=False)
        if view.state["autoPan"] != [state.pan_x, state.pan_y]:
            view.setAutoPan(x=state.pan_x, y=state.pan_y)
        if view.state["autoVisibleOnly"] != [False, state.visible_x]:
            view.setAutoVisible(x=False, y=state.visible_x)
        view.updateAutoRange()

    @QtCore.Slot(bool)
    def _set_zoom_enabled(self, enabled: bool) -> None:
        self._zoom_enabled = enabled
        self._apply_zoom_mode()
        self._sync_plot_controls()

    def _apply_zoom_mode(self) -> None:
        view_box = self._plot_widget.getViewBox()
        view_box.setMouseMode(view_box.RectMode if self._zoom_enabled else view_box.PanMode)

    def _request_plot_presentation(
        self,
        option: str,
        value: bool | float | str,
    ) -> None:
        # The WorkspaceTabRegistry is the sole accepted presentation owner.
        # Do not mutate local state or visible pyqtgraph items until it has
        # validated and stored this intent.
        self.plotPresentationIntentRequested.emit(option, value)

    def _apply_plot_presentation(
        self,
        state: ProjectPresentationState,
        *,
        axis_range: _AxisRangeSession | None = None,
    ) -> None:
        item = self._plot_widget.getPlotItem()
        item.setLogMode(x=state.log_x, y=state.log_y)
        item.invertX(state.x_inverted)
        item.invertY(state.y_inverted)
        item.showGrid(x=state.grid_x, y=state.grid_y, alpha=state.grid_opacity)
        self._sync_plot_controls()
        if self._legend is not None:
            self._legend.setVisible(state.legend_visible)
        if axis_range is not None:
            self._apply_axis_range_state(axis_range, state)

    def set_theme(self, *, dark_mode: bool) -> None:
        dark = bool(dark_mode)
        self._plot_widget.setBackground(plot_background_color(dark))
        foreground = plot_foreground_color(dark)
        if self._legend is not None:
            self._legend.setLabelTextColor(foreground)
        for axis_name in ("left", "bottom"):
            axis = self._plot_widget.getPlotItem().getAxis(axis_name)
            axis.setPen(foreground)
            axis.setTextPen(foreground)


_MAX_RENDERED_COPY_CELLS = 1_000_000
_MAX_RENDERED_COPY_CHARACTERS = 64 * 1024 * 1024


def _visible_sample_mask(
    x: np.ndarray, y: np.ndarray, view_range: Sequence[Sequence[float]],
    *, log_x: bool, log_y: bool,
) -> np.ndarray:
    """Compare existing samples in the native ViewBox coordinate space."""

    with np.errstate(divide="ignore", invalid="ignore"):
        mapped_x = np.log10(x) if log_x else x
        mapped_y = np.log10(y) if log_y else y
    return (
        np.isfinite(mapped_x) & np.isfinite(mapped_y)
        & (mapped_x >= view_range[0][0]) & (mapped_x <= view_range[0][1])
        & (mapped_y >= view_range[1][0]) & (mapped_y <= view_range[1][1])
    )


def _has_visible_samples(
    roster: Sequence[_RenderedPointBlock], view_range: Sequence[Sequence[float]],
    *, log_x: bool, log_y: bool,
) -> bool:
    """Stop at the first matching batch; never construct export columns."""

    for block in roster:
        for chunk in block.chunks:
            for _name, y in chunk.series:
                for start in range(0, chunk.x.size, 4096):
                    sample = slice(start, start + 4096)
                    if _visible_sample_mask(
                        chunk.x[sample], y[sample], view_range, log_x=log_x, log_y=log_y,
                    ).any():
                        return True
    return False


def _visible_point_roster(
    roster: Sequence[_RenderedPointBlock], view_range: Sequence[Sequence[float]],
    *, log_x: bool, log_y: bool,
) -> tuple[_RenderedPointBlock, ...]:
    """Create call-local, independently paired series for the existing serializer."""

    visible = []
    for block in roster:
        units = dict(block.y_units)
        for name, label in block.y_names:
            chunks = []
            for chunk in block.chunks:
                y = dict(chunk.series)[name]
                mask = _visible_sample_mask(chunk.x, y, view_range, log_x=log_x, log_y=log_y)
                if mask.any():
                    chunks.append(_RenderedPointChunk(
                        x=chunk.x[mask], series=((name, y[mask]),),
                        sample_time=None if chunk.sample_time is None else chunk.sample_time[mask],
                    ))
            if chunks:
                visible.append(replace(
                    block, chunks=tuple(chunks), y_names=((name, label),),
                    y_units=((name, units[name]),),
                ))
    return tuple(visible)


def _iter_rendered_point_tsv_chunks(
    roster: Sequence[_RenderedPointBlock],
    *,
    rows_per_chunk: int = 2048,
    action_name: str = "Copy Plotted Data",
) -> tuple[str, ...]:
    """Serialize the captured roster as one rectangular TSV table."""

    values = tuple(roster)
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive.")
    nonempty = tuple(
        block
        for block in values
        if any(chunk.x.size for chunk in block.chunks)
    )
    if not nonempty:
        raise ValueError("No plotted data is available to copy.")
    # Per block: its column parts, the rows kept after de-duplication, and the row count.
    _BlockPlan = tuple[tuple[tuple[np.ndarray, ...], ...], np.ndarray | None, int]
    block_specs: list[tuple[_RenderedPointBlock, tuple[str, ...], _BlockPlan, int, bool]] = []
    planned_rows = 0
    planned_width = 0
    previous_group = None
    for block in nonempty:
        chunks = tuple(chunk for chunk in block.chunks if chunk.x.size)
        series_names = tuple(name for name, _values in chunks[0].series)
        if any(
            tuple(name for name, _values in chunk.series) != series_names
            for chunk in chunks[1:]
        ):
            raise ValueError("rendered point chunks must preserve series order.")
        with_time = not block.x_is_time and chunks[0].sample_time is not None
        columns = (
            *((tuple(chunk.sample_time for chunk in chunks),) if with_time else ()),
            tuple(chunk.x for chunk in chunks),
            *(tuple(dict(chunk.series)[name] for chunk in chunks) for name in series_names),
        )
        rows = sum(int(chunk.x.size) for chunk in chunks)
        keep = None
        if block.kind != ResultLayerKind.EXPERIMENTAL.value and rows > 1:
            # Chunk joins repeat shared endpoints; dataset rows keep repeats.
            # One column at a time keeps the budget check ahead of the matrix.
            repeated = np.ones(rows - 1, dtype=bool)
            for parts in columns:
                column = np.concatenate(parts).astype(float, copy=False)
                repeated &= (column[1:] == column[:-1]) | (np.isnan(column[1:]) & np.isnan(column[:-1]))
            keep = np.concatenate(([True], ~repeated))
            rows = int(keep.sum())
        width = len(columns)
        plan = (columns, keep, rows)
        group = (block.kind, block.group_id)
        separator = bool(block_specs) and group != previous_group
        previous_group = group
        block_specs.append((block, series_names, plan, width, separator))
        planned_rows = max(planned_rows, rows)
        planned_width += width + separator
    planned_cells = (planned_rows + 1) * planned_width
    if planned_cells > _MAX_RENDERED_COPY_CELLS:
        raise ValueError(
            f"{action_name} payload exceeds the bounded cell limit "
            f"({_MAX_RENDERED_COPY_CELLS:,})."
        )

    combined_headers: list[str] = []
    for block, series_names, _rows, _width, separator in block_specs:
        if separator:
            combined_headers.append("")
        chunks = tuple(chunk for chunk in block.chunks if chunk.x.size)
        time_name = block.time_column_name if block.kind == ResultLayerKind.EXPERIMENTAL.value else "Time"
        if block.left_endpoint:
            time_name = f"{time_name} [left endpoint]"
        headers = []
        if not block.x_is_time and chunks[0].sample_time is not None:
            headers.append(encode_clipboard_text(copy_column_header(block.label, time_name, "s")))
        headers.append(encode_clipboard_text(copy_column_header(
            block.label,
            time_name if block.x_is_time else block.x_name,
            block.x_unit,
        )))
        y_names, y_units = dict(block.y_names), dict(block.y_units)
        headers.extend(
            encode_clipboard_text(copy_column_header(block.label, y_names[name], y_units[name]))
            for name in series_names
        )
        # Extend the one combined header in roster order.  Each block remains
        # an adjacent, independently identified column group.
        combined_headers.extend(headers)

    def iter_block_rows(plan: _BlockPlan) -> Iterator[tuple[str, ...]]:
        columns, keep, _rows = plan
        matrix = np.column_stack([np.concatenate(parts).astype(float, copy=False) for parts in columns])
        if keep is not None:
            matrix = matrix[keep]
        for row in matrix:
            yield tuple(repr(value) if math.isfinite(value) else "" for value in row.tolist())

    output_chunks: list[str] = []
    total_characters = 0
    output = io.StringIO()
    output.write(encode_tsv_row(combined_headers))
    rows_in_chunk = 1
    row_iterators = [
        iter_block_rows(plan)
        for _block, _series_names, plan, _width, _separator in block_specs
    ]
    for row_index in range(planned_rows):
        row: list[str] = []
        for iterator, (_block, _series_names, plan, width, separator) in zip(
            row_iterators, block_specs, strict=True
        ):
            if separator:
                row.append("")
            if row_index < plan[2]:
                row.extend(next(iterator))
            else:
                row.extend("" for _ in range(width))
        output.write(encode_tsv_row(row))
        rows_in_chunk += 1
        if rows_in_chunk >= rows_per_chunk:
            piece = output.getvalue()
            total_characters += len(piece)
            if total_characters > _MAX_RENDERED_COPY_CHARACTERS:
                raise ValueError(
                    f"{action_name} payload exceeds the bounded output-size limit."
                )
            output_chunks.append(piece)
            output = io.StringIO()
            rows_in_chunk = 0
    piece = output.getvalue()
    total_characters += len(piece)
    if total_characters > _MAX_RENDERED_COPY_CHARACTERS:
        raise ValueError(
            f"{action_name} payload exceeds the bounded output-size limit."
        )
    if piece:
        output_chunks.append(piece)
    return tuple(output_chunks)
