"""PyQtGraph workspace surface for one immutable result input."""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Iterator

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.project_presentation_state import (
    SimulationSeriesAxisSelection,
)
from kindred.core.result_repository import SimulationResultValue
from kindred.gui.display_name_policy import literal_tooltip
from kindred.gui.theme_manager import plot_background_color, plot_foreground_color
from kindred.gui.ui_helpers import clamp_initial_size
from kindred.gui.interactive_preview import InteractivePreviewRequest
from kindred.gui.plot_config import (
    CONCENTRATION_DISPLAY_UNIT_FACTORS,
    TIME_DISPLAY_UNIT_FACTORS,
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
)

pg = require_pyqtgraph(context="Kindred GUI")

__all__ = ["PyQtGraphPlotPanel"]


@dataclass(frozen=True, slots=True)
class _PlotPresentation:
    log_x: bool = False
    log_y: bool = False
    legend_visible: bool = True
    x_inverted: bool = False
    y_inverted: bool = False


@dataclass(frozen=True, slots=True)
class _AxisRangeSession:
    auto: bool = True
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None


_EPSILON = 1e-9
_DisplayAnnotation = tuple[str, str, Mapping[str, object], float]
_AnnotationKey = tuple[str, str, tuple[tuple[str, str], ...]]
_ANNOTATION_ACCENTS = ("#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2")


@dataclass(frozen=True, slots=True)
class _InteractiveDisplayLayer:
    projection: InteractiveDisplayProjection
    opacity: float


@dataclass(frozen=True, slots=True)
class _InteractiveDisplayTransition:
    source: tuple[_InteractiveDisplayLayer, ...]
    target: InteractiveDisplayProjection
    started_at: float
    interpolate: bool


@dataclass(frozen=True, slots=True)
class _InteractiveDisplayState:
    layers: tuple[_InteractiveDisplayLayer, ...]


def _same_interactive_topology(
    left: InteractiveDisplayCurve,
    right: InteractiveDisplayCurve,
) -> bool:
    if left.segment_ids is None or right.segment_ids is None:
        return left.segment_ids is None and right.segment_ids is None
    return bool(
        np.array_equal(left.segment_ids, right.segment_ids)
        and left.segment_keys == right.segment_keys
        and np.array_equal(left.discontinuity_ids, right.discontinuity_ids)
        and left.intervention_connectors == right.intervention_connectors
    )


@dataclass(frozen=True, slots=True)
class _RenderedPointChunk:
    """One exact point chunk handed to a static or interactive plot item."""

    x: np.ndarray
    series: tuple[tuple[str, np.ndarray], ...]
    source_row: np.ndarray | None = None
    segment_ids: np.ndarray | None = None
    segment_keys: tuple[str, ...] | None = None
    discontinuity_ids: np.ndarray | None = None

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
        for name in ("source_row", "segment_ids", "discontinuity_ids"):
            values = getattr(self, name)
            if values is None:
                continue
            values = np.asarray(values)
            values.setflags(write=False)
            if values.size != x.size:
                raise ValueError(f"rendered point {name} must align with X.")
            object.__setattr__(self, name, values)
        if self.segment_keys is not None:
            keys = tuple(str(value) for value in self.segment_keys)
            if len(keys) != x.size:
                raise ValueError("rendered point segment_keys must align with X.")
            object.__setattr__(self, "segment_keys", keys)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "series", series)


@dataclass(frozen=True, slots=True)
class _RenderedPointBlock:
    """Private sampled roster entry; it is not result or export authority."""

    block_id: str
    label: str
    kind: str
    x_name: str
    x_unit: str | None
    y_unit: str | None
    x_provenance: Mapping[str, object]
    series_provenance: tuple[tuple[str, Mapping[str, object]], ...]
    chunks: tuple[_RenderedPointChunk, ...]

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
        object.__setattr__(
            self,
            "y_unit",
            None if self.y_unit is None else str(self.y_unit),
        )
        object.__setattr__(
            self,
            "x_provenance",
            MappingProxyType(dict(self.x_provenance)),
        )
        object.__setattr__(
            self,
            "series_provenance",
            tuple(
                (str(name), MappingProxyType(dict(provenance)))
                for name, provenance in self.series_provenance
            ),
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
    plotPresentationIntentRequested = QtCore.Signal(str, bool)
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
        self._annotation_labels: dict[
            tuple[object, ...], tuple[object, QtCore.QPointF]
        ] = {}
        self._zoom_enabled = False
        self._presentation = _PlotPresentation()
        self._axis_range = _AxisRangeSession()
        self._x_time_unit = "s"
        self._x_concentration_unit = "M"
        self._y_concentration_unit = "M"
        self._render_input: ResultRenderInput | None = None
        self._copy_all_available = False
        self._rendered_point_roster: tuple[_RenderedPointBlock, ...] = ()
        self._settling_static_input = False
        self._interactive_timer = QtCore.QTimer(self)
        self._interactive_timer.setSingleShot(True)
        self._interactive_timer.timeout.connect(self._advance_interactive_display)
        self._interactive_items: dict[tuple[object, ...], Any] = {}
        self._interactive_visible: tuple[_InteractiveDisplayLayer, ...] | None = None
        self._interactive_transition: _InteractiveDisplayTransition | None = None
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

        self._plot_widget = pg.PlotWidget(self)
        self._plot_widget.setObjectName("plotViewport")
        self._plot_widget.plotItem.setContentsMargins(0, 0, 0, 10)
        self._plot_widget.setBackground(plot_background_color(False))
        self._plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self._legend = self._plot_widget.addLegend(
            offset=(-10, 10),
            labelTextColor=plot_foreground_color(self._plot_widget.backgroundBrush().color().lightness() < 128),
        )
        self._plot_widget.getPlotItem().setMenuEnabled(False)
        self._plot_widget.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._plot_widget.customContextMenuRequested.connect(
            self._show_plot_context_menu
        )
        self._axis_toolbar = AxisToolbar(self)
        self._axis_toolbar.xChanged.connect(self.xSeriesRequested.emit)
        self._axis_toolbar.ySelectionChanged.connect(self.ySeriesRequested.emit)
        self._axis_toolbar.axisRangeChanged.connect(self._apply_axis_range)
        self._axis_toolbar.unitChanged.connect(self._apply_display_unit_change)
        self._plot_widget.getViewBox().sigRangeChangedManually.connect(
            self._on_manual_navigation
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

    def apply_result_input(self, render_input: ResultRenderInput) -> None:
        """Retain and draw one immutable result input and its live aggregate."""

        previous_input = self._render_input
        self._rendered_point_roster = ()
        self._settling_static_input = True
        try:
            plotted_set_ids = {
                str(block.block_id)
                for block in render_input.blocks
                if block.kind
                in {ResultLayerKind.EXPLICIT, ResultLayerKind.PREVIEW}
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
                or self._interactive_transition is not None
            ):
                self._reconcile_cached_interactive_display(
                    previous_input,
                    render_input,
                )
            if not (
                self._interactive_results
                or self._interactive_pending
                or self._interactive_transition
                or self._interactive_visible
                or self._interactive_target_set_ids
            ):
                self._end_interactive_gesture_session()
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

        projections = tuple(
            layer.projection for layer in self._interactive_visible or ()
        )
        if self._interactive_pending is not None:
            projections = (*projections, self._interactive_pending)
        if self._interactive_transition is not None:
            projections = (
                *projections,
                self._interactive_transition.target,
                *(layer.projection for layer in self._interactive_transition.source),
            )
        if not projections:
            return
        if not self._cached_interactive_display_compatible(
            previous_input,
            render_input,
            projections,
        ):
            self._clear_interactive_projection()
            return

        visible = tuple(
            _InteractiveDisplayLayer(
                self._filter_cached_interactive_projection(
                    layer.projection,
                    render_input,
                ),
                layer.opacity,
            )
            for layer in self._interactive_visible or ()
        )
        visible = tuple(
            layer for layer in visible if layer.projection is not None
        )
        if not visible and self._interactive_pending is not None:
            # A pending source-less target has already been projected.  It is
            # safe to expose it as the retained frame after compatibility is
            # established, while cancelling its obsolete transition custody.
            projected = self._filter_cached_interactive_projection(
                self._interactive_pending,
                render_input,
            )
            if projected is not None:
                visible = (_InteractiveDisplayLayer(projected, 1.0),)
        self._interactive_pending = None
        self._interactive_transition = None
        self._cancel_interactive_tick()
        if not visible:
            self._clear_interactive_projection()
            return
        state = _InteractiveDisplayState(visible)
        if not self._render_interactive_display(state):
            return
        self._interactive_visible = visible
        self._interactive_rendered_frame = True
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
                names_by_target.setdefault(str(key[2]), set()).add(str(key[3]))
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
            selection.series_name
            for selection in render_input.selected_y_series
            if isinstance(selection, SimulationSeriesAxisSelection)
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
            if wanted is not None and str(key[3]) not in wanted:
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
        groups = (
            (("copy_all", "Copy All"), ("copy_plotted", "Copy Plotted Data")),
            (("zoom", "Zoom"), ("auto_range", "Auto Range")),
            (
                ("legend_visible", "Show Legend"),
                ("reference_lines_visible", "Show Reference"),
                ("intervention_annotations_visible", "Show Intervention Indicators"),
                ("intervention_labels_visible", "Show Intervention Labels"),
            ),
            (
                ("log_x", "Log X"), ("log_y", "Log Y"),
                ("x_inverted", "Invert X Axis"), ("y_inverted", "Invert Y Axis"),
            ),
            (("sampling", "Downsampling: 1k points/curve"), ("warnings", "Warning details…")),
        )
        for index, group in enumerate(groups):
            if index:
                self._plot_menu.addSeparator()
            for key, text in group:
                action = self._plot_menu.addAction(text)
                action.setCheckable(key not in {"copy_all", "copy_plotted", "warnings"})
                self._plot_actions[key] = action
        actions = self._plot_actions
        actions["copy_all"].setToolTip(
            "Copy full-resolution results and references for all initial-condition "
            "sets in this mechanism, including data not currently shown, plus linked "
            "experimental data. Uses canonical units and ignores plot selection and downsampling."
        )
        actions["copy_plotted"].setToolTip(
            "Copy plotted series using the selected axes, display units, and downsampling. "
            "Includes points outside the current view."
        )
        actions["zoom"].setToolTip("Drag a rectangle to zoom. When off, drag to pan.")
        actions["auto_range"].setToolTip("Automatically scale axes to fit all visible data.")
        actions["legend_visible"].setToolTip("Show the names and colors of the plotted curves.")
        actions["reference_lines_visible"].setToolTip("Show reference curves alongside current results.")
        actions["intervention_annotations_visible"].setToolTip(
            "Show intervention lines and shaded regions. Turning this on or off "
            "also turns labels on or off."
        )
        actions["intervention_labels_visible"].setToolTip(
            "Show draggable text labels for interventions. Turning this on also enables indicators."
        )
        actions["log_x"].setToolTip("Display the X axis on a logarithmic scale.")
        actions["log_y"].setToolTip("Display the Y axis on a logarithmic scale.")
        actions["x_inverted"].setToolTip("Reverse the direction of the X axis.")
        actions["y_inverted"].setToolTip("Reverse the direction of the Y axis.")
        actions["sampling"].setToolTip(
            "Plot about 1,000 points per curve, preserving intervention boundaries. "
            "Copy All keeps full resolution."
        )
        actions["warnings"].setToolTip("View details of warnings for the current plot.")
        actions["copy_all"].triggered.connect(self.copyAllRequested.emit)
        actions["copy_plotted"].triggered.connect(self._copy_rendered_points_to_clipboard)
        actions["zoom"].triggered.connect(self._set_zoom_enabled)
        actions["auto_range"].triggered.connect(self._set_auto_range)
        actions["reference_lines_visible"].triggered.connect(self.referenceLayerVisibilityRequested)
        actions["intervention_annotations_visible"].triggered.connect(self.interventionAnnotationsVisibilityRequested)
        actions["intervention_labels_visible"].triggered.connect(self.interventionLabelsVisibilityRequested)
        for option in ("legend_visible", "log_x", "log_y", "x_inverted", "y_inverted"):
            actions[option].triggered.connect(
                lambda enabled, key=option: self._request_plot_presentation(key, enabled)
            )
        actions["sampling"].triggered.connect(self._set_coarse_sampling)
        actions["warnings"].triggered.connect(self._show_completion_warning_details)
        self._plot_menu.aboutToShow.connect(self._sync_plot_controls)

    def _sync_plot_controls(self) -> None:
        """Project current owner state; QAction.triggered emits user commands only."""
        values = {
            "zoom": self._zoom_enabled,
            "auto_range": self._axis_range.auto,
            "sampling": self._sampling_limit is not None,
            "legend_visible": self._presentation.legend_visible,
            "log_x": self._presentation.log_x,
            "log_y": self._presentation.log_y,
            "x_inverted": self._presentation.x_inverted,
            "y_inverted": self._presentation.y_inverted,
        }
        for key in ("reference_lines_visible", "intervention_annotations_visible", "intervention_labels_visible"):
            values[key] = bool(self._render_input and getattr(self._render_input, key))
        for key, checked in values.items():
            self._plot_actions[key].setChecked(checked)
        self._plot_actions["copy_all"].setEnabled(self._copy_all_available)
        self._plot_actions["copy_plotted"].setEnabled(
            any(chunk.x.size for block in self._rendered_point_roster for chunk in block.chunks)
        )
        self._plot_actions["warnings"].setEnabled(bool(self._completion_warning_badge.text()))
        self._plot_actions["auto_range"].setEnabled(not self._interactive_gesture_active)
        self._axis_toolbar.set_manual_range_controls(
            visible=not self._axis_range.auto,
            enabled=not self._interactive_gesture_active,
        )

    def _show_plot_context_menu(self, position: QtCore.QPoint) -> None:
        self._plot_menu.exec(self._plot_widget.mapToGlobal(position))

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

    def set_persisted_plot_state(
        self,
        *,
        log_x: bool,
        log_y: bool,
        legend_visible: bool,
        x_inverted: bool,
        y_inverted: bool,
    ) -> None:
        """Apply one Integration-owned persisted Plot projection."""

        state = _PlotPresentation(
            log_x=bool(log_x),
            log_y=bool(log_y),
            legend_visible=bool(legend_visible),
            x_inverted=bool(x_inverted),
            y_inverted=bool(y_inverted),
        )
        self._presentation = state
        self._apply_plot_presentation(state)
        if self._render_input is not None:
            self._clear_annotation_items()
            self._render_annotations(self._render_input, state)

    def drop_tab_session_state(self) -> None:
        self.dispose_project_content()
        self._zoom_enabled = False
        self._presentation = _PlotPresentation()
        self._sampling_limit = None
        self._axis_range = _AxisRangeSession()
        self._x_time_unit = "s"
        self._x_concentration_unit = "M"
        self._y_concentration_unit = "M"

    def reset_plot_session_state(self) -> None:
        """Project replacement/reopen resets non-archived Plot session truth."""

        self.dispose_project_content()
        self._end_interactive_gesture_session()
        self._zoom_enabled = False
        self._presentation = _PlotPresentation()
        self._sampling_limit = None
        self._axis_range = _AxisRangeSession()
        self._x_time_unit = "s"
        self._x_concentration_unit = "M"
        self._y_concentration_unit = "M"
        self._axis_toolbar.set_axis_unit_choices(
            x_units=tuple(TIME_DISPLAY_UNIT_FACTORS),
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
            and any(
                layer.projection.curves for layer in self._interactive_visible
            )
        )

    @property
    def _interactive_has_pending_batch(self) -> bool:
        return bool(
            self._interactive_results
            or self._interactive_pending
            or self._interactive_transition
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

        # The visible frame and its physical items are the next transition's
        # source. Only the pending target and aggregate belong to the old
        # request and must be discarded.
        self._interactive_results.clear()
        self._interactive_pending = None
        self._interactive_transition = None
        self._interactive_target_set_ids = target_set_ids
        self._interactive_request = request
        self._interactive_frozen = False
        self._cancel_interactive_tick()

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
            target = self._interactive_pending
            self._interactive_pending = None
            source = self._interactive_visible or ()
            self._interactive_transition = _InteractiveDisplayTransition(
                source,
                target,
                now,
                self._can_interpolate_interactive(source, target),
            )
        state = self._materialize_interactive(now)
        if state is None:
            return
        if self._render_interactive_display(state):
            self._interactive_visible = state.layers
            self._interactive_last_presented_at = now
            self._interactive_projection_sampling_limit = self._sampling_limit
            self._interactive_projection_sampling_known = True
        if self._interactive_transition is not None or self._interactive_pending is not None:
            self._schedule_interactive_tick()

    def _clear_interactive_display(self, *, reset_currentness: bool = False) -> None:
        had_physical_state = bool(
            self._interactive_visible
            or self._interactive_rendered_frame
            or self._interactive_pending
            or self._interactive_transition
        )
        if had_physical_state:
            if not self._render_interactive_display(_InteractiveDisplayState(())):
                return
            self._interactive_visible = None
            self._interactive_rendered_frame = False
            self._interactive_last_presented_at = None
            self._interactive_projection_sampling_limit = None
            self._interactive_projection_sampling_known = False
        self._interactive_pending = None
        self._interactive_transition = None
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
        for layer in self._interactive_visible or ():
            known.update(self._interactive_projection_target_ids(layer.projection))
        known.update(self._interactive_projection_target_ids(self._interactive_pending))
        if self._interactive_transition is not None:
            known.update(self._interactive_projection_target_ids(self._interactive_transition.target))
            for layer in self._interactive_transition.source:
                known.update(self._interactive_projection_target_ids(layer.projection))
        affected = targets.intersection(known)
        if not affected:
            return True
        visible = tuple(
            _InteractiveDisplayLayer(filtered, layer.opacity)
            for layer in self._interactive_visible or ()
            if (filtered := self._filter_interactive_projection(layer.projection, affected)) is not None
        )
        if visible and self._render_input is not None:
            # Restore every retained item from semantic input first; rendering
            # the surviving transient targets will suppress only those targets.
            self._render_result_scene(self._render_input)
        if not self._render_interactive_display(_InteractiveDisplayState(visible)):
            return False
        self._interactive_visible = visible or None
        self._interactive_pending = None
        self._interactive_transition = None
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
        self._interactive_rendered_frame = bool(self._interactive_visible)
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
        self._interactive_transition = None
        self._cancel_interactive_tick()
        if self._interactive_visible or self._interactive_rendered_frame:
            self._render_interactive_display(_InteractiveDisplayState(()))
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
        self._interactive_transition = None
        self._cancel_interactive_tick()
        state = _InteractiveDisplayState(
            (_InteractiveDisplayLayer(projection, 1.0),)
        )
        if self._render_interactive_display(state):
            self._interactive_visible = state.layers
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
        if self._render_input is None:
            return None
        return compose_interactive_display(
            result,
            tab_id=self._render_input.tab_id,
            set_id=set_id,
            render_input=self._render_input,
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
        return replace(
            curve,
            x=curve.x[sample],
            y=curve.y[sample],
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
    def _can_interpolate_interactive(
        source: tuple[_InteractiveDisplayLayer, ...],
        target: InteractiveDisplayProjection,
    ) -> bool:
        if len(source) != 1 or source[0].opacity < 1.0 - _EPSILON:
            return False
        source_curves = source[0].projection.curves
        target_curves = target.curves
        if tuple(curve.key for curve in source_curves) != tuple(curve.key for curve in target_curves):
            return False
        return all(
            left.markers == right.markers
            and left.x.shape == right.x.shape
            and left.y.shape == right.y.shape
            and np.array_equal(left.x, right.x)
            and _same_interactive_topology(left, right)
            for left, right in zip(source_curves, target_curves, strict=True)
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

    def _materialize_interactive(
        self,
        now: float,
    ) -> _InteractiveDisplayState | None:
        if self._interactive_transition is None:
            if self._interactive_visible is None:
                return None
            return _InteractiveDisplayState(self._interactive_visible)
        transition = self._interactive_transition
        progress = min(
            1.0,
            max(0.0, (now - transition.started_at) / 0.12),
        )
        eased = progress * progress * (3.0 - 2.0 * progress)
        layers: tuple[_InteractiveDisplayLayer, ...]
        if transition.interpolate:
            projection = self._interpolate_interactive_projection(
                transition.source[0].projection,
                transition.target,
                eased,
            )
            layers = (_InteractiveDisplayLayer(projection, 1.0),)
        else:
            layers = tuple(
                _InteractiveDisplayLayer(
                    item.projection,
                    item.opacity * (1.0 - eased),
                )
                for item in transition.source
                if item.opacity * (1.0 - eased) > _EPSILON
            )
            if eased > _EPSILON:
                layers = (
                    *layers,
                    _InteractiveDisplayLayer(transition.target, eased),
                )
            if not layers:
                layers = (_InteractiveDisplayLayer(transition.target, 1.0),)
        if progress >= 1.0:
            layers = (_InteractiveDisplayLayer(transition.target, 1.0),)
            self._interactive_transition = None
        return _InteractiveDisplayState(layers)

    @staticmethod
    def _interpolate_interactive_projection(
        source: InteractiveDisplayProjection,
        target: InteractiveDisplayProjection,
        progress: float,
    ) -> InteractiveDisplayProjection:
        curves = tuple(
            replace(left, y=left.y + (right.y - left.y) * progress)
            for left, right in zip(source.curves, target.curves, strict=True)
        )
        annotations = tuple(
            (role, set_id, payload, opacity * weight)
            for values, weight in (
                (source.intervention_annotations, 1.0 - progress),
                (target.intervention_annotations, progress),
            )
            for role, set_id, payload, opacity in values
            if opacity * weight > _EPSILON
        )
        return replace(
            target, curves=curves,
            intervention_annotations=PyQtGraphPlotPanel._merge_annotation_values(annotations),
        )

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
        state: _InteractiveDisplayState,
    ) -> bool:
        self._rendered_point_roster = ()
        if not state.layers:
            self._release_interactive_display()
            return True

        existing = dict(self._interactive_items)
        rendered: dict[tuple[object, ...], Any] = {}
        rendered_points: list[_RenderedPointBlock] = []
        render_input = self._render_input
        transient_set_ids = {
            str(curve.key[2])
            for layer in state.layers
            for curve in layer.projection.curves
            if len(curve.key) >= 3
            and curve.key[0:2] == ("simulation", "foreground")
        }
        transient_reference_set_ids = {
            str(curve.key[2])
            for layer in state.layers
            for curve in layer.projection.curves
            if len(curve.key) >= 3
            and curve.key[0:2] == ("simulation", "ghost")
        }
        replaced_annotation_owners = (
            {("foreground", set_id) for set_id in transient_set_ids}
            | {("ghost", set_id) for set_id in transient_reference_set_ids}
            | {(role, set_id)
            for layer in state.layers
            for role, set_id, _payload, _opacity
            in layer.projection.intervention_annotations}
        )
        _x_unit, _y_unit, x_scale, y_scale = (
            self._display_unit_state(render_input)
            if render_input is not None
            else (None, None, 1.0, 1.0)
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
                    rendered_points.append(rendered_block)
        for layer_index, layer in enumerate(state.layers):
            for curve in layer.projection.curves:
                style = "marker" if curve.markers else "line"
                color = QtGui.QColor(curve.color)
                opacity = layer.opacity * curve.opacity
                visual_break_indices = tuple(
                    chunk_index
                    for chunk_index, (start, _end) in enumerate(
                        self._interactive_curve_chunks(curve)[1:], 1
                    )
                    if any(
                        float(curve.x[start]) == float(connector[1])
                        for connector in curve.intervention_connectors
                    )
                    or (
                        render_input is not None
                        and not self._has_time_x(render_input)
                        and (
                            curve.x[start] != curve.x[start - 1]
                            or curve.y[start] != curve.y[start - 1]
                        )
                    )
                )
                for chunk_index, (start, end) in enumerate(
                    self._interactive_curve_chunks(
                        curve,
                        visual_break_indices=visual_break_indices,
                    )
                ):
                    displayed_x = self._scaled_display_values(
                        curve.x[start:end],
                        x_scale,
                    )
                    displayed_y = self._scaled_display_values(
                        curve.y[start:end],
                        y_scale,
                    )
                    # Retained static items own the stable legend rows.  The
                    # transient projection supplies pixels only.
                    curve_name = None
                    item_key = (
                        layer_index,
                        *curve.key,
                        "segment",
                        chunk_index,
                        style,
                    )
                    item = existing.get(item_key)
                    if item is None:
                        if curve.markers:
                            item = self._plot_widget.plot(
                                displayed_x,
                                displayed_y,
                                pen=None,
                                symbol="o",
                                symbolSize=curve.point_size,
                                symbolBrush=pg.mkBrush(color),
                                symbolPen=pg.mkPen(color),
                                name=curve_name,
                            )
                        else:
                            item = self._plot_widget.plot(
                                displayed_x,
                                displayed_y,
                                pen=pg.mkPen(
                                    color=color,
                                    width=2,
                                    style=(
                                        QtCore.Qt.PenStyle.DashLine
                                        if curve.dashed
                                        else QtCore.Qt.PenStyle.SolidLine
                                    ),
                                ),
                                name=curve_name,
                            )
                    else:
                        item.setData(displayed_x, displayed_y)
                        if curve.markers:
                            item.setSymbolSize(curve.point_size)
                            item.setSymbolBrush(pg.mkBrush(color))
                            item.setSymbolPen(pg.mkPen(color))
                        else:
                            item.setPen(
                                pg.mkPen(
                                    color=color,
                                    width=2,
                                    style=(
                                        QtCore.Qt.PenStyle.DashLine
                                        if curve.dashed
                                        else QtCore.Qt.PenStyle.SolidLine
                                    ),
                                )
                            )
                    self._update_plot_item_legend(item, curve_name)
                    item.setOpacity(opacity)
                    rendered[item_key] = item
                for connector in curve.intervention_connectors:
                    boundary_index, boundary_time, pre_value, post_value = connector
                    displayed_time = boundary_time * x_scale
                    displayed_pre = pre_value * y_scale
                    displayed_post = post_value * y_scale
                    item_key = (
                        layer_index,
                        *curve.key,
                        "intervention-boundary",
                        boundary_index,
                    )
                    item = existing.get(item_key)
                    pen = pg.mkPen(
                        color=color,
                        width=2,
                        style=(
                            QtCore.Qt.PenStyle.DashLine
                            if curve.dashed
                            else QtCore.Qt.PenStyle.SolidLine
                        ),
                    )
                    if item is None:
                        item = self._plot_widget.plot(
                            [displayed_time, displayed_time],
                            [displayed_pre, displayed_post],
                            pen=pen,
                            name=None,
                        )
                    else:
                        item.setData(
                            [displayed_time, displayed_time],
                            [displayed_pre, displayed_post],
                        )
                        item.setPen(pen)
                    item.setOpacity(opacity)
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
                (role, set_id, payload, opacity * layer.opacity)
                for layer in state.layers
                for role, set_id, payload, opacity
                in layer.projection.intervention_annotations
                if opacity * layer.opacity > _EPSILON
            )
            if len(state.layers) > 1:
                interactive_annotations = self._merge_annotation_values(
                    interactive_annotations
                )
            self._render_annotations(
                render_input,
                self._presentation,
                annotations=(*static_annotations, *interactive_annotations),
            )
        # Copy Plotted Data is a semantic settled projection. During a cross-fade
        # the transition source and target are both physical layers, but only
        # the target belongs in the copy roster; hidden static target items
        # are already represented by that projection above.
        settled_projection = (
            self._interactive_transition.target
            if self._interactive_transition is not None
            else state.layers[-1].projection
            if state.layers
            else None
        )
        if render_input is not None and settled_projection is not None:
            for curve in settled_projection.curves:
                rendered_block = self._rendered_block_for_interactive_curve(
                    curve,
                    render_input,
                )
                if rendered_block is not None and any(
                    chunk.x.size for chunk in rendered_block.chunks
                ):
                    rendered_points.append(rendered_block)
        self._rendered_point_roster = tuple(rendered_points)
        if rendered and not self._interactive_rendered_frame:
            self._begin_interactive_gesture_session()
        self._interactive_rendered_frame = bool(rendered)
        return bool(rendered) or not state.layers

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
        *,
        visual_break_indices: Sequence[int] | None = None,
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
        starts = [0, *changes] if visual_break_indices is None else [
            0,
            *(changes[index - 1] for index in visual_break_indices),
        ]
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
        dialog.setWindowTitle("Completion warning details")
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

    def _display_unit_state(
        self,
        render_input: ResultRenderInput,
    ) -> tuple[str | None, str | None, float, float]:
        if self._has_time_x(render_input):
            x_unit = self._x_time_unit
            x_scale = 1.0 / TIME_DISPLAY_UNIT_FACTORS[x_unit]
        elif render_input.x_is_concentration:
            x_unit = self._x_concentration_unit
            x_scale = 1.0 / CONCENTRATION_DISPLAY_UNIT_FACTORS[x_unit]
        else:
            x_unit = None
            x_scale = 1.0
        if render_input.y_is_concentration:
            y_unit = self._y_concentration_unit
            y_scale = 1.0 / CONCENTRATION_DISPLAY_UNIT_FACTORS[y_unit]
        else:
            y_unit = None
            y_scale = 1.0
        return x_unit, y_unit, x_scale, y_scale

    def _configure_axis_unit_controls(
        self,
        render_input: ResultRenderInput,
    ) -> None:
        x_unit, y_unit, _x_scale, _y_scale = self._display_unit_state(
            render_input
        )
        if self._has_time_x(render_input):
            x_units = tuple(TIME_DISPLAY_UNIT_FACTORS)
        elif render_input.x_is_concentration:
            x_units = tuple(CONCENTRATION_DISPLAY_UNIT_FACTORS)
        else:
            x_units = ()
        y_units = (
            tuple(CONCENTRATION_DISPLAY_UNIT_FACTORS)
            if render_input.y_is_concentration
            else ()
        )
        self._axis_toolbar.set_axis_unit_choices(
            x_units=x_units,
            selected_x=x_unit,
            y_units=y_units,
            selected_y=y_unit,
        )

    def _display_axis_labels(
        self,
        render_input: ResultRenderInput,
    ) -> tuple[str, str]:
        x_unit, y_unit, _x_scale, _y_scale = self._display_unit_state(
            render_input
        )
        if self._has_time_x(render_input):
            x_label = f"Time ({x_unit})"
        elif x_unit is not None:
            x_label = f"{render_input.x_label} ({x_unit})"
        else:
            x_label = render_input.x_label
        y_label = (
            f"Concentration ({y_unit})"
            if y_unit is not None
            else render_input.y_label
        )
        return x_label, y_label

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
                    render_input.selected_y_uses_default_all
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
        sample, chunks = self._sample_and_chunks(block, sampling_limit)
        render_input = self._render_input
        x_unit, y_unit, x_scale, y_scale = (
            self._display_unit_state(render_input)
            if render_input is not None
            else (None, None, 1.0, 1.0)
        )
        x = self._scaled_display_values(block.t[sample], x_scale)
        sampled_series = tuple(
            (
                name,
                self._scaled_display_values(values[sample], y_scale),
            )
            for name, values in block.series
        )
        source_row = (
            None if block.source_row is None else block.source_row[sample]
        )
        segment_ids = (
            None if block.segment_ids is None else block.segment_ids[sample]
        )
        discontinuity_ids = (
            None
            if block.discontinuity_ids is None
            else block.discontinuity_ids[sample]
        )
        segment_keys = (
            None
            if block.segment_keys is None
            else tuple(block.segment_keys[int(index)] for index in sample)
        )
        rendered_chunks = tuple(
            _RenderedPointChunk(
                x=x[start:end],
                series=tuple(
                    (name, values[start:end]) for name, values in sampled_series
                ),
                source_row=(
                    None if source_row is None else source_row[start:end]
                ),
                segment_ids=(
                    None if segment_ids is None else segment_ids[start:end]
                ),
                segment_keys=(
                    None if segment_keys is None else segment_keys[start:end]
                ),
                discontinuity_ids=(
                    None
                    if discontinuity_ids is None
                    else discontinuity_ids[start:end]
                ),
            )
            for start, end in chunks
        )
        rendered_block = _RenderedPointBlock(
            block_id=block.block_id,
            label=block.label,
            kind=block.kind.value,
            x_name=block.x_name,
            x_unit=x_unit,
            y_unit=y_unit,
            x_provenance=block.x_provenance,
            series_provenance=block.series_provenance,
            chunks=rendered_chunks,
        )
        return rendered_block, rendered_chunks

    def _rendered_block_for_interactive_curve(
        self,
        curve: object,
        render_input: ResultRenderInput,
    ) -> _RenderedPointBlock | None:
        key = tuple(curve.key)
        if len(key) < 4 or key[0] != "simulation":
            return None
        role, set_id, series_name = key[1], str(key[2]), str(key[3])
        if role == "ghost":
            source = next(
                (
                    block
                    for block in render_input.blocks
                    if block.block_id == set_id
                    and block.kind is ResultLayerKind.REFERENCE
                ),
                None,
            )
        else:
            source = next(
                (
                    block
                    for block in render_input.blocks
                    if block.block_id == set_id
                    and block.kind
                    in {ResultLayerKind.EXPLICIT, ResultLayerKind.PREVIEW}
                ),
                None,
            )
        if source is None:
            return None
        x_unit, y_unit, x_scale, y_scale = self._display_unit_state(
            render_input
        )
        series_provenance = dict(source.series_provenance)
        curve_chunks = tuple(
            _RenderedPointChunk(
                x=self._scaled_display_values(
                    curve.x[start:end],
                    x_scale,
                ),
                series=(
                    (
                        series_name,
                        self._scaled_display_values(
                            curve.y[start:end],
                            y_scale,
                        ),
                    ),
                ),
                segment_ids=(
                    None
                    if curve.segment_ids is None
                    else curve.segment_ids[start:end]
                ),
                segment_keys=(
                    None
                    if curve.segment_keys is None
                    else curve.segment_keys[start:end]
                ),
                discontinuity_ids=(
                    None
                    if curve.discontinuity_ids is None
                    else curve.discontinuity_ids[start:end]
                ),
            )
            for start, end in self._interactive_curve_chunks(curve)
        )
        return _RenderedPointBlock(
            block_id=set_id,
            label=source.label,
            kind=source.kind.value,
            x_name=source.x_name,
            x_unit=x_unit,
            y_unit=y_unit,
            x_provenance=source.x_provenance,
            series_provenance=(
                (series_name, series_provenance.get(series_name, {})),
            ),
            chunks=curve_chunks,
        )

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
        colors = dict(block.color_by_series)
        item_store = rendered_items if rendered_items is not None else {}
        existing_store = existing_items if existing_items is not None else {}
        role = (block.kind.value, block.block_id)
        visual_breaks: dict[str, set[int]] = {
            name: set() for name, _values in block.series
        }
        if block.result is not None:
            result = block.result
            result_chunks = self._execution_region_chunks(
                np.asarray(result.segment_ids).reshape(-1),
                tuple(result.segment_keys),
                np.asarray(result.discontinuity_ids).reshape(-1),
            )
            transition_times = tuple(
                float(result.t[start]) for start, _end in result_chunks[1:]
            )
            requested = set(result.requested_algebra_outputs)
            x_name = None if block.x_axis.series_name == "t" else block.x_axis.series_name
            boundary_by_time = {
                float(item.time): (dict(item.pre_state), dict(item.post_state))
                for item in block.presentation_boundaries
            }
            states = tuple(boundary_by_time.get(time) for time in transition_times)
            visual_breaks = {
                name: {
                    chunk_index
                    for chunk_index, state in enumerate(states, 1)
                    if (
                        ((x_name is not None and x_name in requested) or name in requested)
                        if state is None
                        else any(
                            axis_name not in state[0]
                            or axis_name not in state[1]
                            or state[0][axis_name] != state[1][axis_name]
                            for axis_name in (x_name, name)
                            if axis_name is not None
                        )
                    )
                }
                for name, _values in block.series
            }
        for series_name, _values in block.series:
            color = colors[series_name]
            physical_chunks: list[tuple[np.ndarray, np.ndarray]] = []
            for chunk_index, chunk in enumerate(rendered_chunks):
                x = chunk.x
                y = dict(chunk.series)[series_name]
                if chunk_index and chunk_index not in visual_breaks[series_name]:
                    prior_x, prior_y = physical_chunks[-1]
                    physical_chunks[-1] = (
                        np.concatenate((prior_x, x)),
                        np.concatenate((prior_y, y)),
                    )
                else:
                    physical_chunks.append((x, y))
            for chunk_index, (x, y) in enumerate(physical_chunks):
                singleton = x.size == 1
                marker = bool(block.markers or singleton)
                item_key = (role, series_name, chunk_index, "marker" if marker else "line")
                item = existing_store.get(item_key)
                name = f"{block.label}: {series_name}"
                if item is None:
                    if marker:
                        item = self._plot_widget.plot(
                            x, y, pen=None, symbol=(block.point_shape if block.markers else "o"),
                            symbolSize=(block.point_size if block.markers else 5.0),
                            symbolBrush=pg.mkBrush(color), symbolPen=pg.mkPen(color),
                            name=name if chunk_index == 0 else None,
                        )
                    else:
                        item = self._plot_widget.plot(
                            x, y,
                            pen=pg.mkPen(color=color, width=2, style=(QtCore.Qt.PenStyle.DashLine if block.dashed else QtCore.Qt.PenStyle.SolidLine)),
                            name=name if chunk_index == 0 else None,
                        )
                else:
                    item.setData(x, y)
                    if marker:
                        item.setSymbolBrush(pg.mkBrush(color))
                        item.setSymbolPen(pg.mkPen(color))
                        item.setSymbol(block.point_shape if block.markers else "o")
                        item.setSymbolSize(block.point_size if block.markers else 5.0)
                    else:
                        item.setPen(pg.mkPen(color=color, width=2, style=(QtCore.Qt.PenStyle.DashLine if block.dashed else QtCore.Qt.PenStyle.SolidLine)))
                self._update_plot_item_legend(item, name if chunk_index == 0 else None)
                item.setVisible(True)
                item.setOpacity(block.opacity)
                item_store[item_key] = item
        if rendered_points is not None and any(
            chunk.x.size for chunk in rendered_block.chunks
        ):
            rendered_points.append(rendered_block)
        render_input = self._render_input
        _x_unit, _y_unit, x_scale, y_scale = (
            self._display_unit_state(render_input)
            if render_input is not None
            else (None, None, 1.0, 1.0)
        )
        for boundary_index, species_name, boundary_time, pre_value, post_value in self._intervention_connector_entries(block):
            color = colors[species_name]
            item_key = (role, "intervention-boundary", boundary_index, species_name)
            pen = pg.mkPen(color=color, width=2, style=(QtCore.Qt.PenStyle.DashLine if block.dashed else QtCore.Qt.PenStyle.SolidLine))
            displayed_time = boundary_time * x_scale
            displayed_pre = pre_value * y_scale
            displayed_post = post_value * y_scale
            item = existing_store.get(item_key)
            if item is None:
                item = self._plot_widget.plot([displayed_time, displayed_time], [displayed_pre, displayed_post], pen=pen, name=None)
            else:
                item.setData([displayed_time, displayed_time], [displayed_pre, displayed_post])
                item.setPen(pen)
            item.setVisible(True)
            item.setOpacity(block.opacity)
            item_store[item_key] = item

    @staticmethod
    def _intervention_connector_entries(
        block: ResultDataBlock,
    ) -> tuple[tuple[int, str, float, float, float], ...]:
        """Return exact vertical connectors for a time-X simulation block."""

        if block.kind is ResultLayerKind.EXPERIMENTAL:
            return ()
        if not (
            isinstance(block.x_axis, SimulationSeriesAxisSelection)
            and block.x_axis.series_name == "t"
        ):
            return ()
        series_names = {name for name, _values in block.series}
        entries: list[tuple[int, str, float, float, float]] = []
        for boundary_index, boundary in enumerate(block.presentation_boundaries):
            pre_by_name = dict(boundary.pre_state)
            post_by_name = dict(boundary.post_state)
            for species_name in pre_by_name:
                if species_name not in series_names:
                    continue
                pre_value = pre_by_name[species_name]
                post_value = post_by_name[species_name]
                if pre_value == post_value:
                    continue
                entries.append(
                    (
                        boundary_index,
                        species_name,
                        float(boundary.time),
                        float(pre_value),
                        float(post_value),
                    )
                )
        return tuple(entries)

    def _sample_and_chunks(
        self,
        block: ResultDataBlock,
        sampling_limit: int | None,
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        sample = self._sample_indices(block.t.size, sampling_limit)
        if block.kind is not ResultLayerKind.EXPERIMENTAL:
            region_endpoints = self._execution_region_endpoints(block)
            sample = np.unique(np.concatenate((sample, region_endpoints)))
            if (
                block.segment_ids is None
                or block.segment_keys is None
                or block.discontinuity_ids is None
            ):
                raise ValueError("simulation plot block lacks executor identity.")
            segment_ids = block.segment_ids[sample]
            segment_keys = tuple(block.segment_keys[int(index)] for index in sample)
            chunks = self._execution_region_chunks(
                segment_ids,
                segment_keys,
                block.discontinuity_ids[sample],
            )
        else:
            chunks = ((0, sample.size),)
        return sample, chunks

    def _render_analysis(self, render_input: ResultRenderInput) -> None:
        self._parameter_table.update_parameters(
            {name: (value, unit) for name, value, unit in render_input.parameters}
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
        presentation: _PlotPresentation,
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
        presentation: _PlotPresentation,
    ) -> None:
        if (
            not render_input.intervention_annotations_visible
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

        view_range = self._plot_widget.getPlotItem().getViewBox().viewRange()
        x_min, x_max = (float(value) for value in view_range[0])
        y_min, y_max = (float(value) for value in view_range[1])
        x_span = x_max - x_min
        y_span = y_max - y_min
        if not np.isfinite(x_span) or x_span <= 0.0:
            x_span = 1.0
        if not np.isfinite(y_span) or y_span <= 0.0:
            y_min, y_max, y_span = 0.0, 1.0, 1.0
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
        rendered_labels: dict[
            tuple[object, ...], tuple[object, QtCore.QPointF]
        ] = {}
        for lane, (occurrence, candidate) in enumerate(label_candidates.items()):
            annotation, weighted_start, weight = candidate
            lane_offset = 0.10 + 0.08 * (lane % 4)
            label_y = (
                y_min + y_span * lane_offset
                if presentation.y_inverted
                else y_max - y_span * lane_offset
            )
            anchor = QtCore.QPointF(
                weighted_start / weight
                + x_span * (-0.012 if presentation.x_inverted else 0.012),
                label_y,
            )
            prior = existing_labels.get(occurrence)
            accent = self._annotation_accent(annotation)
            if prior is None:
                label = pg.TextItem(
                    str(annotation.get("label") or "intervention"),
                    color=(238, 238, 238), anchor=(0.0, 0.5),
                    border=pg.mkPen(accent), fill=pg.mkBrush(30, 30, 30, 210),
                )
                label.setFlag(
                    QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True
                )
                label.setZValue(11)
                self._plot_widget.addItem(label, ignoreBounds=True)
                offset = QtCore.QPointF()
            else:
                label, prior_anchor = prior
                offset = label.pos() - prior_anchor
                label.setText(str(annotation.get("label") or "intervention"))
                label.border = pg.mkPen(accent)
                label.update()
            label.setPos(anchor + offset)
            label.setOpacity(min(1.0, weight))
            rendered_labels[occurrence] = (label, anchor)
        for occurrence, (label, _anchor) in existing_labels.items():
            if occurrence not in rendered_labels:
                self._remove_annotation_item(label)
        self._annotation_labels = rendered_labels

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

    @staticmethod
    def _execution_region_chunks(
        segment_ids: np.ndarray,
        segment_keys: tuple[str, ...],
        discontinuity_ids: np.ndarray,
    ) -> tuple[tuple[int, int], ...]:
        count = int(segment_ids.size)
        if count == 0:
            return ()
        starts = [0]
        starts.extend(
            index
            for index in range(1, count)
            if (
                int(segment_ids[index]) != int(segment_ids[index - 1])
                or segment_keys[index] != segment_keys[index - 1]
                or int(discontinuity_ids[index]) != int(discontinuity_ids[index - 1])
            )
        )
        return tuple(
            (start, starts[index + 1] if index + 1 < len(starts) else count)
            for index, start in enumerate(starts)
        )

    def _clear_plot_items(self) -> None:
        self._plot_widget.clear()
        self._plot_items_by_key.clear()
        self._annotation_items.clear()
        self._annotation_labels.clear()
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
        for label, _anchor in self._annotation_labels.values():
            self._remove_annotation_item(label)
        self._annotation_labels.clear()

    @staticmethod
    def _scaled_optional(value: float | None, ratio: float) -> float | None:
        return None if value is None else float(value) * ratio

    def _apply_display_unit_change(self, axis: str, unit: str) -> None:
        render_input = self._render_input
        if render_input is None:
            return
        _old_x_unit, _old_y_unit, old_x_scale, old_y_scale = (
            self._display_unit_state(render_input)
        )
        if axis == "x" and self._has_time_x(render_input):
            if unit not in TIME_DISPLAY_UNIT_FACTORS:
                return
            self._x_time_unit = unit
        elif axis == "x" and render_input.x_is_concentration:
            if unit not in CONCENTRATION_DISPLAY_UNIT_FACTORS:
                return
            self._x_concentration_unit = unit
        elif axis == "y" and render_input.y_is_concentration:
            if unit not in CONCENTRATION_DISPLAY_UNIT_FACTORS:
                return
            self._y_concentration_unit = unit
        else:
            return
        _new_x_unit, _new_y_unit, new_x_scale, new_y_scale = (
            self._display_unit_state(render_input)
        )
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
        self._clear_annotation_items()
        self.apply_result_input(render_input)

    @QtCore.Slot(bool)
    def _set_coarse_sampling(self, enabled: bool) -> None:
        self._sampling_limit = 1000 if enabled else None
        if self._render_input is not None:
            self.apply_result_input(self._render_input)
        self._sync_plot_controls()

    def _capture_view_range(self) -> _AxisRangeSession:
        current_x, current_y = self._plot_widget.getViewBox().viewRange()
        return _AxisRangeSession(
            auto=False,
            x_min=self._view_to_raw_coordinate(current_x[0], log_enabled=self._presentation.log_x),
            x_max=self._view_to_raw_coordinate(current_x[1], log_enabled=self._presentation.log_x),
            y_min=self._view_to_raw_coordinate(current_y[0], log_enabled=self._presentation.log_y),
            y_max=self._view_to_raw_coordinate(current_y[1], log_enabled=self._presentation.log_y),
        )

    @QtCore.Slot(bool)
    def _set_auto_range(self, enabled: bool) -> None:
        self._axis_range = _AxisRangeSession() if enabled else self._capture_view_range()
        self._apply_axis_range_state(self._axis_range, self._presentation)
        self._project_manual_ranges()

    def _on_manual_navigation(self, _axes: object) -> None:
        self._plot_widget.getViewBox().disableAutoRange()
        self._axis_range = self._capture_view_range()
        self._project_manual_ranges()

    def _project_manual_ranges(self) -> None:
        state = self._axis_range
        self._axis_toolbar.set_manual_ranges(state.x_min, state.x_max, state.y_min, state.y_max)
        self._sync_plot_controls()

    @QtCore.Slot()
    def _apply_axis_range(self) -> None:
        presentation = self._presentation
        x_min, x_max, y_min, y_max = self._axis_toolbar.get_manual_ranges()
        auto = self._axis_range.auto
        if not auto:
            current_x, current_y = (
                self._plot_widget.getPlotItem().getViewBox().viewRange()
            )
            if x_min is None or x_max is None or x_max <= x_min:
                x_min = self._view_to_raw_coordinate(
                    current_x[0],
                    log_enabled=presentation.log_x,
                )
                x_max = self._view_to_raw_coordinate(
                    current_x[1],
                    log_enabled=presentation.log_x,
                )
            if y_min is None or y_max is None or y_max <= y_min:
                y_min = self._view_to_raw_coordinate(
                    current_y[0],
                    log_enabled=presentation.log_y,
                )
                y_max = self._view_to_raw_coordinate(
                    current_y[1],
                    log_enabled=presentation.log_y,
                )
        state = _AxisRangeSession(
            auto=auto,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        self._axis_range = state
        if not auto:
            self._axis_toolbar.set_manual_ranges(
                state.x_min,
                state.x_max,
                state.y_min,
                state.y_max,
            )
        self._apply_axis_range_state(state, presentation)

    def _apply_axis_range_state(
        self,
        state: _AxisRangeSession,
        presentation: _PlotPresentation,
    ) -> None:
        if state.auto:
            # With no data, autoRange pads the existing range on every redraw.
            if any(bound is not None for bound in self._plot_widget.getViewBox().childrenBounds()):
                self._plot_widget.autoRange()
            self._plot_widget.enableAutoRange()
            return
        self._plot_widget.getPlotItem().getViewBox().disableAutoRange()
        x_min = self._raw_to_view_coordinate(
            state.x_min,
            log_enabled=presentation.log_x,
        )
        x_max = self._raw_to_view_coordinate(
            state.x_max,
            log_enabled=presentation.log_x,
        )
        if x_min is not None and x_max is not None and x_max > x_min:
            self._plot_widget.setXRange(
                x_min,
                x_max,
                padding=0,
            )
        y_min = self._raw_to_view_coordinate(
            state.y_min,
            log_enabled=presentation.log_y,
        )
        y_max = self._raw_to_view_coordinate(
            state.y_max,
            log_enabled=presentation.log_y,
        )
        if y_min is not None and y_max is not None and y_max > y_min:
            self._plot_widget.setYRange(
                y_min,
                y_max,
                padding=0,
            )

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
        enabled: bool,
    ) -> None:
        # The WorkspaceTabRegistry is the sole accepted presentation owner.
        # Do not mutate local state or visible pyqtgraph items until it has
        # validated and stored this intent.
        self.plotPresentationIntentRequested.emit(str(option), bool(enabled))

    def _apply_plot_presentation(
        self,
        state: _PlotPresentation,
        *,
        axis_range: _AxisRangeSession | None = None,
    ) -> None:
        item = self._plot_widget.getPlotItem()
        item.setLogMode(x=state.log_x, y=state.log_y)
        item.invertX(state.x_inverted)
        item.invertY(state.y_inverted)
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


def _rendered_column_header(
    label: str,
    name: str,
    provenance: Mapping[str, object],
    unit: str | None,
) -> str:
    displayed_name = f"{name} ({unit})" if unit is not None else name
    header = f"{label}::{displayed_name}"
    identity = {
        key: str(provenance[key])
        for key in (
            "dataset_id",
            "raw_observation_id",
            "dataset_name",
            "raw_column",
            "model_target",
            "set_id",
        )
        if provenance.get(key) not in {None, ""}
    }
    if identity:
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        header = f"{header} [provenance={encoded}]"
    return encode_clipboard_text(header)


def _iter_rendered_point_tsv_chunks(
    roster: Sequence[_RenderedPointBlock],
    *,
    rows_per_chunk: int = 2048,
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
    block_specs: list[tuple[_RenderedPointBlock, tuple[str, ...], int, int, bool]] = []
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
        if block.kind == ResultLayerKind.EXPERIMENTAL.value:
            if any(chunk.source_row is None for chunk in chunks):
                raise ValueError("experimental rendered points lack source_row.")
            metadata_width = 1
        else:
            if any(
                chunk.segment_ids is None
                or chunk.segment_keys is None
                or chunk.discontinuity_ids is None
                for chunk in chunks
            ):
                raise ValueError("simulation rendered points lack topology.")
            metadata_width = 3
        rows = sum(int(chunk.x.size) for chunk in chunks)
        width = 1 + metadata_width + len(series_names)
        group = (
            block.kind,
            dict(block.series_provenance)[series_names[0]]["dataset_id"]
            if block.kind == ResultLayerKind.EXPERIMENTAL.value
            else block.block_id,
        )
        separator = bool(block_specs) and group != previous_group
        previous_group = group
        block_specs.append((block, series_names, rows, width, separator))
        planned_rows = max(planned_rows, rows)
        planned_width += width + separator
    planned_cells = (planned_rows + 1) * planned_width
    if planned_cells > _MAX_RENDERED_COPY_CELLS:
        raise ValueError(
            "Copy Plotted Data payload exceeds the bounded cell limit "
            f"({_MAX_RENDERED_COPY_CELLS:,})."
        )

    combined_headers: list[str] = []
    for block, series_names, _rows, _width, separator in block_specs:
        if separator:
            combined_headers.append("")
        chunks = tuple(chunk for chunk in block.chunks if chunk.x.size)
        headers = [
            _rendered_column_header(
                block.label,
                block.x_name,
                block.x_provenance,
                block.x_unit,
            )
        ]
        if block.kind == ResultLayerKind.EXPERIMENTAL.value:
            if any(chunk.source_row is None for chunk in chunks):
                raise ValueError("experimental rendered points lack source_row.")
            headers.append(encode_clipboard_text(f"{block.label} — source_row"))
        else:
            if any(
                chunk.segment_ids is None
                or chunk.segment_keys is None
                or chunk.discontinuity_ids is None
                for chunk in chunks
            ):
                raise ValueError("simulation rendered points lack topology.")
            headers.extend(
                (
                    encode_clipboard_text(f"{block.label} — segment_id"),
                    encode_clipboard_text(f"{block.label} — segment_key"),
                    encode_clipboard_text(
                        f"{block.label} — discontinuity_id"
                    ),
                )
            )
        provenance_by_name = dict(block.series_provenance)
        headers.extend(
            _rendered_column_header(
                block.label,
                name,
                provenance_by_name.get(name, MappingProxyType({})),
                block.y_unit,
            )
            for name in series_names
        )
        # Extend the one combined header in roster order.  Each block remains
        # an adjacent, independently identified column group.
        combined_headers.extend(headers)

    def iter_block_rows(
        block: _RenderedPointBlock,
        series_names: tuple[str, ...],
    ) -> Iterator[tuple[str, ...]]:
        for chunk in tuple(chunk for chunk in block.chunks if chunk.x.size):
            series_by_name = dict(chunk.series)
            for index in range(chunk.x.size):
                row: list[str] = [repr(float(chunk.x[index]))]
                if block.kind == ResultLayerKind.EXPERIMENTAL.value:
                    row.append(str(int(chunk.source_row[index])))
                else:
                    row.extend(
                        (
                            str(int(chunk.segment_ids[index])),
                            encode_clipboard_text(chunk.segment_keys[index]),
                            str(int(chunk.discontinuity_ids[index])),
                        )
                    )
                row.extend(
                    repr(float(series_by_name[name][index]))
                    for name in series_names
                )
                yield tuple(row)

    output_chunks: list[str] = []
    total_characters = 0
    output = io.StringIO()
    output.write(encode_tsv_row(combined_headers))
    rows_in_chunk = 1
    row_iterators = [
        iter_block_rows(block, series_names)
        for block, series_names, _rows, _width, _separator in block_specs
    ]
    for row_index in range(planned_rows):
        row: list[str] = []
        for iterator, (_block, _series_names, rows, width, separator) in zip(
            row_iterators, block_specs, strict=True
        ):
            if separator:
                row.append("")
            if row_index < rows:
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
                    "Copy Plotted Data payload exceeds the bounded output-size limit."
                )
            output_chunks.append(piece)
            output = io.StringIO()
            rows_in_chunk = 0
    piece = output.getvalue()
    total_characters += len(piece)
    if total_characters > _MAX_RENDERED_COPY_CHARACTERS:
        raise ValueError(
            "Copy Plotted Data payload exceeds the bounded output-size limit."
        )
    if piece:
        output_chunks.append(piece)
    return tuple(output_chunks)
