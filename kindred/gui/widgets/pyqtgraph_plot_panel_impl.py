"""PyQtGraph workspace surface for one immutable result projection."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtWidgets

from kindred.gui.widgets.axis_toolbar import AxisToolbar
from kindred.gui.plot_config import PlotPresentationIntent
from kindred.gui.widgets.parameter_statistics_table import ParameterStatisticsTable
from kindred.gui.widgets.species_statistics_table import SpeciesStatisticsTable
from kindred.gui.workspace_result_presenter import (
    ResultDataBlock,
    ResultLayerKind,
    WorkspacePlotLayer,
    WorkspaceResultPresenter,
    WorkspaceResultProjection,
)

logger = logging.getLogger(__name__)

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - plot_config rejects this before construction
    pg = None

__all__ = ["PyQtGraphPlotPanel"]

_FALLBACK_COLORS = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
)


def _fallback_color(name: str) -> str:
    return _FALLBACK_COLORS[
        sum((index + 1) * ord(char) for index, char in enumerate(str(name)))
        % len(_FALLBACK_COLORS)
    ]


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


class PyQtGraphPlotPanel(QtWidgets.QWidget):
    """Physical plot/analysis widget; semantic visibility stays in presentation."""

    referenceLayerVisibilityRequested = QtCore.Signal(bool)
    interventionAnnotationsVisibilityRequested = QtCore.Signal(bool)
    xSeriesRequested = QtCore.Signal(str)
    ySeriesRequested = QtCore.Signal(object)
    parametricModeRequested = QtCore.Signal(bool)
    plotPresentationIntentRequested = QtCore.Signal(object)
    renderFailed = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        embed_analysis_tabs: bool = True,
        workspace_splitter_object_name: str = "plotWorkspaceSplitter",
        enable_axis_inversion_actions: bool = True,
        enable_reference_layer_toggle_action: bool = True,
        enable_copy_visible_data_action: bool = True,
    ) -> None:
        if pg is None:
            raise ImportError("PyQtGraphPlotPanel requires pyqtgraph.")
        super().__init__(parent)
        self._enable_axis_inversion_actions = bool(enable_axis_inversion_actions)
        self._enable_reference_layer_toggle_action = bool(
            enable_reference_layer_toggle_action
        )
        self._enable_copy_visible_data_action = bool(
            enable_copy_visible_data_action
        )
        self._applied_projection: WorkspaceResultProjection | None = None
        self._rendered_blocks: tuple[ResultDataBlock, ...] = ()
        self._copy_all_enabled = False
        self._sampling_limit_by_tab: dict[str, int | None] = {}
        self._curve_items: list[object] = []
        self._annotation_items: list[object] = []
        self._manual_annotations_by_tab: dict[
            str, list[tuple[str, float, float]]
        ] = {}
        self._scalar_guides_by_tab: dict[str, list[str]] = {}
        self._interaction_mode_by_tab: dict[str, str] = {}
        self._presentation_by_tab: dict[str, _PlotPresentation] = {}
        self._axis_range_by_tab: dict[str, _AxisRangeSession] = {}
        self._dark_mode = False
        self._copy_chunks = None
        self._copy_output: io.StringIO | None = None
        self._copy_title = "Copy"
        self._copy_timer = QtCore.QTimer(self)
        self._copy_timer.setInterval(0)
        self._copy_timer.timeout.connect(self._continue_copy)

        self._plot_widget = pg.PlotWidget(self)
        self._plot_widget.setObjectName("plotViewport")
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self._legend = self._plot_widget.addLegend()
        self._plot_widget.getPlotItem().setMenuEnabled(False)
        self._plot_widget.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._plot_widget.customContextMenuRequested.connect(
            self._show_context_menu
        )

        self._axis_toolbar = AxisToolbar(self)
        self._axis_toolbar.xChanged.connect(self.xSeriesRequested.emit)
        self._axis_toolbar.ySelectionChanged.connect(self.ySeriesRequested.emit)
        self._axis_toolbar.parametricToggled.connect(
            self.parametricModeRequested.emit
        )
        self._axis_toolbar.optionsRequested.connect(self._on_axis_option)
        self._axis_toolbar.axisRangeChanged.connect(self._apply_axis_range)
        self._axis_toolbar.addGuideRequested.connect(self._add_scalar_guide)
        self._axis_toolbar.interactionModeRequested.connect(
            self._set_interaction_mode
        )
        self._axis_toolbar.presentationOptionRequested.connect(
            self._request_plot_presentation
        )

        plot_host = QtWidgets.QWidget(self)
        plot_layout = QtWidgets.QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(4)
        plot_layout.addWidget(self._axis_toolbar)
        plot_layout.addWidget(self._plot_widget, 1)

        self._parameter_table = ParameterStatisticsTable(self)
        self._species_table = SpeciesStatisticsTable(self)
        self._details_tabs = QtWidgets.QTabWidget(self)
        self._details_tabs.addTab(self._parameter_table, "Parameters")
        self._details_tabs.addTab(self._species_table, "Statistics")

        self._main_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Vertical,
            self,
        )
        self._main_splitter.setObjectName(str(workspace_splitter_object_name))
        self._main_splitter.addWidget(plot_host)
        self._main_splitter.addWidget(self._details_tabs)
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._main_splitter)
        if not embed_analysis_tabs:
            self._details_tabs.hide()

    def workspace_splitter(self) -> QtWidgets.QSplitter:
        return self._main_splitter

    def analysis_tabs_for_workspace(
        self,
        *,
        detached: bool,
    ) -> QtWidgets.QTabWidget:
        """Return the exact analysis surface in its requested composition."""

        if bool(detached) and self._details_tabs.parentWidget() is self._main_splitter:
            self._details_tabs.setParent(None)
            self._details_tabs.show()
        return self._details_tabs

    @property
    def applied_projection(self) -> WorkspaceResultProjection | None:
        return self._applied_projection

    def apply_workspace_result_projection(
        self,
        projection: WorkspaceResultProjection,
    ) -> None:
        """Apply one projection; failure invalidates this panel's projection."""

        if not isinstance(projection, WorkspaceResultProjection):
            raise TypeError(
                "apply_workspace_result_projection requires "
                "WorkspaceResultProjection."
            )
        try:
            rendered = self._render_projection(projection)
        except Exception as exc:
            logger.exception("Workspace projection render failed")
            # A failed projection is ordinary unavailability for this panel:
            # never retain a stale scene or export authority.
            try:
                self._clear_plot_items()
            except Exception:
                logger.exception("Failed to clear rejected plot projection")
            self._applied_projection = None
            self._rendered_blocks = ()
            self._copy_all_enabled = False
            self.renderFailed.emit(str(exc))
            raise
        self._applied_projection = projection
        self._rendered_blocks = rendered
        self._copy_all_enabled = bool(projection.copy_all_blocks)

    def set_persisted_plot_state(
        self,
        tab_id: str,
        *,
        log_x: bool,
        log_y: bool,
        legend_visible: bool,
        x_inverted: bool,
        y_inverted: bool,
    ) -> None:
        """Apply one Integration-owned, per-tab persisted Plot projection."""

        target = str(tab_id or "").strip()
        if not target:
            raise ValueError("tab_id must be nonblank.")
        state = _PlotPresentation(
            log_x=bool(log_x),
            log_y=bool(log_y),
            legend_visible=bool(legend_visible),
            x_inverted=bool(x_inverted),
            y_inverted=bool(y_inverted),
        )
        self._presentation_by_tab[target] = state
        if (
            self._applied_projection is not None
            and self._applied_projection.tab_id == target
        ):
            self._apply_plot_presentation(state)

    def drop_tab_session_state(self, tab_id: str) -> None:
        target = str(tab_id)
        self._manual_annotations_by_tab.pop(target, None)
        self._scalar_guides_by_tab.pop(target, None)
        self._interaction_mode_by_tab.pop(target, None)
        self._presentation_by_tab.pop(target, None)
        self._sampling_limit_by_tab.pop(target, None)
        self._axis_range_by_tab.pop(target, None)

    def reset_plot_session_state(self) -> None:
        """Project replacement/reopen resets non-archived Plot session truth."""

        self._manual_annotations_by_tab.clear()
        self._scalar_guides_by_tab.clear()
        self._interaction_mode_by_tab.clear()
        self._presentation_by_tab.clear()
        self._sampling_limit_by_tab.clear()
        self._axis_range_by_tab.clear()
        self._axis_toolbar.set_interaction_mode("pan")
        self._axis_toolbar.set_coarse_sampling(False)
        self._axis_toolbar.set_manual_ranges(None, None, None, None)
        self._axis_toolbar.set_auto_range(True)
        self._apply_interaction_mode("pan")

    def rendered_point_blocks(self) -> tuple[ResultDataBlock, ...]:
        return self._rendered_blocks

    def clear_workspace_result_projection(self) -> None:
        """Clear rejected public truth without affecting other tab-local state."""

        try:
            self._clear_plot_items()
            self._parameter_table.setRowCount(0)
            self._species_table.setRowCount(0)
        finally:
            self._applied_projection = None
            self._rendered_blocks = ()
            self._copy_all_enabled = False

    def set_copy_all_enabled(self, enabled: bool) -> None:
        """Receive the active workspace's physical Copy capability."""

        self._copy_all_enabled = bool(enabled)

    def _render_projection(
        self,
        projection: WorkspaceResultProjection,
    ) -> tuple[ResultDataBlock, ...]:
        self._clear_plot_items()
        block_by_id = {
            block.block_id: block
            for block in projection.copy_all_blocks
        }
        rendered_blocks: list[ResultDataBlock] = []
        available_y = set(projection.available_y_series)
        x_candidates = set(available_y)
        if projection.selected_x_series != "t":
            x_candidates.add(projection.selected_x_series)
        for layer in projection.plot_layers:
            x_candidates.update(name for name, _values in layer.series)
            block = block_by_id.get(layer.layer_id)
            if block is None:
                continue
            rendered = self._render_layer(
                layer,
                block,
                sampling_limit=self._sampling_limit_by_tab.get(
                    projection.tab_id
                ),
            )
            if rendered is not None:
                rendered_blocks.append(rendered)

        self._axis_toolbar.set_x_candidates(
            ["t", *sorted(x_candidates)],
            default=projection.selected_x_series,
        )
        self._axis_toolbar.set_y_candidates(
            [
                (
                    name,
                    projection.selected_y_uses_default_all
                    or name in projection.selected_y_series,
                )
                for name in sorted(available_y)
            ]
        )
        self._axis_toolbar.set_parametric(projection.parametric_mode)
        self._axis_toolbar.set_coarse_sampling(
            self._sampling_limit_by_tab.get(projection.tab_id) is not None
        )
        interaction = self._interaction_mode_by_tab.setdefault(
            projection.tab_id,
            "pan",
        )
        self._axis_toolbar.set_interaction_mode(interaction)
        self._apply_interaction_mode(interaction)
        self._render_annotations(projection)
        self._render_manual_annotations(projection.tab_id)
        self._render_scalar_guides(projection)
        self._render_analysis(projection)
        axis_range = self._axis_range_by_tab.setdefault(
            projection.tab_id,
            _AxisRangeSession(),
        )
        self._axis_toolbar.set_manual_ranges(
            axis_range.x_min,
            axis_range.x_max,
            axis_range.y_min,
            axis_range.y_max,
        )
        self._axis_toolbar.set_auto_range(axis_range.auto)
        self._apply_axis_range_state(axis_range)
        self._apply_plot_presentation(
            self._presentation_by_tab.get(
                projection.tab_id,
                _PlotPresentation(),
            )
        )
        return tuple(rendered_blocks)

    def _render_layer(
        self,
        layer: WorkspacePlotLayer,
        block: ResultDataBlock,
        *,
        sampling_limit: int | None,
    ) -> ResultDataBlock | None:
        sample = self._sample_indices(block.t.size, sampling_limit)
        if layer.kind is not block.kind:
            raise ValueError("plot layer and data block kinds must match.")
        if block.kind is not ResultLayerKind.EXPERIMENTAL:
            region_endpoints = self._execution_region_endpoints(block)
            sample = np.unique(
                np.concatenate(
                    (
                        sample,
                        region_endpoints,
                    )
                )
            )
        x = block.t[sample]
        if block.kind is ResultLayerKind.EXPERIMENTAL:
            if block.source_row is None:
                raise ValueError("experimental plot block lacks source_row.")
            segment_ids = None
            segment_keys = None
            discontinuity_ids = None
            source_row = block.source_row[sample]
            chunks = ((0, x.size),)
        else:
            if (
                block.segment_ids is None
                or block.segment_keys is None
                or block.discontinuity_ids is None
            ):
                raise ValueError("simulation plot block lacks executor identity.")
            segment_ids = block.segment_ids[sample]
            segment_keys = tuple(
                block.segment_keys[int(index)] for index in sample
            )
            discontinuity_ids = block.discontinuity_ids[sample]
            source_row = None
            chunks = self._execution_region_chunks(
                segment_ids,
                segment_keys,
                discontinuity_ids,
            )
        colors = dict(layer.color_by_series)
        rendered_series: list[tuple[str, np.ndarray]] = []
        for series_name, values in block.series:
            y = values[sample]
            color = colors.get(series_name, _fallback_color(series_name))
            name = f"{layer.label}: {series_name}"
            for chunk_index, (start, end) in enumerate(chunks):
                singleton = end - start == 1
                if layer.markers or singleton:
                    item = self._plot_widget.plot(
                        x[start:end],
                        y[start:end],
                        pen=None,
                        symbol="o",
                        symbolSize=(
                            layer.point_size if layer.markers else 5.0
                        ),
                        symbolBrush=pg.mkBrush(color),
                        symbolPen=pg.mkPen(color),
                        name=name if chunk_index == 0 else None,
                    )
                else:
                    pen = pg.mkPen(
                        color=color,
                        width=2,
                        style=(
                            QtCore.Qt.PenStyle.DashLine
                            if layer.dashed
                            else QtCore.Qt.PenStyle.SolidLine
                        ),
                    )
                    item = self._plot_widget.plot(
                        x[start:end],
                        y[start:end],
                        pen=pen,
                        name=name if chunk_index == 0 else None,
                    )
                item.setOpacity(layer.opacity)
                self._curve_items.append(item)
            rendered_series.append((series_name, y))
        if not rendered_series:
            return None
        return ResultDataBlock(
            block_id=block.block_id,
            label=block.label,
            kind=block.kind,
            t=x,
            series=tuple(rendered_series),
            segment_ids=segment_ids,
            segment_keys=segment_keys,
            discontinuity_ids=discontinuity_ids,
            source_row=source_row,
            x_name=block.x_name,
            series_provenance=block.series_provenance,
            x_provenance=block.x_provenance,
        )

    def _render_analysis(self, projection: WorkspaceResultProjection) -> None:
        self._parameter_table.update_parameters(
            {
                name: (value, unit)
                for name, value, unit in projection.parameters
            }
        )
        block = projection.primary_statistics
        if block is None:
            self._species_table.setRowCount(0)
            return
        residual = dict(projection.scalar_values).get("residual_mse")
        self._species_table.update_results(
            block.t,
            dict(block.series),
            residual_mse=residual,
        )

    def _render_annotations(self, projection: WorkspaceResultProjection) -> None:
        if (
            not projection.intervention_annotations_visible
            or (
                projection.parametric_mode
                and projection.selected_x_series != "t"
            )
        ):
            return
        for annotation in projection.intervention_annotations:
            if "start" in annotation and "end" in annotation:
                region = pg.LinearRegionItem(
                    values=(
                        float(annotation["start"]),
                        float(annotation["end"]),
                    ),
                    movable=False,
                    brush=pg.mkBrush(255, 190, 80, 35),
                    pen=pg.mkPen(220, 150, 40, 100),
                )
                self._plot_widget.addItem(region)
                self._annotation_items.append(region)
                continue
            if "time" not in annotation:
                continue
            line = pg.InfiniteLine(
                pos=float(annotation["time"]),
                angle=90,
                movable=False,
                pen=pg.mkPen("#b36b00", width=1, style=QtCore.Qt.PenStyle.DashLine),
                label=str(annotation.get("label") or "intervention"),
            )
            self._plot_widget.addItem(line)
            self._annotation_items.append(line)

    def _render_manual_annotations(self, tab_id: str) -> None:
        foreground = "#f3f3f3" if self._dark_mode else "#202020"
        background = "#303030cc" if self._dark_mode else "#ffffffe6"
        border = "#bbbbbb" if self._dark_mode else "#555555"
        for text, x, y in self._manual_annotations_by_tab.get(tab_id, ()):
            item = pg.TextItem(
                text=str(text),
                color=foreground,
                fill=pg.mkBrush(background),
                border=pg.mkPen(border),
                anchor=(0.5, 0.5),
            )
            item.setPos(float(x), float(y))
            self._plot_widget.addItem(item)
            self._annotation_items.append(item)

    def _render_scalar_guides(
        self,
        projection: WorkspaceResultProjection,
    ) -> None:
        scalars = dict(projection.scalar_values)
        for name in self._scalar_guides_by_tab.get(projection.tab_id, ()):
            if name not in scalars:
                continue
            line = pg.InfiniteLine(
                pos=float(scalars[name]),
                angle=0,
                movable=False,
                pen=pg.mkPen(
                    "#777777",
                    width=1,
                    style=QtCore.Qt.PenStyle.DotLine,
                ),
                label=name,
            )
            self._plot_widget.addItem(line)
            self._annotation_items.append(line)

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
                    int(block.segment_ids[index])
                    != int(block.segment_ids[index - 1])
                    or block.segment_keys[index]
                    != block.segment_keys[index - 1]
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
                or int(discontinuity_ids[index])
                != int(discontinuity_ids[index - 1])
            )
        )
        return tuple(
            (start, starts[index + 1] if index + 1 < len(starts) else count)
            for index, start in enumerate(starts)
        )

    @QtCore.Slot(object)
    def _add_scalar_guide(self, requested_name: object) -> None:
        projection = self._applied_projection
        scalars = dict(projection.scalar_values) if projection is not None else {}
        if not scalars:
            return
        name = str(requested_name or "")
        if name not in scalars:
            names = sorted(scalars)
            name, accepted = QtWidgets.QInputDialog.getItem(
                self,
                "Add Guide from Scalar",
                "Scalar:",
                names,
                0,
                False,
            )
            if not accepted:
                return
        guides = self._scalar_guides_by_tab.setdefault(
            projection.tab_id,
            [],
        )
        if name not in guides:
            guides.append(name)
        self.apply_workspace_result_projection(projection)

    def _clear_plot_items(self) -> None:
        self._plot_widget.clear()
        self._curve_items.clear()
        self._annotation_items.clear()
        self._legend = self._plot_widget.addLegend()

    @QtCore.Slot(str, object)
    def _on_axis_option(self, action: str, value: object) -> None:
        normalized = str(action)
        projection = self._applied_projection
        if normalized == "clear_scalar_guides":
            if projection is None:
                return
            self._scalar_guides_by_tab.pop(projection.tab_id, None)
            self.apply_workspace_result_projection(projection)
            return
        if normalized != "sampling":
            return
        if projection is None:
            return
        tab_id = projection.tab_id
        previous_limit = self._sampling_limit_by_tab.get(tab_id)
        self._sampling_limit_by_tab[tab_id] = (
            1000 if str(value) == "coarse" else None
        )
        try:
            self.apply_workspace_result_projection(projection)
        except Exception:
            self._sampling_limit_by_tab[tab_id] = previous_limit
            raise

    @QtCore.Slot()
    def _apply_axis_range(self) -> None:
        projection = self._applied_projection
        if projection is None:
            return
        x_min, x_max, y_min, y_max = self._axis_toolbar.get_manual_ranges()
        auto = self._axis_toolbar.is_auto_range()
        if not auto:
            current_x, current_y = (
                self._plot_widget.getPlotItem().getViewBox().viewRange()
            )
            if (
                x_min is None
                or x_max is None
                or x_max <= x_min
            ):
                x_min, x_max = float(current_x[0]), float(current_x[1])
            if (
                y_min is None
                or y_max is None
                or y_max <= y_min
            ):
                y_min, y_max = float(current_y[0]), float(current_y[1])
        state = _AxisRangeSession(
            auto=auto,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        self._axis_range_by_tab[projection.tab_id] = state
        if not auto:
            self._axis_toolbar.set_manual_ranges(
                state.x_min,
                state.x_max,
                state.y_min,
                state.y_max,
            )
        self._apply_axis_range_state(state)

    def _apply_axis_range_state(self, state: _AxisRangeSession) -> None:
        if state.auto:
            self._plot_widget.autoRange()
            self._plot_widget.enableAutoRange()
            return
        self._plot_widget.getPlotItem().getViewBox().disableAutoRange()
        if (
            state.x_min is not None
            and state.x_max is not None
            and state.x_max > state.x_min
        ):
            self._plot_widget.setXRange(
                state.x_min,
                state.x_max,
                padding=0,
            )
        if (
            state.y_min is not None
            and state.y_max is not None
            and state.y_max > state.y_min
        ):
            self._plot_widget.setYRange(
                state.y_min,
                state.y_max,
                padding=0,
            )

    @QtCore.Slot(QtCore.QPoint)
    def _show_context_menu(self, position: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        projection = self._applied_projection
        if self._enable_copy_visible_data_action:
            copy_all = menu.addAction("Copy All")
            copy_all.setEnabled(
                bool(
                    self._copy_all_enabled
                    and projection is not None
                    and projection.copy_all_blocks
                )
            )
            copy_all.triggered.connect(self._copy_all_to_clipboard)
            copy_rendered = menu.addAction("Copy Rendered Points")
            copy_rendered.setEnabled(bool(self._rendered_blocks))
            copy_rendered.triggered.connect(self._copy_rendered_to_clipboard)
            menu.addSeparator()
        if self._enable_reference_layer_toggle_action:
            reference = menu.addAction("Show Reference")
            reference.setCheckable(True)
            reference.setChecked(
                bool(projection and projection.reference_lines_visible)
            )
            reference.toggled.connect(self.referenceLayerVisibilityRequested)
        annotations = menu.addAction("Show Intervention Annotations")
        annotations.setCheckable(True)
        annotations.setChecked(
            bool(projection and projection.intervention_annotations_visible)
        )
        annotations.toggled.connect(
            self.interventionAnnotationsVisibilityRequested
        )
        menu.addSeparator()
        add_annotation = menu.addAction("Add Annotation…")
        add_annotation.setEnabled(projection is not None)
        add_annotation.triggered.connect(self._add_manual_annotation)
        clear_annotations = menu.addAction("Clear All Annotations")
        clear_annotations.setEnabled(
            bool(
                projection
                and self._manual_annotations_by_tab.get(projection.tab_id)
            )
        )
        clear_annotations.triggered.connect(self._clear_manual_annotations)
        state = (
            self._presentation_by_tab.get(projection.tab_id, _PlotPresentation())
            if projection is not None
            else _PlotPresentation()
        )
        menu.addSeparator()
        for text, option, checked in (
            ("Log X", "log_x", state.log_x),
            ("Log Y", "log_y", state.log_y),
            ("Show Legend", "legend_visible", state.legend_visible),
            ("Invert X Axis", "x_inverted", state.x_inverted),
            ("Invert Y Axis", "y_inverted", state.y_inverted),
        ):
            if (
                option in {"x_inverted", "y_inverted"}
                and not self._enable_axis_inversion_actions
            ):
                continue
            action = menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(bool(checked))
            action.toggled.connect(
                lambda enabled, key=option: self._request_plot_presentation(
                    key,
                    bool(enabled),
                )
            )
        menu.exec(self._plot_widget.mapToGlobal(position))

    @QtCore.Slot()
    def _add_manual_annotation(self) -> None:
        projection = self._applied_projection
        if projection is None:
            return
        text, accepted = QtWidgets.QInputDialog.getText(
            self,
            "Add Plot Annotation",
            "Text:",
        )
        annotation_text = str(text)
        if not accepted or not annotation_text.strip():
            return
        x_range, y_range = self._plot_widget.getPlotItem().viewRange()
        x = (float(x_range[0]) + float(x_range[1])) / 2.0
        y = (float(y_range[0]) + float(y_range[1])) / 2.0
        annotations = self._manual_annotations_by_tab.setdefault(
            projection.tab_id,
            [],
        )
        annotations.append((annotation_text, x, y))
        self.apply_workspace_result_projection(projection)

    @QtCore.Slot()
    def _clear_manual_annotations(self) -> None:
        projection = self._applied_projection
        if projection is None:
            return
        annotations = self._manual_annotations_by_tab.get(projection.tab_id)
        if not annotations:
            return
        annotations.clear()
        self.apply_workspace_result_projection(projection)

    @QtCore.Slot(str)
    def _set_interaction_mode(self, mode: str) -> None:
        normalized = str(mode or "").strip().lower()
        if normalized not in {"pan", "rectangle_zoom"}:
            raise ValueError(
                "Plot interaction mode must be Pan or Rectangle Zoom."
            )
        projection = self._applied_projection
        if projection is not None:
            self._interaction_mode_by_tab[projection.tab_id] = normalized
        self._axis_toolbar.set_interaction_mode(normalized)
        self._apply_interaction_mode(normalized)

    def _apply_interaction_mode(self, mode: str) -> None:
        view_box = self._plot_widget.getPlotItem().getViewBox()
        view_box.setMouseMode(
            view_box.PanMode
            if str(mode) == "pan"
            else view_box.RectMode
        )

    def _request_plot_presentation(
        self,
        option: str,
        enabled: bool,
    ) -> None:
        intent = PlotPresentationIntent(str(option), bool(enabled))
        projection = self._applied_projection
        if projection is not None:
            state = self._presentation_by_tab.get(
                projection.tab_id,
                _PlotPresentation(),
            )
            values = {
                "log_x": state.log_x,
                "log_y": state.log_y,
                "legend_visible": state.legend_visible,
                "x_inverted": state.x_inverted,
                "y_inverted": state.y_inverted,
            }
            values[intent.option] = intent.enabled
            state = _PlotPresentation(**values)
            self._presentation_by_tab[projection.tab_id] = state
            self._apply_plot_presentation(state)
        self.plotPresentationIntentRequested.emit(intent)

    def _apply_plot_presentation(self, state: _PlotPresentation) -> None:
        item = self._plot_widget.getPlotItem()
        item.setLogMode(x=state.log_x, y=state.log_y)
        item.invertX(state.x_inverted)
        item.invertY(state.y_inverted)
        self._axis_toolbar.set_plot_presentation_state(
            log_x=state.log_x,
            log_y=state.log_y,
            legend_visible=state.legend_visible,
            x_inverted=state.x_inverted,
            y_inverted=state.y_inverted,
        )
        if self._legend is not None:
            self._legend.setVisible(state.legend_visible)

    @QtCore.Slot()
    def _copy_all_to_clipboard(self) -> None:
        projection = self._applied_projection
        if (
            not self._copy_all_enabled
            or projection is None
            or not projection.copy_all_blocks
        ):
            QtWidgets.QMessageBox.critical(
                self,
                "Copy All",
                "No current result data is available to copy.",
            )
            return
        self._start_responsive_copy(
            projection.copy_all_blocks,
            title="Copy All",
        )

    @QtCore.Slot()
    def _copy_rendered_to_clipboard(self) -> None:
        self._start_responsive_copy(
            self._rendered_blocks,
            title="Copy Rendered Points",
        )

    def _start_responsive_copy(
        self,
        blocks: tuple[ResultDataBlock, ...],
        *,
        title: str,
    ) -> None:
        if self._copy_timer.isActive():
            return
        try:
            self._copy_chunks = iter(
                WorkspaceResultPresenter.iter_blocks_tsv_chunks(
                    tuple(blocks),
                )
            )
            self._copy_output = io.StringIO()
            self._copy_title = str(title)
            self._copy_timer.start()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                str(title),
                str(exc),
            )

    @QtCore.Slot()
    def _continue_copy(self) -> None:
        chunks = self._copy_chunks
        output = self._copy_output
        if chunks is None or output is None:
            self._copy_timer.stop()
            return
        try:
            output.write(next(chunks))
        except StopIteration:
            self._copy_timer.stop()
            text = output.getvalue()
            self._copy_chunks = None
            self._copy_output = None
            QtWidgets.QApplication.clipboard().setText(text)
        except Exception as exc:
            self._copy_timer.stop()
            self._copy_chunks = None
            self._copy_output = None
            QtWidgets.QMessageBox.critical(
                self,
                self._copy_title,
                str(exc),
            )

    def set_theme(self, *, dark_mode: bool) -> None:
        dark = bool(dark_mode)
        self._dark_mode = dark
        self._plot_widget.setBackground("#1e1e1e" if dark else "w")
        foreground = "#dddddd" if dark else "#222222"
        for axis_name in ("left", "bottom"):
            axis = self._plot_widget.getPlotItem().getAxis(axis_name)
            axis.setPen(foreground)
            axis.setTextPen(foreground)
        if self._applied_projection is not None:
            self.apply_workspace_result_projection(self._applied_projection)
