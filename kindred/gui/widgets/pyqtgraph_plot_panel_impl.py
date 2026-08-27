"""PyQtGraph workspace surface for one immutable result input."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.project_presentation_state import (
    SimulationSeriesAxisSelection,
)
from kindred.core.result_repository import SimulationResultValue
from kindred.gui.interactive_preview import InteractivePreviewRequest
from kindred.gui.plot_config import PlotPresentationIntent, require_pyqtgraph
from kindred.gui.widgets.axis_toolbar import AxisToolbar
from kindred.gui.widgets.interactive_plot_transition import (
    InteractiveDisplayController,
    InteractiveDisplayState,
)
from kindred.gui.widgets.parameter_statistics_table import ParameterStatisticsTable
from kindred.gui.widgets.species_statistics_table import SpeciesStatisticsTable
from kindred.gui.workspace_result_presenter import (
    ResultDataBlock,
    ResultRenderInput,
    ResultLayerKind,
    encode_clipboard_text,
    encode_tsv_row,
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
    copyAllRequested = QtCore.Signal()
    xSeriesRequested = QtCore.Signal(object)
    ySeriesRequested = QtCore.Signal(object)
    plotPresentationIntentRequested = QtCore.Signal(object)
    resultDiagnostic = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sampling_limit: int | None = None
        self._plot_items_by_key: dict[tuple[object, ...], object] = {}
        self._annotation_items: list[object] = []
        self._interaction_mode = "pan"
        self._presentation = _PlotPresentation()
        self._axis_range = _AxisRangeSession()
        self._render_input: ResultRenderInput | None = None
        self._copy_all_available = False
        self._rendered_point_roster: tuple[_RenderedPointBlock, ...] = ()
        self._settling_static_input = False
        self._interactive_timer = QtCore.QTimer(self)
        self._interactive_timer.setSingleShot(True)
        self._interactive_timer.timeout.connect(self._advance_interactive_display)
        self._interactive_items: dict[tuple[object, ...], Any] = {}
        self._interactive_locked_view: (
            tuple[tuple[float, float], tuple[float, float]] | None
        ) = None
        self._interactive_display = InteractiveDisplayController(
            render=self._render_interactive_display,
            request_tick=self._request_interactive_tick,
            cancel_tick=self._interactive_timer.stop,
        )

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
            self._show_plot_context_menu
        )
        self._axis_toolbar = AxisToolbar(self)
        self._axis_toolbar.xChanged.connect(self.xSeriesRequested.emit)
        self._axis_toolbar.ySelectionChanged.connect(self.ySeriesRequested.emit)
        self._axis_toolbar.axisRangeChanged.connect(self._apply_axis_range)
        self._axis_toolbar.interactionModeRequested.connect(self._set_interaction_mode)
        self._axis_toolbar.optionsRequested.connect(self._on_axis_option)
        self._axis_toolbar.presentationOptionRequested.connect(
            self._request_plot_presentation
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
        self._completion_warning_badge.setVisible(False)

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

    def apply_result_input(self, render_input: ResultRenderInput) -> bool:
        """Retain and draw one immutable result input."""

        locked_view = self._interactive_locked_view
        self._rendered_point_roster = ()
        self._settling_static_input = True
        try:
            self._interactive_display.prepare_result_input(render_input)
            self._render_input = render_input
            self._render_result_scene(render_input)
            self._interactive_display.commit_result_input(render_input)
            if self._interactive_display.has_visible_frame and locked_view is not None:
                self._restore_interactive_view(locked_view)
            elif not self._interactive_display.has_renderable_frame:
                self._end_interactive_range_session()
            if render_input.unavailable_x_diagnostic is not None:
                self.resultDiagnostic.emit(render_input.unavailable_x_diagnostic)
            return True
        except Exception:
            # A failed physical apply must not leave the prior scene, tables,
            # warnings, or rendered-copy roster attached to a new semantic
            # input.  MainWindow also clears its semantic cache at this seam.
            self.clear_result_input()
            raise
        finally:
            self._settling_static_input = False

    def apply_interactive_preview_result(
        self,
        tab_id: str,
        set_id: str,
        generation: int,
        request: InteractivePreviewRequest,
        result: SimulationResultValue,
    ) -> bool:
        """Admit one transient result against the retained immutable input."""

        render_input = self._render_input
        if render_input is None or render_input.tab_id != str(tab_id):
            return False
        return self._interactive_display.offer_result(
            result,
            set_id=str(set_id),
            generation=int(generation),
            target_set_ids=tuple(request.target_set_ids),
            render_input=render_input,
        )

    def set_copy_all_available(self, available: bool) -> None:
        """Project semantic Copy All eligibility into the physical menu."""

        self._copy_all_available = bool(available)

    def _show_plot_context_menu(self, position: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        copy_all = menu.addAction("Copy All")
        copy_all.setEnabled(self._copy_all_available)
        copy_all.triggered.connect(self.copyAllRequested.emit)
        menu.addSeparator()
        copy_rendered = menu.addAction("Copy Rendered Points")
        copy_rendered.setEnabled(
            any(chunk.x.size for block in self._rendered_point_roster for chunk in block.chunks)
        )
        copy_rendered.triggered.connect(self._copy_rendered_points_to_clipboard)
        menu.addSeparator()
        reference = menu.addAction("Show Reference")
        reference.setCheckable(True)
        reference.setChecked(
            bool(self._render_input and self._render_input.reference_lines_visible)
        )
        reference.toggled.connect(self.referenceLayerVisibilityRequested)
        annotations = menu.addAction("Show Intervention Annotations")
        annotations.setCheckable(True)
        annotations.setChecked(
            bool(
                self._render_input
                and self._render_input.intervention_annotations_visible
            )
        )
        annotations.toggled.connect(self.interventionAnnotationsVisibilityRequested)
        menu.exec(self._plot_widget.mapToGlobal(position))

    @QtCore.Slot()
    def _copy_rendered_points_to_clipboard(self) -> None:
        roster = self._rendered_point_roster
        if not any(chunk.x.size for block in roster for chunk in block.chunks):
            raise ValueError("No rendered points are available to copy.")
        output = io.StringIO()
        output.writelines(_iter_rendered_point_tsv_chunks(roster))
        QtWidgets.QApplication.clipboard().setText(output.getvalue())

    def retire_interactive_preview_targets(
        self,
        target_set_ids: tuple[str, ...],
        *,
        generation: int | None = None,
    ) -> bool:
        """Retire selected transient targets at an admitted generation."""

        retired = self._interactive_display.retire_targets(
            tuple(str(value) for value in target_set_ids),
            generation=generation,
        )
        if retired or generation is None:
            retired = self._retire_static_preview_targets(target_set_ids) or retired
        return retired

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
        return retired

    def stop_interactive_display(self) -> bool:
        """Cancel visual work and freeze the last physically stable display."""

        if (
            not self._interactive_display.has_visible_frame
            and not self._interactive_display.has_pending_batch
        ):
            return False
        self._interactive_display.stop()
        self._end_interactive_range_session()
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
        if self._interactive_locked_view is not None:
            self._restore_interactive_view(self._interactive_locked_view)
        if self._render_input is not None:
            self._clear_annotation_items()
            self._render_annotations(self._render_input, state)

    def apply_application_plot_preferences(
        self,
        *,
        color_for_species: Callable[[str, Sequence[str]], object],
        known_species: Sequence[str],
        experimental_point_size: float,
        experimental_point_opacity: float,
    ) -> None:
        """Apply changed application styling to already-rendered plot items."""

        render_input = self._render_input
        if render_input is None:
            return
        point_size = float(experimental_point_size)
        point_opacity = float(experimental_point_opacity)
        known = tuple(known_species)
        for block in render_input.blocks:
            if not block.visible:
                continue
            _sample, chunks = self._sample_and_chunks(block, self._sampling_limit)
            role = (block.kind.value, block.block_id)
            colors: dict[str, object] = {}
            for series_name, selection in block.series_axes:
                color_name = (
                    selection.series_name
                    if isinstance(selection, SimulationSeriesAxisSelection)
                    else selection.raw_observation_id
                )
                colors[series_name] = color_for_species(color_name, known)
            opacity = (
                point_opacity
                if block.kind is ResultLayerKind.EXPERIMENTAL
                else block.opacity
            )
            size = point_size if block.kind is ResultLayerKind.EXPERIMENTAL else block.point_size
            for series_name, _values in block.series:
                color = colors[series_name]
                for chunk_index, (start, end) in enumerate(chunks):
                    marker = bool(block.markers or end - start == 1)
                    item_key = (
                        role,
                        series_name,
                        chunk_index,
                        "marker" if marker else "line",
                    )
                    item = self._plot_items_by_key.get(item_key)
                    if item is None:
                        continue
                    if marker:
                        item.setSymbolBrush(pg.mkBrush(color))
                        item.setSymbolPen(pg.mkPen(color))
                        item.setSymbolSize(size if block.markers else 5.0)
                    else:
                        item.setPen(
                            pg.mkPen(
                                color=color,
                                width=2,
                                style=(
                                    QtCore.Qt.PenStyle.DashLine
                                    if block.dashed
                                    else QtCore.Qt.PenStyle.SolidLine
                                ),
                            )
                        )
                    item.setOpacity(opacity)

    def drop_tab_session_state(self) -> None:
        self.dispose_project_content()
        self._interaction_mode = "pan"
        self._presentation = _PlotPresentation()
        self._sampling_limit = None
        self._axis_range = _AxisRangeSession()

    def reset_plot_session_state(self) -> None:
        """Project replacement/reopen resets non-archived Plot session truth."""

        self.dispose_project_content()
        self._end_interactive_range_session()
        self._interaction_mode = "pan"
        self._presentation = _PlotPresentation()
        self._sampling_limit = None
        self._axis_range = _AxisRangeSession()
        self._axis_toolbar.set_interaction_mode("pan")
        self._axis_toolbar.set_coarse_sampling(False)
        self._axis_toolbar.set_manual_ranges(None, None, None, None)
        self._axis_toolbar.set_auto_range(True)
        self._apply_interaction_mode("pan")

    def dispose_project_content(self) -> None:
        """Stop plot-owned asynchronous work before project content teardown."""

        self._render_input = None
        self._copy_all_available = False
        self._rendered_point_roster = ()
        self._interactive_display.clear(reset_currentness=True)

    def clear_result_input(self) -> None:
        """Clear the current result scene and its transient display."""

        self._render_input = None
        self._copy_all_available = False
        self._rendered_point_roster = ()
        self.discard_interactive_display()
        self._end_interactive_range_session()
        self._clear_plot_items()
        self._parameter_table.setRowCount(0)
        self._species_table.setRowCount(0)
        self._clear_completion_warning()
    def _request_interactive_tick(self, delay_seconds: float) -> None:
        milliseconds = max(0, int(np.ceil(float(delay_seconds) * 1000.0)))
        self._interactive_timer.start(milliseconds)

    @QtCore.Slot()
    def _advance_interactive_display(self) -> None:
        had_rendered_frame = self._interactive_display.has_renderable_frame
        self._interactive_display.tick()
        if (
            not had_rendered_frame
            and self._interactive_display.has_renderable_frame
        ):
            self._begin_interactive_range_session()

    def _begin_interactive_range_session(self) -> None:
        view_box = self._plot_widget.getPlotItem().getViewBox()
        x_range, y_range = view_box.viewRange()
        self._interactive_locked_view = (
            (float(x_range[0]), float(x_range[1])),
            (float(y_range[0]), float(y_range[1])),
        )
        self._axis_toolbar.set_interactive_gesture_active(True)
        self._restore_interactive_view(self._interactive_locked_view)

    def _end_interactive_range_session(self) -> None:
        self._interactive_locked_view = None
        self._axis_toolbar.set_interactive_gesture_active(False)

    def _restore_interactive_view(
        self,
        view: tuple[tuple[float, float], tuple[float, float]],
    ) -> None:
        view_box = self._plot_widget.getPlotItem().getViewBox()
        view_box.disableAutoRange()
        self._plot_widget.setXRange(view[0][0], view[0][1], padding=0)
        self._plot_widget.setYRange(view[1][0], view[1][1], padding=0)

    def _render_interactive_display(
        self,
        state: InteractiveDisplayState,
    ) -> bool:
        self._rendered_point_roster = ()
        if not state.layers:
            self._release_interactive_display()
            return True

        existing = dict(self._interactive_items)
        rendered: dict[tuple[object, ...], Any] = {}
        rendered_points: list[_RenderedPointBlock] = []
        render_input = self._render_input
        if render_input is not None:
            for block in render_input.blocks:
                if block.kind is not ResultLayerKind.EXPERIMENTAL or not block.visible:
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
                for chunk_index, (start, end) in enumerate(
                    self._interactive_curve_chunks(curve)
                ):
                    curve_name = curve.label if chunk_index == 0 else None
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
                                curve.x[start:end],
                                curve.y[start:end],
                                pen=None,
                                symbol="o",
                                symbolSize=curve.point_size,
                                symbolBrush=pg.mkBrush(color),
                                symbolPen=pg.mkPen(color),
                                name=curve_name,
                            )
                        else:
                            item = self._plot_widget.plot(
                                curve.x[start:end],
                                curve.y[start:end],
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
                        item.setData(curve.x[start:end], curve.y[start:end])
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
                if render_input is not None:
                    rendered_block = self._rendered_block_for_interactive_curve(
                        curve,
                        render_input,
                    )
                    if rendered_block is not None and any(
                        chunk.x.size for chunk in rendered_block.chunks
                    ):
                        rendered_points.append(rendered_block)
                for connector in curve.intervention_connectors:
                    boundary_index, boundary_time, pre_value, post_value = connector
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
                            [boundary_time, boundary_time],
                            [pre_value, post_value],
                            pen=pen,
                            name=None,
                        )
                    else:
                        item.setData(
                            [boundary_time, boundary_time],
                            [pre_value, post_value],
                        )
                        item.setPen(pen)
                    item.setOpacity(opacity)
                    rendered[item_key] = item
        for item_key, item in existing.items():
            if item_key not in rendered:
                self._remove_interactive_plot_item(item)
        foreground_set_ids = {
            str(curve.key[2])
            for layer in state.layers
            for curve in layer.projection.curves
            if len(curve.key) >= 3
            and curve.key[0:2] == ("simulation", "foreground")
        }
        self._reconcile_static_simulation_items(foreground_set_ids)
        if self._interactive_locked_view is not None:
            self._restore_interactive_view(self._interactive_locked_view)
        self._interactive_items = rendered
        self._rendered_point_roster = tuple(rendered_points)
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
        starts = [0]
        starts.extend(
            index
            for index in range(1, count)
            if (
                int(segment_ids[index]) != int(segment_ids[index - 1])
                or segment_keys[index] != segment_keys[index - 1]
                or int(discontinuities[index])
                != int(discontinuities[index - 1])
            )
        )
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

        The controller owns the semantic state and invokes this operation by
        presenting an empty state.  External scene/lifecycle boundaries call
        it after clearing or superseding that state.  Keep the order explicit:
        transient rows/items disappear first, static simulation items return
        through PlotItem attachment (which restores their named legend rows),
        then range custody ends and the panel's authoritative range is
        reapplied.
        """

        self._interactive_display._cancel_scheduled_tick()
        self._clear_interactive_items()
        self._reconcile_static_simulation_items(())
        self._end_interactive_range_session()
        self._apply_axis_range_state(self._axis_range, self._presentation)
        if self._render_input is not None and not self._settling_static_input:
            self._render_result_scene(self._render_input)

    def _plot_item_is_attached(self, item: object) -> bool:
        """Read physical PlotItem membership; no parallel attachment state."""

        return item in self._plot_widget.getPlotItem().items

    def _legend_item_is_attached(self, item: object) -> bool:
        legend = self._legend
        if legend is None:
            return False
        return legend.getLabel(item) is not None

    def _detach_static_simulation_item(self, item: object) -> None:
        if self._legend_item_is_attached(item) and self._legend is not None:
            self._legend.removeItem(item)
        if self._plot_item_is_attached(item):
            self._plot_widget.getPlotItem().removeItem(item)

    def _reattach_static_simulation_item(self, item: object) -> None:
        if not self._plot_item_is_attached(item):
            self._plot_widget.getPlotItem().addItem(item)

    def _reconcile_static_simulation_items(
        self,
        detached_set_ids: object,
    ) -> None:
        """Physically detach or reattach rendered static simulation items."""

        detached = {str(set_id) for set_id in detached_set_ids}
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
            set_id = str(role[1])
            should_detach = set_id in detached
            plot_attached = self._plot_item_is_attached(item)
            legend_attached = self._legend_item_is_attached(item)
            is_attached = plot_attached or legend_attached
            if should_detach:
                if not is_attached:
                    continue
                self._detach_static_simulation_item(item)
            elif not plot_attached:
                self._reattach_static_simulation_item(item)

    def discard_interactive_display(self) -> None:
        self._interactive_display.clear(reset_currentness=True)

    def _clear_completion_warning(self) -> None:
        self._completion_warning_badge.setText("")
        self._completion_warning_badge.setToolTip("")
        self._completion_warning_badge.setAccessibleDescription("")
        self._completion_warning_badge.setVisible(False)

    def _render_result_scene(
        self,
        render_input: ResultRenderInput,
    ) -> None:
        self._rendered_point_roster = ()
        self._reconcile_static_simulation_items(())
        existing_items = dict(self._plot_items_by_key)
        self._clear_annotation_items()
        self._plot_widget.setLabel("bottom", render_input.x_label)
        self._plot_widget.setLabel("left", render_input.y_label)
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
        self._axis_toolbar.set_coarse_sampling(self._sampling_limit is not None)
        interaction = self._interaction_mode
        self._axis_toolbar.set_interaction_mode(interaction)
        self._apply_interaction_mode(interaction)
        presentation = self._presentation
        self._render_annotations(render_input, presentation)
        self._render_analysis(render_input)
        axis_range = self._axis_range
        self._axis_toolbar.set_manual_ranges(axis_range.x_min, axis_range.x_max, axis_range.y_min, axis_range.y_max)
        self._axis_toolbar.set_auto_range(axis_range.auto)
        self._apply_plot_presentation(presentation, axis_range=axis_range)
        self._project_completion_warning(render_input)
        self._rendered_point_roster = tuple(rendered_points)

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
        x = block.t[sample]
        sampled_series = tuple(
            (name, values[sample]) for name, values in block.series
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
            candidates = (
                block
                for block in render_input.blocks
                if block.block_id == set_id
                and block.kind is ResultLayerKind.REFERENCE
            )
            source = next(candidates, None)
            if source is None:
                source = next(
                    (
                        block
                        for block in render_input.blocks
                        if block.block_id == set_id
                        and block.kind is ResultLayerKind.EXPLICIT
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
        series_provenance = dict(source.series_provenance)
        curve_chunks = tuple(
            _RenderedPointChunk(
                x=curve.x[start:end],
                series=((series_name, curve.y[start:end]),),
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
            kind=(
                ResultLayerKind.REFERENCE.value
                if role == "ghost"
                and any(
                    block.block_id == set_id
                    and block.kind is ResultLayerKind.REFERENCE
                    for block in render_input.blocks
                )
                else source.kind.value
            ),
            x_name=source.x_name,
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
        for series_name, _values in block.series:
            color = colors[series_name]
            for chunk_index, chunk in enumerate(rendered_chunks):
                y = dict(chunk.series)[series_name]
                singleton = chunk.x.size == 1
                marker = bool(block.markers or singleton)
                item_key = (role, series_name, chunk_index, "marker" if marker else "line")
                item = existing_store.get(item_key)
                name = f"{block.label}: {series_name}"
                if item is None:
                    if marker:
                        item = self._plot_widget.plot(
                            chunk.x, y, pen=None, symbol="o",
                            symbolSize=(block.point_size if block.markers else 5.0),
                            symbolBrush=pg.mkBrush(color), symbolPen=pg.mkPen(color),
                            name=name if chunk_index == 0 else None,
                        )
                    else:
                        item = self._plot_widget.plot(
                            chunk.x, y,
                            pen=pg.mkPen(color=color, width=2, style=(QtCore.Qt.PenStyle.DashLine if block.dashed else QtCore.Qt.PenStyle.SolidLine)),
                            name=name if chunk_index == 0 else None,
                        )
                else:
                    item.setData(chunk.x, y)
                    if marker:
                        item.setSymbolBrush(pg.mkBrush(color))
                        item.setSymbolPen(pg.mkPen(color))
                        item.setSymbolSize(block.point_size if block.markers else 5.0)
                    else:
                        item.setPen(pg.mkPen(color=color, width=2, style=(QtCore.Qt.PenStyle.DashLine if block.dashed else QtCore.Qt.PenStyle.SolidLine)))
                self._update_plot_item_legend(item, name if chunk_index == 0 else None)
                item.setOpacity(block.opacity)
                item_store[item_key] = item
        if rendered_points is not None and any(
            chunk.x.size for chunk in rendered_block.chunks
        ):
            rendered_points.append(rendered_block)
        for boundary_index, species_name, boundary_time, pre_value, post_value in self._intervention_connector_entries(block):
            color = colors[species_name]
            item_key = (role, "intervention-boundary", boundary_index, species_name)
            pen = pg.mkPen(color=color, width=2, style=(QtCore.Qt.PenStyle.DashLine if block.dashed else QtCore.Qt.PenStyle.SolidLine))
            item = existing_store.get(item_key)
            if item is None:
                item = self._plot_widget.plot([boundary_time, boundary_time], [pre_value, post_value], pen=pen, name=None)
            else:
                item.setData([boundary_time, boundary_time], [pre_value, post_value])
                item.setPen(pen)
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
        visible = bool(diagnostic or render_input.completion_warning_details)
        self._completion_warning_badge.setText(
            "Selected X unavailable"
            if diagnostic is not None
            else "Completed with warnings"
        )
        warning_details = render_input.completion_warning_tooltip
        details = "\n\n".join(
            detail
            for detail in (diagnostic, warning_details)
            if detail
        )
        if not visible:
            self._completion_warning_badge.setText("")
            self._completion_warning_badge.setToolTip("")
            self._completion_warning_badge.setAccessibleDescription("")
            self._completion_warning_badge.setVisible(False)
            return
        self._completion_warning_badge.setToolTip(details if visible else "")
        self._completion_warning_badge.setAccessibleDescription(
            details if visible else ""
        )
        self._completion_warning_badge.setVisible(visible)

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
    ) -> None:
        if not render_input.intervention_annotations_visible:
            return
        for annotation in render_input.intervention_annotations:
            if "start" in annotation and "end" in annotation:
                start = self._raw_to_view_coordinate(
                    annotation["start"],
                    log_enabled=presentation.log_x,
                )
                end = self._raw_to_view_coordinate(
                    annotation["end"],
                    log_enabled=presentation.log_x,
                )
                if start is None or end is None:
                    continue
                region = pg.LinearRegionItem(
                    values=(start, end),
                    movable=False,
                    brush=pg.mkBrush(255, 190, 80, 35),
                    pen=pg.mkPen(220, 150, 40, 100),
                )
                self._plot_widget.addItem(region)
                self._annotation_items.append(region)
                continue
            if "time" not in annotation:
                continue
            time = self._raw_to_view_coordinate(
                annotation["time"],
                log_enabled=presentation.log_x,
            )
            if time is None:
                continue
            line = pg.InfiniteLine(
                pos=time,
                angle=90,
                movable=False,
                pen=pg.mkPen("#b36b00", width=1, style=QtCore.Qt.PenStyle.DashLine),
                label=str(annotation.get("label") or "intervention"),
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
        self._rendered_point_roster = ()
        self._legend = self._plot_widget.addLegend()

    def _clear_annotation_items(self) -> None:
        for item in self._annotation_items:
            try:
                self._plot_widget.removeItem(item)
            except (RuntimeError, TypeError, ValueError):
                pass
        self._annotation_items.clear()

    @QtCore.Slot(str, object)
    def _on_axis_option(self, action: str, value: object) -> None:
        if str(action) == "sampling":
            self._sampling_limit = 1000 if str(value) == "coarse" else None
            render_input = self._render_input
            if render_input is not None:
                if (
                    self._interactive_display.has_visible_frame
                    or self._interactive_display.has_pending_batch
                ):
                    self.discard_interactive_display()
                else:
                    self._render_result_scene(render_input)

    @QtCore.Slot()
    def _apply_axis_range(self) -> None:
        presentation = self._presentation
        x_min, x_max, y_min, y_max = self._axis_toolbar.get_manual_ranges()
        auto = self._axis_toolbar.is_auto_range()
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

    @QtCore.Slot(str)
    def _set_interaction_mode(self, mode: str) -> None:
        normalized = str(mode or "").strip().lower()
        if normalized not in {"pan", "rectangle_zoom"}:
            raise ValueError("Plot interaction mode must be Pan or Rectangle Zoom.")
        self._interaction_mode = normalized
        self._axis_toolbar.set_interaction_mode(normalized)
        self._apply_interaction_mode(normalized)

    def _apply_interaction_mode(self, mode: str) -> None:
        view_box = self._plot_widget.getPlotItem().getViewBox()
        view_box.setMouseMode(
            view_box.PanMode if str(mode) == "pan" else view_box.RectMode
        )

    def _request_plot_presentation(
        self,
        option: str,
        enabled: bool,
    ) -> None:
        intent = PlotPresentationIntent(str(option), bool(enabled))
        # The WorkspaceTabRegistry is the sole accepted presentation owner.
        # Do not mutate local state or visible pyqtgraph items until its
        # current-content token has accepted this intent.
        self.plotPresentationIntentRequested.emit(intent)

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
        self._axis_toolbar.set_plot_presentation_state(
            log_x=state.log_x,
            log_y=state.log_y,
            legend_visible=state.legend_visible,
            x_inverted=state.x_inverted,
            y_inverted=state.y_inverted,
        )
        if self._legend is not None:
            self._legend.setVisible(state.legend_visible)
        if axis_range is not None:
            self._apply_axis_range_state(axis_range, state)

    def set_theme(self, *, dark_mode: bool) -> None:
        dark = bool(dark_mode)
        self._plot_widget.setBackground("#1e1e1e" if dark else "w")
        foreground = "#dddddd" if dark else "#222222"
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
) -> str:
    header = f"{label}::{name}"
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
    """Serialize only the private arrays captured for the visible projection."""

    values = tuple(roster)
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive.")
    nonempty = tuple(
        block
        for block in values
        if any(chunk.x.size for chunk in block.chunks)
    )
    if not nonempty:
        raise ValueError("No rendered points are available to copy.")
    planned_cells = 0
    for block in nonempty:
        first_chunk = next(chunk for chunk in block.chunks if chunk.x.size)
        columns = 2 + len(first_chunk.series)
        if block.kind != ResultLayerKind.EXPERIMENTAL.value:
            columns += 3
        rows = sum(int(chunk.x.size) for chunk in block.chunks)
        planned_cells += (rows + 1) * columns
    if planned_cells > _MAX_RENDERED_COPY_CELLS:
        raise ValueError(
            "Rendered Copy payload exceeds the bounded cell limit "
            f"({_MAX_RENDERED_COPY_CELLS:,})."
        )

    output_chunks: list[str] = []
    total_characters = 0
    for block in nonempty:
        chunks = tuple(chunk for chunk in block.chunks if chunk.x.size)
        series_names = tuple(name for name, _values in chunks[0].series)
        if any(
            tuple(name for name, _values in chunk.series) != series_names
            for chunk in chunks[1:]
        ):
            raise ValueError("rendered point chunks must preserve series order.")
        headers: list[str] = [
            _rendered_column_header(
                block.label,
                block.x_name,
                block.x_provenance,
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
            )
            for name in series_names
        )
        output = io.StringIO()
        output.write(encode_tsv_row(headers))
        rows_in_chunk = 0
        for chunk in chunks:
            series_by_name = dict(chunk.series)
            for index in range(chunk.x.size):
                row: list[str] = [f"{float(chunk.x[index]):.17g}"]
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
                    f"{float(series_by_name[name][index]):.17g}"
                    for name in series_names
                )
                output.write(encode_tsv_row(row))
                rows_in_chunk += 1
                if rows_in_chunk >= rows_per_chunk:
                    piece = output.getvalue()
                    total_characters += len(piece)
                    if total_characters > _MAX_RENDERED_COPY_CHARACTERS:
                        raise ValueError(
                            "Rendered Copy payload exceeds the bounded output-size limit."
                        )
                    output_chunks.append(piece)
                    output = io.StringIO()
                    rows_in_chunk = 0
        piece = output.getvalue()
        total_characters += len(piece)
        if total_characters > _MAX_RENDERED_COPY_CHARACTERS:
            raise ValueError(
                "Rendered Copy payload exceeds the bounded output-size limit."
            )
        if piece:
            output_chunks.append(piece)
    return tuple(output_chunks)
