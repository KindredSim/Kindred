"""PyQtGraph workspace surface for one immutable result projection."""

from __future__ import annotations

import io
from dataclasses import dataclass
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.project_presentation_state import (
    AxisSeriesSelection,
    ExperimentalAxisSelection,
    SimulationSeriesAxisSelection,
)
from kindred.gui.interactive_preview import (
    InteractivePreviewBatchIdentity,
    InteractivePreviewPresentedReceipt,
)
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
    ResultLayerKind,
    SimulationResultLayerIdentity,
    WorkspacePlotLayer,
    WorkspaceResultPresenter,
    WorkspaceResultProjection,
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


class PyQtGraphPlotPanel(QtWidgets.QWidget):
    """Physical plot/analysis widget; semantic visibility stays in presentation."""

    referenceLayerVisibilityRequested = QtCore.Signal(bool)
    interventionAnnotationsVisibilityRequested = QtCore.Signal(bool)
    xSeriesRequested = QtCore.Signal(object)
    ySeriesRequested = QtCore.Signal(object)
    plotPresentationIntentRequested = QtCore.Signal(object)
    resultDiagnostic = QtCore.Signal(str)
    renderFailed = QtCore.Signal(str)
    interactiveFrameDisposition = QtCore.Signal(object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        enable_axis_inversion_actions: bool = True,
        enable_reference_layer_toggle_action: bool = True,
        enable_copy_visible_data_action: bool = True,
    ) -> None:
        super().__init__(parent)
        self._enable_axis_inversion_actions = bool(enable_axis_inversion_actions)
        self._enable_reference_layer_toggle_action = bool(
            enable_reference_layer_toggle_action
        )
        self._enable_copy_visible_data_action = bool(enable_copy_visible_data_action)
        self._applied_projection: WorkspaceResultProjection | None = None
        self._rendered_blocks: tuple[ResultDataBlock, ...] = ()
        # A panel is created and destroyed with exactly one mechanism tab by
        # MechanismWorkspaceView.  These presentation/session values are
        # therefore panel-local; retaining tab-keyed maps here only creates a
        # second, stale routing surface inside the tab-owned widget.
        self._sampling_limit: int | None = None
        self._curve_items: list[object] = []
        self._plot_items_by_key: dict[tuple[object, ...], object] = {}
        self._physical_items_by_axis: dict[
            tuple[AxisSeriesSelection, bool], list[object]
        ] = {}
        self._annotation_items: list[object] = []
        self._manual_annotations: list[tuple[str, float, float]] = []
        self._scalar_guides: list[str] = []
        self._interaction_mode = "pan"
        self._presentation = _PlotPresentation()
        self._axis_range = _AxisRangeSession()
        self._dark_mode = False
        self._copy_generation = 0
        self._active_copy_generation: int | None = None
        self._copy_chunks = None
        self._copy_output: io.StringIO | None = None
        self._copy_title = "Copy"
        self._copy_timer = QtCore.QTimer(self)
        self._copy_timer.setInterval(0)
        self._copy_timer.timeout.connect(self._continue_copy)
        self._interactive_timer = QtCore.QTimer(self)
        self._interactive_timer.setSingleShot(True)
        self._interactive_timer.timeout.connect(self._advance_interactive_display)
        self._interactive_items: dict[tuple[object, ...], Any] = {}
        self._interactive_foreground_items_by_set: dict[str, tuple[object, ...]] = {}
        self._interactive_masked_items: dict[int, tuple[object, float, str]] = {}
        self._interactive_locked_view: (
            tuple[tuple[float, float], tuple[float, float]] | None
        ) = None
        self._interactive_display = InteractiveDisplayController(
            render=self._render_interactive_display,
            disposition=self.interactiveFrameDisposition.emit,
            request_tick=self._request_interactive_tick,
            cancel_tick=self._interactive_timer.stop,
            render_failed=self.renderFailed.emit,
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
        self._plot_widget.customContextMenuRequested.connect(self._show_context_menu)

        self._axis_toolbar = AxisToolbar(self)
        self._axis_toolbar.xChanged.connect(self.xSeriesRequested.emit)
        self._axis_toolbar.ySelectionChanged.connect(self.ySeriesRequested.emit)
        self._axis_toolbar.optionsRequested.connect(self._on_axis_option)
        self._axis_toolbar.axisRangeChanged.connect(self._apply_axis_range)
        self._axis_toolbar.addGuideRequested.connect(self._add_scalar_guide)
        self._axis_toolbar.interactionModeRequested.connect(self._set_interaction_mode)
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

    def xlsx_export_projection(self) -> WorkspaceResultProjection | None:
        """Return this page's export authority, or None when unavailable."""

        projection = self._applied_projection
        if projection is None:
            return None
        if (
            projection.copy_all_blocks
            and projection.unavailable_x_diagnostic is None
        ):
            return projection
        return None

    def apply_workspace_result_projection(
        self,
        projection: WorkspaceResultProjection,
    ) -> None:
        """Apply one completed projection exactly at its immutable endpoint."""

        if not isinstance(projection, WorkspaceResultProjection):
            raise TypeError(
                "apply_workspace_result_projection requires WorkspaceResultProjection."
            )
        self._cancel_copy()
        locked_view = self._interactive_locked_view
        # The controller removes or transfers queued/visible approximation
        # before any exact/base scene can publish.  This also retires a stopped
        # overlay before a new axis/series projection changes its semantics.
        self._interactive_display.prepare_workspace_projection(projection)
        try:
            rendered = self._render_projection(projection)
        except Exception as exc:
            # A failed projection is ordinary unavailability for this panel:
            # never retain a stale scene or export authority.
            self._clear_rejected_result_scene()
            self.renderFailed.emit(str(exc))
            raise
        self._applied_projection = projection
        self._rendered_blocks = rendered
        self._interactive_display.commit_workspace_projection(projection)
        if (
            self._interactive_display.has_visible_frame
            and locked_view is not None
        ):
            self._restore_interactive_view(locked_view)
        elif not self._interactive_display.has_current_batch:
            self._end_interactive_range_session()
        if projection.unavailable_x_diagnostic is not None:
            self.resultDiagnostic.emit(projection.unavailable_x_diagnostic)

    def apply_interactive_preview_frame(self, frame: object) -> None:
        """Route one frame into the controller-owned physical protocol."""

        projection = self._applied_projection
        if projection is None:
            from kindred.gui.interactive_preview import InteractivePreviewFrame

            if not isinstance(frame, InteractivePreviewFrame):
                raise TypeError("interactive frame must be an InteractivePreviewFrame.")
            disposition = InteractiveDisplayController.reject_unrenderable(
                frame,
                message="Interactive plot has no applied projection.",
            )
            self.interactiveFrameDisposition.emit(disposition)
            return
        from kindred.gui.interactive_preview import InteractivePreviewFrame

        if not isinstance(frame, InteractivePreviewFrame):
            raise TypeError("interactive frame must be an InteractivePreviewFrame.")
        needs_range = not self._interactive_display.has_current_batch
        self._interactive_display.offer_frame(
            frame,
            projection=projection,
        )
        if needs_range and self._interactive_display.has_renderable_frame:
            self._begin_interactive_range_session()

    def apply_interactive_preview_terminal_outcome(self, outcome: object) -> bool:
        """Route one exact terminal fact into controller settlement."""

        projection = self._applied_projection
        if projection is None:
            return False
        from kindred.gui.interactive_preview import InteractivePreviewTerminalOutcome

        if not isinstance(outcome, InteractivePreviewTerminalOutcome):
            raise TypeError(
                "Interactive Preview terminal outcome must be typed."
            )
        changed = self._interactive_display.apply_terminal_outcome(
            outcome,
            projection=projection,
        )
        if changed and not self._interactive_display.has_current_batch:
            self._end_interactive_range_session()
        return changed

    def interactive_exact_release_batch(
        self,
    ) -> InteractivePreviewPresentedReceipt | None:
        """Consume the controller's post-render receipt for exact release."""

        had_current_batch = self._interactive_display.has_current_batch
        receipt = self._interactive_display.exact_release_batch()
        if had_current_batch and not self._interactive_display.has_current_batch:
            self._end_interactive_range_session()
        return receipt

    def allocate_interactive_preview_batch(
        self,
        *,
        tab_id: str,
        document_identity: tuple[int, str],
        target_set_ids: tuple[str, ...],
    ) -> InteractivePreviewBatchIdentity:
        """Mint a controller-owned generation and immutable batch identity."""

        return self._interactive_display.allocate_batch_identity(
            tab_id=tab_id,
            document_identity=document_identity,
            target_set_ids=target_set_ids,
        )

    def apply_interactive_preview_retirement(self, retirement: object) -> bool:
        """Apply one lifecycle retirement intent through the controller."""

        from kindred.gui.interactive_preview import InteractivePreviewRetirement

        if not isinstance(retirement, InteractivePreviewRetirement):
            raise TypeError("Interactive Preview retirement must be typed.")
        projection = self._applied_projection
        if projection is None or retirement.tab_id != projection.tab_id:
            return False
        if retirement.target_set_ids:
            changed = self._interactive_display.retire_targets(
                retirement.target_set_ids
            )
        else:
            changed = self._interactive_display.retire(clear_visible=False)
        if changed:
            self._end_interactive_range_session()
        return changed

    def stop_interactive_display(self) -> bool:
        """Cancel visual work and freeze the last physically stable display."""

        if (
            not self._interactive_display.has_visible_frame
            and not self._interactive_display.has_current_batch
        ):
            return False
        self._interactive_display.stop()
        # Stop is an emergency freeze boundary.  The gesture-owned range is
        # no longer valid once slider custody is gone; subsequent ordinary
        # plot preference changes must not restore that stale lock.
        self._end_interactive_range_session()
        return True

    def apply_application_plot_preferences(
        self,
        *,
        color_for_species: Callable[[str, tuple[str, ...]], QtGui.QColor],
        known_species: tuple[str, ...],
        experimental_point_size: int,
        experimental_point_opacity: float,
    ) -> None:
        """Apply committed global plot preferences to existing items only."""

        projection = self._applied_projection
        if projection is None:
            return
        try:
            if not callable(color_for_species):
                raise TypeError("color_for_species must be callable.")
            provenance_by_axis: dict[AxisSeriesSelection, str] = {}
            for block in projection.copy_all_blocks:
                provenance_by_name = dict(block.series_provenance)
                for name, selection in block.series_axes:
                    if not isinstance(selection, ExperimentalAxisSelection):
                        continue
                    provenance = provenance_by_name.get(name)
                    if provenance is None:
                        raise ValueError(
                            "Experimental plot item lacks color provenance."
                        )
                    color_name = str(
                        provenance.get("model_target")
                        or provenance.get("raw_column")
                        or ""
                    )
                    if not color_name:
                        raise ValueError(
                            "Experimental plot item has no color identity."
                        )
                    previous = provenance_by_axis.setdefault(
                        selection,
                        color_name,
                    )
                    if previous != color_name:
                        raise ValueError(
                            "Experimental plot item color provenance conflicts."
                        )
            view_box = self._plot_widget.getPlotItem().getViewBox()
            previous_auto_range = view_box.autoRangeEnabled()
            view_box.state["autoRange"] = [False, False]
            view_box._autoRangeNeedsUpdate = False
            try:
                for (selection, is_experimental_marker), items in tuple(
                    self._physical_items_by_axis.items()
                ):
                    if isinstance(selection, SimulationSeriesAxisSelection):
                        color_name = selection.series_name
                    else:
                        color_name = provenance_by_axis.get(selection, "")
                        if not color_name:
                            raise ValueError(
                                "Experimental plot item lacks applied provenance."
                            )
                    color = color_for_species(color_name, tuple(known_species))
                    if not isinstance(color, QtGui.QColor) or not color.isValid():
                        raise ValueError("Resolved plot color must be a valid QColor.")
                    for item in items:
                        self._apply_item_color(item, color)
                        if is_experimental_marker:
                            item.setSymbolSize(float(experimental_point_size))
                            item.setOpacity(float(experimental_point_opacity))
            finally:
                # pyqtgraph 0.13.7's public enable/disable methods flush a
                # pending range when disabling and queue one when enabling.
                # Restore only the pre-existing capability, never a style-change
                # bounds update, so prepareForPaint cannot recompose this view.
                view_box.state["autoRange"] = list(previous_auto_range)
                view_box._autoRangeNeedsUpdate = False
            if self._interactive_display.has_visible_frame:
                if self._interactive_locked_view is not None:
                    self._restore_interactive_view(self._interactive_locked_view)
                self._mask_interactive_foreground_items(
                    set(self._interactive_display.masking_intent)
                )
        except Exception as exc:
            self._clear_rejected_result_scene()
            self.renderFailed.emit(str(exc))
            raise

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
        if self._applied_projection is not None:
            self._apply_plot_presentation(state)
            if self._interactive_locked_view is not None:
                self._restore_interactive_view(self._interactive_locked_view)

    def drop_tab_session_state(self) -> None:
        self._discard_interactive_display()
        self._manual_annotations.clear()
        self._scalar_guides.clear()
        self._interaction_mode = "pan"
        self._presentation = _PlotPresentation()
        self._sampling_limit = None
        self._axis_range = _AxisRangeSession()

    def reset_plot_session_state(self) -> None:
        """Project replacement/reopen resets non-archived Plot session truth."""

        self._discard_interactive_display()
        self._manual_annotations.clear()
        self._scalar_guides.clear()
        self._interaction_mode = "pan"
        self._presentation = _PlotPresentation()
        self._sampling_limit = None
        self._axis_range = _AxisRangeSession()
        self._axis_toolbar.set_interaction_mode("pan")
        self._axis_toolbar.set_coarse_sampling(False)
        self._axis_toolbar.set_manual_ranges(None, None, None, None)
        self._axis_toolbar.set_auto_range(True)
        self._apply_interaction_mode("pan")

    def clear_workspace_result_projection(self) -> None:
        """Clear rejected public truth without affecting other tab-local state."""

        self._clear_rejected_result_scene()

    def _clear_rejected_result_scene(self) -> None:
        """Revoke only this physical page's applied result authority."""

        self._cancel_copy()
        self._discard_interactive_display()
        for clear in (
            self._clear_plot_items,
            lambda: self._parameter_table.setRowCount(0),
            lambda: self._species_table.setRowCount(0),
            self._clear_completion_warning,
        ):
            try:
                clear()
            except Exception:
                pass
        self._applied_projection = None
        self._rendered_blocks = ()

    def _request_interactive_tick(self, delay_seconds: float) -> None:
        milliseconds = max(0, int(np.ceil(float(delay_seconds) * 1000.0)))
        self._interactive_timer.start(milliseconds)

    @QtCore.Slot()
    def _advance_interactive_display(self) -> None:
        self._interactive_display.tick()

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
    ) -> None:
        # Prevalidate the complete desired roster before touching masks, plot
        # items, or the view.  This keeps malformed second items from turning
        # into a partial physical commit.
        desired_keys: set[tuple[object, ...]] = set()
        for layer_index, layer in enumerate(state.layers):
            projection = layer.projection
            if not projection.tab_id or not projection.batch_id:
                raise ValueError("Interactive display projection identity is invalid.")
            layer_opacity = float(layer.opacity)
            if not np.isfinite(layer_opacity) or not 0.0 <= layer_opacity <= 1.0:
                raise ValueError("Interactive display layer opacity is invalid.")
            for curve in projection.curves:
                style = "marker" if curve.markers else "line"
                color = QtGui.QColor(curve.color)
                if not color.isValid():
                    raise ValueError(
                        f"Interactive curve {curve.label!r} has an invalid color."
                    )
                curve_opacity = float(curve.opacity)
                if not np.isfinite(curve_opacity) or not 0.0 <= curve_opacity <= 1.0:
                    raise ValueError("Interactive curve opacity is invalid.")
                x_values = np.asarray(curve.x)
                y_values = np.asarray(curve.y)
                if x_values.ndim != 1 or y_values.ndim != 1:
                    raise ValueError("Interactive curve coordinates must be one-dimensional.")
                if x_values.size != y_values.size:
                    raise ValueError("Interactive curve coordinates must have equal length.")
                for chunk_index, (_start, _end) in enumerate(
                    self._interactive_curve_chunks(curve)
                ):
                    item_key = (
                        layer_index,
                        *curve.key,
                        "segment",
                        chunk_index,
                        style,
                    )
                    if item_key in desired_keys:
                        raise ValueError("Interactive display item roster contains duplicates.")
                    desired_keys.add(item_key)
                for connector in curve.intervention_connectors:
                    if len(connector) != 4:
                        raise ValueError("Interactive connector must contain four values.")
                    boundary_index, boundary_time, pre_value, post_value = connector
                    if not all(
                        np.isfinite(float(value))
                        for value in (boundary_index, boundary_time, pre_value, post_value)
                    ):
                        raise ValueError("Interactive connector values must be finite.")
                    item_key = (
                        layer_index,
                        *curve.key,
                        "intervention-boundary",
                        boundary_index,
                    )
                    if item_key in desired_keys:
                        raise ValueError("Interactive display item roster contains duplicates.")
                    desired_keys.add(item_key)

        existing = dict(self._interactive_items)
        old_masked = dict(self._interactive_masked_items)
        old_masked_opacities: dict[int, tuple[object, float]] = {}
        physical_mask_candidates = [
            item for item, _opacity, _set_id in old_masked.values()
        ]
        physical_mask_candidates.extend(
            item
            for items in self._interactive_foreground_items_by_set.values()
            for item in items
        )
        for item in physical_mask_candidates:
            opacity = getattr(item, "opacity", None)
            if callable(opacity):
                old_masked_opacities[id(item)] = (item, float(opacity()))
        old_view = self._capture_interactive_physical_view()
        rendered: dict[tuple[object, ...], Any] = {}
        created_items: list[object] = []
        removed_items: list[object] = []
        reused_snapshots: dict[int, tuple[object, dict[str, object]]] = {}
        try:
            for layer_index, layer in enumerate(state.layers):
                for curve in layer.projection.curves:
                    style = "marker" if curve.markers else "line"
                    color = QtGui.QColor(curve.color)
                    opacity = layer.opacity * curve.opacity
                    for chunk_index, (start, end) in enumerate(
                        self._interactive_curve_chunks(curve)
                    ):
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
                                    name=None,
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
                                    name=None,
                                )
                            # Journal each new item immediately after creation.
                            created_items.append(item)
                        else:
                            reused_snapshots.setdefault(
                                id(item),
                                (item, self._snapshot_interactive_plot_item(item)),
                            )
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
                        item.setOpacity(opacity)
                        rendered[item_key] = item
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
                            created_items.append(item)
                        else:
                            reused_snapshots.setdefault(
                                id(item),
                                (item, self._snapshot_interactive_plot_item(item)),
                            )
                            item.setData(
                                [boundary_time, boundary_time],
                                [pre_value, post_value],
                            )
                            item.setPen(pen)
                        item.setOpacity(opacity)
                        rendered[item_key] = item
            for item_key, item in existing.items():
                if item_key not in rendered:
                    # Record before removal: a partially completed Qt remove
                    # still needs an exact reattach if the call raises.
                    removed_items.append(item)
                    self._remove_interactive_plot_item(item)
            # Masks are part of the same call-local commit, after the desired
            # item roster has fully rendered and before maps/view are swapped.
            self._mask_interactive_foreground_items(set(state.masked_set_ids))
            locked_view = self._interactive_locked_view
            if locked_view is not None:
                self._restore_interactive_view(locked_view)
            self._interactive_items = rendered
        except Exception:
            # Restore removals, remove newly created objects, restore every
            # reused item snapshot, and put masks/view back exactly as found.
            for item in reversed(removed_items):
                self._reattach_interactive_plot_item(item)
            for item in reversed(created_items):
                self._remove_plot_item(item)
            for item, snapshot in reused_snapshots.values():
                self._restore_interactive_plot_item(item, snapshot)
            self._interactive_masked_items = old_masked
            for item, opacity in old_masked_opacities.values():
                try:
                    item.setOpacity(opacity)
                except (RuntimeError, TypeError, ValueError):
                    pass
            self._restore_interactive_physical_view(old_view)
            self._interactive_items = existing
            raise

    @staticmethod
    def _snapshot_interactive_plot_item(item: object) -> dict[str, object]:
        """Capture only the mutable surface touched by transient rendering."""

        snapshot: dict[str, object] = {}
        get_data = getattr(item, "getData", None)
        if callable(get_data):
            data = get_data()
            if isinstance(data, tuple) and len(data) == 2:
                snapshot["data"] = tuple(
                    None if values is None else np.array(values, copy=True)
                    for values in data
                )
        for name in (
            "opacity",
            "pen",
            "symbol",
            "symbolSize",
            "symbolBrush",
            "symbolPen",
        ):
            getter = getattr(item, name, None)
            if callable(getter):
                try:
                    snapshot[name] = getter()
                except (RuntimeError, TypeError, ValueError):
                    pass
        return snapshot

    @staticmethod
    def _restore_interactive_plot_item(
        item: object,
        snapshot: dict[str, object],
    ) -> None:
        data = snapshot.get("data")
        set_data = getattr(item, "setData", None)
        if data is not None and callable(set_data):
            values = tuple(data)
            set_data(values[0], values[1])
        setters = {
            "symbol": "setSymbol",
            "symbolSize": "setSymbolSize",
            "symbolBrush": "setSymbolBrush",
            "symbolPen": "setSymbolPen",
            "pen": "setPen",
            "opacity": "setOpacity",
        }
        for name, setter_name in setters.items():
            if name not in snapshot:
                continue
            setter = getattr(item, setter_name, None)
            if callable(setter):
                setter(snapshot[name])

    def _capture_interactive_physical_view(
        self,
    ) -> tuple[
        tuple[tuple[float, float], tuple[float, float]] | None,
        tuple[object, ...] | None,
    ]:
        view_box = self._plot_widget.getPlotItem().getViewBox()
        try:
            ranges = view_box.viewRange()
            view = (
                (float(ranges[0][0]), float(ranges[0][1])),
                (float(ranges[1][0]), float(ranges[1][1])),
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            view = None
        auto_range = None
        state = getattr(view_box, "state", None)
        if isinstance(state, Mapping):
            raw = state.get("autoRange")
            if raw is not None:
                try:
                    auto_range = tuple(raw)
                except TypeError:
                    auto_range = None
        return view, auto_range

    def _restore_interactive_physical_view(
        self,
        snapshot: tuple[
            tuple[tuple[float, float], tuple[float, float]] | None,
            tuple[object, ...] | None,
        ],
    ) -> None:
        view, auto_range = snapshot
        if view is None:
            return
        view_box = self._plot_widget.getPlotItem().getViewBox()
        disable = getattr(view_box, "disableAutoRange", None)
        if callable(disable):
            disable()
        self._plot_widget.setXRange(view[0][0], view[0][1], padding=0)
        self._plot_widget.setYRange(view[1][0], view[1][1], padding=0)
        if auto_range is not None:
            state = getattr(view_box, "state", None)
            if isinstance(state, Mapping):
                state["autoRange"] = list(auto_range)

    def _remove_interactive_plot_item(self, item: object) -> None:
        """Remove one stale item and let a failure abort the render call."""

        if self._legend is not None:
            try:
                self._legend.removeItem(item)
            except (KeyError, RuntimeError, TypeError, ValueError):
                pass
        self._plot_widget.removeItem(item)

    def _reattach_interactive_plot_item(self, item: object) -> None:
        add_item = getattr(self._plot_widget, "addItem", None)
        if not callable(add_item):
            raise RuntimeError("Interactive plot cannot reattach a removed item.")
        scene = getattr(item, "scene", None)
        if callable(scene):
            try:
                if scene() is not None:
                    return
            except (RuntimeError, TypeError, ValueError):
                pass
        add_item(item)

    @staticmethod
    def _interactive_curve_chunks(
        curve: object,
    ) -> tuple[tuple[int, int], ...]:
        """Split a transient curve at every executor-issued topology change."""

        x = getattr(curve, "x")
        count = int(x.size)
        if count == 0:
            return ()
        segment_ids = getattr(curve, "segment_ids", None)
        segment_keys = getattr(curve, "segment_keys", None)
        discontinuities = getattr(curve, "discontinuity_ids", None)
        if segment_ids is None or segment_keys is None or discontinuities is None:
            return ((0, count),)
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

    def _mask_interactive_foreground_items(self, set_ids: set[str]) -> None:
        """Reconcile completed foreground opacity to the exact overlay roster."""

        active = {str(set_id) for set_id in set_ids}
        for item_key, (item, opacity, set_id) in tuple(
            self._interactive_masked_items.items()
        ):
            if set_id in active:
                continue
            self._interactive_masked_items.pop(item_key, None)
            try:
                item.setOpacity(opacity)
            except (RuntimeError, TypeError, ValueError):
                pass

        for set_id in active:
            for item in self._interactive_foreground_items_by_set.get(set_id, ()):
                item_key = id(item)
                if item_key in self._interactive_masked_items:
                    continue
                opacity = getattr(item, "opacity", None)
                if not callable(opacity):
                    continue
                current = float(opacity())
                self._interactive_masked_items[item_key] = (
                    item,
                    current,
                    set_id,
                )
                item.setOpacity(0.0)

    def _restore_interactive_foreground_items(
        self,
        *,
        keep_set_ids: set[str] | None = None,
    ) -> None:
        """Restore completed foreground opacity outside an active overlay roster."""

        keep = None if keep_set_ids is None else {str(value) for value in keep_set_ids}
        masked = tuple(self._interactive_masked_items.items())
        for item_key, (item, opacity, set_id) in masked:
            if keep is not None and set_id in keep:
                continue
            self._interactive_masked_items.pop(item_key, None)
            try:
                item.setOpacity(opacity)
            except (RuntimeError, TypeError, ValueError):
                pass

    def _discard_interactive_display(self) -> None:
        self._restore_interactive_foreground_items()
        self._interactive_display.clear(reset_currentness=True)
        self._clear_interactive_items()
        self._end_interactive_range_session()

    def _clear_completion_warning(self) -> None:
        self._completion_warning_badge.setText("")
        self._completion_warning_badge.setToolTip("")
        self._completion_warning_badge.setAccessibleDescription("")
        self._completion_warning_badge.setVisible(False)

    def _render_projection(
        self,
        projection: WorkspaceResultProjection,
    ) -> tuple[ResultDataBlock, ...]:
        self._restore_interactive_foreground_items()
        existing_items = dict(self._plot_items_by_key)
        self._clear_annotation_items()
        self._plot_widget.setLabel("bottom", projection.x_label)
        self._plot_widget.setLabel("left", projection.y_label)
        block_by_id = {block.block_id: block for block in projection.copy_all_blocks}
        rendered_blocks: list[ResultDataBlock] = []
        rendered_items: dict[tuple[object, ...], object] = {}
        physical_items: dict[tuple[AxisSeriesSelection, bool], list[object]] = {}
        foreground_items: dict[str, list[object]] = {}
        reference_items: list[object] = []
        self._curve_items = []
        for layer in projection.plot_layers:
            if not layer.visible:
                continue
            block = block_by_id.get(layer.layer_id)
            if block is None:
                continue
            rendered = self._render_layer(
                layer,
                block,
                sampling_limit=self._sampling_limit,
                existing_items=existing_items,
                rendered_items=rendered_items,
                physical_items=physical_items,
                foreground_items=foreground_items,
                reference_items=reference_items,
            )
            if rendered is not None:
                rendered_blocks.append(rendered)

        for item_key, item in existing_items.items():
            if item_key in rendered_items:
                continue
            self._remove_plot_item(item)

        self._plot_items_by_key = rendered_items
        self._physical_items_by_axis = physical_items
        reference_ids = {id(item) for item in reference_items}
        self._interactive_foreground_items_by_set = {
            set_id: tuple(item for item in items if id(item) not in reference_ids)
            for set_id, items in foreground_items.items()
        }
        self._axis_toolbar.set_x_candidates(
            projection.axis_candidates,
            selected=projection.selected_x_series,
        )
        y_candidates = tuple(
            candidate
            for candidate in projection.axis_candidates
            if not (
                isinstance(
                    candidate.selection,
                    SimulationSeriesAxisSelection,
                )
                and candidate.selection.series_name == "t"
            )
        )
        derived_overlay_y = set(projection.derived_overlay_y_series)
        locked_checked: dict[AxisSeriesSelection, str] = {}
        for layer in projection.plot_layers:
            if layer.kind is not ResultLayerKind.EXPERIMENTAL:
                continue
            provenance_by_name = dict(layer.series_provenance)
            for name, selection in layer.series_axes:
                if selection not in derived_overlay_y:
                    continue
                mapped_model = str(
                    provenance_by_name.get(name, {}).get("model_target")
                    or ""
                )
                if not mapped_model:
                    raise ValueError(
                        "Derived overlay Y identity lacks mapped-model "
                        "provenance."
                    )
                previous = locked_checked.setdefault(selection, mapped_model)
                if previous != mapped_model:
                    raise ValueError(
                        "Derived overlay Y identity has conflicting "
                        "mapped-model provenance."
                    )
        if set(locked_checked) != derived_overlay_y:
            raise ValueError(
                "Derived overlay Y identities must match active projected "
                "overlay layers."
            )
        self._axis_toolbar.set_y_candidates(
            [
                (
                    candidate,
                    projection.selected_y_uses_default_all
                    or candidate.selection in projection.selected_y_series
                    or candidate.selection in derived_overlay_y,
                )
                for candidate in y_candidates
            ],
            locked_checked=locked_checked,
        )
        self._axis_toolbar.set_coarse_sampling(
            self._sampling_limit is not None
        )
        interaction = self._interaction_mode
        self._axis_toolbar.set_interaction_mode(interaction)
        self._apply_interaction_mode(interaction)
        presentation = self._presentation
        self._render_annotations(projection, presentation)
        self._render_manual_annotations(presentation)
        self._render_scalar_guides(projection, presentation)
        self._render_analysis(projection)
        axis_range = self._axis_range
        self._axis_toolbar.set_manual_ranges(
            axis_range.x_min,
            axis_range.x_max,
            axis_range.y_min,
            axis_range.y_max,
        )
        self._axis_toolbar.set_auto_range(axis_range.auto)
        self._apply_plot_presentation(
            presentation,
            axis_range=axis_range,
            refresh_geometry=False,
        )
        self._project_completion_warning(projection)
        return tuple(rendered_blocks)

    @staticmethod
    def _physical_layer_identity(layer: WorkspacePlotLayer) -> object:
        """Map semantic role changes onto the compatible retained-result slot."""

        identity = layer.layer_id
        if isinstance(identity, SimulationResultLayerIdentity) and identity.kind in {
            ResultLayerKind.EXPLICIT,
            ResultLayerKind.REFERENCE,
        }:
            return SimulationResultLayerIdentity(
                ResultLayerKind.REFERENCE,
                identity.set_id,
            )
        return identity

    def _update_plot_item_legend(
        self,
        item: object,
        name: str | None,
    ) -> None:
        set_name = getattr(item, "setName", None)
        if callable(set_name):
            set_name(name)
        if name is None or self._legend is None:
            return
        get_label = getattr(self._legend, "getLabel", None)
        label = get_label(item) if callable(get_label) else None
        set_text = getattr(label, "setText", None)
        if callable(set_text):
            set_text(name)

    def _remove_plot_item(self, item: object) -> None:
        if self._legend is not None:
            try:
                self._legend.removeItem(item)
            except (KeyError, RuntimeError, TypeError, ValueError):
                pass
        try:
            self._plot_widget.removeItem(item)
        except (RuntimeError, TypeError, ValueError):
            pass

    def _render_layer(
        self,
        layer: WorkspacePlotLayer,
        block: ResultDataBlock,
        *,
        sampling_limit: int | None,
        existing_items: Mapping[tuple[object, ...], object] | None = None,
        rendered_items: dict[tuple[object, ...], object] | None = None,
        physical_items: dict[tuple[AxisSeriesSelection, bool], list[object]]
        | None = None,
        foreground_items: dict[str, list[object]] | None = None,
        reference_items: list[object] | None = None,
    ) -> ResultDataBlock | None:
        if layer.kind is not block.kind:
            raise ValueError("plot layer and data block kinds must match.")
        sample, chunks = self._sample_and_chunks(block, sampling_limit)
        x = block.t[sample]
        if block.kind is ResultLayerKind.EXPERIMENTAL:
            if block.source_row is None:
                raise ValueError("experimental plot block lacks source_row.")
            segment_ids = None
            segment_keys = None
            discontinuity_ids = None
            source_row = block.source_row[sample]
        else:
            if (
                block.segment_ids is None
                or block.segment_keys is None
                or block.discontinuity_ids is None
            ):
                raise ValueError("simulation plot block lacks executor identity.")
            segment_ids = block.segment_ids[sample]
            segment_keys = tuple(block.segment_keys[int(index)] for index in sample)
            discontinuity_ids = block.discontinuity_ids[sample]
            source_row = None
        colors = dict(layer.color_by_series)
        axis_by_series = dict(block.series_axes)
        rendered_series: list[tuple[str, np.ndarray]] = []
        item_store = rendered_items if rendered_items is not None else {}
        physical_store = physical_items if physical_items is not None else {}
        existing_store = existing_items if existing_items is not None else {}
        foreground_store = (
            foreground_items if foreground_items is not None else {}
        )
        foreground_set_id = (
            layer.layer_id.set_id
            if isinstance(layer.layer_id, SimulationResultLayerIdentity)
            else None
        )
        foreground_layer = layer.kind in {
            ResultLayerKind.EXPLICIT,
            ResultLayerKind.PREVIEW,
        }
        reference_store = reference_items if reference_items is not None else []
        physical_layer_identity = self._physical_layer_identity(layer)
        for series_name, values in block.series:
            y = values[sample]
            color = colors[series_name]
            axis_selection = axis_by_series[series_name]
            is_experimental_marker = block.kind is ResultLayerKind.EXPERIMENTAL
            name = f"{layer.label}: {series_name}"
            for chunk_index, (start, end) in enumerate(chunks):
                singleton = end - start == 1
                marker = bool(layer.markers or singleton)
                item_key = (
                    physical_layer_identity,
                    series_name,
                    chunk_index,
                    "marker" if marker else "line",
                )
                item = existing_store.get(item_key)
                if item is None:
                    if marker:
                        item = self._plot_widget.plot(
                            x[start:end],
                            y[start:end],
                            pen=None,
                            symbol="o",
                            symbolSize=(layer.point_size if layer.markers else 5.0),
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
                else:
                    item.setData(x[start:end], y[start:end])
                    if marker:
                        item.setSymbolBrush(pg.mkBrush(color))
                        item.setSymbolPen(pg.mkPen(color))
                        item.setSymbolSize(layer.point_size if layer.markers else 5.0)
                    else:
                        item.setPen(
                            pg.mkPen(
                                color=color,
                                width=2,
                                style=(
                                    QtCore.Qt.PenStyle.DashLine
                                    if layer.dashed
                                    else QtCore.Qt.PenStyle.SolidLine
                                ),
                            )
                        )
                self._update_plot_item_legend(
                    item,
                    name if chunk_index == 0 else None,
                )
                item.setOpacity(layer.opacity)
                item_store[item_key] = item
                self._curve_items.append(item)
                if foreground_layer and foreground_set_id is not None:
                    foreground_store.setdefault(foreground_set_id, []).append(item)
                if layer.kind is ResultLayerKind.REFERENCE:
                    reference_store.append(item)
                physical_store.setdefault(
                    (axis_selection, is_experimental_marker),
                    [],
                ).append(item)
            rendered_series.append((series_name, y))
        for (
            boundary_index,
            species_name,
            boundary_time,
            pre_value,
            post_value,
        ) in self._intervention_connector_entries(layer, block):
            color = colors[species_name]
            item_key = (
                physical_layer_identity,
                "intervention-boundary",
                boundary_index,
                species_name,
            )
            item = existing_store.get(item_key)
            pen = pg.mkPen(
                color=color,
                width=2,
                style=(
                    QtCore.Qt.PenStyle.DashLine
                    if layer.dashed
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
            item.setOpacity(layer.opacity)
            item_store[item_key] = item
            self._curve_items.append(item)
            if foreground_layer and foreground_set_id is not None:
                foreground_store.setdefault(foreground_set_id, []).append(item)
            if layer.kind is ResultLayerKind.REFERENCE:
                reference_store.append(item)
            physical_store.setdefault(
                (SimulationSeriesAxisSelection(species_name), False),
                [],
            ).append(item)
        if not rendered_series:
            return None
        return ResultDataBlock(
            block_id=block.block_id,
            label=block.label,
            kind=block.kind,
            t=x,
            series=tuple(rendered_series),
            series_axes=block.series_axes,
            x_axis=block.x_axis,
            result_entry=block.result_entry,
            segment_ids=segment_ids,
            segment_keys=segment_keys,
            discontinuity_ids=discontinuity_ids,
            source_row=source_row,
            x_name=block.x_name,
            series_provenance=block.series_provenance,
            x_provenance=block.x_provenance,
        )

    @staticmethod
    def _intervention_connector_entries(
        layer: WorkspacePlotLayer,
        block: ResultDataBlock,
    ) -> tuple[tuple[int, str, float, float, float], ...]:
        """Return exact vertical connectors for a time-X simulation block."""

        if layer.kind is ResultLayerKind.EXPERIMENTAL:
            return ()
        if not (
            isinstance(block.x_axis, SimulationSeriesAxisSelection)
            and block.x_axis.series_name == "t"
        ):
            return ()
        series_names = {name for name, _values in block.series}
        entries: list[tuple[int, str, float, float, float]] = []
        for boundary_index, boundary in enumerate(layer.presentation_boundaries):
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

    def _render_analysis(self, projection: WorkspaceResultProjection) -> None:
        self._parameter_table.update_parameters(
            {name: (value, unit) for name, value, unit in projection.parameters}
        )
        block = projection.primary_statistics
        if block is None:
            self._species_table.setRowCount(0)
            return
        self._species_table.update_results(
            canonical_time=block.t,
            species_data=dict(block.series),
            intervention_execution=(
                block.result_entry.result.intervention_execution
            ),
            presentation_boundaries=(
                block.result_entry.result.presentation_boundaries
            ),
            accepted_result_key=(
                getattr(block.result_entry, "key", None)
                if block.result_entry is not None
                else None
            ),
        )
    def _project_completion_warning(
        self,
        projection: WorkspaceResultProjection,
    ) -> None:
        diagnostic = projection.unavailable_x_diagnostic
        visible = bool(diagnostic or projection.completion_warnings)
        self._completion_warning_badge.setText(
            "Selected X unavailable"
            if diagnostic is not None
            else "Completed with warnings"
        )
        warning_details = projection.completion_warning_tooltip
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
        """Map one view coordinate back to raw semantic plot coordinates."""

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
        projection: WorkspaceResultProjection,
        presentation: _PlotPresentation,
    ) -> None:
        if not projection.intervention_annotations_visible:
            return
        for annotation in projection.intervention_annotations:
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

    def _render_manual_annotations(
        self,
        presentation: _PlotPresentation,
    ) -> None:
        foreground = "#f3f3f3" if self._dark_mode else "#202020"
        background = "#303030cc" if self._dark_mode else "#ffffffe6"
        border = "#bbbbbb" if self._dark_mode else "#555555"
        for text, x, y in self._manual_annotations:
            view_x = self._raw_to_view_coordinate(
                x,
                log_enabled=presentation.log_x,
            )
            view_y = self._raw_to_view_coordinate(
                y,
                log_enabled=presentation.log_y,
            )
            if view_x is None or view_y is None:
                continue
            item = pg.TextItem(
                text=str(text),
                color=foreground,
                fill=pg.mkBrush(background),
                border=pg.mkPen(border),
                anchor=(0.5, 0.5),
            )
            item.setPos(view_x, view_y)
            self._plot_widget.addItem(item)
            self._annotation_items.append(item)

    def _render_scalar_guides(
        self,
        projection: WorkspaceResultProjection,
        presentation: _PlotPresentation,
    ) -> None:
        scalars = dict(projection.scalar_values)
        for name in self._scalar_guides:
            if name not in scalars:
                continue
            position = self._raw_to_view_coordinate(
                scalars[name],
                log_enabled=presentation.log_y,
            )
            if position is None:
                continue
            line = pg.InfiniteLine(
                pos=position,
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
        if name not in self._scalar_guides:
            self._scalar_guides.append(name)
        self.apply_workspace_result_projection(projection)

    def _clear_plot_items(self) -> None:
        self._restore_interactive_foreground_items()
        self._plot_widget.clear()
        self._curve_items.clear()
        self._plot_items_by_key.clear()
        self._physical_items_by_axis.clear()
        self._interactive_foreground_items_by_set.clear()
        self._annotation_items.clear()
        self._legend = self._plot_widget.addLegend()

    def _clear_annotation_items(self) -> None:
        for item in self._annotation_items:
            try:
                self._plot_widget.removeItem(item)
            except (RuntimeError, TypeError, ValueError):
                pass
        self._annotation_items.clear()

    def _cancel_copy(self) -> None:
        chunks = self._copy_chunks
        self._copy_generation += 1
        self._active_copy_generation = None
        self._copy_timer.stop()
        self._copy_chunks = None
        self._copy_output = None
        close = getattr(chunks, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    @staticmethod
    def _apply_item_color(item: object, color: QtGui.QColor) -> None:
        options = getattr(item, "opts", {})
        symbol = options.get("symbol") if isinstance(options, dict) else None
        if symbol not in {None, "none"}:
            item.setSymbolBrush(pg.mkBrush(color))
            item.setSymbolPen(pg.mkPen(color))
            return
        pen = options.get("pen") if isinstance(options, dict) else None
        if pen is None:
            item.setPen(pg.mkPen(color))
            return
        updated_pen = QtGui.QPen(pen)
        updated_pen.setColor(QtGui.QColor(color))
        item.setPen(updated_pen)

    @QtCore.Slot(str, object)
    def _on_axis_option(self, action: str, value: object) -> None:
        normalized = str(action)
        projection = self._applied_projection
        if normalized == "clear_scalar_guides":
            if projection is None:
                return
            self._scalar_guides.clear()
            self.apply_workspace_result_projection(projection)
            return
        if normalized != "sampling":
            return
        if projection is None:
            return
        previous_limit = self._sampling_limit
        self._sampling_limit = 1000 if str(value) == "coarse" else None
        try:
            self.apply_workspace_result_projection(projection)
        except Exception:
            self._sampling_limit = previous_limit
            raise

    @QtCore.Slot()
    def _apply_axis_range(self) -> None:
        projection = self._applied_projection
        if projection is None:
            return
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

    @QtCore.Slot(QtCore.QPoint)
    def _show_context_menu(self, position: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        projection = self._applied_projection
        if self._enable_copy_visible_data_action:
            copy_all = menu.addAction("Copy All")
            copy_all.setEnabled(self._copy_projection_is_eligible(projection))
            copy_all.triggered.connect(self._copy_all_to_clipboard)
            copy_rendered = menu.addAction("Copy Rendered Points")
            copy_rendered.setEnabled(
                self._copy_projection_is_eligible(projection)
                and bool(self._rendered_blocks)
            )
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
        annotations.toggled.connect(self.interventionAnnotationsVisibilityRequested)
        menu.addSeparator()
        add_annotation = menu.addAction("Add Annotation…")
        add_annotation.setEnabled(projection is not None)
        add_annotation.triggered.connect(self._add_manual_annotation)
        clear_annotations = menu.addAction("Clear All Annotations")
        clear_annotations.setEnabled(
            bool(projection and self._manual_annotations)
        )
        clear_annotations.triggered.connect(self._clear_manual_annotations)
        state = self._presentation if projection is not None else _PlotPresentation()
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
        presentation = self._presentation
        x_view = (float(x_range[0]) + float(x_range[1])) / 2.0
        y_view = (float(y_range[0]) + float(y_range[1])) / 2.0
        x = self._view_to_raw_coordinate(
            x_view,
            log_enabled=presentation.log_x,
        )
        y = self._view_to_raw_coordinate(
            y_view,
            log_enabled=presentation.log_y,
        )
        if x is None or y is None:
            return
        self._manual_annotations.append((annotation_text, x, y))
        self.apply_workspace_result_projection(projection)

    @QtCore.Slot()
    def _clear_manual_annotations(self) -> None:
        projection = self._applied_projection
        if projection is None:
            return
        if not self._manual_annotations:
            return
        self._manual_annotations.clear()
        self.apply_workspace_result_projection(projection)

    @QtCore.Slot(str)
    def _set_interaction_mode(self, mode: str) -> None:
        normalized = str(mode or "").strip().lower()
        if normalized not in {"pan", "rectangle_zoom"}:
            raise ValueError("Plot interaction mode must be Pan or Rectangle Zoom.")
        projection = self._applied_projection
        if projection is not None:
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
        refresh_geometry: bool = True,
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
        projection = self._applied_projection
        if refresh_geometry and projection is not None:
            self._clear_annotation_items()
            self._render_annotations(projection, state)
            self._render_manual_annotations(state)
            self._render_scalar_guides(projection, state)
        if axis_range is None and projection is not None:
            axis_range = self._axis_range
        if axis_range is not None:
            self._apply_axis_range_state(axis_range, state)

    @QtCore.Slot()
    def _copy_all_to_clipboard(self) -> None:
        projection = self._applied_projection
        if projection is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Copy All",
                "No current result data is available to copy.",
            )
            return
        try:
            WorkspaceResultPresenter.validate_copy(projection)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Copy All", str(exc))
            return
        self._start_responsive_copy(
            projection.copy_all_blocks,
            title="Copy All",
        )

    @QtCore.Slot()
    def _copy_rendered_to_clipboard(self) -> None:
        projection = self._applied_projection
        if projection is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Copy Rendered Points",
                "No current result data is available to copy.",
            )
            return
        try:
            WorkspaceResultPresenter.validate_copy(projection)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Copy Rendered Points",
                str(exc),
            )
            return
        if not self._rendered_blocks:
            return
        self._start_responsive_copy(
            self._rendered_blocks,
            title="Copy Rendered Points",
        )

    @staticmethod
    def _copy_projection_is_eligible(
        projection: WorkspaceResultProjection | None,
    ) -> bool:
        return bool(
            projection is not None
            and projection.copy_all_blocks
            and projection.unavailable_x_diagnostic is None
        )

    def _start_responsive_copy(
        self,
        blocks: tuple[ResultDataBlock, ...],
        *,
        title: str,
    ) -> None:
        self._cancel_copy()
        generation = self._copy_generation
        try:
            self._copy_chunks = iter(
                WorkspaceResultPresenter.iter_blocks_tsv_chunks(
                    tuple(blocks),
                )
            )
            self._copy_output = io.StringIO()
            self._copy_title = str(title)
            self._active_copy_generation = generation
            self._copy_timer.start()
        except Exception as exc:
            if generation == self._copy_generation:
                self._active_copy_generation = None
                self._copy_chunks = None
                self._copy_output = None
            QtWidgets.QMessageBox.critical(
                self,
                str(title),
                str(exc),
            )

    @QtCore.Slot()
    def _continue_copy(self) -> None:
        generation = self._active_copy_generation
        chunks = self._copy_chunks
        output = self._copy_output
        if (
            generation is None
            or generation != self._copy_generation
            or chunks is None
            or output is None
        ):
            self._copy_timer.stop()
            return
        try:
            output.write(next(chunks))
        except StopIteration:
            self._copy_timer.stop()
            text = output.getvalue()
            self._copy_chunks = None
            self._copy_output = None
            self._active_copy_generation = None
            if generation == self._copy_generation:
                QtWidgets.QApplication.clipboard().setText(text)
        except Exception as exc:
            self._copy_timer.stop()
            self._copy_chunks = None
            self._copy_output = None
            self._active_copy_generation = None
            if generation != self._copy_generation:
                return
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
