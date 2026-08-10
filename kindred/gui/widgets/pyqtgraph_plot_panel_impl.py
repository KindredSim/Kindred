"""PyQtGraph workspace surface for one immutable result projection."""

from __future__ import annotations

import io
from dataclasses import dataclass, replace
from collections.abc import Callable, Mapping
from time import monotonic

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.project_presentation_state import (
    AxisSeriesSelection,
    ExperimentalAxisSelection,
    SimulationSeriesAxisSelection,
)
from kindred.core.result_repository import PreviewResultKey
from kindred.gui.plot_config import PlotPresentationIntent, require_pyqtgraph
from kindred.gui.widgets.axis_toolbar import AxisToolbar
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


@dataclass(frozen=True, slots=True)
class _PreviewCurveSnapshot:
    item_key: tuple[object, ...]
    x: np.ndarray
    y: np.ndarray
    opacity: float
    options: Mapping[str, object]


@dataclass(slots=True)
class _PreviewCurveAnimation:
    item: object
    start_x: np.ndarray
    start_y: np.ndarray
    target_x: np.ndarray
    target_y: np.ndarray
    start_opacity: float
    target_opacity: float
    interpolate_coordinates: bool


@dataclass(slots=True)
class _PreviewTransition:
    started_at: float
    animations: tuple[_PreviewCurveAnimation, ...]
    transient_items: tuple[tuple[tuple[object, ...], object], ...]
    auto_range: bool
    start_range: tuple[tuple[float, float], tuple[float, float]] | None
    target_range: tuple[tuple[float, float], tuple[float, float]] | None


class PyQtGraphPlotPanel(QtWidgets.QWidget):
    """Physical plot/analysis widget; semantic visibility stays in presentation."""

    referenceLayerVisibilityRequested = QtCore.Signal(bool)
    interventionAnnotationsVisibilityRequested = QtCore.Signal(bool)
    xSeriesRequested = QtCore.Signal(object)
    ySeriesRequested = QtCore.Signal(object)
    plotPresentationIntentRequested = QtCore.Signal(object)
    resultDiagnostic = QtCore.Signal(str)
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
        super().__init__(parent)
        self._enable_axis_inversion_actions = bool(enable_axis_inversion_actions)
        self._enable_reference_layer_toggle_action = bool(
            enable_reference_layer_toggle_action
        )
        self._enable_copy_visible_data_action = bool(enable_copy_visible_data_action)
        self._applied_projection: WorkspaceResultProjection | None = None
        self._analysis_set_id: str | None = None
        self._rendered_blocks: tuple[ResultDataBlock, ...] = ()
        self._sampling_limit_by_tab: dict[str, int | None] = {}
        self._curve_items: list[object] = []
        self._plot_items_by_key: dict[tuple[object, ...], object] = {}
        self._physical_items_by_axis: dict[
            tuple[AxisSeriesSelection, bool], list[object]
        ] = {}
        self._annotation_items: list[object] = []
        self._manual_annotations_by_tab: dict[str, list[tuple[str, float, float]]] = {}
        self._scalar_guides_by_tab: dict[str, list[str]] = {}
        self._interaction_mode_by_tab: dict[str, str] = {}
        self._presentation_by_tab: dict[str, _PlotPresentation] = {}
        self._axis_range_by_tab: dict[str, _AxisRangeSession] = {}
        self._preview_endpoint_keys: set[PreviewResultKey] = set()
        self._preview_origin_snapshots: tuple[_PreviewCurveSnapshot, ...] = ()
        self._preview_origin_range: (
            tuple[tuple[float, float], tuple[float, float]] | None
        ) = None
        self._preview_origin_auto_range = False
        self._preview_transition: _PreviewTransition | None = None
        self._preview_transition_duration_s = 0.18
        self._preview_transition_timer = QtCore.QTimer(self)
        self._preview_transition_timer.setInterval(16)
        self._preview_transition_timer.timeout.connect(self._advance_preview_transition)
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

    def xlsx_export_projection(self) -> WorkspaceResultProjection | None:
        """Return this page's export authority, or None when unavailable."""

        projection = self._applied_projection
        if projection is None:
            return None
        if projection.copy_all_blocks or projection.unavailable_x_diagnostic:
            return projection
        return None

    @QtCore.Slot(object, bool)
    def capture_preview_transition_origins(
        self,
        set_ids: object,
        animate: bool,
    ) -> None:
        """Capture only visible pixels before an exact Slider admission fence."""

        try:
            targets = frozenset(
                str(set_id).strip()
                for set_id in tuple(set_ids)  # type: ignore[arg-type]
            )
        except TypeError as exc:
            raise TypeError("set_ids must be an iterable of set IDs.") from exc
        if not targets or any(not set_id for set_id in targets):
            raise ValueError("set_ids must contain nonblank set IDs.")
        if not bool(animate):
            self._preview_endpoint_keys.clear()
            self._clear_preview_transition_pixels()
            return
        snapshots = self._snapshot_visible_preview_items(targets)
        self._stop_preview_transition(remove_transient=True)
        self._preview_origin_snapshots = snapshots
        projection = self._applied_projection
        self._preview_origin_auto_range = bool(
            projection is not None
            and self._axis_range_by_tab.get(
                projection.tab_id,
                _AxisRangeSession(),
            ).auto
        )
        self._preview_origin_range = (
            self._current_view_range() if self._preview_origin_auto_range else None
        )

    @QtCore.Slot(object, bool)
    def authorize_preview_endpoint_presentation(
        self,
        keys: object,
        animate: bool,
    ) -> None:
        """Authorize animation for exact keys without changing semantic truth."""

        try:
            captured = tuple(keys)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "keys must be an iterable of PreviewResultKey values."
            ) from exc
        if not captured or any(
            not isinstance(key, PreviewResultKey) for key in captured
        ):
            raise TypeError("keys must contain at least one PreviewResultKey value.")
        tab_ids = {key.tab_id for key in captured}
        if len(tab_ids) != 1:
            raise ValueError("Preview endpoint keys must belong to one tab.")
        projection = self._applied_projection
        if projection is not None and tab_ids != {projection.tab_id}:
            raise ValueError("Preview endpoint keys do not belong to this panel.")
        self._preview_endpoint_keys = set(captured) if bool(animate) else set()
        if not animate:
            self._clear_preview_transition_pixels()

    @QtCore.Slot(object)
    def retire_preview_endpoint_presentations(self, keys: object) -> None:
        """Retire widget-only pixels when their exact owner keys are fenced."""

        try:
            retired = tuple(keys)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "keys must be an iterable of PreviewResultKey values."
            ) from exc
        if any(not isinstance(key, PreviewResultKey) for key in retired):
            raise TypeError("retired keys must be PreviewResultKey values.")
        self._preview_endpoint_keys.difference_update(retired)
        QtCore.QTimer.singleShot(0, self._settle_retired_preview_endpoints)

    @QtCore.Slot()
    def _settle_retired_preview_endpoints(self) -> None:
        # A newer exact admission installs its keys synchronously after retiring
        # the prior endpoint. Only an unclaimed retirement clears held pixels.
        if not self._preview_endpoint_keys:
            self._clear_preview_transition_pixels()

    def apply_workspace_result_projection(
        self,
        projection: WorkspaceResultProjection,
    ) -> None:
        """Apply one projection; failure invalidates this panel's projection."""

        if not isinstance(projection, WorkspaceResultProjection):
            raise TypeError(
                "apply_workspace_result_projection requires WorkspaceResultProjection."
            )
        animated_keys = {
            layer.result_entry.key
            for layer in projection.plot_layers
            if (
                layer.kind is ResultLayerKind.PREVIEW
                and layer.result_entry is not None
                and isinstance(layer.result_entry.key, PreviewResultKey)
                and layer.result_entry.key in self._preview_endpoint_keys
            )
        }
        origin_snapshots = tuple(
            snapshot
            for snapshot in self._preview_origin_snapshots
            if self._preview_item_set_id(snapshot.item_key)
            in {key.set_id for key in animated_keys}
        )
        origin_range = self._preview_origin_range
        origin_auto_range = self._preview_origin_auto_range
        self._stop_preview_transition(remove_transient=True)
        try:
            rendered = self._render_projection(projection)
            if animated_keys:
                self._start_preview_endpoint_transition(
                    origin_snapshots,
                    target_set_ids=frozenset(key.set_id for key in animated_keys),
                    origin_range=origin_range,
                    auto_range=origin_auto_range,
                )
                self._preview_endpoint_keys.difference_update(animated_keys)
                self._preview_origin_snapshots = ()
                self._preview_origin_range = None
                self._preview_origin_auto_range = False
            elif self._preview_origin_snapshots:
                self._install_held_preview_origin_pixels()
        except Exception as exc:
            # A failed projection is ordinary unavailability for this panel:
            # never retain a stale scene or export authority.
            self._clear_rejected_result_scene()
            self.renderFailed.emit(str(exc))
            raise
        self._applied_projection = projection
        self._rendered_blocks = rendered
        if projection.unavailable_x_diagnostic is not None:
            self.resultDiagnostic.emit(projection.unavailable_x_diagnostic)

    @staticmethod
    def _preview_item_set_id(item_key: tuple[object, ...]) -> str | None:
        if not item_key:
            return None
        identity = item_key[0]
        if (
            isinstance(identity, SimulationResultLayerIdentity)
            and identity.kind is ResultLayerKind.PREVIEW
        ):
            return identity.set_id
        return None

    @staticmethod
    def _snapshot_preview_item(
        item_key: tuple[object, ...],
        item: object,
    ) -> _PreviewCurveSnapshot | None:
        get_data = getattr(item, "getData", None)
        if not callable(get_data):
            return None
        x_values, y_values = get_data()
        if x_values is None or y_values is None:
            return None
        x = np.asarray(x_values, dtype=float).copy()
        y = np.asarray(y_values, dtype=float).copy()
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
            return None
        opacity_getter = getattr(item, "opacity", None)
        opacity = float(opacity_getter()) if callable(opacity_getter) else 1.0
        raw_options = getattr(item, "opts", {})
        options = dict(raw_options) if isinstance(raw_options, Mapping) else {}
        return _PreviewCurveSnapshot(
            item_key=item_key,
            x=x,
            y=y,
            opacity=opacity,
            options=options,
        )

    def _snapshot_visible_preview_items(
        self,
        set_ids: frozenset[str],
    ) -> tuple[_PreviewCurveSnapshot, ...]:
        snapshots: list[_PreviewCurveSnapshot] = []
        item_keys: set[tuple[object, ...]] = set()
        for item_key, item in self._plot_items_by_key.items():
            if self._preview_item_set_id(item_key) not in set_ids:
                continue
            snapshot = self._snapshot_preview_item(item_key, item)
            if snapshot is not None:
                snapshots.append(snapshot)
                item_keys.add(item_key)
        transition = self._preview_transition
        if transition is not None:
            for transient_index, (item_key, item) in enumerate(
                transition.transient_items
            ):
                if self._preview_item_set_id(item_key) not in set_ids:
                    continue
                snapshot = self._snapshot_preview_item(item_key, item)
                if snapshot is not None:
                    # A topology-changing crossfade may have both its fading
                    # source and fading-in endpoint under the same semantic
                    # curve key.  Keep both visible pixels for a seamless
                    # mid-transition retarget; the suffix is widget-only.
                    if item_key in item_keys:
                        snapshot = replace(
                            snapshot,
                            item_key=(
                                *item_key,
                                "preview-transition-origin",
                                transient_index,
                            ),
                        )
                    snapshots.append(snapshot)
                    item_keys.add(snapshot.item_key)
        return tuple(snapshots)

    @staticmethod
    def _preview_item_options(snapshot: _PreviewCurveSnapshot) -> dict[str, object]:
        return {
            name: snapshot.options[name]
            for name in (
                "pen",
                "shadowPen",
                "fillLevel",
                "fillBrush",
                "symbol",
                "symbolSize",
                "symbolPen",
                "symbolBrush",
                "connect",
                "stepMode",
            )
            if name in snapshot.options
        }

    def _clone_preview_snapshot(
        self,
        snapshot: _PreviewCurveSnapshot,
    ) -> object:
        item = pg.PlotDataItem(
            snapshot.x,
            snapshot.y,
            **self._preview_item_options(snapshot),
        )
        item.setOpacity(snapshot.opacity)
        self._plot_widget.addItem(item)
        return item

    def _current_view_range(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        x_range, y_range = self._plot_widget.getPlotItem().getViewBox().viewRange()
        return (
            (float(x_range[0]), float(x_range[1])),
            (float(y_range[0]), float(y_range[1])),
        )

    def _set_transition_view_range(
        self,
        value: tuple[tuple[float, float], tuple[float, float]],
    ) -> None:
        view_box = self._plot_widget.getPlotItem().getViewBox()
        view_box.state["autoRange"] = [False, False]
        view_box.setRange(
            xRange=value[0],
            yRange=value[1],
            padding=0.0,
            disableAutoRange=False,
        )
        view_box.state["autoRange"] = [False, False]

    def _install_held_preview_origin_pixels(self) -> None:
        transient = tuple(
            (snapshot.item_key, self._clone_preview_snapshot(snapshot))
            for snapshot in self._preview_origin_snapshots
        )
        self._preview_transition = _PreviewTransition(
            started_at=monotonic(),
            animations=(),
            transient_items=transient,
            auto_range=self._preview_origin_auto_range,
            start_range=self._preview_origin_range,
            target_range=None,
        )
        if self._preview_origin_auto_range and self._preview_origin_range is not None:
            self._set_transition_view_range(self._preview_origin_range)

    def _start_preview_endpoint_transition(
        self,
        origins: tuple[_PreviewCurveSnapshot, ...],
        *,
        target_set_ids: frozenset[str],
        origin_range: tuple[tuple[float, float], tuple[float, float]] | None,
        auto_range: bool,
    ) -> None:
        targets = tuple(
            (item_key, item)
            for item_key, item in self._plot_items_by_key.items()
            if self._preview_item_set_id(item_key) in target_set_ids
        )
        if not targets:
            return
        target_range = self._current_view_range() if auto_range else None
        origin_by_key = {snapshot.item_key: snapshot for snapshot in origins}
        target_snapshots = {
            item_key: self._snapshot_preview_item(item_key, item)
            for item_key, item in targets
        }
        target_keys = {item_key for item_key, _item in targets}
        matching_topology = (
            bool(origins)
            and set(origin_by_key) == target_keys
            and all(
                (target := target_snapshots.get(item_key)) is not None
                and origin_by_key[item_key].x.shape == target.x.shape
                and origin_by_key[item_key].y.shape == target.y.shape
                for item_key in target_keys
            )
        )
        animations: list[_PreviewCurveAnimation] = []
        transient: list[tuple[tuple[object, ...], object]] = []
        if matching_topology:
            for item_key, item in targets:
                origin = origin_by_key[item_key]
                target = target_snapshots[item_key]
                assert target is not None
                item.setData(origin.x, origin.y)
                item.setOpacity(origin.opacity)
                animations.append(
                    _PreviewCurveAnimation(
                        item=item,
                        start_x=origin.x,
                        start_y=origin.y,
                        target_x=target.x,
                        target_y=target.y,
                        start_opacity=origin.opacity,
                        target_opacity=target.opacity,
                        interpolate_coordinates=True,
                    )
                )
        else:
            for origin in origins:
                ghost = self._clone_preview_snapshot(origin)
                transient.append((origin.item_key, ghost))
                animations.append(
                    _PreviewCurveAnimation(
                        item=ghost,
                        start_x=origin.x,
                        start_y=origin.y,
                        target_x=origin.x,
                        target_y=origin.y,
                        start_opacity=origin.opacity,
                        target_opacity=0.0,
                        interpolate_coordinates=False,
                    )
                )
            for item_key, item in targets:
                target = target_snapshots[item_key]
                assert target is not None
                item.setOpacity(0.0)
                animations.append(
                    _PreviewCurveAnimation(
                        item=item,
                        start_x=target.x,
                        start_y=target.y,
                        target_x=target.x,
                        target_y=target.y,
                        start_opacity=0.0,
                        target_opacity=target.opacity,
                        interpolate_coordinates=False,
                    )
                )
        self._preview_transition = _PreviewTransition(
            started_at=monotonic(),
            animations=tuple(animations),
            transient_items=tuple(transient),
            auto_range=bool(auto_range),
            start_range=origin_range,
            target_range=target_range,
        )
        if auto_range and origin_range is not None and target_range is not None:
            self._set_transition_view_range(origin_range)
        self._preview_transition_timer.start()

    @QtCore.Slot()
    def _advance_preview_transition(self) -> None:
        transition = self._preview_transition
        if transition is None or not transition.animations:
            self._preview_transition_timer.stop()
            return
        elapsed = max(0.0, monotonic() - transition.started_at)
        linear = min(1.0, elapsed / self._preview_transition_duration_s)
        progress = linear * linear * (3.0 - 2.0 * linear)
        for animation in transition.animations:
            if animation.interpolate_coordinates:
                animation.item.setData(
                    animation.start_x
                    + (animation.target_x - animation.start_x) * progress,
                    animation.start_y
                    + (animation.target_y - animation.start_y) * progress,
                )
            animation.item.setOpacity(
                animation.start_opacity
                + (animation.target_opacity - animation.start_opacity) * progress
            )
        if (
            transition.auto_range
            and transition.start_range is not None
            and transition.target_range is not None
        ):
            self._set_transition_view_range(
                (
                    tuple(
                        start + (end - start) * progress
                        for start, end in zip(
                            transition.start_range[0],
                            transition.target_range[0],
                        )
                    ),
                    tuple(
                        start + (end - start) * progress
                        for start, end in zip(
                            transition.start_range[1],
                            transition.target_range[1],
                        )
                    ),
                )
            )
        if linear >= 1.0:
            self._finish_preview_transition()

    def _finish_preview_transition(self) -> None:
        transition = self._preview_transition
        if transition is None:
            return
        self._preview_transition_timer.stop()
        for animation in transition.animations:
            if animation.interpolate_coordinates:
                animation.item.setData(animation.target_x, animation.target_y)
            animation.item.setOpacity(animation.target_opacity)
        for _item_key, item in transition.transient_items:
            try:
                self._plot_widget.removeItem(item)
            except (RuntimeError, TypeError, ValueError):
                pass
        if transition.auto_range and transition.target_range is not None:
            self._set_transition_view_range(transition.target_range)
            self._plot_widget.getPlotItem().getViewBox().enableAutoRange()
        self._preview_transition = None

    def _stop_preview_transition(self, *, remove_transient: bool) -> None:
        transition = self._preview_transition
        self._preview_transition_timer.stop()
        if transition is None:
            return
        if remove_transient:
            for _item_key, item in transition.transient_items:
                try:
                    self._plot_widget.removeItem(item)
                except (RuntimeError, TypeError, ValueError):
                    pass
        self._preview_transition = None

    def _clear_preview_transition_pixels(self) -> None:
        self._stop_preview_transition(remove_transient=True)
        self._preview_origin_snapshots = ()
        self._preview_origin_range = None
        self._preview_origin_auto_range = False
        projection = self._applied_projection
        if (
            projection is not None
            and self._axis_range_by_tab.get(
                projection.tab_id,
                _AxisRangeSession(),
            ).auto
        ):
            self._plot_widget.getPlotItem().getViewBox().enableAutoRange()

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
                        provenance.get("model_species")
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
        except Exception as exc:
            self._clear_rejected_result_scene()
            self.renderFailed.emit(str(exc))
            raise

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
        if any(key.tab_id == target for key in self._preview_endpoint_keys) or (
            self._applied_projection is not None
            and self._applied_projection.tab_id == target
        ):
            self._preview_endpoint_keys.clear()
            self._clear_preview_transition_pixels()
        self._manual_annotations_by_tab.pop(target, None)
        self._scalar_guides_by_tab.pop(target, None)
        self._interaction_mode_by_tab.pop(target, None)
        self._presentation_by_tab.pop(target, None)
        self._sampling_limit_by_tab.pop(target, None)
        self._axis_range_by_tab.pop(target, None)

    def reset_plot_session_state(self) -> None:
        """Project replacement/reopen resets non-archived Plot session truth."""

        self._preview_endpoint_keys.clear()
        self._clear_preview_transition_pixels()
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

    def clear_workspace_result_projection(self) -> None:
        """Clear rejected public truth without affecting other tab-local state."""

        self._clear_rejected_result_scene()

    def _clear_rejected_result_scene(self) -> None:
        """Revoke only this physical page's applied result authority."""

        self._cancel_copy()
        self._preview_endpoint_keys.clear()
        self._clear_preview_transition_pixels()
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
        self._analysis_set_id = None

    def _clear_completion_warning(self) -> None:
        self._completion_warning_badge.setText("Completed with warnings")
        self._completion_warning_badge.setToolTip("")
        self._completion_warning_badge.setVisible(False)

    def _render_projection(
        self,
        projection: WorkspaceResultProjection,
    ) -> tuple[ResultDataBlock, ...]:
        existing_items = dict(self._plot_items_by_key)
        self._clear_annotation_items()
        self._plot_widget.setLabel("bottom", projection.x_label)
        self._plot_widget.setLabel("left", projection.y_label)
        block_by_id = {block.block_id: block for block in projection.copy_all_blocks}
        rendered_blocks: list[ResultDataBlock] = []
        rendered_items: dict[tuple[object, ...], object] = {}
        physical_items: dict[tuple[AxisSeriesSelection, bool], list[object]] = {}
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
                sampling_limit=self._sampling_limit_by_tab.get(projection.tab_id),
                existing_items=existing_items,
                rendered_items=rendered_items,
                physical_items=physical_items,
            )
            if rendered is not None:
                rendered_blocks.append(rendered)

        for item_key, item in existing_items.items():
            if item_key in rendered_items:
                continue
            self._remove_plot_item(item)

        self._plot_items_by_key = rendered_items
        self._physical_items_by_axis = physical_items
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
        self._axis_toolbar.set_y_candidates(
            [
                (
                    candidate,
                    projection.selected_y_uses_default_all
                    or candidate.selection in projection.selected_y_series,
                )
                for candidate in y_candidates
            ]
        )
        self._axis_toolbar.set_coarse_sampling(
            self._sampling_limit_by_tab.get(projection.tab_id) is not None
        )
        interaction = self._interaction_mode_by_tab.setdefault(
            projection.tab_id,
            "pan",
        )
        self._axis_toolbar.set_interaction_mode(interaction)
        self._apply_interaction_mode(interaction)
        presentation = self._presentation_by_tab.get(
            projection.tab_id,
            _PlotPresentation(),
        )
        self._render_annotations(projection, presentation)
        self._render_manual_annotations(projection.tab_id, presentation)
        self._render_scalar_guides(projection, presentation)
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
        has_retained_reference = any(
            layer.kind is ResultLayerKind.REFERENCE
            and isinstance(layer.layer_id, SimulationResultLayerIdentity)
            and layer.layer_id.set_id == projection.current_set_id
            for layer in projection.plot_layers
        )
        if (
            projection.primary_layer_id is None
            and has_retained_reference
            and self._analysis_set_id == projection.current_set_id
        ):
            return
        self._parameter_table.update_parameters(
            {name: (value, unit) for name, value, unit in projection.parameters}
        )
        block = projection.primary_statistics
        if block is None:
            self._species_table.setRowCount(0)
            self._analysis_set_id = (
                projection.current_set_id
                if projection.primary_layer_id is not None
                else None
            )
            return
        self._species_table.update_results(
            canonical_time=block.t,
            species_data=dict(block.series),
            intervention_execution=(
                block.result_entry.result.intervention_execution
            ),
            accepted_result_key=(
                getattr(block.result_entry, "key", None)
                if block.result_entry is not None
                else None
            ),
        )
        self._analysis_set_id = projection.current_set_id

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
        details = (
            diagnostic
            if diagnostic is not None
            else projection.completion_warning_tooltip
        )
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
        tab_id: str,
        presentation: _PlotPresentation,
    ) -> None:
        foreground = "#f3f3f3" if self._dark_mode else "#202020"
        background = "#303030cc" if self._dark_mode else "#ffffffe6"
        border = "#bbbbbb" if self._dark_mode else "#555555"
        for text, x, y in self._manual_annotations_by_tab.get(tab_id, ()):
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
        for name in self._scalar_guides_by_tab.get(projection.tab_id, ()):
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
        guides = self._scalar_guides_by_tab.setdefault(
            projection.tab_id,
            [],
        )
        if name not in guides:
            guides.append(name)
        self.apply_workspace_result_projection(projection)

    def _clear_plot_items(self) -> None:
        self._stop_preview_transition(remove_transient=False)
        self._plot_widget.clear()
        self._curve_items.clear()
        self._plot_items_by_key.clear()
        self._physical_items_by_axis.clear()
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
        self._copy_timer.stop()
        self._copy_chunks = None
        self._copy_output = None

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
            self._scalar_guides_by_tab.pop(projection.tab_id, None)
            self.apply_workspace_result_projection(projection)
            return
        if normalized != "sampling":
            return
        if projection is None:
            return
        tab_id = projection.tab_id
        previous_limit = self._sampling_limit_by_tab.get(tab_id)
        self._sampling_limit_by_tab[tab_id] = 1000 if str(value) == "coarse" else None
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
        presentation = self._presentation_by_tab.get(
            projection.tab_id,
            _PlotPresentation(),
        )
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
        self._axis_range_by_tab[projection.tab_id] = state
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
            copy_all.setEnabled(projection is not None)
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
        annotations.toggled.connect(self.interventionAnnotationsVisibilityRequested)
        menu.addSeparator()
        add_annotation = menu.addAction("Add Annotation…")
        add_annotation.setEnabled(projection is not None)
        add_annotation.triggered.connect(self._add_manual_annotation)
        clear_annotations = menu.addAction("Clear All Annotations")
        clear_annotations.setEnabled(
            bool(projection and self._manual_annotations_by_tab.get(projection.tab_id))
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
        presentation = self._presentation_by_tab.get(
            projection.tab_id,
            _PlotPresentation(),
        )
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
            raise ValueError("Plot interaction mode must be Pan or Rectangle Zoom.")
        projection = self._applied_projection
        if projection is not None:
            self._interaction_mode_by_tab[projection.tab_id] = normalized
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
            self._render_manual_annotations(projection.tab_id, state)
            self._render_scalar_guides(projection, state)
        if axis_range is None and projection is not None:
            axis_range = self._axis_range_by_tab.get(projection.tab_id)
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
        if projection is not None:
            try:
                WorkspaceResultPresenter.validate_copy(projection)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Copy Rendered Points",
                    str(exc),
                )
                return
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
