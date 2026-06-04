# kindred/gui/widgets/pyqtgraph_plot_panel_impl.py
"""High-performance plot panel using PyQtGraph (GPU-accelerated)."""

from __future__ import annotations

from functools import partial
import logging
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, NamedTuple

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from kindred.core.datasets.observation_payload import dense_view_from_observations, observations_from_payload
from kindred.gui.color_manager import ColorManager
from kindred.gui.ports import (
    CopyAllDisplayBlock,
    CopyAllExportPlan,
    CopyAllMissingItem,
    PlotDisplayLayer,
    PlotDisplayLayersPayload,
    PlotLayerKind,
)

# Direct imports required to avoid circular dependency with widgets/__init__.py
from kindred.gui.widgets.axis_toolbar import AxisToolbar
from kindred.gui.widgets.dataset_overlay_panel import DatasetOverlayPanel
from kindred.gui.widgets.species_statistics_table import SpeciesStatisticsTable
from kindred.gui.widgets.parameter_statistics_table import ParameterStatisticsTable
from ..ui_helpers import make_pyqtgraph_fallback_widget

logger = logging.getLogger(__name__)

__all__ = [
    "PyQtGraphPlotPanel",
    "PYQTGRAPH_AVAILABLE",
]


def _try_float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _try_1d_float_array(value: object) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def _intervention_annotations_from_provenance(provenance: Mapping[str, object] | None) -> List[Dict[str, object]]:
    if not isinstance(provenance, Mapping):
        return []
    annotations: List[Dict[str, object]] = []
    schedule = provenance.get("intervention_schedule_executable")
    if isinstance(schedule, Mapping):
        for event in list(schedule.get("instant_events") or ()):
            if not isinstance(event, Mapping):
                continue
            time_value = _try_float(event.get("time"))
            if time_value is None:
                continue
            annotations.append(
                {
                    "time": float(time_value),
                    "kind": "instant",
                    "label": f"{event.get('op', 'event')} {event.get('species', '')}".strip(),
                }
            )
        for event in list(schedule.get("repeated_events") or ()):
            if not isinstance(event, Mapping):
                continue
            start = _try_float(event.get("start"))
            every = _try_float(event.get("every"))
            try:
                count = int(event.get("count") or 0)
            except Exception:
                count = 0
            if start is None or every is None or count <= 0:
                continue
            for idx in range(count):
                annotations.append(
                    {
                        "time": float(start + every * float(idx)),
                        "kind": "pulse",
                        "label": f"pulse {event.get('species', '')}".strip(),
                    }
                )
        for interval in list(schedule.get("intervals") or ()):
            if not isinstance(interval, Mapping):
                continue
            start = _try_float(interval.get("start"))
            end = _try_float(interval.get("end"))
            if start is None or end is None or end <= start:
                continue
            annotations.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "kind": str(interval.get("kind") or "interval"),
                    "label": f"{interval.get('kind', 'interval')} {interval.get('species', '')}".strip(),
                }
            )
    for event in list(provenance.get("intervention_trigger_events") or ()):
        if not isinstance(event, Mapping):
            continue
        time_value = _try_float(event.get("time"))
        if time_value is None:
            continue
        annotations.append(
            {
                "time": float(time_value),
                "kind": "trigger",
                "label": f"trigger {event.get('trigger_species', '')}".strip(),
            }
        )
    return sorted(
        annotations,
        key=lambda item: (
            float(item.get("time", item.get("start", 0.0))),
            str(item.get("label") or ""),
        ),
    )


# Try to import pyqtgraph
try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    pg = None


if PYQTGRAPH_AVAILABLE:
    class _OverlaySeries(NamedTuple):
        dataset: str
        species: str
        x: np.ndarray
        y: np.ndarray
        logical_x_axis_name: str
        resolved_x_column: Optional[str]
        resolved_y_column: str

    class _DetailInspectorDock(QtWidgets.QFrame):
        """Compact secondary inspector wrapper with a small default height hint."""

        def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
            super().__init__(parent)
            self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Minimum,
            )

        def sizeHint(self) -> QtCore.QSize:
            hint = super().sizeHint()
            return QtCore.QSize(max(hint.width(), 320), 136)

        def minimumSizeHint(self) -> QtCore.QSize:
            hint = super().minimumSizeHint()
            return QtCore.QSize(max(hint.width(), 260), 120)

    def _resolve_dataset_species(
        species_name: str,
        species_dict: Dict[str, np.ndarray]
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Resolve a mechanism species name to a dataset column name and values.

        This function implements flexible matching to handle common naming patterns
        in experimental datasets (e.g., "A" matching "A_conc").

        Matching rules (applied in order, first match wins):
        1. Exact match: species_name is directly in species_dict
        2. Case-insensitive exact match
        3. Suffix-based match: species_name + common suffixes ("_conc", "_concentration")

        Parameters
        ----------
        species_name : str
            Mechanism species name (e.g., "A", "PBMP")
        species_dict : dict
            Dataset species columns mapping names to concentration arrays

        Returns
        -------
        tuple
            (resolved_key, values) if match found, (None, None) otherwise
        """
        if not species_name or not species_dict:
            return None, None

        # Rule 1: Exact match
        if species_name in species_dict:
            return species_name, species_dict[species_name]

        # Build lookup for case-insensitive matching
        lower_to_key = {key.lower(): key for key in species_dict.keys()}
        species_lower = species_name.lower()

        # Rule 2: Case-insensitive exact match
        if species_lower in lower_to_key:
            matched_key = lower_to_key[species_lower]
            return matched_key, species_dict[matched_key]

        # Rule 3: Suffix-based matching (common concentration column patterns)
        suffixes = ["_conc", "_concentration"]
        for suffix in suffixes:
            candidate_lower = species_lower + suffix
            if candidate_lower in lower_to_key:
                matched_key = lower_to_key[candidate_lower]
                return matched_key, species_dict[matched_key]

        # No match found
        return None, None

    class PyQtGraphPlotPanel(QtWidgets.QWidget):
        """
        High-performance plot panel using PyQtGraph.

        Features:
        - GPU-accelerated rendering (OpenGL)
        - Handles millions of points smoothly
        - Real-time updates (60+ FPS)
        - Interactive zoom, pan, mouse tracking
        - AxisToolbar integration (X-axis selection, Y-axis selection, parametric mode)
        - Custom X-axis selection (plot any species vs any species)
        - Parametric mode support
        - Legend with show/hide toggle
        - Dark theme support

        Performance:
        - Handles 1M+ points smoothly with GPU acceleration
        """

        # Signal emitted when series visibility changes
        seriesVisibilityChanged = QtCore.Signal(str, bool)
        referenceLayerVisibilityRequested = QtCore.Signal(bool)

        def __init__(
            self,
            parent: Optional[QtWidgets.QWidget] = None,
            *,
            embed_analysis_tabs: bool = True,
            workspace_splitter_object_name: Optional[str] = None,
            enable_axis_inversion_actions: bool = False,
            enable_reference_layer_toggle_action: bool = False,
            enable_copy_visible_data_action: bool = False,
        ):
            """
            Initialize PyQtGraph plot panel.

            Parameters
            ----------
            parent : QWidget, optional
                Parent widget
            """
            super().__init__(parent)

            # Data storage
            self._t: Optional[np.ndarray] = None
            self._series: Dict[str, np.ndarray] = {}
            self._visible: Dict[str, bool] = {}
            self._colors: Dict[str, tuple] = {}
            self._owned_species_keys: Set[str] = set()
            self._owned_species_roster_explicit: bool = False
            self._plot_items: Dict[str, pg.PlotDataItem] = {}
            self._plot_item_signatures: Dict[str, tuple[object, ...]] = {}
            self._dataset_scatter_items: Dict[str, pg.ScatterPlotItem] = {}
            self._dataset_model_items: Dict[str, pg.PlotDataItem] = {}
            self._overlay_items: Dict[Tuple[str, str], pg.ScatterPlotItem] = {}
            self._overlay_datasets: Dict[str, Dict[str, np.ndarray]] = {}
            self._overlay_symbols: Dict[str, str] = {}
            self._active_overlay_series: List[_OverlaySeries] = []
            self._visible_overlay_series: List[_OverlaySeries] = []
            self._export_all_overlay_series_unfiltered: List[_OverlaySeries] = []
            self._export_all_overlay_series: List[_OverlaySeries] = []
            self._active_overlay_warnings: List[str] = []
            self._visible_overlay_warnings: List[str] = []
            self._export_all_overlay_warnings_unfiltered: List[str] = []
            self._export_all_overlay_warnings: List[str] = []
            self._export_all_overlay_cache_dirty: bool = True
            self._dark_mode = False
            self._scalar_values: Dict[str, float] = {}
            self._y_selection_user_touched: bool = False
            self._preserved_y_selection_visibility: Dict[str, bool] = {}
            self._auto_range_enabled: bool = True
            self._manual_range_values: tuple[
                Optional[float],
                Optional[float],
                Optional[float],
                Optional[float],
            ] = (None, None, None, None)

            # Batch simulation overlays (multiple initial-condition sets overlaid as lines)
            self._simulation_set_label: Optional[str] = None
            self._simulation_set_id: Optional[str] = None
            self._simulation_layer_id: Optional[str] = None
            self._simulation_overlays: List[PlotDisplayLayer] = []
            self._reference_layers_hydratable: bool = False

            # Axis control (for parametric mode and custom X-axis)
            self._x_axis_name: str = "t"  # Current X-axis variable
            self._parametric_mode: bool = False  # Parametric plotting mode

            # Plot enhancements (v0.2.0)
            self._log_x: bool = False
            self._log_y: bool = False
            self._invert_x_axis: bool = False
            self._invert_y_axis: bool = False
            self._enable_axis_inversion_actions = bool(enable_axis_inversion_actions)
            self._enable_reference_layer_toggle_action = bool(enable_reference_layer_toggle_action)
            self._enable_copy_visible_data_action = bool(enable_copy_visible_data_action)
            self._copy_all_export_plan_provider: Optional[Callable[[], Optional[CopyAllExportPlan]]] = None
            self._copy_status_text_callback: Optional[Callable[[str], None]] = None
            self._annotations: List[pg.TextItem] = []
            self._intervention_annotations: List[Dict[str, object]] = []
            self._intervention_annotation_items: List[object] = []
            self._intervention_annotation_signature: tuple[object, ...] | None = None
            self._show_intervention_annotations: bool = False
            self._sampling_mode: str = "dense"
            self._sampling_target: int = 1000
            self._export_scope_preference: str = "axis"
            self._guide_items: List[pg.InfiniteLine] = []
            self._analysis_tabs_detached = False

            # Setup UI
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            # Create PyQtGraph PlotWidget
            self._plot_widget = pg.PlotWidget()
            self._plot_widget.setObjectName("plotViewport")
            self._plot_item = self._plot_widget.getPlotItem()
            self._plot_widget.plotItem.setContentsMargins(0, 0, 0, 10)

            # Set white background (user requested light background)
            self._plot_widget.setBackground('w')

            # Configure plot appearance
            self._plot_item.setLabel('bottom', 'Time', units='s')
            self._plot_item.setLabel('left', 'Concentration', units='M')
            self._plot_item.showGrid(x=True, y=True, alpha=0.3)
            self._legend = self._plot_item.addLegend()
            self._legend_visible = True

            # Enable antialiasing for smoother lines
            self._plot_widget.setAntialiasing(True)

            # Setup context menu
            self._plot_widget.setContextMenuPolicy(Qt.CustomContextMenu)
            self._plot_widget.customContextMenuRequested.connect(self._show_context_menu)

            # Disable PyQtGraph's native context menus (prevents double-popups in some builds)
            self._plot_item.setMenuEnabled(False)
            vb = self._plot_item.getViewBox()
            vb.setMenuEnabled(False)
            vb.sigRangeChangedManually.connect(self._on_view_range_changed_manually)
            self._apply_axis_inversion_state()

            self._toolbar = AxisToolbar(self, orientation="horizontal")
            self._toolbar.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )

            self._legend_toggle_btn = QtWidgets.QCheckBox("Legend", self._toolbar)
            self._legend_toggle_btn.setChecked(True)
            self._legend_toggle_btn.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self._legend_toggle_btn.toggled.connect(self._toggle_legend)
            toolbar_layout = self._toolbar.layout()
            toolbar_layout.insertWidget(toolbar_layout.count() - 1, self._legend_toggle_btn, 0, Qt.AlignVCenter)

            self._control_strip = QtWidgets.QWidget(self)
            control_layout = QtWidgets.QHBoxLayout(self._control_strip)
            control_layout.setContentsMargins(8, 6, 8, 2)
            control_layout.setSpacing(6)
            control_layout.addWidget(self._toolbar, stretch=1)
            layout.addWidget(self._control_strip, stretch=0)

            self._plot_surface = QtWidgets.QFrame(self)
            self._plot_surface.setObjectName("mainPlotSurface")
            self._plot_surface.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            plot_surface_layout = QtWidgets.QVBoxLayout(self._plot_surface)
            plot_surface_layout.setContentsMargins(8, 0, 8, 0)
            plot_surface_layout.setSpacing(0)
            self._plot_widget.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            plot_surface_layout.addWidget(self._plot_widget, stretch=1)

            # Vertical splitter: plot area (top) | compact detail dock (bottom)
            self._main_splitter = QtWidgets.QSplitter(Qt.Vertical, self)
            if workspace_splitter_object_name:
                self._main_splitter.setObjectName(str(workspace_splitter_object_name))
            self._main_splitter.setChildrenCollapsible(False)
            self._main_splitter.setHandleWidth(8)
            self._main_splitter.addWidget(self._plot_surface)
            self._main_splitter.setCollapsible(0, False)

            self._details_dock = _DetailInspectorDock(self._main_splitter)
            details_dock_layout = QtWidgets.QVBoxLayout(self._details_dock)
            details_dock_layout.setContentsMargins(0, 0, 0, 0)
            details_dock_layout.setSpacing(0)

            self._details_tabs = QtWidgets.QTabWidget(self._details_dock)
            self._details_tabs.setObjectName("mainPlotDetailTabs")
            self._details_tabs.setDocumentMode(True)
            self._details_tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
            details_dock_layout.addWidget(self._details_tabs, stretch=1)

            stats_container = QtWidgets.QWidget(self._details_tabs)
            stats_layout = QtWidgets.QVBoxLayout(stats_container)
            stats_layout.setContentsMargins(8, 8, 8, 8)
            stats_layout.setSpacing(4)

            stats_selector_row = QtWidgets.QHBoxLayout()
            stats_selector_row.setContentsMargins(0, 0, 0, 0)
            stats_selector_row.setSpacing(6)

            stats_selector_label = QtWidgets.QLabel("Select Result Set:", stats_container)
            stats_selector_row.addWidget(stats_selector_label, stretch=0)

            self._stats_result_selector = QtWidgets.QComboBox(stats_container)
            self._stats_result_selector.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            stats_selector_row.addWidget(self._stats_result_selector, stretch=1)
            stats_selector_row.addStretch(1)

            stats_layout.addLayout(stats_selector_row, stretch=0)

            self._stats_table = SpeciesStatisticsTable(stats_container)
            stats_layout.addWidget(self._stats_table, stretch=1)

            self._stats_results_map: Dict[str, Dict[str, object]] = {}
            self._stats_result_selector.currentTextChanged.connect(self._on_stats_result_selector_changed)

            self._param_table = ParameterStatisticsTable(self._details_tabs)
            self._param_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

            self._overlay_panel = DatasetOverlayPanel(self._details_tabs)
            self._overlay_panel.selectionChanged.connect(self._on_overlay_selection_changed)
            self._overlay_panel.styleChanged.connect(self._on_overlay_style_changed)

            self._details_tabs.addTab(stats_container, "Statistics")
            self._details_tabs.addTab(self._param_table, "Parameters")
            self._details_tabs.setCurrentWidget(stats_container)

            self._main_splitter.addWidget(self._details_dock)
            self._main_splitter.setCollapsible(1, True)
            self._main_splitter.setStretchFactor(0, 5)
            self._main_splitter.setStretchFactor(1, 1)
            layout.addWidget(self._main_splitter, stretch=1)
            QtCore.QTimer.singleShot(0, self._apply_default_workspace_splitter_sizes)

            if not bool(embed_analysis_tabs):
                self.detach_analysis_tabs_for_dock()

            # Connect toolbar signals
            self._toolbar.xChanged.connect(self._on_x_axis_changed)
            self._toolbar.ySelectionChanged.connect(self._on_y_selection_changed)
            self._toolbar.parametricToggled.connect(self._on_parametric_toggled)
            self._toolbar.axisRangeChanged.connect(self._on_axis_range_changed)
            self._toolbar.optionsRequested.connect(self._on_toolbar_option_requested)
            self._toolbar.addGuideRequested.connect(self._on_add_guide_requested)
            logger.debug("Connected axis range changed signal")

            logger.debug("PyQtGraphPlotPanel initialized")

        def _apply_default_workspace_splitter_sizes(self) -> None:
            splitter = getattr(self, "_main_splitter", None)
            if splitter is None:
                return
            total = sum(int(size) for size in splitter.sizes())
            if total <= 0:
                total = max(int(self.height()), 720)
            if bool(getattr(self, "_analysis_tabs_detached", False)):
                splitter.setSizes([max(1, total), 0])
                return
            detail = max(120, int(round(total * 0.16)))
            detail = min(detail, max(140, int(total * 0.18)))
            # Ensure the plot surface keeps at least 200px
            if total - detail < 200:
                detail = max(0, total - 200)
            plot = max(1, total - detail)
            splitter.setSizes([plot, detail])

        def analysis_tabs_widget(self) -> Optional[QtWidgets.QWidget]:
            return getattr(self, "_details_tabs", None)

        def workspace_splitter(self) -> Optional[QtWidgets.QSplitter]:
            return getattr(self, "_main_splitter", None)

        def detach_analysis_tabs_for_dock(self) -> Optional[QtWidgets.QWidget]:
            tabs = getattr(self, "_details_tabs", None)
            if tabs is None:
                return None
            if bool(getattr(self, "_analysis_tabs_detached", False)):
                return tabs

            details_dock = getattr(self, "_details_dock", None)
            if details_dock is not None:
                layout = details_dock.layout()
                if layout is not None:
                    layout.removeWidget(tabs)
                tabs.setParent(None)
                details_dock.hide()
                details_dock.setMinimumHeight(0)
                details_dock.setMaximumHeight(0)

            splitter = getattr(self, "_main_splitter", None)
            if splitter is not None:
                splitter.setSizes([max(1, sum(int(size) for size in splitter.sizes())), 0])

            self._analysis_tabs_detached = True
            return tabs

        def set_data(
            self,
            t: np.ndarray,
            series: Dict[str, np.ndarray],
            *,
            label: Optional[str] = None,
            primary_set_id: Optional[str] = None,
            layer_id: Optional[str] = None,
            owned_species: Optional[Sequence[str]] = None,
        ) -> None:
            self._set_primary_simulation_layer(
                t=t,
                series=series,
                label=label,
                primary_set_id=primary_set_id,
                layer_id=layer_id,
                owned_species=owned_species,
            )
            self._simulation_overlays = []
            self._intervention_annotations = []
            self._show_intervention_annotations = False
            self._refresh_simulation_display()
            logger.debug(f"Data set: {len(self._t)} points, {len(self._series)} series")

        def set_display_layers(self, payload: PlotDisplayLayersPayload) -> None:
            if not isinstance(payload, PlotDisplayLayersPayload):
                raise TypeError("set_display_layers requires a PlotDisplayLayersPayload")
            self._reference_layers_hydratable = bool(payload.reference_layers_hydratable)
            primary_layer = self._primary_display_layer_from_payload(payload)
            if primary_layer is None:
                self.clear_display_transaction_state()
                return
            self._set_primary_simulation_layer(
                t=primary_layer.x,
                series=primary_layer.y,
                label=primary_layer.label,
                primary_set_id=primary_layer.source_id,
                layer_id=primary_layer.layer_id,
                owned_species=self._plot_layer_color_domain(primary_layer),
            )
            primary_layer_id = str(primary_layer.layer_id or "").strip()
            self._simulation_overlays = [
                layer
                for layer in payload.layers
                if layer.kind is not PlotLayerKind.PRIMARY_SERIES
                and str(layer.layer_id or "").strip() != primary_layer_id
            ]
            self._intervention_annotations = [
                dict(annotation)
                for annotation in payload.intervention_annotations
                if isinstance(annotation, Mapping)
            ]
            self._show_intervention_annotations = bool(payload.show_intervention_annotations)
            self._refresh_simulation_display()

        def request_reference_layers_visible(self, visible: bool) -> None:
            self.referenceLayerVisibilityRequested.emit(bool(visible))

        @staticmethod
        def _primary_display_layer_from_payload(
            payload: PlotDisplayLayersPayload,
        ) -> Optional[PlotDisplayLayer]:
            primary_layer_id = str(payload.primary_layer_id or "").strip()
            if primary_layer_id:
                for layer in payload.layers:
                    if str(layer.layer_id or "").strip() == primary_layer_id:
                        return layer
            for layer in payload.layers:
                if layer.kind is PlotLayerKind.PRIMARY_SERIES:
                    return layer
            return None

        @staticmethod
        def _plot_layer_color_domain(layer: PlotDisplayLayer) -> Tuple[str, ...]:
            metadata = layer.style_metadata if isinstance(layer.style_metadata, Mapping) else {}
            raw_domain = metadata.get("color_domain") if isinstance(metadata, Mapping) else ()
            return tuple(str(name).strip() for name in (raw_domain or ()) if str(name).strip())

        def _set_primary_simulation_layer(
            self,
            *,
            t: object,
            series: Mapping[str, object],
            label: Optional[str],
            primary_set_id: Optional[str],
            layer_id: Optional[str],
            owned_species: Optional[Sequence[str]],
        ) -> None:
            preserve_y_selection = bool(self._y_selection_user_touched)
            previous_visibility = dict(self._visible)
            previous_series_names = set(self._series.keys())
            if preserve_y_selection and not previous_series_names and self._preserved_y_selection_visibility:
                previous_visibility = dict(self._preserved_y_selection_visibility)
                previous_series_names = set(previous_visibility.keys())
            self._t = np.asarray(t, dtype=float).reshape(-1)
            self._series = {str(k): np.asarray(v, dtype=float).reshape(-1) for k, v in dict(series or {}).items()}
            new_series_names = set(self._series.keys())
            roster_overlap = previous_series_names & new_series_names
            previous_selection_was_empty = not any(
                bool(previous_visibility.get(name, False))
                for name in previous_series_names
            )
            if preserve_y_selection and roster_overlap:
                if previous_selection_was_empty:
                    self._visible = {k: False for k in self._series.keys()}
                else:
                    self._visible = {
                        k: bool(previous_visibility.get(k, True)) if k in previous_series_names else True
                        for k in self._series.keys()
                    }
            else:
                self._visible = {k: True for k in self._series.keys()}
                if preserve_y_selection:
                    self._y_selection_user_touched = False
            self._preserved_y_selection_visibility = {}
            color_manager = ColorManager.instance()
            provided_owned = {str(name).strip() for name in (owned_species or []) if str(name).strip()}
            series_keys = {str(name).strip() for name in self._series.keys() if str(name).strip()}
            if provided_owned:
                self._owned_species_keys = set(provided_owned)
                self._owned_species_roster_explicit = True
                color_manager.seed_species(sorted(self._owned_species_keys))
            else:
                self._owned_species_keys = set(series_keys)
                self._owned_species_roster_explicit = False
                if self._owned_species_keys:
                    color_manager.seed_species(sorted(self._owned_species_keys))

            self._simulation_set_label = str(label) if label else None
            self._simulation_set_id = str(primary_set_id or "").strip() or None
            self._simulation_layer_id = str(layer_id or "").strip() or (
                f"result:{self._simulation_set_id}" if self._simulation_set_id else "result:live"
            )

        def _refresh_simulation_display(self) -> None:
            self._assign_colors()
            known_species = self._active_overlay_known_species()
            self._overlay_panel.refresh_color_swatches(known_species=known_species or None)

            self._update_toolbar()
            self._update_plot()

        @staticmethod
        def _overlay_item_key(*, layer_id: str, species: str) -> str:
            return f"{str(layer_id)}:{str(species)}"

        @staticmethod
        def _is_reference_layer(layer: PlotDisplayLayer) -> bool:
            return isinstance(layer, PlotDisplayLayer) and layer.kind is PlotLayerKind.REFERENCE_SERIES

        @staticmethod
        def _overlay_layer_id(layer: PlotDisplayLayer) -> str:
            layer_id = str(layer.layer_id or "").strip()
            if layer_id:
                return layer_id
            source_id = str(layer.source_id or "").strip()
            return f"{layer.kind.value}:{source_id}" if source_id else layer.kind.value

        @staticmethod
        def _overlay_y_map(layer: PlotDisplayLayer) -> Dict[str, object]:
            return {str(name): values for name, values in dict(layer.y or {}).items() if str(name)}

        @staticmethod
        def _overlay_x_values(layer: PlotDisplayLayer) -> object:
            return layer.x

        def _overlay_display_label(self, layer: PlotDisplayLayer) -> str:
            return str(layer.label or layer.source_id or layer.layer_id or "").strip()

        def display_layer_snapshot(self) -> Dict[str, object]:
            layers: List[Dict[str, object]] = []
            if self._t is not None and self._series:
                primary_layer_id = str(self._simulation_layer_id or "").strip() or "result:live"
                item_identities = {}
                for species in self._series:
                    item_key = self._overlay_item_key(layer_id=primary_layer_id, species=str(species))
                    item = self._plot_items.get(item_key)
                    if item is not None:
                        item_identities[item_key] = id(item)
                layers.append(
                    {
                        "layer_id": primary_layer_id,
                        "kind": PlotLayerKind.PRIMARY_SERIES.value,
                        "label": str(self._simulation_set_label or "Results"),
                        "source_id": str(self._simulation_set_id or ""),
                        "visible": True,
                        "item_identities": item_identities,
                    }
                )
            for layer in list(self._simulation_overlays or []):
                if not isinstance(layer, PlotDisplayLayer):
                    continue
                layer_id = self._overlay_layer_id(layer)
                series_map = self._overlay_y_map(layer)
                item_identities = {}
                for species in series_map:
                    item_key = self._overlay_item_key(layer_id=layer_id, species=str(species))
                    item = self._plot_items.get(item_key)
                    if item is not None:
                        item_identities[item_key] = id(item)
                layers.append(
                    {
                        "layer_id": layer_id,
                        "kind": layer.kind.value,
                        "label": self._overlay_display_label(layer),
                        "source_id": str(layer.source_id or ""),
                        "visible": bool(layer.visible),
                        "item_identities": item_identities,
                    }
                )
            if self._intervention_annotations:
                layers.append(
                    {
                        "layer_id": "annotations:interventions",
                        "kind": "annotation",
                        "label": "Interventions",
                        "source_id": "",
                        "visible": bool(self._show_intervention_annotations),
                        "item_identities": {},
                    }
                )
            return {"layers": layers}

        def clear_display_transaction_state(self, *, preserve_y_selection_state: bool = False) -> None:
            preserved_visibility = (
                dict(self._visible)
                if bool(preserve_y_selection_state) and bool(self._y_selection_user_touched)
                else {}
            )
            preserved_touched = bool(preserve_y_selection_state) and bool(self._y_selection_user_touched)
            self.clear()
            if preserved_touched:
                self._y_selection_user_touched = True
                self._preserved_y_selection_visibility = preserved_visibility
            self.set_statistics_results({}, prefer="")

        def has_display_data(self) -> bool:
            return bool(self._series) and self._t is not None

        def display_owned_species(self) -> Optional[Tuple[str, ...]]:
            if not bool(getattr(self, "_owned_species_roster_explicit", False)):
                return None
            owned_species = tuple(
                sorted(str(name) for name in (self._owned_species_keys or set()) if str(name))
            )
            return owned_species or None

        def render_dataset_layers(
            self,
            *,
            data_t: object,
            dataset_series: Dict[str, np.ndarray],
            model_t: object = None,
            model_series: Optional[Dict[str, np.ndarray]] = None,
            visible_species: Sequence[str],
            xlabel: str,
            ylabel: str,
        ) -> None:
            """Render dataset-tab scatter/model layers without clearing unrelated backend state."""
            series_map = {
                str(name): _try_1d_float_array(values)
                for name, values in (dataset_series or {}).items()
                if str(name)
            }
            active_dataset = {name: values for name, values in series_map.items() if values.size}
            visible = {str(name) for name in (visible_species or []) if str(name)}

            model_map = {
                str(name): _try_1d_float_array(values)
                for name, values in (model_series or {}).items()
                if str(name)
            }
            if isinstance(data_t, dict):
                data_t_map = {
                    str(name): _try_1d_float_array(values)
                    for name, values in data_t.items()
                    if str(name)
                }
                t_arr = np.asarray([], dtype=float)
            else:
                data_t_map = {}
                t_arr = _try_1d_float_array(data_t)
            if isinstance(model_t, dict):
                model_t_map = {
                    str(name): _try_1d_float_array(values)
                    for name, values in model_t.items()
                    if str(name)
                }
                model_t_arr = np.asarray([], dtype=float)
            else:
                model_t_map = {}
                model_t_arr = _try_1d_float_array(model_t) if model_t is not None else np.asarray([], dtype=float)

            self._set_dataset_axis_labels(xlabel=xlabel, ylabel=ylabel)

            if (t_arr.size == 0 and not data_t_map) or not active_dataset:
                self._prune_dataset_scatter_items(set())
                self._prune_dataset_model_items(set())
                return

            color_manager = ColorManager.instance()
            color_manager.seed_species(active_dataset.keys())

            active_scatter_keys: Set[str] = set()
            active_model_keys: Set[str] = set()
            for species_name, y_data in active_dataset.items():
                species_t = data_t_map.get(species_name, t_arr)
                if y_data.shape[0] != species_t.shape[0]:
                    logger.warning(
                        "Dataset layer length mismatch for %s: %s vs %s",
                        species_name,
                        int(y_data.shape[0]),
                        int(species_t.shape[0]),
                    )
                    continue

                scatter_color = color_manager.get_species_rgb(species_name, known_species=tuple(active_dataset.keys()))
                brush = pg.mkBrush(*scatter_color, 150)
                active_scatter_keys.add(species_name)
                self._upsert_dataset_scatter_item(
                    key=species_name,
                    x_data=species_t,
                    y_data=y_data,
                    brush=brush,
                    size=8,
                    name=f"{species_name} (data)",
                )
                self._dataset_scatter_items[species_name].setVisible(species_name in visible)

                model_values = model_map.get(species_name)
                species_model_t = model_t_map.get(species_name, model_t_arr)
                if species_model_t.size == 0 or model_values is None:
                    continue
                if model_values.shape[0] != species_model_t.shape[0]:
                    logger.warning(
                        "Dataset model length mismatch for %s: %s vs %s",
                        species_name,
                        int(model_values.shape[0]),
                        int(species_model_t.shape[0]),
                    )
                    continue

                pen = pg.mkPen(color=scatter_color, width=2)
                active_model_keys.add(species_name)
                self._upsert_dataset_model_item(
                    key=species_name,
                    x_data=species_model_t,
                    y_data=model_values,
                    pen=pen,
                    name=f"{species_name} (model)",
                )
                self._dataset_model_items[species_name].setVisible(species_name in visible)

            self._prune_dataset_scatter_items(active_scatter_keys)
            self._prune_dataset_model_items(active_model_keys)

        def stats_table(self) -> SpeciesStatisticsTable:
            """Return the species statistics table widget."""
            return self._stats_table

        def parameter_table(self) -> ParameterStatisticsTable:
            """Return the solver-parameter table widget."""
            return self._param_table

        def overlay_panel(self) -> DatasetOverlayPanel:
            """Return the dataset overlay panel widget."""
            return self._overlay_panel

        def set_scalar_values(self, scalars: Dict[str, float]) -> None:
            """Store algebra scalar outputs for guide selection."""
            cleaned: Dict[str, float] = {}
            for name, value in (scalars or {}).items():
                val = _try_float(value)
                if val is None:
                    continue
                cleaned[str(name)] = float(val)
            self._scalar_values = cleaned
            if self._t is not None and self._series:
                self._update_toolbar()

        def selected_series(self) -> List[str]:
            """Return the currently selected Y-series names."""
            return [name for name in self._series.keys() if self._visible.get(name, False)]

        def get_export_scope_preference(self) -> str:
            """Return the preferred export scope for CSV dialogs."""
            return self._export_scope_preference

        def transaction_export_axis_state(self, scope: str) -> Dict[str, object]:
            """Return export presentation state without exposing simulation data authority."""
            normalized_scope = str(scope or "axis")
            if normalized_scope == "axis":
                y_names = self._axis_scope_series_names()
                if not y_names:
                    raise ValueError("Select at least one Y-series before exporting.")
            else:
                y_names = list(self._series.keys())
            x_name = self._x_axis_name or "t"
            _x_data, derived_label = self._get_x_data()
            return {
                "x_name": str(x_name),
                "x_header": str(derived_label or x_name),
                "y_names": tuple(str(name) for name in y_names if str(name)),
            }

        def _get_clipboard(self):
            """Clipboard accessor seam (monkeypatchable in tests)."""
            try:
                app = QtWidgets.QApplication.instance()
                return app.clipboard() if app is not None else None
            except Exception:
                return None

        def export_payload(self) -> Optional[Dict[str, object]]:
            """
            Return a standardized export payload for CSV export code.

            Returns
            -------
            dict or None
                {'t': np.ndarray, 'series': Dict[str, np.ndarray]}
            """
            if self._copy_all_export_plan_provider is not None:
                return None
            if self._t is None or not self._series:
                return None
            layers = [
                dict(layer)
                for layer in self.display_layer_snapshot().get("layers", [])
                if not (
                    isinstance(layer, Mapping)
                    and str(layer.get("kind") or "") == PlotLayerKind.REFERENCE_SERIES.value
                    and layer.get("visible") is False
                )
            ]
            overlays = [
                {
                    "layer_id": self._overlay_layer_id(layer),
                    "kind": layer.kind.value,
                    "label": self._overlay_display_label(layer),
                    "source_id": str(layer.source_id or ""),
                    "visible": bool(layer.visible),
                }
                for layer in list(self._simulation_overlays or [])
                if isinstance(layer, PlotDisplayLayer)
                and bool(layer.visible)
            ]
            return {
                "t": np.asarray(self._t, dtype=float).reshape(-1),
                "series": dict(self._series),
                "layers": layers,
                "overlays": overlays,
                "intervention_annotations": [dict(item) for item in self._intervention_annotations],
            }

        def set_intervention_annotations_from_provenance(self, provenance: Mapping[str, object] | None) -> None:
            self._intervention_annotations = _intervention_annotations_from_provenance(provenance)
            self._refresh_intervention_annotations()

        def set_intervention_annotations_visible(self, visible: bool) -> None:
            self._show_intervention_annotations = bool(visible)
            self._refresh_intervention_annotations()

        def intervention_annotation_state(self) -> Dict[str, object]:
            return {
                "intervention_annotations": [
                    dict(item)
                    for item in list(self._intervention_annotations or [])
                    if isinstance(item, Mapping)
                ],
                "show_intervention_annotations": bool(self._show_intervention_annotations),
            }

        def set_selected_series(self, names: Sequence[str]) -> None:
            """Apply a specific selection of Y-series."""
            valid = [str(n) for n in names if str(n) in self._series]
            target = set(valid)
            self._y_selection_user_touched = True
            for series_name in self._series.keys():
                self._visible[series_name] = series_name in target
            self._project_y_selection_to_toolbar()
            self._update_plot()

        def _series_names_compatible_with_x(
            self,
            names: Sequence[str],
            series_map: Dict[str, object],
            x_array: np.ndarray,
            *,
            require_visible: bool,
        ) -> List[str]:
            compatible: List[str] = []
            seen: Set[str] = set()
            for raw_name in names:
                name = str(raw_name)
                if name in seen:
                    continue
                seen.add(name)
                if name not in series_map:
                    continue
                if require_visible and not self._visible.get(name, True):
                    continue
                y_array = _try_1d_float_array(series_map.get(name))
                if y_array.size == 0 or y_array.shape[0] != x_array.shape[0]:
                    continue
                compatible.append(name)
            return compatible

        @staticmethod
        def _simulation_overlay_owned_species(layer: PlotDisplayLayer) -> Tuple[str, ...]:
            style_metadata = layer.style_metadata if isinstance(layer.style_metadata, Mapping) else {}
            color_domain = style_metadata.get("color_domain") if isinstance(style_metadata, Mapping) else ()
            return tuple(
                str(name).strip()
                for name in (color_domain or ())
                if str(name).strip()
            )

        @staticmethod
        def _simulation_overlay_candidate_series_names(
            layer: PlotDisplayLayer,
            fallback_names: Sequence[str],
        ) -> List[str]:
            fallback = []
            seen: Set[str] = set()
            for raw_name in fallback_names or ():
                name = str(raw_name)
                if not name or name in seen:
                    continue
                seen.add(name)
                fallback.append(name)
            y_series = tuple(
                str(name).strip()
                for name in (layer.y_series or ())
                if str(name).strip()
            )
            if y_series:
                display_set = set(y_series)
                visible_display = [name for name in fallback if name in display_set]
                if visible_display:
                    return visible_display
                return []
            return []

        def _visible_selected_series_names(self) -> List[str]:
            return [name for name in self._series.keys() if self._visible.get(name, False)]

        def _project_y_selection_to_toolbar(self) -> None:
            self._toolbar.select_y(self._visible_selected_series_names())

        def _current_primary_renderable_series_names(
            self,
            names: Sequence[str],
            *,
            require_visible: bool,
        ) -> List[str]:
            primary_basis = self._current_primary_plot_basis()
            if primary_basis is None:
                return []
            candidate_names: List[str] = []
            seen: Set[str] = set()
            for raw_name in names or ():
                name = str(raw_name)
                if not name or name in seen:
                    continue
                seen.add(name)
                candidate_names.append(name)
            return self._series_names_compatible_with_x(
                candidate_names,
                self._series,
                primary_basis[2],
                require_visible=require_visible,
            )

        def _axis_scope_series_names(self) -> List[str]:
            return list(self._visible_selected_series_names())

        def _series_header_text(self, series_name: str) -> str:
            return str(series_name)

        def _visible_overlay_copy_series_names(self) -> List[str]:
            return self._axis_scope_series_names()

        def _visible_primary_copy_series_names(self) -> List[str]:
            return list(
                self._current_primary_renderable_series_names(
                    self._axis_scope_series_names(),
                    require_visible=False,
                )
            )

        @staticmethod
        def _apply_sample_indices(array: np.ndarray, sample_idx: object) -> np.ndarray:
            values = _try_1d_float_array(array)
            if values.size == 0:
                return values
            if isinstance(sample_idx, slice):
                return values
            return values[sample_idx]

        def _current_primary_plot_basis(
            self,
        ) -> Optional[Tuple[str, str, np.ndarray, np.ndarray, Optional[np.ndarray], object]]:
            x_name = str(self._x_axis_name or "t")
            x_data, x_label = self._get_x_data()
            x_array = _try_1d_float_array(x_data)
            if x_array.size == 0:
                return None
            sample_idx = self._get_sampling_indices(x_array.shape[0])
            x_plot = self._apply_sample_indices(x_array, sample_idx)
            t_array = _try_1d_float_array(self._t)
            t_plot: Optional[np.ndarray] = None
            if t_array.size != 0 and t_array.shape[0] == x_array.shape[0]:
                t_plot = self._apply_sample_indices(t_array, sample_idx)
            return x_name, x_label, x_array, x_plot, t_plot, sample_idx

        @staticmethod
        def _qualified_copy_header(block_label: Optional[str], column_label: str) -> str:
            label = str(block_label or "").strip()
            column = str(column_label or "").strip()
            if not label:
                return column
            return f"{label}::{column}"

        def _copy_series_header(self, block_label: Optional[str], series_name: str) -> str:
            return self._qualified_copy_header(block_label, self._series_header_text(series_name))

        def _primary_copy_block_label(self) -> str:
            return str(self._simulation_set_label or "").strip()

        @staticmethod
        def _dataset_overlay_block_label(dataset: object) -> str:
            label = str(dataset or "").strip()
            return f"Dataset overlay: {label}" if label else "Dataset overlay"

        def _append_copy_column(
            self,
            columns: List[Tuple[str, np.ndarray]],
            *,
            header: str,
            values: object,
        ) -> None:
            array = _try_1d_float_array(values)
            if array.size == 0:
                return
            columns.append((str(header), array))

        def set_copy_all_export_plan_provider(
            self,
            provider: Optional[Callable[[], Optional[CopyAllExportPlan]]],
        ) -> None:
            self._copy_all_export_plan_provider = provider if callable(provider) else None

        def set_copy_status_text_callback(
            self,
            callback: Optional[Callable[[str], None]],
        ) -> None:
            self._copy_status_text_callback = callback if callable(callback) else None

        def _set_copy_status_text(self, message: str) -> bool:
            callback = self._copy_status_text_callback
            if not callable(callback):
                return False
            callback(str(message))
            return True

        def _current_copy_axis_spec(self) -> Tuple[str, str]:
            x_name = str(self._x_axis_name or "t")
            _, x_label = self._get_x_data()
            if x_name != "t" and x_name not in self._series:
                return "t", "Time (s)"
            return x_name, str(x_label or "Time (s)")

        def _build_visible_overlay_copy_blocks(
            self,
            *,
            x_name: str,
            x_label: str,
            visible_y_names: Sequence[str],
            excluded_set_ids: Optional[Set[str]] = None,
        ) -> List[List[Tuple[str, np.ndarray]]]:
            blocks: List[List[Tuple[str, np.ndarray]]] = []
            excluded = {str(set_id) for set_id in (excluded_set_ids or set()) if str(set_id)}
            for layer in list(self._simulation_overlays or []):
                if not isinstance(layer, PlotDisplayLayer):
                    continue
                if not bool(layer.visible):
                    continue
                source_id = str(layer.source_id or "").strip()
                if source_id and source_id in excluded:
                    continue
                block_label = self._overlay_display_label(layer)
                overlay_series = self._overlay_y_map(layer)
                if not isinstance(overlay_series, dict):
                    continue
                if x_name == "t":
                    x_overlay = self._overlay_x_values(layer)
                else:
                    x_overlay = overlay_series.get(x_name)
                x_overlay_array = _try_1d_float_array(x_overlay)
                if x_overlay_array.size == 0:
                    continue
                overlay_sample_idx = self._get_sampling_indices(x_overlay_array.shape[0])
                x_overlay_plot = self._apply_sample_indices(x_overlay_array, overlay_sample_idx)

                overlay_columns: List[Tuple[str, np.ndarray]] = []
                self._append_copy_column(
                    overlay_columns,
                    header=self._qualified_copy_header(block_label, x_label),
                    values=x_overlay_plot,
                )

                overlay_y_names = self._simulation_overlay_candidate_series_names(layer, visible_y_names)
                overlay_y_names = self._series_names_compatible_with_x(
                    overlay_y_names,
                    overlay_series,
                    x_overlay_array,
                    require_visible=False,
                )
                overlay_y_added = 0
                for name in overlay_y_names:
                    y_array = _try_1d_float_array(overlay_series.get(name))
                    if y_array.size == 0 or y_array.shape[0] != x_overlay_array.shape[0]:
                        continue
                    y_plot = self._apply_sample_indices(y_array, overlay_sample_idx)
                    self._append_copy_column(
                        overlay_columns,
                        header=self._copy_series_header(block_label, name),
                        values=y_plot,
                    )
                    overlay_y_added += 1
                if overlay_y_added:
                    blocks.append(overlay_columns)
            return blocks

        def _build_dataset_overlay_copy_blocks(
            self,
            *,
            x_label: str,
        ) -> List[List[Tuple[str, np.ndarray]]]:
            blocks: List[List[Tuple[str, np.ndarray]]] = []
            for overlay in list(self._visible_overlay_series or []):
                x_overlay_array = _try_1d_float_array(overlay.x)
                y_overlay_array = _try_1d_float_array(overlay.y)
                if x_overlay_array.size == 0 or y_overlay_array.size == 0:
                    continue
                x_overlay_plot, y_overlay_plot = self._sample_xy(x_overlay_array, y_overlay_array)
                block_label = self._dataset_overlay_block_label(overlay.dataset)
                dataset_columns: List[Tuple[str, np.ndarray]] = []
                self._append_copy_column(
                    dataset_columns,
                    header=self._qualified_copy_header(block_label, x_label),
                    values=x_overlay_plot,
                )
                self._append_copy_column(
                    dataset_columns,
                    header=self._copy_series_header(block_label, str(overlay.species)),
                    values=y_overlay_plot,
                )
                if dataset_columns:
                    blocks.append(dataset_columns)
            return blocks

        def _display_block_current_x_values(
            self,
            *,
            t_values: np.ndarray,
            series_values: Dict[str, np.ndarray],
            x_name: str,
        ) -> np.ndarray:
            if x_name == "t":
                return t_values
            x_values = _try_1d_float_array(series_values.get(x_name))
            if x_values.size == 0 or x_values.shape[0] != t_values.shape[0]:
                return np.asarray([], dtype=float)
            return x_values

        def _build_displayed_simulation_copy_block(
            self,
            display_block: CopyAllDisplayBlock,
            *,
            x_name: str,
            x_label: str,
            visible_y_names: Sequence[str],
        ) -> Tuple[Optional[List[Tuple[str, np.ndarray]]], Optional[str]]:
            t_values = _try_1d_float_array(display_block.t)
            if t_values.size == 0:
                return None, "no_simulation_data"
            x_values = self._display_block_current_x_values(t_values=t_values, series_values=display_block.series, x_name=x_name)
            if x_values.size == 0:
                return None, "current_x_unavailable"

            block_label = str(display_block.label or "").strip()
            display_columns: List[Tuple[str, np.ndarray]] = []
            self._append_copy_column(
                display_columns,
                header=self._qualified_copy_header(block_label, "Time (s)"),
                values=t_values,
            )
            if x_name != "t":
                self._append_copy_column(
                    display_columns,
                    header=self._qualified_copy_header(block_label, x_label),
                    values=x_values,
                )

            y_names = self._series_names_compatible_with_x(
                visible_y_names,
                display_block.series,
                x_values,
                require_visible=False,
            )
            y_added = 0
            for name in y_names:
                y_values = _try_1d_float_array(display_block.series.get(name))
                if y_values.size == 0 or y_values.shape[0] != x_values.shape[0]:
                    continue
                self._append_copy_column(
                    display_columns,
                    header=self._copy_series_header(block_label, name),
                    values=y_values,
                )
                y_added += 1
            if y_added <= 0:
                return None, "no_visible_series"
            return display_columns, None

        def _build_copy_all_blocks(
            self,
            plan: CopyAllExportPlan,
        ) -> Tuple[List[List[Tuple[str, np.ndarray]]], List[CopyAllMissingItem]]:
            x_name, x_label = self._current_copy_axis_spec()

            blocks: List[List[Tuple[str, np.ndarray]]] = []
            missing_items: List[CopyAllMissingItem] = list(plan.missing_items or [])

            for display_block in list(plan.display_blocks or []):
                set_id = str(display_block.set_id or "").strip()
                block_columns, missing_reason = self._build_displayed_simulation_copy_block(
                    display_block,
                    x_name=x_name,
                    x_label=x_label,
                    visible_y_names=list(display_block.display_species),
                )
                if block_columns is not None:
                    blocks.append(block_columns)
                    continue
                missing_items.append(
                    CopyAllMissingItem(
                        set_id=set_id,
                        label=str(display_block.label or ""),
                        popup_label=str(display_block.label or set_id or "Requested Show simulation"),
                        reason=str(missing_reason or "no_simulation_data"),
                    )
                )

            if not blocks and not missing_items:
                raise ValueError("No visible simulation series are available to copy.")
            return blocks, missing_items

        @staticmethod
        def _copy_all_reason_text(reason: str) -> str:
            reason_key = str(reason or "").strip()
            return {
                "preview_pending": "Preview pending",
                "no_cached_results": "Result not cached (evicted)",
                "invalid_cache_entry": "Invalid cached result",
                "current_x_unavailable": "Current X-axis data unavailable",
                "no_visible_series": "No visible series available",
                "no_simulation_data": "No simulation data available",
                "failed_result": "Simulation failed",
                "missing_result": "Missing result",
                "semantic_unavailable": "No semantic displayable result",
                "unavailable": "Unavailable",
            }.get(reason_key, "Unavailable")

        def _confirm_copy_all_missing_items(self, missing_items: Sequence[CopyAllMissingItem]) -> bool:
            entries = [item for item in missing_items if isinstance(item, CopyAllMissingItem)]
            if not entries:
                return True
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            box.setWindowTitle("Copy All")
            box.setText("Some requested Show simulations are unavailable for truthful export.")
            lines = [
                f"{str(item.popup_label or item.label or item.set_id or 'Requested Show simulation')}: "
                f"{self._copy_all_reason_text(item.reason)}"
                for item in entries
            ]
            box.setInformativeText("\n".join(lines) + "\n\nCopy available data anyway?")
            yes_button = box.addButton("Yes", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            no_button = box.addButton("No", QtWidgets.QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(no_button)
            box.setEscapeButton(no_button)
            box.exec()
            return box.clickedButton() is yes_button

        def _show_copy_all_missing_items(self, missing_items: Sequence[CopyAllMissingItem]) -> None:
            entries = [item for item in missing_items if isinstance(item, CopyAllMissingItem)]
            if not entries:
                return
            lines = [
                f"{str(item.popup_label or item.label or item.set_id or 'Requested Show simulation')}: "
                f"{self._copy_all_reason_text(item.reason)}"
                for item in entries
            ]
            QtWidgets.QMessageBox.warning(
                self,
                "Copy All",
                "No displayed simulation data can be copied.\n\n" + "\n".join(lines),
            )

        def _build_visible_copy_blocks(self) -> List[List[Tuple[str, np.ndarray]]]:
            if self._t is None or not self._series:
                raise ValueError("No simulation data is available to copy.")

            primary_visible_y_names = self._visible_primary_copy_series_names()
            overlay_visible_y_names = self._visible_overlay_copy_series_names()

            primary_basis = self._current_primary_plot_basis()
            if primary_basis is None:
                raise ValueError("The current X-axis has no visible data to copy.")
            x_name, x_label, x_array, x_plot, t_plot, sample_idx = primary_basis

            blocks: List[List[Tuple[str, np.ndarray]]] = []
            primary_label = self._primary_copy_block_label()
            primary_columns: List[Tuple[str, np.ndarray]] = []

            if t_plot is not None:
                self._append_copy_column(
                    primary_columns,
                    header=self._qualified_copy_header(primary_label, "Time (s)"),
                    values=t_plot,
                )
            if x_name != "t":
                self._append_copy_column(
                    primary_columns,
                    header=self._qualified_copy_header(primary_label, x_label),
                    values=x_plot,
                )

            primary_y_added = 0
            for name in primary_visible_y_names:
                y_array = _try_1d_float_array(self._series.get(name))
                if y_array.size == 0 or y_array.shape[0] != x_array.shape[0]:
                    continue
                y_array = self._apply_sample_indices(y_array, sample_idx)
                if y_array.size == 0 or y_array.shape[0] != x_plot.shape[0]:
                    continue
                self._append_copy_column(
                    primary_columns,
                    header=self._copy_series_header(primary_label, name),
                    values=y_array,
                )
                primary_y_added += 1
            if primary_y_added:
                blocks.append(primary_columns)

            blocks.extend(
                self._build_visible_overlay_copy_blocks(
                    x_name=x_name,
                    x_label=x_label,
                    visible_y_names=overlay_visible_y_names,
                )
            )
            blocks.extend(self._build_dataset_overlay_copy_blocks(x_label=x_label))

            return blocks

        @staticmethod
        def _flatten_copy_blocks(blocks: Sequence[Sequence[Tuple[str, np.ndarray]]]) -> List[Tuple[str, np.ndarray]]:
            columns: List[Tuple[str, np.ndarray]] = []
            non_empty_blocks = [list(block) for block in blocks if list(block)]
            for idx, block in enumerate(non_empty_blocks):
                if idx > 0:
                    columns.append(("", np.asarray([], dtype=float)))
                columns.extend(block)
            return columns

        @staticmethod
        def _copy_columns_to_rows(columns: Sequence[Tuple[str, np.ndarray]]) -> Tuple[List[str], List[List[object]]]:
            if not columns:
                raise ValueError("No visible plot data is available to copy.")
            max_len = max(values.shape[0] for _, values in columns)
            header = [label for label, _ in columns]
            rows: List[List[object]] = []
            for idx in range(max_len):
                row: List[object] = []
                for _, values in columns:
                    row.append(values[idx] if idx < values.shape[0] else "")
                rows.append(row)
            return header, rows

        @staticmethod
        def _rows_to_tsv(header: Sequence[object], rows: Sequence[Sequence[object]]) -> str:
            def _cell(value: object) -> str:
                if value == "":
                    return ""
                return str(value)

            lines = ["\t".join(_cell(cell) for cell in header)]
            for row in rows:
                lines.append("\t".join(_cell(cell) for cell in row))
            return "\n".join(lines)

        def _copy_visible_data(self) -> None:
            try:
                columns = self._flatten_copy_blocks(self._build_visible_copy_blocks())
                header, rows = self._copy_columns_to_rows(columns)
                clipboard = self._get_clipboard()
                if clipboard is None:
                    raise ValueError("Clipboard is unavailable.")
                clipboard.setText(self._rows_to_tsv(header, rows))
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Copy Visible Data", str(exc))
            except Exception as exc:
                logger.exception("Failed to copy visible plot data: %s", exc)
                QtWidgets.QMessageBox.warning(
                    self,
                    "Copy Visible Data",
                    f"Failed to copy visible plot data: {exc}",
                )

        def _copy_all(self) -> None:
            try:
                provider = self._copy_all_export_plan_provider
                if provider is None:
                    if self._set_copy_status_text("No active simulation display transaction is available to copy."):
                        return
                    raise ValueError("Copy All is unavailable.")
                plan = provider()
                if plan is None:
                    if self._set_copy_status_text("No active simulation display transaction is available to copy."):
                        return
                    raise ValueError("Copy All is unavailable.")
                blocks, missing_items = self._build_copy_all_blocks(plan)
                if not blocks:
                    self._show_copy_all_missing_items(missing_items)
                    return
                if missing_items and not self._confirm_copy_all_missing_items(missing_items):
                    return
                columns = self._flatten_copy_blocks(blocks)
                header, rows = self._copy_columns_to_rows(columns)
                clipboard = self._get_clipboard()
                if clipboard is None:
                    raise ValueError("Clipboard is unavailable.")
                clipboard.setText(self._rows_to_tsv(header, rows))
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Copy All", str(exc))
            except Exception as exc:
                logger.exception("Failed to copy all plot data: %s", exc)
                QtWidgets.QMessageBox.warning(
                    self,
                    "Copy All",
                    f"Failed to copy all plot data: {exc}",
                )

        def update_statistics(
            self,
            t: np.ndarray,
            series: Dict[str, np.ndarray],
            chi_squared: Optional[float] = None,
        ) -> None:
            """Update the species statistics table with latest results."""
            label = str(getattr(self, "_simulation_set_label", "") or "").strip() or "Results"
            layer_id = str(getattr(self, "_simulation_layer_id", "") or "").strip() or "result:live"
            self.set_statistics_results(
                {
                    layer_id: {
                        "t": t,
                        "series": series,
                        "chi_squared": chi_squared,
                        "label": label,
                        "layer_id": layer_id,
                        "kind": PlotLayerKind.PRIMARY_SERIES,
                        "set_id": str(getattr(self, "_simulation_set_id", "") or ""),
                    }
                },
                prefer=layer_id,
            )

        def set_statistics_results(
            self,
            results_map: Dict[str, Dict[str, object]],
            *,
            prefer: Optional[str] = None,
        ) -> None:
            """
            Provide the full set of results available for the statistics table.

            Parameters
            ----------
            results_map : dict
                Mapping of semantic layer ID -> {'t': array, 'series': {species: array}, 'label': display label}
            prefer : str, optional
                Preferred semantic layer ID or display label (used after refresh).
            """
            cleaned: Dict[str, Dict[str, object]] = {}
            for raw_key, payload in (results_map or {}).items():
                fallback_key = str(raw_key or "").strip()
                if not fallback_key:
                    continue
                if not isinstance(payload, dict):
                    continue
                t_payload = payload.get("t")
                series_payload = payload.get("series")
                if t_payload is None or not isinstance(series_payload, dict) or not series_payload:
                    continue
                cleaned_payload = dict(payload)
                layer_key = str(cleaned_payload.get("layer_id") or fallback_key).strip()
                if not layer_key:
                    continue
                cleaned_payload["layer_id"] = layer_key
                cleaned_payload.setdefault("kind", PlotLayerKind.RESULT_SERIES)
                cleaned_payload.setdefault("label", fallback_key)
                cleaned[layer_key] = cleaned_payload

            self._stats_results_map = cleaned

            previous = str(self._stats_result_selector.currentData() or self._stats_result_selector.currentText() or "").strip()
            preferred = str(prefer or "").strip()
            visible_cleaned = self._visible_stats_results_map()
            next_selection = self._stats_result_key_for_preference(preferred, visible_cleaned) or (
                previous if previous and previous in visible_cleaned else next(iter(visible_cleaned.keys()), "")
            )

            self._stats_result_selector.blockSignals(True)
            try:
                self._stats_result_selector.clear()
                for layer_key, payload in visible_cleaned.items():
                    self._stats_result_selector.addItem(
                        self._stats_result_display_label(layer_key, payload),
                        layer_key,
                    )
                if next_selection:
                    for index in range(self._stats_result_selector.count()):
                        if str(self._stats_result_selector.itemData(index) or "") == next_selection:
                            self._stats_result_selector.setCurrentIndex(index)
                            break
            finally:
                self._stats_result_selector.blockSignals(False)

            self._stats_result_selector.setEnabled(self._stats_result_selector.count() > 1)
            if next_selection:
                self._render_statistics_for_label(next_selection)
            else:
                self._stats_table.setRowCount(0)

        def _visible_stats_results_map(self) -> Dict[str, Dict[str, object]]:
            return dict(self._stats_results_map or {})

        @staticmethod
        def _stats_result_display_label(layer_key: str, payload: Mapping[str, object]) -> str:
            label = str(payload.get("label") or "").strip()
            return label or str(layer_key)

        @classmethod
        def _stats_result_key_for_preference(
            cls,
            preferred: str,
            visible_results: Mapping[str, Mapping[str, object]],
        ) -> str:
            preferred_s = str(preferred or "").strip()
            if not preferred_s:
                return ""
            if preferred_s in visible_results:
                return preferred_s
            for layer_key, payload in visible_results.items():
                if cls._stats_result_display_label(str(layer_key), payload) == preferred_s:
                    return str(layer_key)
            return ""

        def _on_stats_result_selector_changed(self, label: str) -> None:
            layer_key = str(self._stats_result_selector.currentData() or "").strip()
            if not layer_key:
                layer_key = str(label or "").strip()
            if not layer_key:
                self._stats_table.setRowCount(0)
                return
            self._render_statistics_for_label(layer_key)

        def _render_statistics_for_label(self, layer_key: str) -> None:
            payload = self._stats_results_map.get(str(layer_key))
            if not isinstance(payload, dict):
                self._stats_table.setRowCount(0)
                return
            t_payload = payload.get("t")
            series_payload = payload.get("series")
            if t_payload is None or not isinstance(series_payload, dict):
                self._stats_table.setRowCount(0)
                return
            chi_squared_value: Optional[float] = None
            raw_chi_squared = payload.get("chi_squared")
            if raw_chi_squared is not None:
                try:
                    chi_squared_value = float(raw_chi_squared)
                except Exception:
                    chi_squared_value = None

            t_arr = _try_1d_float_array(t_payload)
            if t_arr.size == 0:
                self._stats_table.setRowCount(0)
                return

            normalized_series: Dict[str, np.ndarray] = {}
            for name, values in series_payload.items():
                arr = _try_1d_float_array(values)
                if arr.size == 0:
                    continue
                normalized_series[str(name)] = arr

            if not normalized_series:
                self._stats_table.setRowCount(0)
                return

            self._stats_table.update_results(t_arr, normalized_series, chi_squared=chi_squared_value)

        def update_parameters(self, parameters: Dict[str, Tuple[float, str]]) -> None:
            """Update the solver parameter table."""
            try:
                self._param_table.update_parameters(parameters)
            except Exception as exc:
                logger.debug("Failed to update parameter table: %s", exc)

        def set_overlay_catalog(self, datasets: Dict[str, Dict[str, np.ndarray]]) -> None:
            """
            Provide the currently loaded experimental datasets for overlays.

            Parameters
            ----------
            datasets : dict
                Mapping of dataset name -> canonical dataset payload or local dense overlay payload.
            """
            normalized: Dict[str, Dict[str, np.ndarray]] = {}
            for name, payload in (datasets or {}).items():
                try:
                    observations = observations_from_payload(payload)
                    if observations:
                        t_raw, species_raw = dense_view_from_observations(observations)
                    else:
                        t_raw = np.asarray(payload.get("t", []), dtype=float).reshape(-1)
                        species_raw = payload.get("species") or {}
                    if t_raw.size == 0 or not species_raw:
                        continue
                    species_norm: Dict[str, np.ndarray] = {}
                    for sp_name, values in species_raw.items():
                        arr = np.asarray(values, dtype=float).reshape(-1)
                        if arr.size:
                            species_norm[str(sp_name)] = arr
                    if species_norm:
                        normalized[name] = {"t": t_raw, "species": species_norm}
                        if observations:
                            normalized[name]["observations"] = observations
                except Exception as exc:
                    logger.warning("Failed to normalize dataset '%s' for overlay: %s", name, exc)

            self._overlay_datasets = normalized
            self._assign_overlay_styles()
            self._overlay_panel.set_datasets(normalized)
            self._update_plot()

        def active_overlays(self) -> List[str]:
            """Return dataset names currently active as overlays."""
            return self._overlay_panel.selected_datasets()

        def overlay_snapshot(self) -> Dict[str, object]:
            """Return metadata about overlay state for provenance."""
            return {
                "selected": list(self._overlay_panel.selected_datasets()),
                "available": sorted(self._overlay_datasets.keys()),
                "x_axis": self._x_axis_name,
                "parametric": self._parametric_mode,
            }

        def _assign_colors(self):
            """Assign globally owned colors to species and neutral colors to derived series."""
            color_manager = ColorManager.instance()
            self._colors.clear()
            owned_keys = set(self._owned_species_keys or set())
            if owned_keys:
                color_manager.seed_species(sorted(owned_keys))
            for name in self._series.keys():
                canonical = color_manager.resolve_species_key(name, known_species=tuple(owned_keys) if owned_keys else None)
                if not owned_keys or canonical in owned_keys:
                    self._colors[name] = color_manager.get_species_rgb(name, known_species=tuple(owned_keys) if owned_keys else None)
                else:
                    non_species = color_manager.get_non_species_color(name)
                    self._colors[name] = (non_species.red(), non_species.green(), non_species.blue())

        def _assign_overlay_styles(self):
            """Assign deterministic marker styles to overlay datasets."""
            color_manager = ColorManager.instance()
            self._overlay_symbols.clear()
            for idx, name in enumerate(sorted(self._overlay_datasets.keys())):
                self._overlay_symbols[name] = color_manager.get_dataset_symbol(idx)

        def _active_overlay_known_species(self) -> tuple[str, ...]:
            if not bool(getattr(self, "_owned_species_roster_explicit", False)):
                return ()
            return tuple(str(name) for name in (self._owned_species_keys or set()) if str(name))

        def _overlay_display_color(self, species_key: str) -> tuple[int, int, int]:
            """Return the current display color for a resolved overlay dataset column."""
            color_manager = ColorManager.instance()
            color_key = str(species_key or "").strip()
            known_species = self._active_overlay_known_species()
            color = color_manager.get_display_series_color(
                color_key,
                known_species=known_species or None,
            )
            return (int(color.red()), int(color.green()), int(color.blue()))

        def refresh_overlay_presentation_for_current_roster(self) -> None:
            """Keep visible overlay markers aligned with current-roster swatch semantics."""
            known_species = self._active_overlay_known_species()
            self._overlay_panel.refresh_color_swatches(known_species=known_species or None)
            self._refresh_visible_overlay_warnings_for_current_roster()
            if not self._export_all_overlay_cache_dirty:
                self._refresh_export_all_overlay_roster_view()
            self._overlay_panel.set_status_messages(self._visible_overlay_warnings)
            self._draw_overlay_series(list(self._active_overlay_series))

        def _overlay_series_for_current_roster(
            self,
            overlays: Sequence[_OverlaySeries],
        ) -> List[_OverlaySeries]:
            return list(overlays or [])

        def _refresh_visible_overlay_warnings_for_current_roster(self) -> None:
            self._visible_overlay_warnings = [
                warning
                for warning in self._active_overlay_warnings
                if self._overlay_warning_matches_current_roster(warning)
            ]

        def _overlay_warning_matches_current_roster(self, warning: str) -> bool:
            color_manager = ColorManager.instance()
            known_species = self._active_overlay_known_species()
            if not known_species:
                return True
            species_name = self._overlay_warning_species_name(str(warning or ""))
            if not species_name:
                return True
            if species_name == "t":
                return True
            owned_species_key = color_manager.resolve_known_species_key(species_name, known_species)
            if owned_species_key is None:
                return False
            return color_manager.resolve_current_species_key(
                owned_species_key,
                known_species=known_species,
            ) is not None

        @staticmethod
        def _overlay_warning_species_name(warning_text: str) -> Optional[str]:
            marker = "species '"
            start = warning_text.find(marker)
            if start >= 0:
                start += len(marker)
            else:
                start = warning_text.find("'")
                if start < 0:
                    return None
                start += len("'")
            end = warning_text.find("'", start)
            if end < 0:
                return None
            return str(warning_text[start:end] or "").strip() or None

        def _clear_overlay_series_caches(self) -> None:
            self._active_overlay_series = []
            self._visible_overlay_series = []
            self._export_all_overlay_series_unfiltered = []
            self._export_all_overlay_series = []
            self._active_overlay_warnings = []
            self._visible_overlay_warnings = []
            self._export_all_overlay_warnings_unfiltered = []
            self._export_all_overlay_warnings = []
            self._export_all_overlay_cache_dirty = True

        def _resolve_overlay_x_source(
            self,
            x_name: str,
            payload: Dict[str, Dict[str, np.ndarray]],
        ) -> Tuple[Optional[str], Optional[np.ndarray]]:
            """Resolve overlay X values once while preserving current exact-match semantics."""
            if x_name == "t":
                return None, _try_1d_float_array(payload.get("t"))
            species_payload = payload.get("species") or {}
            if not isinstance(species_payload, dict):
                return None, None
            x_source = species_payload.get(x_name)
            if x_source is None:
                return None, None
            return x_name, _try_1d_float_array(x_source)

        def _rebuild_overlay_series_caches(self, axis_scope_series: Sequence[str]) -> None:
            """
            Rebuild all overlay-record caches before any downstream consumer uses them.

            Built overlay records are fully resolved snapshots. Draw, refresh,
            copy, and export paths must use only these records and must
            not call _resolve_dataset_species() again.
            """
            axis_candidate_names = list(axis_scope_series or [])
            self._active_overlay_series, self._active_overlay_warnings = self._build_overlay_series(axis_candidate_names)
            self._refresh_visible_overlay_warnings_for_current_roster()

        def _ensure_export_all_overlay_cache(self) -> None:
            if not self._export_all_overlay_cache_dirty:
                self._refresh_export_all_overlay_roster_view()
                return
            overlay_series, raw_warnings = self._build_overlay_series(
                list(self._series.keys())
            )
            self._export_all_overlay_series_unfiltered = list(overlay_series)
            self._export_all_overlay_warnings_unfiltered = [
                msg for msg in raw_warnings
                if ": no column matching species " not in msg
            ]
            self._refresh_export_all_overlay_roster_view()
            self._export_all_overlay_cache_dirty = False

        def _refresh_export_all_overlay_roster_view(self) -> None:
            self._export_all_overlay_series = list(self._export_all_overlay_series_unfiltered)
            self._export_all_overlay_warnings = [
                warning
                for warning in self._export_all_overlay_warnings_unfiltered
                if self._overlay_warning_matches_current_roster(warning)
            ]

        def _get_sampling_indices(self, length: int):
            """Return slice or index array for downsampling plots."""
            if self._sampling_mode == "dense" or length <= self._sampling_target:
                return slice(None)
            if length <= 0:
                return np.array([], dtype=int)
            target = max(2, min(int(self._sampling_target), int(length)))
            return np.unique(np.linspace(0, length - 1, num=target, dtype=int))

        def _sample_xy(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Apply sampling to X/Y arrays for plotting."""
            indices = self._get_sampling_indices(x_data.shape[0])
            if isinstance(indices, slice):
                return x_data, y_data
            return x_data[indices], y_data[indices]

        _SUBSCRIPT_CHAR_MAP = {
            "0": "₀",
            "1": "₁",
            "2": "₂",
            "3": "₃",
            "4": "₄",
            "5": "₅",
            "6": "₆",
            "7": "₇",
            "8": "₈",
            "9": "₉",
            "+": "₊",
            "-": "₋",
            "(": "₍",
            ")": "₎",
            "a": "ₐ",
            "e": "ₑ",
            "h": "ₕ",
            "i": "ᵢ",
            "j": "ⱼ",
            "k": "ₖ",
            "l": "ₗ",
            "m": "ₘ",
            "n": "ₙ",
            "o": "ₒ",
            "p": "ₚ",
            "r": "ᵣ",
            "s": "ₛ",
            "t": "ₜ",
            "u": "ᵤ",
            "v": "ᵥ",
            "x": "ₓ",
        }

        def _format_species_set_label(self, species: str, set_label: Optional[str]) -> str:
            """Format legend labels like A(set1), preferring a subscript set label when feasible."""
            species = str(species)
            set_label = str(set_label or "").strip()
            if not set_label:
                return species
            sub_chars: List[str] = []
            for ch in set_label:
                mapped = self._SUBSCRIPT_CHAR_MAP.get(ch)
                if mapped is None:
                    mapped = self._SUBSCRIPT_CHAR_MAP.get(ch.lower())
                if mapped is None:
                    return f"{species}({set_label})"
                sub_chars.append(mapped)
            return f"{species}₍{''.join(sub_chars)}₎"

        def _upsert_curve_item(
            self,
            *,
            key: str,
            x_data: np.ndarray,
            y_data: np.ndarray,
            pen: object,
            name: Optional[str] = None,
            visible: bool = True,
        ) -> None:
            name_s = str(name or key)
            visible_b = bool(visible)
            signature = self._curve_item_signature(name=name_s, pen=pen, visible=visible_b)
            item = self._plot_items.get(key)
            if item is None:
                item = self._plot_item.plot(x_data, y_data, name=name_s, pen=pen)
                self._plot_items[key] = item
                self._plot_item_signatures[key] = signature
                item.setVisible(visible_b)
                return
            previous_signature = self._plot_item_signatures.get(key)
            if previous_signature is None or previous_signature[0] != name_s:
                self._update_curve_item_name(item, name_s)
            if not self._curve_item_data_matches(item, x_data, y_data):
                item.setData(x_data, y_data)
            if previous_signature is None or previous_signature[1] != signature[1]:
                item.setPen(pen)
            if previous_signature is None or previous_signature[2] != visible_b:
                item.setVisible(visible_b)
            self._plot_item_signatures[key] = signature

        @classmethod
        def _curve_item_signature(cls, *, name: str, pen: object, visible: bool) -> tuple[object, ...]:
            return (str(name), cls._pen_signature(pen), bool(visible))

        @staticmethod
        def _pen_signature(pen: object) -> tuple[object, ...]:
            color = getattr(pen, "color", lambda: None)()
            color_signature: object
            if color is not None and all(hasattr(color, attr) for attr in ("red", "green", "blue", "alpha")):
                color_signature = (int(color.red()), int(color.green()), int(color.blue()), int(color.alpha()))
            else:
                color_signature = repr(color)
            width_getter = getattr(pen, "widthF", None)
            width = float(width_getter()) if callable(width_getter) else None
            style_getter = getattr(pen, "style", None)
            style = style_getter() if callable(style_getter) else None
            return (color_signature, width, str(style))

        def _update_curve_item_name(self, item: object, name: str) -> None:
            new_name = str(name)
            opts = getattr(item, "opts", None)
            if isinstance(opts, dict):
                current = str(opts.get("name") or "")
                if current == new_name:
                    return
                opts["name"] = new_name
            label_getter = getattr(self._legend, "getLabel", None)
            if callable(label_getter):
                try:
                    label = label_getter(item)
                    setter = getattr(label, "setText", None)
                    if callable(setter):
                        setter(new_name)
                except Exception:
                    pass

        @staticmethod
        def _curve_item_data_matches(item: object, x_data: np.ndarray, y_data: np.ndarray) -> bool:
            data_getter = getattr(item, "getData", None)
            if not callable(data_getter):
                return False
            try:
                existing_x, existing_y = data_getter()
            except Exception:
                return False
            try:
                return bool(
                    np.array_equal(np.asarray(existing_x, dtype=float).reshape(-1), np.asarray(x_data, dtype=float).reshape(-1))
                    and np.array_equal(
                        np.asarray(existing_y, dtype=float).reshape(-1),
                        np.asarray(y_data, dtype=float).reshape(-1),
                    )
                )
            except Exception:
                return False

        def _prune_curve_items(self, active_keys: Set[str]) -> None:
            for key in list(self._plot_items.keys()):
                if key in active_keys:
                    continue
                item = self._plot_items.pop(key, None)
                self._plot_item_signatures.pop(key, None)
                if item is None:
                    continue
                self._plot_item.removeItem(item)

        def _upsert_dataset_scatter_item(
            self,
            *,
            key: str,
            x_data: np.ndarray,
            y_data: np.ndarray,
            brush: object,
            size: float,
            name: str,
        ) -> None:
            item = self._dataset_scatter_items.get(key)
            if item is None:
                item = pg.ScatterPlotItem(
                    x=x_data,
                    y=y_data,
                    pen=None,
                    brush=brush,
                    size=size,
                    name=name,
                )
                self._plot_item.addItem(item)
                self._dataset_scatter_items[key] = item
                return
            item.setData(x=x_data, y=y_data, pen=None, brush=brush, size=size)

        def _prune_dataset_scatter_items(self, active_keys: Set[str]) -> None:
            for key in list(self._dataset_scatter_items.keys()):
                if key in active_keys:
                    continue
                item = self._dataset_scatter_items.pop(key, None)
                if item is None:
                    continue
                self._plot_item.removeItem(item)

        def _upsert_dataset_model_item(
            self,
            *,
            key: str,
            x_data: np.ndarray,
            y_data: np.ndarray,
            pen: object,
            name: str,
        ) -> None:
            item = self._dataset_model_items.get(key)
            if item is None:
                item = self._plot_item.plot(x_data, y_data, name=name, pen=pen)
                self._dataset_model_items[key] = item
                return
            item.setData(x_data, y_data)
            item.setPen(pen)

        def _prune_dataset_model_items(self, active_keys: Set[str]) -> None:
            for key in list(self._dataset_model_items.keys()):
                if key in active_keys:
                    continue
                item = self._dataset_model_items.pop(key, None)
                if item is None:
                    continue
                self._plot_item.removeItem(item)

        def _upsert_dataset_overlay_item(
            self,
            *,
            key: Tuple[str, str],
            x_data: np.ndarray,
            y_data: np.ndarray,
            pen: object,
            brush: object,
            size: float,
            symbol: str,
            name: str,
        ) -> None:
            scatter = self._overlay_items.get(key)
            if scatter is None:
                scatter = pg.ScatterPlotItem(
                    x_data,
                    y_data,
                    pen=pen,
                    brush=brush,
                    size=size,
                    symbol=symbol,
                    name=name,
                )
                scatter.setZValue(5)
                self._plot_item.addItem(scatter)
                self._overlay_items[key] = scatter
                return
            scatter.setData(x=x_data, y=y_data, pen=pen, brush=brush, size=size, symbol=symbol)
            scatter.setZValue(5)

        def _prune_dataset_overlay_items(self, active_keys: Set[Tuple[str, str]]) -> None:
            for key in list(self._overlay_items.keys()):
                if key in active_keys:
                    continue
                item = self._overlay_items.pop(key, None)
                if item is None:
                    continue
                self._plot_item.removeItem(item)

        def _update_plot(self):
            """Update plot with current data, visibility settings, and axis configuration."""
            self._clear_overlay_series_caches()
            self._overlay_panel.set_status_messages([])
            if self._t is None:
                return

            primary_basis = self._current_primary_plot_basis()
            if primary_basis is None:
                return
            _x_name, x_label, x_array, x_plot, _t_plot, sample_idx = primary_basis
            selected_visible_series = self._visible_selected_series_names()
            axis_scope_series = self._axis_scope_series_names()
            selected_primary_series = self._current_primary_renderable_series_names(
                selected_visible_series,
                require_visible=False,
            )

            # Update axis labels dynamically
            self._plot_item.setLabel('bottom', x_label)
            if not self._parametric_mode:
                self._plot_item.setLabel('left', 'Concentration', units='M')
            else:
                # In parametric mode, Y label depends on selected series
                if selected_primary_series:
                    self._plot_item.setLabel('left', f'{selected_primary_series[0]}', units='M')

            # Add visible series (only those selected in toolbar)
            active_curve_keys: Set[str] = set()
            for name in selected_primary_series:
                y_data = np.asarray(self._series[name], dtype=float).reshape(-1)
                color = self._colors.get(name, (100, 100, 100))
                pen = pg.mkPen(color=color, width=2)

                y_plot = self._apply_sample_indices(y_data, sample_idx)
                label = self._format_species_set_label(name, self._simulation_set_label)
                primary_layer_id = str(self._simulation_layer_id or "").strip() or "result:live"
                item_key = self._overlay_item_key(layer_id=primary_layer_id, species=name)
                active_curve_keys.add(item_key)
                self._upsert_curve_item(
                    key=item_key,
                    x_data=x_plot,
                    y_data=y_plot,
                    pen=pen,
                    name=label,
                )

            # Batch simulation overlays (additional initial-condition sets as lines)
            sim_overlays = list(self._simulation_overlays or [])
            if sim_overlays:
                color_manager = ColorManager.instance()
                x_name = self._x_axis_name or "t"
                for idx, layer in enumerate(sim_overlays):
                    if not isinstance(layer, PlotDisplayLayer):
                        continue
                    set_label = self._overlay_display_label(layer)
                    is_reference_layer = self._is_reference_layer(layer)
                    layer_visible = bool(layer.visible)
                    if not set_label:
                        continue
                    t_overlay = self._overlay_x_values(layer)
                    series_overlay = self._overlay_y_map(layer)
                    if t_overlay is None or not isinstance(series_overlay, dict):
                        continue
                    t_arr = _try_1d_float_array(t_overlay)
                    if t_arr.size == 0:
                        continue
                    if x_name == "t":
                        x_overlay = t_arr
                    else:
                        x_source = series_overlay.get(x_name)
                        if x_source is None:
                            continue
                        x_overlay = np.asarray(x_source, dtype=float).reshape(-1)
                        if x_overlay.size == 0:
                            continue

                    idx_overlay = self._get_sampling_indices(x_overlay.shape[0])
                    x_plot_overlay = x_overlay if isinstance(idx_overlay, slice) else x_overlay[idx_overlay]
                    style = color_manager.get_dataset_line_style(idx)
                    overlay_species = self._series_names_compatible_with_x(
                        self._simulation_overlay_candidate_series_names(layer, axis_scope_series),
                        series_overlay,
                        x_overlay,
                        require_visible=False,
                    )
                    for species in overlay_species:
                        y_source = series_overlay.get(species)
                        if y_source is None:
                            continue
                        y_arr = np.asarray(y_source, dtype=float).reshape(-1)
                        if y_arr.shape[0] != x_overlay.shape[0]:
                            continue
                        y_plot_overlay = y_arr if isinstance(idx_overlay, slice) else y_arr[idx_overlay]

                        overlay_owned_species = self._simulation_overlay_owned_species(layer)
                        if overlay_owned_species:
                            base_color = color_manager.get_species_rgb(
                                species,
                                known_species=overlay_owned_species,
                            )
                        else:
                            base_color = self._colors.get(species, (100, 100, 100))
                        try:
                            r, g, b = base_color
                        except Exception:
                            r, g, b = (100, 100, 100)

                        overlay_label = self._format_species_set_label(species, set_label)
                        layer_id = self._overlay_layer_id(layer)
                        overlay_key = self._overlay_item_key(layer_id=layer_id, species=species)
                        overlay_name = overlay_label
                        if is_reference_layer:
                            pen = pg.mkPen(color=(r, g, b, 90), width=1.2, style=Qt.PenStyle.DashLine)
                        else:
                            pen = pg.mkPen(color=(r, g, b, 180), width=1.6, style=style)
                        active_curve_keys.add(overlay_key)
                        self._upsert_curve_item(
                            key=overlay_key,
                            x_data=x_plot_overlay,
                            y_data=y_plot_overlay,
                            pen=pen,
                            name=overlay_name,
                            visible=layer_visible,
                        )

            self._prune_curve_items(active_curve_keys)

            self._rebuild_overlay_series_caches(axis_scope_series)
            self._draw_overlay_series(list(self._active_overlay_series))
            self._overlay_panel.set_status_messages(self._visible_overlay_warnings)
            self._refresh_intervention_annotations()
            self._refresh_view_after_plot_update()

        def _refresh_intervention_annotations(self) -> None:
            signature = self._intervention_annotations_signature()
            if signature == self._intervention_annotation_signature:
                return
            for item in list(self._intervention_annotation_items):
                try:
                    self._plot_item.removeItem(item)
                except Exception:
                    pass
            self._intervention_annotation_items = []
            self._intervention_annotation_signature = signature
            if (
                not self._show_intervention_annotations
                or self._t is None
                or self._parametric_mode
                or self._x_axis_name != "t"
            ):
                return
            for annotation in list(self._intervention_annotations or []):
                start_value = _try_float(annotation.get("start"))
                end_value = _try_float(annotation.get("end"))
                label = str(annotation.get("label") or annotation.get("kind") or "intervention")
                if start_value is not None and end_value is not None and end_value > start_value:
                    try:
                        region = pg.LinearRegionItem(
                            values=(float(start_value), float(end_value)),
                            movable=False,
                            brush=pg.mkBrush(210, 110, 0, 32),
                            pen=pg.mkPen(color=(210, 110, 0), width=1, style=Qt.DashLine),
                        )
                        region.setZValue(8)
                        self._plot_item.addItem(region, ignoreBounds=True)
                        self._intervention_annotation_items.append(region)
                    except Exception:
                        logger.debug("Failed to draw intervention interval annotation.", exc_info=True)
                    for boundary_label, boundary_time in (("start", start_value), ("end", end_value)):
                        line = pg.InfiniteLine(
                            angle=90,
                            pos=float(boundary_time),
                            movable=False,
                            pen=pg.mkPen(color=(210, 110, 0), width=1, style=Qt.DashLine),
                            label=f"{label} {boundary_label}",
                            labelOpts={"position": 0.95, "color": (110, 70, 0)},
                        )
                        line.setZValue(9)
                        self._plot_item.addItem(line, ignoreBounds=True)
                        self._intervention_annotation_items.append(line)
                    continue
                time_value = _try_float(annotation.get("time"))
                if time_value is None:
                    continue
                line = pg.InfiniteLine(
                    angle=90,
                    pos=float(time_value),
                    movable=False,
                    pen=pg.mkPen(color=(210, 110, 0), width=1, style=Qt.DashLine),
                    label=label,
                    labelOpts={"position": 0.95, "color": (110, 70, 0)},
                )
                line.setZValue(9)
                self._plot_item.addItem(line, ignoreBounds=True)
                self._intervention_annotation_items.append(line)

        def _intervention_annotations_signature(self) -> tuple[object, ...]:
            annotation_parts: list[tuple[tuple[str, str], ...]] = []
            for annotation in list(self._intervention_annotations or []):
                if not isinstance(annotation, dict):
                    continue
                annotation_parts.append(
                    tuple(
                        sorted(
                            (str(key), repr(value))
                            for key, value in annotation.items()
                        )
                    )
                )
            t_size = 0 if self._t is None else int(np.asarray(self._t).reshape(-1).size)
            return (
                bool(self._show_intervention_annotations),
                bool(self._parametric_mode),
                str(self._x_axis_name or ""),
                t_size,
                tuple(annotation_parts),
            )

        def _refresh_view_after_plot_update(self) -> None:
            """Keep auto-range truthful across live result updates without overriding manual ranges."""
            if not bool(self._auto_range_enabled):
                return
            try:
                self._plot_item.enableAutoRange(x=True, y=True)
                self._plot_item.autoRange()
            except Exception as exc:
                logger.debug("Failed to auto-range plot after data update: %s", exc, exc_info=True)

        def _build_overlay_series(self, selected_series: List[str]) -> Tuple[List[_OverlaySeries], List[str]]:
            """Compute fully resolved overlay records for the current axis and dataset subset."""
            overlays: List[_OverlaySeries] = []
            warnings: List[str] = []
            x_name = self._x_axis_name or "t"
            active = self._overlay_panel.selected_datasets()

            # Get per-dataset enabled species/columns from overlay panel
            enabled_by_dataset = self._overlay_panel.selected_dataset_species()

            for dataset_name in active:
                payload = self._overlay_datasets.get(dataset_name)
                if not payload:
                    continue
                species_payload = payload["species"]
                enabled_for_dataset = enabled_by_dataset.get(dataset_name)
                enabled_species = None
                if enabled_for_dataset is not None:
                    enabled_species = {
                        key: species_payload[key]
                        for key in enabled_for_dataset
                        if key in species_payload
                    }
                resolved_x_column, base_x_array = self._resolve_overlay_x_source(x_name, payload)
                if base_x_array is None:
                    warnings.append(f"{dataset_name}: missing '{x_name}' values")
                    continue
                if base_x_array.size == 0:
                    warnings.append(f"{dataset_name}: '{resolved_x_column or x_name}' has no data")
                    continue

                for species in selected_series:
                    # Enabled dataset columns are authoritative for what can render/export.
                    # Species-X views skip silently when the enabled subset does not expose
                    # a given simulation species. Time-axis views still warn on true dataset
                    # mismatches, so probe the full payload only to classify misses.
                    resolved_key = None
                    y_source = None
                    if enabled_species is not None:
                        resolved_key, y_source = _resolve_dataset_species(species, enabled_species)
                    else:
                        resolved_key, y_source = _resolve_dataset_species(species, species_payload)
                    if y_source is None:
                        if enabled_species is not None:
                            if x_name != "t":
                                continue
                            full_resolved_key, _ = _resolve_dataset_species(species, species_payload)
                            if full_resolved_key is not None:
                                continue
                        available = sorted(species_payload.keys())
                        warnings.append(
                            f"{dataset_name}: no column matching species '{species}'. "
                            f"Available: {', '.join(available[:5])}" +
                            (f" (and {len(available) - 5} more)" if len(available) > 5 else "")
                        )
                        continue
                    if enabled_species is not None and resolved_key not in enabled_species:
                        # Guard against future resolver changes selecting a disabled column.
                        continue

                    observation_spec = None
                    if x_name == "t":
                        raw_obs = payload.get("observations")
                        if isinstance(raw_obs, dict):
                            observation_spec = raw_obs.get(resolved_key) or raw_obs.get(species)
                    if isinstance(observation_spec, dict):
                        entry_x_array = np.asarray(observation_spec.get("t", []), dtype=float).reshape(-1)
                        entry_y_array = np.asarray(observation_spec.get("y", []), dtype=float).reshape(-1)
                    else:
                        entry_x_array = base_x_array
                        entry_y_array = np.asarray(y_source, dtype=float).reshape(-1)
                    if entry_y_array.size == 0:
                        warnings.append(f"{dataset_name}: '{resolved_key}' has no data")
                        continue
                    if entry_y_array.shape[0] != entry_x_array.shape[0]:
                        warnings.append(
                            f"{dataset_name}: '{resolved_key}' length ({entry_y_array.shape[0]}) != '{x_name}' ({entry_x_array.shape[0]})"
                        )
                        continue
                    overlays.append(
                        _OverlaySeries(
                            dataset_name,
                            species,
                            entry_x_array,
                            entry_y_array,
                            x_name,
                            resolved_x_column,
                            str(resolved_key),
                        )
                    )

                if x_name == "t":
                    selected_columns = (
                        set(enabled_species.keys())
                        if enabled_species is not None
                        else set(species_payload.keys())
                    )
                    resolved_columns = {entry.resolved_y_column for entry in overlays if entry.dataset == dataset_name}
                    color_manager = ColorManager.instance()
                    known_species = self._active_overlay_known_species()
                    for raw_column in sorted(selected_columns - resolved_columns):
                        current_roster_species = (
                            color_manager.resolve_current_species_key(
                                str(raw_column),
                                known_species=known_species,
                            )
                            if known_species
                            else color_manager.resolve_current_species_key(str(raw_column))
                        )
                        if (
                            color_manager.resolve_known_species_key(str(raw_column), known_species) is not None
                            or current_roster_species is not None
                        ):
                            continue
                        y_source = species_payload.get(raw_column)
                        if y_source is None:
                            continue
                        y_array = np.asarray(y_source, dtype=float).reshape(-1)
                        if y_array.size == 0:
                            warnings.append(f"{dataset_name}: '{raw_column}' has no data")
                            continue
                        if y_array.shape[0] != base_x_array.shape[0]:
                            warnings.append(
                                f"{dataset_name}: '{raw_column}' length ({y_array.shape[0]}) != '{x_name}' ({base_x_array.shape[0]})"
                            )
                            continue
                        overlays.append(
                            _OverlaySeries(
                                dataset_name,
                                str(raw_column),
                                base_x_array,
                                y_array,
                                x_name,
                                resolved_x_column,
                                str(raw_column),
                            )
                        )

            return overlays, warnings

        def _draw_overlay_series(self, overlays: List[_OverlaySeries]) -> None:
            """Render overlay scatter markers using species-owned colors and dataset markers."""
            self._visible_overlay_series = self._overlay_series_for_current_roster(overlays)
            # Get current dataset styling from overlay panel (size and opacity)
            style = self._overlay_panel.dataset_style()
            active_overlay_keys: Set[Tuple[str, str]] = set()

            for entry in self._visible_overlay_series:
                color = self._overlay_display_color(entry.resolved_y_column)
                symbol = self._overlay_symbols.get(entry.dataset, 'o')

                # Apply user-configured opacity to alpha channel
                alpha = style.opacity
                brush = pg.mkBrush(color[0], color[1], color[2], alpha)
                pen = pg.mkPen(color=color, width=1.4)

                name = f"{entry.dataset}: {entry.species}"
                x_plot, y_plot = self._sample_xy(
                    np.asarray(entry.x, dtype=float).reshape(-1),
                    np.asarray(entry.y, dtype=float).reshape(-1),
                )
                overlay_key = (entry.dataset, entry.species)
                active_overlay_keys.add(overlay_key)
                self._upsert_dataset_overlay_item(
                    key=overlay_key,
                    x_data=x_plot,
                    y_data=y_plot,
                    pen=pen,
                    brush=brush,
                    size=style.size,
                    symbol=symbol,
                    name=name,
                )
            self._prune_dataset_overlay_items(active_overlay_keys)

        def build_visible_export(self, scope: str) -> Tuple[List[str], List[List[object]]]:
            """
            Build header + rows for CSV export including overlays.

            Parameters
            ----------
            scope : str
                "axis" to use current plot-axis selections, "all" for all series.
            """
            if self._copy_all_export_plan_provider is not None:
                raise ValueError(
                    "Main simulation plot export must be requested through the active simulation display transaction."
                )

            series = self._series
            if not series:
                raise ValueError("No simulation data available to export.")

            if scope == "axis":
                axis_candidate_names = self._axis_scope_series_names()
                if not axis_candidate_names:
                    raise ValueError("Select at least one Y-series before exporting.")
                overlay_series = list(self._visible_overlay_series)
                warnings = list(self._visible_overlay_warnings)
            else:
                self._ensure_export_all_overlay_cache()
                axis_candidate_names = list(series.keys())
                overlay_series = list(self._export_all_overlay_series)
                warnings = list(self._export_all_overlay_warnings)

            x_name = self._x_axis_name or "t"
            x_data, derived_label = self._get_x_data()
            if x_data is None:
                raise ValueError(f"The selected X-axis '{x_name}' has no data to export.")

            x_array = np.asarray(x_data, dtype=float).reshape(-1)
            if x_array.size == 0:
                raise ValueError("X-axis has no points to export.")

            x_header = derived_label or x_name
            blocks: List[List[Tuple[str, np.ndarray]]] = []
            primary_y_names = self._current_primary_renderable_series_names(
                axis_candidate_names,
                require_visible=False,
            )
            if primary_y_names:
                primary_columns: List[Tuple[str, np.ndarray]] = [(x_header, x_array)]
                for name in primary_y_names:
                    arr = np.asarray(series[name], dtype=float).reshape(-1)
                    if arr.shape[0] != x_array.shape[0]:
                        continue
                    primary_columns.append((self._series_header_text(name), arr))
                if len(primary_columns) > 1:
                    blocks.append(primary_columns)

            for layer in list(self._simulation_overlays or []):
                if not isinstance(layer, PlotDisplayLayer):
                    continue
                if not bool(layer.visible):
                    continue
                block_label = self._overlay_display_label(layer)
                overlay_series_map = self._overlay_y_map(layer)
                if not isinstance(overlay_series_map, dict):
                    continue
                if x_name == "t":
                    x_overlay = self._overlay_x_values(layer)
                else:
                    x_overlay = overlay_series_map.get(x_name)
                x_overlay_array = _try_1d_float_array(x_overlay)
                if x_overlay_array.size == 0:
                    continue
                overlay_y_names = self._series_names_compatible_with_x(
                    self._simulation_overlay_candidate_series_names(layer, axis_candidate_names),
                    overlay_series_map,
                    x_overlay_array,
                    require_visible=False,
                )
                if not overlay_y_names:
                    continue
                overlay_columns: List[Tuple[str, np.ndarray]] = [
                    (f"{block_label}::{x_header}", x_overlay_array)
                ]
                for name in overlay_y_names:
                    arr = np.asarray(overlay_series_map[name], dtype=float).reshape(-1)
                    if arr.shape[0] != x_overlay_array.shape[0]:
                        continue
                    overlay_columns.append((f"{block_label}::{self._series_header_text(name)}", arr))
                if len(overlay_columns) > 1:
                    blocks.append(overlay_columns)

            active_overlays = self._overlay_panel.selected_datasets()
            if warnings and active_overlays:
                warning_msg = "\n".join(f" - {msg}" for msg in warnings)
                raise ValueError(
                    "Cannot export overlay datasets until issues are resolved:\n" + warning_msg
                )

            for entry in overlay_series:
                x_header_ds = f"{entry.dataset}::{x_header}"
                y_header_ds = f"{entry.dataset}::{self._series_header_text(entry.species)}"
                blocks.append([(x_header_ds, entry.x), (y_header_ds, entry.y)])

            columns: List[Tuple[str, np.ndarray]] = [col for block in blocks for col in block]
            if not columns:
                raise ValueError("No valid Y-series found to export.")

            max_len = max(col[1].shape[0] for col in columns)
            header = [col[0] for col in columns]
            rows: List[List[object]] = []
            for idx in range(max_len):
                row: List[object] = []
                for _, values in columns:
                    if idx < values.shape[0]:
                        row.append(values[idx])
                    else:
                        row.append("")
                rows.append(row)

            return header, rows

        def _set_dataset_axis_labels(self, *, xlabel: str, ylabel: str) -> None:
            x_text = str(xlabel or "Time")
            y_text = str(ylabel or "Concentration")
            x_units = "s" if x_text == "Time" else None
            y_units = "M" if y_text == "Concentration" else None
            self._plot_item.setLabel("bottom", x_text, units=x_units)
            self._plot_item.setLabel("left", y_text, units=y_units)

        def _update_toolbar(self):
            """Update AxisToolbar with available variables."""
            if self._t is None:
                return

            # X-axis candidates: time + all species
            x_candidates = ["t"] + list(self._series.keys())
            if self._x_axis_name not in x_candidates:
                self._x_axis_name = "t"
            self._toolbar.set_x_candidates(x_candidates, default=self._x_axis_name)

            # Y-axis candidates: all species (checked by default if visible)
            y_candidates = [(name, self._visible.get(name, True)) for name in self._series.keys()]
            scalar_names = [
                name for name in sorted(self._scalar_values.keys())
                if name not in self._series
            ]
            for name in scalar_names:
                y_candidates.append((name, False))
            self._toolbar.set_y_candidates(
                y_candidates,
                disabled=scalar_names,
            )

        def visible_series(self):
            """Get list of visible series names."""
            return [name for name, vis in self._visible.items() if vis]

        def set_series_visible(self, name: str, visible: bool) -> None:
            """
            Set visibility of a series.

            Parameters
            ----------
            name : str
                Series name
            visible : bool
                Whether the series should be visible
            """
            if name not in self._series:
                return
            prev = self._visible.get(name, True)
            self._visible[name] = bool(visible)
            if prev != self._visible[name]:
                self._project_y_selection_to_toolbar()
                self.seriesVisibilityChanged.emit(name, self._visible[name])
                self._update_plot()

        def visible(self, name: str) -> bool:
            """
            Check if a series is visible.

            Parameters
            ----------
            name : str
                Series name

            Returns
            -------
            bool
                True if visible, False otherwise
            """
            return self._visible.get(name, False)

        def _get_x_data(self) -> Tuple[Optional[np.ndarray], str]:
            """
            Get X-axis data and label based on current selection.

            Returns
            -------
            tuple
                (x_data array, x_label string)
            """
            if self._x_axis_name == "t":
                return self._t, "Time (s)"
            elif self._x_axis_name in self._series:
                # X-axis is one of the species
                return self._series[self._x_axis_name], f"[{self._x_axis_name}] (M)"
            else:
                # Unknown X-axis, fall back to time
                return self._t, "Time (s)"

        def _on_x_axis_changed(self, name: str) -> None:
            """Handle X-axis selection change from toolbar."""
            self._x_axis_name = name
            self._update_plot()

        def _on_y_selection_changed(self, selected: List[str]) -> None:
            """Handle Y-axis selection change from toolbar."""
            self._y_selection_user_touched = True
            # Update visibility based on toolbar selection
            for series_name in self._series.keys():
                self._visible[series_name] = series_name in selected
            self._update_plot()

        def _on_view_range_changed_manually(self, *_args) -> None:
            """Switch to manual range mode after PyQtGraph reports user range interaction."""
            toolbar = getattr(self, "_toolbar", None)
            if toolbar is None:
                return
            self._sync_toolbar_manual_ranges_to_view()
            if not bool(self._auto_range_enabled):
                return
            self._auto_range_enabled = False
            self._project_toolbar_auto_range(False)

        def _project_toolbar_auto_range(self, enabled: bool) -> None:
            """Project the plot-owned range mode into the toolbar."""
            self._toolbar.set_auto_range(bool(enabled))

        def _sync_toolbar_manual_ranges_to_view(self) -> None:
            """Copy the current plot view limits into toolbar manual range fields."""
            x_range, y_range = self._plot_item.viewRange()
            ranges = (
                float(x_range[0]),
                float(x_range[1]),
                float(y_range[0]),
                float(y_range[1]),
            )
            self._manual_range_values = ranges
            self._toolbar.set_manual_ranges(*ranges)

        def _on_parametric_toggled(self, enabled: bool) -> None:
            """Handle parametric mode toggle from toolbar."""
            self._parametric_mode = enabled
            self._update_plot()

        def _on_toolbar_option_requested(self, action: str, data: object) -> None:
            """Handle option menu selections from the axis toolbar."""
            action_key = str(action).strip().lower()
            if action_key == "sampling":
                mode = str(data).strip().lower()
                if mode in {"dense", "coarse"}:
                    self._sampling_mode = mode
                    self._update_plot()
                return
            if action_key == "export_scope":
                scope = str(data).strip().lower()
                if scope in {"axis", "visible"}:
                    self._export_scope_preference = "axis"
                elif scope == "all":
                    self._export_scope_preference = "all"
                return
            logger.debug("Unhandled toolbar option: %s (%s)", action, data)

        def _add_guide_line(self, value: float) -> None:
            guide = pg.InfiniteLine(
                angle=0,
                pos=value,
                movable=True,
                pen=pg.mkPen(color=(150, 0, 150), width=1, style=Qt.DashLine),
            )
            guide.setZValue(10)
            self._plot_item.addItem(guide, ignoreBounds=True)
            self._guide_items.append(guide)

        def _on_add_guide_requested(self, from_scalar: object) -> None:
            """Add a horizontal guide line at a user-provided value."""
            scalars = self._scalar_values or {}
            scalar_name: Optional[str] = None

            if isinstance(from_scalar, str) and from_scalar:
                scalar_name = from_scalar
            elif scalars:
                options = []
                option_map: Dict[str, str] = {}
                for name, value in sorted(scalars.items(), key=lambda item: item[0]):
                    label = f"{name} = {value:.6g}"
                    options.append(label)
                    option_map[label] = name
                selection, ok = QtWidgets.QInputDialog.getItem(
                    self,
                    "Add Guide",
                    "Select scalar for guide:",
                    options,
                    0,
                    False,
                )
                if not ok or not selection:
                    return
                scalar_name = option_map.get(selection)

            if scalar_name and scalar_name in scalars:
                self._add_guide_line(float(scalars[scalar_name]))
                return

            label = "Guide value"
            if isinstance(from_scalar, str) and from_scalar:
                label = f"Value for {from_scalar}"

            value, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Add Guide",
                label,
                0.0,
                -1e12,
                1e12,
                6,
            )
            if not ok:
                return
            self._add_guide_line(value)

        def _toggle_legend(self, visible: bool) -> None:
            """Toggle legend visibility."""
            self._legend_visible = visible
            if self._legend is not None:
                self._legend.setVisible(visible)
            logger.debug(f"Legend visibility: {visible}")

        def _on_overlay_selection_changed(self, _names: List[str]) -> None:
            """Redraw when overlay checklist changes."""
            self._assign_colors()
            self._update_plot()

        def _on_overlay_style_changed(self) -> None:
            """Redraw when dataset point styling changes."""
            self._update_plot()

        def _on_axis_range_changed(self) -> None:
            """Handle axis range change from toolbar (auto/manual toggle or manual values changed)."""
            auto_enabled = bool(self._toolbar.is_auto_range())
            if auto_enabled:
                ranges = self._manual_range_values
            else:
                ranges = self._toolbar.get_manual_ranges()
                self._manual_range_values = ranges
            self._auto_range_enabled = auto_enabled
            logger.debug("Axis range changed: auto=%s, ranges=%s", auto_enabled, ranges)
            self._apply_toolbar_axis_range_command()

        def _apply_toolbar_axis_range_command(self) -> None:
            """Apply the current toolbar range command to the plot-owned ViewBox."""
            if self._auto_range_enabled:
                # Auto range mode: let PyQtGraph auto-scale
                self._plot_item.enableAutoRange()
                logger.debug("Applied auto range")
            else:
                x_min, x_max, y_min, y_max = self._manual_range_values

                # Apply X range if both values are provided
                if x_min is not None and x_max is not None:
                    self._plot_item.setXRange(x_min, x_max, padding=0)
                    logger.debug(f"Applied manual X range: [{x_min}, {x_max}]")

                # Apply Y range if both values are provided
                if y_min is not None and y_max is not None:
                    self._plot_item.setYRange(y_min, y_max, padding=0)
                    logger.debug(f"Applied manual Y range: [{y_min}, {y_max}]")

        def clear(self):
            """Clear all plot data."""
            self._plot_item.clear()
            self._t = None
            self._series = {}
            self._visible = {}
            self._colors = {}
            self._plot_items = {}
            self._plot_item_signatures = {}
            self._dataset_scatter_items = {}
            self._dataset_model_items = {}
            self._overlay_items = {}
            self._active_overlay_series = []
            self._visible_overlay_series = []
            self._export_all_overlay_series_unfiltered = []
            self._export_all_overlay_series = []
            self._active_overlay_warnings = []
            self._visible_overlay_warnings = []
            self._export_all_overlay_warnings_unfiltered = []
            self._export_all_overlay_warnings = []
            self._export_all_overlay_cache_dirty = True
            self._annotations = []
            self._intervention_annotations = []
            self._intervention_annotation_items = []
            self._guide_items = []
            self._scalar_values = {}
            self._y_selection_user_touched = False
            self._preserved_y_selection_visibility = {}
            self._auto_range_enabled = True
            self._manual_range_values = (None, None, None, None)
            self._simulation_set_label = None
            self._simulation_set_id = None
            self._simulation_layer_id = None
            self._simulation_overlays = []
            self._reference_layers_hydratable = False
            self._owned_species_keys = set()
            self._owned_species_roster_explicit = False
            self._toolbar.set_y_candidates([])
            self._toolbar.set_manual_ranges(None, None, None, None)
            self._project_toolbar_auto_range(True)

        # ==================== Plot Enhancements (v0.2.0) ====================

        def _show_context_menu(self, position):
            """Show context menu for plot enhancements."""
            menu = QtWidgets.QMenu(self)

            # Log scale actions
            log_menu = menu.addMenu("Log Scale")

            log_x_action = log_menu.addAction("Log X-Axis")
            log_x_action.setCheckable(True)
            log_x_action.setChecked(self._log_x)
            log_x_action.triggered.connect(self._toggle_log_x)

            log_y_action = log_menu.addAction("Log Y-Axis")
            log_y_action.setCheckable(True)
            log_y_action.setChecked(self._log_y)
            log_y_action.triggered.connect(self._toggle_log_y)

            if self._enable_axis_inversion_actions:
                direction_menu = menu.addMenu("Axis Direction")

                invert_x_action = direction_menu.addAction("Invert X-Axis")
                invert_x_action.setCheckable(True)
                invert_x_action.setChecked(self._invert_x_axis)
                invert_x_action.triggered.connect(self._toggle_invert_x)

                invert_y_action = direction_menu.addAction("Invert Y-Axis")
                invert_y_action.setCheckable(True)
                invert_y_action.setChecked(self._invert_y_axis)
                invert_y_action.triggered.connect(self._toggle_invert_y)

            menu.addSeparator()

            if self._enable_reference_layer_toggle_action:
                ghost_action = menu.addAction("Show Canonical Reference Lines")
                ghost_action.setCheckable(True)
                ghost_action.setChecked(self._reference_layers_visible_from_projection())
                ghost_action.setEnabled(self._reference_layer_action_available())
                ghost_action.toggled.connect(self.request_reference_layers_visible)
                menu.addSeparator()

            # Axis range actions
            axis_range_action = menu.addAction("Custom Axis Ranges...")
            axis_range_action.triggered.connect(self._show_axis_range_dialog)

            menu.addSeparator()

            # Annotation actions
            annotation_menu = menu.addMenu("Annotations")
            show_intervention_action = annotation_menu.addAction("Show Intervention Schedule Annotations")
            show_intervention_action.setCheckable(True)
            show_intervention_action.setChecked(bool(self._show_intervention_annotations))
            show_intervention_action.triggered.connect(self.set_intervention_annotations_visible)

            add_annotation_action = annotation_menu.addAction("Add Text Annotation...")
            add_annotation_action.triggered.connect(self._add_annotation)

            if self._annotations:
                clear_annotations_action = annotation_menu.addAction("Clear All Annotations")
                clear_annotations_action.triggered.connect(self._clear_annotations)

            menu.addSeparator()

            if self._copy_all_export_plan_provider is not None:
                copy_all_action = menu.addAction("Copy All")
                copy_all_action.triggered.connect(self._copy_all)

            if self._enable_copy_visible_data_action:
                copy_visible_action = menu.addAction("Copy Visible Data")
                copy_visible_action.triggered.connect(self._copy_visible_data)

            export_action = menu.addAction("Export Plot...")
            export_action.triggered.connect(self._export_plot)
            mouse_menu = menu.addMenu("Mouse Mode")
            vb = self._plot_item.getViewBox()

            pan_action = mouse_menu.addAction("Pan (3-Button)")
            pan_action.triggered.connect(
                partial(vb.setMouseMode, pg.ViewBox.PanMode)
            )

            rect_action = mouse_menu.addAction("Rect Zoom (1-Button)")
            rect_action.triggered.connect(
                partial(vb.setMouseMode, pg.ViewBox.RectMode)
            )

            menu.addSeparator()

            # Reset view action
            reset_action = menu.addAction("Reset View")
            reset_action.triggered.connect(self._reset_view)

            # Show menu at cursor position
            menu.exec_(self._plot_widget.mapToGlobal(position))

        def _reference_layer_action_available(self) -> bool:
            return bool(getattr(self, "_reference_layers_hydratable", False)) or self._has_reference_layers()

        def _has_reference_layers(self) -> bool:
            for layer in (self._simulation_overlays or []):
                if not isinstance(layer, PlotDisplayLayer):
                    continue
                if self._is_reference_layer(layer):
                    return True
            return False

        def _reference_layers_visible_from_projection(self) -> bool:
            reference_layers = [
                layer
                for layer in (self._simulation_overlays or [])
                if isinstance(layer, PlotDisplayLayer) and self._is_reference_layer(layer)
            ]
            if not reference_layers:
                return False
            return all(bool(layer.visible) for layer in reference_layers)

        def _apply_axis_inversion_state(self) -> None:
            viewbox = self._plot_item.getViewBox()
            viewbox.invertX(self._invert_x_axis)
            viewbox.invertY(self._invert_y_axis)

        def _toggle_invert_x(self):
            """Toggle X-axis inversion."""
            self._invert_x_axis = not self._invert_x_axis
            self._apply_axis_inversion_state()
            logger.info(f"X-axis inverted: {self._invert_x_axis}")

        def _toggle_invert_y(self):
            """Toggle Y-axis inversion."""
            self._invert_y_axis = not self._invert_y_axis
            self._apply_axis_inversion_state()
            logger.info(f"Y-axis inverted: {self._invert_y_axis}")

        def _toggle_log_x(self):
            """Toggle X-axis log scale."""
            self._log_x = not self._log_x
            self._plot_item.setLogMode(x=self._log_x, y=self._log_y)
            logger.info(f"X-axis log scale: {self._log_x}")

        def _toggle_log_y(self):
            """Toggle Y-axis log scale."""
            self._log_y = not self._log_y
            self._plot_item.setLogMode(x=self._log_x, y=self._log_y)
            logger.info(f"Y-axis log scale: {self._log_y}")

        def _show_axis_range_dialog(self):
            """Show dialog for custom axis ranges."""
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Custom Axis Ranges")
            dialog.setModal(True)

            layout = QtWidgets.QFormLayout(dialog)

            # Get current ranges
            x_range = self._plot_item.viewRange()[0]
            y_range = self._plot_item.viewRange()[1]

            # X-axis range
            x_min_spin = QtWidgets.QDoubleSpinBox()
            x_min_spin.setDecimals(15)
            x_min_spin.setRange(-1e10, 1e10)
            x_min_spin.setValue(x_range[0])

            x_max_spin = QtWidgets.QDoubleSpinBox()
            x_max_spin.setDecimals(15)
            x_max_spin.setRange(-1e10, 1e10)
            x_max_spin.setValue(x_range[1])

            # Y-axis range
            y_min_spin = QtWidgets.QDoubleSpinBox()
            y_min_spin.setDecimals(15)
            y_min_spin.setRange(-1e10, 1e10)
            y_min_spin.setValue(y_range[0])

            y_max_spin = QtWidgets.QDoubleSpinBox()
            y_max_spin.setDecimals(15)
            y_max_spin.setRange(-1e10, 1e10)
            y_max_spin.setValue(y_range[1])

            layout.addRow("X Min:", x_min_spin)
            layout.addRow("X Max:", x_max_spin)
            layout.addRow("Y Min:", y_min_spin)
            layout.addRow("Y Max:", y_max_spin)

            # Buttons
            button_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addRow(button_box)

            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                ranges = (
                    float(x_min_spin.value()),
                    float(x_max_spin.value()),
                    float(y_min_spin.value()),
                    float(y_max_spin.value()),
                )
                self._auto_range_enabled = False
                self._manual_range_values = ranges
                self._project_toolbar_auto_range(False)
                # Apply ranges
                self._plot_item.setRange(
                    xRange=(ranges[0], ranges[1]),
                    yRange=(ranges[2], ranges[3]),
                    padding=0,
                )
                self._toolbar.set_manual_ranges(*ranges)
                logger.info(
                    f"Custom axis ranges applied: X=[{x_min_spin.value()}, {x_max_spin.value()}], "
                    f"Y=[{y_min_spin.value()}, {y_max_spin.value()}]"
                )

        def _add_annotation(self):
            """Add text annotation to plot."""
            # Get annotation text
            text, ok = QtWidgets.QInputDialog.getText(
                self,
                "Add Annotation",
                "Enter annotation text:"
            )

            if not ok or not text:
                return

            # Get position (center of current view)
            view_range = self._plot_item.viewRange()
            x_pos = (view_range[0][0] + view_range[0][1]) / 2
            y_pos = (view_range[1][0] + view_range[1][1]) / 2

            # Create text item
            annotation = pg.TextItem(
                text=text,
                color='k' if not self._dark_mode else 'w',
                anchor=(0.5, 0.5),
                border=pg.mkPen('k' if not self._dark_mode else 'w'),
                fill=pg.mkBrush(255, 255, 255, 200) if not self._dark_mode else pg.mkBrush(30, 30, 30, 200)
            )
            annotation.setPos(x_pos, y_pos)

            # Make it draggable (by setting movable flag)
            # Note: TextItem doesn't have built-in drag support, but we can add it
            self._plot_item.addItem(annotation)
            self._annotations.append(annotation)

            logger.info(f"Added annotation: {text} at ({x_pos}, {y_pos})")

        def _clear_annotations(self):
            """Clear all annotations from plot."""
            for annotation in self._annotations:
                self._plot_item.removeItem(annotation)
            self._annotations.clear()
            logger.info("Cleared all annotations")

        def _export_plot(self):
            """Open pyqtgraph's export dialog for the plot scene."""
            scene = self._plot_item.scene()
            scene.contextMenuItem = self._plot_item
            scene.showExportDialog()

        def _reset_view(self):
            """Reset plot view to auto range."""
            self._auto_range_enabled = True
            self._project_toolbar_auto_range(True)
            self._plot_item.enableAutoRange(x=True, y=True)
            self._plot_item.autoRange()
            self._sync_toolbar_manual_ranges_to_view()
            self._apply_axis_inversion_state()
            logger.info("Reset plot view to auto range")

        def set_theme(self, dark_mode: bool = False):
            """
            Apply color theme to plot.

            Parameters
            ----------
            dark_mode : bool
                If True, use dark theme colors
            """
            self._dark_mode = bool(dark_mode)

            if dark_mode:
                # Dark theme
                self._plot_widget.setBackground('#1e1e1e')
                self._plot_item.getAxis('bottom').setPen('#e0e0e0')
                self._plot_item.getAxis('left').setPen('#e0e0e0')
                self._plot_item.getAxis('bottom').setTextPen('#e0e0e0')
                self._plot_item.getAxis('left').setTextPen('#e0e0e0')

                # Grid color
                self._plot_item.showGrid(x=True, y=True, alpha=0.2)
            else:
                # Light theme
                self._plot_widget.setBackground('w')
                self._plot_item.getAxis('bottom').setPen('k')
                self._plot_item.getAxis('left').setPen('k')
                self._plot_item.getAxis('bottom').setTextPen('k')
                self._plot_item.getAxis('left').setTextPen('k')

                # Grid color
                self._plot_item.showGrid(x=True, y=True, alpha=0.3)

            # Refresh plot
            self._update_plot()

else:
    # PyQtGraph not available - provide a stub that warns users
    class PyQtGraphPlotPanel(QtWidgets.QWidget):
        """Stub class when PyQtGraph is not available."""

        # Signal emitted when series visibility changes
        seriesVisibilityChanged = QtCore.Signal(str, bool)
        referenceLayerVisibilityRequested = QtCore.Signal(bool)

        def __init__(
            self,
            parent: Optional[QtWidgets.QWidget] = None,
            *,
            embed_analysis_tabs: bool = True,
            workspace_splitter_object_name: Optional[str] = None,
            enable_axis_inversion_actions: bool = False,
            enable_reference_layer_toggle_action: bool = False,
            enable_copy_visible_data_action: bool = False,
        ):
            super().__init__(parent)
            _ = workspace_splitter_object_name
            _ = enable_axis_inversion_actions
            _ = enable_reference_layer_toggle_action
            _ = enable_copy_visible_data_action
            layout = QtWidgets.QVBoxLayout(self)
            layout.addWidget(make_pyqtgraph_fallback_widget(self))
            self._main_splitter = None
            self._details_tabs = None
            self._analysis_tabs_detached = not bool(embed_analysis_tabs)

        def set_data(self, t, series, **_kwargs):
            """Stub method."""
            pass

        def set_display_layers(self, payload):
            self.clear()

        def request_reference_layers_visible(self, visible):
            self.referenceLayerVisibilityRequested.emit(bool(visible))

        def set_intervention_annotations_from_provenance(self, provenance):
            """Stub method."""
            pass

        def set_intervention_annotations_visible(self, visible):
            """Stub method."""
            pass

        def render_dataset_layers(
            self,
            *,
            data_t,
            dataset_series,
            model_t=None,
            model_series=None,
            visible_species=(),
            xlabel="Time",
            ylabel="Concentration",
        ):
            """Stub method."""
            pass

        def clear(self):
            """Stub method."""
            pass

        def set_theme(self, dark_mode=False):
            """Stub method."""
            pass

        def visible_series(self):
            """Stub method."""
            return []

        def set_series_visible(self, name: str, visible: bool) -> None:
            """Stub method."""
            pass

        def visible(self, name: str) -> bool:
            """Stub method."""
            return False

        def set_overlay_catalog(self, datasets):
            """Stub method."""
            pass

        def active_overlays(self):
            """Stub method."""
            return []

        def overlay_snapshot(self):
            """Stub method."""
            return {}

        def build_visible_export(self, scope: str):
            """Stub method."""
            raise RuntimeError("PyQtGraph is required for overlay exports.")

        def analysis_tabs_widget(self):
            """Stub method."""
            return None

        def workspace_splitter(self):
            """Stub method."""
            return None

        def detach_analysis_tabs_for_dock(self):
            """Stub method."""
            return None
