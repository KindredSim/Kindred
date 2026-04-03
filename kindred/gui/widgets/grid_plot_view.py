# kindred/gui/widgets/grid_plot_view.py
"""
Grid view for displaying multiple dataset fits simultaneously.

Shows all fitted datasets in a grid layout (e.g., 2x2, 3x2) for comparison.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Tuple, Optional, Dict, Any, Sequence
import math

import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui

from kindred.gui.color_manager import ColorManager
from kindred.gui.plot_config import try_import_pyqtgraph
from kindred.gui.diagnostics import record_best_effort_failure

logger = logging.getLogger(__name__)

__all__ = ["GridPlotView"]


class GridPlotView(QtWidgets.QWidget):
    """
    Display multiple dataset fits in a grid layout.

    Automatically arranges plots in an optimal grid (e.g., 2x2 for 4 datasets).
    Each subplot shows data + fit + fit quality metric.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        # Establish core state up front so constructor error paths are safe.
        self._legend_visible = True
        self._selected_species_list: List[str] = []  # List of selected species for display
        self._pg = None
        self._graphics_layout = None

        # Store dataset information
        self._datasets: List[Dict[str, Any]] = []
        self._plot_items: List[Any] = []
        self._plot_series_items: List[Dict[str, Any]] = []
        self._plot_legends: List[Any] = []
        self._structure_key: Optional[Tuple[Tuple[str, ...], int, int]] = None
        self._available_species_cache: Optional[Tuple[str, ...]] = None
        self._pending_datasets_lock = threading.Lock()
        self._pending_datasets_update: Optional[List[Dict[str, Any]]] = None
        self._pending_datasets_invoke_scheduled = False
        self._warned_off_thread_update = False
        self._redraw_pending = False
        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(0)
        self._redraw_timer.timeout.connect(self._apply_scheduled_redraw)

        # When locked, prevent PyQtGraph autorange from changing view ranges during live updates.
        self._autorange_locked = False
        self._locked_view_ranges: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        self._locked_autorange_prev: Dict[int, Tuple[bool, bool]] = {}
        self._locked_left_axis_width: Optional[int] = None
        self._locked_bottom_axis_height: Optional[int] = None
        self._scrollbar_policies_prev: Optional[Tuple[QtCore.Qt.ScrollBarPolicy, QtCore.Qt.ScrollBarPolicy]] = None
        self._warned_shape_mismatches: set[tuple[str, str, int, int]] = set()
        self._best_effort_failures: set[str] = set()
        self._best_effort_failure_counts: dict[str, int] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._controls_widget = QtWidgets.QWidget(self)
        self._controls_widget.setObjectName("grid_plot_view_controls")
        controls_layout = QtWidgets.QHBoxLayout(self._controls_widget)
        controls_layout.setContentsMargins(2, 2, 2, 0)
        controls_layout.setSpacing(6)

        self._legend_toggle_btn = QtWidgets.QCheckBox("Show Legend")
        self._legend_toggle_btn.setChecked(True)
        self._legend_toggle_btn.toggled.connect(self._toggle_legend)
        controls_layout.addWidget(self._legend_toggle_btn)

        species_label = QtWidgets.QLabel("Species")
        controls_layout.addWidget(species_label)
        self._species_list = QtWidgets.QListWidget()
        self._species_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self._species_list.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self._species_list.itemSelectionChanged.connect(self._on_species_selection_changed)
        species_label.setBuddy(self._species_list)
        controls_layout.addWidget(self._species_list)

        controls_layout.addStretch()
        layout.addWidget(self._controls_widget)

        ok, pg, _exc = try_import_pyqtgraph()
        self._pg = pg if ok else None

        if self._pg is None:
            # Fallback if PyQtGraph not available
            label = QtWidgets.QLabel("PyQtGraph not available\nInstall with: pip install pyqtgraph")
            label.setAlignment(QtWidgets.Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("padding: 20px;")
            layout.addWidget(label)
            return

        # Create PyQtGraph GraphicsLayoutWidget
        self._graphics_layout = self._pg.GraphicsLayoutWidget()
        # Always force a non-OpenGL viewport for this grid. On Windows, QOpenGLWidget/native
        # viewports can exhibit z-order / clipping glitches during splitter and run-state
        # transitions (e.g., "sliding under" sibling panels). This view updates frequently
        # during Global Fit, so stability is preferred over OpenGL acceleration here.
        try:
            self._graphics_layout.useOpenGL(False)
        except Exception as exc:
            self._record_best_effort_failure(
                "disable_opengl_viewport",
                message="GridPlotView: unable to disable OpenGL viewport",
                exc=exc,
            )
        layout.addWidget(self._graphics_layout)

        # Initialize with empty state
        self._schedule_redraw()

    def _record_best_effort_failure(
        self,
        key: str,
        *,
        message: str,
        exc: Optional[BaseException] = None,
        max_logs: int = 3,
    ) -> None:
        record_best_effort_failure(self, str(key), message=message, exc=exc, log=logger, max_logs=int(max_logs))

    def _is_gui_thread(self) -> bool:
        try:
            return QtCore.QThread.currentThread() == self.thread()
        except Exception:  # pragma: no cover - defensive
            return True

    @staticmethod
    def _freeze_dataset_payload(dataset: Dict[str, Any]) -> Dict[str, Any]:
        def _freeze_1d(values: object) -> np.ndarray:
            try:
                return np.asarray(values, dtype=float).reshape(-1).copy()
            except Exception:
                return np.asarray([], dtype=float)

        frozen = dict(dataset or {})
        frozen["name"] = str(frozen.get("name") or "")
        for key in ("data_x", "data_y", "model_x", "model_y"):
            if key in frozen and frozen.get(key) is not None:
                frozen[key] = _freeze_1d(frozen.get(key))
        model_series = frozen.get("model_series")
        if isinstance(model_series, dict):
            frozen["model_series"] = {str(k): _freeze_1d(v) for k, v in model_series.items()}
        all_species = frozen.get("all_species")
        if isinstance(all_species, dict):
            frozen["all_species"] = {str(k): _freeze_1d(v) for k, v in all_species.items()}
        if "current_species" in frozen:
            frozen["current_species"] = str(frozen.get("current_species") or "")
        if "x_label" in frozen:
            frozen["x_label"] = str(frozen.get("x_label") or "")
        if "x_units" in frozen:
            frozen["x_units"] = None if frozen.get("x_units") is None else str(frozen.get("x_units"))
        return frozen

    @QtCore.Slot()
    def _schedule_redraw(self) -> None:
        if self._pg is None or self._graphics_layout is None:
            return
        if not self._is_gui_thread():
            # Only schedule timers from the owning Qt thread.
            QtCore.QMetaObject.invokeMethod(
                self,
                "_schedule_redraw",
                QtCore.Qt.ConnectionType.QueuedConnection,
            )
            return
        self._redraw_pending = True
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    @QtCore.Slot()
    def _apply_pending_datasets_update_on_gui_thread(self) -> None:
        if not self._is_gui_thread():
            return
        pending: Optional[List[Dict[str, Any]]] = None
        with self._pending_datasets_lock:
            pending = self._pending_datasets_update
            self._pending_datasets_update = None
            self._pending_datasets_invoke_scheduled = False
        if pending is None:
            return
        self._datasets = pending
        self._schedule_redraw()

    @QtCore.Slot()
    def _apply_scheduled_redraw(self) -> None:
        if not self._is_gui_thread():
            return
        if not self._redraw_pending:
            return
        self._redraw_pending = False
        self._update_species_selector()
        self._update_grid()

    def _set_scrollbars_locked(self, locked: bool) -> None:
        if self._graphics_layout is None:
            return
        locked = bool(locked)
        if locked:
            if self._scrollbar_policies_prev is None:
                try:
                    self._scrollbar_policies_prev = (
                        self._graphics_layout.horizontalScrollBarPolicy(),
                        self._graphics_layout.verticalScrollBarPolicy(),
                    )
                except Exception:
                    self._scrollbar_policies_prev = None
            try:
                self._graphics_layout.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                self._graphics_layout.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            except Exception as exc:
                self._record_best_effort_failure(
                    "lock_scrollbars",
                    message="GridPlotView: failed to lock scrollbars",
                    exc=exc,
                )
            return

        # unlocked: restore prior policies (if captured)
        if self._scrollbar_policies_prev is not None:
            h_policy, v_policy = self._scrollbar_policies_prev
            try:
                self._graphics_layout.setHorizontalScrollBarPolicy(h_policy)
                self._graphics_layout.setVerticalScrollBarPolicy(v_policy)
            except Exception as exc:
                self._record_best_effort_failure(
                    "restore_scrollbars",
                    message="GridPlotView: failed to restore scrollbar policies",
                    exc=exc,
                )
        self._scrollbar_policies_prev = None

    def _snapshot_locked_axis_geometry(self) -> None:
        """
        Pick stable axis gutter sizes for the duration of a running/locked session.

        GridPlotView rebuilds PlotItems (and thus AxisItems) on each refresh; without
        fixed gutters, AxisItem text-space expansion can shift the viewport and cause
        a brief "compress" flicker as layout settles.
        """
        left_w = 0.0
        bottom_h = 0.0
        for plot in list(self._plot_items or []):
            try:
                left_axis = plot.getAxis("left")
                bottom_axis = plot.getAxis("bottom")
            except (AttributeError, KeyError, RuntimeError) as exc:
                self._record_best_effort_failure(
                    "measure_axes_missing",
                    message="GridPlotView: missing axis on plot item while snapshotting locked axis geometry",
                    exc=exc,
                )
                continue
            try:
                left_w = max(left_w, float(left_axis.boundingRect().width()))
            except Exception as exc:
                self._record_best_effort_failure(
                    "measure_left_axis_width",
                    message="GridPlotView: failed to measure left axis width while snapshotting locked axis geometry",
                    exc=exc,
                )
            try:
                bottom_h = max(bottom_h, float(bottom_axis.boundingRect().height()))
            except Exception as exc:
                self._record_best_effort_failure(
                    "measure_bottom_axis_height",
                    message="GridPlotView: failed to measure bottom axis height while snapshotting locked axis geometry",
                    exc=exc,
                )

        # Conservative defaults + headroom, clamped to avoid absurd gutters.
        if left_w <= 1.0:
            left_w = 80.0
        else:
            left_w = min(140.0, max(80.0, left_w + 12.0))
        if bottom_h <= 1.0:
            bottom_h = 45.0
        else:
            bottom_h = min(95.0, max(45.0, bottom_h + 8.0))

        self._locked_left_axis_width = int(math.ceil(left_w))
        self._locked_bottom_axis_height = int(math.ceil(bottom_h))

    def _apply_locked_axis_style(self, plot: Any) -> None:
        if not bool(getattr(self, "_autorange_locked", False)):
            return
        try:
            left_axis = plot.getAxis("left")
            bottom_axis = plot.getAxis("bottom")
        except Exception:
            return

        for axis in (left_axis, bottom_axis):
            try:
                axis.setStyle(autoExpandTextSpace=False)
            except Exception as exc:
                self._record_best_effort_failure(
                    "locked_axis_style",
                    message="GridPlotView: failed to apply locked axis style",
                    exc=exc,
                )

        if self._locked_left_axis_width is not None:
            try:
                left_axis.setWidth(float(self._locked_left_axis_width))
            except Exception as exc:
                self._record_best_effort_failure(
                    "locked_left_axis_width",
                    message="GridPlotView: failed to set locked left axis width",
                    exc=exc,
                )
        if self._locked_bottom_axis_height is not None:
            try:
                bottom_axis.setHeight(float(self._locked_bottom_axis_height))
            except Exception as exc:
                self._record_best_effort_failure(
                    "locked_bottom_axis_height",
                    message="GridPlotView: failed to set locked bottom axis height",
                    exc=exc,
                )

    def _dataset_structure_key(self, datasets: Sequence[Dict[str, Any]]) -> Tuple[Tuple[str, ...], int, int]:
        ids = tuple(str((ds or {}).get("name") or "") for ds in (datasets or []))
        n_rows, n_cols = self._calculate_grid_size(len(ids))
        return (ids, int(n_rows), int(n_cols))

    def _configure_plot_static(self, plot: Any) -> None:
        plot.setLabel("bottom", "Time", units="s")
        plot.setLabel("left", "Concentration", units="M")
        plot.showGrid(x=True, y=True, alpha=0.3)
        font = QtGui.QFont()
        font.setPointSize(7)
        for axis in ["left", "bottom"]:
            try:
                plot.getAxis(axis).setStyle(tickFont=font)
                plot.getAxis(axis).setTickFont(font)
            except Exception as exc:
                self._record_best_effort_failure(
                    f"tick_font:{axis}",
                    message=f"GridPlotView: failed to set tick font for axis={axis}",
                    exc=exc,
                )

    def _ensure_legend_state(self, plot: Any, idx: int) -> None:
        want = bool(self._legend_visible and len(self._selected_species_list) > 1)
        legend = self._plot_legends[idx] if idx < len(self._plot_legends) else None
        if want:
            if legend is None:
                try:
                    legend = plot.addLegend(offset=(5, 5))
                except Exception:
                    legend = None
                if idx < len(self._plot_legends):
                    self._plot_legends[idx] = legend
            if legend is not None:
                try:
                    legend.setVisible(True)
                except Exception as exc:
                    self._record_best_effort_failure(
                        "legend_show",
                        message="GridPlotView: failed to show legend",
                        exc=exc,
                    )
        else:
            if legend is not None:
                try:
                    legend.setVisible(False)
                except Exception as exc:
                    self._record_best_effort_failure(
                        "legend_hide",
                        message="GridPlotView: failed to hide legend",
                        exc=exc,
                    )

    def _ensure_curve_item(self, plot: Any, idx: int, key: str, *, is_model: bool, species_name: str, color: Tuple[int, int, int]) -> Any:
        series_map = self._plot_series_items[idx]
        item = series_map.get(key)
        if item is not None:
            return item
        if self._pg is None:
            return None

        if is_model:
            item = plot.plot(
                [],
                [],
                pen=self._pg.mkPen(color=color, width=2),
                name=species_name,
            )
        else:
            item = plot.plot(
                [],
                [],
                pen=None,
                symbol="o",
                symbolSize=5,
                symbolBrush=self._pg.mkBrush(*color, 150),
                symbolPen=self._pg.mkPen(color=color, width=1),
                name=None,
            )
        series_map[key] = item
        return item

    def _update_plot_data_in_place(self) -> None:
        if self._pg is None or self._graphics_layout is None:
            return
        if not self._plot_items or len(self._plot_items) != len(self._datasets):
            return
        if len(self._plot_series_items) != len(self._plot_items):
            return

        color_manager = ColorManager.instance()
        available_species = []
        for dataset in self._datasets:
            all_species = dataset.get("all_species", {}) if isinstance(dataset.get("all_species", {}), dict) else {}
            available_species.extend(str(name) for name in all_species.keys() if str(name))
        color_manager.seed_species(available_species)

        for idx, dataset in enumerate(self._datasets):
            plot = self._plot_items[idx]
            self._ensure_legend_state(plot, idx)

            name = dataset.get("name")
            x_label = str(dataset.get("x_label") or "Time")
            x_units = dataset.get("x_units")
            data_x = np.asarray(dataset.get("data_x", []), dtype=float).reshape(-1)
            chi_squared = dataset.get("chi_squared")
            r_squared = dataset.get("r_squared")
            all_species = dataset.get("all_species", {}) if isinstance(dataset.get("all_species", {}), dict) else {}
            model_x = dataset.get("model_x")
            model_y = dataset.get("model_y")
            model_series = dataset.get("model_series") or {}
            current_species = dataset.get("current_species")

            try:
                if x_units:
                    plot.setLabel("bottom", x_label, units=str(x_units))
                else:
                    plot.setLabel("bottom", x_label)
                plot.setLabel("left", "Concentration", units="M")
            except Exception as exc:
                self._record_best_effort_failure(
                    "plot_labels",
                    message=f"GridPlotView: failed to update plot labels for dataset={name!r}",
                    exc=exc,
                )

            # Title with fit quality.
            title = str(name or "")
            if chi_squared is not None:
                try:
                    chi_val = float(chi_squared)
                except Exception:
                    chi_val = None
                if chi_val is not None:
                    title = f"{title} (χ² = {chi_val:.3e})"
                    color = self._get_color_for_chi_squared(chi_val)
                else:
                    color = (51, 51, 51)
            elif r_squared is not None:
                try:
                    r2_val = float(r_squared)
                except Exception:
                    r2_val = None
                if r2_val is not None:
                    title = f"{title} (R² = {r2_val:.3f})"
                    color = self._get_color_for_r_squared(r2_val)
                else:
                    color = (51, 51, 51)
            else:
                color = (51, 51, 51)
            try:
                plot.setTitle(title, color=color, size="10pt")
            except Exception as exc:
                self._record_best_effort_failure(
                    "plot_title",
                    message=f"GridPlotView: failed to set plot title for dataset={name!r}",
                    exc=exc,
                )

            active_keys: set[str] = set()
            for species_name in list(self._selected_species_list or []):
                color_rgb = color_manager.get_species_rgb(species_name, known_species=tuple(available_species))

                data_key = f"{species_name}::data"
                model_key = f"{species_name}::model"

                if species_name in all_species:
                    y_data = np.asarray(all_species[species_name], dtype=float).reshape(-1)
                    data_item = self._ensure_curve_item(
                        plot,
                        idx,
                        data_key,
                        is_model=False,
                        species_name=species_name,
                        color=color_rgb,
                    )
                    if data_item is not None:
                        try:
                            data_item.setSymbol("o")
                            data_item.setSymbolSize(5)
                            data_item.setSymbolBrush(self._pg.mkBrush(*color_rgb, 150))
                            data_item.setSymbolPen(self._pg.mkPen(color=color_rgb, width=1))
                            if data_x.size == y_data.size:
                                data_item.setData(data_x, y_data)
                                data_item.setVisible(True)
                            else:
                                warn_key = (str(name or ""), data_key, int(data_x.size), int(y_data.size))
                                if warn_key not in self._warned_shape_mismatches:
                                    self._warned_shape_mismatches.add(warn_key)
                                    logger.warning(
                                        "GridPlotView: skip data curve with shape mismatch dataset=%s series=%s x_len=%d y_len=%d",
                                        name,
                                        species_name,
                                        int(data_x.size),
                                        int(y_data.size),
                                    )
                                try:
                                    data_item.setVisible(False)
                                except Exception as exc:
                                    self._record_best_effort_failure(
                                        "hide_data_curve",
                                        message=(
                                            f"GridPlotView: failed to hide data curve dataset={name!r} "
                                            f"species={species_name!r}"
                                        ),
                                        exc=exc,
                                    )
                        except Exception as exc:
                            self._record_best_effort_failure(
                                "update_data_curve",
                                message=(
                                    f"GridPlotView: failed to update data curve dataset={name!r} "
                                    f"species={species_name!r}"
                                ),
                                exc=exc,
                            )
                    active_keys.add(data_key)

                    # Model overlay (multi-series preferred, fallback to single model_y).
                    y_model = None
                    if isinstance(model_series, dict) and species_name in model_series:
                        y_model = model_series[species_name]
                    elif model_y is not None and (current_species == species_name or len(self._selected_species_list) == 1):
                        y_model = model_y

                    if y_model is not None:
                        try:
                            y_model_arr = np.asarray(y_model, dtype=float).reshape(-1)
                        except Exception:
                            y_model_arr = None
                        if y_model_arr is not None:
                            x_model = model_x if model_x is not None else data_x
                            try:
                                x_model_arr = np.asarray(x_model, dtype=float).reshape(-1)
                            except Exception:
                                x_model_arr = data_x
                            model_item = self._ensure_curve_item(
                                plot,
                                idx,
                                model_key,
                                is_model=True,
                                species_name=species_name,
                                color=color_rgb,
                            )
                            if model_item is not None:
                                try:
                                    model_item.setPen(self._pg.mkPen(color=color_rgb, width=2))
                                    if x_model_arr.size == y_model_arr.size:
                                        model_item.setData(x_model_arr, y_model_arr)
                                        model_item.setVisible(True)
                                    else:
                                        warn_key = (str(name or ""), model_key, int(x_model_arr.size), int(y_model_arr.size))
                                        if warn_key not in self._warned_shape_mismatches:
                                            self._warned_shape_mismatches.add(warn_key)
                                            logger.warning(
                                                "GridPlotView: skip model curve with shape mismatch dataset=%s series=%s x_len=%d y_len=%d",
                                                name,
                                                species_name,
                                                int(x_model_arr.size),
                                                int(y_model_arr.size),
                                            )
                                        try:
                                            model_item.setVisible(False)
                                        except Exception as exc:
                                            self._record_best_effort_failure(
                                                "hide_model_curve",
                                                message=(
                                                    f"GridPlotView: failed to hide model curve dataset={name!r} "
                                                    f"species={species_name!r}"
                                                ),
                                                exc=exc,
                                            )
                                except Exception as exc:
                                    self._record_best_effort_failure(
                                        "update_model_curve",
                                        message=(
                                            f"GridPlotView: failed to update model curve dataset={name!r} "
                                            f"species={species_name!r}"
                                        ),
                                        exc=exc,
                                    )
                            active_keys.add(model_key)

            # Hide any previously created curves not active in this update.
            for key, item in list(self._plot_series_items[idx].items()):
                if key in active_keys:
                    continue
                try:
                    item.setVisible(False)
                except Exception as exc:
                    self._record_best_effort_failure(
                        "hide_inactive_curve",
                        message=f"GridPlotView: failed to hide inactive curve {key!r}",
                        exc=exc,
                    )

    def _rebuild_structure(self) -> None:
        if self._pg is None or self._graphics_layout is None:
            return
        self._graphics_layout.setUpdatesEnabled(False)
        try:
            self._graphics_layout.clear()
            self._plot_items.clear()
            self._plot_series_items.clear()
            self._plot_legends.clear()

            n_datasets = len(self._datasets)
            if n_datasets == 0:
                text = self._pg.TextItem(
                    "No datasets loaded\n\nAdd datasets using the Data Manager",
                    color=(150, 150, 150),
                    anchor=(0.5, 0.5),
                )
                placeholder_plot = self._graphics_layout.addPlot()
                placeholder_plot.hideAxis("left")
                placeholder_plot.hideAxis("bottom")
                placeholder_plot.addItem(text)
                text.setPos(0.5, 0.5)
                self._structure_key = self._dataset_structure_key([])
                return

            _n_rows, n_cols = self._calculate_grid_size(n_datasets)
            for idx in range(n_datasets):
                row = idx // n_cols
                col = idx % n_cols
                plot = self._graphics_layout.addPlot(row=row, col=col)
                self._plot_items.append(plot)
                self._plot_series_items.append({})
                self._plot_legends.append(None)
                self._configure_plot_static(plot)

                if bool(getattr(self, "_autorange_locked", False)):
                    self._apply_locked_axis_style(plot)
                    vb = plot.getViewBox()
                    try:
                        vb.enableAutoRange(x=False, y=False)
                    except Exception as exc:
                        self._record_best_effort_failure(
                            "disable_autorange_rebuild",
                            message=f"GridPlotView: failed to disable autorange while rebuilding subplot={idx}",
                            exc=exc,
                        )
                    frozen = self._locked_view_ranges.get(idx)
                    if frozen:
                        try:
                            plot.setXRange(float(frozen[0][0]), float(frozen[0][1]), padding=0)
                            plot.setYRange(float(frozen[1][0]), float(frozen[1][1]), padding=0)
                        except Exception as exc:
                            self._record_best_effort_failure(
                                "restore_frozen_range",
                                message=f"GridPlotView: failed to restore frozen view range for subplot={idx}",
                                exc=exc,
                            )

            self._structure_key = self._dataset_structure_key(self._datasets)
        finally:
            self._graphics_layout.setUpdatesEnabled(True)

    def add_dataset(
        self,
        name: str,
        data_x: np.ndarray,
        data_y: np.ndarray,
        model_x: Optional[np.ndarray] = None,
        model_y: Optional[np.ndarray] = None,
        model_series: Optional[Dict[str, np.ndarray]] = None,
        chi_squared: Optional[float] = None,
        r_squared: Optional[float] = None,
        all_species: Optional[Dict[str, np.ndarray]] = None,
        current_species: Optional[str] = None,
    ):
        """
        Add a dataset to the grid view.

        Parameters
        ----------
        name : str
            Dataset name
        data_x, data_y : np.ndarray
            Experimental data
        model_x, model_y : np.ndarray, optional
            Model fit
        model_series : dict, optional
            Multi-series model fit {species_name: y_model_aligned_to_model_x}
        chi_squared, r_squared : float, optional
            Fit quality metrics
        all_species : dict, optional
            Dictionary of all species data {name: y_array}
        current_species : str, optional
            Name of the currently displayed species
        """
        if self._pg is None or self._graphics_layout is None:
            return

        dataset = {
            'name': name,
            'data_x': np.asarray(data_x),
            'data_y': np.asarray(data_y),
            'model_x': np.asarray(model_x) if model_x is not None else None,
            'model_y': np.asarray(model_y) if model_y is not None else None,
            'model_series': {k: np.asarray(v) for k, v in (model_series or {}).items()} if model_series else None,
            'chi_squared': chi_squared,
            'r_squared': r_squared,
            'all_species': all_species if all_species else {current_species or 'Data': data_y},
            'current_species': current_species or 'Data',
        }
        # Defensive copy: never retain references to caller-owned buffers.
        self._datasets.append(self._freeze_dataset_payload(dataset))
        self._schedule_redraw()

    def set_datasets(self, datasets: Sequence[Dict[str, Any]]) -> None:
        """Replace all datasets and redraw once (avoids per-dataset redraw churn)."""
        if self._pg is None or self._graphics_layout is None:
            return

        normalized: List[Dict[str, Any]] = []
        for dataset in datasets or []:
            if not isinstance(dataset, dict):
                continue
            normalized.append(dict(dataset))

        # Defensive copy: ensure all payload arrays are owned/contiguous and detached from caller memory.
        frozen = [self._freeze_dataset_payload(ds) for ds in normalized]

        if not self._is_gui_thread():
            if not self._warned_off_thread_update:
                self._warned_off_thread_update = True
                logger.warning("GridPlotView.set_datasets called off GUI thread; marshaling update to GUI thread.")
            with self._pending_datasets_lock:
                self._pending_datasets_update = frozen
                schedule = not self._pending_datasets_invoke_scheduled
                self._pending_datasets_invoke_scheduled = True
            if schedule:
                QtCore.QMetaObject.invokeMethod(
                    self,
                    "_apply_pending_datasets_update_on_gui_thread",
                    QtCore.Qt.ConnectionType.QueuedConnection,
                )
            return

        self._datasets = frozen
        self._schedule_redraw()

    def clear_datasets(self):
        """Remove all datasets from the grid."""
        if self._pg is None or self._graphics_layout is None:
            return

        if not self._is_gui_thread():
            with self._pending_datasets_lock:
                self._pending_datasets_update = []
                schedule = not self._pending_datasets_invoke_scheduled
                self._pending_datasets_invoke_scheduled = True
            if schedule:
                QtCore.QMetaObject.invokeMethod(
                    self,
                    "_apply_pending_datasets_update_on_gui_thread",
                    QtCore.Qt.ConnectionType.QueuedConnection,
                )
            return

        self._datasets.clear()
        self._plot_items.clear()
        self._plot_series_items.clear()
        self._plot_legends.clear()
        self._structure_key = None
        self._selected_species_list = []
        self._species_list.blockSignals(True)
        self._species_list.clear()
        self._species_list.blockSignals(False)
        self._schedule_redraw()

    def set_controls_visible(self, visible: bool) -> None:
        """Show or hide the built-in legend/species control row."""
        self._controls_widget.setVisible(bool(visible))

    def set_autorange_locked(self, locked: bool) -> None:
        """
        When locked, disable PyQtGraph autorange on all subplots and preserve the current
        view ranges across redraws (prevents visible "zoom blips" during live updates).
        """
        locked = bool(locked)
        if locked == bool(getattr(self, "_autorange_locked", False)):
            return

        self._autorange_locked = locked
        if locked:
            self._set_scrollbars_locked(True)
            self._snapshot_plot_view_ranges()
            self._snapshot_locked_axis_geometry()
            for idx, plot in enumerate(list(self._plot_items)):
                vb = getattr(plot, "getViewBox", lambda: None)()
                if vb is None:
                    continue
                self._apply_locked_axis_style(plot)
                try:
                    self._locked_autorange_prev[idx] = tuple(bool(x) for x in vb.autoRangeEnabled())
                except Exception:
                    self._locked_autorange_prev[idx] = (True, True)
                try:
                    vb.enableAutoRange(x=False, y=False)
                except Exception as exc:
                    self._record_best_effort_failure(
                        "disable_autorange_lock",
                        message=f"GridPlotView: failed to disable autorange while locking subplot={idx}",
                        exc=exc,
                    )
        else:
            self._set_scrollbars_locked(False)
            for idx, plot in enumerate(list(self._plot_items)):
                vb = getattr(plot, "getViewBox", lambda: None)()
                if vb is None:
                    continue
                try:
                    left_axis = plot.getAxis("left")
                    bottom_axis = plot.getAxis("bottom")
                    left_axis.setStyle(autoExpandTextSpace=True)
                    bottom_axis.setStyle(autoExpandTextSpace=True)
                    left_axis.setWidth(None)
                    bottom_axis.setHeight(None)
                except Exception as exc:
                    self._record_best_effort_failure(
                        "restore_axis_styles",
                        message=f"GridPlotView: failed to restore axis styles while unlocking subplot={idx}",
                        exc=exc,
                    )
                prev = self._locked_autorange_prev.get(idx, (True, True))
                try:
                    vb.enableAutoRange(x=bool(prev[0]), y=bool(prev[1]))
                except Exception as exc:
                    self._record_best_effort_failure(
                        "restore_autorange",
                        message=f"GridPlotView: failed to restore autorange while unlocking subplot={idx}",
                        exc=exc,
                    )
            self._locked_view_ranges.clear()
            self._locked_autorange_prev.clear()
            self._locked_left_axis_width = None
            self._locked_bottom_axis_height = None

    def _snapshot_plot_view_ranges(self) -> None:
        if not self._plot_items:
            return
        for idx, plot in enumerate(list(self._plot_items)):
            vb = getattr(plot, "getViewBox", lambda: None)()
            if vb is None:
                continue
            try:
                view = vb.viewRange()
                x0, x1 = view[0]
                y0, y1 = view[1]
                x_range = (float(x0), float(x1))
                y_range = (float(y0), float(y1))
            except (TypeError, ValueError, IndexError):
                continue
            self._locked_view_ranges[idx] = (x_range, y_range)

    def set_species_selection(self, species_names: Sequence[str]) -> None:
        """
        Update the multi-select list with the provided species names.

        Parameters
        ----------
        species_names : Sequence[str]
            Names to select in the list. Non-existent names are ignored.
        """
        if not species_names or self._species_list.count() == 0:
            return

        names_lower = {name.lower(): name for name in species_names if isinstance(name, str)}
        if not names_lower:
            return

        prev_selected = list(getattr(self, "_selected_species_list", []) or [])
        prev_lower = {str(name).lower() for name in prev_selected if str(name).strip()}
        if prev_lower and prev_lower == set(names_lower.keys()):
            return
        self._species_list.blockSignals(True)
        try:
            for idx in range(self._species_list.count()):
                item = self._species_list.item(idx)
                item.setSelected(False)
                item_lower = item.text().lower()
                if item_lower in names_lower:
                    item.setSelected(True)
            if not self._species_list.selectedItems() and self._species_list.count() > 0:
                self._species_list.item(0).setSelected(True)
        finally:
            self._species_list.blockSignals(False)

        new_selected = [item.text() for item in self._species_list.selectedItems()]
        self._selected_species_list = new_selected
        if new_selected != prev_selected:
            self._schedule_redraw()

    def _calculate_grid_size(self, n_datasets: int) -> Tuple[int, int]:
        """
        Calculate optimal grid dimensions for n datasets.

        Returns (n_rows, n_cols)
        """
        if n_datasets == 0:
            return (1, 1)
        elif n_datasets == 1:
            return (1, 1)
        elif n_datasets == 2:
            return (1, 2)
        elif n_datasets <= 4:
            return (2, 2)
        elif n_datasets <= 6:
            return (2, 3)
        elif n_datasets <= 9:
            return (3, 3)
        else:
            # General case: aim for roughly square grid
            cols = math.ceil(math.sqrt(n_datasets))
            rows = math.ceil(n_datasets / cols)
            return (rows, cols)

    def _update_grid(self):
        """Update grid plots, rebuilding structure only when necessary."""
        if self._pg is None or self._graphics_layout is None:
            return

        if bool(getattr(self, "_autorange_locked", False)):
            self._snapshot_plot_view_ranges()

        desired_key = self._dataset_structure_key(self._datasets)
        needs_rebuild = (
            self._structure_key != desired_key
            or len(self._plot_items) != len(self._datasets)
            or len(self._plot_series_items) != len(self._datasets)
        )
        if needs_rebuild:
            self._rebuild_structure()
        self._update_plot_data_in_place()

    def _plot_dataset(self, plot: Any, dataset: Dict[str, Any], idx: int):
        """Plot a single dataset on the given PlotItem."""
        if self._pg is None:
            return
        # Extract data
        name = dataset['name']
        data_x = dataset['data_x']
        chi_squared = dataset['chi_squared']
        r_squared = dataset['r_squared']
        all_species = dataset.get('all_species', {})
        model_x = dataset.get('model_x')
        model_y = dataset.get('model_y')
        model_series = dataset.get('model_series') or {}
        current_species = dataset.get('current_species')

        color_manager = ColorManager.instance()
        color_manager.seed_species(all_species.keys())

        # Configure plot
        plot.setLabel('bottom', 'Time', units='s')
        plot.setLabel('left', 'Concentration', units='M')
        plot.showGrid(x=True, y=True, alpha=0.3)

        # Build title with fit quality
        title = name
        if chi_squared is not None:
            title += f" (χ² = {chi_squared:.3e})"
            color = self._get_color_for_chi_squared(chi_squared)
        elif r_squared is not None:
            title += f" (R² = {r_squared:.3f})"
            color = self._get_color_for_r_squared(r_squared)
        else:
            color = (51, 51, 51)

        # Set title with color
        plot.setTitle(title, color=color, size='10pt')

        # Plot each selected species
        for species_name in self._selected_species_list:
            if species_name not in all_species:
                continue

            species_data = all_species[species_name]
            color = color_manager.get_species_rgb(species_name, known_species=tuple(all_species.keys()))

            # Plot experimental data as scatter points
            plot.plot(
                data_x, species_data,
                pen=None,
                symbol='o',
                symbolSize=5,
                symbolBrush=self._pg.mkBrush(*color, 150),
                symbolPen=self._pg.mkPen(color=color, width=1),
                name=None,
            )

            # Plot model overlay (multi-series preferred, fallback to single model_y)
            y_model = None
            if isinstance(model_series, dict) and species_name in model_series:
                y_model = model_series[species_name]
            elif model_y is not None and (current_species == species_name or len(self._selected_species_list) == 1):
                y_model = model_y

            if y_model is not None:
                x_model = model_x if model_x is not None else data_x
                plot.plot(
                    x_model,
                    y_model,
                    pen=self._pg.mkPen(color=color, width=2),
                    name=species_name,
                )

        # Add legend if we have multiple species
        if len(self._selected_species_list) > 1 and self._legend_visible:
            plot.addLegend(offset=(5, 5))

        # Reduce font sizes for compact display
        font = QtGui.QFont()
        font.setPointSize(7)
        for axis in ['left', 'bottom']:
            plot.getAxis(axis).setStyle(tickFont=font)
            plot.getAxis(axis).setTickFont(font)

    def _toggle_legend(self, visible: bool) -> None:
        """Toggle legend visibility and redraw grid."""
        self._legend_visible = visible
        self._schedule_redraw()

    def _on_species_selection_changed(self) -> None:
        """Handle species multi-selection change."""
        selected_items = self._species_list.selectedItems()
        self._selected_species_list = [item.text() for item in selected_items]

        # Redraw grid with new species selection
        self._schedule_redraw()

    def _update_species_selector(self) -> None:
        """Update species selector with all available species across datasets."""
        all_species_names = set()

        for dataset in self._datasets:
            all_species = dataset.get('all_species', {})
            all_species_names.update(all_species.keys())

        if not all_species_names:
            return

        sorted_species = tuple(sorted(all_species_names))
        if sorted_species == tuple(getattr(self, "_available_species_cache", None) or ()):
            return
        self._available_species_cache = sorted_species

        # Update list widget (block signals to prevent recursion)
        self._species_list.blockSignals(True)
        current_selection = self._selected_species_list.copy()
        self._species_list.clear()

        for species in list(sorted_species):
            item = QtWidgets.QListWidgetItem(species)
            self._species_list.addItem(item)
            # Restore selection or select first species by default
            if species in current_selection or (not current_selection and species == sorted_species[0]):
                item.setSelected(True)

        # Update selected list
        self._selected_species_list = [item.text() for item in self._species_list.selectedItems()]
        self._species_list.blockSignals(False)

    def _get_color_for_chi_squared(self, chi_squared: float) -> Tuple[int, int, int]:
        """Get color based on chi-squared value."""
        if chi_squared < 0.1:
            return (34, 170, 85)  # Green
        elif chi_squared < 1.0:
            return (221, 170, 85)  # Yellow/orange
        else:
            return (221, 68, 68)  # Red

    def _get_color_for_r_squared(self, r_squared: float) -> Tuple[int, int, int]:
        """Get color based on R-squared value."""
        if r_squared > 0.95:
            return (34, 170, 85)  # Green
        elif r_squared > 0.85:
            return (221, 170, 85)  # Yellow/orange
        else:
            return (221, 68, 68)  # Red

    def set_dark_mode(self, enabled: bool):
        """
        Set dark mode for the grid.

        Parameters
        ----------
        enabled : bool
            If True, use dark theme. Otherwise, use light theme.
        """
        if self._pg is None or self._graphics_layout is None:
            return

        if enabled:
            bg_color = (30, 30, 30)
            text_color = (224, 224, 224)
        else:
            bg_color = (248, 248, 248)
            text_color = (0, 0, 0)

        # Set background on the GraphicsLayoutWidget (not individual PlotItems)
        self._graphics_layout.setBackground(bg_color)

        # Update axis colors for all plots
        for plot in self._plot_items:
            for axis in ['left', 'bottom']:
                plot.getAxis(axis).setTextPen(text_color)
                plot.getAxis(axis).setPen(text_color)
