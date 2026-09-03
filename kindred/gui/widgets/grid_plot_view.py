"""Lean Fit-local grid plot with injected species-color resolution."""

from __future__ import annotations

import io
import math
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import TypeAlias

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.display_name_policy import (
    PLOT_TITLE_LABEL_MAX_CHARS,
    STATUS_ITEM_LABEL_MAX_CHARS,
    compact_plot_title_label,
    compact_text,
)
from kindred.gui.fitting.worker import FitCurveDataset, FitCurveYSemantic
from kindred.gui.plot_config import require_pyqtgraph
from kindred.gui.workspace_result_presenter import (
    encode_clipboard_text,
    encode_tsv_row,
)

__all__ = ["DatasetPlotPresentation", "GridPlotView"]

pg = require_pyqtgraph(context="Kindred GUI")

ColorResolver = Callable[[str], object]
DatasetPlotPresentation: TypeAlias = tuple[str, str]
_FIT_COPY_COLUMNS = 15
_MAX_FIT_COPY_CELLS = 1_000_000
_MAX_FIT_COPY_CHARACTERS = 64 * 1024 * 1024
_FIT_COPY_ROWS_PER_CHUNK = 2048
_LOCKED_LEFT_AXIS_WIDTH = 92
_LOCKED_BOTTOM_AXIS_HEIGHT = 48


def _color_value(value: object) -> object:
    if isinstance(value, QtGui.QColor):
        return value
    if isinstance(value, (str, tuple, list)):
        return value
    return str(value)


def _fit_y_axis_label(
    y_semantics: tuple[tuple[str, FitCurveYSemantic], ...],
    visible_y_names: Iterable[str],
) -> tuple[str, str | None]:
    visible = tuple(dict.fromkeys(str(name) for name in visible_y_names))
    semantics = dict(y_semantics)
    if visible and all(
        semantics.get(name) is FitCurveYSemantic.CONCENTRATION
        for name in visible
    ):
        return "Concentration", "M"
    return "Value", None


class GridPlotView(QtWidgets.QWidget):
    """Render immutable live and terminal Fit curves in the Fit-local plot."""

    copyFitDataRequested = QtCore.Signal()

    def __init__(
        self,
        *,
        color_resolver: ColorResolver | None = None,
        show_generic_controls: bool = True,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if color_resolver is not None and not callable(color_resolver):
            raise TypeError("color_resolver must be callable.")
        self._color_resolver = color_resolver
        self._color_by_species: Mapping[str, object] = MappingProxyType({})
        self._curves: tuple[FitCurveDataset, ...] = ()
        self._dataset_presentation: Mapping[str, DatasetPlotPresentation] = (
            MappingProxyType({})
        )
        self._pg = pg
        self._legend_visible = True
        self._selected_species_list: list[str] = []
        self._available_species_cache: tuple[str, ...] = ()
        self._seen_nonempty_payload = False
        self._plot_items: list[object] = []
        self._plot_series_items: list[dict[str, object]] = []
        self._plot_legends: list[object | None] = []
        self._plot_display_state: dict[str, tuple[object, ...]] = {}
        self._structure_key: tuple[tuple[str, ...], int, int] | None = None
        self._dark_mode = False
        self._live_layout_locked = False
        self._live_layout_lock_pending = False
        self._locked_left_axis_width: int | None = None
        self._locked_bottom_axis_height: int | None = None
        self._scrollbar_policies_prev: (
            tuple[QtCore.Qt.ScrollBarPolicy, QtCore.Qt.ScrollBarPolicy] | None
        ) = None
        self._copy_chunks = None
        self._copy_output: io.StringIO | None = None
        self._copy_timer = QtCore.QTimer(self)
        self._copy_timer.setInterval(0)
        self._copy_timer.timeout.connect(self._continue_copy)

        self._context_menu = QtWidgets.QMenu(self)
        self._context_actions = {
            key: self._context_menu.addAction(label)
            for key, label in (
                ("copy", "Copy Fit Data"),
                ("range", "Auto Range"),
                ("pan", "Pan"),
                ("rect", "Rectangle Zoom"),
                ("legend", "Show Legend"),
                ("log_x", "Log X"),
                ("log_y", "Log Y"),
                ("invert_x", "Invert X"),
                ("invert_y", "Invert Y"),
            )
        }
        for action in tuple(self._context_actions.values())[2:]:
            action.setCheckable(True)
        self._mouse_mode_group = QtGui.QActionGroup(self)
        self._mouse_mode_group.setExclusive(True)
        self._mouse_mode_group.addAction(self._context_actions["pan"])
        self._mouse_mode_group.addAction(self._context_actions["rect"])
        self._context_plot_index: int | None = None
        for action in self._context_actions.values():
            action.triggered.connect(self._context_action_triggered)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._controls_widget = QtWidgets.QWidget(self)
        self._controls_widget.setObjectName("grid_plot_view_controls")
        controls_layout = QtWidgets.QHBoxLayout(self._controls_widget)
        controls_layout.setContentsMargins(2, 2, 2, 0)
        controls_layout.setSpacing(6)

        self._legend_toggle_btn = QtWidgets.QCheckBox("Show Legend", self)
        self._legend_toggle_btn.setChecked(True)
        self._legend_toggle_btn.toggled.connect(self._toggle_legend)
        controls_layout.addWidget(self._legend_toggle_btn)

        species_label = QtWidgets.QLabel("Species", self._controls_widget)
        controls_layout.addWidget(species_label)
        self._species_list = QtWidgets.QListWidget(self._controls_widget)
        self._species_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        self._species_list.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self._species_list.itemSelectionChanged.connect(
            self._on_species_selection_changed
        )
        species_label.setBuddy(self._species_list)
        controls_layout.addWidget(self._species_list)
        controls_layout.addStretch()
        layout.addWidget(self._controls_widget)
        self._controls_widget.setVisible(bool(show_generic_controls))

        self._canvas = pg.GraphicsLayoutWidget(self)
        self._canvas.scene().installEventFilter(self)
        try:
            self._canvas.useOpenGL(False)
        except (AttributeError, RuntimeError):
            pass
        layout.addWidget(self._canvas)

    @staticmethod
    def _normalize_dataset_presentation(
        presentation: Mapping[str, DatasetPlotPresentation],
    ) -> Mapping[str, DatasetPlotPresentation]:
        if not isinstance(presentation, Mapping):
            raise TypeError("presentation must be a mapping.")
        normalized: dict[str, DatasetPlotPresentation] = {}
        for dataset_id, value in presentation.items():
            if not isinstance(dataset_id, str) or not dataset_id:
                raise TypeError("presentation keys must be non-empty strings.")
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or not all(isinstance(item, str) for item in value)
            ):
                raise TypeError(
                    "presentation values must be (title, tooltip) string tuples."
                )
            normalized[dataset_id] = value
        return MappingProxyType(normalized)

    def set_projection(
        self,
        curves: Iterable[FitCurveDataset],
        presentation: Mapping[str, DatasetPlotPresentation],
    ) -> bool:
        """Install curves and presentation together, then redraw exactly once."""

        self._cancel_copy()
        if len(self._plot_items) == len(self._curves):
            for curve, plot in zip(self._curves, self._plot_items, strict=True):
                self._plot_display_state[curve.dataset_id] = self._plot_display_state_for_plot(plot)
        values = tuple(curves)
        if any(not isinstance(item, FitCurveDataset) for item in values):
            raise TypeError("curves must contain FitCurveDataset values.")
        normalized_presentation = self._normalize_dataset_presentation(
            presentation
        )
        species_names = tuple(
            dict.fromkeys(
                species
                for curve in values
                for species, _series in (*curve.model_y, *curve.observed_y)
            )
        )
        if species_names and self._color_resolver is None:
            raise ValueError("Fit curves require an injected species-color resolver.")
        self._color_by_species = MappingProxyType(
            {
                species: _color_value(self._color_resolver(species))
                for species in species_names
            }
            if self._color_resolver is not None
            else {}
        )
        self._dataset_presentation = normalized_presentation
        self._curves = values
        if not values:
            self._seen_nonempty_payload = False
            self._selected_species_list = []
            self._available_species_cache = ()
        return self._redraw()

    def _plot_display_state_for_plot(self, plot: object) -> tuple[object, ...]:
        state = plot.getViewBox().getState(copy=True)
        return (state["mouseMode"], tuple(state["logMode"]), state["xInverted"],
                state["yInverted"], tuple(state["autoRange"]),
                tuple(tuple(axis) for axis in state["viewRange"]))

    def _restore_plot_display_state(
        self,
        plot: object,
        dataset_id: str,
    ) -> None:
        """Restore one dataset's display state after a PlotItem rebuild."""

        state = self._plot_display_state.get(dataset_id)
        if state is None:
            return
        view_box = plot.getViewBox()
        mouse_mode, log_mode, x_inverted, y_inverted, auto_range, view_range = state
        view_box.setMouseMode(mouse_mode)
        plot.setLogMode(x=log_mode[0], y=log_mode[1])
        plot.invertX(bool(x_inverted))
        plot.invertY(bool(y_inverted))
        view_box.enableAutoRange(x=auto_range[0], y=auto_range[1])
        view_box.setRange(
            xRange=view_range[0], yRange=view_range[1], padding=0,
            disableAutoRange=False)

    def _prepare_context_menu(self, plot_index: int) -> None:
        plot = self._plot_items[plot_index]
        view_box = plot.getViewBox()
        state = view_box.getState(copy=False)
        mouse_mode = state["mouseMode"]
        self._context_plot_index = plot_index
        actions = self._context_actions
        actions["copy"].setEnabled(bool(self._curves))
        log_mode = state["logMode"]
        for key, value in (
            ("pan", mouse_mode == pg.ViewBox.PanMode),
            ("rect", mouse_mode == pg.ViewBox.RectMode),
            ("legend", self._legend_visible),
            ("log_x", log_mode[0]),
            ("log_y", log_mode[1]),
            ("invert_x", state["xInverted"]),
            ("invert_y", state["yInverted"]),
        ):
            actions[key].setChecked(bool(value))

    def _show_context_menu(
        self,
        scene_position: QtCore.QPointF,
        global_position: QtCore.QPoint,
    ) -> None:
        plot_index = next((index for index, plot in enumerate(self._plot_items)
                           if plot.sceneBoundingRect().contains(scene_position)), None)
        if plot_index is None:
            self._context_plot_index = None
            return
        self._prepare_context_menu(plot_index)
        self._context_menu.exec(global_position)
        self._context_plot_index = None

    def eventFilter(self, watched: object, event: QtCore.QEvent) -> bool:
        if watched is self._canvas.scene() and event.type() == QtCore.QEvent.Type.GraphicsSceneContextMenu:
            self._show_context_menu(event.scenePos(), event.screenPos())
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _context_action_triggered(self, checked: bool) -> None:
        kind = next(key for key, action in self._context_actions.items() if action is self.sender())
        if kind == "copy":
            self.copyFitDataRequested.emit()
            return
        index = self._context_plot_index
        if index is None:
            return
        plot = self._plot_items[index]
        view_box = plot.getViewBox()
        if kind == "range":
            view_box.enableAutoRange(x=True, y=True)
            view_box.autoRange()
        elif kind in {"pan", "rect"} and checked:
            view_box.setMouseMode(pg.ViewBox.PanMode if kind == "pan" else pg.ViewBox.RectMode)
        elif kind == "legend":
            self._toggle_legend(checked)
            return
        elif kind in {"log_x", "log_y"}:
            before = self._plot_display_state_for_plot(plot)
            plot.setLogMode(**{kind[-1]: bool(checked)})
            auto_range = before[4]
            view_box.enableAutoRange(x=auto_range[0], y=auto_range[1])
            if not all(bool(item) for item in auto_range):
                view_range = before[5]
                view_box.setRange(
                    xRange=view_range[0], yRange=view_range[1], padding=0,
                    disableAutoRange=False)
        elif kind in {"invert_x", "invert_y"}:
            getattr(plot, f"invert{kind[-1].upper()}")(bool(checked))
        else:
            return
        self._plot_display_state[self._curves[index].dataset_id] = self._plot_display_state_for_plot(plot)

    def copy_eligible(self) -> bool:
        """Whether this Fit-local panel has immutable curves to copy."""

        return bool(self._curves)

    def copy_projection_to_clipboard(
        self,
        curves: Iterable[FitCurveDataset],
    ) -> None:
        """Copy a fresh canonical projection without using retained plot data."""

        values = tuple(curves)
        if any(not isinstance(item, FitCurveDataset) for item in values):
            raise TypeError("curves must contain FitCurveDataset values.")
        if not values:
            raise ValueError("No Fit curves are available to copy.")
        if self._copy_timer.isActive():
            return
        self._copy_chunks = iter(self._iter_copy_chunks(values))
        self._copy_output = io.StringIO()
        self._copy_timer.start()

    def _cancel_copy(self) -> None:
        """Cancel one pending asynchronous full-resolution Copy operation."""

        self._copy_timer.stop()
        self._copy_chunks = None
        self._copy_output = None

    @staticmethod
    def _iter_copy_chunks(
        curves: tuple[FitCurveDataset, ...],
    ) -> Iterable[str]:
        row_count = sum(
            len(values)
            for dataset in curves
            for _species, values in dataset.objective_model_y
        )
        cell_count = (row_count + 1) * _FIT_COPY_COLUMNS
        if cell_count > _MAX_FIT_COPY_CELLS:
            raise ValueError(
                "Fit Copy payload exceeds the bounded full-resolution cell "
                f"limit ({_MAX_FIT_COPY_CELLS:,})."
            )
        header = encode_tsv_row(
            (
                "dataset_id",
                "dataset_label",
                "series",
                "raw_observation_id",
                "raw_column",
                "source_row",
                "segment_id",
                "segment_key",
                "discontinuity_id",
                "selection_provenance",
                "x_name",
                "observed_x",
                "observed_y",
                "model_x",
                "model_y",
            )
        )
        total_characters = 0
        output = io.StringIO()
        output.write(header)
        rows_in_chunk = 0
        for dataset in curves:
            model_x = dict(dataset.objective_model_x)
            model_y = dict(dataset.objective_model_y)
            observed_x = dict(dataset.observed_x)
            observed_y = dict(dataset.observed_y)
            segment_ids = dict(dataset.objective_model_segment_ids)
            segment_keys = dict(dataset.objective_model_segment_keys)
            discontinuity_ids = dict(dataset.objective_model_discontinuity_ids)
            selection = dict(dataset.objective_selection_provenance)
            observation_ids = dict(dataset.objective_observation_ids)
            raw_columns = dict(dataset.objective_raw_columns)
            source_rows = dict(dataset.objective_source_rows)
            for species, model_y_values in model_y.items():
                rows = zip(
                    observed_x[species],
                    observed_y[species],
                    model_x[species],
                    model_y_values,
                    strict=True,
                )
                for index, (obs_x, obs_y, selected_x, selected_y) in enumerate(rows):
                    output.write(
                        encode_tsv_row(
                            (
                                encode_clipboard_text(dataset.dataset_id),
                                encode_clipboard_text(dataset.display_name),
                                encode_clipboard_text(species),
                                encode_clipboard_text(
                                    observation_ids[species][index]
                                ),
                                encode_clipboard_text(
                                    raw_columns[species][index]
                                ),
                                str(source_rows[species][index]),
                                str(segment_ids[species][index]),
                                encode_clipboard_text(
                                    segment_keys[species][index]
                                ),
                                str(discontinuity_ids[species][index]),
                                encode_clipboard_text(
                                    selection[species][index]
                                ),
                                encode_clipboard_text(dataset.x_name),
                                f"{float(obs_x):.17g}",
                                f"{float(obs_y):.17g}",
                                f"{float(selected_x):.17g}",
                                f"{float(selected_y):.17g}",
                            )
                        )
                    )
                    rows_in_chunk += 1
                    if rows_in_chunk >= _FIT_COPY_ROWS_PER_CHUNK:
                        chunk = output.getvalue()
                        total_characters += len(chunk)
                        if total_characters > _MAX_FIT_COPY_CHARACTERS:
                            raise ValueError(
                                "Fit Copy payload exceeds the bounded "
                                "output-size limit "
                                f"({_MAX_FIT_COPY_CHARACTERS // (1024 * 1024)} MiB)."
                            )
                        yield chunk
                        output = io.StringIO()
                        rows_in_chunk = 0
        chunk = output.getvalue()
        total_characters += len(chunk)
        if total_characters > _MAX_FIT_COPY_CHARACTERS:
            raise ValueError(
                "Fit Copy payload exceeds the bounded output-size limit "
                f"({_MAX_FIT_COPY_CHARACTERS // (1024 * 1024)} MiB)."
            )
        if chunk:
            yield chunk

    @QtCore.Slot()
    def _continue_copy(self) -> None:
        chunks = self._copy_chunks
        output = self._copy_output
        if chunks is None or output is None:
            self._cancel_copy()
            return
        try:
            output.write(next(chunks))
        except StopIteration:
            text = output.getvalue()
            self._cancel_copy()
            application = QtWidgets.QApplication.instance()
            if application is None:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Copy Fit Data",
                    "A QApplication is required to copy Fit data.",
                )
                return
            application.clipboard().setText(text)
        except Exception as exc:
            self._cancel_copy()
            QtWidgets.QMessageBox.critical(
                self,
                "Copy Fit Data",
                str(exc),
            )

    @staticmethod
    def _dataset_structure_key(
        curves: tuple[FitCurveDataset, ...],
    ) -> tuple[tuple[str, ...], int, int]:
        identifiers = tuple(
            f"{curve.dataset_id}\x00{curve.display_name}\x00{curve.x_name}"
            for curve in curves
        )
        rows, columns = GridPlotView._calculate_grid_size(len(identifiers))
        return identifiers, rows, columns

    @staticmethod
    def _calculate_grid_size(n_datasets: int) -> tuple[int, int]:
        """Use the historical 1x2/2x2/2x3/3x3 grid breakpoints."""

        if n_datasets <= 1:
            return 1, 1
        if n_datasets == 2:
            return 1, 2
        if n_datasets <= 4:
            return 2, 2
        if n_datasets <= 6:
            return 2, 3
        if n_datasets <= 9:
            return 3, 3
        columns = math.ceil(math.sqrt(n_datasets))
        return math.ceil(n_datasets / columns), columns

    def _configure_plot_static(self, plot: object, x_name: str) -> None:
        """Apply the historical static axes, grid, and tick geometry."""

        plot.setMenuEnabled(False)
        x_label = "Time" if x_name.casefold() == "t" else x_name
        if x_name.casefold() == "t":
            plot.setLabel("bottom", x_label, units="s")
        else:
            plot.setLabel("bottom", x_label)
        plot.setLabel("left", "Value")
        plot.showGrid(x=True, y=True, alpha=0.3)
        font = QtGui.QFont()
        font.setPointSize(7)
        for axis_name in ("left", "bottom"):
            try:
                axis = plot.getAxis(axis_name)
                axis.setStyle(tickFont=font)
                axis.setTickFont(font)
            except Exception:
                pass
        self._apply_dark_mode_to_plot(plot)
        if self._live_layout_locked:
            self._apply_live_axis_style(plot)

    def _apply_dark_mode_to_plot(self, plot: object) -> None:
        text_color = (224, 224, 224) if self._dark_mode else (0, 0, 0)
        for axis_name in ("left", "bottom"):
            try:
                axis = plot.getAxis(axis_name)
                axis.setTextPen(text_color)
                axis.setPen(text_color)
            except Exception:
                pass

    def _ensure_legend_state(
        self,
        plot: object,
        index: int,
        active_keys: set[str],
    ) -> None:
        want_legend = bool(self._legend_visible and len(active_keys) > 1)
        legend = self._plot_legends[index]
        if want_legend and legend is None:
            try:
                legend = plot.addLegend(offset=(5, 5))
            except Exception:
                legend = None
            self._plot_legends[index] = legend
        if legend is not None:
            legend.clear()
            if want_legend:
                for key in sorted(active_keys):
                    item = self._plot_series_items[index][key]
                    species_name, kind = key.rsplit("::", 1)
                    compact_species = compact_text(
                        species_name,
                        max_chars=STATUS_ITEM_LABEL_MAX_CHARS,
                    )
                    legend.addItem(
                        item,
                        f"{compact_species.display} — {kind}",
                    )
            legend.setVisible(want_legend)

    def _ensure_curve_item(
        self,
        plot: object,
        index: int,
        key: str,
        *,
        is_model: bool,
        color: object,
    ) -> object:
        series_map = self._plot_series_items[index]
        item = series_map.get(key)
        if item is not None:
            return item
        if is_model:
            item = plot.plot(
                [],
                [],
                pen=self._pg.mkPen(color=color, width=2),
                name=None,
            )
        else:
            item = plot.plot(
                [],
                [],
                pen=None,
                symbol="o",
                symbolSize=5,
                symbolBrush=self._pg.mkBrush(color),
                symbolPen=self._pg.mkPen(color=color, width=1),
                name=None,
            )
        series_map[key] = item
        return item

    @staticmethod
    def _with_segment_gaps(
        x_values: tuple[float, ...],
        y_values: tuple[float, ...],
        segment_ids: tuple[int, ...],
        segment_keys: tuple[str, ...],
        discontinuity_ids: tuple[int, ...],
    ) -> tuple[list[float], list[float]]:
        if not (
            len(x_values)
            == len(y_values)
            == len(segment_ids)
            == len(segment_keys)
            == len(discontinuity_ids)
        ):
            raise ValueError("Fit model curve identity lengths do not align.")
        x_render: list[float] = []
        y_render: list[float] = []
        for index, (x_value, y_value) in enumerate(zip(x_values, y_values)):
            if index and (
                segment_ids[index] != segment_ids[index - 1]
                or segment_keys[index] != segment_keys[index - 1]
                or discontinuity_ids[index] != discontinuity_ids[index - 1]
            ):
                x_render.append(math.nan)
                y_render.append(math.nan)
            x_render.append(x_value)
            y_render.append(y_value)
        return x_render, y_render

    def _update_plot_data_in_place(self) -> None:
        if len(self._plot_items) != len(self._curves):
            return
        model_x_by_dataset = [dict(curve.model_x) for curve in self._curves]
        model_y_by_dataset = [dict(curve.model_y) for curve in self._curves]
        observed_x_by_dataset = [dict(curve.observed_x) for curve in self._curves]
        observed_y_by_dataset = [dict(curve.observed_y) for curve in self._curves]
        model_segments_by_dataset = [
            dict(curve.model_segment_ids) for curve in self._curves
        ]
        model_segment_keys_by_dataset = [
            dict(curve.model_segment_keys) for curve in self._curves
        ]
        model_discontinuities_by_dataset = [
            dict(curve.model_discontinuity_ids) for curve in self._curves
        ]

        for dataset_index, curve in enumerate(self._curves):
            plot = self._plot_items[dataset_index]
            presentation = self._dataset_presentation.get(curve.dataset_id)
            if presentation is None:
                compact_name = compact_plot_title_label(
                    curve.display_name,
                    max_chars=PLOT_TITLE_LABEL_MAX_CHARS,
                )
                title = compact_name.display
                tooltip = f"{compact_name.full}\nDataset ID: {curve.dataset_id}"
            else:
                title, tooltip = presentation
            title_state = (title, tooltip, self._dark_mode)
            if getattr(plot, "_kindred_title_state", None) != title_state:
                title_color = (224, 224, 224) if self._dark_mode else (0, 0, 0)
                plot.setTitle(title, color=title_color, size="10pt")
                setattr(plot, "_kindred_title_state", title_state)
            if hasattr(plot, "setToolTip"):
                plot.setToolTip(tooltip)

            active_keys: set[str] = set()
            model_x = model_x_by_dataset[dataset_index]
            model_y = model_y_by_dataset[dataset_index]
            observed_x = observed_x_by_dataset[dataset_index]
            observed_y = observed_y_by_dataset[dataset_index]
            model_segments = model_segments_by_dataset[dataset_index]
            model_segment_keys = model_segment_keys_by_dataset[dataset_index]
            model_discontinuities = model_discontinuities_by_dataset[dataset_index]
            visible_y_names = tuple(
                name
                for name in self._selected_species_list
                if name in model_y or name in observed_y
            )
            y_label, y_units = _fit_y_axis_label(
                curve.y_semantics,
                visible_y_names,
            )
            plot.setLabel("left", y_label, units=y_units)

            for species in self._selected_species_list:
                color = self._color_by_species[species]
                if species in observed_y:
                    observed_item = self._ensure_curve_item(
                        plot,
                        dataset_index,
                        f"{species}::observed",
                        is_model=False,
                        color=color,
                    )
                    observed_item.setSymbol("o")
                    observed_item.setSymbolSize(5)
                    observed_item.setSymbolBrush(self._pg.mkBrush(color))
                    observed_item.setSymbolPen(self._pg.mkPen(color=color, width=1))
                    observed_item.setData(
                        list(observed_x[species]),
                        list(observed_y[species]),
                    )
                    observed_item.setVisible(True)
                    active_keys.add(f"{species}::observed")

                if species in model_y:
                    model_item = self._ensure_curve_item(
                        plot,
                        dataset_index,
                        f"{species}::model",
                        is_model=True,
                        color=color,
                    )
                    model_item.setPen(self._pg.mkPen(color=color, width=2))
                    model_render_x, model_render_y = self._with_segment_gaps(
                        model_x[species],
                        model_y[species],
                        model_segments[species],
                        model_segment_keys[species],
                        model_discontinuities[species],
                    )
                    model_item.setData(model_render_x, model_render_y)
                    model_item.setVisible(True)
                    active_keys.add(f"{species}::model")

            for key, item in self._plot_series_items[dataset_index].items():
                if key in active_keys:
                    continue
                item.setVisible(False)
            self._ensure_legend_state(plot, dataset_index, active_keys)

    def _rebuild_structure(self) -> None:
        self._canvas.setUpdatesEnabled(False)
        try:
            self._canvas.clear()
            self._plot_items.clear()
            self._plot_series_items.clear()
            self._plot_legends.clear()
            if not self._curves:
                text = self._pg.TextItem(
                    "No Fit curves yet",
                    color=(150, 150, 150),
                    anchor=(0.5, 0.5),
                )
                placeholder = self._canvas.addPlot(row=0, col=0)
                placeholder.setMenuEnabled(False)
                placeholder.hideAxis("left")
                placeholder.hideAxis("bottom")
                placeholder.addItem(text)
                text.setPos(0.5, 0.5)
                self._structure_key = self._dataset_structure_key(())
                return

            _rows, columns = self._calculate_grid_size(len(self._curves))
            for index, curve in enumerate(self._curves):
                row, column = divmod(index, columns)
                plot = self._canvas.addPlot(row=row, col=column)
                self._plot_items.append(plot)
                self._plot_series_items.append({})
                self._plot_legends.append(None)
                self._configure_plot_static(plot, curve.x_name)
                self._restore_plot_display_state(plot, curve.dataset_id)
                if self._live_layout_locked:
                    self._apply_live_axis_style(plot)
            self._structure_key = self._dataset_structure_key(self._curves)
        finally:
            self._canvas.setUpdatesEnabled(True)

    def _set_scrollbars_locked(self, locked: bool) -> None:
        if locked:
            if self._scrollbar_policies_prev is None:
                try:
                    self._scrollbar_policies_prev = (
                        self._canvas.horizontalScrollBarPolicy(),
                        self._canvas.verticalScrollBarPolicy(),
                    )
                except Exception:
                    self._scrollbar_policies_prev = None
            try:
                self._canvas.setHorizontalScrollBarPolicy(
                    QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
                )
                self._canvas.setVerticalScrollBarPolicy(
                    QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
                )
            except Exception:
                pass
            return
        if self._scrollbar_policies_prev is not None:
            try:
                self._canvas.setHorizontalScrollBarPolicy(
                    self._scrollbar_policies_prev[0]
                )
                self._canvas.setVerticalScrollBarPolicy(
                    self._scrollbar_policies_prev[1]
                )
            except Exception:
                pass
        self._scrollbar_policies_prev = None

    def _snapshot_live_axis_geometry(self) -> None:
        left_width = 0.0
        bottom_height = 0.0
        for plot in self._plot_items:
            try:
                left_width = max(
                    left_width,
                    float(plot.getAxis("left").boundingRect().width()),
                )
                bottom_height = max(
                    bottom_height,
                    float(plot.getAxis("bottom").boundingRect().height()),
                )
            except Exception:
                continue
        self._locked_left_axis_width = int(
            math.ceil(
                min(
                    140.0,
                    max(
                        float(_LOCKED_LEFT_AXIS_WIDTH),
                        left_width + 12.0,
                    ),
                )
            )
        )
        self._locked_bottom_axis_height = int(
            math.ceil(
                min(
                    95.0,
                    max(
                        float(_LOCKED_BOTTOM_AXIS_HEIGHT),
                        bottom_height + 8.0,
                    ),
                )
            )
        )

    def _apply_live_axis_style(self, plot: object) -> None:
        if not self._live_layout_locked:
            return
        try:
            left_axis = plot.getAxis("left")
            bottom_axis = plot.getAxis("bottom")
            left_axis.setStyle(autoExpandTextSpace=False)
            bottom_axis.setStyle(autoExpandTextSpace=False)
            left_axis.setWidth(
                float(self._locked_left_axis_width or _LOCKED_LEFT_AXIS_WIDTH)
            )
            bottom_axis.setHeight(
                float(self._locked_bottom_axis_height or _LOCKED_BOTTOM_AXIS_HEIGHT)
            )
        except Exception:
            pass

    def _update_species_selector(self) -> None:
        available = tuple(
            sorted(
                {
                    species
                    for curve in self._curves
                    for species, _values in (*curve.model_y, *curve.observed_y)
                    if species
                }
            )
        )
        if available == self._available_species_cache:
            return
        self._available_species_cache = available
        first_payload = bool(available) and not self._seen_nonempty_payload
        previous_selection = set(self._selected_species_list)
        self._species_list.blockSignals(True)
        try:
            self._species_list.clear()
            for species in available:
                compact_species = compact_text(
                    species,
                    max_chars=STATUS_ITEM_LABEL_MAX_CHARS,
                )
                item = QtWidgets.QListWidgetItem(compact_species.display)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, species)
                item.setToolTip(compact_species.full)
                self._species_list.addItem(item)
                if first_payload or species in previous_selection:
                    item.setSelected(True)
            self._selected_species_list = [
                self._species_item_name(item)
                for item in self._species_list.selectedItems()
            ]
        finally:
            self._species_list.blockSignals(False)
        if available:
            self._seen_nonempty_payload = True

    @staticmethod
    def _species_item_name(item: QtWidgets.QListWidgetItem) -> str:
        return str(item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text())

    def _on_species_selection_changed(self) -> None:
        self._selected_species_list = [
            self._species_item_name(item) for item in self._species_list.selectedItems()
        ]
        self._redraw()

    def _toggle_legend(self, visible: bool) -> None:
        self._legend_visible = bool(visible)
        if self._legend_toggle_btn.isChecked() != self._legend_visible:
            self._legend_toggle_btn.blockSignals(True)
            try:
                self._legend_toggle_btn.setChecked(self._legend_visible)
            finally:
                self._legend_toggle_btn.blockSignals(False)
        self._redraw()

    def set_dark_mode(self, enabled: bool) -> None:
        """Apply the historical Fit-grid dark/light background and axis pens."""

        self._dark_mode = bool(enabled)
        try:
            self._canvas.setBackground(
                (30, 30, 30) if self._dark_mode else (248, 248, 248)
            )
        except Exception:
            pass
        for plot in self._plot_items:
            self._apply_dark_mode_to_plot(plot)
        self._redraw()

    def set_live_layout_locked(self, locked: bool) -> None:
        """Stabilize live grid geometry without freezing numerical ranges."""

        requested = bool(locked)
        if requested and (
            self._live_layout_locked or self._live_layout_lock_pending
        ):
            return
        if requested:
            if not self._curves or not self._plot_items:
                self._live_layout_lock_pending = True
                return
            self._activate_live_layout_lock()
            return

        self._live_layout_lock_pending = False
        if not self._live_layout_locked:
            return
        self._live_layout_locked = False
        self._set_scrollbars_locked(False)
        for plot in self._plot_items:
            try:
                left_axis = plot.getAxis("left")
                bottom_axis = plot.getAxis("bottom")
                left_axis.setStyle(autoExpandTextSpace=True)
                bottom_axis.setStyle(autoExpandTextSpace=True)
                left_axis.setWidth(None)
                bottom_axis.setHeight(None)
            except Exception:
                pass
        self._locked_left_axis_width = None
        self._locked_bottom_axis_height = None
        self._redraw()

    def _activate_live_layout_lock(self) -> None:
        """Freeze established axis geometry while live values continue changing."""

        self._live_layout_lock_pending = False
        self._live_layout_locked = True
        self._set_scrollbars_locked(True)
        self._snapshot_live_axis_geometry()
        for plot in self._plot_items:
            self._apply_live_axis_style(plot)

    def _establish_live_layout_lock(self) -> None:
        """Autorange the first live payload, then stabilize only its layout."""

        if (
            not self._live_layout_lock_pending
            or not self._curves
            or len(self._plot_items) != len(self._curves)
        ):
            return
        for plot in self._plot_items:
            view_box = plot.getViewBox()
            try:
                view_box.enableAutoRange(x=True, y=True)
                view_box.autoRange()
                view_box.enableAutoRange(x=True, y=True)
            except Exception:
                pass
        self._activate_live_layout_lock()

    def _update_grid(self) -> None:
        desired_key = self._dataset_structure_key(self._curves)
        if (
            self._structure_key != desired_key
            or len(self._plot_items) != len(self._curves)
            or len(self._plot_series_items) != len(self._curves)
        ):
            self._rebuild_structure()
        self._update_plot_data_in_place()
        self._establish_live_layout_lock()

    def _redraw(self) -> bool:
        self._update_species_selector()
        self._update_grid()
        return bool(
            self._curves
            and len(self._plot_items) == len(self._curves)
            and len(self._plot_series_items) == len(self._curves)
        )
