"""Lean Fit-local grid plot with injected species-color resolution."""

from __future__ import annotations

import io
import html
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import TypeAlias

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.curve_copy_headers import copy_column_header
from kindred.core.quantity_units import Unit, MOLAR, SECOND, format_unit_for_display
from kindred.gui.fitting.curve_projection import (
    focused_units, prepared_fit_units, scale_values, focused_model_chunks, focused_model_rows,
)
from kindred.gui.display_name_policy import (
    PLOT_TITLE_LABEL_MAX_CHARS,
    compact_plot_title_label,
)
from kindred.gui.fitting.launch import FitExecutionRequest
from kindred.gui.fitting.execution import (
    FitCurveDataset, _observed_curve_parts,
)
from kindred.gui.theme_manager import plot_background_color, plot_foreground_color
from kindred.gui.plot_config import require_pyqtgraph
from kindred.gui.workspace_result_presenter import (
    encode_clipboard_text,
    encode_tsv_row,
)

__all__ = ["DatasetPlotPresentation", "GridPlotView"]

pg = require_pyqtgraph(context="Kindred GUI")

ColorResolver = Callable[[str], object]
DatasetPlotPresentation: TypeAlias = tuple[str, str]
_MAX_FIT_COPY_CELLS = 1_000_000
_MAX_FIT_COPY_CHARACTERS = 64 * 1024 * 1024
_FIT_COPY_ROWS_PER_CHUNK = 2048


def _color_value(value: object) -> object:
    if isinstance(value, QtGui.QColor):
        return value
    if isinstance(value, (str, tuple, list)):
        return value
    return str(value)


def _fit_y_axis_label(
    quantity_units: tuple[tuple[str, Unit | None], ...],
    visible_y_names: Iterable[str],
    observed_unit: Unit = MOLAR,
) -> tuple[str, str | None]:
    # Observations retain their admitted concentration meaning.
    units = dict(quantity_units)
    visible = tuple(dict.fromkeys(visible_y_names))
    if observed_unit is not None and observed_unit.compatible(MOLAR) and visible and all(units.get(name) == observed_unit for name in visible):
        return "Concentration", format_unit_for_display(observed_unit)
    return "Value", None


def _fit_copy_columns(dataset):
    """Project complete retained data independently of widget display history."""
    targets = tuple(name for name, _ in dataset.observed_y)
    units = dict(dataset.simulation.quantity_units)
    display = dict(focused_units(units, dataset.simulation.presentation_units, dataset.x_name, targets))
    definitions = dict(dataset.model_definitions)
    rows = focused_model_rows(dataset)
    columns = []

    def add(group, name, values, unit, definition=None):
        header = copy_column_header(f"{dataset.display_name} — {group}", name,
                                    None if unit is None else unit.symbol, definition)
        columns.append((encode_clipboard_text(header), values))

    add("Model", "t", rows[:, 0], SECOND)
    if dataset.x_name != "t":
        name = dataset.x_name
        add("Model", name, scale_values(rows[:, 1], units.get(name), display[name]), display[name], definitions.get(name))
    for index, name in enumerate(targets):
        add("Model", name, scale_values(rows[:, index+2], units.get(name), display[name]), display[name], definitions.get(name))
    observed_x, times, raw = dict(dataset.observed_x), dict(dataset.observed_time), dict(dataset.observed_raw_columns)
    for name, values in dataset.observed_y:
        group = f"Observed {raw[name]}"
        if dataset.x_name != "t" and times[name]:
            add(group, "Recorded time", times[name], SECOND)
        x_name = dataset.observed_x_name or "t"
        x_unit = display[dataset.x_name]
        if x_unit is None or not dataset.observed_x_unit.compatible(x_unit):
            x_unit = dataset.observed_x_unit
        y_unit = display[name]
        if y_unit is None or not dataset.observed_y_unit.compatible(y_unit):
            y_unit = dataset.observed_y_unit
        add(group, x_name, scale_values(observed_x[name], dataset.observed_x_unit, x_unit), x_unit)
        add(group, raw[name], scale_values(values, dataset.observed_y_unit, y_unit), y_unit)
    return columns


class GridPlotView(QtWidgets.QWidget):
    """Render immutable live and terminal Fit curves in the Fit-local plot."""

    copyFitDataRequested = QtCore.Signal(str)

    def __init__(
        self,
        *,
        color_resolver: ColorResolver | None = None,
        observation_style: tuple[str, float, float],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if color_resolver is not None and not callable(color_resolver):
            raise TypeError("color_resolver must be callable.")
        self._color_resolver = color_resolver
        self._color_by_species: Mapping[str, object] = MappingProxyType({})
        self._curves: tuple[FitCurveDataset, ...] = ()
        self._pg = pg
        self._legend_visible = True
        self._observation_style = observation_style
        self._plot_items: dict[str, object] = {}
        self._request: FitExecutionRequest | None = None
        self._dataset_ids: tuple[str, ...] = ()
        self._plot_series_items: dict[str, dict[tuple[str, str], object]] = {}
        self._structure_key: tuple = ()
        self._grid_key: tuple | None = None
        self._dark_mode = False
        self._copy_allowed = True
        self._copy_chunks = None
        self._copy_output: io.StringIO | None = None
        self._copy_timer = QtCore.QTimer(self)
        self._copy_timer.setInterval(0)
        self._copy_timer.timeout.connect(self._continue_copy)
        self._layout_timer = QtCore.QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._update_grid)

        self._context_menu = QtWidgets.QMenu(self)
        self._context_menu.setToolTipsVisible(True)
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
        self._context_actions["range"].setToolTip(
            "Fit both axes to the data once. Automatic ranging remains off afterward."
        )
        self._context_actions["copy"].setToolTip(
            "Copy the model predictions and observations for this dataset at full resolution."
        )
        self._context_actions["pan"].setToolTip("Drag the plot to move the visible area.")
        self._context_actions["rect"].setToolTip(
            "Drag a rectangle to zoom into part of the plot."
        )
        self._context_actions["legend"].setToolTip("Show curve names and colors.")
        self._context_actions["log_x"].setToolTip(
            "Use a logarithmic horizontal scale. Only positive X values can appear."
        )
        self._context_actions["log_y"].setToolTip(
            "Use a logarithmic vertical scale. Only positive Y values can appear."
        )
        self._context_actions["invert_x"].setToolTip(
            "Reverse the direction of the horizontal axis."
        )
        self._context_actions["invert_y"].setToolTip(
            "Reverse the direction of the vertical axis."
        )
        for action in tuple(self._context_actions.values())[2:]:
            action.setCheckable(True)
        self._mouse_mode_group = QtGui.QActionGroup(self)
        self._mouse_mode_group.setExclusive(True)
        self._mouse_mode_group.addAction(self._context_actions["pan"])
        self._mouse_mode_group.addAction(self._context_actions["rect"])
        self._context_target: tuple[str, object] | None = None
        for action in self._context_actions.values():
            action.triggered.connect(self._context_action_triggered)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        legend_row = QtWidgets.QHBoxLayout()
        self._legend_label = QtWidgets.QLabel(self)
        self._legend_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._legend_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                         QtWidgets.QSizePolicy.Policy.Fixed)
        legend_row.addWidget(self._legend_label, 1)
        layout.addLayout(legend_row)
        self._canvas = pg.GraphicsLayoutWidget(self)
        self._canvas.scene().installEventFilter(self)
        self._canvas.scene().sigMouseClicked.connect(self._on_plot_click)
        try:
            self._canvas.useOpenGL(False)
        except (AttributeError, RuntimeError):
            pass
        self._canvas.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._canvas.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._scroll.setWidget(self._canvas)
        self._scroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self._scroll.viewport().installEventFilter(self)
        self._canvas.viewport().installEventFilter(self)
        layout.addWidget(self._scroll)


    def prepare(self, request: FitExecutionRequest, dataset_ids: tuple[str, ...]) -> None:
        """Prepare observations and empty model items without running simulations."""
        self.cancel_copy()
        self._curves = ()
        self._request = request
        self._dataset_ids = dataset_ids
        configs = {item.dataset_id: item for item in request.dispatch.prepared_datasets}
        structure = tuple((identity, configs[identity].selected_observations,
                           configs[identity].parametric_x_observation)
                          for identity in dataset_ids)
        changed = structure != self._structure_key
        self._reconcile_structure(structure)
        self._refresh_appearance()
        for identity in dataset_ids:
            if changed:
                self._plot_items[identity].getViewBox().autoRange()
        self._update_grid()
        self._context_actions["copy"].setEnabled(False)

    def finish_preparation(self) -> None:
        """Lay out and paint hidden views too, then fix native axis gutters."""
        self.ensurePolished()
        self.layout().activate()
        self.grab()  # Activate hidden scroll-area layouts at the stack viewport size.
        self._update_grid()
        self._canvas.grab()
        for plot in self._plot_items.values():
            for side in ("left", "bottom"):
                axis = plot.getAxis(side)
                if side == "left":
                    axis.setWidth(axis.width())
                else:
                    axis.setHeight(axis.height())
                axis.setStyle(autoExpandTextSpace=False, autoReduceTextSpace=False)
        self._update_grid()
        self._canvas.ci.layout.activate()
        self._canvas.grab()

    def set_projection(
        self,
        curves: Iterable[FitCurveDataset],
        presentation: Mapping[str, DatasetPlotPresentation],
    ) -> None:
        """Retain full Copy data and update only existing model-series data."""
        self.cancel_copy()
        self._curves = tuple(curves)
        for curve in self._curves:
            if curve.dataset_id in presentation:
                self._plot_items[curve.dataset_id].setToolTip(presentation[curve.dataset_id][1])
        self._update_models()
        self._context_actions["copy"].setEnabled(self.copy_eligible())

    def _update_models(self):
        curves = {curve.dataset_id: curve for curve in self._curves}
        for identity, series in self._plot_series_items.items():
            curve = curves.get(identity)
            if curve is None:
                for kind, name in series:
                    if kind == "model":
                        self._update_series(identity, kind, name, ((), ()))
                continue
            units = dict(curve.simulation.quantity_units)
            display = dict(focused_units(units, curve.simulation.presentation_units,
                                        curve.x_name, tuple(name for name, _ in curve.observed_y)))
            chunks = focused_model_chunks(curve)
            for name, _ in curve.observed_y:
                xs, ys = [], []
                for chunk in chunks:
                    if xs:
                        xs.append(math.nan)
                        ys.append(math.nan)
                    xs.extend(scale_values(chunk.x, units.get(curve.x_name), display[curve.x_name]))
                    ys.extend(scale_values(dict(chunk.series)[name], units.get(name), display[name]))
                self._update_series(identity, "model", name, (tuple(xs), tuple(ys)))

    def _prepare_context_menu(self, dataset_id: str) -> None:
        plot = self._plot_items[dataset_id]
        view_box = plot.getViewBox()
        state = view_box.getState(copy=False)
        mouse_mode = state["mouseMode"]
        self._context_target = (dataset_id, plot)
        actions = self._context_actions
        actions["copy"].setEnabled(self.copy_eligible())
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
        dataset_id = next((identity for identity, plot in self._plot_items.items()
                           if plot.isVisible() and plot.sceneBoundingRect().contains(scene_position)), None)
        if dataset_id is None:
            return
        self._prepare_context_menu(dataset_id)
        self._context_menu.exec(global_position)
        self._context_target = None

    def _on_plot_click(self, event: object) -> None:
        if event.button() == QtCore.Qt.MouseButton.RightButton and not event.isAccepted():
            event.accept()
            self._show_context_menu(event.scenePos(), event.screenPos().toPoint())

    def eventFilter(self, watched: object, event: QtCore.QEvent) -> bool:
        if (event.type() == QtCore.QEvent.Type.Wheel
                and watched is self._canvas.viewport()
                and len(self._dataset_ids) > 1):
            # Scroll the dataset grid before the plot or its axes consume the wheel.
            self._scroll.wheelEvent(event)
            event.accept()
            return True
        if watched is self._scroll.viewport() and event.type() == QtCore.QEvent.Type.Resize:
            self._layout_timer.start(0)
        if watched is self._canvas.scene() and event.type() == QtCore.QEvent.Type.GraphicsSceneContextMenu:
            if event.reason() != QtWidgets.QGraphicsSceneContextMenuEvent.Reason.Mouse:
                self._show_context_menu(event.scenePos(), event.screenPos())
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _context_action_triggered(self, checked: bool) -> None:
        kind = next(key for key, action in self._context_actions.items() if action is self.sender())
        target = self._context_target
        if target is None:
            return
        dataset_id, plot = target
        if self._plot_items.get(dataset_id) is not plot:
            return
        if kind == "copy":
            if self.copy_eligible():
                self.copyFitDataRequested.emit(dataset_id)
            return
        view_box = plot.getViewBox()
        if kind == "range":
            view_box.autoRange()
            view_box.enableAutoRange(x=False, y=False)
        elif kind in {"pan", "rect"} and checked:
            view_box.setMouseMode(pg.ViewBox.PanMode if kind == "pan" else pg.ViewBox.RectMode)
        elif kind == "legend":
            self._toggle_legend(checked)
            return
        elif kind in {"log_x", "log_y"}:
            before = view_box.viewRange()
            plot.setLogMode(**{kind[-1]: bool(checked)})
            view_box.autoRange()
            view_box.enableAutoRange(x=False, y=False)
            if kind == "log_x":
                view_box.setYRange(*before[1], padding=0)
            else:
                view_box.setXRange(*before[0], padding=0)
        elif kind in {"invert_x", "invert_y"}:
            getattr(plot, f"invert{kind[-1].upper()}")(bool(checked))
        else:
            return

    def copy_eligible(self) -> bool:
        """Whether this Fit-local panel has immutable curves to copy."""

        return self._copy_allowed and bool(self._curves)

    def set_copy_allowed(self, allowed: bool) -> None:
        self._copy_allowed = bool(allowed)
        self._context_actions["copy"].setEnabled(self.copy_eligible())

    def copy_projection_to_clipboard(
        self,
        curves: Iterable[FitCurveDataset],
    ) -> None:
        """Copy the Results owner's immutable plotted arrays for the given scope."""

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

    def cancel_copy(self) -> None:
        """Cancel one pending asynchronous full-resolution Copy operation."""

        self._copy_timer.stop()
        self._copy_chunks = None
        self._copy_output = None

    def _iter_copy_chunks(
        self, curves: tuple[FitCurveDataset, ...],
    ) -> Iterable[str]:
        columns: list[tuple[str, Sequence[float]]] = []
        for dataset in curves:
            dataset_columns = _fit_copy_columns(dataset)
            if columns and dataset_columns:
                columns.append(("", ()))
            columns.extend(dataset_columns)
        if not columns:
            raise ValueError("No Fit curves are available to copy.")
        row_count = max(len(values) for _header, values in columns)
        cell_count = (row_count + 1) * len(columns)
        if cell_count > _MAX_FIT_COPY_CELLS:
            raise ValueError(
                "Fit Copy payload exceeds the bounded full-resolution cell "
                f"limit ({_MAX_FIT_COPY_CELLS:,})."
            )

        total_characters = 0
        output = io.StringIO()
        output.write(encode_tsv_row([header for header, _values in columns]))
        rows_in_chunk = 0
        for row_index in range(row_count):
            row: list[str] = []
            for _header, values in columns:
                if row_index >= len(values):
                    row.append("")
                else:
                    value = float(values[row_index])
                    row.append(repr(value) if math.isfinite(value) else "")
            output.write(encode_tsv_row(row))
            rows_in_chunk += 1
            if rows_in_chunk >= _FIT_COPY_ROWS_PER_CHUNK:
                chunk = output.getvalue()
                total_characters += len(chunk)
                if total_characters > _MAX_FIT_COPY_CHARACTERS:
                    raise ValueError(
                        "Fit Copy payload exceeds the bounded output-size limit "
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
            self.cancel_copy()
            return
        try:
            output.write(next(chunks))
        except StopIteration:
            text = output.getvalue()
            self.cancel_copy()
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
            self.cancel_copy()
            QtWidgets.QMessageBox.critical(
                self,
                "Copy Fit Data",
                str(exc),
            )

    def _panel_geometry(self) -> tuple[int, int, int, int]:
        """Reserve the data floor plus the axes' measured native text space."""

        scale = self.fontMetrics().height() / 14.0
        left = max((plot.getAxis("left").width() for plot in self._plot_items.values()), default=40)
        bottom = max((plot.getAxis("bottom").height() for plot in self._plot_items.values()), default=30)
        top = max((plot.titleLabel.height() for plot in self._plot_items.values()), default=0)
        # PlotItem's own layout has one pixel on either side of the ViewBox.
        return (math.ceil(250 * scale + left + 2),
                math.ceil(170 * scale + bottom + top + 2),
                math.ceil(left), math.ceil(bottom))

    def _grid_geometry(self) -> tuple[tuple[int, int], QtCore.QSize, int]:
        """Resolve canvas geometry without depending on currently visible bars."""

        maximum = self._scroll.maximumViewportSize()
        panel_width, panel_height, left, _bottom = self._panel_geometry()
        scale = self.fontMetrics().height() / 14.0
        horizontal = vertical = False
        while True:
            width = maximum.width() - (
                self._scroll.verticalScrollBar().sizeHint().width() if vertical else 0
            )
            height = maximum.height() - (
                self._scroll.horizontalScrollBar().sizeHint().height() if horizontal else 0
            )
            capacity = max(1, (width - 18) // (panel_width + 6))
            rows = max(1, math.ceil(len(self._dataset_ids) / capacity))
            columns = max(1, math.ceil(len(self._dataset_ids) / rows))
            minimum_width = columns * (panel_width + 6) + 18
            canvas_width = max(width, minimum_width)
            if rows == 1:
                available_data_height = height - 24 - (panel_height - 170 * scale)
                landscape_width = columns * (
                    max(170 * scale, available_data_height) * 1.5 + left + 8
                ) + 18
                canvas_width = min(canvas_width, max(minimum_width, math.floor(landscape_width)))
            data_width = (canvas_width - 18) / columns - 6 - left - 2
            actual_panel_height = panel_height + max(
                0, math.ceil(data_width * 2 / 3 - 170 * scale)
            )
            canvas_height = rows * (actual_panel_height + 6) + 18
            needs_horizontal = canvas_width > width
            needs_vertical = canvas_height > height
            if (not needs_horizontal or horizontal) and (not needs_vertical or vertical):
                return (rows, columns), QtCore.QSize(canvas_width, canvas_height), height
            # Reserve each required bar once. Shrinking the available area may
            # require the other bar, but cannot undo this call's reservation.
            horizontal = horizontal or needs_horizontal
            vertical = vertical or needs_vertical

    def _configure_plot_static(self, plot: object, x_name: str) -> None:
        """Apply common font-scaled axes and grid geometry."""

        plot.setMenuEnabled(False)
        x_label = "Time" if x_name == "t" else x_name
        if x_name == "t":
            plot.setLabel("bottom", x_label, units=format_unit_for_display(SECOND))
        else:
            plot.setLabel("bottom", html.escape(x_label))
        plot.showGrid(x=True, y=True, alpha=0.3)
        font = self.font()
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.enableAutoSIPrefix(False)
            if axis.style["tickFont"] != font:
                axis.setWidth(None) if axis_name == "left" else axis.setHeight(None)
            axis.setTickFont(font)
            axis.setStyle(autoExpandTextSpace=True, autoReduceTextSpace=True)
            axis.label.setFont(font)
        self._apply_dark_mode_to_plot(plot)

    def _apply_dark_mode_to_plot(self, plot: object) -> None:
        text_color = plot_foreground_color(self._dark_mode)
        for axis_name in ("left", "bottom"):
            try:
                axis = plot.getAxis(axis_name)
                axis.setTextPen(text_color)
                axis.setPen(text_color)
            except Exception:
                pass

    def legend_text(self) -> tuple[str, str]:
        entries = []
        names = []
        for species, color in self._color_by_species.items():
            names.append(species)
            entries.append(f'<span style="color:{self._pg.mkColor(color).name()}">●</span> '
                           + html.escape(species))
        marker = {"o": "●", "s": "■", "t": "▼", "d": "◆", "x": "×", "+": "+"}[self._observation_style[0]]
        return (" &nbsp; ".join(entries) + f" &nbsp; — Model &nbsp; {marker} Observed",
                "\n".join(names) + "\n\n— Model line\nObserved markers")

    def _update_legend(self) -> None:
        rich, plain = self.legend_text()
        document = QtGui.QTextDocument()
        document.setDefaultFont(self.font())
        document.setHtml(rich)
        overflow = document.idealWidth() > max(1, self.contentsRect().width())
        if overflow:
            marker = {"o": "●", "s": "■", "t": "▼", "d": "◆", "x": "×", "+": "+"}[self._observation_style[0]]
            rich = f"{len(self._color_by_species)} series &nbsp; — Model &nbsp; {marker} Observed"
        self._legend_label.setText(rich)
        self._legend_label.setFixedHeight(self.fontMetrics().lineSpacing())
        self._legend_label.setAccessibleName(plain)
        self._legend_label.setVisible(self._legend_visible and bool(self._dataset_ids))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_legend()

    def set_observation_style(self, shape: str, size: float, opacity: float) -> None:
        if self._observation_style != (shape, size, opacity):
            self._observation_style = (shape, size, opacity)
            self._refresh_appearance()

    def _reconcile_structure(self, structure: tuple) -> None:
        ids = set(self._dataset_ids)
        for identity in tuple(self._plot_items):
            if identity in ids:
                continue
            if self._context_target is not None and self._context_target[0] == identity:
                self._context_target = None
                self._context_menu.close()
            plot = self._plot_items.pop(identity)
            self._canvas.ci.removeItem(plot)
            plot.deleteLater()
            self._plot_series_items.pop(identity)
        for identity, observations, _x in structure:
            names = tuple(item.model_target for item in observations)
            if identity not in self._plot_items:
                plot = self._pg.PlotItem(enableMenu=False)
                plot.getViewBox().enableAutoRange(x=False, y=False)
                self._plot_items[identity] = plot
                self._plot_series_items[identity] = {}
            series = self._plot_series_items[identity]
            active = {(kind, name) for name in names for kind in ("observed", "model")}
            for key in tuple(series):
                if key not in active:
                    item = series.pop(key)
                    self._plot_items[identity].removeItem(item)
                    item.deleteLater()
        if structure != self._structure_key:
            self._grid_key = None
        self._structure_key = structure

    def _update_panel(self, identity, display_name, x_name, quantity_units, names, x_display, y_display) -> None:
        plot = self._plot_items[identity]
        name = compact_plot_title_label(display_name, max_chars=PLOT_TITLE_LABEL_MAX_CHARS)
        title, tooltip = name.display, f"{name.full}\nDataset ID: {identity}"
        label, units = _fit_y_axis_label(quantity_units, names, y_display)
        state = (x_name, title, tooltip, label, units, x_display, len(self._dataset_ids) > 1,
                 self._dark_mode, self.font().toString())
        if getattr(plot, "_kindred_appearance", None) != state:
            self._configure_plot_static(plot, x_name)
            plot.setTitle(html.escape(title) if len(self._dataset_ids) > 1 else "",
                          color=plot_foreground_color(self._dark_mode),
                          size=f"{self.font().pointSizeF()}pt")
            plot.setToolTip(tooltip)
            plot.setLabel("left", label, units=units)
            model_x = dict(quantity_units).get(x_name)
            observed_x = SECOND if x_name == "t" else MOLAR
            common_x = (x_display is not None and model_x is not None
                        and model_x.compatible(x_display) and observed_x.compatible(x_display))
            plot.setLabel("bottom", "Time" if x_name == "t" else html.escape(x_name),
                          units=format_unit_for_display(x_display) if common_x else None)
            plot._kindred_appearance = state

    def _update_series(self, identity: str, kind: str, species: str, data: tuple) -> None:
        series = self._plot_series_items[identity]
        key = (kind, species)
        item = series.get(key)
        if item is None:
            item = self._plot_items[identity].plot([], [], pen=None)
            series[key] = item
        item.setVisible(True)
        color = self._pg.mkColor(self._color_by_species[species])
        style = (color.rgba(), self._observation_style if kind == "observed" else None)
        if getattr(item, "_kindred_style", None) != style:
            if kind == "observed":
                shape, size, opacity = self._observation_style
                color.setAlphaF(opacity)
                item.setSymbol(shape)
                item.setSymbolSize(size)
                item.setSymbolBrush(self._pg.mkBrush(color))
                item.setSymbolPen(self._pg.mkPen(color=color, width=1))
            else:
                item.setPen(self._pg.mkPen(color=color, width=2))
            item._kindred_style = style
        if getattr(item, "_kindred_data", None) != data:
            x, y = data
            item.setData(x, y, connect="finite")
            item._kindred_data = data

    def _toggle_legend(self, visible: bool) -> None:
        self._legend_visible = bool(visible)
        self._update_legend()

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.FontChange and hasattr(self, "_scroll"):
            self._refresh_appearance()
            self.finish_preparation()

    def set_dark_mode(self, enabled: bool) -> None:
        """Refresh background, labels, and resolved species colors together."""
        self._dark_mode = bool(enabled)
        self._canvas.setBackground(plot_background_color(self._dark_mode))
        self._refresh_appearance()

    def _update_grid(self) -> None:
        self._layout_timer.stop()
        if not self._dataset_ids:
            return
        grid_size, canvas_size, available_height = self._grid_geometry()
        if (self._canvas.minimumSize() != canvas_size
                or self._canvas.maximumSize() != canvas_size):
            self._canvas.setFixedSize(canvas_size)
        vertical = (QtCore.Qt.AlignmentFlag.AlignVCenter
                    if grid_size[0] == 1 and canvas_size.height() <= available_height
                    else QtCore.Qt.AlignmentFlag.AlignTop)
        self._scroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | vertical)
        key = (self._dataset_ids, grid_size)
        if key != self._grid_key:
            for slot in self._plot_items.values():
                if slot in self._canvas.ci.items:
                    self._canvas.ci.removeItem(slot)
            for index, identity in enumerate(self._dataset_ids):
                row, column = divmod(index, grid_size[1])
                self._canvas.ci.addItem(self._plot_items[identity], row=row, col=column)
            self._grid_key = key

    def _refresh_appearance(self) -> None:
        request = self._request
        if request is None:
            return
        configs = {item.dataset_id: item for item in request.dispatch.prepared_datasets}
        mapped = {item.dataset_id: item for item in request.setup.origin.mapped_datasets}
        species_names = tuple(dict.fromkeys(
            item.model_target for identity in self._dataset_ids
            for item in configs[identity].selected_observations))
        if species_names and self._color_resolver is None:
            raise ValueError("Fit curves require an injected species-color resolver.")
        self._color_by_species = MappingProxyType({
            species: _color_value(self._color_resolver(species)) for species in species_names})
        for identity in self._dataset_ids:
            config = configs[identity]
            names = tuple(item.model_target for item in config.selected_observations)
            x = config.parametric_x_observation
            x_name = "t" if x is None else x.model_target
            quantities, display_pairs = prepared_fit_units(request.setup.origin.prepared_dsl, x_name, names)
            display = dict(display_pairs)
            x_display = display[x_name]
            y_units = tuple(display[name] for name in names)
            y_display = y_units[0] if y_units and all(unit == y_units[0] for unit in y_units) else None
            self._update_panel(identity, mapped[identity].display_name, x_name,
                               display_pairs, names, x_display, y_display)
            observed_x, observed_y = _observed_curve_parts(config.selected_observations, x_observations=x)
            for species, y in observed_y:
                self._update_series(identity, "observed", species, (
                    scale_values(dict(observed_x)[species], SECOND if x is None else MOLAR, x_display),
                    scale_values(y, MOLAR, display[species])))
                if ("model", species) not in self._plot_series_items[identity]:
                    self._update_series(identity, "model", species, ((), ()))
        self._update_legend()
        self._update_models()
