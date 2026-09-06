"""Lean Fit-local grid plot with injected species-color resolution."""

from __future__ import annotations

import io
import html
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import TypeAlias

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.display_name_policy import (
    PLOT_TITLE_LABEL_MAX_CHARS,
    compact_plot_title_label,
)
from kindred.gui.fitting.worker import FitCurveDataset, FitCurveYSemantic
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


def _fit_copy_columns(
    dataset: FitCurveDataset,
) -> list[tuple[str, Sequence[object], str]]:
    """Build one dataset's model and observation columns in plot-series order."""

    dataset_label = f"{dataset.display_name} [{dataset.dataset_id}]"
    dataset_identity = {
        "dataset_id": dataset.dataset_id,
        "dataset_name": dataset.display_name,
    }

    def header(
        label: str,
        name: str,
        identity: Mapping[str, str],
        unit: str | None = None,
    ) -> str:
        displayed_name = f"{name} ({unit})" if unit is not None else name
        provenance = json.dumps(
            {**dataset_identity, **identity}, sort_keys=True, separators=(",", ":")
        )
        return encode_clipboard_text(
            f"{label}::{displayed_name} [provenance={provenance}]"
        )

    model_x = dict(dataset.model_x)
    model_y = dict(dataset.model_y)
    segment_ids = dict(dataset.model_segment_ids)
    segment_keys = dict(dataset.model_segment_keys)
    discontinuity_ids = dict(dataset.model_discontinuity_ids)
    selection = dict(dataset.model_selection_provenance)
    series_names = sorted(model_y)
    x_unit = "s" if dataset.x_name.casefold() == "t" else None
    _y_label, y_unit = _fit_y_axis_label(dataset.y_semantics, series_names)
    model_groups: dict[tuple[object, ...], list[str]] = {}
    for name in series_names:
        # Repeated X values on different execution segments are separate grids.
        grid = (
            model_x[name], segment_ids[name], segment_keys[name],
            discontinuity_ids[name],
        )
        model_groups.setdefault(grid, []).append(name)

    columns: list[tuple[str, Sequence[object], str]] = []
    for index, names in enumerate(model_groups.values(), 1):
        suffix = f" {index}" if len(model_groups) > 1 else ""
        label = f"{dataset_label} — Model{suffix}"
        first = names[0]
        x_identity = {} if x_unit is not None else {"model_target": dataset.x_name}
        columns.append(
            (header(label, dataset.x_name, x_identity, x_unit), model_x[first], "float")
        )
        columns.extend(
            (
                encode_clipboard_text(f"{label} — {field}"), values, kind,
            )
            for field, values, kind in (
                ("segment_id", segment_ids[first], "int"),
                ("segment_key", segment_keys[first], "text"),
                ("discontinuity_id", discontinuity_ids[first], "int"),
            )
        )
        for name in names:
            identity = {"model_target": name}
            kinds = selection.get(name, ())
            varying_selection = bool(kinds) and len(set(kinds)) > 1
            if kinds and not varying_selection:
                identity["selection_provenance"] = kinds[0]
            columns.append((header(label, name, identity, y_unit), model_y[name], "float"))
            if varying_selection:
                columns.append(
                    (header(label, f"{name} — selection_provenance", {"model_target": name}), kinds, "text")
                )

    observed_x = dict(dataset.observed_x)
    observed_y = dict(dataset.observed_y)
    observation_ids = dict(dataset.observed_observation_ids)
    raw_columns = dict(dataset.observed_raw_columns)
    source_rows = dict(dataset.observed_source_rows)
    for name in sorted(observed_y):
        label = f"{dataset_label} — Observed {name}"
        if dataset.observed_x_identity is None:
            x_identity = {"raw_observation_id": observation_ids[name]}
        else:
            observation_id, raw_column = dataset.observed_x_identity
            x_identity = {
                "raw_observation_id": observation_id,
                "raw_column": raw_column,
                "model_target": dataset.x_name,
            }
        y_identity = {
            "raw_observation_id": observation_ids[name],
            "raw_column": raw_columns[name],
            "model_target": name,
        }
        columns.extend(
            (
                (header(label, dataset.x_name, x_identity, x_unit), observed_x[name], "float"),
                (encode_clipboard_text(f"{label} — source_row"), source_rows[name], "int"),
                (header(label, raw_columns[name], y_identity, y_unit), observed_y[name], "float"),
            )
        )
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
        self._dataset_presentation: Mapping[str, DatasetPlotPresentation] = (
            MappingProxyType({})
        )
        self._pg = pg
        self._legend_visible = True
        self._observation_style = observation_style
        self._plot_items: list[object] = []
        self._plot_series_items: list[dict[str, object]] = []
        self._plot_display_state: dict[str, tuple[object, ...]] = {}
        self._structure_key: tuple | None = None
        self._dark_mode = False
        self._copy_allowed = True
        self._copy_chunks = None
        self._copy_output: io.StringIO | None = None
        self._copy_timer = QtCore.QTimer(self)
        self._copy_timer.setInterval(0)
        self._copy_timer.timeout.connect(self._continue_copy)
        self._layout_timer = QtCore.QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._redraw)

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

        legend_row = QtWidgets.QHBoxLayout()
        self._legend_label = QtWidgets.QLabel(self)
        self._legend_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._legend_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                         QtWidgets.QSizePolicy.Policy.Fixed)
        legend_row.addWidget(self._legend_label, 1)
        layout.addLayout(legend_row)

        self._canvas = pg.GraphicsLayoutWidget(self)
        self._canvas.scene().installEventFilter(self)
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
        layout.addWidget(self._scroll)


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
        normalized_presentation = MappingProxyType(dict(presentation))
        self._dataset_presentation = normalized_presentation
        self._curves = values
        self._context_actions["copy"].setEnabled(self.copy_eligible())
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
        plot_index = next((index for index, plot in enumerate(self._plot_items)
                           if plot.sceneBoundingRect().contains(scene_position)), None)
        if plot_index is None:
            self._context_plot_index = None
            return
        self._prepare_context_menu(plot_index)
        self._context_menu.exec(global_position)
        self._context_plot_index = None

    def eventFilter(self, watched: object, event: QtCore.QEvent) -> bool:
        if watched is self._scroll.viewport() and event.type() == QtCore.QEvent.Type.Resize:
            self._layout_timer.start(0)
        if watched is self._canvas.scene() and event.type() == QtCore.QEvent.Type.GraphicsSceneContextMenu:
            self._show_context_menu(event.scenePos(), event.screenPos())
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _context_action_triggered(self, checked: bool) -> None:
        kind = next(key for key, action in self._context_actions.items() if action is self.sender())
        index = self._context_plot_index
        if kind == "copy":
            if index is not None and 0 <= index < len(self._curves):
                self.copyFitDataRequested.emit(self._curves[index].dataset_id)
            return
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

    def _cancel_copy(self) -> None:
        """Cancel one pending asynchronous full-resolution Copy operation."""

        self._copy_timer.stop()
        self._copy_chunks = None
        self._copy_output = None

    @staticmethod
    def _iter_copy_chunks(
        curves: tuple[FitCurveDataset, ...],
    ) -> Iterable[str]:
        columns: list[tuple[str, Sequence[object], str]] = []
        for dataset in curves:
            dataset_columns = _fit_copy_columns(dataset)
            if columns and dataset_columns:
                columns.append(("", (), "text"))
            columns.extend(dataset_columns)
        if not columns:
            raise ValueError("No Fit curves are available to copy.")
        row_count = max(len(values) for _header, values, _kind in columns)
        cell_count = (row_count + 1) * len(columns)
        if cell_count > _MAX_FIT_COPY_CELLS:
            raise ValueError(
                "Fit Copy payload exceeds the bounded full-resolution cell "
                f"limit ({_MAX_FIT_COPY_CELLS:,})."
            )

        total_characters = 0
        output = io.StringIO()
        output.write(encode_tsv_row([header for header, _values, _kind in columns]))
        rows_in_chunk = 0
        for row_index in range(row_count):
            row: list[str] = []
            for _header, values, kind in columns:
                if row_index >= len(values):
                    row.append("")
                elif kind == "float":
                    row.append(repr(float(values[row_index])))
                elif kind == "int":
                    row.append(str(int(values[row_index])))
                else:
                    row.append(encode_clipboard_text(str(values[row_index])))
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

    def _panel_geometry(self) -> tuple[int, int, int, int]:
        """Reserve the data floor plus the axes' measured native text space."""

        scale = self.fontMetrics().height() / 14.0
        left = max((plot.getAxis("left").width() for plot in self._plot_items), default=40)
        bottom = max((plot.getAxis("bottom").height() for plot in self._plot_items), default=30)
        top = max((plot.titleLabel.height() for plot in self._plot_items), default=0)
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
            rows = max(1, math.ceil(len(self._curves) / capacity))
            columns = max(1, math.ceil(len(self._curves) / rows))
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

    def _dataset_structure_key(
        self, curves: tuple[FitCurveDataset, ...], grid_size: tuple[int, int]
    ) -> tuple:
        return (tuple((curve.dataset_id, curve.x_name) for curve in curves),
                grid_size, self.font().toString())

    def _configure_plot_static(self, plot: object, x_name: str) -> None:
        """Apply common font-scaled axes and grid geometry."""

        plot.setMenuEnabled(False)
        x_label = "Time" if x_name.casefold() == "t" else x_name
        if x_name.casefold() == "t":
            plot.setLabel("bottom", x_label, units="s")
        else:
            plot.setLabel("bottom", html.escape(x_label))
        plot.setLabel("left", "Value")
        plot.showGrid(x=True, y=True, alpha=0.3)
        font = self.font()
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
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
        self._legend_label.setVisible(self._legend_visible and bool(self._curves))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_legend()

    def set_observation_style(self, shape: str, size: float, opacity: float) -> None:
        self._observation_style = (shape, size, opacity)
        self._redraw()

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
                symbol=self._observation_style[0],
                symbolSize=self._observation_style[1],
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
                title_color = plot_foreground_color(self._dark_mode)
                plot.setTitle(html.escape(title) if len(self._curves) > 1 else "",
                              color=title_color, size=f"{self.font().pointSizeF()}pt")
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
                for name in self._color_by_species
                if name in model_y or name in observed_y
            )
            y_label, y_units = _fit_y_axis_label(
                curve.y_semantics,
                visible_y_names,
            )
            plot.setLabel("left", y_label, units=y_units)

            for species in self._color_by_species:
                color = self._color_by_species[species]
                if species in observed_y:
                    observed_item = self._ensure_curve_item(
                        plot,
                        dataset_index,
                        f"{species}::observed",
                        is_model=False,
                        color=color,
                    )
                    shape, size, opacity = self._observation_style
                    marker_color = self._pg.mkColor(color)
                    marker_color.setAlphaF(opacity)
                    observed_item.setSymbol(shape)
                    observed_item.setSymbolSize(size)
                    observed_item.setSymbolBrush(self._pg.mkBrush(marker_color))
                    observed_item.setSymbolPen(self._pg.mkPen(color=marker_color, width=1))
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

    def _rebuild_structure(self, grid_size: tuple[int, int]) -> None:
        self._canvas.setUpdatesEnabled(False)
        try:
            ids = tuple(curve.dataset_id for curve in self._curves)
            previous_ids = () if self._structure_key is None else tuple(item[0] for item in self._structure_key[0])
            if ids and ids == previous_ids and len(self._plot_items) == len(ids):
                _rows, columns = grid_size
                for plot in self._plot_items:
                    self._canvas.ci.removeItem(plot)
                for index, (curve, plot) in enumerate(zip(self._curves, self._plot_items)):
                    row, column = divmod(index, columns)
                    self._canvas.ci.addItem(plot, row=row, col=column)
                    self._configure_plot_static(plot, curve.x_name)
                self._structure_key = self._dataset_structure_key(self._curves, grid_size)
                return
            self._canvas.clear()
            self._plot_items.clear()
            self._plot_series_items.clear()
            if not self._curves:
                text = self._pg.TextItem(
                    "No Fit curves yet",
                    color=plot_foreground_color(self._dark_mode),
                    anchor=(0.5, 0.5),
                )
                placeholder = self._canvas.addPlot(row=0, col=0)
                placeholder.setMenuEnabled(False)
                placeholder.hideAxis("left")
                placeholder.hideAxis("bottom")
                placeholder.addItem(text)
                text.setPos(0.5, 0.5)
                self._structure_key = self._dataset_structure_key((), grid_size)
                return

            _rows, columns = grid_size
            for index, curve in enumerate(self._curves):
                row, column = divmod(index, columns)
                plot = self._canvas.addPlot(row=row, col=column)
                self._plot_items.append(plot)
                self._plot_series_items.append({})
                self._configure_plot_static(plot, curve.x_name)
                for axis_name in ("left", "bottom"):
                    plot.getAxis(axis_name).geometryChanged.connect(self._layout_timer.start)
                self._restore_plot_display_state(plot, curve.dataset_id)
            self._structure_key = self._dataset_structure_key(self._curves, grid_size)
        finally:
            self._canvas.setUpdatesEnabled(True)

    def _toggle_legend(self, visible: bool) -> None:
        self._legend_visible = bool(visible)
        self._update_legend()

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.FontChange and hasattr(self, "_scroll"):
            self._layout_timer.start(0)

    def set_dark_mode(self, enabled: bool) -> None:
        """Refresh background, labels, and resolved species colors together."""

        self._dark_mode = bool(enabled)
        try:
            self._canvas.setBackground(
                plot_background_color(self._dark_mode)
            )
        except Exception:
            pass
        for plot in self._plot_items:
            self._apply_dark_mode_to_plot(plot)
        # Qt can deliver font changes while traversing the styled widget tree.
        # Rebuild plot structure only after that traversal has finished.
        self._layout_timer.start(0)

    def _update_grid(self) -> None:
        grid_size, canvas_size, available_height = self._grid_geometry()
        if (self._canvas.minimumSize() != canvas_size
                or self._canvas.maximumSize() != canvas_size):
            self._canvas.setFixedSize(canvas_size)
        vertical = (QtCore.Qt.AlignmentFlag.AlignVCenter
                    if grid_size[0] == 1 and canvas_size.height() <= available_height
                    else QtCore.Qt.AlignmentFlag.AlignTop)
        self._scroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | vertical)
        desired_key = self._dataset_structure_key(self._curves, grid_size)
        if (
            self._structure_key != desired_key
            or len(self._plot_items) != len(self._curves)
            or len(self._plot_series_items) != len(self._curves)
        ):
            self._rebuild_structure(grid_size)
        self._update_plot_data_in_place()
        for curve, plot in zip(self._curves, self._plot_items):
            if curve.dataset_id not in self._plot_display_state:
                view_box = plot.getViewBox()
                view_box.autoRange()
                view_box.enableAutoRange(x=True, y=True)
                self._plot_display_state[curve.dataset_id] = self._plot_display_state_for_plot(plot)


    def _redraw(self) -> bool:
        self._layout_timer.stop()
        species_names = tuple(
            dict.fromkeys(
                species
                for curve in self._curves
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
        self._update_legend()
        self._update_grid()
        return bool(
            self._curves
            and len(self._plot_items) == len(self._curves)
            and len(self._plot_series_items) == len(self._curves)
        )
