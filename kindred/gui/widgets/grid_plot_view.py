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
        self._plot_items: dict[str, object] = {}
        self._slots: dict[str, object] = {}
        self._plot_series_items: dict[str, dict[tuple[str, str], object]] = {}
        self._structure_key: tuple = ()
        self._grid_key: tuple | None = None
        self._jobs: list[tuple[str, str, str | None]] = []
        self._cursor = 0
        self._render_revision = 0
        self._pass_revision = 0
        self._render_pending = False
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(1)
        self._render_timer.timeout.connect(self._render_step)
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
    ) -> None:
        """Retain complete curves immediately; render the visible view incrementally."""

        self.cancel_copy()
        self._curves = tuple(curves)
        self._dataset_presentation = MappingProxyType(dict(presentation))
        structure = tuple((curve.dataset_id, tuple(name for name, _ in curve.observed_y),
                           tuple(name for name, _ in curve.model_y)) for curve in self._curves)
        if structure != self._structure_key:
            self._reconcile_structure(structure)
        self._context_actions["copy"].setEnabled(self.copy_eligible())
        self._refresh_appearance()
        self._update_grid()

    def _schedule_render(self) -> None:
        if self._render_pending and self.isVisible() and not self._render_timer.isActive():
            self._render_timer.start()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._schedule_render()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self._render_timer.stop()
        self._layout_timer.stop()
        super().hideEvent(event)

    def _plot_display_state_for_plot(self, plot: object) -> tuple[object, ...]:
        state = plot.getViewBox().getState(copy=True)
        return (state["mouseMode"], tuple(state["logMode"]), state["xInverted"],
                state["yInverted"], tuple(state["autoRange"]),
                tuple(tuple(axis) for axis in state["viewRange"]))

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

    def _configure_plot_static(self, plot: object, x_name: str) -> None:
        """Apply common font-scaled axes and grid geometry."""

        plot.setMenuEnabled(False)
        x_label = "Time" if x_name.casefold() == "t" else x_name
        if x_name.casefold() == "t":
            plot.setLabel("bottom", x_label, units="s")
        else:
            plot.setLabel("bottom", html.escape(x_label))
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
        if self._observation_style != (shape, size, opacity):
            self._observation_style = (shape, size, opacity)
            self._refresh_appearance()

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

    def _reconcile_structure(self, structure: tuple) -> None:
        self._render_timer.stop()
        ids = {curve.dataset_id for curve in self._curves}
        for identity in tuple(self._slots):
            if identity in ids:
                continue
            if self._context_target is not None and self._context_target[0] == identity:
                self._context_target = None
                self._context_menu.close()
            slot = self._slots.pop(identity)
            self._canvas.ci.removeItem(slot)
            slot.deleteLater()
            self._plot_items.pop(identity, None)
            self._plot_series_items.pop(identity, None)
        self._jobs = []
        for identity, observed, model in structure:
            if identity not in self._slots:
                self._slots[identity] = self._pg.GraphicsWidget()
            series = self._plot_series_items.get(identity, {})
            active = {(kind, name) for kind, names in (("observed", observed), ("model", model))
                      for name in names}
            for key in tuple(series):
                if key not in active:
                    item = series.pop(key)
                    self._plot_items[identity].removeItem(item)
                    item.deleteLater()
            self._jobs.append((identity, "panel", None))
            self._jobs.extend((identity, "observed", name) for name in observed)
            self._jobs.extend((identity, "model", name) for name in model)
            self._jobs.append((identity, "finish", None))
        self._structure_key = structure
        self._cursor = 0
        self._grid_key = None
        if not structure:
            self._layout_timer.stop()
            self._render_pending = False

    def _render_step(self) -> None:
        if not self.isVisible() or not self._render_pending:
            return
        if self._cursor == 0:
            self._pass_revision = self._render_revision
        identity, kind, species = self._jobs[self._cursor]
        curve = next(curve for curve in self._curves if curve.dataset_id == identity)
        if kind == "panel":
            self._update_panel(curve)
        elif kind == "finish":
            plot = self._plot_items[identity]
            if not plot.isVisible():
                plot.show()
                plot.getViewBox().autoRange()
                plot.getViewBox().enableAutoRange(x=True, y=True)
        else:
            self._update_series(curve, kind, species)
        self._cursor += 1
        if self._cursor == len(self._jobs):
            self._cursor = 0
            self._render_pending = self._pass_revision != self._render_revision
        self._schedule_render()

    def _update_panel(self, curve: FitCurveDataset) -> None:
        identity = curve.dataset_id
        plot = self._plot_items.get(identity)
        if plot is None:
            placeholder = self._slots[identity]
            row, column = self._canvas.ci.items[placeholder][0]
            self._canvas.ci.removeItem(placeholder)
            placeholder.deleteLater()
            plot = self._pg.PlotItem(enableMenu=False)
            policy = plot.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            plot.setSizePolicy(policy)
            plot.hide()
            plot.getViewBox().enableAutoRange(x=False, y=False)
            self._canvas.ci.addItem(plot, row=row, col=column)
            self._slots[identity] = self._plot_items[identity] = plot
            self._plot_series_items[identity] = {}
            for side in ("left", "bottom"):
                plot.getAxis(side).geometryChanged.connect(self._layout_timer.start)
        presentation = self._dataset_presentation.get(identity)
        if presentation is None:
            name = compact_plot_title_label(curve.display_name, max_chars=PLOT_TITLE_LABEL_MAX_CHARS)
            title, tooltip = name.display, f"{name.full}\nDataset ID: {identity}"
        else:
            title, tooltip = presentation
        names = tuple(dict.fromkeys(name for name, _ in (*curve.model_y, *curve.observed_y)))
        label, units = _fit_y_axis_label(curve.y_semantics, names)
        state = (curve.x_name, title, tooltip, label, units, len(self._curves) > 1,
                 self._dark_mode, self.font().toString())
        if getattr(plot, "_kindred_appearance", None) != state:
            self._configure_plot_static(plot, curve.x_name)
            plot.setTitle(html.escape(title) if len(self._curves) > 1 else "",
                          color=plot_foreground_color(self._dark_mode),
                          size=f"{self.font().pointSizeF()}pt")
            plot.setToolTip(tooltip)
            plot.setLabel("left", label, units=units)
            plot._kindred_appearance = state

    def _update_series(self, curve: FitCurveDataset, kind: str, species: str) -> None:
        identity = curve.dataset_id
        series = self._plot_series_items[identity]
        key = (kind, species)
        item = series.get(key)
        if item is None:
            item = self._plot_items[identity].plot([], [], pen=None)
            series[key] = item
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
        if kind == "observed":
            data = (dict(curve.observed_x)[species], dict(curve.observed_y)[species])
        else:
            data = (dict(curve.model_x)[species], dict(curve.model_y)[species],
                    dict(curve.model_segment_ids)[species], dict(curve.model_segment_keys)[species],
                    dict(curve.model_discontinuity_ids)[species])
        if getattr(item, "_kindred_data", None) != data:
            x, y = self._with_segment_gaps(*data) if kind == "model" else data
            item.setData(x, y)
            item._kindred_data = data

    def _toggle_legend(self, visible: bool) -> None:
        self._legend_visible = bool(visible)
        self._update_legend()

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.FontChange and hasattr(self, "_scroll"):
            self._refresh_appearance()
            self._layout_timer.start(0)

    def set_dark_mode(self, enabled: bool) -> None:
        """Refresh background, labels, and resolved species colors together."""
        self._dark_mode = bool(enabled)
        self._canvas.setBackground(plot_background_color(self._dark_mode))
        self._refresh_appearance()

    def _update_grid(self) -> None:
        self._layout_timer.stop()
        if not self._curves:
            return
        grid_size, canvas_size, available_height = self._grid_geometry()
        if (self._canvas.minimumSize() != canvas_size
                or self._canvas.maximumSize() != canvas_size):
            self._canvas.setFixedSize(canvas_size)
        vertical = (QtCore.Qt.AlignmentFlag.AlignVCenter
                    if grid_size[0] == 1 and canvas_size.height() <= available_height
                    else QtCore.Qt.AlignmentFlag.AlignTop)
        self._scroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | vertical)
        key = (tuple(curve.dataset_id for curve in self._curves), grid_size)
        if key != self._grid_key:
            for slot in self._slots.values():
                if slot in self._canvas.ci.items:
                    self._canvas.ci.removeItem(slot)
            for index, curve in enumerate(self._curves):
                row, column = divmod(index, grid_size[1])
                self._canvas.ci.addItem(self._slots[curve.dataset_id], row=row, col=column)
            self._grid_key = key

    def _refresh_appearance(self) -> None:
        species_names = tuple(dict.fromkeys(
            species for curve in self._curves
            for species, _series in (*curve.model_y, *curve.observed_y)))
        if species_names and self._color_resolver is None:
            raise ValueError("Fit curves require an injected species-color resolver.")
        self._color_by_species = MappingProxyType({
            species: _color_value(self._color_resolver(species)) for species in species_names})
        self._update_legend()
        self._render_revision += 1
        self._render_pending = bool(self._jobs)
        self._schedule_render()
