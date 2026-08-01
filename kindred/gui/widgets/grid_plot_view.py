"""Lean Fit-local grid plot with injected species-color resolution."""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Callable, Iterable

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.fitting.worker import FitCurveDataset
from kindred.gui.plot_config import try_import_pyqtgraph

__all__ = ["GridPlotView"]

ColorResolver = Callable[[str], object]
_FIT_COPY_COLUMNS = 10
_MAX_FIT_COPY_CELLS = 1_000_000
_MAX_FIT_COPY_CHARACTERS = 64 * 1024 * 1024
_FIT_COPY_ROWS_PER_CHUNK = 2048


def _default_color(name: str) -> str:
    digest = hashlib.sha256(str(name).encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") % 360
    return QtGui.QColor.fromHsv(hue, 180, 220).name()


def _color_value(value: object) -> object:
    if isinstance(value, QtGui.QColor):
        return value
    if isinstance(value, (str, tuple, list)):
        return value
    return str(value)


class GridPlotView(QtWidgets.QWidget):
    """Render immutable live/terminal Fit curves without shared plot authority."""

    def __init__(
        self,
        *,
        color_resolver: ColorResolver | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if color_resolver is not None and not callable(color_resolver):
            raise TypeError("color_resolver must be callable.")
        self._color_resolver = color_resolver or _default_color
        self._curves: tuple[FitCurveDataset, ...] = ()
        self._pg = None
        self._canvas = None
        self._message = None
        self._copy_chunks = None
        self._copy_output: io.StringIO | None = None
        self._copy_timer = QtCore.QTimer(self)
        self._copy_timer.setInterval(0)
        self._copy_timer.timeout.connect(self._continue_copy)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        ok, pg, _error = try_import_pyqtgraph()
        if ok:
            self._pg = pg
            self._canvas = pg.GraphicsLayoutWidget(self)
            try:
                self._canvas.useOpenGL(False)
            except (AttributeError, RuntimeError):
                pass
            layout.addWidget(self._canvas)
        else:
            self._message = QtWidgets.QLabel(
                "Plotting is unavailable; Fit values and Apply remain usable.",
                self,
            )
            self._message.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._message)

    def set_curves(self, curves: Iterable[FitCurveDataset]) -> None:
        values = tuple(curves)
        if any(not isinstance(item, FitCurveDataset) for item in values):
            raise TypeError("curves must contain FitCurveDataset values.")
        self._curves = values
        self._redraw()

    def clear(self) -> None:
        self._curves = ()
        self._redraw()

    def copy_all_to_clipboard(self) -> None:
        if self._copy_timer.isActive():
            return
        self._copy_chunks = iter(self._iter_copy_chunks())
        self._copy_output = io.StringIO()
        self._copy_timer.start()

    def _iter_copy_chunks(self) -> Iterable[str]:
        row_count = sum(
            len(values)
            for dataset in self._curves
            for values_by_kind in (dataset.model_y, dataset.observed_y)
            for _species, values in values_by_kind
        )
        cell_count = (row_count + 1) * _FIT_COPY_COLUMNS
        if cell_count > _MAX_FIT_COPY_CELLS:
            raise ValueError(
                "Fit Copy payload exceeds the bounded full-resolution cell "
                f"limit ({_MAX_FIT_COPY_CELLS:,})."
            )
        header = (
            "dataset\tkind\tseries\traw_observation_id\tsource_row"
            "\tsegment_id\tdiscontinuity\tx_name\tx\ty"
        )
        total_characters = 0
        output = io.StringIO()
        output.write(header)
        rows_in_chunk = 0
        for dataset in self._curves:
            for kind, axes, values in (
                ("model", dict(dataset.model_x), dict(dataset.model_y)),
                (
                    "observed",
                    dict(dataset.observed_x),
                    dict(dataset.observed_y),
                ),
            ):
                segment_ids = (
                    dict(dataset.model_segment_ids)
                    if kind == "model"
                    else {}
                )
                source_rows = (
                    dict(dataset.observed_source_rows)
                    if kind == "observed"
                    else {}
                )
                raw_ids = (
                    dict(dataset.observed_raw_ids)
                    if kind == "observed"
                    else {}
                )
                for species, y_values in values.items():
                    x_values = axes.get(species, ())
                    for index, (x_value, y_value) in enumerate(
                        zip(x_values, y_values, strict=True)
                    ):
                        segment_id = (
                            segment_ids[species][index]
                            if species in segment_ids
                            else ""
                        )
                        discontinuity = bool(
                            kind == "model"
                            and index > 0
                            and segment_ids[species][index]
                            != segment_ids[species][index - 1]
                        )
                        output.write("\n")
                        output.write(
                            "\t".join(
                                (
                                    dataset.display_name,
                                    kind,
                                    species,
                                    raw_ids.get(species, ""),
                                    (
                                        str(source_rows[species][index])
                                        if species in source_rows
                                        else ""
                                    ),
                                    str(segment_id),
                                    "1" if discontinuity else "0",
                                    dataset.x_name,
                                    f"{float(x_value):.17g}",
                                    f"{float(y_value):.17g}",
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
            self._copy_timer.stop()
            return
        try:
            output.write(next(chunks))
        except StopIteration:
            self._copy_timer.stop()
            text = output.getvalue()
            self._copy_chunks = None
            self._copy_output = None
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
            self._copy_timer.stop()
            self._copy_chunks = None
            self._copy_output = None
            QtWidgets.QMessageBox.critical(
                self,
                "Copy Fit Data",
                str(exc),
            )

    def _redraw(self) -> None:
        if self._canvas is None or self._pg is None:
            if self._message is not None:
                self._message.setText(
                    "No Fit curves yet."
                    if not self._curves
                    else "Plotting is unavailable; Fit values and Apply remain usable."
                )
            return
        self._canvas.clear()
        if not self._curves:
            plot = self._canvas.addPlot(row=0, col=0, title="No Fit curves yet")
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            return
        columns = max(1, int(math.ceil(math.sqrt(len(self._curves)))))
        for index, dataset in enumerate(self._curves):
            row, column = divmod(index, columns)
            plot = self._canvas.addPlot(
                row=row,
                col=column,
                title=dataset.display_name,
            )
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel(
                "bottom",
                "Time" if dataset.x_name == "t" else dataset.x_name,
            )
            plot.setLabel("left", "Value")
            model_x = dict(dataset.model_x)
            observed_x = dict(dataset.observed_x)
            displayed_series_count = (
                len(dataset.model_y) + len(dataset.observed_y)
            )
            if displayed_series_count > 1:
                plot.addLegend()
            model_segments = dict(dataset.model_segment_ids)
            for species, values in dataset.model_y:
                color = _color_value(self._color_resolver(species))
                x_values = model_x.get(species, ())
                segment_values = model_segments[species]
                start = 0
                for index in range(1, len(values) + 1):
                    if (
                        index < len(values)
                        and segment_values[index] == segment_values[start]
                    ):
                        continue
                    plot.plot(
                        list(x_values[start:index]),
                        list(values[start:index]),
                        pen=self._pg.mkPen(color=color, width=2),
                        name=(
                            f"{species} — model"
                            if start == 0
                            else None
                        ),
                    )
                    start = index
            for species, values in dataset.observed_y:
                color = _color_value(self._color_resolver(species))
                plot.plot(
                    list(observed_x.get(species, ())),
                    list(values),
                    pen=None,
                    symbol="o",
                    symbolSize=6,
                    symbolBrush=self._pg.mkBrush(color),
                    symbolPen=self._pg.mkPen(color=color),
                    name=f"{species} — observed",
                )
