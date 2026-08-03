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
_FIT_COPY_COLUMNS = 14
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
    """Render immutable live and terminal Fit curves in the Fit-local plot."""

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
            for _species, values in dataset.objective_model_y
        )
        cell_count = (row_count + 1) * _FIT_COPY_COLUMNS
        if cell_count > _MAX_FIT_COPY_CELLS:
            raise ValueError(
                "Fit Copy payload exceeds the bounded full-resolution cell "
                f"limit ({_MAX_FIT_COPY_CELLS:,})."
            )
        header = (
            "dataset\tseries\traw_observation_id\traw_column\tsource_row"
            "\tsegment_id\tsegment_key\tdiscontinuity_id\tselection_provenance"
            "\tx_name\tobserved_x\tobserved_y\tmodel_x\tmodel_y"
        )
        total_characters = 0
        output = io.StringIO()
        output.write(header)
        rows_in_chunk = 0
        for dataset in self._curves:
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
                    output.write("\n")
                    output.write(
                        "\t".join(
                            (
                                dataset.display_name,
                                species,
                                observation_ids[species][index],
                                raw_columns[species][index],
                                str(source_rows[species][index]),
                                str(segment_ids[species][index]),
                                segment_keys[species][index],
                                str(discontinuity_ids[species][index]),
                                selection[species][index],
                                dataset.x_name,
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
            model_segment_keys = dict(dataset.model_segment_keys)
            model_discontinuities = dict(dataset.model_discontinuity_ids)
            for species, values in dataset.model_y:
                color = _color_value(self._color_resolver(species))
                x_values = model_x.get(species, ())
                segment_values = model_segments[species]
                segment_key_values = model_segment_keys[species]
                discontinuity_values = model_discontinuities[species]
                start = 0
                for index in range(1, len(values) + 1):
                    if (
                        index < len(values)
                        and segment_values[index] == segment_values[start]
                        and segment_key_values[index] == segment_key_values[start]
                        and discontinuity_values[index]
                        == discontinuity_values[start]
                    ):
                        continue
                    singleton = index - start == 1
                    plot.plot(
                        list(x_values[start:index]),
                        list(values[start:index]),
                        pen=self._pg.mkPen(color=color, width=2),
                        symbol="o" if singleton else None,
                        symbolSize=5 if singleton else None,
                        symbolBrush=self._pg.mkBrush(color) if singleton else None,
                        symbolPen=self._pg.mkPen(color=color) if singleton else None,
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
