"""The PyQtGraph data item every Kindred result plot draws with."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import pyqtgraph as pg

__all__ = ["PlotDataItem"]



class PlotDataItem(pg.PlotDataItem):
    """Mark lone displayed samples of a line, and treat scatter axes with no
    finite mapped samples as having no bounds."""

    # Samples of instantaneous jump paths, which never get a lone-sample mark.
    _connectors: bool | Sequence[bool] = False

    def setData(self, *args, connectors: bool | Sequence[bool] = False, **kwargs) -> None:
        self._connectors = connectors
        super().setData(*args, **kwargs)

    def updateItems(self, styleUpdate=True):
        super().updateItems(styleUpdate)
        if self.opts["symbol"] is not None or self.opts["pen"] is None:
            return
        # A displayed sample with no defined neighbour has no segment to draw;
        # judged after axis mapping, as a log axis leaves nonpositive values out.
        x, y = self.getData()
        if x is None or not len(x):
            return
        known = np.isfinite(x) & np.isfinite(y)
        lone = (known & ~np.r_[False, known[:-1]] & ~np.r_[known[1:], False]
                & ~np.asarray(self._connectors, dtype=bool))
        if lone.any():
            color = pg.mkPen(self.opts["pen"]).color()
            self.scatter.setData(
                x=x[lone], y=y[lone], symbol="o", size=5, pen=pg.mkPen(color), brush=pg.mkBrush(color),
            )
            self.scatter.show()

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if not self.curve.isVisible() and self.scatter.isVisible():
            # 0.13.7 checks for empty scatter data before, but not after,
            # filtering nonfinite values for percentiles. Log mapping can
            # remove every finite value, including within a visible-X window.
            data = self.scatter.data
            values = data["x" if ax == 0 else "y"]
            if orthoRange is not None:
                other = data["y" if ax == 0 else "x"]
                values = values[(other >= orthoRange[0]) & (other <= orthoRange[1])]
            if not np.isfinite(values).any():
                return (None, None)
        return super().dataBounds(ax, frac=frac, orthoRange=orthoRange)
