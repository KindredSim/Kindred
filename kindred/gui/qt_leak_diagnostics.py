from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

from PySide6 import QtCore, QtWidgets

logger = logging.getLogger(__name__)


def qt_leak_diagnostics_enabled() -> bool:
    return str(os.environ.get("KINDRED_DEBUG_QT_LEAKS") or "").strip() == "1"


def maybe_log_qt_leak_snapshot(
    root: QtWidgets.QWidget,
    *,
    milestone: str,
    tables: Optional[Iterable[QtWidgets.QWidget]] = None,
) -> None:
    """
    Log a bounded snapshot of Qt widget counts to help detect GUI table leaks.

    Gated by KINDRED_DEBUG_QT_LEAKS=1 and logs at most once per (root, milestone).
    """
    if not qt_leak_diagnostics_enabled():
        return

    try:
        seen = getattr(root, "_qt_leak_diag_seen", None)
    except Exception:
        return
    if not isinstance(seen, set):
        seen = set()
        try:
            setattr(root, "_qt_leak_diag_seen", seen)
        except Exception:
            return
    if str(milestone) in seen:
        return
    seen.add(str(milestone))

    try:
        widgets_total = len(root.findChildren(QtWidgets.QWidget))
        combos_total = len(root.findChildren(QtWidgets.QComboBox))
    except Exception:
        return

    table_summaries: list[str] = []
    for tbl in list(tables or []):
        try:
            name = tbl.objectName() or tbl.__class__.__name__
        except Exception:
            name = tbl.__class__.__name__

        if isinstance(tbl, QtWidgets.QTableWidget):
            try:
                table_summaries.append(f"{name}(rows={tbl.rowCount()} cols={tbl.columnCount()})")
            except Exception:
                table_summaries.append(f"{name}(QTableWidget)")
        elif isinstance(tbl, QtWidgets.QTableView):
            try:
                model = tbl.model()
                rows = int(model.rowCount()) if model is not None else -1
                cols = int(model.columnCount()) if model is not None else -1
                table_summaries.append(f"{name}(rows={rows} cols={cols})")
            except Exception:
                table_summaries.append(f"{name}(QTableView)")
        else:
            table_summaries.append(f"{name}({tbl.__class__.__name__})")

    logger.info(
        "QT leak diag (%s): widgets=%s combobox=%s tables=%s",
        str(milestone),
        int(widgets_total),
        int(combos_total),
        ", ".join(table_summaries) if table_summaries else "(none)",
    )


def schedule_qt_leak_snapshot_after_event_cycles(
    root: QtWidgets.QWidget,
    *,
    milestone: str,
    cycles: int,
    tables: Optional[Iterable[QtWidgets.QWidget]] = None,
) -> None:
    """
    Schedule a one-time snapshot after a bounded number of event-loop cycles.

    This is intended for diagnosing leaks that appear during idle processing.
    """
    if not qt_leak_diagnostics_enabled():
        return
    try:
        key = f"_qt_leak_cycles_scheduled__{milestone}"
        if bool(getattr(root, key, False)):
            return
        setattr(root, key, True)
    except Exception:
        return

    remaining = int(cycles)

    def _tick() -> None:
        nonlocal remaining
        if remaining <= 0:
            maybe_log_qt_leak_snapshot(root, milestone=milestone, tables=tables)
            return
        remaining -= 1
        try:
            if isinstance(root, QtWidgets.QWidget) and not root.isVisible():
                return
        except Exception:
            return
        QtCore.QTimer.singleShot(0, _tick)

    QtCore.QTimer.singleShot(0, _tick)

