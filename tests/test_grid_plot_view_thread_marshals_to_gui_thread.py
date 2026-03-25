from __future__ import annotations

import threading

import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


def test_grid_plot_view_set_datasets_from_worker_thread_is_marshaled_to_gui_thread(qt_app, monkeypatch) -> None:
    """
    Regression guardrail: GridPlotView must never touch Qt/PyQtGraph state off the GUI thread.

    This test calls `set_datasets()` from a non-GUI Python thread and asserts that the
    internal redraw happens on the QApplication thread (not the caller thread).
    """
    from kindred.gui.widgets.grid_plot_view import GridPlotView

    view = GridPlotView()
    try:
        gui_thread_ident = int(threading.get_ident())
        called_thread_idents: list[int] = []

        def _record_update_grid():
            called_thread_idents.append(int(threading.get_ident()))

        # Avoid exercising PyQtGraph/Qt painting in this test; we only care about thread marshaling.
        monkeypatch.setattr(view, "_update_species_selector", lambda: None)
        monkeypatch.setattr(view, "_update_grid", _record_update_grid)

        datasets = [
            {
                "name": "ds",
                "data_x": np.linspace(0.0, 1.0, 5, dtype=float),
                "data_y": np.linspace(0.0, 1.0, 5, dtype=float),
                "all_species": {"A": np.linspace(0.0, 1.0, 5, dtype=float)},
                "current_species": "A",
            }
        ]

        exc: list[BaseException] = []

        def _call_from_thread():
            try:
                view.set_datasets(datasets)
            except BaseException as e:  # pragma: no cover - defensive
                exc.append(e)

        t = threading.Thread(target=_call_from_thread, daemon=True)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), "Worker thread did not finish."
        if exc:
            raise exc[0]

        # Allow any queued GUI-thread work to run.
        for _ in range(50):
            qt_app.processEvents()

        assert called_thread_idents, "Expected GridPlotView to schedule a redraw."
        assert called_thread_idents[-1] == gui_thread_ident
    finally:
        view.close()
        qt_app.processEvents()
