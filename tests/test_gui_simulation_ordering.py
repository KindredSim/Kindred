from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

pytestmark = [pytest.mark.gui, pytest.mark.slow]


def _payload(t_len: int, mechanism_text: str, solver_config: dict) -> dict:
    t = np.linspace(0.0, 1.0, int(t_len))
    Y = np.vstack(
        [
            np.linspace(1.0, 0.5, t.size),
            np.linspace(0.0, 0.5, t.size),
        ]
    )
    return {
        "t": t,
        "Y": Y,
        "species_names": ["A", "B"],
        "mechanism": None,
        "mechanism_text": mechanism_text,
        "solver_config": dict(solver_config or {}),
    }


def test_run_does_not_auto_run_pending_slider_simulation(main_window, qtbot, monkeypatch):
    """
    Regression: pressing Run while a slider-release commit is pending should not
    trigger an extra fast-mode (slider) simulation after the Run completes.

    Pre-fix behavior:
    - Release starts a commit timer.
    - Run starts a full simulation.
    - Commit timer fires during the run, marks `_pending_slider_simulation=True`.
    - When the run finishes, the pending slider simulation runs and overwrites the plot.
    """
    main_window._mechanism_editor._reactions_text.setPlainText(
        "equilibrium: A <-> B ; kf=0.2, K=0.5\n"
        "reaction: B -> C ; k=0.1\n"
        "init: A=1, B=0, C=0\n"
    )
    main_window._extract_and_populate_variables()

    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("K1")

    created: list[dict] = []
    full_run_holder: dict[str, object] = {"worker": None}
    full_run_points = int(main_window._num_points_spinbox.value())

    class _ControlledWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        result_ready = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(
            self,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent=None,
            prepared=None,
            include_mechanism_in_result_payload=True,
        ):
            super().__init__(parent)
            self._running = False
            self._mechanism_text = str(mechanism_text)
            self._solver_config = dict(solver_config or {})
            self._include_mechanism_in_result_payload = bool(include_mechanism_in_result_payload)
            n_points = int(self._solver_config.get("grid", {}).get("N", 2) or 2)
            self._payload = _payload(n_points, self._mechanism_text, self._solver_config)
            created.append({"n_points": n_points})
            self._auto_finish = n_points != full_run_points
            if not self._auto_finish:
                full_run_holder["worker"] = self

        def start(self):
            self._running = True
            if self._auto_finish:
                self.result_ready.emit(self._payload)
                self._running = False

        def complete(self):
            self.result_ready.emit(self._payload)
            self._running = False

        def cancel(self):
            self._running = False

        def isRunning(self):
            return self._running

        def wait(self, *_args, **_kwargs):
            return True

        def terminate(self):
            self._running = False

    monkeypatch.setattr("kindred.gui.simulation_worker.SimulationWorker", _ControlledWorker)

    # Simulate a K1 drag+release to arm the slider-release commit timer.
    preview = main_window._preview_session
    main_window._on_slider_drag_started("K1")
    slider_widget = sliders._sliders["K1"]
    slider_widget.setValue(min(slider_widget.maximum(), slider_widget.value() + 50))
    qtbot.waitUntil(lambda: "K1" in preview._pending_slider_values, timeout=1000)
    main_window._on_slider_drag_finished("K1")

    # Start a full Run while the release commit is still pending.
    main_window.simulation_controller.run_simulation()
    assert full_run_holder["worker"] is not None

    # Allow the release timer to fire during the run (so it can schedule a pending slider sim).
    qtbot.waitUntil(lambda: not preview._slider_release_commit_timer.isActive(), timeout=2000)

    # Complete the full run and flush events so any pending slider sim executes.
    full_worker = full_run_holder["worker"]
    assert hasattr(full_worker, "complete")
    full_worker.complete()  # type: ignore[union-attr]
    QtWidgets.QApplication.processEvents()
    qtbot.wait(50)
    QtWidgets.QApplication.processEvents()

    # Expected behavior: Run should be the only simulation started.
    assert len(created) == 1
