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
    assert sliders.has_variable("Keq1")

    created: list[dict] = []
    full_run_holder: dict[str, object] = {"worker": None}
    full_run_points = int(main_window._num_points_spinbox.value())

    class _ControlledWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        result_ready = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(
            self,
            *,
            owner,
            simulation_plan_payload,
            include_mechanism_in_result_payload=True,
            parent=None,
        ):
            super().__init__(parent)
            from kindred.core.simulation_plan import SimulationPlan

            _ = owner
            request = (
                SimulationPlan.from_payload(dict(simulation_plan_payload or {}))
                .to_execution_request()
                .to_payload()
            )
            self._running = False
            self._mechanism_text = str(request.get("mechanism_text") or "")
            self._solver_config = dict(request.get("solver_config") or {})
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

    monkeypatch.setattr("kindred.gui.simulation_worker.ContainedSimulationWorker", _ControlledWorker)

    # Simulate a Keq1 drag+release to arm the slider-release commit timer.
    preview = main_window._preview_session
    main_window._on_slider_drag_started("Keq1")
    slider_widget = sliders._sliders["Keq1"]
    slider_widget.setValue(min(slider_widget.maximum(), slider_widget.value() + 50))
    qtbot.waitUntil(lambda: "Keq1" in preview._pending_slider_values, timeout=1000)
    main_window._on_slider_drag_finished("Keq1")

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


def test_run_submits_intervention_schedule_plan_and_surfaces_scheduled_result(main_window, qtbot, monkeypatch):
    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=set; species=A; time=0.0; value=3.0",
        ]
    )
    main_window._mechanism_editor._reactions_text.setPlainText(mechanism_text)
    main_window._extract_and_populate_variables()

    captured: dict[str, object] = {}

    class _ImmediateWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        result_ready = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(
            self,
            *,
            owner,
            simulation_plan_payload,
            include_mechanism_in_result_payload=True,
            parent=None,
        ):
            super().__init__(parent)
            from kindred.core.simulation_plan import SimulationPlan

            _ = owner
            _ = include_mechanism_in_result_payload
            self._running = False
            self._request_payload = (
                SimulationPlan.from_payload(dict(simulation_plan_payload or {}))
                .to_execution_request()
                .to_payload()
            )
            captured["request_payload"] = dict(self._request_payload)

        def start(self):
            from kindred.core.simulation_preparation import prepare_simulation_worker_run
            from kindred.core.simulation_result_payload import build_simulation_success_payload
            from kindred.core.simulator.solvers import solve_ode

            self._running = True
            prepared = prepare_simulation_worker_run(execution_request=self._request_payload)
            result = solve_ode(prepared.request)
            payload = build_simulation_success_payload(
                result=result,
                y=result.Y,
                species_names=prepared.species_names,
                base_species_count=len(prepared.species_names),
                algebra_scalars={},
                algebra_errors=[],
                warnings=[],
                solver=str(prepared.request.solver),
                mechanism=prepared.mechanism,
                mechanism_text=str(self._request_payload.get("mechanism_text") or ""),
                solver_config=dict(self._request_payload.get("solver_config") or {}),
                extra_fields={},
            )
            captured["payload"] = payload
            self.result_ready.emit(payload)
            self._running = False

        def cancel(self):
            self._running = False

        def isRunning(self):
            return self._running

        def wait(self, *_args, **_kwargs):
            return True

        def terminate(self):
            self._running = False

    monkeypatch.setattr("kindred.gui.simulation_worker.ContainedSimulationWorker", _ImmediateWorker)

    main_window.simulation_controller.run_simulation()
    qtbot.waitUntil(lambda: "payload" in captured, timeout=2000)
    QtWidgets.QApplication.processEvents()

    request_payload = captured["request_payload"]
    schedule_payload = request_payload["intervention_schedule"]  # type: ignore[index]
    payload = captured["payload"]
    species_names = list(payload["species_names"])  # type: ignore[index]
    a_index = species_names.index("A")
    plot_payload = main_window.main_plot().export_payload()

    assert schedule_payload["instant_events"][0]["value"] == 3.0
    assert float(np.asarray(payload["Y"])[a_index, 0]) == pytest.approx(3.0)  # type: ignore[index]
    assert payload["provenance"]["has_intervention_schedule"] is True  # type: ignore[index]
    assert plot_payload is not None
    assert float(np.asarray(plot_payload["series"]["A"])[0]) == pytest.approx(3.0)
