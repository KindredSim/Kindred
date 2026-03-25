from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6 import QtWidgets

from kindred.core.simulator import solvers
from kindred.core.exceptions import SolverError, create_solver_error


def test_solver_failure_raises_solvererror(monkeypatch):
    """Solver should raise SolverError after exhausting implicit alternatives."""

    class DummySolution:
        def __init__(self):
            self.success = False
            self.message = "forced failure"
            self.t = np.array([0.0, 0.5])

    call_count = {"n": 0}

    def _fail(*args, **kwargs):
        call_count["n"] += 1
        return DummySolution()

    monkeypatch.setattr(solvers, "_solve_ivp", _fail)

    req = solvers.SimulationRequest(
        rhs=lambda t, y: -y,
        t_span=(0.0, 1.0),
        y0=np.array([1.0]),
        solver="BDF",
        grid={"N": 4},
    )

    with pytest.raises(SolverError) as exc_info:
        solvers.solve_ode(req)

    # Should attempt primary + alternatives and include context in the error
    assert call_count["n"] == 2  # BDF + Radau
    message = str(exc_info.value)
    assert "Solver 'BDF'" in message
    assert "attempted methods: BDF, Radau" in message
    assert "forced failure" in message


def test_solver_records_successful_fallback(monkeypatch):
    """When the primary SciPy method fails but a fallback succeeds, record provenance."""

    class DummySolution:
        def __init__(self, success: bool, message: str, t_eval):
            self.success = success
            self.message = message
            self.t = np.asarray(t_eval, float)
            self.y = np.vstack([np.ones_like(self.t)])

    calls = []

    def _fake_solve_ivp(*args, **kwargs):
        method = kwargs.get("method")
        calls.append(method)
        t_eval = kwargs.get("t_eval")
        if t_eval is None:
            t_eval = np.array([0.0, 1.0])
        if method == "LSODA":
            return DummySolution(False, "LSODA failure", t_eval[:1])
        return DummySolution(True, "ok", t_eval)

    monkeypatch.setattr(solvers, "_solve_ivp", _fake_solve_ivp)

    req = solvers.SimulationRequest(
        rhs=lambda t, y: -y,
        t_span=(0.0, 1.0),
        y0=np.array([1.0]),
        solver="LSODA",
        grid={"N": 2},
    )

    out = solvers.solve_ode(req)

    assert calls == ["LSODA", "Radau"]
    assert out.fallback_occurred is True
    assert out.fallback_message == "LSODA failed; succeeded with Radau"
    assert out.provenance.get("solver_requested") == "LSODA"
    assert out.provenance.get("solver_used") == "Radau"
    assert "solver" not in out.provenance


@pytest.mark.gui
@pytest.mark.slow
def test_main_window_shows_solver_error(main_window, monkeypatch):
    """Main window should surface solver errors to the user."""

    mock_critical = MagicMock(return_value=QtWidgets.QMessageBox.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", mock_critical)

    mock_plot = MagicMock()
    mock_plot._visible = True
    mock_plot._series = {}
    mock_plot.visible_series.return_value = []
    mock_plot.set_series_visible.return_value = None
    mock_plot.update_statistics.return_value = None
    mock_plot.overlay_snapshot = None

    mock_viewport = MagicMock()
    mock_table = MagicMock()
    mock_table.viewport.return_value = mock_viewport
    mock_plot.stats_table.return_value = mock_table

    main_window.set_data = lambda *args, **kwargs: None
    main_window._plot_tabs = MagicMock()
    main_window._plot_tabs._main_plot = mock_plot

    error = create_solver_error(
        "BDF",
        0.5,
        "forced failure; attempted methods: BDF, Radau",
    )

    main_window.simulation_controller.on_simulation_error(str(error))

    assert mock_critical.called
    error_text = mock_critical.call_args[0][2]
    assert "solver" in error_text.lower()
    assert "bdf" in error_text.lower()
    assert "attempted methods" in error_text.lower()
