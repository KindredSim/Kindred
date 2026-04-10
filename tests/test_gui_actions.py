from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np
from PySide6 import QtCore, QtWidgets

from kindred.gui.main_window import MainWindow
from kindred.gui.plot_config import is_pyqtgraph_available
from tests.worker_stubs import ImmediateWorker

pytestmark = pytest.mark.gui


def _capture_messagebox(monkeypatch, kind: str):
    """Return a list capturing text passed to QMessageBox."""
    seen: list[str] = []

    def _collector(_parent, _title, message):
        seen.append(message)
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, kind, _collector)
    return seen


def _load_minimal_dataset(window: MainWindow):
    """Register a deterministic dataset for fitting tests."""
    t = np.linspace(0.0, 4.0, 5)
    series = {"A": np.linspace(1.0, 0.2, 5)}
    payload = {"t": t, "species": series}
    window._on_dataset_loaded("exp.csv", payload)
    window._right_panel._data_manager._datasets["exp.csv"] = payload


def test_run_simulation_requires_mechanism(main_window: MainWindow, monkeypatch):
    """Run Simulation action must block early when the mechanism is invalid."""
    main_window._mechanism_editor._reactions_text.setPlainText("")
    warnings = _capture_messagebox(monkeypatch, "warning")
    main_window.simulation_controller.run_simulation()
    assert warnings == []
    assert main_window._status_label.text() == "Cannot run: mechanism has errors. Fix and try again."
    assert main_window.simulation_controller.run_state.simulation_worker is None


def test_run_simulation_uses_worker_stub(main_window: MainWindow, monkeypatch):
    """Happy-path simulation hooks should update provenance when worker succeeds."""
    monkeypatch.setattr(
        "kindred.gui.simulation_worker.SimulationWorker",
        ImmediateWorker,
    )
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=0.5\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window.simulation_controller.run_simulation()
    assert "species_names" in main_window._last_simulation_provenance
    assert "Simulation complete" in main_window._status_label.text()


def test_handle_export_config_writes_csv(main_window: MainWindow, tmp_path: Path):
    """Export handler should write CSV rows for the current simulation."""
    class _Plot:
        def export_payload(self):
            return {"t": np.array([0.0, 1.0]), "series": {"A": np.array([1.0, 0.5])}}

    main_window._plot_tabs.get_current_plot = lambda: _Plot()
    path = tmp_path / "out.csv"
    config = {"path": str(path), "mode": "legacy", "scope": "all"}
    main_window.project_controller.handle_export_config(config)

    contents = path.read_text().strip().splitlines()
    assert contents[0] == "t,[A]"
    assert len(contents) == 3


def test_handle_export_config_warns_on_mismatch(main_window: MainWindow, monkeypatch):
    """Export handler should warn when series lengths mismatch the time axis."""
    class _BrokenPlot:
        def export_payload(self):
            return {"t": np.array([0.0, 1.0]), "series": {"A": np.array([1.0])}}

    captured = _capture_messagebox(monkeypatch, "warning")
    main_window._plot_tabs.get_current_plot = lambda: _BrokenPlot()
    config = {"path": "unused.csv", "mode": "legacy", "scope": "all"}
    main_window.project_controller.handle_export_config(config)
    assert captured and "length" in captured[0]


def test_open_docs_opens_browser_when_configured(main_window: MainWindow, monkeypatch):
    """Help → Documentation should open the browser when a URL is configured."""
    calls = []
    monkeypatch.setattr("kindred.gui.main_window.DOCUMENTATION_URL", "https://example.com")
    monkeypatch.setattr("webbrowser.open", lambda url: calls.append(url))
    main_window._open_docs()
    assert calls == ["https://example.com"]
    assert "Documentation opened" in main_window._status_label.text()


def test_open_docs_falls_back_without_url(main_window: MainWindow, monkeypatch):
    """When no documentation URL exists, a friendly information dialog is shown."""
    monkeypatch.setattr("kindred.gui.main_window.DOCUMENTATION_URL", None)
    infos = _capture_messagebox(monkeypatch, "information")
    main_window._open_docs()
    assert infos and "online documentation" in infos[0].lower()
    assert "tutorial" not in infos[0].lower()


def test_open_docs_browser_failure_does_not_mention_tutorials(main_window: MainWindow, monkeypatch):
    """When the browser fails to open, the fallback message must not reference hidden Tutorials."""
    monkeypatch.setattr("kindred.gui.main_window.DOCUMENTATION_URL", "https://example.com")
    monkeypatch.setattr("webbrowser.open", lambda url: (_ for _ in ()).throw(OSError("simulated")))
    infos = _capture_messagebox(monkeypatch, "information")
    main_window._open_docs()
    assert infos
    assert "could not be opened" in infos[0].lower()
    assert "tutorial" not in infos[0].lower()


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_axis_toolbar_options_update_state(main_window: MainWindow):
    """Axis toolbar options should update sampling and export scope preferences."""
    plot = main_window._plot_tabs._main_plot
    t = np.linspace(0.0, 10.0, 2000)
    series = {"A": np.sin(t), "B": np.cos(t)}
    plot.set_data(t, series)

    plot._on_toolbar_option_requested("sampling", "coarse")
    assert plot._sampling_mode == "coarse"
    x_data, _ = plot._plot_items["A"].getData()
    assert len(x_data) <= plot._sampling_target

    plot._on_toolbar_option_requested("export_scope", "all")
    assert plot.get_export_scope_preference() == "all"
    plot._on_toolbar_option_requested("export_scope", "visible")
    assert plot.get_export_scope_preference() == "axis"


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_axis_toolbar_add_guide(main_window: MainWindow, monkeypatch):
    """Adding a guide from the toolbar should create a guide line."""
    plot = main_window._plot_tabs._main_plot
    t = np.linspace(0.0, 1.0, 5)
    plot.set_data(t, {"A": np.linspace(0.0, 1.0, 5)})
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getDouble",
        lambda *args, **kwargs: (1.23, True),
    )
    start_count = len(plot._guide_items)
    plot._on_add_guide_requested(None)
    assert len(plot._guide_items) == start_count + 1


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_axis_toolbar_scalar_guide_selection(main_window: MainWindow, monkeypatch):
    """Scalar guides should use algebra scalar values when available."""
    plot = main_window._plot_tabs._main_plot
    t = np.linspace(0.0, 1.0, 5)
    plot.set_data(t, {"A": np.linspace(0.0, 1.0, 5)})
    plot.set_scalar_values({"k_eq": 1.23})

    display = "k_eq = 1.23"
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *args, **kwargs: (display, True),
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("getDouble should not be called when scalar is selected")

    monkeypatch.setattr(QtWidgets.QInputDialog, "getDouble", _fail)

    plot._on_add_guide_requested(None)
    assert plot._guide_items
    assert abs(plot._guide_items[-1].value() - 1.23) < 1e-6

    y_list = plot._toolbar._y_list
    scalar_item = None
    for i in range(y_list.count()):
        item = y_list.item(i)
        if item.text() == "k_eq":
            scalar_item = item
            break
    assert scalar_item is not None
    assert not (scalar_item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled)
