import numpy as np
import pytest

from PySide6 import QtWidgets

from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult
from kindred.gui.fitting.window import FittingWindow


pytestmark = [pytest.mark.gui]


def _make_success_result(*, chi_sq: float = 1.234) -> GlobalFitResult:
    t = np.linspace(0.0, 1.0, 5)
    y = np.linspace(1.0, 0.5, t.size)
    model = np.linspace(1.0, 0.4, t.size)
    residual = model - y
    return GlobalFitResult(
        success=True,
        shared_params={"k1": 1.23},
        dataset_params={"ds1": {"init:A": 2.0}},
        uncertainties=None,
        global_chi_squared=float(chi_sq),
        global_r_squared=0.9,
        dataset_info=[
            DatasetFitInfo(
                dataset_id="ds1",
                r_squared=0.9,
                chi_squared=0.1,
                rmse=0.1,
                mae=0.1,
                residuals=residual.copy(),
                n_points=int(residual.size),
                weight=1.0,
            )
        ],
        nfev=10,
        message="ok",
        covariance=None,
        objective_residuals=residual.copy(),
        model_series={"ds1": {"A": model.copy()}},
        residual_series={"ds1": {"A": residual.copy()}},
    )


def _make_window():
    t = np.linspace(0.0, 1.0, 5)
    y = np.linspace(1.0, 0.5, t.size)
    model = np.linspace(1.0, 0.4, t.size)
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 1.23, "min": 0.01, "max": 10.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t.copy(),
                "species_data": {"A": y.copy()},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": model.copy()}},
        dataset_payloads=[{"id": "ds1", "t": t.copy(), "y": np.vstack([y.copy()]), "species": ["A"]}],
        dataset_weights={"ds1": 1.0},
    )


def test_fit_completion_shows_success_popup(qt_app, monkeypatch):
    window = _make_window()
    calls = []

    def _fake_information(parent, title, text, *args, **kwargs):
        calls.append((str(title), str(text)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "information", _fake_information)
    try:
        window.show()
        qt_app.processEvents()
        result = _make_success_result(chi_sq=3.21)
        window._handle_global_fit_complete({"result": result})
        assert calls, "Expected Optimization Complete popup on successful fit."
        assert calls[-1][0] == "Optimization Complete"
        assert "Chi" in calls[-1][1] or "χ²" in calls[-1][1]
        assert "3.21" in calls[-1][1]
    finally:
        window.close()


def test_graphs_tab_removed(qt_app):
    window = _make_window()
    try:
        titles = [window._tabs.tabText(i) for i in range(window._tabs.count())]
        assert "Graphs" not in titles
    finally:
        window.close()


def test_subset_viewer_button_removed(qt_app):
    window = _make_window()
    try:
        texts = [btn.text() for btn in window.findChildren(QtWidgets.QPushButton)]
        assert all("Subset Viewer" not in text for text in texts)
    finally:
        window.close()


def test_apply_to_project_replaces_legacy_final_apply_buttons(qt_app):
    window = _make_window()
    try:
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        assert button is not None
        assert combo is not None
        assert button.text() == "Apply to Project"
        assert [combo.itemText(i) for i in range(combo.count())] == [
            "Parameters only",
            "Initial conditions only",
            "Parameters and initial conditions",
        ]

        texts = [btn.text() for btn in window.findChildren(QtWidgets.QPushButton)]
        assert "Apply Parameters" not in texts
        assert "Apply ICs" not in texts
    finally:
        window.close()
