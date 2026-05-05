from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


class _DummyClipboard:
    def __init__(self) -> None:
        self.last_text = ""

    def setText(self, text: str, *_args, **_kwargs) -> None:  # noqa: N802 - Qt-style
        self.last_text = str(text)

    def text(self) -> str:
        return str(self.last_text)


def _make_window():
    from kindred.gui.fitting.window import FittingWindow
    from kindred.core.simulation_preparation import PreparedSimulationMetadata

    t = np.linspace(0.0, 1.0, 6)
    y_a = np.linspace(1.0, 0.5, t.size)
    y_b = np.linspace(0.2, 0.9, t.size)

    mechanism_text = "rxn: A -> B; k1=0.2"
    reactions_text = "rxn: A -> B; k1=0.2"

    def simulation_func(_params):
        return {"t": t.copy(), "species": {"A": y_a.copy(), "B": y_b.copy()}}

    prepared_meta = PreparedSimulationMetadata(
        version=1,
        mechanism_text_sha256=hashlib.sha256(mechanism_text.encode("utf-8")).hexdigest(),
        mechanism_text_len=len(mechanism_text),
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        temperature_K=298.15,
        solver_requested="BDF",
        solver_normalized="BDF",
        solver_warning=None,
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        wegscheider_cyclicity_enabled=False,
        initial_prefix="init:",
    )
    simulation_func._kindred_prepared_simulation_meta = prepared_meta  # type: ignore[attr-defined]

    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t.copy(),
            "species_data": {"A": y_a.copy(), "B": y_b.copy()},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]
    dataset_payloads = [
        {
            "id": "ds1",
            "t": t.copy(),
            "y": np.vstack([y_a.copy()]),
            "species": ["A"],
        }
    ]
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=simulation_func,
        mechanism_text_getter=lambda: mechanism_text,
        reactions_text_getter=lambda: reactions_text,
        dataset_params={"ds1": {"init:A": 1.0}},
        dataset_variable_params={
            "ds1": {"init:B": {"initial": 0.2, "min": 0.0, "max": 10.0, "log10": False}}
        },
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0},
    )


def test_copy_buttons_disabled_before_any_run(qt_app):
    """Before any fit run, the Run Stamp footer button is disabled."""
    from PySide6 import QtWidgets

    window = _make_window()
    try:
        btn = window.findChild(QtWidgets.QPushButton, "global_fit_results_summary_footer_button")
        assert btn is not None
        assert btn.isEnabled() is False
    finally:
        window.close()
        qt_app.processEvents()


def test_copy_short_and_json_stamp_updates_clipboard(qt_app, monkeypatch):
    from PySide6 import QtCore, QtWidgets

    import kindred.gui.fitting.run_results_tab as run_results_tab_mod

    dummy_clipboard = _DummyClipboard()
    monkeypatch.setattr(run_results_tab_mod, "_get_clipboard", lambda: dummy_clipboard)

    captured = {"payloads": []}

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["payloads"].append({"args": args, "kwargs": dict(kwargs)})

        def start(self):
            return

        def isRunning(self):
            return False

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    window = _make_window()
    try:
        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        window._start_fit()
        qt_app.processEvents()

        assert window._run_results_tab._last_run_stamp_short

        # Open the stamp dialog to access the copy buttons
        dialog = run_results_tab_mod.RunStampDialog(
            window._run_results_tab._last_run_stamp,
            window._run_results_tab._last_run_stamp_hash,
            window._run_results_tab._last_run_stamp_short,
            parent=window,
        )
        try:
            copy_button = dialog.findChild(QtWidgets.QPushButton, "global_fit_copy_stamp_button")
            assert copy_button is not None
            copy_button.click()
            qt_app.processEvents()
            assert dummy_clipboard.last_text == window._run_results_tab._last_run_stamp_short

            copy_json_button = dialog.findChild(QtWidgets.QPushButton, "global_fit_copy_stamp_json_button")
            assert copy_json_button is not None
            copy_json_button.click()
            qt_app.processEvents()
            parsed = json.loads(dummy_clipboard.last_text)
            assert parsed["mode"] == "global"
            assert "kindred_version" in parsed
            assert "datasets" in parsed
            assert "fit_targets_applied" in parsed
        finally:
            dialog.close()
    finally:
        window.close()
        qt_app.processEvents()
