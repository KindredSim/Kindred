from __future__ import annotations

from dataclasses import replace
import importlib.resources
from types import SimpleNamespace
import numpy as np
import pytest


def _seed_one_dataset(main_window) -> None:
    data_panel = main_window._right_panel._data_manager
    data_panel._datasets.clear()
    t = np.linspace(0.0, 1.0, 6)
    data_panel._datasets["ds1"] = {
        "t": t.copy(),
        "species": {
            "A": np.linspace(1.0, 0.5, t.size),
            "B": np.linspace(0.0, 0.4, t.size),
        },
    }


def _seed_simple_mechanism(main_window) -> None:
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )


@pytest.mark.gui
def test_fitting_mixin_run_global_fit_delegates_to_launch_owner(main_window, monkeypatch):
    from kindred.gui.fitting import GlobalFitLaunchContext

    _seed_one_dataset(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    captured: dict[str, object] = {}

    def fake_launch(context):
        captured["context"] = context
        return None

    monkeypatch.setattr("kindred.gui.mixins.fitting_mixin.launch_global_fit_session", fake_launch, raising=False)

    try:
        main_window._run_global_fit()
    finally:
        for window in list(getattr(main_window, "_active_fit_windows", []) or []):
            window.close()

    context = captured.get("context")
    assert isinstance(context, GlobalFitLaunchContext)
    assert context.parent is main_window
    assert context.dataset_manager is main_window._dataset_manager
    assert callable(context.reactions_text_getter)
    assert callable(context.reactions_text_setter)


@pytest.mark.gui
def test_fitting_package_launch_owner_builds_window_payloads(main_window, monkeypatch):
    from PySide6 import QtWidgets

    from kindred.core.fitting_evaluation import SerialFittingEvaluator
    from kindred.gui.fitting import launch_global_fit_session

    _seed_one_dataset(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    captured: dict[str, object] = {}

    class _FakeWindow(QtWidgets.QDialog):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["kwargs"] = dict(kwargs)

        def setWindowTitle(self, *_args):
            return None

        def show(self):
            return None

        def raise_(self):
            return None

        def activateWindow(self):
            return None

    context = replace(main_window._build_global_fit_launch_context(), window_factory=_FakeWindow)
    window = launch_global_fit_session(context)
    assert isinstance(window, QtWidgets.QDialog)
    kwargs = captured.get("kwargs")
    assert isinstance(kwargs, dict)
    assert "simulation_func" in kwargs
    assert isinstance(kwargs["simulation_func"], SerialFittingEvaluator)
    prepared_payload = kwargs["simulation_func"].context.execution_request.to_payload()["prepared_payload"]
    assert isinstance(prepared_payload, dict)
    assert "rhs" not in prepared_payload
    assert kwargs.get("dataset_payloads")
    assert kwargs["dataset_payloads"][0]["id"] == "ds1"
    assert callable(kwargs.get("project_apply_callback"))


@pytest.mark.gui
def test_fitting_package_launch_owner_preserves_invalid_payload_results(main_window, monkeypatch):
    from PySide6 import QtWidgets

    from kindred.gui.fitting import launch_global_fit_session

    _seed_one_dataset(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    captured: dict[str, object] = {}

    class _FakeWindow(QtWidgets.QDialog):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["kwargs"] = dict(kwargs)

        def setWindowTitle(self, *_args):
            return None

        def show(self):
            return None

        def raise_(self):
            return None

        def activateWindow(self):
            return None

    monkeypatch.setattr(
        "kindred.gui.fitting.launch._coerce_dataset_payload",
        lambda **_kwargs: SimpleNamespace(
            state="invalid",
            payload=None,
            error="Dataset 'ds1' has invalid x_obs for X='X'.",
        ),
    )

    context = replace(main_window._build_global_fit_launch_context(), window_factory=_FakeWindow)
    window = launch_global_fit_session(context)
    assert isinstance(window, QtWidgets.QDialog)
    kwargs = captured.get("kwargs")
    assert isinstance(kwargs, dict)
    assert kwargs.get("dataset_payloads") == []
    payload_results = kwargs.get("dataset_payload_results")
    assert isinstance(payload_results, dict)
    assert payload_results["ds1"].state == "invalid"
    assert "invalid x_obs" in str(payload_results["ds1"].error)


def test_fitting_launch_owner_module_does_not_import_fit_dialog() -> None:
    source = importlib.resources.files("kindred.gui.fitting").joinpath("launch.py").read_text(encoding="utf-8")

    assert "from kindred.gui.fit_dialog import" not in source
    assert "import kindred.gui.fit_dialog" not in source
