from __future__ import annotations

import numpy as np
import pytest


def _dataset_spec(dataset_id: str = "ds1"):
    from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec

    t = np.linspace(0.0, 1.0, 5)
    return FitDatasetSpec(
        dataset_id=dataset_id,
        t_exp=t,
        species_list=["A"],
        y_matrix=np.vstack([np.ones_like(t)]),
        point_count=int(t.size),
        x_name="t",
        x_obs=None,
        x_mode="auto",
    )


def test_fit_global_accepts_typed_dataset_specs() -> None:
    from kindred.core.analysis.global_fitting import fit_global
    from kindred.core.fitting_evaluation import CallableFittingEvaluator

    spec = _dataset_spec()

    def simulate(params):
        t = spec.t_exp
        return {"t": t, "species": {"A": np.full_like(t, float(params["k"]))}}

    result = fit_global(
        CallableFittingEvaluator(simulate),
        [spec],
        {"k": 1.0},
        max_nfev=2,
    )

    assert "ds1" in result.dataset_params


def test_global_fit_worker_accepts_typed_dataset_specs() -> None:
    from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult
    from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
    from kindred.gui.fitting.worker import GlobalFitWorker

    captured: dict[str, object] = {}
    spec = _dataset_spec()

    def fake_fit_global(_simulate, datasets, _shared_params, **_kwargs):
        captured["datasets"] = datasets
        return GlobalFitResult(
            success=True,
            shared_params={"k": 1.0},
            dataset_params={"ds1": {}},
            uncertainties=None,
            global_chi_squared=0.0,
            global_r_squared=1.0,
            dataset_info=[
                DatasetFitInfo(
                    dataset_id="ds1",
                    r_squared=1.0,
                    chi_squared=0.0,
                    rmse=0.0,
                    mae=0.0,
                    residuals=np.zeros(1, dtype=float),
                    n_points=1,
                    weight=1.0,
                )
            ],
            nfev=1,
            message="ok",
            covariance=None,
            objective_residuals=np.zeros(1, dtype=float),
            model_series={"ds1": {}},
            residual_series={"ds1": {}},
        )

    def simulation(_params):
        return {"t": spec.t_exp, "species": {"A": np.ones_like(spec.t_exp)}}

    worker = GlobalFitWorker(
        [spec],
        {"k": 1.0},
        fit_evaluator=simulation,
        fit_func=fake_fit_global,
    )

    payload = worker._execute()

    assert payload is not None
    assert isinstance(captured["datasets"], list)
    assert isinstance(captured["datasets"][0], FitDatasetSpec)


@pytest.mark.gui
def test_global_fit_worker_emits_structured_failure_payload(qtbot) -> None:
    from kindred.gui.fitting.worker import GlobalFitWorker

    spec = _dataset_spec()

    def simulation(_params):
        return {"t": spec.t_exp, "species": {"A": np.ones_like(spec.t_exp)}}

    def fake_fit_global(*_args, **_kwargs):
        raise RuntimeError("fit exploded")

    worker = GlobalFitWorker(
        [spec],
        {"k": 1.0},
        fit_evaluator=simulation,
        fit_func=fake_fit_global,
    )

    with qtbot.waitSignal(worker.error, timeout=3000) as blocker:
        worker.start()

    worker.wait(1000)
    payload = blocker.args[0]
    assert isinstance(payload, dict)
    assert payload["kind"] == "fitting_error"
    assert payload["message"] == "fit exploded"


@pytest.mark.gui
def test_fitting_window_passes_typed_dataset_specs_to_worker(qt_app, monkeypatch) -> None:
    from PySide6 import QtCore

    from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
    from kindred.gui.fitting.window import FittingWindow

    t = np.linspace(0.0, 1.0, 5)
    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t.copy(),
            "species_data": {"A": np.ones_like(t)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]
    dataset_payloads = [{"id": "ds1", "t": t.copy(), "y": np.vstack([np.ones_like(t)]), "species": ["A"]}]
    captured: dict[str, object] = {}

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(self, datasets, *args, **kwargs):
            super().__init__()
            captured["datasets"] = list(datasets)

        def start(self):
            return

        def isRunning(self):
            return False

        def cancel(self):
            return

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": np.ones_like(t)}},
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0},
    )
    try:
        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = window._collect_dataset_selection()
        window._start_global_fit(config, selection)

        datasets = captured.get("datasets")
        assert isinstance(datasets, list)
        assert isinstance(datasets[0], FitDatasetSpec)
        assert datasets[0].dataset_id == "ds1"
    finally:
        window.close()
        qt_app.processEvents()


@pytest.mark.gui
def test_fitting_window_preserves_invalid_payload_reason_during_rebuild(qt_app, monkeypatch) -> None:
    from PySide6 import QtWidgets

    from kindred.gui.fitting.window import FittingWindow

    t = np.linspace(0.0, 1.0, 5)
    captured: list[str] = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, _title, text, *args, **kwargs: captured.append(str(text)) or QtWidgets.QMessageBox.StandardButton.Ok,
    )

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t.copy(),
                "species_data": {"A": np.ones_like(t)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": np.ones_like(t)}},
        dataset_payloads=[{"id": "ds1", "t": t.copy(), "y": np.vstack([np.ones_like(t)]), "species": ["A"]}],
        dataset_weights={"ds1": 1.0},
    )
    try:
        window._dataset_entries[0]["x_name"] = "X"
        window._dataset_entries[0]["x_obs"] = object()
        window._dataset_entries[0]["x_mapping_mode"] = "time_guided"

        window._rebuild_selected_payload_lookup()

        payloads = window._datasets_payloads_for_run(["ds1"])
        assert payloads is None
        assert captured
        assert "invalid payload" in captured[-1].lower()
        assert "invalid x_obs" in captured[-1].lower()
    finally:
        window.close()
        qt_app.processEvents()


@pytest.mark.gui
def test_fitting_window_rebuilds_fit_evaluator_when_launch_deferred(qt_app, monkeypatch) -> None:
    from PySide6 import QtCore, QtWidgets

    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
    from kindred.core.fitting_process_lanes import fitting_process_lane_payload_from_evaluator
    from kindred.gui.fitting.window import FittingWindow

    t = np.linspace(0.0, 1.0, 5)
    warnings: list[str] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, _title, text, *args, **kwargs: warnings.append(str(text)) or QtWidgets.QMessageBox.StandardButton.Ok,
    )

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)
        bestUpdated = QtCore.Signal(dict)

        def __init__(self, datasets, shared_params, *, fit_evaluator=None, **kwargs):
            super().__init__()
            captured["datasets"] = list(datasets)
            captured["shared_params"] = dict(shared_params)
            captured["fit_evaluator"] = fit_evaluator

        def start(self):
            return

        def isRunning(self):
            return False

        def cancel(self):
            return

    build_calls: list[tuple[str, tuple[str, ...], str, float, float]] = []

    def _build_simulation(mechanism_text, param_names, *, solver, rtol, atol):
        build_calls.append((str(mechanism_text), tuple(param_names), str(solver), float(rtol), float(atol)))
        context = prepare_fitting_execution_context(
            mechanism_text=str(mechanism_text),
            param_names=list(param_names),
            t_end=1.0,
            num_points=5,
            solver=str(solver),
            rtol=float(rtol),
            atol=float(atol),
            initial_prefix="init:",
        )
        return SerialFittingEvaluator(context)

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t.copy(),
                "species_data": {"A": np.ones_like(t)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        simulation_func=None,
        simulation_builder=_build_simulation,
        mechanism_text_getter=lambda: mechanism_text,
        dataset_payloads=[{"id": "ds1", "t": t.copy(), "y": np.vstack([np.ones_like(t)]), "species": ["A"]}],
        dataset_weights={"ds1": 1.0},
    )
    try:
        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = window._collect_dataset_selection()
        window._start_global_fit(config, selection)

        assert build_calls
        assert warnings == []
        assert isinstance(captured.get("fit_evaluator"), SerialFittingEvaluator)
        assert type(captured["fit_evaluator"]) is SerialFittingEvaluator
        assert fitting_process_lane_payload_from_evaluator(captured["fit_evaluator"]) is not None
    finally:
        window.close()
        qt_app.processEvents()
