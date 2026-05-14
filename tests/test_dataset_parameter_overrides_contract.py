from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_fit_global_accepts_typed_dataset_overrides() -> None:
    from kindred.core.analysis.dataset_parameter_overrides import (
        FitDatasetParameterOverrides,
        FitDatasetVariableParamSpec,
    )
    from kindred.core.analysis.global_fitting import fit_global

    def simulate(params):
        k = params["k"]
        init_a = params.get("init:A", 1.0)
        t = np.linspace(0.0, 5.0, 40)
        return {"t": t, "A": init_a * np.exp(-k * t)}

    t = np.linspace(0.0, 5.0, 40)
    datasets = [
        {"id": "ds1", "t": t, "y": 1.0 * np.exp(-0.4 * t), "species": "A"},
        {"id": "ds2", "t": t, "y": 2.0 * np.exp(-0.4 * t), "species": "A"},
    ]
    overrides = [
        FitDatasetParameterOverrides(
            dataset_id="ds1",
            variable_params={"init:A": FitDatasetVariableParamSpec(initial=0.5, minimum=0.1, maximum=3.0)},
        ),
        FitDatasetParameterOverrides(
            dataset_id="ds2",
            variable_params={"init:A": FitDatasetVariableParamSpec(initial=0.5, minimum=0.1, maximum=5.0)},
        ),
    ]

    result = fit_global(
        simulate,
        datasets,
        {"k": 0.2},
        dataset_overrides=overrides,
    )

    assert result.completion.status == "ok"
    assert pytest.approx(result.shared_params["k"], rel=1e-2) == 0.4
    assert pytest.approx(result.dataset_params["ds1"]["init:A"], rel=1e-2) == 1.0
    assert pytest.approx(result.dataset_params["ds2"]["init:A"], rel=1e-2) == 2.0


@pytest.mark.unit
def test_run_stamp_accepts_typed_dataset_overrides() -> None:
    from kindred.core.analysis.dataset_parameter_overrides import (
        FitDatasetParameterOverrides,
        FitDatasetVariableParamSpec,
    )
    from kindred.gui.fitting.run_stamp import build_global_fit_run_stamp

    stamp = build_global_fit_run_stamp(
        dataset_rows=[{"id": "ds1", "label": "ds1", "include": True, "weight": 1.0}],
        included_ids=["ds1"],
        applied_fit_targets={"ds1": ["A"]},
        weights_used={"ds1": 1.0},
        weight_mode="custom",
        fit_config={
            "parameters": {"k1": 0.2},
            "fixed_params": {},
            "bounds": {"k1": (0.01, 1.0)},
            "log10_params": {"k1": False},
            "method": "trf",
            "max_nfev": 10,
            "seed": 7,
            "parallel_starts": 1,
        },
        mechanism_text="rxn: A -> B; k1=0.2",
        reactions_text="rxn: A -> B; k1=0.2",
        dataset_overrides=[
            FitDatasetParameterOverrides(
                dataset_id="ds1",
                fixed_params={"init:A": 1.0},
                variable_params={
                    "init:B": FitDatasetVariableParamSpec(
                        initial=0.2,
                        minimum=0.0,
                        maximum=10.0,
                        log10=False,
                    )
                },
            )
        ],
    )

    assert stamp["dataset_params"]["ds1"]["init:A"] == "1"
    assert stamp["dataset_variable_params"]["ds1"]["init:B"]["max"] == "10"


@pytest.mark.gui
def test_fitting_window_passes_typed_dataset_overrides_to_worker(qt_app, monkeypatch) -> None:
    from PySide6 import QtCore

    from kindred.core.analysis.dataset_parameter_overrides import FitDatasetParameterOverrides
    from kindred.core.simulation_preparation import PreparedSimulationMetadata
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

    def simulation_func(_params):
        return {"t": t.copy(), "species": {"A": np.ones_like(t)}}

    simulation_func._kindred_prepared_simulation_meta = PreparedSimulationMetadata(  # type: ignore[attr-defined]
        version=1,
        mechanism_text_sha256="abc",
        mechanism_text_len=3,
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
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

    captured: dict[str, object] = {}

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        bestUpdated = QtCore.Signal(dict)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["dataset_overrides"] = kwargs.get("dataset_overrides")

        def start(self):
            return

        def isRunning(self):
            return False

        def cancel(self):
            return

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _FakeWorker)

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=simulation_func,
        mechanism_text_getter=lambda: "rxn: A -> B; k1=0.2",
        reactions_text_getter=lambda: "rxn: A -> B; k1=0.2",
        dataset_params={"ds1": {"init:A": 1.0}},
        dataset_variable_params={
            "ds1": {"init:B": {"initial": 0.2, "min": 0.0, "max": 10.0, "log10": False}}
        },
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0},
    )
    try:
        config = window._params_ics_tab.collect_parameter_config()
        assert config is not None
        window.run_fit()

        overrides = captured.get("dataset_overrides")
        assert isinstance(overrides, list)
        assert len(overrides) == 1
        assert isinstance(overrides[0], FitDatasetParameterOverrides)
        assert overrides[0].dataset_id == "ds1"
        assert overrides[0].fixed_params["init:A"] == pytest.approx(1.0)
        assert overrides[0].variable_params["init:B"].maximum == pytest.approx(10.0)
    finally:
        window.close()
        qt_app.processEvents()
