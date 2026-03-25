from __future__ import annotations

import numpy as np
import pytest

from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
from kindred.core.analysis.global_fitting import (
    _GlobalFitObjective,
    _normalize_weights,
)
from kindred.core.exceptions import FitSimulationError


pytestmark = [pytest.mark.unit]


def _make_payload(dataset_id: str, *, point_count: int = 3) -> FitDatasetSpec:
    t_exp = np.linspace(0.0, 1.0, point_count, dtype=float)
    return FitDatasetSpec(
        dataset_id=dataset_id,
        t_exp=t_exp,
        species_list=["A"],
        y_matrix=np.zeros((1, point_count), dtype=float),
        point_count=point_count,
        x_name="t",
        x_obs=None,
        x_mode="auto",
    )


def test_normalize_weights_rejects_unknown_dataset_ids() -> None:
    payloads = [_make_payload("ds1"), _make_payload("ds2")]

    with pytest.raises(ValueError, match="ghost"):
        _normalize_weights(payloads, {"ds1": 2.0, "ghost": 1.0})


def test_failed_param_snapshot_namespaces_dataset_specific_values() -> None:
    snapshot = _GlobalFitObjective._build_failed_param_snapshot(
        ds_id="ds2",
        shared_params={"k": 0.5},
        full_params={"k": 0.5, "init:A": 2.25, "offset": 7.0},
    )

    assert snapshot == {
        "k": pytest.approx(0.5),
        "ds2::init:A": pytest.approx(2.25),
        "ds2::offset": pytest.approx(7.0),
    }


def test_failed_result_preserves_namespaced_dataset_variable_params(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting

    def _fake_fit_parameters(*_args, **_kwargs):
        raise FitSimulationError(
            "fatal failure",
            failed_params={
                "k": 0.75,
                "ds1::init:A": 1.25,
                "ds2::init:A": 2.5,
            },
        )

    monkeypatch.setattr(global_fitting, "fit_parameters", _fake_fit_parameters)

    result = global_fitting.fit_global(
        lambda _params: {"t": np.asarray([0.0, 1.0], dtype=float), "A": np.asarray([1.0, 1.0], dtype=float)},
        datasets=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0], dtype=float), "y": np.asarray([1.0, 0.8], dtype=float), "species": "A"},
            {"id": "ds2", "t": np.asarray([0.0, 1.0], dtype=float), "y": np.asarray([1.0, 0.6], dtype=float), "species": "A"},
        ],
        shared_params={"k": 0.2},
        dataset_variable_params={
            "ds1": {"init:A": {"initial": 0.3, "min": 0.1, "max": 3.0}},
            "ds2": {"init:A": {"initial": 0.4, "min": 0.1, "max": 4.0}},
        },
        max_nfev=1,
    )

    assert result.success is False
    assert result.shared_params["k"] == pytest.approx(0.75)
    assert result.dataset_params["ds1"]["init:A"] == pytest.approx(1.25)
    assert result.dataset_params["ds2"]["init:A"] == pytest.approx(2.5)


def test_global_chi_squared_uses_full_objective_residuals_for_parametric_x_penalties(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    t_obs = np.linspace(0.0, 1.0, 6, dtype=float)
    x_obs = np.full_like(t_obs, 10.0, dtype=float)
    y_obs = np.linspace(0.0, 2.0, t_obs.size, dtype=float).reshape(1, -1)

    datasets = [
        {
            "id": "ds1",
            "t": t_obs.copy(),
            "y": y_obs.copy(),
            "species": ["Y"],
            "x_name": "X",
            "x_obs": x_obs.copy(),
            "x_mapping_mode": "auto",
        }
    ]

    t_sim = np.linspace(0.0, 1.0, 101, dtype=float)

    def simulation_func(_params):
        return {"t": t_sim.copy(), "species": {"X": t_sim.copy(), "Y": np.zeros_like(t_sim)}}

    def _fake_least_squares(fun, x0, **_kwargs):
        residuals = np.asarray(fun(np.asarray(x0, dtype=float)), dtype=float).reshape(-1)

        class _Result:
            pass

        result = _Result()
        result.x = np.asarray(x0, dtype=float)
        result.success = True
        result.message = "fake"
        result.nfev = 1
        result.fun = residuals
        result.jac = np.zeros((residuals.size, result.x.size), dtype=float)
        return result

    monkeypatch.setattr(
        fitting_optimization,
        "load_scipy_optimize",
        lambda: (_fake_least_squares, lambda *_a, **_k: None),
    )

    result = global_fitting.fit_global(
        simulation_func,
        datasets=datasets,
        shared_params={"k1": 0.2},
        weights=None,
        method="trf",
        max_nfev=1,
    )

    residuals = np.asarray(result.objective_residuals, dtype=float).reshape(-1)
    y_flat = y_obs.reshape(-1)
    expected_chi_squared = float(np.mean(residuals**2))
    expected_r_squared = 1.0 - float(np.sum(residuals**2)) / float(np.sum((y_flat - float(np.mean(y_flat))) ** 2))

    assert residuals.size == 2 * t_obs.size
    assert result.global_chi_squared == pytest.approx(expected_chi_squared)
    assert result.global_r_squared == pytest.approx(expected_r_squared)
