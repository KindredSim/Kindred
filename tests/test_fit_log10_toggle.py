import numpy as np
import pytest

from kindred.core.analysis.global_fitting import fit_global


def test_fit_global_log10_shared_param_converts_bounds_and_values():
    true_k = 0.7

    def simulate(params):
        k = params["k"]
        t = np.linspace(0.0, 2.0, 40)
        return {"t": t, "A": np.exp(-k * t)}

    t = np.linspace(0.0, 2.0, 40)
    y = np.exp(-true_k * t)
    datasets = [{"id": "ds", "t": t, "y": y, "species": "A"}]

    result = fit_global(
        simulate,
        datasets,
        {"k": 0.2},
        bounds={"k": (0.05, 5.0)},
        log10_params={"k": True},
        max_nfev=200,
    )

    assert result.success
    assert pytest.approx(result.shared_params["k"], rel=2e-2) == true_k


def test_fit_global_log10_rejects_non_positive_bounds():
    def simulate(params):
        t = np.linspace(0.0, 1.0, 10)
        return {"t": t, "A": np.exp(-params["k"] * t)}

    t = np.linspace(0.0, 1.0, 10)
    datasets = [{"id": "ds", "t": t, "y": np.exp(-0.5 * t), "species": "A"}]

    with pytest.raises(ValueError):
        fit_global(
            simulate,
            datasets,
            {"k": 0.2},
            bounds={"k": (-1.0, 1.0)},
            log10_params={"k": True},
            max_nfev=10,
        )

