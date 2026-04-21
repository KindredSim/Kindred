import re

import numpy as np

from kindred.gui.controllers.dataset_manager import DatasetManager
import pytest

pytestmark = pytest.mark.integration



def test_prepare_fit_job_uses_solver_settings_getter(monkeypatch):
    """
    Regression: solver/rtol/atol must flow into the objective builder for
    dataset_manager-driven fit entry points (e.g., local objective jobs).
    """
    captured = {}

    def _fake_build_fitting_objective(**kwargs):
        captured["solver"] = kwargs["solver"]
        captured["rtol"] = float(kwargs["rtol"])
        captured["atol"] = float(kwargs["atol"])
        return lambda _params: np.zeros(5, dtype=float)

    monkeypatch.setattr("kindred.core.fitting_objective.build_fitting_objective", _fake_build_fitting_objective)

    dataset = {
        "t": np.linspace(0.0, 1.0, 5),
        "species": {"A": np.ones(5, dtype=float)},
    }

    dm = DatasetManager(
        plot_tabs=None,
        dataset_resolver=lambda name: dataset if name == "ds1" else None,
        solver_settings_getter=lambda: {"solver": "Radau", "rtol": 1e-3, "atol": 1e-9},
    )

    dm.prepare_fit_job(
        config={"dataset": "ds1", "parameters": {"k1": 0.1}},
        mechanism_text="reaction: A -> B; k=0.1\ninitial: A=1.0\ninitial: B=0.0",
        state_network_text="",
        temperature_K=298.15,
    )

    assert captured == {"solver": "Radau", "rtol": 1e-3, "atol": 1e-9}


def test_prepare_fit_job_does_not_double_append_state_network(monkeypatch):
    """
    Regression: some workflows may pass a full DSL that already includes the
    '# State Network' section while also supplying state_network_text. Ensure we
    do not append the state network twice.
    """
    captured = {}

    def _fake_build_fitting_objective(**kwargs):
        captured["mechanism_text"] = kwargs["mechanism_text"]
        return lambda _params: np.zeros(5, dtype=float)

    monkeypatch.setattr("kindred.core.fitting_objective.build_fitting_objective", _fake_build_fitting_objective)

    dataset = {
        "t": np.linspace(0.0, 1.0, 5),
        "species": {"A": np.ones(5, dtype=float)},
    }

    state_network_text = "States: A,B\nedge: A->B"
    mechanism_text_with_network = "\n".join(
        [
            "reaction: A -> B; k=0.1",
            "initial: A=1.0",
            "initial: B=0.0",
            "",
            "# State Network",
            state_network_text,
        ]
    )

    dm = DatasetManager(
        plot_tabs=None,
        dataset_resolver=lambda name: dataset if name == "ds1" else None,
        solver_settings_getter=lambda: {"solver": "Radau", "rtol": 1e-6, "atol": 1e-12},
    )

    dm.prepare_fit_job(
        config={"dataset": "ds1", "parameters": {"k1": 0.1}},
        mechanism_text=mechanism_text_with_network,
        state_network_text=state_network_text,
        temperature_K=298.15,
    )

    text = str(captured["mechanism_text"])
    headers = re.findall(r"(?im)^\s*#\s*state\s+network\b", text)
    assert len(headers) == 1
