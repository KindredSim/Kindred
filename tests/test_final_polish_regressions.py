import numpy as np
import pytest
from PySide6 import QtCore

import kindred.core.fitting_objective as fitting_module


@pytest.mark.unit
def test_build_fitting_objective_does_not_swallow_unexpected_exception(monkeypatch) -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(fitting_module, "prepare_fitting_objective_context", _boom)

    t_exp = np.asarray([0.0, 1.0], dtype=float)
    y_exp = np.asarray([0.0, 1.0], dtype=float)

    with pytest.raises(RuntimeError, match="boom"):
        fitting_module.build_fitting_objective(
            mechanism_text="A -> B",
            param_names=["k1"],
            t_exp=t_exp,
            y_exp=y_exp,
            target_species="A",
        )


@pytest.mark.gui
def test_simulation_progress_does_not_force_process_events(main_window, monkeypatch) -> None:
    calls = {"n": 0}

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return None

    monkeypatch.setattr(QtCore.QCoreApplication, "processEvents", _counting)

    main_window.simulation_controller.on_simulation_progress(10, "hello")

    assert calls["n"] == 0
