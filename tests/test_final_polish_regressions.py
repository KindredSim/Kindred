import pytest
from PySide6 import QtCore

pytestmark = [pytest.mark.gui]


def test_simulation_progress_does_not_force_process_events(main_window, monkeypatch) -> None:
    calls = {"n": 0}

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return None

    monkeypatch.setattr(QtCore.QCoreApplication, "processEvents", _counting)

    main_window.simulation_controller.on_simulation_progress(10, "hello")

    assert calls["n"] == 0
