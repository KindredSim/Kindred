from __future__ import annotations


import pytest


pytestmark = [pytest.mark.unit]


def test_startup_debug_mode_disables_stderr_redirection(monkeypatch):
    import kindred.gui.startup as startup

    monkeypatch.setenv("KINDRED_DEBUG_STARTUP", "1")
    assert startup.startup_debug_enabled() is True
    assert startup.should_redirect_stderr(startup_debug=True) is False


@pytest.mark.gui
def test_main_window_init_does_not_create_batch_executor(qtbot, monkeypatch):
    import kindred.gui.main_window as mw
    import kindred.gui.controllers.simulation_controller as sc

    called = {"value": False}

    def _boom(*_args, **_kwargs):
        called["value"] = True
        raise AssertionError("batch executor should not be created during startup")

    monkeypatch.setattr(sc, "_default_batch_executor_factory", _boom)
    window = mw.MainWindow()
    qtbot.addWidget(window)
    assert called["value"] is False
    window.close()
