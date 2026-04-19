from __future__ import annotations


import pytest


@pytest.mark.unit
def test_startup_debug_mode_disables_stderr_redirection(monkeypatch):
    import kindred.gui.startup as startup

    monkeypatch.setenv("KINDRED_DEBUG_STARTUP", "1")
    assert startup.startup_debug_enabled() is True
    assert startup.should_redirect_stderr(startup_debug=True) is False
