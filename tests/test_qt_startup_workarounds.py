from __future__ import annotations

import os

import pytest


pytestmark = [pytest.mark.unit]


def test_qt_startup_workaround_auto_wsl_enables_software(monkeypatch):
    import kindred.gui.startup as startup

    monkeypatch.setenv("WSL_INTEROP", "1")
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("QT_OPENGL", raising=False)
    monkeypatch.delenv("KINDRED_QT_OPENGL", raising=False)

    calls = []

    def _capture(attr, enabled):
        calls.append((attr, enabled))

    applied = startup.apply_qt_startup_workarounds(set_qt_attribute=_capture)
    assert applied is True
    assert os.environ.get("QT_OPENGL") == "software"
    assert calls and calls[0][1] is True


def test_qt_startup_workaround_default_override_disables_changes(monkeypatch):
    import kindred.gui.startup as startup

    monkeypatch.setenv("WSL_INTEROP", "1")
    monkeypatch.setenv("KINDRED_QT_OPENGL", "default")
    monkeypatch.delenv("QT_OPENGL", raising=False)

    calls = []

    def _capture(attr, enabled):
        calls.append((attr, enabled))

    applied = startup.apply_qt_startup_workarounds(set_qt_attribute=_capture)
    assert applied is False
    assert "QT_OPENGL" not in os.environ
    assert not calls


def test_qt_startup_workaround_software_override_forces_changes(monkeypatch):
    import kindred.gui.startup as startup

    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setenv("KINDRED_QT_OPENGL", "software")
    monkeypatch.delenv("QT_OPENGL", raising=False)

    calls = []

    def _capture(attr, enabled):
        calls.append((attr, enabled))

    applied = startup.apply_qt_startup_workarounds(set_qt_attribute=_capture)
    assert applied is True
    assert os.environ.get("QT_OPENGL") == "software"
    assert calls and calls[0][1] is True
