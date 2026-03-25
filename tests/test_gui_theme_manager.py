import pytest

import qdarktheme

from kindred.gui.theme_manager import ThemeManager


class _DummyPlotTabs:
    pass


@pytest.mark.gui
def test_theme_manager_apply_calls_qdarktheme_setup_theme(monkeypatch):
    calls: list[str] = []

    def _fake_setup_theme(theme: str) -> None:
        calls.append(theme)

    monkeypatch.setattr(qdarktheme, "setup_theme", _fake_setup_theme, raising=False)

    tm = ThemeManager(_DummyPlotTabs())
    tm.apply(True)
    assert calls == ["dark"]

    calls.clear()
    tm.apply(False)
    assert calls == ["light"]


@pytest.mark.gui
def test_theme_manager_apply_refreshes_with_theme_bool(monkeypatch):
    tm = ThemeManager(_DummyPlotTabs())

    refresh_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _fake_refresh(*args, **kwargs) -> None:
        refresh_calls.append((args, kwargs))

    monkeypatch.setattr(tm, "refresh", _fake_refresh)

    tm.apply(True)
    assert refresh_calls, "expected ThemeManager.apply() to call refresh()"
    assert refresh_calls[-1] == ((True,), {})
