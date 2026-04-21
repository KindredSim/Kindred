from __future__ import annotations
import pytest

pytestmark = pytest.mark.unit



def test_theme_manager_apply_light_runs_on_first_call(monkeypatch) -> None:
    import kindred.gui.theme_manager as theme_manager

    calls: list[str] = []
    monkeypatch.setattr(
        theme_manager.qdarktheme,
        "setup_theme",
        lambda theme: calls.append(theme),
        raising=False,
    )

    refreshed: list[bool] = []

    def _fake_refresh(self, is_dark: bool | None = None) -> None:
        refreshed.append(bool(is_dark))

    monkeypatch.setattr(theme_manager.ThemeManager, "refresh", _fake_refresh)

    class _StubPlotTabs:
        _main_plot = None
        _dataset_plots = []
        _grid_view = None

    tm = theme_manager.ThemeManager(_StubPlotTabs())
    tm.apply(False)

    assert calls == ["light"]
    assert refreshed == [False]
