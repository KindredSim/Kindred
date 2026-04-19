from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]


def test_build_profile_and_template_managers_loads_profiles_once(monkeypatch) -> None:
    from kindred.config.profiles import ProfileManager
    from kindred.gui.app_wiring import build_profile_and_template_managers

    calls: list[int] = []

    def _spy_load_profiles(self) -> None:
        calls.append(id(self))

    monkeypatch.setattr(ProfileManager, "load_profiles", _spy_load_profiles)

    managers = build_profile_and_template_managers()

    assert isinstance(managers.profile_manager, ProfileManager)
    assert len(calls) == 1
