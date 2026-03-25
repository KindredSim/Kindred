from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gui]


def test_shared_main_window_fixtures_are_not_available(request) -> None:
    for fixture_name in ("shared_main_window", "reset_shared_main_window", "main_window_init_counter"):
        with pytest.raises(pytest.FixtureLookupError):
            request.getfixturevalue(fixture_name)
