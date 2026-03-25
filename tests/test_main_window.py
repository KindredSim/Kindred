import pytest


pytestmark = pytest.mark.gui


def test_get_solver_settings_falls_back_to_radau(main_window):
    main_window._initial_solver = None
    settings = main_window._get_solver_settings()
    assert settings["solver"] == "Radau"


def test_clear_profile_resets_solver_to_radau(main_window):
    # Ensure we start from a non-default value so this test is meaningful.
    main_window._initial_solver = "BDF"
    main_window._clear_profile()
    assert main_window._initial_solver == "Radau"

