import pytest


pytestmark = pytest.mark.gui


def test_get_solver_settings_falls_back_to_bdf(main_window):
    main_window._initial_solver = None
    settings = main_window._get_solver_settings()
    assert settings["solver"] == "BDF"


def test_clear_profile_resets_solver_to_bdf(main_window):
    # Ensure we start from a non-default value so this test is meaningful.
    main_window._initial_solver = "Radau"
    main_window._clear_profile()
    assert main_window._initial_solver == "BDF"


def test_profile_manager_normalizes_invalid_solver_to_bdf():
    from kindred.config.profiles import Profile, ProfileManager

    loaded = Profile.from_dict({"name": "Loaded", "solver": {"method": "unknown_solver_name"}})
    manager = ProfileManager(auto_load=False)
    created = manager.create_profile("Created", solver_method="unknown_solver_name")

    assert loaded.solver_method == "BDF"
    assert created.solver_method == "BDF"


def test_profile_from_dict_rejects_non_mapping_solver_section():
    from kindred.config.profiles import Profile

    with pytest.raises(TypeError, match="solver"):
        Profile.from_dict({"name": "Loaded", "solver": "not-a-dict"})


def test_profile_from_dict_resets_invalid_tolerances_to_defaults():
    from kindred.config.profiles import Profile

    loaded = Profile.from_dict(
        {"name": "Loaded", "solver": {"method": "unknown_solver_name", "rtol": float("nan"), "atol": float("inf")}}
    )

    assert loaded.solver_method == "BDF"
    assert loaded.rtol == pytest.approx(1e-6)
    assert loaded.atol == pytest.approx(1e-12)


def test_profile_manager_create_profile_resets_invalid_tolerances_to_defaults():
    from kindred.config.profiles import ProfileManager

    manager = ProfileManager(auto_load=False)
    created = manager.create_profile(
        "Created",
        solver_method="unknown_solver_name",
        rtol=float("nan"),
        atol=float("inf"),
    )

    assert created.solver_method == "BDF"
    assert created.rtol == pytest.approx(1e-6)
    assert created.atol == pytest.approx(1e-12)
