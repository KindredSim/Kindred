"""Tests for kindred.gui.project_schema — single source of truth for project defaults."""
import pytest


EXPECTED_KEYS = {
    "mechanism",
    "notes",
    "state_network",
    "batch_initial_conditions",
    "solver",
    "rtol",
    "atol",
    "use_sparse_jacobian",
    "wegscheider_cyclicity_enabled",
    "max_parallel_batch_workers",
    "limit_blas_threads_per_worker",
    "temperature_K",
    "simulation_time",
    "num_points",
    "fitting_method",
    "fitting_max_nfev",
    "fitting_ftol",
    "fitting_xtol",
    "fitting_use_parallel",
    "fitting_use_seed",
    "fitting_seed",
    "fitting_solver",
    "fitting_rtol",
    "fitting_atol",
}


@pytest.mark.unit
class TestProjectDefaults:
    def test_schema_completeness(self):
        from kindred.gui.project_schema import get_default_project_payload

        payload = get_default_project_payload()
        assert set(payload.keys()) == EXPECTED_KEYS

    def test_mutable_safety(self):
        from kindred.gui.project_schema import get_default_project_payload

        first = get_default_project_payload()
        first["batch_initial_conditions"]["A"] = 1.0
        first["mechanism"] = "mutated"

        second = get_default_project_payload()
        assert second["batch_initial_conditions"] == {}
        assert second["mechanism"] == ""

    def test_value_types(self):
        from kindred.gui.project_schema import get_default_project_payload

        p = get_default_project_payload()
        assert isinstance(p["mechanism"], str)
        assert isinstance(p["notes"], str)
        assert isinstance(p["state_network"], str)
        assert isinstance(p["batch_initial_conditions"], dict)
        assert isinstance(p["solver"], str)
        assert isinstance(p["rtol"], float)
        assert isinstance(p["atol"], float)
        assert isinstance(p["use_sparse_jacobian"], bool)
        assert isinstance(p["wegscheider_cyclicity_enabled"], bool)
        assert isinstance(p["max_parallel_batch_workers"], int)
        assert isinstance(p["limit_blas_threads_per_worker"], bool)
        assert isinstance(p["temperature_K"], float)
        assert isinstance(p["simulation_time"], str)
        assert isinstance(p["num_points"], int)

    def test_schema_version_matches(self):
        from kindred.gui.project_schema import PROJECT_SCHEMA_VERSION

        assert PROJECT_SCHEMA_VERSION == 4

    def test_solver_default_matches_core(self):
        from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME
        from kindred.gui.project_schema import PROJECT_DEFAULTS

        assert PROJECT_DEFAULTS["solver"] == DEFAULT_SOLVER_NAME


EXPECTED_DUAL_PERSISTED_KEYS = {
    "solver",
    "rtol",
    "atol",
    "use_sparse_jacobian",
    "wegscheider_cyclicity_enabled",
    "max_parallel_batch_workers",
    "limit_blas_threads_per_worker",
    "temperature_K",
    "simulation_time",
    "num_points",
    "fitting_method",
    "fitting_max_nfev",
    "fitting_ftol",
    "fitting_xtol",
    "fitting_use_parallel",
    "fitting_use_seed",
    "fitting_seed",
    "fitting_solver",
    "fitting_rtol",
    "fitting_atol",
}


@pytest.mark.unit
class TestQSettingsKeyMap:
    def test_map_covers_exactly_dual_persisted_keys(self):
        from kindred.gui.project_schema import QSETTINGS_KEY_MAP

        assert set(QSETTINGS_KEY_MAP.keys()) == EXPECTED_DUAL_PERSISTED_KEYS

    def test_all_map_keys_exist_in_project_defaults(self):
        from kindred.gui.project_schema import PROJECT_DEFAULTS, QSETTINGS_KEY_MAP

        for key in QSETTINGS_KEY_MAP:
            assert key in PROJECT_DEFAULTS, f"QSETTINGS_KEY_MAP key {key!r} missing from PROJECT_DEFAULTS"

    def test_qsettings_paths_are_unique(self):
        from kindred.gui.project_schema import QSETTINGS_KEY_MAP

        paths = list(QSETTINGS_KEY_MAP.values())
        assert len(paths) == len(set(paths)), "Duplicate QSettings paths"


@pytest.mark.gui
class TestGetUserPreferencePayload:
    def test_returns_complete_14_key_payload(self, main_window):
        from kindred.gui.project_schema import get_user_preference_payload

        settings = main_window._settings
        settings.clear()
        settings.sync()

        payload = get_user_preference_payload(settings)
        assert set(payload.keys()) == EXPECTED_KEYS

    def test_reads_qsettings_values_for_dual_persisted_keys(self, main_window):
        from kindred.gui.project_schema import get_user_preference_payload

        settings = main_window._settings
        settings.clear()
        settings.setValue("simulation/solver", "BDF")
        settings.setValue("simulation/rtol", 1e-5)
        settings.setValue("simulation/temperature", 400.0)
        settings.setValue("simulation/points", 250)
        settings.sync()

        payload = get_user_preference_payload(settings)
        assert payload["solver"] == "BDF"
        assert payload["rtol"] == pytest.approx(1e-5)
        assert payload["temperature_K"] == pytest.approx(400.0)
        assert payload["num_points"] == 250

    def test_type_coercion_for_rtol_stored_as_string(self, main_window):
        from kindred.gui.project_schema import get_user_preference_payload

        settings = main_window._settings
        settings.clear()
        settings.setValue("simulation/rtol", "1e-8")
        settings.sync()

        payload = get_user_preference_payload(settings)
        assert isinstance(payload["rtol"], float)
        assert payload["rtol"] == pytest.approx(1e-8)

    def test_workers_clamped_to_minimum_one(self, main_window):
        from kindred.gui.project_schema import get_user_preference_payload

        settings = main_window._settings
        settings.clear()
        settings.setValue("simulation/max_parallel_batch_workers", 0)
        settings.sync()

        payload = get_user_preference_payload(settings)
        assert payload["max_parallel_batch_workers"] >= 1

    def test_project_only_keys_use_factory_defaults(self, main_window):
        from kindred.gui.project_schema import PROJECT_DEFAULTS, get_user_preference_payload

        settings = main_window._settings
        settings.clear()
        settings.sync()

        payload = get_user_preference_payload(settings)
        assert payload["mechanism"] == PROJECT_DEFAULTS["mechanism"]
        assert payload["notes"] == PROJECT_DEFAULTS["notes"]
        assert payload["batch_initial_conditions"] == PROJECT_DEFAULTS["batch_initial_conditions"]
