"""GUI preference payload tests for kindred.gui.project_schema."""

import pytest

pytestmark = [pytest.mark.gui]

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
    "fitting_use_seed",
    "fitting_seed",
    "fitting_solver",
    "fitting_rtol",
    "fitting_atol",
}


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

    def test_workers_clamped_to_shared_ceiling(self, main_window):
        from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
        from kindred.gui.project_schema import get_user_preference_payload

        settings = main_window._settings
        settings.clear()
        settings.setValue("simulation/max_parallel_batch_workers", 200)
        settings.sync()

        payload = get_user_preference_payload(settings)
        assert payload["max_parallel_batch_workers"] == int(MAX_PARALLEL_WORKERS_CEILING)

    def test_project_only_keys_use_factory_defaults(self, main_window):
        from kindred.gui.project_schema import PROJECT_DEFAULTS, get_user_preference_payload

        settings = main_window._settings
        settings.clear()
        settings.sync()

        payload = get_user_preference_payload(settings)
        assert payload["mechanism"] == PROJECT_DEFAULTS["mechanism"]
        assert payload["notes"] == PROJECT_DEFAULTS["notes"]
        assert payload["batch_initial_conditions"] == PROJECT_DEFAULTS["batch_initial_conditions"]
