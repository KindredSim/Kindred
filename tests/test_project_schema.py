"""Tests for kindred.gui.project_schema — single source of truth for project defaults."""
import os

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

    def test_performance_defaults_match_expected_values(self):
        from kindred.gui.project_schema import PROJECT_DEFAULTS

        assert PROJECT_DEFAULTS["use_sparse_jacobian"] is True
        assert PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"] is True
        assert PROJECT_DEFAULTS["max_parallel_batch_workers"] == min(
            max(1, (os.cpu_count() or 1) - 1),
            16,
        )

    def test_runtime_default_batch_workers_caps_high_cpu_count(self):
        from kindred.core import runtime_defaults

        assert runtime_defaults._compute_max_parallel_batch_workers_default(cpu_count=32) == 16

    def test_runtime_default_batch_workers_uses_cpu_minus_one_below_cap(self):
        from kindred.core import runtime_defaults

        assert runtime_defaults._compute_max_parallel_batch_workers_default(cpu_count=8) == 7


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
