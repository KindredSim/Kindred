"""Tests for kindred.gui.project_schema — single source of truth for project defaults."""
import math
import os
import sys

import pytest

from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.gui.fitting.constants import FITTING_MAX_NFEV_RANGE, FITTING_SEED_RANGE
from kindred.gui.project_schema import SIMULATION_NUM_POINTS_RANGE


EXPECTED_KEYS = {
    "mechanism_source",
    "notes",
    "batch_initial_conditions",
    "solver",
    "rtol",
    "atol",
    "use_sparse_jacobian",
    "wegscheider_cyclicity_enabled",
    "max_parallel_batch_workers",
    "batch_runtime_lane_budget",
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


def _complete_current_project_payload() -> dict[str, object]:
    from kindred.gui.project_schema import get_default_project_payload

    payload = get_default_project_payload()
    payload["version"] = "test"
    payload["solver_method"] = str(payload["solver"])
    payload["solver_warning"] = None
    return payload


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
        first["mechanism_source"]["reactions_text"] = "mutated"

        second = get_default_project_payload()
        assert second["batch_initial_conditions"] == {}
        assert second["mechanism_source"] == {
            "reactions_text": "",
            "state_network_dsl": "",
        }

    def test_default_payload_types_and_authoritative_values(self):
        from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME
        from kindred.gui.project_schema import PROJECT_DEFAULTS, get_default_project_payload

        p = get_default_project_payload()
        assert "project_schema_version" not in p
        assert isinstance(p["mechanism_source"], dict)
        assert isinstance(p["mechanism_source"]["reactions_text"], str)
        assert isinstance(p["mechanism_source"]["state_network_dsl"], str)
        assert isinstance(p["notes"], str)
        assert isinstance(p["batch_initial_conditions"], dict)
        assert isinstance(p["solver"], str)
        assert isinstance(p["rtol"], float)
        assert isinstance(p["atol"], float)
        assert isinstance(p["use_sparse_jacobian"], bool)
        assert isinstance(p["wegscheider_cyclicity_enabled"], bool)
        assert isinstance(p["max_parallel_batch_workers"], int)
        assert isinstance(p["batch_runtime_lane_budget"], int)
        assert isinstance(p["limit_blas_threads_per_worker"], bool)
        assert isinstance(p["temperature_K"], float)
        assert isinstance(p["simulation_time"], str)
        assert isinstance(p["num_points"], int)
        assert PROJECT_DEFAULTS["solver"] == DEFAULT_SOLVER_NAME
        assert PROJECT_DEFAULTS["use_sparse_jacobian"] is True
        assert PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"] is True
        assert PROJECT_DEFAULTS["max_parallel_batch_workers"] == min(
            max(1, (os.cpu_count() or 1) - 1),
            16,
        )
        assert (
            PROJECT_DEFAULTS["batch_runtime_lane_budget"]
            == PROJECT_DEFAULTS["max_parallel_batch_workers"]
        )

    def test_project_payload_contract_has_no_schema_version_surface(self):
        import kindred.gui.project_schema as project_schema

        payload = project_schema.get_default_project_payload()

        assert "PROJECT_SCHEMA_VERSION" not in project_schema.__all__
        assert not hasattr(project_schema, "PROJECT_SCHEMA_VERSION")
        assert "project_schema_version" not in project_schema.PROJECT_DEFAULTS
        assert "project_schema_version" not in payload

    @pytest.mark.parametrize("legacy_key", ["project_schema_version", "mechanism", "state_network", "use_advanced_dsl"])
    def test_current_project_payload_rejects_unknown_top_level_fields(self, legacy_key):
        from kindred.gui.project_schema import validate_project_payload

        payload = _complete_current_project_payload()
        payload[legacy_key] = "legacy"

        with pytest.raises(ValueError, match=legacy_key):
            validate_project_payload(payload)

    @pytest.mark.parametrize(
        "missing_key",
        ["version", "notes", "batch_initial_conditions", "temperature_K", "num_points", "fitting_solver"],
    )
    def test_current_project_payload_rejects_missing_required_fields(self, missing_key):
        from kindred.gui.project_schema import validate_project_payload

        payload = _complete_current_project_payload()
        del payload[missing_key]

        with pytest.raises(ValueError, match=missing_key):
            validate_project_payload(payload)

    def test_current_project_payload_rejects_inconsistent_solver_metadata(self):
        from kindred.gui.project_schema import validate_project_payload

        payload = _complete_current_project_payload()
        payload["solver"] = "BDF"
        payload["solver_method"] = "Radau"
        payload["solver_warning"] = None

        with pytest.raises(ValueError, match="solver_method"):
            validate_project_payload(payload)

    @pytest.mark.parametrize(
        "field",
        ["rtol", "atol", "temperature_K", "fitting_ftol", "fitting_xtol", "fitting_rtol", "fitting_atol"],
    )
    @pytest.mark.parametrize(
        "bad_value",
        [math.nan, math.inf, -math.inf, 0.0, -1.0, sys.float_info.max],
    )
    def test_current_project_payload_rejects_invalid_positive_numeric_fields(self, field, bad_value):
        from kindred.gui.project_schema import validate_project_payload

        payload = _complete_current_project_payload()
        payload[field] = bad_value

        with pytest.raises(ValueError, match=field):
            validate_project_payload(payload)

    def test_current_project_payload_accepts_ui_supported_large_fitting_tolerances(self):
        from kindred.gui.project_schema import validate_project_payload

        payload = _complete_current_project_payload()
        payload["fitting_ftol"] = 10.0
        payload["fitting_xtol"] = 10.0
        payload["fitting_rtol"] = 10.0
        payload["fitting_atol"] = 10.0

        validate_project_payload(payload)

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("num_points", SIMULATION_NUM_POINTS_RANGE[0] - 1),
            ("num_points", SIMULATION_NUM_POINTS_RANGE[1] + 1),
            ("max_parallel_batch_workers", 0),
            ("max_parallel_batch_workers", int(MAX_PARALLEL_WORKERS_CEILING) + 1),
            ("batch_runtime_lane_budget", 0),
            ("batch_runtime_lane_budget", int(MAX_PARALLEL_WORKERS_CEILING) + 1),
            ("fitting_max_nfev", FITTING_MAX_NFEV_RANGE[0] - 1),
            ("fitting_max_nfev", FITTING_MAX_NFEV_RANGE[1] + 1),
            ("fitting_seed", FITTING_SEED_RANGE[0] - 1),
            ("fitting_seed", FITTING_SEED_RANGE[1] + 1),
        ],
    )
    def test_current_project_payload_rejects_integer_fields_outside_current_ui_ranges(
        self,
        field,
        bad_value,
    ):
        from kindred.gui.project_schema import validate_project_payload

        payload = _complete_current_project_payload()
        payload[field] = bad_value

        with pytest.raises(ValueError, match=field):
            validate_project_payload(payload)

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("simulation_time", ""),
            ("simulation_time", "not-a-number"),
            ("simulation_time", "nan"),
            ("simulation_time", "inf"),
            ("simulation_time", "0"),
            ("simulation_time", "-1"),
            ("simulation_time", str(sys.float_info.max)),
            ("fitting_method", "bogus"),
            ("fitting_solver", "bogus"),
        ],
    )
    def test_current_project_payload_rejects_semantically_invalid_string_fields(
        self,
        field,
        bad_value,
    ):
        from kindred.gui.project_schema import validate_project_payload

        payload = _complete_current_project_payload()
        payload[field] = bad_value

        with pytest.raises(ValueError, match=field):
            validate_project_payload(payload)

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
    "batch_runtime_lane_budget",
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
