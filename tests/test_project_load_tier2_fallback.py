import pytest

from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.gui.mixins.fitting_mixin import _FITTING_KEY_TO_SHORT
from kindred.gui.project_schema import FITTING_DEFAULTS_KEYS, PROJECT_DEFAULTS

pytestmark = pytest.mark.gui


_NON_FITTING_PAYLOAD_VALUES = {
    "solver": "BDF",
    "rtol": 2e-5,
    "atol": 3e-10,
    "use_sparse_jacobian": True,
    "wegscheider_cyclicity_enabled": True,
    "max_parallel_batch_workers": 9,
    "limit_blas_threads_per_worker": True,
    "temperature_K": 345.0,
    "simulation_time": "42.5",
    "num_points": 211,
}

_NON_FITTING_PREF_VALUES = {
    "solver": "Radau",
    "rtol": 7e-4,
    "atol": 9e-8,
    "use_sparse_jacobian": False,
    "wegscheider_cyclicity_enabled": False,
    "max_parallel_batch_workers": 7,
    "limit_blas_threads_per_worker": False,
    "temperature_K": 321.0,
    "simulation_time": "27.5",
    "num_points": 234,
}

_FITTING_PREF_VALUES = {
    "fitting_method": "dogbox",
    "fitting_max_nfev": 5000,
    "fitting_ftol": 1e-7,
    "fitting_xtol": 2e-7,
    "fitting_use_seed": False,
    "fitting_seed": 99,
    "fitting_solver": "Radau",
    "fitting_rtol": 3e-6,
    "fitting_atol": 4e-11,
}


def _make_project_payload() -> dict[str, object]:
    payload = {
        "mechanism": "A -> B; k=1",
        "batch_initial_conditions": {"sets": {"set1": {"A": 1.0}}, "species": ["A"]},
    }
    payload.update(_NON_FITTING_PAYLOAD_VALUES)
    return payload


def _read_loaded_value(main_window, key: str) -> object:
    if key == "solver":
        return str(main_window._initial_solver)
    if key == "rtol":
        return float(main_window._initial_rtol)
    if key == "atol":
        return float(main_window._initial_atol)
    if key == "use_sparse_jacobian":
        return bool(main_window._use_sparse_jacobian)
    if key == "wegscheider_cyclicity_enabled":
        return bool(main_window._wegscheider_cyclicity_enabled)
    if key == "max_parallel_batch_workers":
        return int(main_window._sim_controller.parallel_batch.max_parallel_workers)
    if key == "limit_blas_threads_per_worker":
        return bool(main_window._sim_controller.parallel_batch.limit_blas_threads_per_worker)
    if key == "temperature_K":
        return float(main_window._temperature_spinbox.value())
    if key == "simulation_time":
        return str(main_window._sim_time_spinbox.text()).strip()
    if key == "num_points":
        return int(main_window._num_points_spinbox.value())
    raise AssertionError(f"Unexpected key: {key}")


def _assert_loaded_value(main_window, key: str, expected: object) -> None:
    actual = _read_loaded_value(main_window, key)
    if isinstance(expected, float):
        assert actual == pytest.approx(expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("missing_key", list(_NON_FITTING_PREF_VALUES))
def test_apply_project_payload_missing_non_fitting_key_uses_tier2_preference(
    main_window, missing_key: str
):
    payload = _make_project_payload()
    payload.pop(missing_key)
    pref_value = _NON_FITTING_PREF_VALUES[missing_key]

    assert pref_value != PROJECT_DEFAULTS[missing_key]

    main_window.config_controller.update_user_preference(missing_key, pref_value)

    main_window._apply_project_payload(payload, record_undo=False)

    _assert_loaded_value(main_window, missing_key, pref_value)


def test_apply_project_payload_clamps_tier2_parallel_worker_fallback(main_window):
    payload = _make_project_payload()
    payload.pop("max_parallel_batch_workers")

    main_window.config_controller.update_user_preference(
        "max_parallel_batch_workers",
        int(MAX_PARALLEL_WORKERS_CEILING) + 17,
    )

    main_window._apply_project_payload(payload, record_undo=False)

    assert (
        int(main_window._sim_controller.parallel_batch.max_parallel_workers)
        == int(MAX_PARALLEL_WORKERS_CEILING)
    )


def test_apply_project_payload_invalid_tier2_parallel_worker_value_ends_at_factory_default(
    main_window, monkeypatch
):
    payload = _make_project_payload()
    payload.pop("max_parallel_batch_workers")
    main_window.config_controller._user_preferences["max_parallel_batch_workers"] = "not-an-int"

    calls: list[str] = []
    original_get_user_preference = main_window.config_controller.get_user_preference

    def _recording_get_user_preference(key: str) -> object:
        calls.append(str(key))
        return original_get_user_preference(key)

    monkeypatch.setattr(
        main_window.config_controller,
        "get_user_preference",
        _recording_get_user_preference,
    )

    main_window._apply_project_payload(payload, record_undo=False)

    assert "max_parallel_batch_workers" in calls
    assert (
        int(main_window._sim_controller.parallel_batch.max_parallel_workers)
        == int(PROJECT_DEFAULTS["max_parallel_batch_workers"])
    )


def test_apply_project_payload_missing_fitting_keys_keep_fitting_defaults_sparse_and_live(
    main_window,
):
    payload = _make_project_payload()
    payload.pop("solver")

    for full_key, value in _FITTING_PREF_VALUES.items():
        main_window.config_controller.update_user_preference(full_key, value)

    main_window.config_controller.update_user_preference("solver", "Radau")

    for full_key in FITTING_DEFAULTS_KEYS:
        payload.pop(full_key, None)

    main_window._apply_project_payload(payload, record_undo=False)

    assert main_window._initial_solver == "Radau"
    assert main_window._fitting_defaults == {}

    session_defaults = main_window._get_fitting_session_defaults()
    for full_key, short_key in _FITTING_KEY_TO_SHORT.items():
        expected = _FITTING_PREF_VALUES[full_key]
        actual = session_defaults[short_key]
        if isinstance(expected, float):
            assert actual == pytest.approx(expected)
        else:
            assert actual == expected


def test_apply_project_payload_missing_temperature_uses_same_tier2_value_for_spinbox_and_stash(
    main_window, monkeypatch
):
    payload = _make_project_payload()
    payload.pop("temperature_K")
    original_set_value = main_window._temperature_spinbox.setValue
    monkeypatch.setattr(main_window, "_update_temperature_mode_indicator", lambda: None)

    def _record_temperature_write(value: float) -> None:
        main_window._temperature_dsl_override_active = True
        original_set_value(value)

    main_window._temperature_spinbox.setValue = _record_temperature_write
    main_window.config_controller.update_user_preference("temperature_K", 333.0)

    main_window._apply_project_payload(payload, record_undo=False)

    assert float(main_window._temperature_spinbox.value()) == pytest.approx(333.0)
    assert float(main_window._pre_dsl_temperature) == pytest.approx(333.0)
