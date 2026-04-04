"""Regression tests for three-tier fitting defaults persistence."""

import pytest
from PySide6 import QtWidgets

from kindred.gui.project_schema import (
    FITTING_DEFAULTS_KEYS,
    PROJECT_DEFAULTS,
    QSETTINGS_KEY_MAP,
)


pytestmark = [pytest.mark.gui]


def test_all_fitting_keys_in_project_defaults_and_qsettings_key_map():
    """All 10 fitting keys exist in both PROJECT_DEFAULTS and QSETTINGS_KEY_MAP."""
    expected_keys = {
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
    assert expected_keys <= set(PROJECT_DEFAULTS.keys())
    assert expected_keys <= set(QSETTINGS_KEY_MAP.keys())
    assert set(FITTING_DEFAULTS_KEYS) == expected_keys


def _make_config_controller():
    """Build a minimal ConfigController with user_preferences initialized."""
    from kindred.gui.controllers.config_controller import ConfigController

    controller = ConfigController.__new__(ConfigController)
    controller._user_preferences = {}
    return controller


def test_three_tier_precedence_factory_default(qt_app):
    """Factory default is returned when QSettings is empty."""
    controller = _make_config_controller()
    for key in FITTING_DEFAULTS_KEYS:
        assert controller.get_user_preference(key) == PROJECT_DEFAULTS[key]


def test_three_tier_precedence_user_pref_overrides_factory(qt_app):
    """User preference (QSettings) overrides factory default."""
    controller = _make_config_controller()
    controller.update_user_preference("fitting_max_nfev", 2000)
    assert controller.get_user_preference("fitting_max_nfev") == 2000


def test_round_trip_kin_save_load(main_window):
    """Serialize project, load it back, verify all 10 fitting defaults survive."""
    main_window._fitting_defaults = {
        "fitting_method": "trf",
        "fitting_max_nfev": 500,
        "fitting_ftol": 1e-8,
        "fitting_xtol": 1e-8,
        "fitting_use_parallel": True,
        "fitting_use_seed": False,
        "fitting_seed": 99,
        "fitting_solver": "Radau",
        "fitting_rtol": 1e-4,
        "fitting_atol": 1e-10,
    }

    payload = main_window._serialize_project_state()
    for key in FITTING_DEFAULTS_KEYS:
        assert key in payload, f"Missing key {key} in serialized payload"

    assert payload["fitting_method"] == "trf"
    assert payload["fitting_max_nfev"] == 500
    assert payload["fitting_solver"] == "Radau"
    assert payload["fitting_seed"] == 99

    # Apply payload back and verify
    main_window._apply_project_payload(payload, record_undo=False)
    for key in FITTING_DEFAULTS_KEYS:
        assert main_window._fitting_defaults[key] == payload[key], (
            f"Mismatch for {key}: {main_window._fitting_defaults[key]} != {payload[key]}"
        )


def test_active_integration_defaults_reads_from_config_defaults(qt_app):
    """_active_integration_defaults_for_ui reads from config_defaults."""
    import numpy as np
    from kindred.gui.fitting.window import FittingWindow

    t = np.linspace(0.0, 1.0, 5)
    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t.copy(),
            "species_data": {"A": np.ones_like(t)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]
    dataset_payloads = [
        {"id": "ds1", "t": t.copy(), "y": np.vstack([np.ones_like(t)]), "species": ["A"]}
    ]

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": np.ones_like(t)}},
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0},
        config_defaults={"solver": "BDF", "rtol": 1e-3, "atol": 1e-9},
    )
    try:
        solver, rtol, atol = window._active_integration_defaults_for_ui()
        assert solver == "BDF"
        assert rtol == 1e-3
        assert atol == 1e-9
    finally:
        window.close()


def test_preferences_action_absent(main_window):
    """No action with objectName 'preferencesAction' exists in menus."""
    from PySide6 import QtGui

    action = main_window.findChild(QtGui.QAction, "preferencesAction")
    assert action is None


def test_keyboard_shortcuts_action_absent(main_window):
    """No action with objectName 'keyboardShortcutsAction' exists in menus."""
    from PySide6 import QtGui

    action = main_window.findChild(QtGui.QAction, "keyboardShortcutsAction")
    assert action is None


def test_fitting_defaults_dialog_updates_all_keys(qt_app, monkeypatch):
    """Fitting Defaults dialog updates all fitting keys via config_controller."""
    from kindred.gui.mixins.fitting_mixin import FittingMixin

    persisted = {}

    class _MockConfigController:
        def get_user_preference(self, key):
            return PROJECT_DEFAULTS.get(key)

        def update_user_preference(self, key, value):
            persisted[key] = value

    class _Host(QtWidgets.QWidget, FittingMixin):
        def __init__(self):
            super().__init__()
            self.config_controller = _MockConfigController()
            self._fitting_defaults = {}
            self._status_label = QtWidgets.QLabel("")

    monkeypatch.setattr(
        QtWidgets.QDialog, "exec", lambda _self: QtWidgets.QDialog.DialogCode.Accepted
    )

    host = _Host()
    try:
        host._configure_fitting()
        # All 10 fitting keys should have been persisted
        expected_keys = set(FITTING_DEFAULTS_KEYS)
        assert expected_keys <= set(persisted.keys()), (
            f"Missing persisted keys: {expected_keys - set(persisted.keys())}"
        )
        # Dialog must not write to document-level _fitting_defaults
        assert host._fitting_defaults == {}
    finally:
        host.close()


def test_load_fitting_defaults_reads_user_prefs_not_document_overrides(main_window):
    """After loading a .kin with fitting_method='dogbox', the Fitting Defaults
    dialog must still show the user's global preference ('trf'), not the
    document override ('dogbox')."""
    # User pref is factory default "trf" (no user override set)
    assert main_window.config_controller.get_user_preference("fitting_method") == "trf"

    # Load a project payload that overrides fitting_method to "dogbox"
    payload = main_window._serialize_project_state()
    payload["fitting_method"] = "dogbox"
    main_window._apply_project_payload(payload, record_undo=False)

    # Live session state should reflect the document override
    assert main_window._fitting_defaults["fitting_method"] == "dogbox"

    # But _load_fitting_defaults (used by the dialog) must return user prefs
    dialog_defaults = main_window._load_fitting_defaults()
    assert dialog_defaults["method"] == "trf", (
        f"Expected user pref 'trf' but got document override '{dialog_defaults['method']}'"
    )


def test_fitting_key_to_short_matches_fitting_defaults_keys():
    """_FITTING_KEY_TO_SHORT must cover exactly the same keys as FITTING_DEFAULTS_KEYS."""
    from kindred.gui.mixins.fitting_mixin import _FITTING_KEY_TO_SHORT

    assert set(_FITTING_KEY_TO_SHORT.keys()) == set(FITTING_DEFAULTS_KEYS)


def test_session_defaults_reads_fitting_defaults_not_user_prefs(main_window):
    """_get_fitting_session_defaults returns document overrides merged with live user prefs."""
    payload = main_window._serialize_project_state()
    payload["fitting_solver"] = "Radau"
    payload["fitting_rtol"] = 1e-4
    main_window._apply_project_payload(payload, record_undo=False)

    assert main_window._fitting_defaults["fitting_solver"] == "Radau"

    session = main_window._get_fitting_session_defaults()
    assert session["solver"] == "Radau"
    assert session["rtol"] == 1e-4

    from kindred.gui.mixins.fitting_mixin import _FITTING_KEY_TO_SHORT

    assert set(session.keys()) == set(_FITTING_KEY_TO_SHORT.values())


def test_dialog_ok_does_not_clobber_project_state(main_window, monkeypatch):
    """Accepting the Fitting Defaults dialog must NOT mutate _fitting_defaults."""
    payload = main_window._serialize_project_state()
    payload["fitting_solver"] = "Radau"
    main_window._apply_project_payload(payload, record_undo=False)
    assert main_window._fitting_defaults["fitting_solver"] == "Radau"

    monkeypatch.setattr(
        QtWidgets.QDialog, "exec", lambda _self: QtWidgets.QDialog.DialogCode.Accepted
    )
    main_window._configure_fitting()

    assert main_window._fitting_defaults["fitting_solver"] == "Radau", (
        "Dialog OK must not overwrite project-level _fitting_defaults"
    )


def test_active_integration_defaults_none_check_not_falsy_or(qt_app):
    """Non-default values in config_defaults must pass through to the result."""
    import numpy as np
    from kindred.gui.fitting.window import FittingWindow

    t = np.linspace(0.0, 1.0, 5)
    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=[{
            "id": "ds1", "label": "ds1",
            "t": t.copy(), "species_data": {"A": np.ones_like(t)},
            "selected_species": ["A"], "weight": 1.0, "include": True,
        }],
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": np.ones_like(t)}},
        dataset_payloads=[
            {"id": "ds1", "t": t.copy(), "y": np.vstack([np.ones_like(t)]), "species": ["A"]}
        ],
        dataset_weights={"ds1": 1.0},
        config_defaults={"solver": "Radau", "rtol": 1e-5, "atol": 1e-11},
    )
    try:
        solver, rtol, atol = window._active_integration_defaults_for_ui()
        assert solver == "Radau"
        assert rtol == 1e-5
        assert atol == 1e-11
    finally:
        window.close()


def test_fitting_defaults_empty_at_startup(main_window):
    """No document loaded at startup means no document overrides."""
    assert main_window._fitting_defaults == {}, (
        f"Expected empty dict at startup, got {main_window._fitting_defaults}"
    )


def test_dialog_change_takes_effect_on_session_defaults(main_window):
    """Changing a fitting preference via config_controller must be reflected
    in _get_fitting_session_defaults() without reload."""
    session_before = main_window._get_fitting_session_defaults()
    assert session_before["max_nfev"] == 1000

    main_window.config_controller.update_user_preference("fitting_max_nfev", 5000)

    session_after = main_window._get_fitting_session_defaults()
    assert session_after["max_nfev"] == 5000, (
        f"Expected 5000 after dialog change, got {session_after['max_nfev']}"
    )


def test_document_override_preserved_after_dialog_change(main_window):
    """A document override must take precedence even after the user changes
    the same key via the Fitting Defaults dialog."""
    payload = main_window._serialize_project_state()
    payload["fitting_max_nfev"] = 500
    main_window._apply_project_payload(payload, record_undo=False)

    session = main_window._get_fitting_session_defaults()
    assert session["max_nfev"] == 500

    main_window.config_controller.update_user_preference("fitting_max_nfev", 9999)

    session_after = main_window._get_fitting_session_defaults()
    assert session_after["max_nfev"] == 500, (
        f"Document override should win, got {session_after['max_nfev']}"
    )


def test_partial_document_load_stores_only_present_keys(main_window):
    """Loading a document with only some fitting keys must store only those
    keys, while _get_fitting_session_defaults still returns all 10 short keys."""
    # Build a payload that has only 3 of the 10 fitting keys
    payload = main_window._serialize_project_state()
    payload["fitting_method"] = "dogbox"
    payload["fitting_max_nfev"] = 500
    payload["fitting_solver"] = "Radau"
    # Ensure the other 7 are absent
    for key in ("fitting_ftol", "fitting_xtol", "fitting_use_parallel",
                "fitting_use_seed", "fitting_seed", "fitting_rtol", "fitting_atol"):
        payload.pop(key, None)

    main_window._apply_project_payload(payload, record_undo=False)

    assert "fitting_method" in main_window._fitting_defaults
    assert "fitting_max_nfev" in main_window._fitting_defaults
    assert "fitting_solver" in main_window._fitting_defaults
    assert "fitting_ftol" not in main_window._fitting_defaults
    assert "fitting_use_parallel" not in main_window._fitting_defaults

    session = main_window._get_fitting_session_defaults()
    from kindred.gui.mixins.fitting_mixin import _FITTING_KEY_TO_SHORT
    assert set(session.keys()) == set(_FITTING_KEY_TO_SHORT.values())
    assert session["ftol"] == PROJECT_DEFAULTS["fitting_ftol"]


def test_save_load_round_trip_preserves_sparsity(main_window):
    """Saving a project with 3 document overrides and reloading must not
    inflate _fitting_defaults to all 10 keys."""
    main_window._fitting_defaults = {
        "fitting_method": "dogbox",
        "fitting_solver": "Radau",
        "fitting_seed": 99,
    }

    payload = main_window._serialize_project_state()

    # Only the 3 override keys should appear in the payload
    assert payload.get("fitting_method") == "dogbox"
    assert payload.get("fitting_solver") == "Radau"
    assert payload.get("fitting_seed") == 99
    assert "fitting_ftol" not in payload
    assert "fitting_max_nfev" not in payload

    # Round-trip: reload and verify sparsity is preserved
    main_window._apply_project_payload(payload, record_undo=False)
    assert set(main_window._fitting_defaults.keys()) == {
        "fitting_method", "fitting_solver", "fitting_seed",
    }

    # Session defaults must still return all 10 short keys
    session = main_window._get_fitting_session_defaults()
    assert session["method"] == "dogbox"
    assert session["solver"] == "Radau"
    assert session["seed"] == 99
    # Non-overridden keys come from tier 2
    assert session["max_nfev"] == PROJECT_DEFAULTS["fitting_max_nfev"]
