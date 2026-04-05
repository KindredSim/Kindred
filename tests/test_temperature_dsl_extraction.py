"""Regression tests for T= extraction from reactions-only DSL.

Covers:
- Spinbox value sync when T= is in reactions text without state network
- Spinbox visibility and enabled state with DSL-derived temperature
- Indicator text reflects DSL temperature source
- blockSignals guard prevents preference persistence of DSL-derived values
- Spinbox reverts to hidden when T= is removed
- Normal isothermal path unaffected when no T= in DSL
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gui]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_reactions(main_window, text: str) -> None:
    """Set reactions text and allow signals to propagate."""
    main_window._mechanism_editor._reactions_text.setPlainText(text)


def _clear_state_network(main_window) -> None:
    """Ensure state network is empty."""
    try:
        main_window._mechanism_editor._state_network_editor.set_state_network_dsl("")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReactionsOnlyTemperatureExtraction:
    """T= in reactions-only DSL (no state network) must sync the spinbox."""

    def test_reactions_only_T_syncs_spinbox_value(self, main_window):
        _clear_state_network(main_window)
        main_window._temperature_spinbox.setValue(298.15)

        _set_reactions(main_window, "T=350\nA -> B ; k=1")

        assert main_window._temperature_spinbox.value() == pytest.approx(350.0)

    def test_reactions_only_T_disables_spinbox(self, main_window):
        _clear_state_network(main_window)
        main_window._temperature_spinbox.setValue(298.15)

        _set_reactions(main_window, "T=350\nA -> B ; k=1")

        assert not main_window._temperature_spinbox.isEnabled()
        # isHidden() checks the widget's own visibility flag (not ancestor state)
        assert not main_window._temperature_spinbox.isHidden()

    def test_reactions_only_T_indicator_shows_dsl_source(self, main_window):
        _clear_state_network(main_window)
        main_window._temperature_spinbox.setValue(298.15)

        _set_reactions(main_window, "T=350\nA -> B ; k=1")

        indicator_text = main_window._temperature_mode_indicator.text()
        assert "350.00" in indicator_text
        assert "from DSL" in indicator_text


class TestWriteBackGuard:
    """DSL-derived write-back must NOT leak into user preferences."""

    def test_dsl_writeback_does_not_persist_as_user_preference(self, main_window):
        _clear_state_network(main_window)
        cc = main_window.config_controller
        # Establish a known baseline preference via manual spinbox edit
        main_window._temperature_spinbox.setValue(300.0)
        assert cc._user_preferences.get("temperature_K") == pytest.approx(300.0)

        _set_reactions(main_window, "T=350\nA -> B ; k=1")

        # blockSignals guard must prevent DSL-derived 350 from leaking into preferences
        assert cc._user_preferences.get("temperature_K") == pytest.approx(300.0)


class TestSpinboxVisibilityToggle:
    """Spinbox visibility must track T= presence in DSL."""

    def test_removing_T_hides_spinbox_and_restores_preference(self, main_window):
        _clear_state_network(main_window)

        # Establish a known user preference via manual spinbox edit
        main_window._temperature_spinbox.setValue(300.0)

        # Add T= — spinbox should become visible with DSL value
        _set_reactions(main_window, "T=350\nA -> B ; k=1")
        assert not main_window._temperature_spinbox.isHidden()
        assert main_window._temperature_spinbox.value() == pytest.approx(350.0)

        # Remove T= — spinbox should hide and revert to user preference
        _set_reactions(main_window, "A -> B ; k=1")
        assert main_window._temperature_spinbox.isHidden()
        assert main_window._temperature_spinbox.value() == pytest.approx(300.0)


class TestNormalIsothermalPath:
    """When no T= in DSL, the spinbox value is used as isothermal temperature."""

    def test_no_T_in_dsl_uses_spinbox_value(self, main_window):
        _clear_state_network(main_window)
        main_window._temperature_spinbox.setValue(310.0)

        _set_reactions(main_window, "A -> B ; k=1")

        assert main_window._temperature_spinbox.value() == pytest.approx(310.0)
        assert main_window._temperature_spinbox.isEnabled()
        indicator_text = main_window._temperature_mode_indicator.text()
        assert "310.00" in indicator_text


class TestPreferenceRestorationOnTRemoval:
    """Removing T= from DSL must restore the spinbox to the user's preference."""

    def test_removing_T_restores_user_preference_value(self, main_window):
        _clear_state_network(main_window)
        cc = main_window.config_controller
        cc.update_user_preference("temperature_K", 300.0)
        main_window._temperature_spinbox.blockSignals(True)
        try:
            main_window._temperature_spinbox.setValue(300.0)
        finally:
            main_window._temperature_spinbox.blockSignals(False)

        # T= overrides the spinbox to 350
        _set_reactions(main_window, "T=350\nA -> B ; k=1")
        assert main_window._temperature_spinbox.value() == pytest.approx(350.0)

        # Remove T= — spinbox must revert to the user preference, not stay at 350
        _set_reactions(main_window, "A -> B ; k=1")
        assert main_window._temperature_spinbox.value() == pytest.approx(300.0)


class TestScheduleOverridesT:
    """Temperature schedule takes precedence over bare T= for the indicator."""

    def test_schedule_indicator_wins_over_bare_T(self, main_window):
        _clear_state_network(main_window)
        main_window._temperature_spinbox.setValue(298.15)

        _set_reactions(
            main_window,
            "T=350\nA -> B ; k=1\ntemp_step: t=[0,50,100], T=[298,350]",
        )

        indicator_text = main_window._temperature_mode_indicator.text()
        assert "from DSL" not in indicator_text
        assert "Schedule" in indicator_text
        # T= still seeds the spinbox value
        assert main_window._temperature_spinbox.value() == pytest.approx(350.0)


class TestProjectLoadTemperatureStash:
    """Removing T= after project load must restore to the project temperature, not startup default."""

    def test_removing_T_after_project_load_restores_project_temperature(self, main_window):
        _clear_state_network(main_window)

        payload = main_window._serialize_project_state()
        payload["mechanism"] = "T=350\nA -> B ; k=1"
        payload["temperature_K"] = 500.0
        main_window._apply_project_payload(payload, record_undo=False)

        # T= overrides the spinbox to 350
        assert main_window._temperature_spinbox.value() == pytest.approx(350.0)

        # Remove T= — spinbox must revert to the project temperature (500), not 298.15
        _set_reactions(main_window, "A -> B ; k=1")
        assert main_window._temperature_spinbox.value() == pytest.approx(500.0)

    def test_T_active_then_project_load_then_remove_restores_latest_project(self, main_window):
        _clear_state_network(main_window)

        # Establish T= override from manual editing
        main_window._temperature_spinbox.setValue(300.0)
        _set_reactions(main_window, "T=350\nA -> B ; k=1")
        assert main_window._temperature_spinbox.value() == pytest.approx(350.0)

        # Load a different project while T= is already active
        payload = main_window._serialize_project_state()
        payload["mechanism"] = "T=400\nC -> D ; k=2"
        payload["temperature_K"] = 600.0
        main_window._apply_project_payload(payload, record_undo=False)
        assert main_window._temperature_spinbox.value() == pytest.approx(400.0)

        # Remove T= — must restore to latest project's temperature (600), not 300
        _set_reactions(main_window, "C -> D ; k=2")
        assert main_window._temperature_spinbox.value() == pytest.approx(600.0)


class TestStashStabilityDuringTValueChanges:
    """Changing the T= value in DSL must not corrupt the pre-override stash."""

    def test_changing_T_value_preserves_original_stash(self, main_window):
        _clear_state_network(main_window)

        # Set a known spinbox value
        main_window._temperature_spinbox.setValue(400.0)

        # Add T=350 — stash captures 400
        _set_reactions(main_window, "T=350\nA -> B ; k=1")
        assert main_window._temperature_spinbox.value() == pytest.approx(350.0)

        # Change to T=500 — stash must NOT update (still 400)
        _set_reactions(main_window, "T=500\nA -> B ; k=1")
        assert main_window._temperature_spinbox.value() == pytest.approx(500.0)

        # Remove T= — must restore to original 400, not 350 or 500
        _set_reactions(main_window, "A -> B ; k=1")
        assert main_window._temperature_spinbox.value() == pytest.approx(400.0)


class TestSerializationWhileTOverrideActive:
    """Saving a project while T= is active must serialize the base temperature, not the T= value."""

    def test_serialize_preserves_base_temperature_not_T_override(self, main_window):
        _clear_state_network(main_window)

        # Set a known base temperature
        main_window._temperature_spinbox.setValue(500.0)

        # Add T= override — spinbox now shows T= value, not the base
        _set_reactions(main_window, "T=350\nA -> B ; k=1")
        assert main_window._temperature_spinbox.value() == pytest.approx(350.0)

        # Serialize while T= is active
        payload = main_window._serialize_project_state()
        assert payload["temperature_K"] == pytest.approx(500.0)

    def test_save_reload_remove_T_round_trip(self, main_window):
        _clear_state_network(main_window)

        main_window._temperature_spinbox.setValue(500.0)
        _set_reactions(main_window, "T=350\nA -> B ; k=1")

        # Save and reload
        payload = main_window._serialize_project_state()
        main_window._apply_project_payload(payload, record_undo=False)

        # T= still overrides
        assert main_window._temperature_spinbox.value() == pytest.approx(350.0)

        # Remove T= — must restore to 500, not 350
        _set_reactions(main_window, "A -> B ; k=1")
        assert main_window._temperature_spinbox.value() == pytest.approx(500.0)
