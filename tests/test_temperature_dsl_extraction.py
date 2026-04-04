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

    def test_removing_T_hides_spinbox_and_keeps_value(self, main_window):
        _clear_state_network(main_window)

        # Add T= — spinbox should become visible (not hidden)
        _set_reactions(main_window, "T=350\nA -> B ; k=1")
        assert not main_window._temperature_spinbox.isHidden()

        # Remove T= — spinbox should hide but keep value
        _set_reactions(main_window, "A -> B ; k=1")
        assert main_window._temperature_spinbox.isHidden()
        assert main_window._temperature_spinbox.value() == pytest.approx(350.0)


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
