from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gui]


def test_main_window_keeps_only_load_bearing_public_mechanism_helper_facades(main_window) -> None:
    removed = (
        "last_mechanism",
        "last_mechanism_context",
        "remember_last_mechanism",
        "is_energy_mode_mechanism",
        "dsl_has_computational_mode_generated_block",
        "sync_energy_mode_temperature_from_mechanism",
        "extract_and_populate_variables",
    )
    for name in removed:
        assert name not in type(main_window).__dict__, f"Dead mechanism-helper forwarder {name} should be removed."

    kept = (
        "set_temperature_override_state",
        "set_temperature_mode_indicator_text",
        "update_temperature_mode_indicator",
        "apply_pending_init_migration",
        "populate_energy_mode_variables_from_mechanism",
    )
    for name in kept:
        assert name in type(main_window).__dict__, f"Load-bearing mechanism-helper facade {name} should remain on MainWindow."
