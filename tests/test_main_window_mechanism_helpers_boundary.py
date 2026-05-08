from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from kindred.gui.main_window_mechanism_helpers import MainWindowMechanismHelpers


@dataclass
class _FakeMechanismHost:
    marker: dict[str, Any] = field(default_factory=dict)
    temperature_override_calls: list[tuple[bool, str]] = field(default_factory=list)
    temperature_mode_indicator_texts: list[str] = field(default_factory=list)
    temperature_mode_indicator_updates: int = 0
    pending_init_calls: list[tuple[dict[str, float], str]] = field(default_factory=list)
    pending_init_failure_invalidations: int = 0
    focused_control_sync_calls: list[bool] = field(default_factory=list)

    def set_temperature_override_state(self, *, enabled: bool, tooltip: str) -> None:
        self.temperature_override_calls.append((bool(enabled), str(tooltip)))

    def set_temperature_mode_indicator_text(self, text: str) -> None:
        self.temperature_mode_indicator_texts.append(str(text))

    def update_temperature_mode_indicator(self) -> None:
        self.temperature_mode_indicator_updates += 1

    def apply_pending_init_migration(self, *, seed: dict[str, float], rewrite: str) -> bool:
        self.pending_init_calls.append((dict(seed), str(rewrite)))
        return True

    def invalidate_pending_init_preserved_results_after_failed_run(self) -> None:
        self.pending_init_failure_invalidations += 1

    def _sync_mechanism_controls_to_focused_batch_set(self, *, use_workspace: bool = True) -> None:
        self.focused_control_sync_calls.append(bool(use_workspace))


pytestmark = [pytest.mark.unit]


def test_mechanism_helpers_do_not_retain_main_window_back_reference() -> None:
    host = _FakeMechanismHost()
    helper = MainWindowMechanismHelpers(host)

    assert not hasattr(helper, "_mw")


def test_mechanism_helpers_snapshot_state_round_trips_through_owner_boundary() -> None:
    host = _FakeMechanismHost()
    helper = MainWindowMechanismHelpers(host)
    mechanism = object()

    helper.remember_last_mechanism(mechanism, "reaction: A -> B; k=1.0", {"solver": "BDF"})

    assert helper.last_mechanism() is mechanism
    assert helper.last_mechanism_context()["dsl_text"] == "reaction: A -> B; k=1.0"
    assert helper.last_mechanism_context()["solver_config"] == {"solver": "BDF"}

    helper.clear_last_mechanism()

    assert helper.last_mechanism() is None
    assert helper.last_mechanism_context() == {}

def test_mechanism_helpers_snapshot_context_returns_copies() -> None:
    host = _FakeMechanismHost()
    helper = MainWindowMechanismHelpers(host)

    helper.remember_last_mechanism(object(), "reaction: A -> B; k=1.0", {"solver": "BDF"})
    snapshot = helper.last_mechanism_context()
    snapshot["solver_config"]["solver"] = "BDF"

    assert helper.last_mechanism_context()["solver_config"] == {"solver": "BDF"}


def test_mechanism_helpers_own_mechanism_state_and_do_not_forward_variable_runtime() -> None:
    host = _FakeMechanismHost()
    helper = MainWindowMechanismHelpers(host)

    assert not hasattr(helper, "_runtime")
    assert not hasattr(helper, "is_energy_mode_mechanism")
    assert not hasattr(helper, "dsl_has_computational_mode_generated_block")
    assert not hasattr(helper, "sync_energy_mode_temperature_from_mechanism")
    assert not hasattr(helper, "populate_energy_mode_variables_from_mechanism")
    assert not hasattr(helper, "extract_and_populate_variables")

    helper.sync_mechanism_controls_to_focused_batch_set(use_workspace=True)
    helper.set_temperature_override_state(enabled=False, tooltip="Overridden by energy-mode DSL (T=...).")
    helper.set_temperature_mode_indicator_text("Temperature: 200.00 K (from DSL)")
    helper.update_temperature_mode_indicator()

    assert (
        helper.apply_pending_init_migration(
            seed={"A": 1.0},
            rewrite="reaction: A -> B; k=1.0",
        )
        is True
    )
    helper.invalidate_pending_init_preserved_results_after_failed_run()

    assert host.temperature_override_calls == [(False, "Overridden by energy-mode DSL (T=...).")]
    assert host.temperature_mode_indicator_texts == ["Temperature: 200.00 K (from DSL)"]
    assert host.temperature_mode_indicator_updates == 1
    assert host.pending_init_calls == [({"A": 1.0}, "reaction: A -> B; k=1.0")]
    assert host.pending_init_failure_invalidations == 1
    assert host.focused_control_sync_calls == [True]


def test_mechanism_helpers_pending_init_guard_fallback_accepts_rewrite_keyword() -> None:
    @dataclass
    class _PartialHost:
        def set_temperature_override_state(self, *, enabled: bool, tooltip: str) -> None:
            _ = (enabled, tooltip)

        def set_temperature_mode_indicator_text(self, text: str) -> None:
            _ = text

        def update_temperature_mode_indicator(self) -> None:
            return

        def apply_pending_init_migration(self, *, seed: dict[str, float], rewrite: str) -> bool:
            _ = (seed, rewrite)
            return True

    helper = MainWindowMechanismHelpers(_PartialHost())

    helper.arm_pending_init_result_invalidation_guard(rewrite="reaction: A -> B; k=1.0")
