from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from kindred.gui.main_window_mechanism_helpers import MainWindowMechanismHelpers


@dataclass
class _FakeVariableRuntime:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def is_energy_mode_mechanism(self, mechanism: object) -> bool:
        self.calls.append(("is_energy_mode_mechanism", (mechanism,), {}))
        return True

    def dsl_has_computational_mode_generated_block(self, mechanism_text: str) -> bool:
        self.calls.append(("dsl_has_computational_mode_generated_block", (mechanism_text,), {}))
        return "Generated" in mechanism_text

    def sync_energy_mode_temperature_from_mechanism(self, mechanism: object) -> None:
        self.calls.append(("sync_energy_mode_temperature_from_mechanism", (mechanism,), {}))

    def populate_energy_mode_variables_from_mechanism(
        self,
        mechanism: object,
        *,
        refresh_sliders: bool,
        preserve_visibility: bool = False,
    ) -> None:
        self.calls.append(
            (
                "populate_energy_mode_variables_from_mechanism",
                (mechanism,),
                {
                    "refresh_sliders": bool(refresh_sliders),
                    "preserve_visibility": bool(preserve_visibility),
                },
            )
        )

    def extract_and_populate_variables(self, *, preserve_visibility: bool = False) -> None:
        self.calls.append(("extract_and_populate_variables", (), {"preserve_visibility": bool(preserve_visibility)}))


@dataclass
class _FakeMechanismHost:
    marker: dict[str, Any] = field(default_factory=dict)
    _variable_runtime: _FakeVariableRuntime = field(default_factory=_FakeVariableRuntime)
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


def test_mechanism_helpers_delegate_full_controller_surface_without_storing_main_window() -> None:
    host = _FakeMechanismHost()
    helper = MainWindowMechanismHelpers(host)
    mechanism = object()

    assert helper.is_energy_mode_mechanism(mechanism) is True
    assert helper.dsl_has_computational_mode_generated_block("# === Generated from Computational Mode ===") is True

    helper.sync_energy_mode_temperature_from_mechanism(mechanism)
    helper.populate_energy_mode_variables_from_mechanism(
        mechanism,
        refresh_sliders=True,
        preserve_visibility=True,
    )
    helper.extract_and_populate_variables(preserve_visibility=True)
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

    assert host._variable_runtime.calls == [
        ("is_energy_mode_mechanism", (mechanism,), {}),
        ("dsl_has_computational_mode_generated_block", ("# === Generated from Computational Mode ===",), {}),
        ("sync_energy_mode_temperature_from_mechanism", (mechanism,), {}),
        (
            "populate_energy_mode_variables_from_mechanism",
            (mechanism,),
            {"refresh_sliders": True, "preserve_visibility": True},
        ),
        ("extract_and_populate_variables", (), {"preserve_visibility": True}),
    ]
    assert host.temperature_override_calls == [(False, "Overridden by energy-mode DSL (T=...).")]
    assert host.temperature_mode_indicator_texts == ["Temperature: 200.00 K (from DSL)"]
    assert host.temperature_mode_indicator_updates == 1
    assert host.pending_init_calls == [({"A": 1.0}, "reaction: A -> B; k=1.0")]
    assert host.pending_init_failure_invalidations == 1
    assert host.focused_control_sync_calls == [True]


def test_mechanism_helpers_pending_init_guard_fallback_accepts_rewrite_keyword() -> None:
    @dataclass
    class _PartialHost:
        _variable_runtime: _FakeVariableRuntime = field(default_factory=_FakeVariableRuntime)

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
