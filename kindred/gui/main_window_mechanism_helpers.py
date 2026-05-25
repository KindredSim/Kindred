from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from kindred.core.mechanism_structure_snapshot import (
    MechanismStructureSnapshot,
    MechanismStructureSnapshotOwner,
)
from kindred.core.mechanism_source import MechanismAuthoringSource

if TYPE_CHECKING:
    from kindred.gui.main_window import MainWindow


class MainWindowMechanismHelpers:
    """Owns cached mechanism snapshots for MainWindow's mechanism-helper seam."""

    def __init__(self, main_window: "MainWindow") -> None:
        self._set_temperature_override_state: Callable[..., None] = main_window.set_temperature_override_state
        self._set_temperature_mode_indicator_text: Callable[[str], None] = main_window.set_temperature_mode_indicator_text
        self._update_temperature_mode_indicator: Callable[[], None] = main_window.update_temperature_mode_indicator
        self._sync_mechanism_controls_to_focused_batch_set: Callable[..., None] = (
            main_window._sync_mechanism_controls_to_focused_batch_set
        )
        self._apply_pending_init_migration: Callable[..., bool] = main_window.apply_pending_init_migration
        self._arm_pending_init_result_invalidation_guard: Callable[..., None] = (
            main_window.arm_pending_init_result_invalidation_guard
        )
        self._invalidate_pending_init_preserved_results_after_failed_run: Callable[[], None] = (
            main_window.invalidate_pending_init_preserved_results_after_failed_run
        )
        self._last_mechanism: object | None = None
        self._last_mechanism_context: dict[str, Any] = {}
        self._structure_owner = MechanismStructureSnapshotOwner()

    def last_mechanism(self) -> object | None:
        return self._last_mechanism

    def last_mechanism_context(self) -> dict[str, Any]:
        context = dict(self._last_mechanism_context or {})
        solver_config = context.get("solver_config")
        if isinstance(solver_config, dict):
            context["solver_config"] = dict(solver_config)
        return context

    def remember_last_mechanism(self, mechanism: object, mechanism_text: str, solver_config: dict[str, Any]) -> None:
        self._last_mechanism = mechanism
        self._last_mechanism_context = {
            "dsl_text": str(mechanism_text),
            "solver_config": dict(solver_config or {}),
            "timestamp": datetime.now().isoformat(),
        }

    def authoritative_structure_snapshot(
        self,
        *,
        source: MechanismAuthoringSource,
        units_identity: tuple[object, ...] = (),
        builder: Callable[[str], object],
    ) -> MechanismStructureSnapshot:
        return self._structure_owner.snapshot_for(
            source=source,
            units_identity=units_identity,
            builder=builder,
        )

    def set_temperature_override_state(self, *, enabled: bool, tooltip: str) -> None:
        self._set_temperature_override_state(enabled=bool(enabled), tooltip=str(tooltip))

    def set_temperature_mode_indicator_text(self, text: str) -> None:
        self._set_temperature_mode_indicator_text(str(text))

    def update_temperature_mode_indicator(self) -> None:
        self._update_temperature_mode_indicator()

    def sync_mechanism_controls_to_focused_batch_set(self, *, use_workspace: bool = True) -> None:
        self._sync_mechanism_controls_to_focused_batch_set(use_workspace=bool(use_workspace))

    def apply_pending_init_migration(
        self,
        *,
        seed_sets: dict[str, dict[str, float]] | None = None,
        rewrite: str,
    ) -> bool:
        normalized = {
            str(set_name): {str(species): float(value) for species, value in dict(values).items()}
            for set_name, values in dict(seed_sets or {}).items()
        }
        return bool(self._apply_pending_init_migration(seed_sets=normalized, rewrite=str(rewrite)))

    def arm_pending_init_result_invalidation_guard(self, *, rewrite: str | None = None) -> None:
        self._arm_pending_init_result_invalidation_guard(rewrite=rewrite)

    def invalidate_pending_init_preserved_results_after_failed_run(self) -> None:
        self._invalidate_pending_init_preserved_results_after_failed_run()

    def clear_last_mechanism(self) -> None:
        self._last_mechanism = None
        self._last_mechanism_context = {}
        self._structure_owner.clear()
