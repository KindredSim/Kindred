from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from kindred.core.mechanism_structure_snapshot import (
    MechanismStructureSnapshot,
    MechanismStructureSnapshotOwner,
)

if TYPE_CHECKING:
    from kindred.gui.main_window import MainWindow
    from kindred.gui.main_window_variable_runtime import MainWindowVariableRuntime


class MainWindowMechanismHelpers:
    """Owns cached mechanism snapshots for MainWindow's mechanism-helper seam."""

    def __init__(self, main_window: "MainWindow") -> None:
        self._runtime = getattr(main_window, "_variable_runtime", None)
        self._set_temperature_override_state: Callable[..., None] = main_window.set_temperature_override_state
        self._set_temperature_mode_indicator_text: Callable[[str], None] = main_window.set_temperature_mode_indicator_text
        self._update_temperature_mode_indicator: Callable[[], None] = main_window.update_temperature_mode_indicator
        self._sync_mechanism_controls_to_focused_batch_set: Callable[..., None] = getattr(
            main_window,
            "_sync_mechanism_controls_to_focused_batch_set",
            lambda **_kwargs: None,
        )
        self._apply_pending_init_migration: Callable[..., bool] = main_window.apply_pending_init_migration
        self._arm_pending_init_result_invalidation_guard: Callable[..., None] = getattr(
            main_window,
            "arm_pending_init_result_invalidation_guard",
            lambda *, rewrite=None: None,
        )
        self._invalidate_pending_init_preserved_results_after_failed_run: Callable[[], None] = getattr(
            main_window,
            "invalidate_pending_init_preserved_results_after_failed_run",
            lambda: None,
        )
        self._last_mechanism: object | None = None
        self._last_mechanism_context: dict[str, Any] = {}
        self._structure_owner = MechanismStructureSnapshotOwner()

    def _runtime_owner(self) -> "MainWindowVariableRuntime":
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("MainWindowMechanismHelpers requires an initialized MainWindowVariableRuntime owner.")
        return runtime

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
        reactions_text: str,
        state_network_text: str = "",
        units_identity: tuple[object, ...] = (),
        builder: Callable[[str], object],
    ) -> MechanismStructureSnapshot:
        return self._structure_owner.snapshot_for(
            reactions_text=str(reactions_text or ""),
            state_network_text=str(state_network_text or ""),
            units_identity=units_identity,
            builder=builder,
        )

    def is_energy_mode_mechanism(self, mechanism: object) -> bool:
        return bool(self._runtime_owner().is_energy_mode_mechanism(mechanism))

    def dsl_has_computational_mode_generated_block(self, mechanism_text: str) -> bool:
        return bool(self._runtime_owner().dsl_has_computational_mode_generated_block(mechanism_text))

    def sync_energy_mode_temperature_from_mechanism(self, mechanism: object) -> None:
        self._runtime_owner().sync_energy_mode_temperature_from_mechanism(mechanism)

    def set_temperature_override_state(self, *, enabled: bool, tooltip: str) -> None:
        self._set_temperature_override_state(enabled=bool(enabled), tooltip=str(tooltip))

    def set_temperature_mode_indicator_text(self, text: str) -> None:
        self._set_temperature_mode_indicator_text(str(text))

    def update_temperature_mode_indicator(self) -> None:
        self._update_temperature_mode_indicator()

    def populate_energy_mode_variables_from_mechanism(
        self,
        mechanism: object,
        *,
        refresh_sliders: bool,
        preserve_visibility: bool = False,
    ) -> None:
        self._runtime_owner().populate_energy_mode_variables_from_mechanism(
            mechanism,
            refresh_sliders=bool(refresh_sliders),
            preserve_visibility=bool(preserve_visibility),
        )

    def extract_and_populate_variables(self, *, preserve_visibility: bool = False) -> None:
        self._runtime_owner().extract_and_populate_variables(
            preserve_visibility=bool(preserve_visibility)
        )

    def sync_mechanism_controls_to_focused_batch_set(self, *, use_workspace: bool = True) -> None:
        self._sync_mechanism_controls_to_focused_batch_set(use_workspace=bool(use_workspace))

    def apply_pending_init_migration(
        self,
        *,
        seed_sets: dict[str, dict[str, float]] | None = None,
        seed: dict[str, float] | None = None,
        rewrite: str,
    ) -> bool:
        if seed_sets is None and seed is not None:
            seed_sets = {"set1": dict(seed)}
        normalized = {
            str(set_name): {str(species): float(value) for species, value in dict(seed).items()}
            for set_name, seed in dict(seed_sets or {}).items()
        }
        try:
            return bool(self._apply_pending_init_migration(seed_sets=normalized, rewrite=str(rewrite)))
        except TypeError:
            legacy_seed = normalized.get("set1")
            if legacy_seed is None:
                raise
            return bool(self._apply_pending_init_migration(seed=legacy_seed, rewrite=str(rewrite)))

    def arm_pending_init_result_invalidation_guard(self, *, rewrite: str | None = None) -> None:
        self._arm_pending_init_result_invalidation_guard(rewrite=rewrite)

    def invalidate_pending_init_preserved_results_after_failed_run(self) -> None:
        self._invalidate_pending_init_preserved_results_after_failed_run()

    def clear_last_mechanism(self) -> None:
        self._last_mechanism = None
        self._last_mechanism_context = {}
        self._structure_owner.clear()
