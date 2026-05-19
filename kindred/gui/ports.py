from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple


@dataclass(frozen=True, slots=True)
class SimulationCompletionDisplayOutcome:
    displayed: bool
    direct_completion_displayed: bool = False
    reason: Optional[str] = None
    primary_set_id: Optional[str] = None
    displayed_set_ids: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        raise TypeError("Use SimulationCompletionDisplayOutcome.displayed explicitly")


@dataclass(frozen=True, slots=True)
class CachedBatchSelectionDisplayOutcome:
    displayed: bool
    reason: Optional[str] = None
    primary_set_id: Optional[str] = None
    displayed_set_ids: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        raise TypeError("Use CachedBatchSelectionDisplayOutcome.displayed explicitly")


@dataclass(frozen=True, slots=True)
class BatchDisplayRefreshOutcome:
    focused_controls_use_workspace: Optional[bool] = None
    displayed: bool = False
    reason: Optional[str] = None
    primary_set_id: Optional[str] = None
    displayed_set_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedBatchSelectionEntry:
    set_id: str
    label: str
    entry: Mapping[str, Any]
    canonical_entry: Mapping[str, Any] | None = None
    workspace_preview_provenance: Dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class CompletedRunDisplayIntent:
    set_ids: tuple[str, ...]
    labels_by_set_id: Mapping[str, str]
    primary_set_id: str
    cache_key: str
    run_id: int | None = None
    request_id: int | None = None


@dataclass(frozen=True, slots=True)
class CompletionDisplayEntry:
    set_id: str
    label: str
    t: Any
    series: Mapping[str, Any]
    algebra_scalars: Mapping[str, object]
    solver_provenance: Mapping[str, Any] | None
    mechanism_text: str
    solver_config: Mapping[str, object]
    warnings: tuple[Mapping[str, Any], ...]
    completion_provenance: Mapping[str, Any] | None

    def to_display_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "t": self.t,
            "series": dict(self.series or {}),
            "algebra_scalars": dict(self.algebra_scalars or {}),
            "solver_provenance": dict(self.solver_provenance or {}),
            "mechanism_text": str(self.mechanism_text or ""),
            "solver_config": dict(self.solver_config or {}),
            "warnings": [dict(warning) for warning in self.warnings if isinstance(warning, Mapping)],
        }
        if isinstance(self.completion_provenance, Mapping):
            payload["completion_provenance"] = dict(self.completion_provenance)
        return payload


@dataclass(frozen=True, slots=True)
class CompletedRunDisplayTransaction:
    intent: CompletedRunDisplayIntent
    completion_entries: tuple[CompletionDisplayEntry, ...]


@dataclass(frozen=True, slots=True)
class CompletedRunDisplayCoverage:
    intent: CompletedRunDisplayIntent | None = None
    transaction: CompletedRunDisplayTransaction | None = None
    missing_set_ids: tuple[str, ...] = ()
    reason: Optional[str] = None


@dataclass(frozen=True, slots=True)
class BatchDisplaySelectionResolution:
    resolved_entries: tuple[ResolvedBatchSelectionEntry, ...] = ()
    reason: Optional[str] = None
    all_selected_sets_resolved: bool = False
    has_workspace_selection: bool = False
    has_resolved_workspace_preview: bool = False
    focused_uses_workspace_controls: bool = False
    focused_has_resolved_entry: bool = False


def _normalize_slider_replay_target_set_ids(values: Sequence[str] | object) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    if isinstance(values, str):
        values = (values,)
    for value in values or ():
        set_id = str(value or "").strip()
        if not set_id or set_id in seen:
            continue
        seen.add(set_id)
        normalized.append(set_id)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class SliderReplayIntent:
    target_set_ids: tuple[str, ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_set_ids",
            _normalize_slider_replay_target_set_ids(self.target_set_ids),
        )
        object.__setattr__(self, "source", str(self.source or "").strip())


class SliderPreviewLifecyclePort(Protocol):
    def submit_slider_preview_replay_intent(
        self,
        intent: SliderReplayIntent,
        *,
        preserve_existing_request: bool = False,
    ) -> None: ...

    def clear_pending_slider_preview_replay(self, *, clear_plot_updates: bool = True) -> None: ...

    def invalidate_slider_preview_work(self) -> None: ...

    def launch_pending_slider_preview_replay(self) -> None: ...


@dataclass(frozen=True)
class SimulationCacheOpResult:
    ok: bool
    operation: str
    message: str = ""
    stats: Optional[Dict[str, Dict[str, int]]] = None
    cache_state_changed: bool = False


class SimulationDialogsPort(Protocol):
    def message_box_warning(self, title: str, message: str) -> None: ...

    def message_box_critical(self, title: str, message: str, *, details: Optional[str] = None) -> None: ...

    def message_box_question(self, title: str, message: str, *, accept_label: str = "Apply") -> bool: ...

    def choose_wegscheider_resolution(
        self,
        title: str,
        message: str,
        choices: Mapping[str, Sequence[Mapping[str, str]]],
    ) -> Optional[Dict[str, str]]: ...


class SimulationSettingsPort(Protocol):
    def settings_set_value(self, key: str, value: object) -> None: ...

    def settings_sync(self) -> None: ...


class SimulationCacheControlsPort(Protocol):
    def set_simulation_cache_caps(
        self,
        *,
        result_cap: int,
        preview_cap: int,
        persist: bool = True,
    ) -> SimulationCacheOpResult: ...

    def simulation_cache_stats(self) -> SimulationCacheOpResult: ...

    def purge_simulation_result_cache(self) -> SimulationCacheOpResult: ...

    def purge_simulation_preview_cache(self) -> SimulationCacheOpResult: ...

    def purge_simulation_all_caches(self) -> SimulationCacheOpResult: ...


class SimulationRunUiPort(Protocol):
    def run_button_is_enabled(self) -> bool: ...

    def set_run_button_enabled(self, enabled: bool) -> None: ...

    def set_runtime_backed_run_controls_ready(self, ready: bool) -> None: ...

    def schedule_runtime_availability_refresh(self) -> None: ...

    def set_stop_button_enabled(self, enabled: bool) -> None: ...

    def set_status_text(self, text: str) -> None: ...

    def set_sim_progress_value(self, value: int) -> None: ...

    def repaint_simulation_widgets(self) -> None: ...

    def set_algebra_status_text(self, text: str, *, details: str | None = None) -> None: ...


class SimulationSliderPort(Protocol):
    def stop_slider_release_commit_timer(self) -> None: ...

    def has_pending_slider_values(self) -> bool: ...

    def has_dirty_transaction(self) -> bool: ...

    def is_mechanism_valid_for_preview(self) -> bool: ...

    def show_preview_unavailable_for_dirty_state(self, message: str) -> None: ...

    def finalize_slider_release_commit(self) -> None: ...

    def stop_variable_update_timer(self) -> None: ...

    def stop_species_slider_update_timer(self) -> None: ...

    def set_slider_triggered_simulation(self, value: bool) -> None: ...

    def slider_triggered_simulation(self) -> bool: ...

    def last_slider_change_name(self) -> str: ...

    def slider_drag_active(self) -> bool: ...

    def suppress_slider_refresh(self) -> bool: ...

    def slider_gesture_target_set_ids_snapshot(self) -> List[str]: ...

    def preview_initials_for_row(self, row: int, baseline: Dict[str, float]) -> Dict[str, float]: ...

    def preview_batch_cache_token(self, rows: Sequence[int]) -> str: ...

    def has_dirty_state_for_set(self, set_id: str) -> bool: ...

    def dirty_state_generation(self, set_id: str) -> int: ...

    def reset_mechanism_workspaces(self, set_ids: Sequence[str]) -> bool: ...

    def discard_concentration_overlays_for_rows(self, rows: Sequence[int]) -> bool: ...

    def discard_concentration_overlays_for_set_ids(self, set_ids: Sequence[str]) -> bool: ...


class SimulationBatchPort(Protocol):
    def batch_rows_for_scope(self, scope: str) -> List[int]: ...

    def batch_set_ids_for_scope(self, scope: str) -> List[str]: ...

    def shown_batch_set_ids(self) -> List[str]: ...

    def slider_edit_target_set_ids(self) -> List[str]: ...

    def focused_batch_set_id(self) -> Optional[str]: ...

    def batch_current_row(self) -> Optional[int]: ...

    def batch_set_id_for_row(self, row: int) -> Optional[str]: ...

    def batch_set_name_for_id(self, set_id: str) -> Optional[str]: ...

    def batch_set_id_for_name(self, name: str) -> Optional[str]: ...

    def batch_preferred_primary_set_id(self, rows: Sequence[int]) -> Optional[str]: ...

    def set_active_batch_selection(self, set_id: str, set_name: str, selected_ids: Sequence[str]) -> None: ...

    def clear_display_selection_state(self) -> None: ...

    def clear_active_preview_selection_state(self) -> None: ...

    def batch_cache_key(
        self,
        *,
        scope_identity: object | None = None,
        mechanism_text: str = "",
        solver_config: Optional[Dict[str, Any]] = None,
        t_end: float = 0.0,
    ) -> str: ...

    def batch_store_row_count(self) -> int: ...

    def batch_store_set_names(self) -> List[str]: ...

    def batch_store_visible_species(self) -> List[str]: ...

    def batch_model_validate_rows(self, rows: Sequence[int]) -> Set[Tuple[int, str]]: ...

    def batch_initials_for_row(self, row: int) -> Dict[str, float]: ...

    def update_batch_row_controls_state(self) -> None: ...

    def sync_batch_species_columns(
        self,
        species_names: Sequence[str],
        *,
        preserve_active_cache: bool = False,
    ) -> None: ...


class SimulationMechanismPort(Protocol):
    def auto_lock_for_run(self) -> bool: ...

    def is_mechanism_ready_for_run(self) -> bool: ...

    def mechanism_reactions_text_raw(self) -> str: ...

    def mechanism_state_network_dsl_raw(self) -> str: ...

    def mechanism_slider_points_value(self) -> Optional[int]: ...

    def mechanism_slider_solver_value(self) -> Optional[str]: ...

    def set_variable_sliders(
        self,
        variables: Dict[str, float],
        *,
        metadata: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> None: ...

    def variable_slider_values(self) -> Dict[str, float]: ...

    def variable_metadata(self) -> Dict[str, Dict[str, object]]: ...

    def clear_variable_sliders(self) -> None: ...

    def has_slider_overrides(self) -> bool: ...

    def simulation_schema_id(self) -> str: ...

    def simulation_param_fingerprint(self, set_id: Optional[str] = None) -> str: ...

    def slider_overrides(self, set_id: Optional[str] = None) -> Dict[str, float]: ...

    def apply_overrides_to_text(self, base_text: str, *, set_id: Optional[str] = None) -> str: ...

    def apply_overrides_to_state_network_dsl(self, base_text: str, *, set_id: Optional[str] = None) -> str: ...

    def get_mechanism_text(self) -> str: ...

    def apply_wegscheider_resolution_source_rewrite(self, reactions_text: str) -> None: ...


class SimulationSolverPort(Protocol):
    def initial_solver_name(self) -> Optional[str]: ...

    def initial_rtol(self) -> Optional[float]: ...

    def initial_atol(self) -> Optional[float]: ...

    def temperature_spinbox_value(self) -> float: ...

    def num_points_spinbox_value(self) -> int: ...

    def sim_time_spinbox_text(self) -> str: ...

    def parse_sim_time_seconds(self) -> float: ...

    def dsl_global_temperature_K(self, dsl_text: str) -> Optional[float]: ...

    def use_sparse_jacobian(self) -> bool: ...

    def wegscheider_cyclicity_enabled(self) -> bool: ...


class SimulationRuntimePort(Protocol):
    def prepare_slider_runtime(
        self,
        param_names: Optional[list[str]] = None,
        *,
        set_id: Optional[str] = None,
    ) -> Optional[object]: ...

    def apply_slider_overrides_to_bindings(self, runtime: object, *, set_id: Optional[str] = None) -> bool: ...

    def set_slider_runtime_dirty(self, value: bool) -> None: ...

    def is_energy_mode_mechanism(self, mechanism: object) -> bool: ...

    def dsl_has_computational_mode_generated_block(self, mechanism_text: str) -> bool: ...

    def sync_energy_mode_temperature_from_mechanism(self, mechanism: object) -> None: ...

    def populate_energy_mode_variables_from_mechanism(
        self,
        mechanism: object,
        *,
        refresh_sliders: bool,
        preserve_visibility: bool = False,
    ) -> None: ...

    def extract_and_populate_variables(self, *, preserve_visibility: bool = False) -> None: ...


class SimulationResultsPort(Protocol):
    def main_plot(self) -> object: ...

    def update_main_plot_parameter_summary(self, parameters: Dict[str, Tuple[float, str]]) -> None: ...

    def set_results_table(self, table: object) -> None: ...

    def publish_simulation_completion_result(
        self,
        *,
        t: Any,
        series: Dict[str, Any],
        cache_key: Optional[str],
        batch_set: Optional[str],
        batch_set_id: Optional[str],
        redraw_valid_set_ids: Optional[Sequence[str]],
        has_redraw_subset: bool,
        slider_triggered: bool,
        explicit_batch_coalescing: bool,
        algebra_scalars: Optional[Mapping[str, object]],
        solver_provenance: Optional[Mapping[str, Any]] = None,
        direct_completion_provenance: Optional[Mapping[str, Any]] = None,
        owned_species: Optional[Sequence[str]] = None,
    ) -> SimulationCompletionDisplayOutcome: ...

    def publish_completed_run_display_transaction(
        self,
        transaction: CompletedRunDisplayTransaction,
    ) -> SimulationCompletionDisplayOutcome: ...

    def publish_cached_batch_selection(
        self,
        *,
        cache_key: str,
        selected_sets: Sequence[str],
        prefer_set: Optional[str] = None,
        cache_store: Optional[object] = None,
        valid_set_ids: Optional[Sequence[str]] = None,
        invalidated_set_ids: Optional[Sequence[str]] = None,
    ) -> "CachedBatchSelectionDisplayOutcome": ...

    def refresh_display_from_focus_and_shown(self) -> "BatchDisplayRefreshOutcome": ...

    def publish_completion_intervention_annotations(
        self,
        solver_provenance: Optional[Mapping[str, Any]],
    ) -> None: ...


class SimulationProvenancePort(Protocol):
    def snapshot_datasets(self) -> Dict[str, Any]: ...

    def last_fit_metadata(self) -> Optional[Dict[str, Any]]: ...

    def set_last_simulation_provenance(self, provenance: Dict[str, Any]) -> None: ...

    def set_last_simulation_ctc(self, ctc: Dict[str, float]) -> None: ...

    def integrate_ctc(
        self,
        t: Any,
        y: Any,
        *,
        uniformity_eps: float,
        tail_strategy: str,
    ) -> Tuple[float, str, bool, float, str]: ...

    def publish_simulation_completion_provenance(
        self,
        *,
        mechanism_text: str,
        solver_method: str,
        solver_label: str,
        solver_warning: Optional[str],
        solver_config: Mapping[str, Any],
        temperature_K: float,
        temperature_source: str,
        energy_unit: Optional[str],
        energy_mode: bool,
        simulation_time: float | str,
        num_points_requested: int,
        species_names: Sequence[str],
        t: Any,
        series: Mapping[str, Any],
        algebra_scalars: Optional[Mapping[str, Any]] = None,
        dataset_overlays: Any = None,
        solver_provenance: Optional[Mapping[str, Any]] = None,
        warnings: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]: ...


class SimulationMechanismHelpersPort(Protocol):
    def last_mechanism(self) -> Optional[object]: ...

    def last_mechanism_context(self) -> Dict[str, Any]: ...

    def remember_last_mechanism(self, mechanism: object, mechanism_text: str, solver_config: Dict[str, Any]) -> None: ...

    def set_temperature_override_state(self, *, enabled: bool, tooltip: str) -> None: ...

    def set_temperature_mode_indicator_text(self, text: str) -> None: ...

    def update_temperature_mode_indicator(self) -> None: ...

    def authoritative_structure_snapshot(
        self,
        *,
        reactions_text: str,
        state_network_text: str = "",
        units_identity: Sequence[object] = (),
        builder: Callable[[str], object],
    ) -> object: ...

    def sync_mechanism_controls_to_focused_batch_set(self, *, use_workspace: bool = True) -> None: ...

    def apply_pending_init_migration(self, *, seed_sets: Dict[str, Dict[str, float]], rewrite: str) -> bool: ...

    def arm_pending_init_result_invalidation_guard(self, *, rewrite: str | None = None) -> None: ...

    def invalidate_pending_init_preserved_results_after_failed_run(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SimulationUiPorts:
    dialogs: SimulationDialogsPort
    settings: SimulationSettingsPort
    run_ui: SimulationRunUiPort
    slider: SimulationSliderPort
    batch: SimulationBatchPort
    mechanism: SimulationMechanismPort
    solver: SimulationSolverPort
    runtime: SimulationRuntimePort
    results: SimulationResultsPort
    provenance: SimulationProvenancePort
    mechanism_helpers: SimulationMechanismHelpersPort
