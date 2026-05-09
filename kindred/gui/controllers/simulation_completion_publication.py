from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from PySide6 import QtCore

from kindred.gui.controllers.simulation_completion_policy import CacheAuthorityState, CompletionPolicyContext

logger = logging.getLogger(__name__)


@dataclass
class CompletionCallbackState:
    run_id: Optional[int]
    request_id: Optional[int]
    batch_set: Optional[str]
    batch_set_id: Optional[str]
    cache_key: Optional[str]
    policy_context: CompletionPolicyContext | None
    ctx: Mapping[str, Any] | None
    shutdown_requested: bool
    is_preview: bool
    slider_triggered: bool
    explicit_batch_coalescing: bool
    simulation_identity: Mapping[str, Any] | None = None
    preview_batch_cache_token: str | None = None
    stale_fast_handoff_after_display: bool = False
    batch_queue_done: bool = True


@dataclass
class CompletionResultState:
    t: Any
    Y: Any
    species_names: Sequence[str]
    algebra_scalars: Mapping[str, Any]
    algebra_errors: Sequence[Any]
    solver_provenance: Mapping[str, Any]
    mechanism: object | None
    base_species_count: int | None
    mechanism_text: str
    solver_config: Mapping[str, Any]
    fallback_occurred: bool
    fallback_message: object | None
    series: Dict[str, Any]
    is_primary: bool
    energy_mode: bool
    redraw_valid_set_ids: object | None
    has_redraw_subset: bool


@dataclass(frozen=True)
class SimulationCompletionPublicationDependencies:
    apply_lifecycle_effects: Callable[..., None]
    record_nonfatal_exception: Callable[[str, BaseException], None]
    queue_slider_plot_update: Callable[..., None]
    finalize_explicit_batch_dirty_reset: Callable[..., Mapping[str, Any]]
    flush_slider_plot_updates: Callable[..., None]
    show_scoped_batch_failure_summary: Callable[..., None]
    has_deferred_preview_replay_intent: Callable[[], bool]
    start_next_batch_simulation: Callable[[], None]


class SimulationCompletionPublicationOwner:
    def __init__(
        self,
        *,
        ui: Any,
        batch_context_owner: Any,
        batch_cache: Any,
        cache_admin: Any,
        completion_policy: Any,
        lifecycle_effect_owner: Any,
        result_materialization_owner: Any,
        dependencies: SimulationCompletionPublicationDependencies,
    ) -> None:
        self._ui = ui
        self._batch_context_owner = batch_context_owner
        self._batch_cache = batch_cache
        self._cache_admin = cache_admin
        self._completion_policy = completion_policy
        self._lifecycle_effect_owner = lifecycle_effect_owner
        self._result_materialization_owner = result_materialization_owner
        self._deps = dependencies

    def publish_success(
        self,
        result: Mapping[str, Any],
        state: CompletionCallbackState,
        *,
        run_id: Optional[int],
        request_id: Optional[int],
        batch_set_id: Optional[str],
        debug_batch_parallel: bool,
    ) -> None:
        try:
            logger.info("Simulation completed successfully")
            if bool(debug_batch_parallel):
                logger.info(
                    "BATCH_PAR completion handler run_id=%s request_id=%s set_id=%s slider=%s ts=%.6f",
                    int(run_id or 0),
                    int(request_id or 0),
                    str(batch_set_id or ""),
                    bool(state.slider_triggered),
                    float(perf_counter()),
                )
            result_for_completion = dict(result)
            result_for_completion.setdefault("mechanism_text", self._ui.mechanism.get_mechanism_text())
            completion = self.build_result_state(result_for_completion)
            self.resolve_cache_key(state)
            self.resolve_result_ownership(completion, state)
            self.apply_materialization_contract(completion, state)
            self.publish_primary_materialization(completion, state)
            self.apply_algebra_status(completion)
            self.publish_cache_truth(state)
            self.publish_cache_entry(completion, state)
            self.apply_pending_init(completion, state)
            self.publish_display(completion, state)
            self.publish_annotations_and_provenance(completion)
            self.apply_pending_init_guard(completion, state)
            self._result_materialization_owner.refresh_primary_result_controls(
                mechanism=completion.mechanism,
                energy_mode=bool(completion.energy_mode),
                slider_triggered=bool(state.slider_triggered),
                is_primary=bool(completion.is_primary),
            )
            state.batch_queue_done = self.advance_batch_success(completion, state)
            if state.batch_queue_done:
                self.finalize_success(completion, state)
        except Exception as exc:
            logger.error("Error displaying results: %s", exc, exc_info=True)
            self._ui.dialogs.message_box_critical(
                "Display Error",
                f"Failed to display results:\n\n{exc}",
            )
            self._deps.apply_lifecycle_effects(
                self._lifecycle_effect_owner.progress_update(status_text="Display failed")
            )
        finally:
            if state.batch_queue_done:
                self.apply_final_effects(state)

    def build_result_state(
        self,
        result: Mapping[str, Any],
    ) -> CompletionResultState:
        t = result["t"]
        Y = result["Y"]
        species_names = result["species_names"]
        algebra_scalars = result.get("algebra_scalars") or {}
        algebra_errors = result.get("algebra_errors") or []
        solver_provenance = result.get("provenance") if isinstance(result.get("provenance"), Mapping) else {}
        mechanism = result.get("mechanism")
        base_species_count = self._base_species_count(result, mechanism=mechanism)
        mechanism_text = str(result.get("mechanism_text", ""))
        solver_config = result.get("solver_config", {})
        solver_config_map = solver_config if isinstance(solver_config, Mapping) else {}
        return CompletionResultState(
            t=t,
            Y=Y,
            species_names=species_names,
            algebra_scalars=algebra_scalars if isinstance(algebra_scalars, Mapping) else {},
            algebra_errors=algebra_errors,
            solver_provenance=solver_provenance if isinstance(solver_provenance, Mapping) else {},
            mechanism=mechanism,
            base_species_count=base_species_count,
            mechanism_text=mechanism_text,
            solver_config=solver_config_map,
            fallback_occurred=bool(result.get("fallback_occurred")),
            fallback_message=result.get("fallback_message"),
            series={str(species_name): Y[i, :] for i, species_name in enumerate(species_names)},
            is_primary=True,
            energy_mode=False,
            redraw_valid_set_ids=None,
            has_redraw_subset=False,
        )

    def resolve_result_ownership(
        self,
        completion: CompletionResultState,
        state: CompletionCallbackState,
    ) -> None:
        completion.redraw_valid_set_ids, completion.has_redraw_subset = self._redraw_scope(state)
        completion.is_primary = self._is_primary(state)
        completion.mechanism = self._result_materialization_owner.resolve_completion_mechanism(
            mechanism=completion.mechanism,
            mechanism_text=completion.mechanism_text,
            solver_config=completion.solver_config,
            is_preview=bool(state.is_preview),
            is_primary=bool(completion.is_primary),
        )

    def apply_materialization_contract(
        self,
        completion: CompletionResultState,
        state: CompletionCallbackState,
    ) -> None:
        completion.energy_mode = self._result_materialization_owner.update_primary_result_materialization_contract(
            mechanism=completion.mechanism,
            mechanism_text=completion.mechanism_text,
            solver_config=completion.solver_config,
            is_preview=bool(state.is_preview),
            is_primary=bool(completion.is_primary),
        )

    def apply_algebra_status(self, completion: CompletionResultState) -> None:
        self._deps.apply_lifecycle_effects(
            self._lifecycle_effect_owner.algebra_status_effect(
                species_names=completion.species_names,
                base_species_count=completion.base_species_count,
                algebra_errors=completion.algebra_errors,
            )
        )

    def resolve_cache_key(self, state: CompletionCallbackState) -> None:
        state.cache_key = self._batch_context_owner.completion_publication_cache_key(
            callback_cache_key=state.cache_key,
            callback_context=state.ctx if isinstance(state.ctx, Mapping) else None,
        )
        state.cache_key = str(state.cache_key) if state.cache_key else None

    def publish_cache_truth(self, state: CompletionCallbackState) -> None:
        self.resolve_cache_key(state)
        if not state.cache_key:
            return
        if state.is_preview:
            preview_scope_ids = (
                state.policy_context.preview_scope_set_ids
                if state.policy_context is not None and state.policy_context.preview_scope_set_ids
                else None
            )
            self._cache_admin.publish_completion_cache_truth(
                is_preview=True,
                cache_key=state.cache_key,
                preview_scope_set_ids=preview_scope_ids,
            )
            return
        cache_reconciliation = self._completion_policy.build_explicit_cache_reconciliation(
            context=self._explicit_cache_policy_context(state),
            cache_state=self._completion_policy_cache_state(),
            cache_key=state.cache_key,
        )
        self._cache_admin.publish_completion_cache_truth(
            is_preview=False,
            cache_key=state.cache_key,
            clear_active_selection_state=cache_reconciliation.clear_active_selection_state,
            active_cache_key=cache_reconciliation.active_cache_key,
            active_cache_preview_token=cache_reconciliation.active_cache_preview_token,
            active_cache_preview_scope_set_ids=cache_reconciliation.active_cache_preview_scope_set_ids,
            active_cache_valid_set_ids=cache_reconciliation.active_cache_valid_set_ids,
            active_cache_invalidated_set_ids=cache_reconciliation.active_cache_invalidated_set_ids,
        )

    def publish_primary_materialization(
        self,
        completion: CompletionResultState,
        state: CompletionCallbackState,
    ) -> None:
        if (not state.is_preview) and completion.is_primary and completion.fallback_occurred:
            solver_name = completion.solver_config.get("solver", "selected solver")
            warning_text = f"The requested stiff solver {solver_name} failed"
            warning_text += f" ({completion.fallback_message})." if completion.fallback_message else "."
            warning_text += " The simulation retried with an alternative stiff SciPy solver."
            logger.warning("Displaying solver fallback warning: %s", warning_text)
            self._ui.dialogs.message_box_warning("Solver fallback", warning_text)

        if state.is_preview or (not completion.is_primary) or completion.mechanism is None:
            return
        self._result_materialization_owner.remember_primary_result_mechanism(
            mechanism=completion.mechanism,
            mechanism_text=completion.mechanism_text,
            solver_config=completion.solver_config,
        )
        if state.policy_context is not None and state.cache_key:
            state.policy_context = self._completion_policy.build_context_update_from_cache_truth(
                context=state.policy_context,
                cache_state=self._completion_policy_cache_state(),
                cache_key=state.cache_key,
            )
            if isinstance(state.ctx, Mapping):
                state.ctx = self._batch_context_owner.callback_context_with_cache_truth(
                    state.ctx,
                    state.policy_context,
                )
                return
            state.ctx = self._batch_context_owner.serialize_completion_policy_context(
                state.policy_context,
                base_context=state.ctx if isinstance(state.ctx, Mapping) else None,
            )

    def publish_cache_entry(
        self,
        completion: CompletionResultState,
        state: CompletionCallbackState,
    ) -> None:
        cache_token = str(state.cache_key or "")
        if not cache_token or not state.batch_set_id:
            return
        cached_mechanism = (
            completion.mechanism
            if self._batch_context_owner.include_mechanism_in_result_payload(
                fast_mode=bool(state.is_preview),
                batch_set_id=state.batch_set_id,
                context=state.ctx if isinstance(state.ctx, Mapping) else {},
            )
            else None
        )
        cache_simulation_identity = dict(state.simulation_identity or {})
        cache_preview_token = None
        if state.is_preview:
            cache_preview_token = state.preview_batch_cache_token
        self._cache_admin.publish_completion_cache(
            cache_key=cache_token,
            cache_token=cache_token,
            set_id=str(state.batch_set_id),
            is_preview=bool(state.is_preview),
            t=completion.t,
            series=completion.series,
            algebra_scalars=(dict(completion.algebra_scalars) if isinstance(completion.algebra_scalars, dict) else None),
            mechanism=cached_mechanism,
            mechanism_text=completion.mechanism_text,
            simulation_identity=cache_simulation_identity,
            solver_config=dict(completion.solver_config) if isinstance(completion.solver_config, dict) else None,
            preview_batch_cache_token=cache_preview_token,
            fallback_occurred=bool(completion.fallback_occurred),
            fallback_message=completion.fallback_message,
            solver_provenance=completion.solver_provenance,
        )

    def apply_pending_init(
        self,
        completion: CompletionResultState,
        state: CompletionCallbackState,
    ) -> None:
        if state.policy_context is None:
            return
        pending_init_completion = self._completion_policy.resolve_pending_init_completion(
            context=state.policy_context,
            batch_set=state.batch_set,
            is_preview=bool(state.is_preview),
            is_primary=bool(completion.is_primary),
        )
        if not pending_init_completion.should_attempt_apply:
            return
        applied = False
        try:
            applied = bool(
                self._ui.mechanism_helpers.apply_pending_init_migration(
                    seed_sets={str(state.batch_set): dict(pending_init_completion.seed_for_ui)},
                    rewrite=str(pending_init_completion.rewrite or ""),
                )
            )
        except Exception:
            applied = False
        if not applied:
            return
        state.policy_context = self._completion_policy.note_pending_init_apply_result(
            context=state.policy_context,
            applied=True,
        )
        state.ctx = self._batch_context_owner.serialize_completion_policy_context(
            state.policy_context,
            base_context=state.ctx if isinstance(state.ctx, Mapping) else None,
        )

    def publish_display(
        self,
        completion: CompletionResultState,
        state: CompletionCallbackState,
    ) -> None:
        if state.cache_key and (state.slider_triggered or state.explicit_batch_coalescing):
            self._deps.queue_slider_plot_update(
                set_id=state.batch_set_id,
                cache_key=str(state.cache_key),
                request_id=state.request_id,
                run_id=state.run_id,
                slider_triggered=bool(state.slider_triggered),
                valid_set_ids=(
                    completion.redraw_valid_set_ids
                    if (state.explicit_batch_coalescing and completion.has_redraw_subset)
                    else None
                ),
                allow_fallback=(not bool(state.explicit_batch_coalescing)),
            )
            return

        owned_species = None
        if completion.mechanism is not None:
            try:
                owned_species = list(completion.mechanism.species_names())
            except Exception:
                owned_species = None
        selected_sets = []
        prefer = None
        if isinstance(state.ctx, Mapping):
            selected_sets = self._ui.batch.batch_set_ids_for_scope("selected")
            current_row = self._ui.batch.batch_current_row()
            if current_row is not None:
                prefer = self._ui.batch.batch_set_id_for_row(int(current_row))
        self._ui.results.publish_simulation_completion_result(
            t=completion.t,
            series=completion.series,
            cache_key=state.cache_key,
            batch_set=state.batch_set,
            batch_set_id=state.batch_set_id,
            selected_sets=selected_sets,
            prefer_set=prefer,
            redraw_valid_set_ids=completion.redraw_valid_set_ids,
            has_redraw_subset=completion.has_redraw_subset,
            slider_triggered=bool(state.slider_triggered),
            explicit_batch_coalescing=bool(state.explicit_batch_coalescing),
            algebra_scalars=completion.algebra_scalars,
            owned_species=owned_species,
        )

    def publish_annotations_and_provenance(
        self,
        completion: CompletionResultState,
    ) -> None:
        try:
            self._ui.results.publish_completion_intervention_annotations(completion.solver_provenance)
        except Exception as exc:
            self._deps.record_nonfatal_exception("Failed to update intervention plot annotations", exc)

        if not completion.is_primary:
            return
        temperature_used = float(self._ui.solver.temperature_spinbox_value())
        energy_unit_used = None
        if completion.mechanism is not None:
            try:
                mmeta = getattr(completion.mechanism, "metadata", {}) or {}
                if isinstance(mmeta, dict) and mmeta.get("temperature_K") is not None:
                    temperature_used = float(mmeta.get("temperature_K"))
                if isinstance(mmeta, dict) and mmeta.get("energy_unit"):
                    energy_unit_used = str(mmeta.get("energy_unit"))
            except Exception as exc:
                self._deps.record_nonfatal_exception(
                    "Failed to read mechanism metadata for simulation provenance",
                    exc,
                )

        sim_time_prov: float | str = str(self._ui.solver.sim_time_spinbox_text()).strip()
        try:
            sim_time_prov = float(self._ui.solver.parse_sim_time_seconds())
        except Exception as exc:
            self._deps.record_nonfatal_exception("Failed to parse simulation time for provenance; keeping text value", exc)

        from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, normalize_solver_name

        solver_label = str(
            completion.solver_config.get("solver_label") or self._ui.solver.initial_solver_name() or DEFAULT_SOLVER_NAME
        ).strip() or DEFAULT_SOLVER_NAME
        solver_method, solver_warning = normalize_solver_name(
            str(completion.solver_config.get("solver") or solver_label)
        )
        overlay_snapshot = getattr(self._ui.results.main_plot(), "overlay_snapshot", None)
        dataset_overlays = overlay_snapshot() if callable(overlay_snapshot) else None
        self._ui.provenance.publish_simulation_completion_provenance(
            mechanism_text=completion.mechanism_text,
            solver_method=str(solver_method),
            solver_label=str(solver_label),
            solver_warning=(str(solver_warning) if solver_warning else None),
            solver_config={
                "rtol": completion.solver_config.get("rtol", self._ui.solver.initial_rtol() or 1e-6),
                "atol": completion.solver_config.get("atol", self._ui.solver.initial_atol() or 1e-12),
            },
            temperature_K=float(temperature_used),
            temperature_source=(
                "dsl" if self._ui.solver.dsl_global_temperature_K(completion.mechanism_text) is not None else "ui"
            ),
            energy_unit=energy_unit_used,
            energy_mode=bool(completion.energy_mode),
            simulation_time=sim_time_prov,
            num_points_requested=int(self._ui.solver.num_points_spinbox_value()),
            species_names=list(completion.species_names),
            t=completion.t,
            series=completion.series,
            algebra_scalars=completion.algebra_scalars,
            dataset_overlays=dataset_overlays,
        )

    def apply_pending_init_guard(
        self,
        completion: CompletionResultState,
        state: CompletionCallbackState,
    ) -> None:
        if state.policy_context is None:
            return
        pending_init_guard_rewrite = self._completion_policy.should_arm_pending_init_guard(
            context=state.policy_context,
            is_preview=bool(state.is_preview),
            is_primary=bool(completion.is_primary),
        )
        if pending_init_guard_rewrite:
            self._ui.mechanism_helpers.arm_pending_init_result_invalidation_guard(
                rewrite=str(pending_init_guard_rewrite)
            )

    def advance_batch_success(
        self,
        completion: CompletionResultState,
        state: CompletionCallbackState,
    ) -> bool:
        if not isinstance(state.ctx, Mapping):
            return not self._active_current_context()
        completion_state = self._batch_context_owner.completion_state(
            state.ctx
        )
        if completion_state is None or not completion_state.active:
            return True
        total = max(1, int(completion_state.total or len(completion_state.queue_ids) or 1))
        if state.stale_fast_handoff_after_display:
            state.ctx = self._batch_context_owner.deactivate_if_active(state.ctx)
            return True
        if isinstance(state.ctx, Mapping) and not self._batch_context_owner.context_matches_current_run_identity(
            state.ctx
        ):
            return True
        if state.policy_context is not None and isinstance(state.ctx, Mapping):
            state.ctx = self._batch_context_owner.serialize_completion_policy_context(
                state.policy_context,
                base_context=state.ctx,
            )
        if completion_state.parallel:
            transition = self._batch_context_owner.record_parallel_success(
                set_id=state.batch_set_id,
                total=total,
            )
            state.ctx = transition.context
            if transition.batch_done:
                return True
            self.publish_batch_progress(
                state.batch_set,
                completed=int(transition.completed_count),
                total=total,
            )
            return False

        transition = self._batch_context_owner.record_serial_success(
            set_id=state.batch_set_id,
            shutdown_requested=state.shutdown_requested,
        )
        state.ctx = transition.context
        if transition.batch_done:
            self._deps.apply_lifecycle_effects(
                self._lifecycle_effect_owner.serial_batch_continue_effects()
            )
            return True
        self.publish_batch_progress(
            state.batch_set,
            completed=int(transition.completed_count),
            total=total,
        )
        self._deps.apply_lifecycle_effects(
            self._lifecycle_effect_owner.serial_batch_continue_effects()
        )
        QtCore.QTimer.singleShot(0, self._deps.start_next_batch_simulation)
        return False

    def publish_batch_progress(
        self,
        batch_set: object,
        *,
        completed: int,
        total: int,
    ) -> None:
        progress_value = None
        if total > 1:
            progress_value = max(0, min(100, int((int(completed) / float(total)) * 100.0)))
        self._deps.apply_lifecycle_effects(
            self._lifecycle_effect_owner.progress_update(
                progress_value=progress_value,
                status_text=f"Completed {batch_set} ({completed}/{total})" if batch_set else None,
            )
        )

    def finalize_success(
        self,
        completion: CompletionResultState,
        state: CompletionCallbackState,
    ) -> None:
        if (not state.is_preview) and isinstance(state.ctx, dict):
            state.ctx = self._deps.finalize_explicit_batch_dirty_reset(
                state.ctx,
                mechanism=completion.mechanism,
                species_names=completion.species_names,
            )
        if state.slider_triggered or state.explicit_batch_coalescing:
            self._deps.flush_slider_plot_updates(
                force=bool(state.explicit_batch_coalescing),
                cache_key=state.cache_key,
                request_id=state.request_id,
                run_id=state.run_id,
            )
        summary = self._batch_context_owner.completion_summary(state.ctx) if isinstance(state.ctx, Mapping) else None
        failed_set_ids = summary.failed_set_ids if summary is not None else ()
        self._deps.apply_lifecycle_effects(
            self._lifecycle_effect_owner.completion_status_effect(
                species_count=len(completion.species_names),
                point_count=len(completion.t),
                failed_set_ids=failed_set_ids,
                is_preview=bool(state.is_preview),
            )
        )
        if summary is not None and summary.failed_set_ids and not bool(state.is_preview):
            self._deps.show_scoped_batch_failure_summary(
                failed_set_ids=summary.failed_set_ids,
                failed_errors=summary.failed_errors,
            )
        logger.info("Displayed results: %s time points", len(completion.t))
        logger.info("Captured simulation provenance and CTC metadata")

    def apply_final_effects(self, state: CompletionCallbackState) -> None:
        has_deferred_preview_replay = self._deps.has_deferred_preview_replay_intent()
        if has_deferred_preview_replay:
            logger.debug("Processing pending slider update after completion")
        cleanup_context = state.ctx if isinstance(state.ctx, Mapping) else {}
        self._deps.apply_lifecycle_effects(
            self._lifecycle_effect_owner.successful_completion_final_effects(
                cleanup_state=self._batch_context_owner.completion_cleanup_state(
                    cleanup_context
                ),
                stale_fast_handoff_after_display=bool(state.stale_fast_handoff_after_display),
                has_deferred_preview_replay=bool(has_deferred_preview_replay),
                shutdown_requested=bool(state.shutdown_requested),
            )
        )

    def _base_species_count(self, result: Mapping[str, Any], *, mechanism: object | None) -> int | None:
        raw_base_species_count = result.get("base_species_count")
        try:
            if raw_base_species_count is not None:
                return max(0, int(raw_base_species_count))
        except Exception:
            pass
        if mechanism is not None:
            try:
                return len(list(mechanism.species_names()))
            except Exception:
                return None
        return None

    def _redraw_scope(self, state: CompletionCallbackState) -> tuple[object | None, bool]:
        if state.is_preview or state.policy_context is None:
            return None, False
        cache_reconciliation = self._completion_policy.build_explicit_cache_reconciliation(
            context=self._explicit_cache_policy_context(state),
            cache_state=self._completion_policy_cache_state(),
            cache_key=state.cache_key,
        )
        return cache_reconciliation.redraw_valid_set_ids, bool(cache_reconciliation.has_redraw_subset)

    def _completion_policy_cache_state(self) -> CacheAuthorityState:
        return CacheAuthorityState(
            active_cache_key=self._batch_cache.active_cache_key,
            active_cache_preview_token=self._batch_cache.active_cache_preview_token,
            active_cache_preview_scope_set_ids=self._batch_cache.active_cache_preview_scope_set_ids,
            active_cache_valid_set_ids=self._batch_cache.active_cache_valid_set_ids,
            active_cache_invalidated_set_ids=self._batch_cache.active_cache_invalidated_set_ids,
        )

    def _explicit_cache_policy_context(self, state: CompletionCallbackState) -> CompletionPolicyContext:
        context = self._batch_context_owner.completion_publication_policy_context(
            callback_context=state.ctx if isinstance(state.ctx, Mapping) else None,
            policy_context=state.policy_context,
        )
        if context is None:
            return CompletionPolicyContext(
                active=False,
                request_id=state.request_id,
                run_id=state.run_id,
                fast_mode=bool(state.is_preview),
                parallel=False,
                keep_lane_pool_alive=False,
            )
        return context

    def _is_primary(self, state: CompletionCallbackState) -> bool:
        if not isinstance(state.ctx, Mapping):
            return not self._active_current_context()
        primary_set = self._batch_context_owner.primary_set_id(
            state.ctx
        )
        if not primary_set:
            return True
        return bool(state.batch_set_id is not None and str(state.batch_set_id) == str(primary_set))

    def _active_current_context(self) -> bool:
        try:
            current_state = self._batch_context_owner.completion_state()
        except Exception:
            return False
        return bool(current_state is not None and current_state.active)
